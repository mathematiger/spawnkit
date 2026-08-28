"""Does the batched service actually pay? Round-trip latency and throughput, per transport.

Four configurations, measured the only way that means anything — with real spawned client processes
issuing real sequential calls, because the thing under test is a round trip and a round trip cannot
be simulated in one process:

* ``queue`` — the default transport: payload pickled onto a shared request queue.
* ``queue+graph`` — the same, with CUDA-graph replay on the forward.
* ``shared`` — the shared-memory transport: one row per client, the queue carries only an id.
* ``shared+graph`` — both.

The number that matters is not one client's latency, it is **aggregate throughput across N clients**,
because that is what the service exists to raise. A per-client latency that grows while total
throughput grows is the service working as designed: the batch got fuller.

Run it directly for a live table::

    python benchmarks/bench_service.py --clients 8 --calls 400
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks._harness import Measurement, summarise, table_header, write_results
from benchmarks._model import build_reference_net
from spawnkit import cuda_hidden_from_children, prepare_cpu_only_worker, shutdown_processes
from spawnkit.service import (
    BatchedInferenceService,
    ModuleReplica,
    ServiceClient,
    SharedRows,
    SharedRowSpec,
    TensorRpc,
)

HIDDEN_DIM = 128
NUM_ACTIONS = 60
SUPPORT = 51
STEP = "step"
BARRIER_TIMEOUT_S = 120.0
"""How long a client waits at the start barrier for its peers to finish warming up."""

RESULT_TIMEOUT_S = 300.0
"""How long the parent waits for a client to report. A client that dies takes its samples with it,
and without a bound the benchmark would wait out the job rather than saying which client failed."""


def build_rpc(shared: bool) -> TensorRpc:
    """The one RPC these benchmarks serve, on either transport.

    :param shared: use the shared-memory transport rather than the queue.
    :return: the configured RPC.
    """
    spec = (
        SharedRowSpec(
            request={"hidden": ((HIDDEN_DIM,), torch.float32), "action": ((1,), torch.long)},
            response={
                "hidden_state": ((HIDDEN_DIM,), torch.float32),
                "reward": ((SUPPORT,), torch.float32),
                "policy": ((NUM_ACTIONS,), torch.float32),
                "value": ((SUPPORT,), torch.float32),
            },
        )
        if shared
        else None
    )
    return TensorRpc(
        STEP,
        method="step",
        input_axes=(0, 0),
        output_fields=("hidden_state", "reward", "policy", "value"),
        shared=spec,
    )


def client_worker(
    client_id: int,
    request_queue: Any,
    response_queue: Any,
    rpc: TensorRpc,
    stop_event: Any,
    shared_rows: SharedRows | None,
    calls: int,
    warmup: int,
    result_queue: Any,
    barrier: Any,
    profile_run: tuple[str, str] | None = None,
) -> None:
    """One client process: issue ``calls`` sequential round trips and report their durations.

    Module-level so it is picklable under ``spawn``. Pins its own thread pool, because a benchmark
    whose clients each fan numpy across every core measures core contention, not transport cost.

    The barrier is what makes the numbers mean anything. Spawned clients come up seconds apart, so
    without it the first client times its early calls against a service that is still alone, and the
    last one times its early calls against a service serving everyone — measured, that put p99 at
    650 ms against a p50 of 2 ms and reported startup skew as tail latency. Every client warms up,
    then waits for all the others, and only then starts the clock.

    ``profile_run`` is ``(run_dir, run_id)`` when ``--profile`` was passed, and ``None`` otherwise —
    the default path constructs a plain client and imports nothing from the profiler.
    """
    prepare_cpu_only_worker("cpu")
    if profile_run is None:
        _run_client_calls(
            ServiceClient(client_id, request_queue, response_queue, [rpc], stop_event, shared_rows),
            client_id, shared_rows, calls, warmup, result_queue, barrier,
        )
        return

    from benchmarks._profiling import ProfiledServiceClient, build_profiler

    run_dir, run_id = profile_run
    with build_profiler(run_dir, run_id, "client") as profiler:
        _run_client_calls(
            ProfiledServiceClient(
                profiler, client_id, request_queue, response_queue, [rpc], stop_event, shared_rows,
            ),
            client_id, shared_rows, calls, warmup, result_queue, barrier,
        )


def _run_client_calls(
    client: ServiceClient,
    client_id: int,
    shared_rows: SharedRows | None,
    calls: int,
    warmup: int,
    result_queue: Any,
    barrier: Any,
) -> None:
    """Warm up, wait at the barrier, then time ``calls`` sequential round trips.

    Split out from :func:`client_worker` so the instrumented and uninstrumented clients run the
    identical loop — a benchmark whose two arms differ in their timing code compares two loops.
    """
    if shared_rows is not None:
        payload: Any = {
            "hidden": torch.zeros(HIDDEN_DIM, dtype=torch.float32),
            "action": torch.zeros(1, dtype=torch.long),
        }
    else:
        payload = (
            np.zeros((1, HIDDEN_DIM), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int64),
        )

    try:
        for _ in range(warmup):
            client.call(STEP, payload)

        barrier.wait(timeout=BARRIER_TIMEOUT_S)

        samples = [0.0] * calls
        clock = time.perf_counter_ns
        for index in range(calls):
            start = clock()
            client.call(STEP, payload)
            samples[index] = (clock() - start) / 1_000_000.0
        result_queue.put((client_id, samples))
    except Exception as exc:  # a dead service must not leave the parent waiting forever
        result_queue.put((client_id, f"error: {exc}"))


def run_configuration(
    label: str,
    clients: int,
    calls: int,
    warmup: int,
    device: str,
    shared: bool,
    graph: bool,
    batch_window_ms: float,
    profile_run: tuple[str, str] | None = None,
) -> Measurement:
    """Measure one transport/graph configuration end to end.

    :param label: the measurement's name.
    :param clients: how many client processes to spawn.
    :param calls: timed calls per client.
    :param warmup: untimed calls per client first.
    :param device: where the service places its model.
    :param shared: use the shared-memory transport.
    :param graph: enable CUDA-graph replay.
    :param batch_window_ms: the service's batch-fill latency floor.
    :param profile_run: ``(run_dir, run_id)`` to instrument every process of this configuration, or
        ``None`` for the uninstrumented default. Instrumented numbers are not comparable with the
        published ones and are never written to ``benchmarks/results/``.
    :return: aggregated round-trip latency across every client, with throughput in its context.
    """
    ctx = mp.get_context("spawn")
    net = build_reference_net(HIDDEN_DIM, NUM_ACTIONS)
    rpc = build_rpc(shared)
    rows = SharedRows(clients, rpc.shared) if rpc.shared is not None else None

    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(clients)]
    result_queue = ctx.Queue()
    stop_event = ctx.Event()
    start_barrier = ctx.Barrier(clients)
    replica = ModuleReplica(net)

    service_kwargs: dict[str, Any] = {
        "build_fn": replica.build,
        "rpcs": [rpc],
        "request_queue": request_queue,
        "response_queues": response_queues,
        "stop_event": stop_event,
        "device": device,
        "sync_fn": replica.sync,
        "sync_interval": 10_000,  # weight syncs are a separate benchmark; keep them out of this one
        "max_batch": clients,
        "batch_window_ms": batch_window_ms,
        "shared_rows": rows,
        "graph_rpcs": (STEP,) if graph else (),
        "name": f"service[{label}]",
    }
    if profile_run is None:
        service: BatchedInferenceService = BatchedInferenceService(**service_kwargs)
    else:
        from benchmarks._profiling import ProfiledService

        service = ProfiledService(profile_run, **service_kwargs)
    # Same spawn context as the clients: the service must not be forked from a parent that has
    # already built a torch module, or it deadlocks in its first parallel region.
    service_process = service.start(ctx)

    workers = []
    # The clients are CPU-only: without this mask each of them opens a CUDA primary context and
    # holds ~414 MiB for nothing, which is both the point of the hygiene benchmark and, here, a
    # confound - N contexts on the same card perturb the service's own allocations.
    with cuda_hidden_from_children():
        for client_id in range(clients):
            worker = ctx.Process(
                target=client_worker,
                args=(
                    client_id, request_queue, response_queues[client_id], rpc, stop_event,
                    rows, calls, warmup, result_queue, start_barrier, profile_run,
                ),
                daemon=True,
            )
            worker.start()
            workers.append(worker)

    samples: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()
    try:
        for _ in range(clients):
            _client_id, payload = result_queue.get(timeout=RESULT_TIMEOUT_S)
            if isinstance(payload, str):
                errors.append(payload)
            else:
                samples.extend(payload)
    finally:
        elapsed = time.perf_counter() - started
        stop_event.set()
        request_queue.put(None)
        shutdown_processes(
            [(f"client_{i}", worker) for i, worker in enumerate(workers)] + [("service", service_process)],
        )

    if errors:
        msg = f"{label}: {len(errors)} client(s) failed: {errors[0]}"
        raise RuntimeError(msg)

    return summarise(
        label,
        samples,
        context={
            "clients": clients,
            "calls_per_client": calls,
            "transport": "shared_mem" if shared else "queue",
            "cuda_graph": graph,
            "device": device,
            "batch_window_ms": batch_window_ms,
            "aggregate_calls_per_s": round(clients * calls / elapsed, 1),
            "wall_s": round(elapsed, 3),
            "profiled": profile_run is not None,
        },
    )


def main() -> None:
    """Run every configuration and write ``benchmarks/results/service.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--calls", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-window-ms", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true", help="fewer calls, for a smoke check")
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "record a lineprofiler.accounting run for every process, so the round trip can be split "
            "into its segments. Off by default: it adds phases to the measured path, so the numbers "
            "it prints are not the published ones and are not written to benchmarks/results/."
        ),
    )
    parser.add_argument(
        "--profile-dir",
        default="benchmarks/scratch/profile",
        help="where --profile writes worker snapshots and trace sidecars (git-ignored)",
    )
    args = parser.parse_args()

    calls = 60 if args.quick else args.calls
    warmup = 10 if args.quick else args.warmup
    on_cuda = args.device.startswith("cuda")

    configurations = [
        ("service/queue", False, False),
        ("service/shared_mem", True, False),
    ]
    if on_cuda:
        configurations += [
            ("service/queue+cudagraph", False, True),
            ("service/shared_mem+cudagraph", True, True),
        ]

    print(table_header())
    measurements = []
    for label, shared, graph in configurations:
        # One run directory per configuration: merging two transports into one attempt would
        # average a queue round trip with a shared-memory one and call the result a round trip.
        profile_run = _profile_run_for(args.profile_dir, label) if args.profile else None
        measurement = run_configuration(
            label, args.clients, calls, warmup, args.device, shared, graph, args.batch_window_ms,
            profile_run,
        )
        measurements.append(measurement)
        print(measurement.line())
        if profile_run is not None:
            print(f"    profile: lineprofiler report {profile_run[0]}")

    if args.profile:
        # An instrumented run is a diagnostic, not a published figure. Writing it to
        # benchmarks/results/ would put phase overhead into a number the README cites.
        print("\n--profile set: results not written (instrumented latencies are not comparable)")
        return
    path = write_results("service.json", measurements)
    print(f"\nwrote {path}")


def _profile_run_for(profile_dir: str, label: str) -> tuple[str, str]:
    """Return ``(run_dir, run_id)`` for one configuration's profile.

    The run id carries a timestamp as well as the configuration's name. A run id identifies one
    *attempt*: re-running into the same directory under a fixed id makes the merge read every
    invocation's workers as one attempt and sum them, which reads as a single very long run.

    :param profile_dir: the root the caller chose.
    :param label: the configuration's name, used for the sub-directory and the attempt id.
    :return: the directory to write into and the run id every process of it shares.
    """
    slug = label.replace("/", "_").replace("+", "_")
    return (str(Path(profile_dir) / slug), f"{slug}-{int(time.time())}")


if __name__ == "__main__":
    main()
