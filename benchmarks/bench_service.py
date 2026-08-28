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
) -> None:
    """One client process: issue ``calls`` sequential round trips and report their durations.

    Module-level so it is picklable under ``spawn``. Pins its own thread pool, because a benchmark
    whose clients each fan numpy across every core measures core contention, not transport cost.

    The barrier is what makes the numbers mean anything. Spawned clients come up seconds apart, so
    without it the first client times its early calls against a service that is still alone, and the
    last one times its early calls against a service serving everyone — measured, that put p99 at
    650 ms against a p50 of 2 ms and reported startup skew as tail latency. Every client warms up,
    then waits for all the others, and only then starts the clock.
    """
    prepare_cpu_only_worker("cpu")
    client = ServiceClient(client_id, request_queue, response_queue, [rpc], stop_event, shared_rows)

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

    service = BatchedInferenceService(
        build_fn=replica.build,
        rpcs=[rpc],
        request_queue=request_queue,
        response_queues=response_queues,
        stop_event=stop_event,
        device=device,
        sync_fn=replica.sync,
        sync_interval=10_000,  # weight syncs are a separate benchmark; keep them out of this one
        max_batch=clients,
        batch_window_ms=batch_window_ms,
        shared_rows=rows,
        graph_rpcs=(STEP,) if graph else (),
        name=f"service[{label}]",
    )
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
                    rows, calls, warmup, result_queue, start_barrier,
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
        measurement = run_configuration(
            label, args.clients, calls, warmup, args.device, shared, graph, args.batch_window_ms,
        )
        measurements.append(measurement)
        print(measurement.line())

    path = write_results("service.json", measurements)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
