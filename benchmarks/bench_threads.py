"""The oversubscription cliff: what an unpinned intra-op thread pool costs a batched service.

torch defaults to one intra-op thread per core. For a service process that shares a node with N
worker processes, that default is wrong in a way that hides from casual measurement.

**Measure it under contention or you will not see it.** An unpinned forward timed on an otherwise
idle node is within ~1.7x of a pinned one — nothing worth a knob. Put the same service on a node
where N clients are also running and the picture changes completely: the service's threads and the
workers oversubscribe the cores between them, and every parallel region waits on a thread that
cannot get scheduled. That is why this benchmark has a ``--contenders`` arm, and why the first
version of it — a single process, alone — reported that the problem did not exist while the
end-to-end service benchmark was measuring a 400x difference from the same cause.

The second place it hides is batch size. Batch 1 takes a serial fast path and is unaffected, so
every unit test and single-client smoke run measures the one case that looks fine.

Run it::

    python -m benchmarks.bench_threads --contenders 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import Any

import torch

from benchmarks._harness import Measurement, measure, table_header, write_results
from benchmarks._model import build_reference_net
from spawnkit import shutdown_processes

BATCH_SIZES = (1, 4, 16, 64, 256)
"""Batch 1 is included precisely because it is the size that hides the problem."""


def cpu_contender(stop_event: Any) -> None:
    """Burn CPU in one thread until told to stop, standing in for a busy worker process.

    Single-threaded on purpose: N of these model N worker processes each pinned to one thread, which
    is what :func:`~spawnkit.hygiene.prepare_cpu_only_worker` gives them. The contention under test is
    between the *service's* thread pool and the workers, not between careless thread pools.

    Module-level so it is picklable under ``spawn``.
    """
    torch.set_num_threads(1)
    x = torch.zeros(64, 64)
    while not stop_event.is_set():
        for _ in range(200):
            x = torch.tanh(x @ x.T) * 0.01


def start_contenders(count: int) -> tuple[list[Any], Any]:
    """Spawn ``count`` busy sibling processes; returns them and their stop event."""
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    processes = [
        ctx.Process(target=cpu_contender, args=(stop_event,), daemon=True) for _ in range(count)
    ]
    for process in processes:
        process.start()
    if processes:
        time.sleep(2.0)  # let them come up and actually start competing before anything is timed
    return processes, stop_event


def measure_grid(
    thread_counts: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    hidden_dim: int,
    num_actions: int,
    depth: int,
    iterations: int,
    device: str,
    contenders: int,
) -> list[Measurement]:
    """Time the reference forward at every (threads, batch) combination.

    :param thread_counts: intra-op thread counts to try.
    :param batch_sizes: rows per forward.
    :param hidden_dim: reference model latent width.
    :param num_actions: reference model action count.
    :param depth: reference model trunk depth.
    :param iterations: timed forwards per cell.
    :param device: where to run.
    :param contenders: busy sibling processes to run alongside; 0 measures an idle node.
    :return: one measurement per cell.
    """
    net = build_reference_net(hidden_dim, num_actions, depth).to(device)
    torch_device = torch.device(device)
    measurements = []

    for threads in thread_counts:
        torch.set_num_threads(threads)
        for batch in batch_sizes:
            hidden = torch.zeros(batch, hidden_dim, device=torch_device)
            action = torch.zeros(batch, 1, dtype=torch.long, device=torch_device)

            def forward(hidden: torch.Tensor = hidden, action: torch.Tensor = action) -> object:
                with torch.inference_mode():
                    return net.step(hidden, action)

            measurements.append(
                measure(
                    f"forward/threads={threads}/batch={batch}",
                    forward,
                    iterations=iterations,
                    warmup=max(10, iterations // 10),
                    sync=torch_device.type == "cuda",
                    context={
                        "threads": threads,
                        "batch": batch,
                        "device": device,
                        "contenders": contenders,
                    },
                ),
            )
    return measurements


def cliff_ratio(measurements: list[Measurement]) -> dict[str, float]:
    """How much worse the worst thread count is than the best, at each batch size.

    A ratio rather than a pair of absolutes because the absolutes are machine-specific while the
    *shape* — fine at batch 1, catastrophic above it — is what a reader needs to recognise on their
    own box.

    :param measurements: the full grid.
    :return: ``{"batch=N": worst/best}``.
    """
    ratios = {}
    for batch in sorted({int(m.context["batch"]) for m in measurements}):
        at_batch = [m.p50_ms for m in measurements if m.context["batch"] == batch]
        best = min(at_batch)
        ratios[f"batch={batch}"] = round(max(at_batch) / best, 1) if best > 0 else float("inf")
    return ratios


def main() -> None:
    """Run the grid and write ``benchmarks/results/threads.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=60)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--contenders",
        type=int,
        default=4,
        help="busy sibling processes during measurement; 0 measures an idle node and will "
             "under-report the effect",
    )
    args = parser.parse_args()

    default_threads = torch.get_num_threads()
    # Deduplicated and sorted so a 2-core box does not measure "8 threads" twice.
    thread_counts = tuple(sorted({1, 2, 8, default_threads}))

    print(f"default intra-op threads on this machine: {default_threads}")
    print(f"contending sibling processes: {args.contenders}")
    if args.contenders == 0:
        print("NOTE: with no contenders this measures an idle node and will under-report the effect.")
    print(table_header())

    contender_processes, contender_stop = start_contenders(args.contenders)
    try:
        measurements = measure_grid(
            thread_counts, BATCH_SIZES, args.hidden_dim, args.num_actions, args.depth,
            args.iterations, args.device, args.contenders,
        )
    finally:
        contender_stop.set()
        shutdown_processes([(f"contender_{i}", p) for i, p in enumerate(contender_processes)])

    for measurement in measurements:
        print(measurement.line())

    ratios = cliff_ratio(measurements)
    print("\nworst/best p50 by batch size:")
    for batch, ratio in ratios.items():
        print(f"  {batch:<12} {ratio:>8.1f}x")

    # Namespaced by device: the CPU and GPU runs measure genuinely different things - the cliff is a
    # CPU thread-pool effect and vanishes when the forward runs on a device - so one filename between
    # them means whichever ran last silently replaces the other's conclusions.
    path = write_results(
        f"threads_{'cuda' if args.device.startswith('cuda') else 'cpu'}.json",
        measurements,
        extra={
            "default_intra_op_threads": default_threads,
            "contenders": args.contenders,
            "worst_over_best_p50": ratios,
        },
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
