"""Shutting down N workers: one shared grace window versus a timeout each.

The claim under test: joining each worker for its own timeout makes teardown cost
``N x timeout``, while a single window shared across the pool costs ``timeout`` regardless of N.

It matters more than a tidy-up detail. At realistic worker counts the per-worker version is a
minute-long shutdown — long enough that an impatient second Ctrl+C forces an unclean exit and drops
the buffered log tail, which is the part that says why the run ended. A shutdown slow enough to be
interrupted loses the diagnosis.

The workers here **ignore SIGTERM** and exit only on their stop event, after a delay. That is the
case that separates the two strategies: a worker that dies instantly makes them look identical, so
measuring with cooperative workers would show nothing and prove nothing.

Run it::

    python -m benchmarks.bench_shutdown --workers 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import time
from typing import Any

from benchmarks._harness import summarise, table_header, write_results
from spawnkit import safe_is_alive, safe_join, shutdown_processes

_WORKER_EXIT_DELAY_S = 0.4
"""How long each worker takes to notice its stop event and wind down."""


def stubborn_worker(stop_event: Any) -> None:
    """A worker that ignores SIGTERM and exits ``_WORKER_EXIT_DELAY_S`` after its stop event.

    Module-level so it is picklable under ``spawn``. Ignoring SIGTERM models the realistic case: a
    worker mid-task does not die the instant it is asked to, and how a teardown handles that is the
    whole difference between the two strategies.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    stop_event.wait(timeout=120)
    time.sleep(_WORKER_EXIT_DELAY_S)


def spawn_pool(workers: int, stop_event: Any) -> list[Any]:
    """Start ``workers`` stubborn workers and wait until they are all alive."""
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=stubborn_worker, args=(stop_event,), daemon=True) for _ in range(workers)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not all(safe_is_alive(p) for p in processes):
        time.sleep(0.05)
    return processes


def time_per_worker_timeout(workers: int, timeout: float) -> float:
    """Teardown cost when every worker is joined for its own full timeout, in milliseconds.

    This is the naive strategy, written out rather than imported, because the point of the benchmark
    is to compare against it.
    """
    stop_event = mp.get_context("spawn").Event()
    processes = spawn_pool(workers, stop_event)
    stop_event.set()

    started = time.perf_counter_ns()
    for process in processes:
        safe_join(process, timeout=timeout)
    for process in processes:
        if safe_is_alive(process):
            process.kill()
            safe_join(process, timeout=0.5)
    return (time.perf_counter_ns() - started) / 1_000_000.0


def time_shared_window(workers: int, timeout: float) -> float:
    """Teardown cost using :func:`~spawnkit.processes.shutdown_processes`, in milliseconds."""
    stop_event = mp.get_context("spawn").Event()
    processes = spawn_pool(workers, stop_event)
    stop_event.set()

    started = time.perf_counter_ns()
    shutdown_processes(
        [(f"worker_{i}", p) for i, p in enumerate(processes)],
        graceful_timeout=timeout,
    )
    return (time.perf_counter_ns() - started) / 1_000_000.0


def main() -> None:
    """Compare both strategies across pool sizes and write ``benchmarks/results/shutdown.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--timeout", type=float, default=1.0, help="per-worker / shared grace window")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(table_header())
    measurements = []
    ratios = {}
    for workers in args.workers:
        results = {}
        for label, fn in (("per_worker_timeout", time_per_worker_timeout), ("shared_window", time_shared_window)):
            samples = [fn(workers, args.timeout) for _ in range(args.repeats)]
            measurement = summarise(
                f"shutdown/{label}/workers={workers}",
                samples,
                context={"workers": workers, "grace_timeout_s": args.timeout, "strategy": label},
            )
            measurements.append(measurement)
            results[label] = measurement.p50_ms
            print(measurement.line())
        ratios[f"workers={workers}"] = round(
            results["per_worker_timeout"] / max(results["shared_window"], 1e-9), 1,
        )

    print("\nper-worker / shared, by pool size:")
    for pool, ratio in ratios.items():
        print(f"  {pool:<14} {ratio:>8.1f}x")

    path = write_results("shutdown.json", measurements, extra={"per_worker_over_shared_p50": ratios})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
