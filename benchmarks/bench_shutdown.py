"""Shutting down N workers: one shared grace window versus a timeout each.

The claim under test: joining each worker for its own timeout makes teardown cost
``N x timeout``, while a single window shared across the pool costs ``timeout`` regardless of N.

It matters more than a tidy-up detail. At realistic worker counts the per-worker version is a
minute-long shutdown — long enough that an impatient second Ctrl+C forces an unclean exit and drops
the buffered log tail, which is the part that says why the run ended. A shutdown slow enough to be
interrupted loses the diagnosis.

**The two strategies only differ for a worker that does not exit in time**, and it is worth being
precise about that rather than overclaiming. This benchmark measures both regimes:

* ``prompt`` — workers exit well inside the grace window. Measured, the two strategies are
  indistinguishable (1.0x at every pool size), because no join ever waits out its timeout. A
  benchmark that only ran this arm would report that the shared window buys nothing.
* ``stuck`` — workers ignore SIGTERM *and* their stop event, so every join burns its full timeout.
  This is where per-worker timeouts cost ``N x timeout`` and one shared window costs ``timeout``.

The stuck arm is not a contrived case. It is the shape a real teardown takes when a worker is inside
a blocking read, a driver call, or a flush to a consumer that has already gone — which is exactly
when you are shutting down.

Run it::

    python -m benchmarks.bench_shutdown --workers 4 8 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import time
from typing import Any

from benchmarks._harness import summarise, table_header, write_results
from spawnkit import safe_is_alive, safe_join, shutdown_processes

PROMPT_EXIT_DELAY_S = 0.2
"""How long a *prompt* worker takes to wind down: well inside any sane grace window."""


def stubborn_worker(stop_event: Any, exit_delay_s: float | None) -> None:
    """A worker that ignores SIGTERM, then exits after ``exit_delay_s`` — or never, if ``None``.

    Ignoring SIGTERM models the realistic case: a worker mid-task does not die the instant it is
    asked to. ``exit_delay_s=None`` is the stuck worker, which exits only when killed, and is the
    only case in which the two teardown strategies differ at all.

    Module-level so it is picklable under ``spawn``.
    """
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if exit_delay_s is None:
        while True:  # only SIGKILL ends this
            time.sleep(1.0)
    stop_event.wait(timeout=120)
    time.sleep(exit_delay_s)


def spawn_pool(workers: int, stop_event: Any, exit_delay_s: float | None) -> list[Any]:
    """Start ``workers`` stubborn workers and wait until they are all alive."""
    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=stubborn_worker, args=(stop_event, exit_delay_s), daemon=True)
        for _ in range(workers)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not all(safe_is_alive(p) for p in processes):
        time.sleep(0.05)
    return processes


def time_per_worker_timeout(workers: int, timeout: float, exit_delay_s: float | None) -> float:
    """Teardown cost when every worker is joined for its own full timeout, in milliseconds.

    This is the naive strategy, written out rather than imported, because the point of the benchmark
    is to compare against it.
    """
    stop_event = mp.get_context("spawn").Event()
    processes = spawn_pool(workers, stop_event, exit_delay_s)
    stop_event.set()

    started = time.perf_counter_ns()
    for process in processes:
        safe_join(process, timeout=timeout)
    for process in processes:
        if safe_is_alive(process):
            process.kill()
            safe_join(process, timeout=0.5)
    return (time.perf_counter_ns() - started) / 1_000_000.0


def time_shared_window(workers: int, timeout: float, exit_delay_s: float | None) -> float:
    """Teardown cost using :func:`~spawnkit.processes.shutdown_processes`, in milliseconds."""
    stop_event = mp.get_context("spawn").Event()
    processes = spawn_pool(workers, stop_event, exit_delay_s)
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
    ratios: dict[str, dict[str, float]] = {"prompt": {}, "stuck": {}}
    for regime, exit_delay in (("prompt", PROMPT_EXIT_DELAY_S), ("stuck", None)):
        for workers in args.workers:
            results = {}
            for label, fn in (
                ("per_worker_timeout", time_per_worker_timeout),
                ("shared_window", time_shared_window),
            ):
                samples = [fn(workers, args.timeout, exit_delay) for _ in range(args.repeats)]
                measurement = summarise(
                    f"shutdown/{regime}/{label}/workers={workers}",
                    samples,
                    context={
                        "workers": workers,
                        "grace_timeout_s": args.timeout,
                        "strategy": label,
                        "regime": regime,
                    },
                )
                measurements.append(measurement)
                results[label] = measurement.p50_ms
                print(measurement.line())
            ratios[regime][f"workers={workers}"] = round(
                results["per_worker_timeout"] / max(results["shared_window"], 1e-9), 1,
            )

    for regime, by_pool in ratios.items():
        print(f"\nper-worker / shared, {regime} workers:")
        for pool, ratio in by_pool.items():
            print(f"  {pool:<14} {ratio:>8.1f}x")

    path = write_results("shutdown.json", measurements, extra={"per_worker_over_shared_p50": ratios})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
