"""The measurement core: time a callable honestly, and write the number where the README can cite it.

Every figure this repo publishes has to come from a file under ``benchmarks/results/``. This module
is what produces those files, so it is deliberately small and deliberately strict about the things
that make a microbenchmark lie:

* **Warm-up is separate from measurement.** The first calls pay for lazy imports, allocator growth,
  library workspaces and — on a GPU — kernel autotuning. Including them reports a startup cost as a
  steady-state one.
* **Percentiles, not means.** A mean over a distribution with a tail tells you neither the typical
  cost nor the bad case. p50 and p99 tell you both, and their ratio is the interesting number when a
  benchmark contends with anything.
* **The clock is read twice per iteration and nothing else happens between the reads.** Timing
  buffers are preallocated, the loop appends nothing, and no formatting or arithmetic runs inside it.
* **A GPU result that was not synchronised is not a result.** ``sync`` drains the device queue before
  stopping the clock, or the number is submission time wearing compute's clothes.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"
"""Where every published number lives. Committed on purpose."""

RESULTS_DIR_ENV = "SPAWNKIT_BENCH_RESULTS_DIR"
"""Env var redirecting results elsewhere, so a run can measure without touching the committed files.

Needed for the one comparison that actually answers "did this change cost anything": both arms in a
single allocation, old code in a ``git worktree`` beside new. Without a redirect the second arm
overwrites the first arm's file — and, worse, leaves a number measured on a busy node sitting in the
working tree looking exactly like a published result. An env var rather than a flag because it is
set once per arm and applies to whichever benchmarks that arm runs.
"""

_NS_PER_MS = 1_000_000.0


@dataclass
class Measurement:
    """One benchmark's timing distribution, in milliseconds, plus whatever context explains it.

    :param name: what was measured; becomes the key the README cites.
    :param iterations: how many timed calls contributed.
    :param p50_ms: median call cost.
    :param p99_ms: 99th percentile call cost.
    :param mean_ms: arithmetic mean, kept because throughput derives from it and not from the median.
    :param min_ms: fastest call seen — the floor the implementation can reach.
    :param ops_per_s: calls per second implied by ``mean_ms``.
    :param context: anything that would change the number if it changed: batch size, worker count,
        device, transport. A measurement without this is not reproducible.
    """

    name: str
    iterations: int
    p50_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    ops_per_s: float
    context: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        """One aligned line for the live console table."""
        return (
            f"{self.name:<38} {self.p50_ms:>9.3f} {self.p99_ms:>9.3f} "
            f"{self.mean_ms:>9.3f} {self.ops_per_s:>12,.0f}"
        )


def table_header() -> str:
    """The console table's header, matching :meth:`Measurement.line`."""
    return (
        f"{'benchmark':<38} {'p50 ms':>9} {'p99 ms':>9} {'mean ms':>9} {'ops/s':>12}\n"
        f"{'-' * 38} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 12}"
    )


def measure(
    name: str,
    fn: Callable[[], Any],
    iterations: int = 1000,
    warmup: int = 50,
    sync: bool = False,
    context: dict[str, Any] | None = None,
) -> Measurement:
    """Time ``fn`` ``iterations`` times after ``warmup`` untimed calls.

    :param name: the measurement's name.
    :param fn: the callable to time. Takes no arguments — bind them with a closure or
        :func:`functools.partial` outside the loop, so argument construction is not measured.
    :param iterations: timed calls.
    :param warmup: untimed calls first. Never zero for anything touching torch or a device.
    :param sync: drain the CUDA queue inside the timed region, before the clock stops. Required for
        any GPU measurement; costs a synchronise per iteration, which is the price of an honest number.
    :param context: what would change this number if it changed.
    :return: the measurement.
    """
    for _ in range(warmup):
        fn()

    device_sync = _cuda_sync() if sync else None
    # Preallocated: appending inside the timed loop would measure list growth on some iterations.
    samples = [0.0] * iterations
    clock = time.perf_counter_ns

    if device_sync is None:
        for index in range(iterations):
            start = clock()
            fn()
            samples[index] = (clock() - start) / _NS_PER_MS
    else:
        for index in range(iterations):
            start = clock()
            fn()
            device_sync()
            samples[index] = (clock() - start) / _NS_PER_MS

    return summarise(name, samples, context)


def summarise(name: str, samples: list[float], context: dict[str, Any] | None = None) -> Measurement:
    """Turn raw per-call milliseconds into a :class:`Measurement`.

    Separate from :func:`measure` because some benchmarks collect their samples themselves — a
    multi-process throughput run times whole batches from the outside, and cannot use the loop above.

    :param name: the measurement's name.
    :param samples: per-call durations in milliseconds. Must be non-empty.
    :param context: what would change this number if it changed.
    :return: the measurement.
    """
    if not samples:
        msg = f"benchmark {name!r} collected no samples"
        raise ValueError(msg)
    ordered = sorted(samples)
    mean_ms = statistics.fmean(ordered)
    return Measurement(
        name=name,
        iterations=len(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p99_ms=_percentile(ordered, 0.99),
        mean_ms=mean_ms,
        min_ms=ordered[0],
        ops_per_s=(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
        context=context or {},
    )


def write_results(filename: str, measurements: Iterable[Measurement], extra: dict[str, Any] | None = None) -> Path:
    """Write measurements to ``benchmarks/results/<filename>`` as JSON, with the machine recorded.

    The environment block is not decoration: a latency number without the CPU, the GPU and the torch
    version it came from cannot be compared against anything, and "we measured 0.39 ms" in a README
    is a claim about a machine as much as about the code.

    :param filename: the result file's name, e.g. ``ipc.json``.
    :param measurements: what to record.
    :param extra: any additional top-level keys, e.g. a derived speed-up ratio.
    :return: the path written. ``$SPAWNKIT_BENCH_RESULTS_DIR`` redirects it away from the committed
        files; see :data:`RESULTS_DIR_ENV`.
    """
    directory = Path(os.environ.get(RESULTS_DIR_ENV) or RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    payload: dict[str, Any] = {
        "environment": environment(),
        "measurements": [asdict(measurement) for measurement in measurements],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def environment() -> dict[str, Any]:
    """Describe the machine well enough that a reader can tell whether a number applies to theirs."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": _cpu_count(),
    }
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info

    info["torch"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def gpu_memory_mib(pids: Iterable[int]) -> dict[int, int]:
    """Per-process VRAM in MiB, as the driver reports it — the column ``nvidia-smi`` prints.

    Deliberately not ``torch.cuda.memory_allocated``: that is the caching allocator's view and cannot
    see a process's CUDA **context**, which is the term that scales with worker count and the whole
    subject of the hygiene benchmark. Measuring the wrong one is how a 414 MiB-per-worker leak stays
    invisible behind a 300 MB allocator figure.

    :param pids: the processes to ask about.
    :return: ``{pid: mib}`` for those the driver attributed memory to; absent pids held none.
    """
    wanted = set(pids)
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    usage: dict[int, int] = {}
    for row in output.splitlines():
        parts = [cell.strip() for cell in row.split(",")]
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        pid = int(parts[0])
        if pid in wanted:
            usage[pid] = int(parts[1])
    return usage


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _cuda_sync() -> Callable[[], None] | None:
    """``torch.cuda.synchronize`` when a device is initialised here, else ``None``."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.synchronize


def _cpu_count() -> int:
    """Cores this process may actually use, which on a scheduled node is not the machine's total."""
    try:
        return len(__import__("os").sched_getaffinity(0))
    except (AttributeError, OSError):
        return __import__("os").cpu_count() or 1
