"""What a CPU-only worker costs in VRAM when nobody hides the GPU from it.

The claim under test: a worker that owns no GPU tensor and performs no device compute still holds a
CUDA **primary context** — hundreds of MiB — as soon as anything in it reaches the CUDA runtime. The
call that does it is rarely one you would guard: a profiler synchronising the device, a library
probing capabilities, an ``is_available()`` inside a dependency.

Measured the only honest way: with the driver's own per-process accounting
(``nvidia-smi --query-compute-apps``), not ``torch.cuda.memory_allocated()``. The allocator's view
cannot see a context, so the term that scales with worker count is exactly the term it omits — which
is why this leak survives in codebases that do monitor VRAM.

Two arms:

* ``unmasked`` — workers spawned normally; each touches the CUDA runtime once.
* ``masked``   — the same workers spawned inside
  :func:`~spawnkit.hygiene.cuda_hidden_from_children`.

Needs a CUDA device. Without one it reports that and exits, rather than writing a result file of
zeros that would later be cited as evidence.

Run it::

    python -m benchmarks.bench_hygiene --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import Any

import torch

from benchmarks._harness import gpu_memory_mib, write_results
from spawnkit import cuda_hidden_from_children, shutdown_processes

_SETTLE_S = 3.0
"""Seconds to let every worker reach the runtime and the driver update its accounting."""


def cpu_only_worker(ready_queue: Any, stop_event: Any) -> None:
    """A worker doing no GPU work that nonetheless touches the CUDA runtime once.

    ``synchronize()`` stands in for whatever reaches the runtime in a real worker. It is the honest
    stand-in rather than a strawman: it is precisely what a profiler's ``sync=True`` phase calls, and
    that is how this was originally discovered — profiling a CPU-only worker gave it a GPU context.

    Module-level so it is picklable under ``spawn``.
    """
    visible = torch.cuda.is_available()
    if visible:
        # The measured split: is_available() costs nothing, synchronize() creates the context.
        torch.cuda.synchronize()
    ready_queue.put((mp.current_process().pid, visible))
    stop_event.wait(timeout=120)


def run_arm(workers: int, mask: bool) -> dict[str, Any]:
    """Spawn ``workers`` CPU-only workers, with or without the mask, and read their VRAM.

    :param workers: how many workers to spawn.
    :param mask: wrap the spawns in :func:`cuda_hidden_from_children`.
    :return: the arm's per-process and total VRAM, plus how many workers still saw a device.
    """
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    stop_event = ctx.Event()
    processes = []

    with cuda_hidden_from_children(mask):
        for _ in range(workers):
            process = ctx.Process(target=cpu_only_worker, args=(ready_queue, stop_event), daemon=True)
            process.start()
            processes.append(process)

    try:
        reports = [ready_queue.get(timeout=180) for _ in range(workers)]
        time.sleep(_SETTLE_S)  # the driver's per-process accounting lags the allocation slightly
        pids = [pid for pid, _visible in reports]
        usage = gpu_memory_mib(pids)
    finally:
        stop_event.set()
        shutdown_processes([(f"worker_{i}", p) for i, p in enumerate(processes)])

    holding = [usage.get(pid, 0) for pid in pids]
    return {
        "workers": workers,
        "masked": mask,
        "saw_a_device": sum(1 for _pid, visible in reports if visible),
        "per_worker_mib": holding,
        "total_mib": sum(holding),
        "workers_holding_vram": sum(1 for mib in holding if mib > 0),
    }


def main() -> None:
    """Run both arms and write ``benchmarks/results/hygiene.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device visible - this benchmark measures VRAM and has nothing to measure.")
        print("Not writing a result file: a file of zeros would later be cited as evidence.")
        return

    print(f"device: {torch.cuda.get_device_name(0)}")
    arms = {}
    for label, mask in (("unmasked", False), ("masked", True)):
        arm = run_arm(args.workers, mask)
        arms[label] = arm
        print(
            f"{label:<10} {arm['workers_holding_vram']}/{arm['workers']} workers hold VRAM, "
            f"total {arm['total_mib']} MiB "
            f"(saw a device: {arm['saw_a_device']}/{arm['workers']})",
        )

    unmasked_total = arms["unmasked"]["total_mib"]
    masked_total = arms["masked"]["total_mib"]
    per_worker = (
        unmasked_total / arms["unmasked"]["workers_holding_vram"]
        if arms["unmasked"]["workers_holding_vram"]
        else 0
    )
    print(f"\nreclaimed by masking: {unmasked_total - masked_total} MiB "
          f"({per_worker:.0f} MiB per CPU-only worker)")

    path = write_results(
        "hygiene.json",
        [],
        extra={
            "arms": arms,
            "reclaimed_mib": unmasked_total - masked_total,
            "per_worker_mib": round(per_worker, 1),
        },
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
