"""spawnkit — the layer below the trainer.

Four failure modes cost a multi-process training job its wall clock, and none of them is about the
model. A CPU-only worker opens a CUDA context and holds ~414 MiB of VRAM it never computes on. A
worker dies of memory exhaustion and the broad ``except Exception`` retries it until the scheduler
gives up. Shutting down N workers costs N x timeout, long enough that the second Ctrl+C drops the
log tail explaining why the run ended. A producer dies quietly and the consumer waits forever on a
buffer nothing will refill. ``spawnkit`` is the code for those four, extracted from a trainer that
hit all of them.

Three tiers, and you can take only the first:

* **hygiene** — :func:`~spawnkit.hygiene.cuda_hidden_from_children`,
  :func:`~spawnkit.hygiene.blas_threads_pinned`,
  :func:`~spawnkit.hygiene.prepare_cpu_only_worker`, :mod:`~spawnkit.seeding`. Keeps a CPU-only
  worker off the GPU and off every core, and makes one seed reproduce a run across spawn.
* **supervision** — :class:`~spawnkit.monitor.WorkerSpec`, :class:`~spawnkit.monitor.WorkerMonitor`,
  :mod:`~spawnkit.oom`, :mod:`~spawnkit.processes`, :mod:`~spawnkit.supply`, :mod:`~spawnkit.run`.
  Notices a death, says which one it was, and stops the run instead of spinning it out.
* **service** — :mod:`spawnkit.service`. One GPU process serving batched inference to N CPU workers,
  with CUDA-graph replay on the hot path. Needs the ``[torch]`` extra.

The tiers import downward only: hygiene knows nothing of supervision, supervision nothing of the
service. ``pip install spawnkit`` is stdlib + numpy; torch arrives with ``spawnkit[torch]``.
"""

from __future__ import annotations

from spawnkit.hygiene import (
    BLAS_THREAD_VARS,
    CUDA_VISIBLE_DEVICES,
    blas_threads_pinned,
    cuda_hidden_from_children,
    prepare_cpu_only_worker,
)
from spawnkit.lifecycle import register_shutdown_signals
from spawnkit.monitor import WorkerMonitor, WorkerSpec
from spawnkit.oom import (
    OOM_EXIT_CODE,
    OutOfMemoryAbortError,
    abort_worker_on_oom,
    is_oom_error,
    process_oom_reason,
    raise_if_oom,
    thread_oom_reason,
)
from spawnkit.processes import (
    MonitoredThread,
    detach_queue_feeder,
    graceful_shutdown,
    safe_close,
    safe_is_alive,
    safe_join,
    safe_kill,
    safe_terminate,
    shutdown_processes,
)
from spawnkit.run import claim_run_dir, default_run_tag, write_run_manifest
from spawnkit.seeding import (
    COLLECTOR_SEED_OFFSET,
    EVALUATOR_SEED_OFFSET,
    seed_worker,
    setup_seed,
    worker_rng,
)
from spawnkit.supply import (
    SupplyStalledError,
    no_live_producers,
    thread_crash_reason,
    thread_stopped_reason,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "BLAS_THREAD_VARS",
    "COLLECTOR_SEED_OFFSET",
    "CUDA_VISIBLE_DEVICES",
    "EVALUATOR_SEED_OFFSET",
    "OOM_EXIT_CODE",
    "MonitoredThread",
    "OutOfMemoryAbortError",
    "SupplyStalledError",
    "WorkerMonitor",
    "WorkerSpec",
    "__version__",
    "abort_worker_on_oom",
    "blas_threads_pinned",
    "claim_run_dir",
    "cuda_hidden_from_children",
    "default_run_tag",
    "detach_queue_feeder",
    "graceful_shutdown",
    "is_oom_error",
    "no_live_producers",
    "prepare_cpu_only_worker",
    "process_oom_reason",
    "raise_if_oom",
    "register_shutdown_signals",
    "safe_close",
    "safe_is_alive",
    "safe_join",
    "safe_kill",
    "safe_terminate",
    "seed_worker",
    "setup_seed",
    "shutdown_processes",
    "thread_crash_reason",
    "thread_oom_reason",
    "thread_stopped_reason",
    "worker_rng",
    "write_run_manifest",
]
