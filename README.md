# spawnkit

**The layer below the trainer.** Process hygiene, worker supervision and batched GPU inference for
spawn-mode multiprocessing in PyTorch.

If you run a training job as N worker processes plus a learner, you have already met some of these
failures. None of them is about your model, all of them cost wall-clock time, and every one is
silent — no traceback, no error, just a job that produces less than it should or nothing at all.

```python
from spawnkit import cuda_hidden_from_children, prepare_cpu_only_worker

with cuda_hidden_from_children():        # in the parent, around start()
    worker.start()                       # the child inherits no GPU

prepare_cpu_only_worker("cpu")           # first line of the child: pin its thread pool
```

---

## The four failures it exists for

**A CPU-only worker holds a CUDA context.** Pinning a worker's *tensors* to the CPU is not enough.
Anything that reaches the CUDA runtime — a profiler synchronising the device, a library probing
capabilities, an `is_available()` inside a dependency — creates that process's CUDA primary context.
Masking from inside the child is already too late: torch caches the visible-device count on first
use, and in a spawned child something reaches CUDA before your code runs. It has to happen in the
parent, around `Process.start()`.

**Out of memory gets retried until the wall clock runs out.** Every long-lived worker loop ends up
wrapped in a broad `except Exception`, which is right for the failures it was written for and
exactly wrong for memory exhaustion. The shape: a queue read raises `Cannot allocate memory`, the
handler logs and continues, and the same error is re-logged 1795 times in three seconds while the
trainer keeps stepping on a buffer nothing will ever refill. `spawnkit.oom` classifies it and ends
the run — non-zero, so a scheduler records a failure rather than a clean finish.

**Teardown costs N × timeout.** Joining each worker for its own timeout makes shutdown scale with the
pool. Long enough, and an impatient second Ctrl+C forces an unclean exit and drops the buffered log
tail — the part that says why the run ended. `shutdown_processes` signals everyone and then waits
once, per phase.

**A producer dies quietly and the consumer waits forever.** A collector thread that raises prints a
traceback and stops; a worker pool that exits leaves its queue empty. From inside the consumer both
look exactly like "nothing has arrived yet", and the job holds its allocation for the rest of its
wall clock without doing work. `MonitoredThread`, `spawnkit.supply` and `WorkerMonitor` make the
death observable and name which worker it was.

---

## Measured

Every number here is read from a file under [`benchmarks/results/`](benchmarks/results/), written by
the scripts in [`benchmarks/`](benchmarks/). Nothing in this README is estimated. All figures below
are from one 40-core node; run the benchmarks on yours, because two of the three effects depend on
how loaded your machine is.

### Pin the service's intra-op threads — 610×

`benchmarks/results/threads.json`. Reference forward (128-wide latent, 6 residual blocks), **with 8
competing processes** on a 40-core node:

| intra-op threads | batch 1 | batch 4 | batch 16 | batch 256 |
| --- | --- | --- | --- | --- |
| 1 | 0.376 ms | 0.455 ms | 0.591 ms | 3.01 ms |
| 40 (torch default) | 0.369 ms | **277.8 ms** | 275.7 ms | 308.5 ms |
| ratio | 1.0× | **610.6×** | 482.8× | 146.4× |

`BatchedInferenceService` defaults to `intra_op_threads=1` because of this. Two things hide it, and
both are reasons a profiler will not find it for you:

- **It needs contention.** The same grid on an *idle* node shows 1.4–1.7×. Profiling the service by
  itself says nothing is wrong.
- **It needs batch > 1.** Batch 1 takes a serial fast path and is exactly 1.0×, so every
  single-client smoke test measures the one case that looks fine.

### Teardown that does not scale with the pool — 4.6× at 16 workers

`benchmarks/results/shutdown.json`. Workers that ignore `SIGTERM` and do not exit on their own, 1 s
grace window:

| stuck workers | per-worker timeout | one shared window | ratio |
| --- | --- | --- | --- |
| 4 | 4.03 s | 2.32 s | 1.7× |
| 8 | 8.05 s | 2.72 s | 3.0× |
| 16 | 16.11 s | 3.54 s | 4.6× |

The ratio grows with the pool, which is the whole point. For workers that *do* exit inside the
window the two strategies are identical (1.0×), and the benchmark reports that arm too.

### Batched service round trip

`benchmarks/results/service.json`. 4 client processes, CPU service, 200 timed calls each after a
warm-up and a start barrier:

| transport | p50 | p99 |
| --- | --- | --- |
| queue | 1.420 ms | 1.903 ms |
| shared-memory rows | 1.321 ms | 1.570 ms |

GPU numbers, including CUDA-graph replay, are not in this table yet — they need a GPU run, and this
README does not carry a figure that has no result file behind it.

---

## Install

```bash
pip install spawnkit           # hygiene + supervision: stdlib and numpy only
pip install spawnkit[torch]    # adds the batched inference service
```

The core tiers deliberately do not depend on torch, so the hygiene layer installs in seconds and is
usable from a process that must not import torch yet.

## The three tiers

They import downward only — hygiene knows nothing of supervision, supervision nothing of the service
— so you can take just the first.

### hygiene

`cuda_hidden_from_children` and `blas_threads_pinned` run in the **parent**, around `Process.start()`;
`prepare_cpu_only_worker` runs in the **child**, first thing. Getting that backwards fails silently.
`spawnkit.seeding` derives each worker's RNG stream from one seed so a run reproduces across spawn.

### supervision

Describe each worker once and hand the list to the monitor:

```python
from spawnkit import WorkerMonitor, WorkerSpec

monitor = WorkerMonitor([
    WorkerSpec("learner", learner, critical=True),
    WorkerSpec("actor-0", actor, producer=True, restart_fn=respawn, max_restarts=3),
    WorkerSpec("evaluator", evaluator, critical=False),
], stop_event)
monitor.watch()          # raises OutOfMemoryAbortError; returns on every allowed end
```

Four policies, in a load-bearing order: OOM deaths are swept **first** (an OOM must not be reported
as a restartable worker or a tolerated one), critical deaths stop the run, tolerated deaths are
logged once, restarts are budgeted. `spawnkit.run` gives every run its own directory atomically, so
two jobs sharing a tag cannot prune each other's checkpoints.

### service

One process owns the device model; N workers hold a client and call it.

```python
from spawnkit.service import BatchedInferenceService, ModuleReplica, ServiceClient, TensorRpc

step = TensorRpc("step", method="forward", input_axes=(0, 0),
                 output_fields=("hidden", "policy", "value"))

replica = ModuleReplica(net)                       # net.share_memory() in the parent
service = BatchedInferenceService(
    build_fn=replica.build, sync_fn=replica.sync, rpcs=[step],
    request_queue=req_q, response_queues=resp_qs, stop_event=stop,
    device="cuda:0", graph_rpcs=("step",),
)
process = service.start()                          # spawn by default, never fork

# in each worker:
out = ServiceClient(rank, req_q, resp_qs[rank], [step], stop).call("step", (hidden, action))
```

VRAM is 1× the model regardless of worker count. Two transports (a queue, or shared-memory rows for
a hot path where pickling is measurable), and optional CUDA-graph replay with a first-use check that
verifies the graphed output against an eager run bit-for-bit and disables itself if they ever differ.

---

## Examples

Both run end to end and are checked by hand on every change; neither is pseudocode.

- [`examples/sb3_hygiene.py`](examples/sb3_hygiene.py) — Stable-Baselines3's `SubprocVecEnv` with
  the two parent-side context managers. You do not have to write your own worker pool to hit these
  failures; measured on a 40-core node, it takes each of 4 workers from 40 torch threads to 1.
- [`examples/ensemble_q_service.py`](examples/ensemble_q_service.py) — K CPU workers stepping
  Gymnasium environments through one batched ensemble-Q service, supervised by a `WorkerMonitor`.
  No tree search, no planning, no learning: it is the plumbing, in the order a launcher uses it.

## When *not* to use this

Being honest about this is more useful than a longer feature list.

- **Single-process training.** None of these failures exist. Use nothing.
- **On-policy RL with a vectorised env** (PPO and friends). The frameworks already own this shape;
  the hygiene helpers still compose with `SubprocVecEnv`, but the service will not fit.
- **Multi-node.** spawnkit is deliberately single-node — one machine, one process tree, no cluster
  scheduler and no network transport. Use Ray if you need to cross machines.
- **Large batches of large tensors.** The service pays off when forwards are launch-bound and small.
  If yours is genuinely GPU-bound, batching it changes little and `intra_op_threads=1` is wrong for
  you — measure before adopting either.
- **You need a task queue.** This is not one. Clients block on their own call; there is no
  scheduling, no priority, and one in-flight request per client by design.

## Documentation

Every public module, class and function carries a NumPy-style docstring that says not just what it
does but why it is shaped that way, usually with the failure that shaped it. Start with
`spawnkit/__init__.py` and the module docstrings of `hygiene`, `oom` and `monitor`.

## License

Apache-2.0.
