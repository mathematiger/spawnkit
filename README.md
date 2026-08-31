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

Every number here is read from a committed file under [`benchmarks/results/`](benchmarks/results/),
written by the scripts in [`benchmarks/`](benchmarks/). Nothing is estimated. Two machines appear
below and each table says which, because a latency figure is a claim about a machine as much as about
the code: a 40-core CPU node, and an A100-SXM4-80GB node with 64 cores.

### A CPU-only worker holds 414 MiB of VRAM — until you mask it

`benchmarks/results/hygiene.json`, A100-80GB, 6 workers that touch the CUDA runtime once and do no
GPU work at all. Read with the driver's own per-process accounting, not the torch allocator, because
the allocator cannot see a CUDA context — which is precisely the term that scales with worker count.

| | workers holding VRAM | total | per worker |
| --- | --- | --- | --- |
| spawned normally | **6 / 6** | 2484 MiB | **414 MiB** |
| spawned inside `cuda_hidden_from_children()` | **0 / 6** | 0 MiB | 0 MiB |

Masked, none of the six could even see a device. At 32 workers the unmasked figure is ~13 GB of a
card doing nothing.

### Pin the service's intra-op threads — 638× on a CPU service

`benchmarks/results/threads_cpu.json`. Reference forward (128-wide latent, 6 residual blocks), **with
8 competing processes** on a 40-core node:

| intra-op threads | batch 1 | batch 4 | batch 16 | batch 256 |
| --- | --- | --- | --- | --- |
| 1 | 0.395 ms | 0.404 ms | 0.588 ms | 2.99 ms |
| 40 (torch default) | 0.421 ms | **257.5 ms** | 287.2 ms | 289.9 ms |
| ratio | 1.1× | **638×** | 488× | 140× |

`BatchedInferenceService` defaults to `intra_op_threads=1` because of this. Three things scope it,
and the first two are why a profiler will not find it for you:

- **It needs contention.** The same grid on an *idle* node shows 1.4–1.7×. Profiling the service by
  itself says nothing is wrong.
- **It needs batch > 1.** Batch 1 takes a serial fast path, so every single-client smoke test
  measures the one case that looks fine.
- **It is a CPU-service effect.** `benchmarks/results/threads_cuda.json`, same grid on the A100, is
  **1.0× at every batch size** — when the forward runs on the device, the CPU thread pool does not
  matter. If your service is on a GPU, this knob is not your problem.

### Batched service round trip, and what CUDA graphs buy

`benchmarks/results/service_cuda.json`. A100-80GB, 8 client processes, 500 timed calls each after a
warm-up and a start barrier:

| transport | p50 | p99 | aggregate |
| --- | --- | --- | --- |
| queue | 1.990 ms | 3.658 ms | 959 calls/s |
| shared-memory rows | 1.528 ms | 2.976 ms | 1014 calls/s |
| queue + CUDA graph | 1.306 ms | 2.351 ms | 1142 calls/s |
| shared-memory + CUDA graph | **0.904 ms** | **1.661 ms** | **1176 calls/s** |

Both optimisations pay and they compose: **2.2× end to end** on the round trip, from removing
pickling on the hot path and replaying the forward from a captured graph rather than relaunching it.
The graph's output is checked against an eager run on first use of every captured shape.

Measured on an otherwise-quiet node. Re-running the same commit while four unrelated jobs share the
machine costs ~15 % per call and about 2.3× of the aggregate throughput, and the 2.2× composed gain
becomes 1.96× — so treat the ratio as the portable claim and the milliseconds as a fact about this
machine. [`docs/benchmarks.md`](docs/benchmarks.md) says how that was separated from a code change.

### Teardown that does not scale with the pool — 4.6× at 16 workers

`benchmarks/results/shutdown.json`. Workers that ignore `SIGTERM` and do not exit on their own, 1 s
grace window:

| stuck workers | per-worker timeout | one shared window | ratio |
| --- | --- | --- | --- |
| 4 | 4.03 s | 2.32 s | 1.7× |
| 8 | 8.07 s | 2.73 s | 3.0× |
| 16 | 16.13 s | 3.55 s | 4.5× |
| 32 | 32.25 s | 5.20 s | **6.2×** |

The left column is linear in the pool size and the right one is not, which is the point — and both
machines produced the same figures to within a few tens of milliseconds, as an O(N)-against-O(1)
difference should.

For workers that *do* exit inside the grace window the two strategies are identical (1.0× at every
pool size), and the benchmark reports that arm too rather than quietly keeping the flattering one.

## Examples

Both run end to end and are checked by hand on every change; neither is pseudocode.

- [`examples/sb3_hygiene.py`](examples/sb3_hygiene.py) — Stable-Baselines3's `SubprocVecEnv` with
  the two parent-side context managers. You do not have to write your own worker pool to hit these
  failures; measured on a 40-core node, it takes each of 4 workers from 40 torch threads to 1.
- [`examples/ensemble_q_service.py`](examples/ensemble_q_service.py) — K CPU workers stepping
  Gymnasium environments through one batched ensemble-Q service, supervised by a `WorkerMonitor`.
  No tree search, no planning, no learning: it is the plumbing, in the order a launcher uses it.

## Limitations

Read this before adopting. Everything here is a real constraint, most of it found by deliberately
attacking the package rather than by using it, and each one is a case where you can get a wrong
answer or a stopped run if you are not aware of it.

### Batching requires a row-independent model — this is the big one

The service serves N clients' requests in **one forward**. That is only correct if no operation in
your model mixes rows. Batch normalisation left in training mode, any normalisation over `dim=0`,
attention or pooling across the batch: all of them make one client's answer depend on who it was
batched with.

Measured, before the guard existed: three clients sent *identical* inputs to a model that subtracts
the batch mean and received `-1`, `0` and `+1`. No exception, no warning, three plausible-looking
numbers, all wrong.

`BatchedInferenceService` verifies this once per RPC on its first multi-request batch
(`verify_row_independence=True` by default), on **both** transports, and refuses to serve a model
that fails. The check costs
one extra forward per request, once. **If you disable it, this becomes your responsibility.** There
is no safe automatic fallback: serving one request per forward would make the service pointless.

### One in-flight request per client

`ServiceClient.call` is synchronous and each client owns one slot. Calling one client from two
threads is detected — the ordering guard raises rather than returning mixed-up answers — but it is
not *supported*. Use one client per worker process. There is no pipelining, no async API, and no
request priority.

### The shared-memory transport is strictly limited

One row per client, shapes fixed at allocation. A request carrying several rows cannot use it, a
call whose tensor width varies per request cannot use it, and value-prefix/recurrent-state layouts
generally cannot either. All of those belong on the queue transport, which has none of those limits
and is the default. The measured gain is real but modest (1.990 ms → 1.528 ms on the A100 above), so
reach for it only when profiling says pickling is a meaningful share of your round trip.

### CUDA graphs assume more than they can check

Graph replay pads a short batch up to a captured bucket size, which is sound only for the same
row-independence property above. The runner verifies its output against an eager run on first use of
each captured shape and disables itself permanently on any mismatch — but that is a safety net over
your reasoning, not a substitute for it. Weight updates must be **in place** (`load_state_dict`); a
resync that rebinds parameters to new tensors silently invalidates every captured graph.

### Single node, single process tree

No multi-node, no cluster scheduler, no network transport, no fault-tolerant reconnection. A client
whose service dies raises; it does not reconnect. If you need to cross machines, use Ray.

### The scheduler is FIFO and has no fairness guarantee

Requests are served in arrival order up to `max_batch`. A slow client is not protected from a fast
one monopolising the queue, and there is no priority, no deadline and no backpressure beyond each
client blocking on its own call.

### Run tags must be a single directory name

`claim_run_dir` refuses a tag containing a path separator, an absolute tag, and an empty one. Tags
usually come from config files, CLI arguments or scheduler variables, and they are joined onto a
path — before this was checked, `"../../escaped"` created the run two levels above the base and an
absolute tag ignored the base entirely, both silently.

### A `restart_fn` must return the new handle

Returning `None` is treated as a *failed* restart: the dead handle is kept so the worker stays
visible, and the run ends with a diagnosis. It used to be stored as-is, which made the worker
invisible — no death to report, nothing to restart, and the monitor spun until the scheduler killed
the job.

### What it does *not* validate

- **`rows_in`** is trusted. An `Rpc` subclass that reports the wrong row count will misattribute
  results between clients silently; `TensorRpc` computes it for you, which is the safer path.
- **Payload dtypes and shapes** beyond the row-count agreement check are your model's problem.
- **Client ids** must be unique and less than the number of response queues. A request naming an
  unroutable id is logged and dropped — it will not take the service down, but that client will
  block until its own timeout.

### Platform

Linux is what it is developed and measured on. It is stdlib-only in the first two tiers so it should
work elsewhere, but `SIGHUP` handling, `/dev/shm` behaviour and the CUDA-context measurements are
POSIX assumptions, and Windows is untested.

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
