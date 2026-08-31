# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release. Everything below is extracted from a multi-process reinforcement-learning trainer
that hit each of these failures in production, and the entries say which failure shaped which piece —
that reasoning is the part worth keeping.

### Added — the hygiene tier

`cuda_hidden_from_children` and `blas_threads_pinned` mask `CUDA_VISIBLE_DEVICES` and the BLAS thread
variables around `Process.start()`; `prepare_cpu_only_worker` pins torch's intra-op pool from inside
the child. The split between parent and child is not stylistic. A spawn child inherits `os.environ`
as it stands at `start()`, and both numpy's BLAS backend and torch's device-count cache read their
values before any application code runs in the child — so masking from inside is already too late,
and it fails silently, with the worker simply holding resources nobody asked it for. Measured on an
A100-40GB, a CPU-only worker that so much as synchronises the device holds 414 MiB of VRAM it never
computes on.

`spawnkit.seeding` derives each worker's stream from one `SeedSequence([seed, offset])`.
`setup_seed` returns the resolved seed rather than mutating a config, so a run that was given no seed
can still record the one it used — a run that cannot say its own seed is not reproducible however
careful the rest of the machinery is.

### Added — the supervision tier

`WorkerSpec` describes a worker once; `WorkerMonitor` applies four policies in an order that is
load-bearing. Out-of-memory deaths are swept **first**, because an OOM reported as a restartable
worker, a tolerated one, or an ordinary critical failure leads somewhere wrong in all three cases.
Critical deaths stop the run, tolerated deaths are logged exactly once, and restarts are budgeted —
unbounded restarts turn a permanent failure into an infinite loop that looks, from outside, like a
working run.

`spawnkit.oom` classifies memory exhaustion by message as well as by type, because the same condition
arrives as `torch.cuda.OutOfMemoryError`, as a bare `RuntimeError` from the shared-memory allocator,
and as a `RuntimeError` re-raised by a queue's pickler. It ends the run non-zero rather than letting
a broad `except Exception` retry it: the shape that motivated this logged the same allocation failure
1795 times in three seconds while the trainer kept stepping on a buffer nothing would refill, and the
job had to be cancelled by hand nine hours later.

`spawnkit.run` claims each run's directory with `mkdir(exist_ok=False)`, which is what makes it
race-free, and writes its manifest atomically.

### Added — the batched inference service

One process owns the device model and serves N clients over a queue or shared-memory rows, so VRAM is
1x the model regardless of worker count. Remote calls are declared as `Rpc` objects — `TensorRpc`
covers the concatenate-in/slice-out case, and subclassing covers batched graphs and recurrent state
that batches on a different axis. Optional CUDA-graph replay carries a first-use check that compares
the graphed output against an eager run bit-for-bit and disables itself permanently on any
disagreement, because padding a short batch up to a captured size is only sound for a row-independent
model and that is a precondition worth verifying rather than assuming.

Responses travel as `dict[str, ndarray]`. The alternative — a positional tuple whose *length* tells
the client which variant produced it — works until two variants have the same arity and then fails by
silently decoding one as the other.

### Fixed before release — four defects an adversarial pass found

Deliberate misuse, rather than use, found each of these. All four failed either silently or fatally.

**A model that mixes rows was served, and served wrongly.** Batching N clients into one forward is
only correct when no operation mixes rows, and nothing checked it. Three clients sending identical
inputs to a model that subtracts the batch mean received -1, 0 and +1 — each depending on who it was
batched with, with no exception and no warning. `verify_row_independence` (on by default) now serves
the first multi-request batch of each RPC twice, batched and per request, and refuses the model if
the answers differ. There is deliberately no automatic fallback: serving one request per forward
would remove the service's reason to exist.

**One unroutable request ended everybody's run, and reported success.** A request naming a client id
with no response queue took the service down and exited **zero**, so the run finished "cleanly"
having served nothing after that point — the silent clean finish this package exists to prevent,
produced by the package itself. Unroutable requests (bad client id, unknown RPC name) are now logged
and dropped while the service keeps serving everyone else, and a genuinely fatal batch failure now
*raises* instead of setting the stop event and returning, so the process exits non-zero.

**Payloads whose arrays disagreed on row count were concatenated anyway.** `rows_in` reads the row
count off the first array, so a request with a 2-row and a 5-row array was split at the wrong
boundary and its rows were handed to neighbouring clients. No exception, no shape error.
`TensorRpc.collate` now rejects it.

**The shared transport's one-row limit surfaced as a reshape error.** A multi-row request produced
`shape '[3]' is invalid for input of size 12`, which describes the symptom and not the cause. It now
names the limit and says to move that RPC to the queue transport.

### Fixed before release — two more from a second adversarial pass

The first pass attacked the service; this one attacked the hygiene and supervision tiers.

**A run tag could place the run anywhere on the filesystem.** `claim_run_dir(base, "../../escaped")`
created a directory two levels above the base, and an absolute tag ignored the base entirely and
wrote to the path it named. Neither raised, and the run's output then went there. Tags come from
config files, CLI arguments and scheduler variables, so this is ordinary input rather than an
attack. A tag must now be a single directory name; empty, `.` and `..` are refused too, the last
because an empty tag resolves to the base itself and puts the next run *beside* it.

**A `restart_fn` returning `None` made the worker invisible and the monitor spun forever.** A `None`
handle means "never started", which is never a fault — so a restart that forgot to return left the
supervisor with no death to report, nothing to restart and no reason to stop. Measured: `watch()`
never returned. That is precisely the run-holds-its-allocation-doing-nothing failure this package
exists to catch, produced by the package itself. A restart yielding no handle is now logged as a
failed restart and the dead handle is kept, so the worker stays visible and the run ends with a
diagnosis.

Two things held up under the same pass and are worth recording: nested `cuda_hidden_from_children`
and `blas_threads_pinned` restore the environment correctly, and `write_run_manifest` leaves no
partial file behind when a value will not serialise.

### Fixed before release — three defects the benchmarks found

**The service forked instead of spawning.** Subclassing `multiprocessing.Process` binds a class to the
platform's default start method, which on Linux is `fork`. Forking a parent that has already built a
torch module deadlocked the service inside its first `Linear.forward`, with every client waiting on a
response that never came. The service is no longer a `Process` subclass; `start()` creates the
process from a context defaulting to spawn.

**Teardown was O(N) in worker count.** `shutdown_processes` documented a shared grace window and
delivered one — for the graceful phase only. Stragglers were then force-stopped one at a time, each
burning a full `terminate_timeout`, making the function slower than the naive per-worker join it
existed to replace: 17.9 s against 4.0 s with 16 stuck workers. All three phases now signal everyone
and wait once, giving 3.5 s against 16.1 s, with the ratio growing as the pool does.

**An unpinned intra-op thread pool cost 610x.** torch's one-thread-per-core default is wrong for a
service sharing a node with worker processes: 277.8 ms against 0.455 ms at batch 4 with 8 contenders
on 40 cores, and p50 582 ms against 1.42 ms end to end. `intra_op_threads` now defaults to 1. The
effect is invisible on an idle node (1.7x) and exactly 1.0x at batch 1, so neither a microbenchmark
nor a single-client smoke test finds it.

### Fixed before release — from review of the extracted modules

- Annotations use `multiprocessing.process.BaseProcess` rather than `Process`. A spawn context
  returns `SpawnProcess`, which typeshed makes a *sibling* of `Process`, so every type-checked caller
  using this package's headline use case failed at the API boundary.
- A producer with a `None` handle no longer ends the run as a clean success — `no_live_producers([None])`
  is `True`, so a run whose producers had not started yet exited zero on its first tick, which is the
  silent clean finish the supervision tier exists to prevent.
- `is_oom_error` drops its ten-link depth cap. The cap was a blind spot rather than a safeguard: an
  OOM wrapped deeper classified as non-OOM and got retried forever. The cycle guard is the seen-set.
- `py.typed` is shipped, without which downstream type checkers silently treat the package as `Any`.
- Importing `spawnkit.service` without torch names the `spawnkit[torch]` extra instead of raising a
  bare `ModuleNotFoundError`.
