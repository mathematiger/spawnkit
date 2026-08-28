# withlineprofiler feedback — a batched inference service, its round trip, and what the report could not say

Against `with_line_profiler 0.8.2` at the head of its unreleased branch, on a 40-core login node (128 cores on the box, 40 in the job's affinity mask, 2.0 TB RAM), Python 3.12, torch 2.11, **no GPU**. Every number below comes from `benchmarks/bench_service.py --clients 4 --device cpu --calls 1500 --profile`, which spawns one service process and four client processes and issues 6,200 real round trips per transport.

The instrumentation lives entirely in `benchmarks/_profiling.py`. `spawnkit` itself does not and will not import a profiler — a library that does forces the choice on everyone who installs it — so the client and the service are instrumented from outside, by subclasses that override their private steps. That constraint is worth stating up front because it shapes what could be measured: nothing here required a hook inside the package, and the `phase` / `trace_begin` / `trace_mark` / `trace_end` / `signal` / `wait_on` set was sufficient to decompose a cross-process round trip from the outside. That is the headline compliment.

Five issues follow. Four are cases where a figure in the report is wrong, missing, or means less than it appears to; the fifth is an output that exists in one format only. All five are backed by a measurement, and each names the function that would carry the fix.

---

## What was measured, for context

The service owns the model; four clients each block on one synchronous call at a time. The round trip decomposes as follows (queue transport, 6,198 complete lifecycles, means per request):

| segment | instrument | mean | share |
|---|---|---|---|
| client encode + `put` | `phase("submit")` | 6.6 µs (p50) | 0.4% |
| `begin → admitted` — queue transit, plus the request sitting unclaimed while the service worked | lifecycle marks | 726.9 µs | 43% |
| `admitted → computed` — collate (16.0 µs p50) + forward (473.8 µs p50) | lifecycle marks | 587.8 µs | 35% |
| `computed → replied` — split (30.9 µs p50) + `put` (20.6 µs p50) | lifecycle marks | 75.2 µs | 4% |
| `replied → end` — the return hop | lifecycle marks | 311.4 µs | 18% |
| client decode | `phase("decode")` | 0.6 µs (p50) | ~0% |

The service is the bottleneck, and the report says so from three independent directions: its phase wall time covers 92.4% of the run; its per-request occupancy is (2.16 s + 0.406 s) / 6,200 = **414 µs**, and four clients queueing behind that predicts a 1.66 ms round trip against the 1.614 ms measured; and its process CPU sits at **99% of a core** (it is pinned to one thread by `intra_op_threads=1`, so that is its ceiling). The clients are not idle for want of work — `busy 98.7% / working 3.2%` — they are queued.

The shared-memory transport's advantage shows up exactly where it should and nowhere else: `replied → end` falls from **311.4 µs to 140.4 µs**, a 2.2x reduction on the one hop that carries the four response fields (290 float32 = 1,160 bytes). The request hop is unchanged (726.9 → 714.0 µs), which is right: the request payload is small and that interval is dominated by service occupancy rather than by transit. The saving is real and it is 171 µs of a round trip whose length is set by a saturated server, so end to end it is a few percent (uninstrumented p50 1.340 ms queue against 1.290 ms shared).

---

## Issue 1 — `CPU peak` in the `RESOURCES` block is quantisation noise, and it is the figure a reader sizes a job with

**Severity: high.** It is a wrong number, in the headline resource block, non-deterministic between identical runs, and it overstated this run's peak CPU demand by **5.8x**.

### What the report said

```
                            used     available      per proc
CPU  peak              8.7 cores      40 cores          1.73
CPU  mean              2.9 cores   (7% of box)          0.58
```

and on the shared-memory configuration of the same benchmark, `CPU peak 19.5 cores`. The truth, from the same worker files: one service process at ~1.06 cores and four clients at 0.08–0.12 cores each — a concurrent peak of about **1.5 cores**. The 0.53-second version of the identical benchmark reported `CPU peak 1.4 cores`. Same code, same machine, same workload; 1.4, 8.7 and 19.5 cores.

### Where it comes from

The per-process readings behind those totals:

```
service   n=4  cpu% = [  0, 106, 108, 107]
client    n=4  cpu% = [725,   9,  10,   8]     <- 725% on a process whose steady state is 9%
client    n=4  cpu% = [  0,  10,   8,  12]
```

`ResourceSampler._run` (`accounting/sampler.py`) writes a baseline row "before any interval elapses", and `_add_process_metrics` reads `psutil.Process.cpu_percent()` on it. That call differences against the priming call spent in `_detect_capabilities`, a thread start and a file open earlier — on the order of a millisecond. `psutil` divides a CPU-time delta quantised to the scheduler tick by that sub-millisecond wall interval.

Measured directly, on a process spinning at exactly 100% of a core, varying only the gap between the priming call and the reading (`SC_CLK_TCK` = 100 Hz, so one tick is 10 ms):

| probe → reading gap | readings of 0% | readings > 200% | max | mean |
|---|---|---|---|---|
| 0.2 ms | 60/60 | 0/60 | 0% | 0% |
| 0.5 ms | 56/60 | 4/60 | **1920%** | 128% |
| 1.4 ms | 51/60 | 9/60 | **703%** | 105% |
| 2.0 ms | 48/60 | 12/60 | 495% | 99% |
| 5.0 ms | 30/60 | 0/60 | 199% | 100% |
| 20 ms | 0/60 | 0/60 | 100% | 99% |
| 1000 ms | 0/60 | 0/60 | 100% | 99% |

The quantisation is unbiased, so the *mean* survives it — but `analysis._combine_cpu` reduces each process with `peak = max(cores)`, and the maximum is precisely the statistic a one-tick artefact captures. From two ticks (20 ms) upward the counter is exact.

The same defect reaches the *final* row, which `_run` writes immediately after the stop signal: if the run ends shortly after an interval boundary, that row differences over the same kind of sub-tick window.

### Why it matters here rather than being cosmetic

`CPU peak` against `available cores` is the one line in the report that answers "how many of these can I run on a node?". This service is pinned to one thread on purpose, and the interesting fact about the run is that it is *at* its one-core ceiling. The report instead offered 8.7 cores of 40, which reads as ample headroom in a job that has none.

### Suggested fix

`accounting/sampler.py`, `ResourceSampler._add_process_metrics`. **Reject a `cpu_percent` reading whose interval is shorter than the counter's own resolution, using the `-1.0` unmeasured sentinel that already exists for exactly this purpose.** Concretely: record the monotonic time of the priming call in `__init__`, and on each row read `cpu_percent()` (so `psutil`'s internal cursor still advances one row per interval) but keep the value only when the elapsed since the previous read clears a few scheduler ticks — `_MIN_TICKS / os.sysconf("SC_CLK_TCK")`, four ticks being 40 ms at the usual 100 Hz, comfortably inside the exact region the table above measures.

This is the general form rather than "skip the first row", and the generality is load-bearing: it covers the final row, and it covers a caller who sets `sample_interval_s=0.005`, for whom *every* row is currently sub-tick noise. All the downstream plumbing is already in place — `_compact` gates on `>= 0`, `_combine_cpu` skips negatives — so the change is confined to one function plus a constant.

**And then print `CpuUsage.max_process`.** It is already computed, already carried into `report_as_dict` as `cpu_cores_max_process`, and its own docstring says *"The heaviest single process's peak. Against `peak / processes` this is the skew"* — but `report._resources_block` never renders it. The RAM row does have this line (`heaviest process held 574.4 MB RSS against a 558.7 MB mean`, a 1.03x skew, which is the uninteresting one). CPU's skew here is 1.06 cores against a 0.30-core mean — the single most useful sentence the resources block could have printed, and the one it withheld. It is worth doing in the same change, and only in the same change: printed today it would say 7.25 cores.

---

## Issue 2 — a lifecycle that lost its intermediate checkpoints is rendered as a segment, and it outranks every real one

**Severity: high.** This is the wrong-numbers class: a plausible row, in the block the docs present as the answer to "decomposing a queue wait", naming a transition nobody instrumented.

### What the report said

Shared-memory configuration, `REQUEST LIFECYCLE`:

```
roundtrip                          9.53s  (6,196 req)
    ├─ begin → admitted               4.42s    46%   714.0us/ea
    ├─ admitted → computed            3.31s    35%   534.2us/ea
    ├─ computed → replied           595.6ms     6%    96.1us/ea
    ├─ replied → end                870.0ms     9%   140.4us/ea
    └─ begin → end                  330.9ms     3%    82.7ms/ea
```

There is no `begin → end` transition in this pipeline. Four requests — each client's very first call, `0:1` through `3:1`, taking 46–121 ms while the service was still building its model — arrived at the merge carrying only their `begin` and `end` marks, because the service's link ring had dropped their three intermediate checkpoints. `tracealign._segments_of` requires `begin` and `end` and nothing else, so those four lifecycles were accepted as complete and contributed the *whole* round trip as a fifth "segment", beside the four segments that decompose it.

Three consequences, in ascending order of harm:

1. **The denominator is inflated by time already counted.** The four real segments sum to 9.20 s; the phantom pushes the total to 9.53 s, so every genuine share is understated by 3.4%.
2. **A transition that does not exist appears in the table**, in the same units and the same tree, in a pipeline the reader is trying to learn the shape of.
3. **It ranks first on the column that matters.** Sorting the segments by `/ea` — which is what a reader does to find the expensive step — puts `begin → end` at 82.7 ms/ea, **59x** the whole round trip's own p50 and ahead of every real step. The most expensive thing in the breakdown is a thing that never happened.

### The drop that caused it is not disclosed anywhere the reader is looking

`TraceBuffer.record_link` counted it correctly: `dropped_links = 16` in the service's file, carried faithfully through `AlignedTrace.dropped_links`. But `dropped_links` is surfaced in exactly one place, `htmltrace._caveats` — a *different page*, which does not render the `REQUEST LIFECYCLE` block at all. The text report never mentions it, and `lineprofiler trace --format json` does not carry it either (its keys are `duration_ns, lanes, spans, arrows, unmatched_waits, dropped_spans, clock_steps, findings, phases` — `dropped_spans` is there and `dropped_links` is not, so a CI gate reading that document cannot see that records were lost).

This is the same shape as the superseded-worker disclosure the changelog already fixed once: the caveat was present, correct, and on a page eighty lines and one file away from the number it qualifies.

### Suggested fix

`accounting/tracealign.py`, `_segments_of` / `lifecycle_segments`:

- **A `begin → end` interval is not a segment when the channel decomposes elsewhere.** Where other requests on the same channel carry intermediate checkpoints, a request that carries none has nothing to decompose and should contribute nothing, exactly as an incomplete lifecycle already does. Where *every* request on a channel is `begin`/`end`-only, the caller instrumented only the ends and one row is the honest answer — so the rule must be conditional on the channel, not unconditional.
- **Count the exclusions and say so**, rather than repairing silently: `4 of 6,200 request(s) carried no intermediate checkpoint and are excluded` under the block. A quietly dropped request is better than a phantom row, but it is not good enough on its own — the reader needs to know the sample shrank.

`accounting/report.py`, `_lifecycle_block` and the header: **surface `dropped_links` in the text report**, with the cause and the two levers (`trace_capacity`, `snapshot_interval_s`), on the block whose numbers it degrades. And add `dropped_links` to `cli._render_trace`'s JSON, beside `dropped_spans`.

---

## Issue 3 — the request lifecycle is means only, so the segment that produces the tail cannot be named

**Severity: medium-high.** For a latency-serving system the tail *is* the question, and the block that decomposes the round trip is the only one in the report with no distribution behind it.

`Segment` (`accounting/tracealign.py`) carries `total_ns`, `count`, and a derived `mean_ns`. Every other timing in the report carries a histogram — `DOMINANT PHASES` prints p50 and p99, `ROUNDTRIPS` prints mean/p50/p95/p99 — but the cross-process decomposition, the one thing a per-process phase table structurally cannot produce, is summarised as a single average.

### What that costs, measured

Grouping the 6,198 round trips of the queue configuration into cohorts by their own total, and taking each cohort's mean per segment:

| cohort | n | round trip | begin→admitted | admitted→computed | computed→replied | replied→end |
|---|---|---|---|---|---|---|
| fastest half | 3,099 | 1,458 µs | 585 µs | 537 µs | 75 µs | 262 µs |
| median decile | 619 | 1,591 µs | 663 µs | 598 µs | 67 µs | 263 µs |
| p90–p99 | 558 | 1,880 µs | 845 µs | 529 µs | 79 µs | 426 µs |
| **slowest 1%** | 62 | **13,453 µs** | **7,469 µs** | **4,666 µs** | 150 µs | 1,168 µs |

The slowest 1% cost 11,862 µs more than the median decile. Of that excess, **57.4% is admission delay** and **34.3% is the forward**; the return hop contributes 7.6%.

The report's mean-only table apportions the round trip 43% / 35% / 4% / **18%**. A reader who takes those shares as a guide to the tail — which is the natural reading, since it is the only breakdown offered — will over-weight the return hop by a factor of 2.4 and under-weight admission. In this package the return hop is precisely the term the shared-memory transport addresses, so the means point at the lever that does *not* fix the tail.

Per-segment percentiles alone do not fully settle it either, and this is worth saying: the p99 of a segment is not necessarily on the same request as the p99 of the total, and here the four segments' p99/p50 ratios (1.53, 1.51, 1.25, 1.89) do not reproduce the 57/34 split above. The statistic that answers the question is the **conditional** one — the breakdown of the slow requests.

### Suggested fix

`accounting/tracealign.py`, `lifecycle_segments`. It already materialises every request's marks in `by_request` and already computes each request's segments one request at a time, so the per-request total is one `sum()` away and the memory order does not change. Sort by that total, take the slowest 1% (or the single slowest, when there are fewer than a hundred requests), and give `Segment` a `tail_total_ns` / `tail_count` pair beside the existing ones.

`accounting/report.py`, `_lifecycle_block`: one extra column, and a note stating the cohort's size and what it cost, so the column cannot be read as anything but conditional:

```
REQUEST LIFECYCLE
roundtrip                         10.54s  (6,198 req)
                                   total   share    per req  slowest 1%
    ├─ begin → admitted            4.51s     43%    726.9us      7.47ms
    ├─ admitted → computed         3.64s     35%    587.8us      4.67ms
    ├─ computed → replied        466.3ms      4%     75.2us     150.2us
    └─ replied → end               1.93s     18%    311.4us      1.17ms

  slowest 1% = each segment's mean over the 61 slowest requests, which took
  13.5ms against a 1.6ms median. A mean cannot say which segment produced a
  tail; this column can.
```

That is the whole finding, in one column, from data the profiler already collects and already discards.

---

## Issue 4 — the request lifecycle exists in the text report and nowhere else

**Severity: medium.** Not a wrong number — a missing output, and the missing one is the block that took the most instrumentation to earn.

`lifecycle_segments` is called from exactly one place: `report._lifecycle_block`. So:

- `lineprofiler report --format json` returns `['caveats', 'findings', 'machine', 'resources', 'roles', 'run', 'workers']`. No lifecycle.
- `lineprofiler report --format html` does not render it, and — because the HTML page embeds `report_as_dict` as its data block — does not carry it in the embedded JSON either.
- `lineprofiler trace --format json` and the timeline page do not carry it, and `trace --fail-over` therefore cannot gate on it.

Every other block of the report is in `report_as_dict`; the docstring's stated purpose is that a merged run "makes a usable assertion target in a test or a CI gate". Wanting the segments in a machine-readable form is not exotic — the reason to instrument a queue boundary at all is usually to watch it over time — and the only route today is to re-implement the decomposition against `merge_run(..., with_trace=True)` and `align_run`, which is literally what had to be written to produce the tables in Issues 2 and 3 above.

There is also a mismatch worth naming: `FINDINGS` is derived from spans only, so the ranked conclusions never mention the lifecycle. On this run finding #1 reads *"`roundtrip/await` spent 97% of its time blocked … service had a phase open for 94% of that wait, so this is a queue"* — true, and one screen above a block that already says *which part* of the queue. Lifecycle marks are trace data, not phase totals, so a finding drawn from them would not violate the rule that findings must rest on spans; but that is a larger change than the rest of this note and is **not** something to bolt on. Carrying the data into the outputs first is the prerequisite, and it is the part worth doing now.

### Suggested fix

`accounting/report.py`, `report_as_dict`: add a `lifecycle` key — channel, segment names in order, totals, counts, means (and the tail figures from Issue 3). Derive the aligned trace there with the existing `_aligned_or_none`.

`accounting/htmlreport.py`, `render_html`: render a section from `report_as_dict(run)["lifecycle"]` — the document is already built there, so this costs no second alignment.

`accounting/cli.py`, `_render_trace`: add `lifecycle` and `dropped_links` to the JSON form.

---

## Issue 5 — `working (on CPU)` is one thread inside named phases, and the report presents it as the role's CPU

**Severity: medium.** The figure is correct for what it measures. Nothing says what it measures, and in this run it accounts for two thirds of a process that has no headroom at all.

### What the report said

```
SERVICE  (1 process, imbalance 1.00)
busy (phase open)              93.7%
working (on CPU)               68.2%
  busy = a phase was open; working = on a CPU inside one. The gap is waiting.
```

The note explains the *gap between the two*. It does not say that both are scoped to the thread that opened the phases. `tracealign.lane_working_share` sums `span.cpu_ns` over leaf spans, and spans exist only on threads that enter a `phase()`.

### What the process was actually doing

Per-thread CPU from `/proc/<pid>/task/*/stat`, sampled over 7.06 s of the queue configuration:

```
process CPU: 7.00 CPU-s = 99% of a core
  tid 217368   (main)     5.76s   81.6% of a core   82.3% of the process
  tid 217600              0.32s    4.5% of a core    4.6% of the process
  tid 217607              0.31s    4.4% of a core    4.4% of the process
  tid 217602              0.31s    4.4% of a core    4.4% of the process
  tid 217601              0.30s    4.3% of a core    4.3% of the process
  (3 further threads used no measurable CPU)
```

Four threads, one per client, at 17.7% of the process between them: the response queues' feeder threads, which is where `multiprocessing.Queue.put` does its pickling. So the service's 99% of a core decomposes as ~68% inside named phases, ~13% on the main thread outside them, and ~18% on threads the lane table cannot see. The service is pinned to one thread by `intra_op_threads=1`, so 99% is its ceiling — and `working (on CPU) 68.2%` invites the opposite conclusion.

The sampler recorded the true figure in the same run (`cpu_percent` ≈ 106–108% on every steady-state row). Both numbers are in the report's own inputs; they are never placed next to each other, and the whole-run `RESOURCES` block pools them across roles so that neither the service's 1.06 cores nor the clients' 0.10 survives into anything printed.

This also silently absorbs the profiler's own flush and sampler threads, which is the one case where a reader particularly wants to know.

### Suggested fix

`accounting/report.py`, `_role_block`. It already receives `run`, so `run.workers_of(role)` and their samples are in hand. Add one line beside the existing two, and one sentence to the note:

```
busy (phase open)              93.7%
working (on CPU)               68.2%
process CPU (all threads)     106.5%  of one core, peak 108.2%
  busy = a phase was open; working = on a CPU inside one. The gap is waiting.
  working covers the threads that open phases, inside them; process CPU is
  every thread of the process, in or out of a phase. A large gap is work on
  threads you have not named.
```

Small, drawn entirely from data the report already loads, and it turns "the service looks two-thirds busy" into "the service is at its ceiling", which is the difference between tuning the transport and adding a second service.

---

## The four issues from the previous round: verified fixed

No GPU was available on this machine, so the three device-related fixes were verified by reading the code actually executing (the local source tree, imported ahead of the installed wheel over `PYTHONPATH`) and by that repository's own test suite, which stubs NVML and torch — not by re-measuring VRAM. Stated plainly because the measurements that originally justified them cannot be reproduced here.

- **`sync=True` opening a CUDA context in a CPU-only process — fixed.** `capabilities.cuda_synchronize(only_when_initialised=True)` returns a wrapper that calls `torch.cuda.synchronize()` only while `torch.cuda.is_initialized()`, and `profiler._resolve_cuda_sync` wires the three-way `cuda_sync` switch to it. The comment that used to claim `is_available()` initialises the driver is gone and the docstring now states the measured 414 MiB correctly.
- **GPU compute reported as "blocked", and called a queue — fixed.** `FLAG_DEVICE_SYNC` is set in `profiler._PhaseScope._record_span` from a live-drain check, and `findings._explain_wait` documents and implements a rule 0 that returns a device explanation before any peer attribution. `report.py` carries the matching exclusions at three sites.
- **The VRAM row that omits the term that scales — fixed.** `sampler` records `cuda_proc_used` per pid, `report` prints `VRAM peak held` beside the allocator row, and `report_as_dict` carries `vram_held_peak`.
- **The `async_work` footnote naming the decisive measurement — fixed, and it is the model for the rest of this note.** `report._async_note` now ends with the batch-1-versus-large-batch comparison and names launch-bound submission as the diagnosis. Nothing in this benchmark ran on a device, so the footnote never fired here; it is present and correct in the source.

---

## What worked well, for calibration

- **The request lifecycle is the reason this profiling session produced an answer at all.** A per-process phase table cannot see the interval between two processes, and that interval is 61% of this round trip. `trace_begin` / `trace_mark` / `trace_end` put it on the page with four call sites and no shared state, and the `sample=` selection-by-key-hash is the right design — it is the only one that keeps a request's checkpoints together across processes. Issues 2, 3 and 4 are all requests to finish this feature, not to replace it; it is the most valuable thing in the package for this workload.

- **`busy` versus `working` per role, read together with the `while X waited, concurrently active: Y` line, settled the bottleneck question in one glance.** `client busy 98.7% / working 3.2%` beside `while client waited, concurrently active: service 94%` is "queued behind the service, not slow, not stalled", stated outright. Issue 5 is a request to state the *scope* of `working`, not a complaint about the pair.

- **`wait%` held up at a granularity I did not expect it to.** A phase that spins on the CPU for a known duration reports `wait 0.0%` at 1 µs, 3 µs, 10 µs and 30 µs, and 0.6–0.7% at 100 µs through 1 ms. So the 42% wait the report showed on a 29.5 µs `write_row` phase is real off-CPU time and not a measurement artefact — which is exactly what I needed to know before chasing it, and I could establish it because the layer is honest enough about `wall - cpu` to make the check obvious.

- **`entries` in `DOMINANT PHASES`, and the counter's `1..4` spread**, are what let the service's batch fill be read at all: 6,200 requests over 3,043 forwards is 2.04 per forward against a `max_batch` of 4, which is the exact number this service's own tuning docs tell a reader to look at. One small thing: the *mean* amount per entry — 2.04 here — is the tuning figure, and it is the one statistic of the three that is not printed. It is recoverable (counter total 6,200 ÷ phase entries 3,043), but it lives on two different lines, and the range `1..4` that *is* printed is the least informative of the three, since a batching service with a cap always reports `1..cap`.

- **The overhead documentation is accurate and was budgeted from, not guessed at.** `trace_mark()` at 1,277 ns and a default phase at 3,909 ns are both in `docs/accounting-recipes.md` and both benchmarked in `bench_accounting.py`, which is what made it possible to decide up front that eleven instrumentation calls per round trip were affordable against a 1.4 ms round trip. Measured end to end the instrumented p50 came out at 1.614 ms against 1.340 ms — a 20% inflation, larger than the call costs alone, the remainder being the 1 Hz snapshot flush this configuration deliberately asks for. Every profiled figure in this document should be read with that 20% in mind; the uninstrumented numbers are quoted beside them for that reason.

- **One trap, which the profiler documents but which is easy to walk into anyway.** `Profiler(run_id=...)` identifies an *attempt*. Deriving that id from a stable configuration name and re-running into the same directory makes `merge_run` read three invocations as one attempt and sum them: 483,213 spans where 43,059 were expected, a report that took over two minutes to render instead of one second, and totals that were three runs deep. The header does say `Processes 15` where five were spawned, so it is discoverable — but the fix is on the caller's side (`benchmarks/bench_service.py` now appends a timestamp to the run id) and it is worth a line in the `run_id` docstring that a *stable* id is only correct for workers of one invocation.
