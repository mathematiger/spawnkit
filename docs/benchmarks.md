# Benchmarks

Every number in the README and these docs comes from a committed JSON file under
`benchmarks/results/`, written by the scripts in `benchmarks/`. Nothing is estimated.

```bash
pip install spawnkit[bench] gymnasium
python -m benchmarks.bench_threads   --contenders 8   # thread oversubscription
python -m benchmarks.bench_service   --clients 8      # round trip, per transport
python -m benchmarks.bench_shutdown  --workers 4 8 16 # teardown strategies
python -m benchmarks.bench_hygiene   --workers 6      # VRAM held by CPU-only workers (needs a GPU)
```

## Re-run them on your machine

Two of the effects depend on how loaded your box is, and the absolutes are machine-specific in any
case. The *shapes* are what transfer.

Concretely, for the service round trip: on a node also running four unrelated training jobs, the
committed `service_cuda.json` figures do not reproduce — every transport is roughly 15 % slower per
call and aggregate throughput is about 2.3x lower. What holds is the comparison the table is making,
shared-memory + CUDA graph against the plain queue: 2.2x on the quiet node the file was written on,
1.96x on the loaded one. Read the ratio as the claim and the milliseconds as the machine.

## Three ways these benchmarks were wrong first

Both are encoded in the scripts now, and both are worth knowing before you write your own.

**A microbenchmark measured the wrong condition.** The thread-oversubscription benchmark originally
timed one process alone and reported that the problem did not exist — 1.7x — while the end-to-end
service benchmark was measuring a 400x difference from the same cause. The effect only appears when
the service competes with worker processes for cores, which is the deployment it is for and not the
condition a microbenchmark defaults to. Hence `--contenders`. It is also exactly 1.0x at batch 1,
so a single-client smoke test hides it independently.

**Spawn skew was reported as tail latency.** The service benchmark's clients come up seconds apart,
so the first client timed its early calls against an idle service and the last against a busy one:
p99 650 ms against a p50 of 2 ms, none of it real. Every client now warms up, waits at a barrier,
and only then starts its clock.

**A committed result file was used as the control arm.** After changing the serving path, the
service benchmark was re-run and read 2.3x lower throughput than `service_cuda.json` — which looks
exactly like a regression, and is not one. Checking out the commit that *wrote* that file and
running it on the same node the same afternoon reproduced the same 2.3x shortfall: the machine had
changed, not the code. A results file records the machine it was written on, and a busy node is a
different machine.

So a before/after question needs both arms in **one allocation**, interleaved A/B/A/B, with the old
code in a `git worktree` rather than reconstructed by hand. Interleaving is what makes the answer
falsifiable: if the two repeats of an arm disagree by as much as the arms disagree with each other,
there is no effect to report. That is what happened here — 2.273 and 2.389 ms for the old code
against 2.307 and 2.437 for the new — and the honest conclusion was "no measurable difference"
rather than either the regression or the improvement the unpaired runs each suggested.

The general lesson is the same in all three: a benchmark that disagrees with another measurement is
usually measuring a different condition, not disproving the effect.

## Adding one

Use `benchmarks/_harness.py`. It separates warm-up from measurement, reports p50/p99 rather than a
mean, keeps the timed region to two clock reads, and requires an explicit `sync` for anything on a
GPU — a device result that was not synchronised is submission time wearing compute's clothes. Write
results with `write_results()`, which stamps the machine into every file, because a latency figure
without its CPU, GPU and torch version cannot be compared with anything.
