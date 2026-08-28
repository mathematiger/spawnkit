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

## Two ways these benchmarks were wrong first

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

The general lesson is the same in both cases: a benchmark that disagrees with an end-to-end
measurement is usually measuring a different condition, not disproving the effect.

## Adding one

Use `benchmarks/_harness.py`. It separates warm-up from measurement, reports p50/p99 rather than a
mean, keeps the timed region to two clock reads, and requires an explicit `sync` for anything on a
GPU — a device result that was not synchronised is submission time wearing compute's clothes. Write
results with `write_results()`, which stamps the machine into every file, because a latency figure
without its CPU, GPU and torch version cannot be compared with anything.
