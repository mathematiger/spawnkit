# spawnkit

Process hygiene, worker supervision and batched GPU inference for spawn-mode multiprocessing in
PyTorch.

```bash
pip install spawnkit           # hygiene + supervision: stdlib and numpy only
pip install spawnkit[torch]    # adds the batched inference service
```

The [README](https://github.com/mathematiger/spawnkit#readme) is the introduction: the four failures
this exists for, the measured numbers, and an honest list of when not to use it. These pages are the
reference.

## Reading order

Every module's docstring says not just what the module does but *why it is shaped that way*, usually
naming the failure that shaped it. They are meant to be read.

1. [`spawnkit.hygiene`](api/spawnkit/hygiene.md) — the parent/child split, and why it cannot move.
2. [`spawnkit.oom`](api/spawnkit/oom.md) — why memory exhaustion needs its own policy.
3. [`spawnkit.monitor`](api/spawnkit/monitor.md) — four supervision policies, in a load-bearing order.
4. [`spawnkit.service`](api/spawnkit/service/index.md) — one process owning the model, N clients.

## A note on the numbers

Every figure in the README and in these docs is read from a committed file under
`benchmarks/results/`, written by the scripts in `benchmarks/`. Nothing is estimated, and a claim
without a result file behind it is not published. Two of the effects depend on how loaded your
machine is, so the benchmarks are meant to be re-run on yours — see [Benchmarks](benchmarks.md).
