# Benchmark procedure

Every performance claim `spawnkit` makes is about someone else's wall clock, so every claim has to be
reproducible from a file in this repository. This document is how a number gets from a machine into
the README, and it is also the only sanctioned path.

## The rule, stated once

**No number is ever invented, estimated, remembered or carried over.**

A figure may appear in the README, in `docs/`, in a docstring, in a changelog entry or in a commit
message **only if** a committed file under `benchmarks/results/` contains it, and the prose cites
that file. Concretely:

- Not allowed: "roughly 400 MiB of VRAM", "about 3x faster", "cuts shutdown from minutes to seconds",
  a figure copied from a different project, a figure from a benchmark that was run but not committed.
- Allowed: a number read out of `benchmarks/results/<name>.json`, with the file named next to it.
- If the claim is worth making and no result file exists, the benchmark has to be written and run
  first. Until then, write `TODO(results): <what is missing>` and leave the claim out.

`benchmarks/results/` is deliberately not git-ignored. Those JSON files are evidence, and a reviewer
must be able to check any published figure without re-running anything.

## 1. Pick the interpreter

Benchmarks that touch the service tier need torch, which the repository's own `.venv` deliberately
does not have. Use the torch-enabled interpreter over `PYTHONPATH` so the benchmark measures this
working tree:

```bash
export SPAWNKIT_TORCH_PYTHON=/path/to/a/torch-enabled/bin/python   # see the conventions document
export PYTHONPATH="$PWD/src"
```

Tier-1 and tier-2 benchmarks (hygiene, seeding, supervision, shutdown timing) are stdlib + numpy and
run under `.venv/bin/python`. Prefer that where it works: fewer moving parts in the measurement.

## 2. Record the machine

A number without its machine is not a number. Every result file carries an `environment` block, and
it is filled in from the machine, not from memory:

```bash
"$SPAWNKIT_TORCH_PYTHON" -c "
import platform, sys
print(platform.platform()); print(sys.version)
try:
    import torch
    print(torch.__version__, torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except ImportError:
    print('no torch')
"
nproc; free -g | head -2
```

Do not put a hostname, a cluster name, a scheduler flag or an absolute home path into the result
file. The hygiene gate scans `benchmarks/results/` like everything else, and it will reject them.
Describe the machine by its hardware ("64 physical cores, one 40 GB datacentre GPU"), not by its
address.

## 3. Run everything under `benchmarks/`

```bash
ls benchmarks/*.py
for bench in benchmarks/*.py; do
    echo "=== $bench"
    "$SPAWNKIT_TORCH_PYTHON" "$bench" --out benchmarks/results/"$(basename "${bench%.py}")".json
done
```

Discipline that decides whether the numbers mean anything:

- **Warm up, then measure.** Report a median and a spread over repeats, never a single timing.
- **One variable at a time.** A benchmark that changes two things measures neither.
- **Idle machine.** A shared or loaded machine produces a number that will not reproduce; note the
  load or do not run.
- **Repeat count and seed go in the file**, so a rerun is a rerun and not a new experiment.

## 4. The result file shape

One JSON file per benchmark, named after it, in `benchmarks/results/`. Self-describing, so a reader
needs nothing but the file:

```json
{
  "benchmark": "shutdown_latency",
  "spawnkit_version": "0.1.0",
  "commit": "<git rev-parse --short HEAD>",
  "timestamp_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "environment": {
    "python": "3.12.x",
    "platform": "Linux-x86_64",
    "torch": "2.x.y+cu128 | null",
    "gpu": "one 40 GB datacentre GPU | null",
    "cpu_cores": 64
  },
  "parameters": {"workers": 8, "repeats": 20, "seed": 0},
  "metrics": {
    "median_seconds": 0.0,
    "p90_seconds": 0.0,
    "baseline_median_seconds": 0.0
  },
  "notes": "what the baseline is and what was held fixed"
}
```

Commit the result file in the same commit as any prose that cites it. A README figure and its
evidence never land separately.

## 5. Update the prose — only from the files

Read the number out of the file; do not retype it from the terminal:

```bash
.venv/bin/python -c "
import json, pathlib
for path in sorted(pathlib.Path('benchmarks/results').glob('*.json')):
    data = json.loads(path.read_text())
    print(path, data['metrics'])
"
```

Then edit the README or `docs/`, and cite the file next to every figure, for example:

> Shutting down eight workers takes a median of 0.4 s rather than 40 s
> (`benchmarks/results/shutdown_latency.json`).

Round in the prose if you like, but round *down* a benefit and *up* a cost, and never past the
precision the file supports.

## 6. Re-running invalidates the old prose

When a benchmark is re-run, the JSON file is replaced, and **every figure in the repository that
cites that file is updated in the same commit**. Find them before editing:

```bash
grep -rn "results/<name>.json" README.md docs/ src/ CHANGELOG.md
```

A figure whose result file has moved on is a wrong figure, not an out-of-date one.

## 7. Close the loop

```bash
.venv/bin/python scripts/hygiene_gate.py       # result files are scanned like everything else
git status --short benchmarks/results/          # the evidence must actually be staged
```

## Checklist

- [ ] Ran on an idle machine; warm-up discarded; repeats and seed recorded.
- [ ] `environment` block filled in from the machine, with no hostname, cluster name or home path.
- [ ] One JSON file per benchmark under `benchmarks/results/`, committed.
- [ ] Every changed figure in README or `docs/` cites its result file.
- [ ] No figure anywhere that no committed file supports.
- [ ] Hygiene gate exits 0.
