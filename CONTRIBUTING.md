# Contributing to spawnkit

Thanks for taking the time. `spawnkit` is small on purpose, so the bar is less "is this useful" than
"does this belong below the trainer" — a change that only makes sense for one training loop probably
belongs in that training loop.

## Getting set up

```bash
git clone https://github.com/mathematiger/spawnkit
cd spawnkit
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

That environment has **no torch**, and that is deliberate: it is the clean-install gate. The hygiene,
supervision and run-identity tiers are stdlib + numpy, and an accidental top-level `import torch` in
them has to fail here rather than on a user's machine. Work on the service tier (`src/spawnkit/service/`)
needs a second, torch-enabled interpreter; reach this working tree from it with
`PYTHONPATH="$PWD/src"` rather than installing into it.

## Before you open a pull request

All four must pass locally. CI runs the same commands on Python 3.10 through 3.13.

```bash
.venv/bin/python scripts/hygiene_gate.py       # forbidden-string gate; must exit 0
.venv/bin/ruff check .
.venv/bin/mypy                                 # the paths come from pyproject's `files`
PATH="$PWD/.venv/bin:$PATH" scripts/ci_pytest.sh local
```

Neither `mypy` nor the test run takes an explicit path or flag here, and that is the point: both
have drifted from what CI ran before, and green locally then meant red on push. The check set lives
in `pyproject.toml` and the pytest invocation in `scripts/ci_pytest.sh`, which is the same script
both CI jobs call.

If you have a CUDA device, also run the GPU-marked tests:

```bash
PYTHONPATH="$PWD/src" /path/to/torch/python -m pytest -q -m gpu
```

## The hygiene gate

`scripts/hygiene_gate.py` fails the build if the tree contains a vendor or institution name, a
machine-local absolute path, a cluster scheduler flag, a private package name, or the shape of a
leaked credential. It is read-only — it reports, and a human decides the right wording.

Run it before every push. If it fires on something legitimate, the fix is almost always to reword;
adding an exclude is a last resort and requires an argument written into the script's module
docstring next to the existing four. The pattern list lives in that script and nowhere else, so
never copy it into a workflow file.

## Design constraints that reviews will hold you to

**The layering rule.** Imports run downward only:

```
tier 1  hygiene      hygiene.py, seeding.py                      stdlib + numpy
tier 2  supervision  monitor.py, run.py, oom.py, processes.py,    stdlib + numpy
                     supply.py, lifecycle.py
tier 3  service      service/                                    needs the [torch] extra
```

Tier 1 knows nothing of tier 2; tier 2 knows nothing of tier 3. A `TYPE_CHECKING`-only import that
crosses upward is still a violation. `pip install spawnkit` must stay a two-second, torch-free
install.

**No invented numbers.** Every performance figure in the README, the docs, a docstring or a changelog
entry must trace to a committed JSON file under `benchmarks/results/`, cited next to the figure.
`benchmarks/results/` is not git-ignored; those files are evidence. If a claim needs a number that no
file supports, leave a `TODO(results): ...` and leave the claim out.

**Style.** Ruff at `line-length = 120` with the numpy docstring convention; mypy with
`disallow_untyped_defs`. `scripts/` is held to the same standard as `src/`. Every public module,
class and function gets a numpy-style docstring with parameters, returns, shapes and raises. One
purpose per function — if the name needs an "and", split it.

**Surgical diffs.** Do not reformat, rename or improve code your change did not have to touch. Do not
delete a `TODO`; resolve it or leave it.

## Tests

New behaviour needs a test. Tests that need a CUDA device carry the `gpu` marker so CI can deselect
them; tests slower than a couple of seconds carry `slow`. A bug fix starts with a test that
reproduces the bug.

Supervision code is concurrency code, and flaky tests there are worse than no tests: prefer
deterministic signals (a process exit code, a queue that has actually drained) over sleeps.

## Commit messages and pull requests

Write the subject line in the imperative and say what changed about observable behaviour, not which
files moved. Keep one logical change per pull request; a refactor and a fix in the same diff cost a
reviewer far more than two pull requests do.

Pull requests should state what problem the change solves, how it was verified, and — for anything
touching the tiers — which tier the code lives in and why.

## Reporting bugs

Use the issue templates. For a supervision bug, the useful details are: how many workers, spawn or
fork, which tier's API you called, the exit codes you saw, and whether the run hung or exited. A
minimal reproduction that runs on CPU is worth more than a stack trace.

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0, the same
licence as the project.
