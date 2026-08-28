# Repository conventions

Working agreements for `spawnkit`. Everything here is enforceable: a rule that no tool checks is a
rule that drifts, so each section names the command that proves it.

## What the package is

`spawnkit` is the layer below the trainer — process hygiene, worker supervision and batched GPU
inference for spawn-mode multiprocessing. The package docstring in `src/spawnkit/__init__.py` is the
canonical statement of scope and positioning. If a change makes that docstring wrong, the docstring
is what has to be revised first, in the same commit.

## The layering rule

Three tiers, and a user may take only the first. Imports run **downward only**:

```
tier 1  hygiene      hygiene.py, seeding.py
                     |  imports nothing from tier 2 or tier 3
tier 2  supervision  monitor.py, run.py, oom.py, processes.py, supply.py, lifecycle.py
                     |  may import tier 1; imports nothing from tier 3
tier 3  service      service/
                     |  may import tier 1 and tier 2
```

Consequences that are not negotiable:

- **Tier 1 and tier 2 are stdlib + numpy.** No `import torch` at module scope anywhere outside
  `service/`. `pip install spawnkit` must stay a two-second install; torch arrives only with the
  `[torch]` extra.
- **Tier 1 never learns about supervision.** `hygiene.prepare_cpu_only_worker` must be usable by
  someone who has never heard of `WorkerMonitor`.
- `_log.py` is shared internal plumbing and belongs to no tier; it may be imported from anywhere and
  may import nothing but the standard library.
- A `TYPE_CHECKING`-only import that crosses a tier upward is still a violation. Restructure instead.

New public names go in `src/spawnkit/__init__.py`'s `__all__`, alphabetically, in the same commit
that adds them.

## No invented numbers

Every figure that appears in the README, in `docs/`, in a docstring or in a commit message —
latency, throughput, VRAM saved, shutdown time, speed-up factor — must trace to a committed JSON
file under `benchmarks/results/`. Cite the file inline, next to the number.

- `benchmarks/results/` is deliberately **not** git-ignored. Those files are evidence, not build
  output.
- If a number is needed and no result file exists, write `TODO(results): <what is missing>` and stop.
  Do not estimate, do not round up something remembered, do not carry a figure over from the code
  this was extracted from — different machine, different measurement.
- Re-running a benchmark replaces the JSON file and updates every number that cites it, in one
  commit. A README figure and its result file are never allowed to disagree.
- The procedure lives in `skills/bench/SKILL.md` (see *Procedure documents* below).

## Two interpreters

This repository is its own clean-install gate, and that only works if the default interpreter cannot
see torch.

**Clean interpreter (default for everything).**

```
/mnt/home/dkoehler/all_projects/spawnkit/.venv/bin/python
```

It has `spawnkit` installed editable, the `[dev]` extra, and **no torch**. Every routine command
runs here:

```bash
.venv/bin/python scripts/hygiene_gate.py
.venv/bin/python -m pytest -q -m "not gpu"
.venv/bin/ruff check .
.venv/bin/mypy src tests scripts
```

Because torch is absent, an accidental top-level `import torch` in tier 1 or tier 2 fails
immediately here rather than surviving to a user's machine. That is the point of the venv, so do not
"fix" a failure by installing torch into it.

**Torch interpreter (tier 3 and benchmarks only).**

There is a second, torch-enabled interpreter elsewhere on this machine. Its path is *not* recorded
in this repository: it lives inside an unrelated private project whose name the hygiene gate
forbids, and hardcoding it would defeat the gate. Export it instead, once per shell:

```bash
export SPAWNKIT_TORCH_PYTHON=/path/to/a/torch-enabled/bin/python
```

and reach `spawnkit` from it over `PYTHONPATH` rather than installing into it:

```bash
PYTHONPATH="$PWD/src" "$SPAWNKIT_TORCH_PYTHON" -m pytest -q -m gpu
PYTHONPATH="$PWD/src" "$SPAWNKIT_TORCH_PYTHON" benchmarks/<name>.py
```

`PYTHONPATH` keeps the torch environment untouched and guarantees the code under test is this
working tree, not an installed copy.

## The hygiene gate is a pre-push requirement

`scripts/hygiene_gate.py` fails on any vendor or institution name, machine-local absolute path,
cluster scheduler flag, private package name, or leaked-credential shape anywhere in the tree.

```bash
.venv/bin/python scripts/hygiene_gate.py    # must exit 0 before every push
```

`.github/workflows/hygiene.yml` runs that same file, so a green local run and a green CI run mean
the same thing. The pattern list lives in the script and nowhere else — never duplicate it into YAML.

The gate's exclude list is short and every entry is argued in the script's module docstring. Adding
an entry means writing that argument. In particular: **this file is exempt from the home-directory
path pattern and nothing else is.** It is the one document that has to name a local interpreter to
be useful, and that exemption is scoped to this filename alone.

## Procedure documents

Two multi-step procedures are written down rather than remembered, as `SKILL.md` files under
`skills/` inside the tooling dot-directory at the repository root. That directory's name is fixed by
external tooling and is one of the strings the hygiene gate forbids in file *content*, which is why
this file refers to those documents by the `skills/<name>/SKILL.md` tail of their path:

- `skills/release/SKILL.md` — cutting a release.
- `skills/bench/SKILL.md` — running benchmarks and updating the numbers they justify.

## Release

Versions live in two places and must agree: `version` in `pyproject.toml` and `__version__` in
`src/spawnkit/__init__.py`. The full procedure — bump, changelog, tag, watch the publish workflow,
verify on PyPI, verify a clean `pip install` — is `skills/release/SKILL.md`. Publishing uses
PyPI Trusted Publishing over OIDC; there is no API token and no repository secret, so nothing about a
release is meant to be done by hand outside that procedure.

## Style

- Ruff with `line-length = 120`, numpy docstring convention; mypy with `disallow_untyped_defs`.
  `scripts/` is held to the same standard as `src/` — docstrings and annotations included.
- Every public module, class and function gets a numpy-style docstring: parameters, returns, shapes,
  raises. Prose that is longer than a docstring belongs in `docs/`.
- One purpose per function; if the name needs an "and", split it. Public functions first, private
  helpers below.
- Surgical diffs. Do not reformat, rename or "improve" code the change did not have to touch.
- Do not delete a `TODO`. Resolve it or leave it.
