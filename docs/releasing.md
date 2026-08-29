# Release procedure

Cutting a release is mechanical, and it is written down because the two failure modes are both
silent: a version that disagrees between `pyproject.toml` and `__version__`, and a tag pushed on a
tree that never passed its own gate. Follow the steps in order. Do not skip step 0.

## 0. Preconditions — refuse to proceed if any fails

The release stops here unless all four are true. This is not advisory; a red gate means the tag does
not get created.

```bash
.venv/bin/python scripts/hygiene_gate.py       # must exit 0
.venv/bin/ruff check .                          # must be clean
.venv/bin/mypy src tests scripts                # must be clean
.venv/bin/python -m pytest -q -m "not gpu"      # must pass
```

Then confirm the remote state:

```bash
git status --porcelain                          # must be empty: no uncommitted work
git rev-parse --abbrev-ref HEAD                 # must be main
gh run list --branch main --limit 5             # the latest ci and hygiene runs must be success
```

If `gh run list` shows the newest commit's `ci` or `hygiene` run as anything but `success`, stop and
fix that first. Releasing on top of a red or still-running CI is the thing this step exists to
prevent.

## 1. Decide the version

Semantic versioning. Pre-1.0, a breaking change to a public name bumps the minor. The version string
is used verbatim in the tag, so decide it once: `X.Y.Z`, and the tag is `vX.Y.Z`.

## 2. Bump it in *both* places

The two must agree exactly. Nothing checks this automatically, which is why it gets its own step.

1. `pyproject.toml` → `[project] version = "X.Y.Z"`
2. `src/spawnkit/__init__.py` → `__version__ = "X.Y.Z"`

Verify:

```bash
grep -n '^version' pyproject.toml
grep -n '^__version__' src/spawnkit/__init__.py
.venv/bin/python -c "import spawnkit, importlib.metadata as m; \
    assert spawnkit.__version__ == m.version('spawnkit'), (spawnkit.__version__, m.version('spawnkit')); \
    print('version agrees:', spawnkit.__version__)"
```

The last command requires a re-install of the editable package if the metadata is stale
(`.venv/bin/pip install -e ".[dev]"`), which is itself a useful check that the build still works.

## 3. Write the changelog section

Generate the raw material from the commits since the previous tag:

```bash
PREV=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
if [ -n "$PREV" ]; then git log --no-merges --pretty='- %s' "$PREV"..HEAD; \
                   else git log --no-merges --pretty='- %s'; fi
```

Then edit it into a `CHANGELOG.md` section — do not paste the raw log. Prepend, newest first:

```markdown
## X.Y.Z — YYYY-MM-DD

### Added
### Changed
### Fixed
### Removed
```

Rules for the prose:

- Drop commits that changed nothing a user can observe (formatting, test-only edits, typo fixes).
- Write from the caller's side: what changed about the behaviour of a public name, not which file
  moved.
- A performance claim in the changelog is a number, and every number obeys the no-invented-numbers
  rule: it cites a file under `benchmarks/results/`. If there is no result file, the claim does not
  go in.
- Name every removed or renamed public symbol explicitly. A rename is a breaking change.

Commit the bump and the changelog together:

```bash
git add pyproject.toml src/spawnkit/__init__.py CHANGELOG.md
git commit -m "Release X.Y.Z"
git push origin main
```

Wait for `ci` and `hygiene` to go green on that commit before tagging.

## 4. Tag and push the tag

```bash
git tag -a vX.Y.Z -m "spawnkit X.Y.Z"
git push origin vX.Y.Z
```

The tag push is what triggers `.github/workflows/publish.yml`. Nothing else does.

## 5. Watch the publish workflow

```bash
gh run watch "$(gh run list --workflow=publish.yml --limit 1 --json databaseId -q '.[0].databaseId')"
```

The workflow runs `ci` and `hygiene` as reusable jobs first, then builds with `python -m build`,
checks the artifacts with `twine check`, and uploads with Trusted Publishing over OIDC. There is no
API token and no repository secret; if the upload step fails with a permissions or trust error, the
fix is in the PyPI project's publisher settings, never a token added to the repository.

## 6. Verify on PyPI

```bash
curl -s https://pypi.org/pypi/spawnkit/json | python -c \
    "import json,sys; d=json.load(sys.stdin); print(d['info']['version']); print(sorted(d['releases'])[-5:])"
```

The reported version must be `X.Y.Z`. Index propagation can lag the upload by a minute or two; retry
before concluding anything went wrong.

## 7. Verify a clean install

The point of the thin dependency list is that a fresh install is fast and torch-free. Prove it in a
throwaway environment, outside the repository:

```bash
SCRATCH=$(mktemp -d)
python -m venv "$SCRATCH/venv"
"$SCRATCH/venv/bin/pip" install --quiet "spawnkit==X.Y.Z"
"$SCRATCH/venv/bin/python" -c "import spawnkit; print(spawnkit.__version__)"
"$SCRATCH/venv/bin/python" -c "import sys; assert 'torch' not in sys.modules; print('torch-free import: ok')"
"$SCRATCH/venv/bin/pip" list | grep -i torch && echo "UNEXPECTED torch dependency" && exit 1
rm -rf "$SCRATCH"
```

The printed version must match the tag. A torch entry in `pip list` means a tier-1 or tier-2 module
grew a torch import and the layering rule was broken — yank the release rather than shipping it.

## 8. Publish the GitHub release

```bash
gh release create vX.Y.Z --title "spawnkit X.Y.Z" --notes-file <(sed -n '/^## X.Y.Z/,/^## /p' CHANGELOG.md)
```

## If something goes wrong

- **Publish failed before upload.** Fix, commit, delete and re-push the tag
  (`git tag -d vX.Y.Z && git push --delete origin vX.Y.Z`), then tag again.
- **Publish succeeded with a bad artifact.** PyPI versions are immutable. Do not attempt to reuse the
  version: yank it on PyPI and release `X.Y.Z+1` with the fix.
- **The two version strings disagreed and it shipped.** Same answer: patch release, both places, and
  add the assertion from step 2 to the release checklist you actually run.
