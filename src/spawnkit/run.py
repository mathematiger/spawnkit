"""Run identity: give every run its own directory, atomically, and write down where it went.

Two runs that share an output directory corrupt each other silently, and the damage is worst where
it is least visible. A checkpoint pruner that globs a directory and keeps the highest-numbered files
will delete a fresh run's checkpoints as fast as it writes them — they are always the lowest-numbered
ones present — while the older run beside it looks perfectly healthy. Both jobs report success. One
of them produced nothing.

:func:`claim_run_dir` closes that by making directory creation the claim, and
:func:`write_run_manifest` closes the follow-on failure: a driver script that re-derives paths from
the tag it *passed in* reads the wrong ones the moment the tag had to be suffixed.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spawnkit._log import get_logger

log = get_logger(__name__)

_SCHEDULER_JOB_ID_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID")
"""Batch schedulers whose job id, when present, is the most useful name a run can have."""


def default_run_tag() -> str:
    """Name an otherwise untagged run: its scheduler job id, or a timestamp plus pid.

    An empty tag means "write straight into the base directory", which every untagged run then
    shares — and a sweep that submits seven concurrent jobs under one project name and no tag has
    all seven pruning each other's output. Every run gets a name of its own; this supplies one when
    the caller did not.

    :return: the tag, e.g. ``job_1234567`` under a scheduler or ``run_20260828_143005_91142`` off it.
    """
    for var in _SCHEDULER_JOB_ID_VARS:
        job_id = os.environ.get(var)
        if job_id:
            return f"job_{job_id}"
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def claim_run_dir(base_dir: Path | str, tag: str) -> tuple[Path, str]:
    """Create the first unused ``base_dir/<tag>``, ``base_dir/<tag>_2``, ... and return it and its name.

    ``mkdir(exist_ok=False)`` is what claims the name, and it is why this is race-free: directory
    creation is atomic, so of two jobs starting in the same second exactly one wins and the other
    gets ``FileExistsError`` and moves to the next suffix. A check-then-create would let both win,
    which is the bug this replaces rather than a theoretical concern.

    The resolved tag is returned because it names the run everywhere else too — checkpoints, log
    files, experiment-tracker run names. **Write it back onto your config before any worker spawns**,
    or the workers will use the tag you asked for and the parent will use the one it got.

    :param base_dir: the directory runs live under; created if missing.
    :param tag: the preferred name, e.g. from :func:`default_run_tag`. Must be a single directory
        name — see :func:`_check_tag_is_a_single_name` for why anything else is refused.
    :return: ``(directory, resolved_tag)``.
    :raises ValueError: if ``tag`` is empty, absolute, or contains a path separator.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from spawnkit import claim_run_dir
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     first = claim_run_dir(Path(tmp), "sweep")[1]
    ...     second = claim_run_dir(Path(tmp), "sweep")[1]
    >>> first, second
    ('sweep', 'sweep_2')
    """
    _check_tag_is_a_single_name(tag)
    base = Path(base_dir)
    suffix = 1
    while True:
        candidate = base / (tag if suffix == 1 else f"{tag}_{suffix}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            suffix += 1
        else:
            return candidate, candidate.name


def write_run_manifest(path: Path | str | None, entries: Mapping[str, Any]) -> None:
    """Write a run's **resolved** paths to ``path`` as JSON; a ``None`` path is a no-op.

    The manifest exists for the next stage of a pipeline. A driver that re-derives the output
    directory from the tag it passed in reads a directory no run ever wrote as soon as
    :func:`claim_run_dir` had to add a suffix — one stale empty directory is enough to turn ``train``
    into ``train_2`` and send every downstream stage looking in the wrong place. Have the run write
    down where it actually went, and have the next stage read that.

    The write is atomic — a temporary file in the same directory, then :func:`os.replace`. A reader
    racing the writer is the normal case here (the next stage often polls for this file), and a
    partial read of a half-written manifest is a wrong path rather than a missing one, which fails
    much later and much less clearly.

    :param path: where to write the manifest; parent directories are created. ``None`` disables it.
    :param entries: the resolved values, e.g. ``{"tag": ..., "run_dir": ..., "checkpoints": ...}``.
        Values must be JSON-serialisable; :class:`~pathlib.Path` is converted for you, at any depth.
    """
    if not path:
        return
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_stringify_paths(entries), indent=2)

    # Same directory, so os.replace is a rename within one filesystem and therefore atomic.
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    log.info("Run manifest: %s", manifest_path)


def _stringify_paths(value: Any) -> Any:
    """Convert every :class:`~pathlib.Path` in a nested structure to a string.

    Recursive because a manifest naturally holds lists of paths — the checkpoints a stage wrote, the
    shards it produced — and a top-level-only conversion turns those into a bare ``TypeError`` at the
    very end of an otherwise successful run.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value


def _check_tag_is_a_single_name(tag: str) -> None:
    """Refuse a tag that would place the run somewhere other than inside ``base_dir``.

    A tag is usually a config value, a CLI argument or a scheduler variable, and it is joined onto a
    path — so a tag containing ``..`` or a leading ``/`` silently relocates the entire run. Measured
    before this check: ``claim_run_dir(base, "../../escaped")`` created a directory two levels
    *above* the base, and an absolute tag ignored the base completely and wrote to the path it
    named. Neither raised, and the run's output then went there.

    An empty tag is refused for a related reason: it resolves to ``base_dir`` itself, so the first
    run claims the base directory and the next becomes ``base_2`` *beside* it rather than within it.

    :param tag: the caller's preferred run name.
    :raises ValueError: if the tag is not a single, ordinary directory name.
    """
    if not tag or tag in {".", ".."}:
        msg = f"run tag {tag!r} is empty or a directory reference; it must name a directory"
        raise ValueError(msg)
    if os.sep in tag or (os.altsep and os.altsep in tag) or Path(tag).is_absolute():
        msg = (
            f"run tag {tag!r} contains a path separator, so it would place the run outside "
            "base_dir rather than inside it. A tag must be a single directory name."
        )
        raise ValueError(msg)
