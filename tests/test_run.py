"""Run identity: the claim has to be atomic, and the run has to write down what it got.

Two runs sharing an output directory is the failure that reports success. Nothing raises, both jobs
finish, and the damage shows up later in whichever artefacts one of them overwrote — a checkpoint
pruner that keeps the highest-numbered files will delete a fresh run's output as fast as it is
written, because those are always the lowest-numbered ones present, while the older run beside it
looks perfectly healthy.

So the tests here are about the two halves of preventing that:

* :func:`~spawnkit.claim_run_dir` must make the *creation* of the directory the claim. A
  check-then-create would let two jobs starting in the same second both decide the name was free,
  which is the bug this replaces — hence a concurrency test rather than only a sequential one.
* :func:`~spawnkit.write_run_manifest` must record the tag the run actually got. A driver that
  re-derives paths from the tag it *passed in* reads a directory nobody wrote the moment a suffix
  had to be added.

Every test that touches the scheduler variables clears all four first: this suite has to give the
same answer on a workstation and inside a batch job.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import pytest

from spawnkit import claim_run_dir, default_run_tag, write_run_manifest

SCHEDULER_JOB_ID_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID")
"""The documented priority order, restated here so a reordering in the library fails a test."""


@pytest.fixture
def no_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as if off a batch scheduler, whatever the machine running the suite is."""
    for var in SCHEDULER_JOB_ID_VARS:
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# claim_run_dir
# ---------------------------------------------------------------------------


def test_claiming_a_free_name_creates_the_directory_and_returns_it(tmp_path: Path) -> None:
    """The claim is the directory: nothing else has to happen for the name to be taken."""
    directory, tag = claim_run_dir(tmp_path, "experiment")

    assert directory == tmp_path / "experiment"
    assert tag == "experiment"
    assert directory.is_dir()


def test_the_base_directory_is_created_when_it_is_missing(tmp_path: Path) -> None:
    """First run of a project should not have to be preceded by a ``mkdir``."""
    directory, _ = claim_run_dir(tmp_path / "runs" / "project", "experiment")

    assert directory.is_dir()


def test_a_taken_name_is_suffixed_rather_than_shared(tmp_path: Path) -> None:
    """The second claimer gets its own directory instead of the first one's."""
    first, first_tag = claim_run_dir(tmp_path, "experiment")
    second, second_tag = claim_run_dir(tmp_path, "experiment")

    assert (first_tag, second_tag) == ("experiment", "experiment_2")
    assert first != second
    assert second.is_dir()


def test_suffixes_keep_climbing_past_the_first_collision(tmp_path: Path) -> None:
    """Numbering continues rather than starting over, so the third run is not the second's."""
    tags = [claim_run_dir(tmp_path, "experiment")[1] for _ in range(3)]

    assert tags == ["experiment", "experiment_2", "experiment_3"]


def test_a_directory_left_behind_by_something_else_is_stepped_over(tmp_path: Path) -> None:
    """An empty stale directory is enough to shift the name, which is why the tag is returned.

    One directory created by hand, or by a job that died before it wrote anything, turns
    ``experiment`` into ``experiment_2`` for every run after it. A caller that re-derives paths from
    the name it asked for is then looking in a directory this run never used.
    """
    (tmp_path / "experiment").mkdir()

    _, tag = claim_run_dir(tmp_path, "experiment")

    assert tag == "experiment_2"


def test_the_returned_tag_matches_the_directory_name(tmp_path: Path) -> None:
    """The tag names the run everywhere else too, so it has to be the resolved one."""
    directory, tag = claim_run_dir(tmp_path, "experiment")
    other_directory, other_tag = claim_run_dir(tmp_path, "experiment")

    assert directory.name == tag
    assert other_directory.name == other_tag


def test_concurrent_claimers_never_share_a_directory(tmp_path: Path) -> None:
    """The race the atomic ``mkdir`` exists for: many jobs starting on the same name at once.

    A sweep that submits its arms together and names them after the project hits this every launch.
    All the claimers are released from a barrier so they are genuinely contending; each must come
    away with a directory of its own, and every one of those directories must exist — a claim
    reported but not created is the same collision one step later.
    """
    claimers = 12
    barrier = threading.Barrier(claimers)
    lock = threading.Lock()
    claimed: list[Path] = []

    def claim() -> None:
        barrier.wait(timeout=10.0)
        directory, _ = claim_run_dir(tmp_path, "sweep")
        with lock:
            claimed.append(directory)

    threads = [threading.Thread(target=claim, name=f"claimer-{index}") for index in range(claimers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)

    assert len(claimed) == claimers, "a claimer failed or deadlocked instead of claiming"
    assert len(set(claimed)) == claimers, "two claimers were handed the same directory"
    assert all(directory.is_dir() for directory in claimed)


# ---------------------------------------------------------------------------
# default_run_tag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("var", SCHEDULER_JOB_ID_VARS)
def test_a_scheduler_job_id_becomes_the_tag(
    var: str,
    monkeypatch: pytest.MonkeyPatch,
    no_scheduler: None,
) -> None:
    """Under a batch scheduler the job id is the most useful name a run can have.

    It is the identifier the queue, the accounting records and the job's own output file already
    agree on, so a directory named after it needs no cross-referencing later.
    """
    monkeypatch.setenv(var, "12345")

    assert default_run_tag() == "job_12345"


def test_the_job_id_variables_are_read_in_their_documented_order(
    monkeypatch: pytest.MonkeyPatch,
    no_scheduler: None,
) -> None:
    """With more than one set, the order decides — and a machine can have several set at once.

    Dropping them one at a time walks the whole priority list in a single pass, so a reordering
    anywhere in it fails here rather than only for whichever variable a test happened to pick.
    """
    for var in SCHEDULER_JOB_ID_VARS:
        monkeypatch.setenv(var, f"id-for-{var}")

    for var in SCHEDULER_JOB_ID_VARS:
        assert default_run_tag() == f"job_id-for-{var}"
        monkeypatch.delenv(var)


def test_an_empty_job_id_is_not_a_job_id(monkeypatch: pytest.MonkeyPatch, no_scheduler: None) -> None:
    """A variable exported as empty must not produce the tag ``job_``, which every run would share."""
    monkeypatch.setenv("SLURM_JOB_ID", "")
    monkeypatch.setenv("PBS_JOBID", "678")

    assert default_run_tag() == "job_678"


def test_the_fallback_tag_carries_the_timestamp_and_the_pid(no_scheduler: None) -> None:
    """Off a scheduler the pid is what keeps two runs started in the same second apart.

    The timestamp alone is not enough: launching a handful of runs from one shell script puts them
    all in the same second, and a shared tag there is the collision this module exists to prevent.
    """
    tag = default_run_tag()

    assert re.fullmatch(rf"run_\d{{8}}_\d{{6}}_{os.getpid()}", tag), tag


def test_the_fallback_tag_is_usable_as_a_directory_name(tmp_path: Path, no_scheduler: None) -> None:
    """Whatever it is made of, it has to be something a directory can be called."""
    directory, tag = claim_run_dir(tmp_path, default_run_tag())

    assert directory.is_dir()
    assert directory.name == tag


# ---------------------------------------------------------------------------
# write_run_manifest
# ---------------------------------------------------------------------------


def test_the_manifest_round_trips_as_json(tmp_path: Path) -> None:
    """The manifest is read by the next stage of a pipeline, so it has to parse."""
    manifest = tmp_path / "manifest.json"

    write_run_manifest(manifest, {"tag": "experiment_2", "steps": 1000, "resumed": False})

    assert json.loads(manifest.read_text()) == {"tag": "experiment_2", "steps": 1000, "resumed": False}


def test_path_values_are_written_as_strings(tmp_path: Path) -> None:
    """Paths are the whole point of the manifest, and JSON has no type for one."""
    manifest = tmp_path / "manifest.json"
    run_dir = tmp_path / "runs" / "experiment_2"

    write_run_manifest(manifest, {"run_dir": run_dir, "checkpoints": run_dir / "checkpoints"})

    written = json.loads(manifest.read_text())
    assert written == {"run_dir": str(run_dir), "checkpoints": str(run_dir / "checkpoints")}
    assert Path(written["run_dir"]) == run_dir


def test_the_manifest_records_the_resolved_tag_rather_than_the_requested_one(tmp_path: Path) -> None:
    """The failure the manifest exists for, written out end to end.

    The caller asks for ``experiment``, gets ``experiment_2`` because a stale directory was in the
    way, and every downstream stage that re-derives the path from the requested name reads an empty
    directory. Writing down what the run actually got is what closes it.
    """
    (tmp_path / "experiment").mkdir()
    directory, tag = claim_run_dir(tmp_path, "experiment")
    manifest = tmp_path / "manifest.json"

    write_run_manifest(manifest, {"tag": tag, "run_dir": directory})

    written = json.loads(manifest.read_text())
    assert written["tag"] == "experiment_2"
    assert Path(written["run_dir"]).is_dir()


def test_missing_parent_directories_are_created(tmp_path: Path) -> None:
    """The manifest often lands beside output that does not exist yet."""
    manifest = tmp_path / "reports" / "nested" / "manifest.json"

    write_run_manifest(manifest, {"tag": "experiment"})

    assert manifest.is_file()


def test_an_existing_manifest_is_replaced(tmp_path: Path) -> None:
    """A rerun writes the current run's paths, not a mixture with the previous run's."""
    manifest = tmp_path / "manifest.json"

    write_run_manifest(manifest, {"tag": "first", "extra": 1})
    write_run_manifest(manifest, {"tag": "second"})

    assert json.loads(manifest.read_text()) == {"tag": "second"}


def test_a_string_path_is_accepted(tmp_path: Path) -> None:
    """Callers hand this whatever their configuration held, which is usually a string."""
    manifest = tmp_path / "manifest.json"

    write_run_manifest(str(manifest), {"tag": "experiment"})

    assert manifest.is_file()


@pytest.mark.parametrize("path", [None, ""])
def test_no_path_means_no_manifest(path: str | None, tmp_path: Path) -> None:
    """Writing a manifest is opt-in: no destination is a no-op, not an error.

    A library that raised here would make the manifest mandatory for every caller who does not
    want one.
    """
    write_run_manifest(path, {"tag": "experiment"})

    assert list(tmp_path.iterdir()) == []


def test_an_empty_manifest_is_still_written(tmp_path: Path) -> None:
    """Empty *entries* is not the same as no path: the file is the signal that the run got this far."""
    manifest = tmp_path / "manifest.json"

    write_run_manifest(manifest, {})

    assert json.loads(manifest.read_text()) == {}


@pytest.mark.parametrize(
    "tag",
    ["../escaped", "../../escaped", "nested/name", "/absolute/path", "", ".", ".."],
)
def test_a_tag_that_would_escape_the_base_directory_is_refused(tmp_path: Path, tag: str) -> None:
    """A run tag is joined onto a path, so a separator in it relocates the whole run.

    Tags come from config files, CLI arguments and scheduler variables. Measured before this check:
    ``claim_run_dir(base, "../../escaped")`` created a directory two levels *above* the base and an
    absolute tag ignored the base entirely — silently, after which the run's output went there. An
    empty tag is refused for the neighbouring reason: it resolves to the base itself, so the next
    run lands *beside* the base rather than inside it.
    """
    with pytest.raises(ValueError, match="run tag"):
        claim_run_dir(tmp_path, tag)


def test_an_ordinary_tag_is_still_accepted(tmp_path: Path) -> None:
    """The guard must not reject the names people actually use."""
    for tag in ("run_1", "job_1234567", "sweep-2026-08-29", "a.b.c", "run_20260829_120000_4242"):
        directory, resolved = claim_run_dir(tmp_path, tag)
        assert resolved == tag
        assert directory.parent == tmp_path
