"""The benchmark harness's one piece of logic that can silently corrupt a published number.

``write_results`` picks where a result file lands. Getting that wrong does not raise — it overwrites
a committed measurement with one taken under different conditions, which then reads as a regression
(or an improvement) that never happened. That is the failure this file exists to catch, and it is
the only thing in ``_harness`` worth a test: the timing helpers are measured against a real clock,
so a unit test of them would assert the clock works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks._harness import RESULTS_DIR, RESULTS_DIR_ENV, Measurement, write_results


def _measurement() -> Measurement:
    return Measurement(
        name="probe", iterations=10, p50_ms=1.0, p99_ms=2.0, mean_ms=1.5, min_ms=0.9, ops_per_s=100.0
    )


def test_results_land_in_the_committed_directory_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var means the published location, because that is what a bare run should update."""
    monkeypatch.delenv(RESULTS_DIR_ENV, raising=False)
    path = write_results("unit_probe.json", [_measurement()])
    try:
        assert path.parent == RESULTS_DIR
    finally:
        path.unlink(missing_ok=True)


def test_the_env_var_redirects_results_away_from_the_committed_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A redirected run must not touch ``benchmarks/results/`` at all.

    This is what lets a before/after comparison run both arms in one allocation without the second
    arm overwriting the first — the only way to tell a code change from a busy machine.
    """
    monkeypatch.setenv(RESULTS_DIR_ENV, str(tmp_path))
    path = write_results("unit_probe.json", [_measurement()])

    assert path == tmp_path / "unit_probe.json"
    assert not (RESULTS_DIR / "unit_probe.json").exists()
    assert json.loads(path.read_text())["measurements"][0]["name"] == "probe"


def test_a_redirect_creates_the_directory_it_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An arm names its own output directory; requiring it to exist first is a step to forget."""
    target = tmp_path / "arm" / "old"
    monkeypatch.setenv(RESULTS_DIR_ENV, str(target))
    path = write_results("unit_probe.json", [_measurement()])

    assert path.parent == target
    assert path.is_file()


def test_an_empty_redirect_falls_back_to_the_committed_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """``export VAR=`` in a shell script sets it to empty rather than unsetting it.

    Treating that as "write to the current working directory" would scatter result files wherever
    the job happened to start, so an empty value means "not set".
    """
    monkeypatch.setenv(RESULTS_DIR_ENV, "")
    path = write_results("unit_probe.json", [_measurement()])
    try:
        assert path.parent == RESULTS_DIR
        assert path.parent != Path()
    finally:
        path.unlink(missing_ok=True)
