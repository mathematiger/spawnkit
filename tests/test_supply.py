"""Unit tests for :mod:`spawnkit.supply`.

The module is three tiny predicates, and every one of them is a trap that can cost a multi-hour run.
The tests are written around the *confusions* rather than the happy path:

* a crash and a clean exit look identical from outside (both: thread not alive) and mean opposite
  things, so :func:`thread_crash_reason` and :func:`thread_stopped_reason` must disagree on exactly
  one case;
* :func:`no_live_producers` returns ``True`` for an empty pool, which is right for one kind of caller
  and wrong for the other — the test pins the value so neither caller's guard can be "simplified"
  away.
"""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

from spawnkit.processes import MonitoredThread
from spawnkit.supply import (
    SupplyStalledError,
    no_live_producers,
    thread_crash_reason,
    thread_stopped_reason,
)

_JOIN_TIMEOUT_S = 5.0


def _crashed_thread(error: BaseException) -> MonitoredThread:
    """Build a MonitoredThread that has run, raised ``error`` and finished."""

    def explode() -> None:
        raise error

    thread = MonitoredThread(target=explode, name="worker")
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)
    assert not thread.is_alive()
    return thread


def _finished_thread() -> MonitoredThread:
    """Build a MonitoredThread that has run to completion without raising."""
    thread = MonitoredThread(target=lambda: None, name="worker")
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)
    assert not thread.is_alive()
    return thread


# ---------------------------------------------------------------------------
# thread_crash_reason
# ---------------------------------------------------------------------------


def test_a_recorded_exception_is_reported_with_its_repr() -> None:
    """The reason string is the whole diagnosis a caller gets, so it must carry the error itself."""
    reason = thread_crash_reason(_crashed_thread(RuntimeError("mmap failed")), "collector")
    assert reason is not None
    assert "collector" in reason
    assert "mmap failed" in reason


def test_a_clean_exit_is_not_a_crash() -> None:
    """The distinction the whole module exists for: finishing is not failing.

    This is what lets the check stay meaningful during shutdown, when every worker thread is
    expected to be gone and only a recorded exception still signals a fault.
    """
    assert thread_crash_reason(_finished_thread(), "collector") is None


def test_a_missing_thread_is_never_a_crash() -> None:
    """``None`` is a role this run never started, not a role that died."""
    assert thread_crash_reason(None, "trainer") is None


def test_a_plain_thread_without_the_attribute_is_not_reported_as_crashed() -> None:
    """``getattr`` default, not ``thread.exception``: a bare Thread must not raise AttributeError."""
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)
    assert thread_crash_reason(thread, "collector") is None


def test_a_base_exception_that_is_not_an_exception_is_still_recorded() -> None:
    """Record a ``SystemExit`` too, since it ends a worker thread just as silently as an error.

    ``MonitoredThread`` catches ``BaseException`` for exactly this reason: a bare
    ``except Exception`` would let it through, which is the failure mode the class exists to close.
    """
    reason = thread_crash_reason(_crashed_thread(SystemExit("worker exited")), "collector")
    assert reason is not None
    assert "worker exited" in reason


# ---------------------------------------------------------------------------
# thread_stopped_reason
# ---------------------------------------------------------------------------


def test_a_crash_is_reported_by_the_stopped_check_too() -> None:
    """The crash reason wins, so the caller sees the cause rather than "is not running"."""
    reason = thread_stopped_reason(_crashed_thread(RuntimeError("boom")), "collector")
    assert reason is not None
    assert "boom" in reason


def test_a_thread_that_finished_without_raising_counts_as_stopped() -> None:
    """The one case the two predicates disagree on, and the reason both of them exist."""
    thread = _finished_thread()
    assert thread_crash_reason(thread, "collector") is None
    assert thread_stopped_reason(thread, "collector") == "collector is not running"


def test_a_live_thread_is_not_stopped() -> None:
    """Runs on every consumer iteration, so the healthy answer has to be ``None``."""
    release = threading.Event()
    thread = MonitoredThread(target=release.wait, name="worker")
    thread.start()
    try:
        assert thread_stopped_reason(thread, "collector") is None
    finally:
        release.set()
        thread.join(timeout=_JOIN_TIMEOUT_S)


def test_a_never_started_thread_reads_as_not_running() -> None:
    """Report a never-started thread as stopped, the same as one that has already finished.

    ``is_alive()`` is False in both cases, and for a worker the caller believes it launched, "never
    started" is exactly as fatal as "already finished".
    """
    thread = MonitoredThread(target=lambda: None, name="worker")
    assert thread_stopped_reason(thread, "collector") == "collector is not running"


def test_a_missing_thread_is_never_stopped() -> None:
    """``None`` is a role this run does not have, not a role that died."""
    assert thread_stopped_reason(None, "trainer") is None


# ---------------------------------------------------------------------------
# no_live_producers
# ---------------------------------------------------------------------------


class _Process:
    """Stands in for a spawned worker; only ``is_alive`` is consulted."""

    def __init__(self, *, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class _ClosedProcess:
    """Stands in for a process whose handle has already been released."""

    def is_alive(self) -> bool:
        raise ValueError("process object is closed")


def test_a_pool_with_one_survivor_is_still_producing() -> None:
    """Producers die independently; only losing all of them stops work reaching the consumer."""
    pool = cast("Any", [_Process(alive=False), _Process(alive=True), _Process(alive=False)])
    assert not no_live_producers(pool)


def test_a_pool_where_everyone_died_has_no_producers() -> None:
    """The condition a consumer loop is actually watching for."""
    assert no_live_producers(cast("Any", [_Process(alive=False), _Process(alive=False)]))


def test_an_empty_pool_counts_as_no_producers() -> None:
    """Pinned deliberately, because the two kinds of caller need opposite things from this answer.

    A run whose producers work until a stop event pairs this with "we expected producers at all", so
    that a consumer-only run still stops when asked; a run whose producers stop at a work budget
    pairs it with a non-empty check, so a serial path that starts no processes is not diagnosed as
    "every producer exited" on its first iteration. Flipping this to ``False`` would silently break
    the first; deleting either caller's guard would break the other.
    """
    assert no_live_producers([])


def test_a_none_entry_is_treated_as_dead_rather_than_raising() -> None:
    """A worker slot that was never filled must not turn liveness checking into an AttributeError."""
    assert no_live_producers(cast("Any", [None, None]))
    assert not no_live_producers(cast("Any", [None, _Process(alive=True)]))


def test_a_process_whose_handle_was_closed_counts_as_dead() -> None:
    """``safe_is_alive`` swallows the ValueError a closed handle raises.

    Shutdown reaps and closes processes, so liveness gets asked about closed handles routinely; a
    raising predicate would abort cleanup partway and leak the workers it had not reached.
    """
    assert no_live_producers(cast("Any", [_ClosedProcess()]))


def test_a_generator_of_processes_is_accepted() -> None:
    """The parameter is an iterable, so a caller may pass a comprehension rather than a list."""
    assert no_live_producers(cast("Any", (_Process(alive=False) for _ in range(3))))


# ---------------------------------------------------------------------------
# SupplyStalledError
# ---------------------------------------------------------------------------


def test_supply_stalled_is_a_runtime_error() -> None:
    """Callers that already catch RuntimeError around a run keep working."""
    assert issubclass(SupplyStalledError, RuntimeError)
    with pytest.raises(RuntimeError, match="nothing can reach the consumer"):
        raise SupplyStalledError("nothing can reach the consumer")
