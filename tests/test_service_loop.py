"""The worker skeleton, and the one piece of it that is not obvious: the fast-retry burst.

Build, resync, collect, handle. Three of those are simple enough that the tests read like their
docstrings. The fourth is a small state machine spread across two functions, and it exists because
both of its neighbouring policies were wrong first:

* retrying a failed weight sync **not at all** serves stale weights until the next scheduled sync,
  which for a sync that merely raced the trainer's write is a needless window of staleness;
* retrying it **forever** is worse, because an allocation failure never clears and the loop then
  logs the same warning for the rest of the job while quietly serving the same stale weights.

So a failure opens a burst of at most :data:`~spawnkit.service.loop._MAX_FAST_SYNC_RETRIES` forced
retries and normal cadence resumes; a forced retry that succeeds ends the burst immediately. Both
halves are counted exactly here, by scripting the collector and counting sync attempts, so nothing
in this file waits on anything.

Every exception raised in these tests is deliberately *not* an out-of-memory error. Both functions
route an OOM through :func:`~spawnkit.oom.abort_worker_on_oom`, which calls ``os._exit`` — a test
that tripped it would take the session with it.
"""

from __future__ import annotations

import logging
import multiprocessing
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("torch")

from spawnkit.service.loop import (
    IDLE,
    STOP,
    build_model_or_stop,
    maybe_sync_weights,
    run_worker_loop,
)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

pytestmark = pytest.mark.timeout(30)

NEVER_SCHEDULED = 1000
"""A sync interval so long that only the very first iteration is on schedule.

That leaves the fast-retry burst as the only thing that can attempt a sync, which is what makes the
attempt count an exact assertion rather than an approximate one.
"""

MAX_FAST_RETRIES = 5
"""The cap the loop enforces, restated here so a change to it fails a test rather than passing one."""


class ScriptedCollector:
    """Returns a fixed sequence of work items, then :data:`STOP`, so the loop always terminates."""

    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self.calls = 0

    def __call__(self) -> Any:
        """Hand out the next scripted item, or stop the loop once the script runs out."""
        self.calls += 1
        if not self._items:
            return STOP
        return self._items.pop(0)


class CountingSync:
    """A weight sync that counts its attempts and fails for the first ``failures`` of them."""

    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.attempts = 0

    def __call__(self) -> None:
        """Attempt a sync, raising a non-OOM error while failures remain."""
        self.attempts += 1
        if self.attempts <= self._failures:
            msg = "the trainer's weights were being written"
            raise RuntimeError(msg)


class Recorder:
    """Collects the work items ``handle_fn`` was given."""

    def __init__(self) -> None:
        self.handled: list[Any] = []

    def __call__(self, work: Any) -> None:
        """Record one handled item."""
        self.handled.append(work)


def failing_build() -> Any:
    """A model build that fails for a reason no retry can fix, and that is not an OOM."""
    msg = "the checkpoint on disk was truncated"
    raise RuntimeError(msg)


@pytest.fixture
def stop_flag() -> Event:
    """The run's shared stop flag, of the type the worker functions are annotated for."""
    return multiprocessing.Event()


def test_build_model_or_stop_returns_the_model(stop_flag: Event) -> None:
    """The happy path returns the built object and leaves the run alone."""
    sentinel = object()

    result = build_model_or_stop(lambda: sentinel, stop_flag, "service")

    assert result is sentinel
    assert not stop_flag.is_set()


def test_a_build_failure_logs_stops_the_run_and_returns_none(
    stop_flag: Event,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A worker that cannot build its model can make no progress, so the whole run ends."""
    with caplog.at_level(logging.ERROR, logger="spawnkit"):
        result = build_model_or_stop(failing_build, stop_flag, "service")

    assert result is None
    assert stop_flag.is_set()
    assert any("failed to build model" in message for message in caplog.messages)
    assert any("truncated" in message for message in caplog.messages)


def test_maybe_sync_weights_skips_an_off_schedule_iteration() -> None:
    """Off schedule and not forced: no attempt at all, and nothing for the caller to retry."""
    sync = CountingSync(failures=0)

    failed = maybe_sync_weights(sync, iters=3, interval=10, name="service")

    assert failed is False
    assert sync.attempts == 0


def test_maybe_sync_weights_syncs_on_schedule() -> None:
    """On schedule, a successful sync also reports ``False`` — the two are indistinguishable to the
    caller on purpose, because neither one calls for a retry.
    """
    sync = CountingSync(failures=0)

    failed = maybe_sync_weights(sync, iters=10, interval=10, name="service")

    assert failed is False
    assert sync.attempts == 1


def test_maybe_sync_weights_reports_a_failure_for_the_caller_to_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``True`` means *attempted and failed*, which is the only case that opens a burst."""
    sync = CountingSync(failures=1)

    with caplog.at_level(logging.WARNING, logger="spawnkit"):
        failed = maybe_sync_weights(sync, iters=0, interval=10, name="service")

    assert failed is True
    assert sync.attempts == 1
    assert any("weight sync failed" in message for message in caplog.messages)


def test_force_syncs_regardless_of_the_schedule() -> None:
    """The burst's forced retries have to bypass the schedule, or there would be no burst."""
    sync = CountingSync(failures=0)

    failed = maybe_sync_weights(sync, iters=7, interval=10, name="service", force=True)

    assert failed is False
    assert sync.attempts == 1


def test_the_loop_leaves_on_stop(stop_flag: Event) -> None:
    """``STOP`` from the collector ends the loop without handling anything."""
    handle = Recorder()
    collect = ScriptedCollector([])

    run_worker_loop(stop_flag, NEVER_SCHEDULED, CountingSync(failures=0), collect, handle, "service")

    assert handle.handled == []
    assert collect.calls == 1


def test_the_loop_leaves_when_the_stop_event_is_already_set(stop_flag: Event) -> None:
    """The stop flag is checked before the collector, so a stopped run does no further work."""
    stop_flag.set()
    collect = ScriptedCollector(["work"])

    run_worker_loop(stop_flag, NEVER_SCHEDULED, CountingSync(failures=0), collect, Recorder(), "service")

    assert collect.calls == 0


def test_an_idle_poll_keeps_looping_without_handling(stop_flag: Event) -> None:
    """Nothing arrived in the poll window; that is the normal state of an unloaded service."""
    handle = Recorder()
    collect = ScriptedCollector([IDLE, IDLE, "work"])

    run_worker_loop(stop_flag, NEVER_SCHEDULED, CountingSync(failures=0), collect, handle, "service")

    assert handle.handled == ["work"]
    assert collect.calls == 4


def test_a_work_item_reaches_handle_fn(stop_flag: Event) -> None:
    """In order, and only the work items — the sentinels never reach the handler."""
    handle = Recorder()
    collect = ScriptedCollector(["first", IDLE, "second"])

    run_worker_loop(stop_flag, NEVER_SCHEDULED, CountingSync(failures=0), collect, handle, "service")

    assert handle.handled == ["first", "second"]


def test_a_failing_sync_opens_a_bounded_burst_and_then_reverts_to_cadence(
    stop_flag: Event,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sync that never recovers costs one scheduled attempt plus the capped burst, and no more.

    With the interval set beyond the run's length, the only scheduled attempt is the one on the
    first iteration. Everything after it is the burst, so the total attempt count is exactly
    ``1 + MAX_FAST_RETRIES`` however many more iterations the loop goes round.
    """
    sync = CountingSync(failures=1000)
    iterations = 12
    collect = ScriptedCollector([f"work-{index}" for index in range(iterations)])
    handle = Recorder()

    with caplog.at_level(logging.WARNING, logger="spawnkit"):
        run_worker_loop(stop_flag, NEVER_SCHEDULED, sync, collect, handle, "service")

    assert len(handle.handled) == iterations
    assert sync.attempts == 1 + MAX_FAST_RETRIES
    assert sum("weight sync failed" in message for message in caplog.messages) == 1 + MAX_FAST_RETRIES


def test_a_forced_retry_that_succeeds_ends_the_burst_early(stop_flag: Event) -> None:
    """The burst is for a sync racing the trainer's write, and one retry is usually all it takes."""
    sync = CountingSync(failures=1)
    collect = ScriptedCollector([f"work-{index}" for index in range(8)])

    run_worker_loop(stop_flag, NEVER_SCHEDULED, sync, collect, Recorder(), "service")

    # The scheduled attempt on iteration one failed; the first forced retry succeeded and closed the
    # burst, so the remaining iterations are back on the (never-reached) schedule.
    assert sync.attempts == 2


def test_an_exhausted_burst_is_reopened_by_the_next_scheduled_sync(stop_flag: Event) -> None:
    """Reverting to cadence is not a permanent give-up — but it is a real gap, not a busy loop.

    With an interval of 8 and a sync that never recovers, each scheduled attempt opens a burst of
    five, the loop then goes two iterations with no attempt at all, and the next scheduled iteration
    opens the next burst: 1 + 5, skip, skip, 1 + 5, skip, skip, 1 + 3 over twenty work items, plus
    one last forced retry on the iteration that collects the stop (the sync runs before the collect).
    """
    sync = CountingSync(failures=1000)
    handle = Recorder()
    collect = ScriptedCollector([f"work-{index}" for index in range(20)])

    run_worker_loop(stop_flag, 8, sync, collect, handle, "service")

    assert len(handle.handled) == 20
    assert sync.attempts == 17
    assert sync.attempts < len(handle.handled)  # some iterations attempted nothing
    assert sync.attempts > 1 + MAX_FAST_RETRIES  # and the schedule opened more than one burst
