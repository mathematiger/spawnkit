"""Unit tests for :mod:`spawnkit.processes`.

Two contracts, and both of them are about failure rather than the happy path.

* The ``safe_*`` wrappers must never raise, whatever state the handle is in (``None``, live,
  finished, or already closed). Cleanup runs *after* workers have started dying, so a helper that
  raised would abort the rest of the cleanup and leak the workers it had not reached yet.
* ``shutdown_processes`` must spend **one** grace window on the whole pool, not one per process, and
  ``graceful_shutdown`` must escalate from ``SIGTERM`` to ``SIGKILL`` for a process that ignores the
  polite signal.

Every real child is started through the ``spawn`` context and joined (or killed) in a ``finally``, so
a failing assertion cannot leak a process into the rest of the session.
"""

from __future__ import annotations

import multiprocessing
import signal
import threading
import time
from typing import TYPE_CHECKING, Any, cast

import pytest

from spawnkit.processes import (
    MonitoredThread,
    detach_queue_feeder,
    graceful_shutdown,
    safe_close,
    safe_is_alive,
    safe_join,
    safe_kill,
    safe_terminate,
    shutdown_processes,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_CONTEXT = multiprocessing.get_context("spawn")
_JOIN_TIMEOUT_S = 30.0
_READY_TIMEOUT_S = 30.0


# ── Picklable process targets ─────────────────────────────────────────────────────────────────────
def _idle_forever() -> None:
    """Idle forever; the default SIGTERM disposition still terminates it (a tractable runaway)."""
    while True:
        time.sleep(0.05)


def _ignore_sigterm_then_idle(ready: Any) -> None:
    """Ignore SIGTERM, announce readiness, then idle: only the SIGKILL escalation can stop this."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(0.05)


def _exit_immediately() -> None:
    """Return at once so the parent sees a finished (reapable) process."""
    return


# ── Process helpers ───────────────────────────────────────────────────────────────────────────────
def _spawn(target: Callable[..., None], *args: Any) -> multiprocessing.Process:
    """Start ``target`` in a spawned child.

    The cast is the ``SpawnProcess``/``Process`` split in the type stubs; the two are the same thing
    at runtime, and the module under test is annotated for the latter.
    """
    process = cast("multiprocessing.Process", _CONTEXT.Process(target=target, args=args))
    process.start()
    return process


def _reaped(process: multiprocessing.Process) -> None:
    """Make sure ``process`` is gone, however the test that started it ended."""
    try:
        if process.is_alive():
            process.kill()
        process.join(timeout=_JOIN_TIMEOUT_S)
    except ValueError:  # the test closed the handle itself
        return


@pytest.fixture
def idle_process() -> Iterator[multiprocessing.Process]:
    """A live child that has to be stopped by the test, and is reaped even when the test fails."""
    process = _spawn(_idle_forever)
    try:
        yield process
    finally:
        _reaped(process)


def _finished_process() -> multiprocessing.Process:
    """A started-then-joined process whose handle is still open (not closed)."""
    process = _spawn(_exit_immediately)
    process.join(timeout=_JOIN_TIMEOUT_S)
    return process


def _closed_process() -> multiprocessing.Process:
    """A process handle that has been released, so every attribute access raises ``ValueError``."""
    process = cast("multiprocessing.Process", _CONTEXT.Process(target=_exit_immediately))
    process.close()
    return process


# ── safe_is_alive ─────────────────────────────────────────────────────────────────────────────────
def test_safe_is_alive_reports_no_for_a_missing_handle() -> None:
    """A worker slot the run never filled is not a running worker."""
    assert safe_is_alive(None) is False


def test_safe_is_alive_follows_a_process_from_live_to_finished(idle_process: multiprocessing.Process) -> None:
    """The predicate every other helper is built on."""
    assert safe_is_alive(idle_process) is True
    idle_process.terminate()
    idle_process.join(timeout=_JOIN_TIMEOUT_S)
    assert safe_is_alive(idle_process) is False


def test_safe_is_alive_treats_a_closed_handle_as_dead() -> None:
    """``close()`` makes every later attribute access raise; cleanup asks anyway, and must not die."""
    assert safe_is_alive(_closed_process()) is False


def test_safe_is_alive_accepts_a_thread_too() -> None:
    """The same helper guards thread handles, which is why its parameter is the union."""
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)
    assert safe_is_alive(thread) is False


# ── safe_join ─────────────────────────────────────────────────────────────────────────────────────
def test_safe_join_tolerates_none_finished_and_closed() -> None:
    """All three shapes turn up during a real shutdown, and none of them may raise."""
    safe_join(None, timeout=0.1)
    safe_join(_finished_process(), timeout=0.1)
    safe_join(_closed_process(), timeout=0.1)


def test_safe_join_returns_when_the_timeout_expires(idle_process: multiprocessing.Process) -> None:
    """It is a bounded wait, not a blocking one: a wedged worker must not stall the cleanup."""
    started = time.monotonic()
    safe_join(idle_process, timeout=0.1)
    assert time.monotonic() - started < 5.0
    assert safe_is_alive(idle_process) is True


# ── safe_terminate / safe_kill ────────────────────────────────────────────────────────────────────
def test_safe_terminate_stops_a_live_process(idle_process: multiprocessing.Process) -> None:
    """The polite signal, for a worker that has not wedged."""
    safe_terminate(idle_process)
    idle_process.join(timeout=_JOIN_TIMEOUT_S)
    assert safe_is_alive(idle_process) is False


def test_safe_kill_stops_a_live_process(idle_process: multiprocessing.Process) -> None:
    """The impolite one, for a worker that has."""
    safe_kill(idle_process)
    idle_process.join(timeout=_JOIN_TIMEOUT_S)
    assert safe_is_alive(idle_process) is False


def test_safe_terminate_and_kill_tolerate_none_and_closed() -> None:
    """``safe_is_alive`` guards the closed handle, so both calls are no-ops rather than errors."""
    safe_terminate(None)
    safe_kill(None)
    closed = _closed_process()
    safe_terminate(closed)
    safe_kill(closed)


# ── safe_close ────────────────────────────────────────────────────────────────────────────────────
def test_safe_close_is_idempotent_and_tolerates_none() -> None:
    """Cleanup may reach the same handle twice; the second pass must not turn into an exception."""
    safe_close(None)
    process = _finished_process()
    safe_close(process)
    safe_close(process)
    assert safe_is_alive(process) is False


# ── graceful_shutdown ─────────────────────────────────────────────────────────────────────────────
def test_graceful_shutdown_tolerates_none_and_finished() -> None:
    """A half-built pool has both, and cleanup runs over it exactly as it would over a full one."""
    graceful_shutdown(None, "never started")
    graceful_shutdown(_finished_process(), "finished")


def test_graceful_shutdown_terminates_a_responsive_process(idle_process: multiprocessing.Process) -> None:
    """The common path: SIGTERM is enough, and the kill escalation never runs."""
    graceful_shutdown(idle_process, "runaway", terminate_timeout=2.0, kill_timeout=1.0)
    assert safe_is_alive(idle_process) is False


def test_graceful_shutdown_escalates_to_sigkill_when_sigterm_is_ignored() -> None:
    """A process that ignores SIGTERM survives ``terminate()`` and is stopped only by the escalation.

    Without the escalation this worker outlives the run, holding whatever it was given.
    """
    ready = _CONTEXT.Event()
    process = _spawn(_ignore_sigterm_then_idle, ready)
    try:
        assert ready.wait(timeout=_READY_TIMEOUT_S), "the child never installed its SIGTERM handler"
        graceful_shutdown(process, "stubborn", terminate_timeout=0.2, kill_timeout=2.0)
        assert safe_is_alive(process) is False
    finally:
        _reaped(process)


# ── detach_queue_feeder ───────────────────────────────────────────────────────────────────────────
def test_detach_queue_feeder_tolerates_none_and_a_double_call() -> None:
    """Cancelling an already-cancelled feeder is a normal second pass through cleanup, not an error."""
    detach_queue_feeder(None)
    queue: Any = _CONTEXT.Queue()
    try:
        detach_queue_feeder(queue)
        detach_queue_feeder(queue)
    finally:
        queue.close()


# ── MonitoredThread ───────────────────────────────────────────────────────────────────────────────
def test_monitored_thread_records_the_exception_that_killed_it() -> None:
    """Record the exception, because otherwise the death is invisible.

    :mod:`threading` prints a traceback and the thread stops, while whatever was waiting on its
    output goes on waiting forever.
    """

    def explode() -> None:
        raise RuntimeError("collector died")

    thread = MonitoredThread(target=explode, name="worker")
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert isinstance(thread.exception, RuntimeError)
    assert "collector died" in str(thread.exception)


def test_monitored_thread_sets_the_stop_event_when_it_fails() -> None:
    """Siblings must wind down at once rather than at the next poll of whatever is watching.

    A supervisor that polls would notice eventually; a run with no supervisor at all would never
    notice, which is the case this event exists for.
    """
    stop_event = threading.Event()

    def explode() -> None:
        raise RuntimeError("boom")

    thread = MonitoredThread(target=explode, name="worker", stop_event=stop_event)
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert stop_event.is_set()


def test_monitored_thread_leaves_the_stop_event_alone_on_a_clean_exit() -> None:
    """A worker finishing its job must not be mistaken for one that failed."""
    stop_event = threading.Event()
    thread = MonitoredThread(target=lambda: None, name="worker", stop_event=stop_event)
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert thread.exception is None
    assert not stop_event.is_set()


def test_monitored_thread_without_a_stop_event_still_records_the_failure() -> None:
    """``stop_event`` is optional; omitting it must not turn a crash into an AttributeError."""

    def explode() -> None:
        raise ValueError("no event here")

    thread = MonitoredThread(target=explode, name="worker")
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert isinstance(thread.exception, ValueError)


def test_monitored_thread_catches_base_exception_not_just_exception() -> None:
    """A ``SystemExit`` raised in a worker thread ends it exactly as silently as an unhandled error.

    ``except Exception`` would let it through, which is the whole failure mode this class exists to
    close — so the wider catch is behaviour, not defensiveness.
    """
    stop_event = threading.Event()

    def leave() -> None:
        raise SystemExit("worker exited")

    thread = MonitoredThread(target=leave, name="worker", stop_event=stop_event)
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert isinstance(thread.exception, SystemExit)
    assert stop_event.is_set()


def test_monitored_thread_does_not_re_raise_into_the_joiner() -> None:
    """The exception is recorded, not propagated: ``join()`` returns normally on a dead worker."""

    def explode() -> None:
        raise RuntimeError("boom")

    thread = MonitoredThread(target=explode, name="worker")
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert not thread.is_alive()
    assert thread.exception is not None


def test_monitored_thread_passes_target_args_and_kwargs_through() -> None:
    """``stop_event`` is popped before ``Thread.__init__``; everything else must survive intact."""
    seen: list[tuple[int, int]] = []
    thread = MonitoredThread(
        target=lambda a, b: seen.append((a, b)),
        args=(1,),
        kwargs={"b": 2},
        name="worker",
        stop_event=threading.Event(),
    )
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)

    assert seen == [(1, 2)]
    assert thread.name == "worker"


# ── shutdown_processes ────────────────────────────────────────────────────────────────────────────
def test_shutdown_processes_reaps_workers_that_exit_on_their_own() -> None:
    """The common path: every worker saw the stop event and is already winding down."""
    processes = [_spawn(_exit_immediately) for _ in range(2)]
    try:
        shutdown_processes([(f"worker_{index}", process) for index, process in enumerate(processes)])
        assert not any(safe_is_alive(process) for process in processes)
    finally:
        for process in processes:
            _reaped(process)


def test_shutdown_processes_kills_a_worker_that_ignores_sigterm() -> None:
    """The escalation reaches the pool path too, or a wedged worker outlives the run."""
    ready = _CONTEXT.Event()
    process = _spawn(_ignore_sigterm_then_idle, ready)
    try:
        assert ready.wait(timeout=_READY_TIMEOUT_S), "the child never installed its SIGTERM handler"
        shutdown_processes([("stubborn", process)], graceful_timeout=0.2, terminate_timeout=0.2)
        assert not safe_is_alive(process)
    finally:
        _reaped(process)


def test_shutdown_processes_shares_one_grace_window_across_the_pool() -> None:
    """Prove the pool shares one grace window rather than one per process.

    Joining each worker for its own timeout costs ``num_workers x timeout``; at realistic worker
    counts that is a minute-long cleanup, long enough that a second Ctrl+C forces an unclean
    ``os._exit()`` and drops the buffered log tail — which is the tail explaining why the run ended.
    Five runaway workers under a 0.2 s window must therefore cost ~0.2 s of *waiting*, not 1.0 s.
    """
    processes = [_spawn(_idle_forever) for _ in range(5)]
    try:
        started = time.monotonic()
        shutdown_processes(
            [(f"worker_{index}", process) for index, process in enumerate(processes)],
            graceful_timeout=0.2,
            terminate_timeout=1.0,
            kill_timeout=0.5,
        )
        elapsed = time.monotonic() - started

        assert not any(safe_is_alive(process) for process in processes)
        # Generous bound: the assertion is "one shared window", not a benchmark. A per-process
        # window would spend 5 x 0.2 s waiting before terminating anything.
        assert elapsed < 0.7, f"shutdown took {elapsed:.2f}s - the grace window is not being shared"
    finally:
        for process in processes:
            _reaped(process)


class _SlowJoinProcess:
    """A handle that records the join timeouts it was handed and then dies on ``terminate()``."""

    join_cost_s = 0.05

    def __init__(self) -> None:
        self.join_timeouts: list[float] = []
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(float(timeout if timeout is not None else -1.0))
        time.sleep(self.join_cost_s)

    def terminate(self) -> None:
        self._alive = False


def test_shutdown_processes_spends_the_window_down_across_the_pool() -> None:
    """The same property, read off the timeouts themselves rather than off the clock.

    Each handle is given whatever is left of the shared deadline, so the timeouts shrink as the pool
    is walked and then floor at the small positive value that keeps the last joins polling rather
    than blocking forever.
    """
    handles = [_SlowJoinProcess() for _ in range(5)]
    graceful_timeout = 0.2

    shutdown_processes(
        cast("Any", [(f"worker_{index}", handle) for index, handle in enumerate(handles)]),
        graceful_timeout=graceful_timeout,
        terminate_timeout=0.2,
        kill_timeout=0.2,
    )

    timeouts = [handle.join_timeouts[0] for handle in handles]
    assert len(timeouts) == 5
    assert timeouts[0] <= graceful_timeout
    assert timeouts == sorted(timeouts, reverse=True), f"the window is not shrinking: {timeouts}"
    assert min(timeouts) == pytest.approx(0.05), "the exhausted window must floor, not go negative"
    assert sum(timeouts) < 5 * graceful_timeout


def test_shutdown_processes_tolerates_none_and_unstarted_entries() -> None:
    """Cleanup runs on a half-built pool too: a worker the run never started is a ``None`` slot."""
    shutdown_processes([("never_built", None), ("also_none", None)])


def test_shutdown_processes_accepts_an_empty_pool() -> None:
    """A run with no child processes at all still calls the same cleanup."""
    shutdown_processes([])
