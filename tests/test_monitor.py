"""What the supervisor decides, and — the load-bearing half — in what order it decides it.

Four policies run on every pass: sweep for out-of-memory deaths, stop on a critical death, note a
tolerated death once, respawn what has budget left. Each one on its own is a couple of lines. The
value is in the sequence, because three of the four verdicts are *wrong answers* for an OOM:

* as a restartable worker it is respawned into the same exhausted node, once per tick, for ever;
* as a non-critical worker it is logged once and the run continues without a feature it needs;
* as an ordinary critical death the run stops — but *cleanly*, and the scheduler records success.

So the OOM tests below are written as "this worker would also qualify as X" cases: each one puts a
worker in front of the monitor that a reordered implementation would classify as something else, and
asserts both the raise and that the other policy never ran. Reorder the four calls in ``watch`` and
they fail; that is what they are for.

The rest covers the two ends of the loop that are easy to get wrong: a tolerated death must be
logged *once* rather than once per tick, and the post-loop re-check must still find a thread's OOM
after the loop has already ended — because a thread that dies of one sets the stop event itself, and
the run would otherwise look like a clean, requested shutdown.

Tick counting, and the doubles used throughout, are described in ``conftest.py``.
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import TYPE_CHECKING

import pytest
from conftest import FakeProcess, FakeThread, MonitorDriver

from spawnkit import (
    OOM_EXIT_CODE,
    MonitoredThread,
    OutOfMemoryAbortError,
    WorkerMonitor,
    WorkerSpec,
)

if TYPE_CHECKING:
    import multiprocessing

TOLERATED_DEATH_MESSAGE = "died - continuing without it"


class RestartRecorder:
    """A ``restart_fn`` that counts its calls and hands back a worker that is dead on arrival.

    Dead on arrival on purpose: a worker that fails for a permanent reason — a bad config, a missing
    file — dies again immediately after every respawn, which is the only case in which the restart
    budget is doing anything.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> multiprocessing.Process:
        """Record one restart and return the replacement handle."""
        self.calls += 1
        return FakeProcess(alive=False, name=f"restarted-{self.calls}")


class DyingProcess(FakeProcess):
    """A live process that dies after ``polls_before_death`` liveness checks.

    Ends a watch after a known amount of *looking* rather than a known amount of time, which is what
    lets a test assert "the monitor went round several times and still only reported once" without
    waiting for a clock.
    """

    def __init__(self, polls_before_death: int, name: str = "dying") -> None:
        super().__init__(alive=True, name=name)
        self.polls_before_death = polls_before_death
        self.polls = 0

    def is_alive(self) -> bool:
        """Report alive until the poll budget is spent."""
        self.polls += 1
        return self.polls <= self.polls_before_death


# ---------------------------------------------------------------------------
# Policy 1 — out-of-memory deaths are swept ahead of every other verdict
# ---------------------------------------------------------------------------


def test_an_oom_death_beats_a_restart_that_still_has_budget(watch_driver: MonitorDriver) -> None:
    """A worker killed by memory exhaustion must not be respawned into the same exhausted node.

    This is the reordering that costs the most: the restart policy is the one verdict that keeps the
    run *going*, so an OOM classified as a restartable death is retried once per tick until the
    scheduler kills the job, with a log full of restarts and no diagnosis.
    """
    restart = RestartRecorder()
    spec = WorkerSpec(
        "collector",
        FakeProcess(alive=False, exitcode=OOM_EXIT_CODE),
        critical=True,
        restart_fn=restart,
        max_restarts=3,
    )

    with pytest.raises(OutOfMemoryAbortError, match="collector"):
        watch_driver.run([spec])

    assert restart.calls == 0, "the restart policy ran on a worker that had died of an OOM"
    assert watch_driver.ticks == 0, "the sweep did not happen first — a whole pass completed"


def test_an_oom_death_beats_being_merely_non_critical(
    watch_driver: MonitorDriver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``critical=False`` says "the run survives without this worker", not "ignore how it died".

    An optional worker that ran out of memory is evidence about the *node*, not about the feature:
    the memory it could not get is memory the rest of the run is also not going to get. The
    tolerated-death line must not be written either — "continuing without it" is a false statement
    about a run that is, correctly, about to end.
    """
    spec = WorkerSpec("evaluator", FakeProcess(alive=False, exitcode=OOM_EXIT_CODE), critical=False)

    with caplog.at_level(logging.WARNING, logger="spawnkit"), pytest.raises(
        OutOfMemoryAbortError, match="evaluator",
    ):
        watch_driver.run([spec])

    assert watch_driver.ticks == 0
    assert [message for message in caplog.messages if TOLERATED_DEATH_MESSAGE in message] == []


def test_an_oom_death_beats_an_ordinary_critical_death(
    watch_driver: MonitorDriver,
    stop_event: threading.Event,
) -> None:
    """Both stop the run — but only one of them stops it *non-zero*.

    A critical death is reported and returns normally, which is a clean finish as far as the process
    exit status and the scheduler are concerned. Memory exhaustion has to escape as an exception, or
    the job that produced nothing is recorded as the job that succeeded.
    """
    spec = WorkerSpec("trainer", FakeProcess(alive=False, exitcode=OOM_EXIT_CODE), critical=True)

    with pytest.raises(OutOfMemoryAbortError, match="trainer"):
        watch_driver.run([spec])

    assert stop_event.is_set(), "the sweep must stop the rest of the run before it raises"


def test_a_sigkilled_process_is_swept_as_an_oom(watch_driver: MonitorDriver) -> None:
    """``-SIGKILL`` is the only way the kernel OOM-killer is ever visible from the parent.

    It gives the worker no chance to run a handler, so there is no exit code to read and no last log
    line; the signal is the whole diagnosis.
    """
    spec = WorkerSpec("worker", FakeProcess(alive=False, exitcode=-signal.SIGKILL))

    with pytest.raises(OutOfMemoryAbortError, match="SIGKILL"):
        watch_driver.run([spec])


def test_a_threads_recorded_oom_is_swept_the_same_way(watch_driver: MonitorDriver) -> None:
    """A thread has no exit status, so its OOM is read from the exception it recorded."""
    crashed = FakeThread(alive=False, exception=MemoryError("cannot allocate memory"))
    spec = WorkerSpec("collector", crashed)

    with pytest.raises(OutOfMemoryAbortError, match="collector"):
        watch_driver.run([spec])


def test_a_healthy_pool_is_never_swept_as_an_oom(watch_driver: MonitorDriver) -> None:
    """The complement, so the three tests above are not passing on a sweep that fires at anything."""
    watch_driver.run(
        [
            WorkerSpec("trainer", FakeThread()),
            WorkerSpec("worker", FakeProcess()),
        ],
        require_producers=False,
    )

    assert watch_driver.ticks == watch_driver.max_ticks


# ---------------------------------------------------------------------------
# Policy 2 — a critical death stops the run, and does not raise
# ---------------------------------------------------------------------------


def test_a_dead_critical_process_stops_the_run(
    watch_driver: MonitorDriver,
    stop_event: threading.Event,
) -> None:
    """No progress is possible without it, so the watch ends and everything else is told to stop."""
    watch_driver.run([WorkerSpec("worker", FakeProcess(alive=False))])

    assert stop_event.is_set()
    assert watch_driver.ticks == 0, "the watch completed a pass instead of returning on the death"


def test_a_dead_critical_thread_stops_the_run(stop_event: threading.Event, watch_driver: MonitorDriver) -> None:
    """A thread that simply stopped is as fatal as one that crashed: neither is doing its job."""
    watch_driver.run([WorkerSpec("trainer", FakeThread(alive=False))])

    assert stop_event.is_set()


def test_a_crashed_thread_is_caught_while_it_is_still_winding_down(
    stop_event: threading.Event,
    watch_driver: MonitorDriver,
) -> None:
    """The recorded exception is checked before liveness, and that ordering is deliberate.

    A worker that has caught something can stay alive for as long as its teardown takes, and for
    that whole window a liveness-only check reports a healthy run.
    """
    watch_driver.run([WorkerSpec("trainer", FakeThread(alive=True, exception=RuntimeError("boom")))])

    assert stop_event.is_set()


def test_a_critical_death_does_not_raise(watch_driver: MonitorDriver) -> None:
    """Reported, not raised: an ordinary worker death is an end the run is allowed to have.

    Stated as its own test because the OOM cases above are only meaningful if this is true — if
    every death raised there would be nothing for the sweep to have to run ahead of.
    """
    watch_driver.run([WorkerSpec("worker", FakeProcess(alive=False, exitcode=1))])


def test_a_live_pool_is_not_reported_as_dead(watch_driver: MonitorDriver, stop_event: threading.Event) -> None:
    """The common case: the monitor loops, decides nothing is wrong, and touches nothing."""
    watch_driver.run(
        [WorkerSpec("trainer", FakeThread()), WorkerSpec("worker", FakeProcess())],
        require_producers=False,
    )

    assert watch_driver.ticks == watch_driver.max_ticks
    assert stop_event.is_set()  # set by the tick budget, not by a verdict


# ---------------------------------------------------------------------------
# Policy 3 — a tolerated death is logged once, and the watch continues
# ---------------------------------------------------------------------------


def test_a_non_critical_death_is_logged_exactly_once(
    watch_driver: MonitorDriver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Once, not once per tick.

    The line that matters is the one naming the *next* thing to go wrong, and a message repeated
    every few seconds for the rest of a multi-hour run buries it — which is the failure mode this
    library was extracted from: 1795 copies of one error in three seconds, and nothing else.
    """
    specs = [
        WorkerSpec("evaluator", FakeProcess(alive=False), critical=False),
        WorkerSpec("worker", FakeProcess()),
    ]

    with caplog.at_level(logging.WARNING, logger="spawnkit"):
        watch_driver.run(specs, require_producers=False)

    reports = [message for message in caplog.messages if TOLERATED_DEATH_MESSAGE in message]
    assert watch_driver.ticks >= 2, "the monitor never took a second look, so once is not proven"
    assert reports == ["evaluator died - continuing without it"]


def test_a_non_critical_death_does_not_stop_the_watch(watch_driver: MonitorDriver) -> None:
    """Monitoring continues with that feature gone; losing an optional worker is not losing the run.

    An evaluator that failed to spawn costs a run its metrics. A monitor that stopped on it would
    cost the run everything else as well.
    """
    watch_driver.run(
        [WorkerSpec("evaluator", FakeProcess(alive=False), critical=False)],
        require_producers=False,
    )

    assert watch_driver.ticks == watch_driver.max_ticks


# ---------------------------------------------------------------------------
# Policy 4 — restarts happen within a budget, and the budget is spent for good
# ---------------------------------------------------------------------------


def test_a_dead_restartable_worker_is_respawned_within_its_budget(watch_driver: MonitorDriver) -> None:
    """Exactly ``max_restarts`` respawns, then the death is permanent.

    A per-pass allowance instead of a per-worker cap turns a permanent failure into an infinite
    loop: one spawn per tick for the rest of the job, each paying full startup cost, for a run that
    was never going to recover — and from the outside it looks exactly like a working run.
    """
    restart = RestartRecorder()
    spec = WorkerSpec(
        "worker",
        FakeProcess(alive=False),
        critical=True,
        restart_fn=restart,
        max_restarts=2,
    )

    watch_driver.run([spec], require_producers=False)

    assert restart.calls == 2
    assert spec.restarts_used == 2


def test_the_budget_being_spent_makes_the_death_terminal(
    watch_driver: MonitorDriver,
    stop_event: threading.Event,
) -> None:
    """After the last restart the worker is judged like any other critical worker: the run stops."""
    spec = WorkerSpec("worker", FakeProcess(alive=False), restart_fn=RestartRecorder(), max_restarts=1)

    watch_driver.run([spec], require_producers=False)

    assert stop_event.is_set()
    assert watch_driver.ticks == 1, "the run should have stopped on the pass after the last restart"


def test_a_restart_replaces_the_handle_rather_than_adding_one(watch_driver: MonitorDriver) -> None:
    """The spec is updated in place, so the caller's list stays the size it asked for.

    The pool the caller holds is also the pool it shuts down. A restart that appended would widen it
    on every crash: more workers than the configuration says, each holding its own handles, with
    nothing reporting the drift.
    """
    specs = [WorkerSpec("worker", FakeProcess(alive=False), restart_fn=RestartRecorder(), max_restarts=1)]
    original = specs[0].handle

    watch_driver.run(specs, require_producers=False)

    assert len(specs) == 1
    assert specs[0].handle is not original


def test_a_restart_fn_without_budget_is_never_called(watch_driver: MonitorDriver) -> None:
    """``max_restarts=0`` disables restarting even though a ``restart_fn`` was supplied.

    The default, and the safe one: a run that would rather stop and be diagnosed than respawn.
    """
    restart = RestartRecorder()

    watch_driver.run([WorkerSpec("worker", FakeProcess(alive=False), restart_fn=restart)])

    assert restart.calls == 0


def test_a_live_restartable_worker_is_left_alone(watch_driver: MonitorDriver) -> None:
    """Only a dead worker is respawned; the budget is not spent on a healthy one."""
    restart = RestartRecorder()

    watch_driver.run(
        [WorkerSpec("worker", FakeProcess(), restart_fn=restart, max_restarts=2)],
        require_producers=False,
    )

    assert restart.calls == 0


# ---------------------------------------------------------------------------
# require_producers — the same fact means opposite things to different runs
# ---------------------------------------------------------------------------


def test_producers_all_finished_ends_the_watch_cleanly(
    watch_driver: MonitorDriver,
    stop_event: threading.Event,
) -> None:
    """For a run whose producers work until told to stop, an empty pool means the run is over."""
    watch_driver.run([WorkerSpec("producer", FakeProcess(alive=False), critical=False, producer=True)])

    assert stop_event.is_set()
    assert watch_driver.ticks == 0, "the watch should end on the pass that finds the pool empty"


def test_an_empty_producer_pool_is_tolerated_when_it_is_not_required(watch_driver: MonitorDriver) -> None:
    """``require_producers=False`` is what makes a consumer-only phase possible.

    Replaying a saved buffer with no producers at all is a normal run; under the default the empty
    pool would read as "they all died" and end it before the first step.
    """
    watch_driver.run(
        [WorkerSpec("producer", FakeProcess(alive=False), critical=False, producer=True)],
        require_producers=False,
    )

    assert watch_driver.ticks == watch_driver.max_ticks


def test_require_producers_does_nothing_when_no_worker_is_a_producer(watch_driver: MonitorDriver) -> None:
    """A pool with nothing *declared* a producer is not a pool whose producers have all finished.

    The distinction is easy to lose — "no live producers" is true of both — and losing it ends every
    run that never declared one on its first pass, with the log line saying it finished.
    """
    watch_driver.run([WorkerSpec("worker", FakeProcess())], require_producers=True)

    assert watch_driver.ticks == watch_driver.max_ticks


def test_one_live_producer_keeps_the_watch_going(watch_driver: MonitorDriver) -> None:
    """A partly-dead pool is still producing; ending the run would throw away the rest of it."""
    watch_driver.run(
        [
            WorkerSpec("producer-0", FakeProcess(alive=False), critical=False, producer=True),
            WorkerSpec("producer-1", FakeProcess(), critical=False, producer=True),
        ],
    )

    assert watch_driver.ticks == watch_driver.max_ticks


# ---------------------------------------------------------------------------
# The post-loop re-check — the death that happens after the loop has ended
# ---------------------------------------------------------------------------


def test_a_threads_oom_is_found_even_when_the_loop_body_never_ran(stop_event: threading.Event) -> None:
    """The bug this module is shaped around, reproduced with a real ``MonitoredThread``.

    A thread that dies of an OOM sets the stop event *itself*, so by the time the supervisor tests
    its ``while`` condition the run is already flagged as stopping and the body never runs. Every
    per-pass policy is skipped, the supervisor returns normally, and the job that ran out of memory
    exits zero. The recorded exception is written before the event is set, which is exactly why it
    is still there to be found afterwards.
    """
    thread = MonitoredThread(target=_raise_out_of_memory, stop_event=stop_event, name="collector")
    thread.start()
    thread.join(timeout=5.0)

    assert stop_event.is_set(), "the thread was supposed to have flagged the stop itself"

    monitor = WorkerMonitor([WorkerSpec("collector", thread)], stop_event, check_interval=0.0)
    with pytest.raises(OutOfMemoryAbortError, match="collector"):
        monitor.watch()


def test_a_sigkilled_process_is_not_re_checked_after_the_loop(stop_event: threading.Event) -> None:
    """The mirror image, and the reason the post-loop sweep looks at threads only.

    Once the run is stopping, shutdown is entitled to ``SIGKILL`` a straggler that ignored its
    terminate — and from the parent that is indistinguishable from the OOM-killer. Re-checking
    processes here would report every forced shutdown as an out-of-memory failure.
    """
    stop_event.set()
    spec = WorkerSpec("worker", FakeProcess(alive=False, exitcode=-signal.SIGKILL))

    WorkerMonitor([spec], stop_event, check_interval=0.0).watch()  # must not raise


def test_the_same_sigkilled_process_does_raise_from_inside_the_loop(watch_driver: MonitorDriver) -> None:
    """The complement of the test above: the exemption is the post-loop sweep, not the signal.

    While the run is *supposed* to be running, nothing is entitled to kill a worker, so the same
    exit status means the OOM-killer and must end the run.
    """
    spec = WorkerSpec("worker", FakeProcess(alive=False, exitcode=-signal.SIGKILL))

    with pytest.raises(OutOfMemoryAbortError):
        watch_driver.run([spec])


def test_an_ordinary_stop_returns_without_raising(stop_event: threading.Event) -> None:
    """A stop requested from elsewhere — a signal handler, a finished job — is a clean end."""
    stop_event.set()

    WorkerMonitor([WorkerSpec("trainer", FakeThread())], stop_event, check_interval=0.0).watch()


# ---------------------------------------------------------------------------
# Status ticks
# ---------------------------------------------------------------------------


def test_the_status_hook_fires_on_the_status_interval_not_every_pass(stop_event: threading.Event) -> None:
    """The hook is for periodic side work — a progress snapshot, a heartbeat file.

    It must follow the status *interval* rather than the check interval, or a supervisor polling
    every few seconds calls it thousands of times an hour. The unpatched interval is used here, so
    the many passes this worker survives produce exactly one call.
    """
    ticks: list[int] = []
    worker = DyingProcess(polls_before_death=12)

    monitor = WorkerMonitor([WorkerSpec("worker", worker)], stop_event, check_interval=0.0)
    monitor.watch(require_producers=False, on_status_tick=lambda: ticks.append(1))

    assert worker.polls > 4, "the monitor did not loop enough times for this to mean anything"
    assert len(ticks) == 1


def test_the_status_hook_follows_the_interval_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
    stop_event: threading.Event,
) -> None:
    """The complement: shorten the interval and the same watch reports many times.

    Together with the test above this pins the hook to the interval rather than to a fixed count —
    one of them fails for any implementation that ignores the constant.
    """
    monkeypatch.setattr("spawnkit.monitor._STATUS_INTERVAL_EARLY_S", 0.0)
    ticks: list[int] = []
    worker = DyingProcess(polls_before_death=12)

    monitor = WorkerMonitor([WorkerSpec("worker", worker)], stop_event, check_interval=0.0)
    monitor.watch(require_producers=False, on_status_tick=lambda: ticks.append(1))

    assert len(ticks) > 1


def test_the_watch_runs_without_a_status_hook(stop_event: threading.Event) -> None:
    """The hook is optional, and ``None`` must not be called."""
    monitor = WorkerMonitor(
        [WorkerSpec("worker", DyingProcess(polls_before_death=4))], stop_event, check_interval=0.0,
    )

    monitor.watch(require_producers=False, on_status_tick=None)


# ---------------------------------------------------------------------------
# A handle that was never started is never a fault
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("critical", [True, False])
def test_a_worker_that_was_never_started_is_never_a_fault(
    critical: bool,
    watch_driver: MonitorDriver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``handle=None`` means "this run has no such worker", which is a configuration, not a death.

    Runs differ in which optional workers they have, and a supervisor that read an absent one as a
    death would stop every run that turned a feature off. Nothing may fire: no sweep, no critical
    verdict, no tolerated-death line, and no restart.
    """
    restart = RestartRecorder()
    spec = WorkerSpec("absent", None, critical=critical, restart_fn=restart, max_restarts=3)

    with caplog.at_level(logging.WARNING, logger="spawnkit"):
        watch_driver.run([spec], require_producers=False)

    assert watch_driver.ticks == watch_driver.max_ticks
    assert restart.calls == 0
    assert [message for message in caplog.messages if "absent" in message] == []


def test_an_absent_worker_beside_a_live_one_changes_nothing(watch_driver: MonitorDriver) -> None:
    """The realistic shape: a pool where one optional worker was never configured."""
    watch_driver.run(
        [
            WorkerSpec("trainer", FakeThread()),
            WorkerSpec("evaluator", None, critical=False),
            WorkerSpec("worker", FakeProcess()),
        ],
        require_producers=False,
    )

    assert watch_driver.ticks == watch_driver.max_ticks


def _raise_out_of_memory() -> None:
    """A thread body that dies the way an exhausted node kills one."""
    raise MemoryError("cannot allocate memory")
