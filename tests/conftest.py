"""Shared fixtures: worker doubles, and a way to run the monitor for an exact number of ticks.

Two ideas carry most of the supervision-tier tests.

**Doubles instead of workers.** ``FakeThread`` and ``FakeProcess`` subclass the real classes rather
than duck-typing them, so the library's own type hints are satisfied and the doubles cannot drift
from the interface they stand in for. They are never started: the supervisor reads ``is_alive()``,
an ``exception`` attribute and an exit status, and spawning a real worker to produce those would add
seconds per test without adding coverage. Real processes are used in exactly one place — the kill
matrix, where the exit statuses themselves are what is under test.

**Ticks instead of seconds.** A supervisor is a loop around a sleep, and a test that waits for it to
go round is a test that is slow when it passes and hangs when it fails. ``MonitorDriver`` shrinks the
status interval to zero so the status hook fires once per iteration, counts those, and sets the stop
event when the budget is spent. "How many times did the monitor look" then becomes an exact number,
no test depends on the wall clock, and a monitor that decides to keep going ends the test rather
than the session.

Nothing here is autouse and nothing here changes global state: every fixture is opt-in, and the two
that patch anything do it through ``monkeypatch`` for the duration of one test.
"""

from __future__ import annotations

import multiprocessing
import threading
from typing import TYPE_CHECKING

import pytest

from spawnkit import WorkerMonitor, WorkerSpec
from spawnkit import monitor as monitor_module

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeThread(threading.Thread):
    """A thread double: it is whatever ``alive`` and ``exception`` say it is, and never runs.

    Stands in for a worker thread wrapped in :class:`~spawnkit.MonitoredThread`, whose death the
    supervisor reads from exactly those two places — the recorded exception first, because a thread
    that has caught something may still take a while to actually exit.
    """

    def __init__(
        self,
        *,
        alive: bool = True,
        exception: BaseException | None = None,
        name: str = "fake-thread",
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._alive = alive
        self.exception = exception

    def is_alive(self) -> bool:
        """Whether this double claims to be running."""
        return self._alive


class FakeProcess(multiprocessing.Process):
    """A process double: it is whatever ``alive`` and ``exitcode`` say it is, and never spawns.

    The exit status is the interesting half. It is the only channel a dead child has left, and the
    supervisor reads three different meanings out of it: :data:`~spawnkit.OOM_EXIT_CODE` for a
    worker that diagnosed its own memory exhaustion, ``-SIGKILL`` for the kernel OOM-killer, and
    anything else for an ordinary death.
    """

    def __init__(
        self,
        *,
        alive: bool = True,
        exitcode: int | None = None,
        name: str = "fake-process",
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._alive = alive
        self._exit_status = exitcode

    def is_alive(self) -> bool:
        """Whether this double claims to be running."""
        return self._alive

    @property
    def exitcode(self) -> int | None:
        """The exit status this double reports, standing in for a reaped child's."""
        return self._exit_status


class MonitorDriver:
    """Run :meth:`~spawnkit.WorkerMonitor.watch` for a bounded, counted number of loop iterations.

    The status hook is the tick. With the module's status intervals shrunk to zero it fires exactly
    once per iteration, at the end of the body, so ``ticks`` counts complete passes: a policy that
    is supposed to run once per pass can be asserted as an exact number rather than a lower bound.
    Reaching ``max_ticks`` sets the stop event, which is what keeps a monitor that decides to keep
    watching from hanging the test session.

    ``ticks`` stays readable after :meth:`run` raises, because half of what this suite asserts is
    that the monitor raised *before* it got round to some other policy.
    """

    def __init__(self, stop_event: threading.Event, max_ticks: int = 3) -> None:
        self.stop_event = stop_event
        self.max_ticks = max_ticks
        self.ticks = 0

    def run(
        self,
        specs: Sequence[WorkerSpec],
        *,
        require_producers: bool = True,
        check_interval: float = 0.0,
    ) -> None:
        """Watch ``specs`` until the monitor ends the watch or the tick budget is spent.

        :param specs: the workers to watch.
        :param require_producers: passed straight through to ``watch``.
        :param check_interval: zero by default — the loop is driven by ticks, not by sleeping.
        """
        monitor = WorkerMonitor(specs, self.stop_event, check_interval=check_interval)
        monitor.watch(require_producers=require_producers, on_status_tick=self._on_tick)

    def _on_tick(self) -> None:
        """Count one completed loop pass, and ask the monitor to stop once the budget is spent."""
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.stop_event.set()


@pytest.fixture
def stop_event() -> threading.Event:
    """The run's shared stop flag, as the supervisor's callers hold it."""
    return threading.Event()


@pytest.fixture
def watch_driver(monkeypatch: pytest.MonkeyPatch, stop_event: threading.Event) -> MonitorDriver:
    """A :class:`MonitorDriver` whose status hook fires once per loop pass.

    The two status-interval constants are patched to zero for the duration of one test, which is
    what turns "log a line every 60 seconds" into "call the hook every pass" without a test ever
    waiting 60 seconds to find out.
    """
    monkeypatch.setattr(monitor_module, "_STATUS_INTERVAL_EARLY_S", 0.0)
    monkeypatch.setattr(monitor_module, "_STATUS_INTERVAL_LATE_S", 0.0)
    return MonitorDriver(stop_event)
