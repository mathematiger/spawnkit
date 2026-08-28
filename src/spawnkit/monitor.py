"""Watch a pool of workers, and say which one died — before the run spends its wall clock finding out.

The failure this exists for has no error message. A worker dies, nothing raises in the parent, and
the run keeps going: the trainer waits on a buffer nothing refills, the queue nobody drains grows,
and the job is cancelled hours later having logged nothing but its last healthy status line. Every
policy below is one lesson from that shape.

Describe each worker once with a :class:`WorkerSpec` — is it critical, may it be restarted, how many
times — and hand the list to :class:`WorkerMonitor`. It applies four rules per tick, and the
**order** is load-bearing:

1. **Out-of-memory deaths are swept first.** An OOM must not be reported as a restartable worker, a
   disabled optional feature, or an ordinary critical failure — every one of those verdicts leads
   somewhere wrong. It is checked ahead of all of them and it ends the run.
2. **Critical deaths stop the run.** No progress is possible without them; continuing only delays
   the diagnosis.
3. **Non-critical deaths are noted once**, and monitoring continues with that feature gone. Logging
   it every tick would bury the line that matters.
4. **Restartable workers are respawned within their budget.** Unbounded restarts turn a permanent
   failure into an infinite loop that looks, from the outside, exactly like a working run.

An empty producer pool is *not* a verdict this module reaches on its own — see
``require_producers`` on :meth:`WorkerMonitor.watch` and the note in :mod:`spawnkit.supply` about
why the same fact means opposite things to different runs.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spawnkit._log import get_logger
from spawnkit.oom import OutOfMemoryAbortError, process_oom_reason, thread_oom_reason
from spawnkit.processes import safe_is_alive
from spawnkit.supply import no_live_producers, thread_stopped_reason

if TYPE_CHECKING:
    import multiprocessing
    from multiprocessing.process import BaseProcess

log = get_logger(__name__)

_SLEEP_TICK_S = 0.1
"""Granularity of the interruptible sleep, so a stop event is noticed within ~100 ms."""

_EARLY_STATUS_WINDOW_S = 300.0
"""How long status is logged at the frequent cadence; startup is when a run most often dies."""

_STATUS_INTERVAL_EARLY_S = 60.0
_STATUS_INTERVAL_LATE_S = 300.0


@dataclass
class WorkerSpec:
    """One worker, and the policy for what its death means.

    :param name: how the worker is named in every log line and failure reason. Make it unique — it
        is the whole diagnosis when the run ends.
    :param handle: the live :class:`multiprocessing.Process` or :class:`threading.Thread`. ``None``
        means "never started", which is never treated as a fault.
    :param critical: when True, this worker's death ends the run. When False the death is logged
        once and monitoring continues without it.
    :param producer: when True, this worker counts toward "is anything still producing" — the pool
        that :meth:`WorkerMonitor.watch`'s ``require_producers`` asks about.
    :param restart_fn: called with no arguments to respawn this worker, returning the new handle.
        ``None`` (the default) means the worker is not restartable.
    :param max_restarts: how many times ``restart_fn`` may be used before the death is treated as
        permanent. Zero disables restarting even when ``restart_fn`` is set.
    """

    name: str
    handle: BaseProcess | threading.Thread | None
    critical: bool = True
    producer: bool = False
    restart_fn: Callable[[], BaseProcess] | None = None
    max_restarts: int = 0

    restarts_used: int = field(default=0, init=False, repr=False)
    death_reported: bool = field(default=False, init=False, repr=False)

    @property
    def is_thread(self) -> bool:
        """Whether this worker is a thread rather than a process (they die differently)."""
        return isinstance(self.handle, threading.Thread)

    @property
    def can_restart(self) -> bool:
        """Whether a restart is configured and this worker's budget still has room."""
        return self.restart_fn is not None and self.restarts_used < self.max_restarts


class WorkerMonitor:
    """Poll a pool of :class:`WorkerSpec` workers until one dies or the run is asked to stop.

    :param specs: the workers to watch. Held by reference: a restarted worker's ``handle`` is
        replaced in place, so the caller's list stays current.
    :param stop_event: the run's shared stop flag. Set by this monitor on a fatal verdict, and
        watched so a stop requested elsewhere ends the loop promptly.
    :param check_interval: seconds between ticks. The sleep is interruptible at ``0.1 s``.

    Examples
    --------
    >>> import threading
    >>> from spawnkit import MonitoredThread, WorkerMonitor, WorkerSpec
    >>> stop = threading.Event()
    >>> worker = MonitoredThread(target=stop.wait, stop_event=stop, name="trainer")
    >>> worker.start()
    >>> monitor = WorkerMonitor(                      # doctest: +SKIP
    ...     [WorkerSpec("trainer", worker, critical=True)], stop, check_interval=1.0,
    ... )
    >>> monitor.watch()                               # doctest: +SKIP
    """

    def __init__(
        self,
        specs: Sequence[WorkerSpec],
        stop_event: threading.Event | multiprocessing.synchronize.Event,
        check_interval: float = 5.0,
    ) -> None:
        self._specs = list(specs)
        self._stop = stop_event
        self._check_interval = check_interval

    def watch(
        self,
        require_producers: bool = True,
        on_status_tick: Callable[[], None] | None = None,
    ) -> None:
        """Watch every worker until one fails, the producers are all done, or the run is stopped.

        Returns normally on every end the run is allowed to have: a critical death (already logged
        and with ``stop_event`` set), an exhausted producer pool, or an external stop. Only memory
        exhaustion raises — because only memory exhaustion must not be mistaken for a clean finish.

        :param require_producers: when True, a producer pool with nothing alive in it ends the watch
            as a *completed* run. Set False when zero live producers is normal — a consumer-only
            phase, or a run whose producers have not started yet. When no spec is marked
            ``producer`` this has no effect.
        :param on_status_tick: called on each status-log interval, for a caller that wants to
            checkpoint something whenever the monitor reports. Exceptions from it are not caught.
        :raises OutOfMemoryAbortError: if any worker, critical or not, died of memory exhaustion.
            That is the one death this method does not merely report: it cannot be retried, a
            restarted worker would meet the same exhausted node, and the run must end non-zero so
            the scheduler records a failure instead of a clean finish.
        """
        log.info("Watching %d workers", len(self._specs))
        started_at = time.monotonic()
        last_status = 0.0

        while not self._stop.is_set():
            self._raise_on_oom_death(self._specs)

            if self._any_critical_death():
                return

            self._report_tolerated_deaths()
            self._restart_dead_workers()

            if require_producers and self._producers_are_done():
                log.info("All producers finished - ending the watch")
                self._stop.set()
                break

            now = time.monotonic()
            interval = (
                _STATUS_INTERVAL_EARLY_S
                if now - started_at < _EARLY_STATUS_WINDOW_S
                else _STATUS_INTERVAL_LATE_S
            )
            if now - last_status >= interval:
                self._log_status()
                if on_status_tick is not None:
                    on_status_tick()
                last_status = now

            self._sleep_or_stop(self._check_interval)

        # The loop can exit without ever running its body: a thread that dies of an OOM sets
        # stop_event itself (MonitoredThread does), so the very next `while` test ends the loop and
        # the run would otherwise look like a clean, requested shutdown. The recorded exception is
        # written before that event is set, so it is still here to be found.
        #
        # Threads only. By this point the shutdown path is entitled to SIGKILL a straggler, which is
        # indistinguishable from the OOM-killer doing it (see oom.process_oom_reason).
        self._raise_on_oom_death([spec for spec in self._specs if spec.is_thread])
        log.info("Worker monitoring stopped")

    def _raise_on_oom_death(self, specs: Iterable[WorkerSpec]) -> None:
        """Stop every worker and raise if any of ``specs`` died of memory exhaustion."""
        reasons = (
            thread_oom_reason(spec.handle, spec.name)  # type: ignore[arg-type]
            if spec.is_thread
            else process_oom_reason(spec.handle, spec.name)  # type: ignore[arg-type]
            for spec in specs
        )
        first = next((reason for reason in reasons if reason is not None), None)
        if first is None:
            return

        log.error("CRITICAL: %s. Stopping the run - out of memory is not retryable.", first)
        self._stop.set()
        raise OutOfMemoryAbortError(first)

    def _any_critical_death(self) -> bool:
        """Whether a critical worker is gone; logs the reason and sets the stop event if so."""
        for spec in self._specs:
            if not spec.critical:
                continue
            reason = self._death_reason(spec)
            if reason is None:
                continue
            # A restartable critical worker is respawned rather than fatal, as long as it has budget.
            if spec.can_restart:
                continue
            log.error("CRITICAL: %s", reason)
            self._stop.set()
            return True
        return False

    def _report_tolerated_deaths(self) -> None:
        """Log each non-critical worker's death exactly once, then keep going without it."""
        for spec in self._specs:
            if spec.critical or spec.death_reported or spec.can_restart:
                continue
            if self._death_reason(spec) is not None:
                log.warning("%s died - continuing without it", spec.name)
                spec.death_reported = True

    def _restart_dead_workers(self) -> None:
        """Respawn every dead restartable worker that still has restart budget."""
        for spec in self._specs:
            if self._stop.is_set():
                return
            if spec.handle is None or safe_is_alive(spec.handle) or not spec.can_restart:
                continue
            assert spec.restart_fn is not None  # can_restart guarantees it
            log.warning(
                "%s died - restarting (%d of %d)", spec.name, spec.restarts_used + 1, spec.max_restarts,
            )
            spec.handle = spec.restart_fn()
            spec.restarts_used += 1

    def _producers_are_done(self) -> bool:
        """Whether producers were declared, at least one was started, and none is still alive.

        The ``is not None`` filter is load-bearing. A producer that was never started has a ``None``
        handle, and ``no_live_producers([None])`` is ``True`` — so without the filter a run whose
        producers had not spawned yet ended on its very first tick, set the stop event, logged "all
        producers finished" and exited zero. That is precisely the silent clean finish this module
        exists to prevent, produced by the module itself.
        """
        producers = [spec.handle for spec in self._specs if spec.producer and spec.handle is not None]
        return bool(producers) and no_live_producers(producers)  # type: ignore[arg-type]

    @staticmethod
    def _death_reason(spec: WorkerSpec) -> str | None:
        """Why ``spec`` is not doing its job, or ``None`` while it is fine."""
        if spec.is_thread:
            return thread_stopped_reason(spec.handle, spec.name)  # type: ignore[arg-type]
        if spec.handle is None:
            return None
        if not safe_is_alive(spec.handle):
            return f"{spec.name} process died unexpectedly"
        return None

    def _log_status(self) -> None:
        """Log one line naming every worker and whether it is alive."""
        marks = " | ".join(
            f"{spec.name}: {'up' if safe_is_alive(spec.handle) else 'down'}" for spec in self._specs
        )
        log.info("[status] %s", marks)

    def _sleep_or_stop(self, duration: float) -> None:
        """Sleep up to ``duration`` seconds, waking early if the stop event is set."""
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return
            time.sleep(min(_SLEEP_TICK_S, max(0.0, deadline - time.monotonic())))
