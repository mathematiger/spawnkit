"""Worker lifecycle: a thread that records why it died, and stopping a pool of processes.

Every ``safe_*`` helper swallows the three exceptions a *torn-down* process or thread raises
(``ValueError`` from a closed handle, ``AssertionError`` from a foreign-process ``terminate``,
``OSError`` from a reaped pid). Shutdown runs after workers have already started dying, so a helper
that raised would abort the rest of the cleanup and leak the workers it had not reached yet. That is
not defensive coding for its own sake — it is the difference between "one worker was already gone"
and "eleven workers are still running after the parent exited".
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Sequence
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import multiprocessing

from spawnkit._log import get_logger

log = get_logger(__name__)

ProcessOrThread = BaseProcess | threading.Thread | None
"""Anything with a lifecycle these helpers can ask about.

:class:`~multiprocessing.process.BaseProcess` rather than :class:`multiprocessing.Process` on
purpose. ``get_context("spawn").Process`` is ``SpawnProcess``, which in typeshed is a **sibling** of
``Process`` and not a subclass — so annotating these with ``Process`` makes every type-checked caller
using a spawn context (this package's headline use case) fail at the API boundary. ``BaseProcess`` is
the common ancestor and admits every start method.
"""

NamedProcesses = Sequence[tuple[str, "BaseProcess | None"]]
"""``(label, process)`` pairs for :func:`shutdown_processes`; a ``None`` process is skipped."""


class MonitoredThread(threading.Thread):
    """A thread that records the exception that killed it, so a supervisor can surface it.

    A worker thread that raises is otherwise *invisible*: :mod:`threading` prints the traceback and
    the thread simply stops, while the code waiting on its output keeps waiting. The expensive shape
    is a dead data collector — the trainer spins on a buffer nothing fills any more, holding its
    GPUs for the rest of the allocation without a gradient step.

    Two behaviours make the death observable:

    * ``exception`` holds what was raised, for :func:`~spawnkit.supply.thread_crash_reason`.
    * ``stop_event``, when given, is **set**, so sibling workers wind down immediately rather than at
      the next poll of whatever supervisor happens to be watching.

    ``BaseException`` rather than ``Exception`` on purpose: the point is that no death is silent, and
    a ``SystemExit`` raised inside a worker thread would otherwise end it just as quietly. The
    exception is recorded, not re-raised — re-raising would only reach :mod:`threading`'s excepthook,
    which is where it was already going.

    Examples
    --------
    >>> import threading
    >>> from spawnkit import MonitoredThread
    >>> stop = threading.Event()
    >>> worker = MonitoredThread(target=lambda: 1 / 0, stop_event=stop, name="collector")
    >>> worker.start(); worker.join()
    >>> type(worker.exception).__name__
    'ZeroDivisionError'
    >>> stop.is_set()
    True
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Popped before super().__init__: Thread rejects keyword arguments it does not define.
        self.stop_event: Any = kwargs.pop("stop_event", None)
        super().__init__(*args, **kwargs)
        self.exception: BaseException | None = None

    def run(self) -> None:
        """Run the thread body, recording (and signalling) any exception that escapes it."""
        try:
            super().run()
        except BaseException as error:
            self.exception = error
            log.error("Thread %s failed: %s", self.name, error, exc_info=True)
            if self.stop_event is not None:
                self.stop_event.set()


def safe_is_alive(p: ProcessOrThread) -> bool:
    """Whether ``p`` is running, treating a torn-down or never-started handle as "no".

    :param p: a process, a thread, or ``None``.
    :return: True only if the handle is live and says so.
    """
    if p is None:
        return False
    try:
        return p.is_alive()
    except (ValueError, AssertionError, OSError):
        return False


def safe_join(p: ProcessOrThread, timeout: float = 3.0) -> None:
    """Join ``p`` for at most ``timeout`` seconds, ignoring a torn-down handle.

    :param p: a process, a thread, or ``None``.
    :param timeout: seconds to wait before returning regardless.
    """
    if p is None:
        return
    try:
        if hasattr(p, "is_alive") and p.is_alive():
            p.join(timeout=timeout)
    except (ValueError, AssertionError, OSError):
        pass


def safe_terminate(p: BaseProcess | None) -> None:
    """Send ``SIGTERM`` to ``p`` if it is running, ignoring a torn-down handle.

    :param p: the process to terminate, or ``None``.
    """
    if p is None:
        return
    try:
        if hasattr(p, "terminate") and safe_is_alive(p):
            p.terminate()
    except (ValueError, AssertionError, OSError):
        pass


def safe_kill(p: BaseProcess | None) -> None:
    """Send ``SIGKILL`` to ``p`` if it is running, ignoring a torn-down handle.

    :param p: the process to kill, or ``None``.
    """
    if p is None:
        return
    try:
        if hasattr(p, "kill") and safe_is_alive(p):
            p.kill()
    except (ValueError, AssertionError, OSError):
        pass


def safe_close(p: BaseProcess | None) -> None:
    """Release ``p``'s handle, ignoring one that is already closed.

    :param p: the process whose handle to close, or ``None``.
    """
    if p is None:
        return
    try:
        if hasattr(p, "close"):
            p.close()
    except (ValueError, AssertionError, OSError):
        pass


def detach_queue_feeder(q: multiprocessing.Queue[Any] | None) -> None:
    """Detach a queue's background feeder thread so a dead consumer cannot wedge interpreter exit.

    When this process has buffered items on ``q`` whose consumer has been terminated, the feeder
    thread blocks forever trying to flush them to the dead pipe, and the interpreter hangs joining
    it at shutdown — the "cleanup complete, then hang" shape, where the last log line is the one
    saying everything went fine. ``cancel_join_thread`` drops that join so the process can exit; the
    buffered items are intentionally discarded, which is correct because nothing is left to read
    them.

    :param q: the queue this process produces to, or ``None``.
    """
    if q is None:
        return
    with contextlib.suppress(ValueError, AssertionError, OSError):
        q.cancel_join_thread()


def graceful_shutdown(
    process: BaseProcess | None,
    name: str,
    terminate_timeout: float = 3.0,
    kill_timeout: float = 1.0,
) -> None:
    """Stop one process: terminate, wait, kill if it is still there, then release the handle.

    :param process: the process to stop, or ``None``.
    :param name: how the process is named in logs.
    :param terminate_timeout: seconds to wait after ``SIGTERM`` before escalating to ``SIGKILL``.
    :param kill_timeout: seconds to wait after ``SIGKILL`` before giving up on the pid.
    """
    if not safe_is_alive(process):
        safe_close(process)
        return

    log.info("  Terminating %s...", name)
    safe_terminate(process)
    safe_join(process, timeout=terminate_timeout)

    if safe_is_alive(process):
        log.warning("  %s ignored SIGTERM; killing it", name)
        safe_kill(process)
        safe_join(process, timeout=kill_timeout)

    safe_close(process)


def shutdown_processes(
    named_processes: NamedProcesses,
    graceful_timeout: float = 3.0,
    terminate_timeout: float = 1.0,
    kill_timeout: float = 0.5,
) -> None:
    """Stop a pool of workers in *three shared windows*, so teardown cost does not scale with N.

    The naive teardown gives each worker its own timeout, making shutdown cost ``N x timeout``. At
    realistic worker counts that is a minute-long cleanup — long enough that an impatient second
    Ctrl+C forces an unclean exit and drops the buffered log tail, which is the part that says why
    the run ended. A shutdown slow enough to be interrupted loses the diagnosis.

    Every phase here is therefore a *signal to everyone, then one wait*:

    1. join everyone within one shared ``graceful_timeout``;
    2. ``SIGTERM`` every straggler, then join them within one shared ``terminate_timeout``;
    3. ``SIGKILL`` whatever is left, then join within one shared ``kill_timeout``.

    Total cost is bounded by ``graceful + terminate + kill`` no matter how many workers there are.
    Sharing only the *first* window is not enough and is an easy mistake to make: measured with 16
    deliberately stuck workers and a 1 s window, a version that force-stopped stragglers one at a
    time took 17.9 s, against 4.0 s for the naive per-worker join it was supposed to beat — the
    graceful phase was shared and the force-stop phase was still serial
    (``benchmarks/results/shutdown.json``).

    The caller sets the stop event (and wakes any consumer blocked on a queue) *before* calling this
    — the grace window is time for workers to notice a signal that has already been sent, not a
    request to stop.

    :param named_processes: ``(label, process)`` pairs; the label appears in the force-stop log.
    :param graceful_timeout: seconds shared across all workers to exit on their own.
    :param terminate_timeout: seconds shared across all stragglers to die on ``SIGTERM``.
    :param kill_timeout: seconds shared across all survivors to die on ``SIGKILL``.
    """
    _join_all_within(named_processes, graceful_timeout)

    stragglers = [(name, p) for name, p in named_processes if safe_is_alive(p)]
    if stragglers:
        for name, process in stragglers:
            log.info("  Force-stopping %s...", name)
            safe_terminate(process)
        _join_all_within(stragglers, terminate_timeout)

        survivors = [(name, p) for name, p in stragglers if safe_is_alive(p)]
        for name, process in survivors:
            log.warning("  %s ignored SIGTERM; killing it", name)
            safe_kill(process)
        _join_all_within(survivors, kill_timeout)

    for _, process in named_processes:
        safe_close(process)


def _join_all_within(named_processes: NamedProcesses, window: float) -> None:
    """Join every process, sharing one ``window`` between them rather than one window each.

    The ``max()`` floor matters: once the window is spent the remaining joins still *poll* rather
    than block forever, so a pool larger than the window can absorb still finishes this phase.
    """
    deadline = time.monotonic() + window
    for _, process in named_processes:
        safe_join(process, timeout=max(0.05, deadline - time.monotonic()))
