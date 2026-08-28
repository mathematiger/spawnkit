"""Is the supply side still alive? — the question a consumer loop has to keep asking.

Work reaches a consumer through a chain: producer processes make it, a collector thread drains their
queue into a buffer, and the consumer takes what lands there. Every link can die *quietly*. A
collector thread that raises prints a traceback and stops; a producer pool that exits leaves its
queue empty. Neither wakes the consumer, and neither is an error the consumer can see — from inside
the loop the two look exactly like "nothing has arrived yet".

The failure that follows is the expensive one: the consumer waits on a buffer nothing will ever fill
again, and the job holds its allocation for the rest of its wall clock without doing work or writing
a log line. The checks here are cheap enough to run on *every* consumer iteration, which is what it
takes to catch it.

They answer only "is it still running, and if not, why" — never "what should happen next". That
decision is not shared and must not be. A pool whose producers run until a stop event means an empty
pool is a finished run; a pool whose producers stop at a work budget means an empty pool *before*
that budget is spent is a failure. Both call the same primitives and reach opposite conclusions,
correctly. Encoding either one here would make the other unreachable.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING

from spawnkit.processes import safe_is_alive

if TYPE_CHECKING:
    import multiprocessing


class SupplyStalledError(RuntimeError):
    """Raised when no further work can reach the consumer.

    Ends the run with a diagnosis instead of letting it spin out its wall clock. Callers pass the
    reason string from :func:`thread_crash_reason` / :func:`thread_stopped_reason` so the message
    names the link that broke.
    """


def thread_crash_reason(thread: threading.Thread | None, label: str) -> str | None:
    """Return the exception a :class:`~spawnkit.processes.MonitoredThread` recorded.

    Deliberately blind to a *clean* exit, so it stays meaningful during shutdown — when every worker
    thread is expected to be gone and only a recorded exception still signals a fault. Use
    :func:`thread_stopped_reason` when a thread that is merely absent is also a problem.

    :param thread: the thread to inspect; ``None`` (a worker this run never started) is never a fault.
    :param label: how the thread is named in the returned reason.
    :return: a one-line reason, or ``None`` if nothing was recorded.
    """
    if thread is None:
        return None
    error = getattr(thread, "exception", None)
    if error is not None:
        return f"{label} crashed: {error!r}"
    return None


def thread_stopped_reason(thread: threading.Thread | None, label: str) -> str | None:
    """Why ``thread`` is no longer doing its job — crashed *or* simply not running — or ``None``.

    A thread that has not been started yet is reported the same way as one that has finished: both
    are "not running", and for a worker the caller believes it launched, both are equally fatal.
    Only call this once the run is past startup and while it is not already winding down, or a
    perfectly normal absence reads as a failure.

    :param thread: the thread to inspect; ``None`` is never a fault.
    :param label: how the thread is named in the returned reason.
    :return: a one-line reason, or ``None`` while the thread is alive.
    """
    crash = thread_crash_reason(thread, label)
    if crash is not None:
        return crash
    if thread is not None and not thread.is_alive():
        return f"{label} is not running"
    return None


def no_live_producers(processes: Iterable[multiprocessing.Process | None]) -> bool:
    """Whether none of ``processes`` is still running.

    An **empty** pool counts as none running. That is the answer both kinds of caller need, but for
    opposite reasons — so neither of them may treat this as "and that is a failure" on its own. Pair
    it with whatever makes an empty pool meaningful in your run: a "we expected producers at all"
    flag, or a work budget that has not been spent yet.

    :param processes: the producer pool, as live process handles.
    :return: ``True`` when not one of them is alive.

    Examples
    --------
    >>> from spawnkit import no_live_producers
    >>> no_live_producers([])
    True
    """
    return not any(safe_is_alive(process) for process in processes)
