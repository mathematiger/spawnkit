"""Out of memory is fatal: end the run, rather than logging the same failure until the wall clock.

Every long-lived worker loop ends up wrapping its body in a broad ``except Exception``, so that one
bad item, one torn checkpoint or one transient queue error cannot take down a multi-hour job. That
is right for the failures those handlers were written for, and exactly wrong for memory exhaustion,
which no retry can clear.

The shape it produces is worth stating once, because it is the reason this module exists. A data
collector's queue read raised::

    unable to mmap 716 bytes from file <filename not specified>: Cannot allocate memory (12)

the handler logged it and ``continue``d, and the loop re-raised and re-logged the same error 1795
times in three seconds. Nothing else stopped — the trainer kept stepping on a buffer that could
never be refilled — and the job had to be cancelled by hand nine hours later, having produced
nothing and reported success.

So memory exhaustion gets its own classification and one policy everywhere: **stop the run**. The
two halves of that policy differ only in where the code is running.

* In the **main process** :func:`raise_if_oom` converts the error into :class:`OutOfMemoryAbortError`,
  which escapes the broad handler, is recorded by :class:`~spawnkit.processes.MonitoredThread`, and
  ends the run non-zero so the scheduler records a failure rather than a clean finish.
* In a **worker process** :func:`abort_worker_on_oom` exits immediately with :data:`OOM_EXIT_CODE`.
  A child cannot raise into its parent, and the parent must not restart it into the same wall, so
  the exit code carries the diagnosis instead: :func:`process_oom_reason` reads it back.

:func:`process_oom_reason` also treats ``SIGKILL`` as an OOM death. That is the *only* way to see
the kernel OOM-killer, which gives a worker no chance to run a handler at all — and which is how
real out-of-memory failures on a shared node usually present.
"""

from __future__ import annotations

import errno
import os
import signal
from typing import TYPE_CHECKING

from spawnkit._log import flush_handlers, get_logger

if TYPE_CHECKING:
    import threading
    from multiprocessing.process import BaseProcess

log = get_logger(__name__)

OOM_EXIT_CODE = 17
"""Exit status a worker process uses to tell its parent it died of memory exhaustion.

Chosen outside the ranges anything else here produces: 0/1 from a normal or crashing interpreter,
and the negative ``-signal`` values ``multiprocessing`` reports for a killed child.
"""

_OOM_MESSAGE_MARKERS = (
    "out of memory",  # torch.cuda.OutOfMemoryError, "CUDA out of memory", cudaErrorMemoryAllocation
    "cannot allocate memory",  # ENOMEM from mmap/fork/shm
    "not enough memory",  # torch's DefaultCPUAllocator
    "unable to mmap",  # torch shared-memory file mapping, i.e. /dev/shm exhausted
)
"""Substrings (matched case-insensitively) that identify an out-of-memory error by its message.

Message matching rather than exception types on purpose: the same condition arrives as
``torch.cuda.OutOfMemoryError``, as a bare ``RuntimeError`` from torch's shared-memory allocator,
and as a ``RuntimeError`` re-raised by ``multiprocessing``'s pickler across a queue — and only the
first of those is a type this module could name without importing torch.
"""


class OutOfMemoryAbortError(RuntimeError):
    """Raised to end a run whose worker hit an unrecoverable out-of-memory condition.

    Distinct from the original error so a handler can tell "this run is over" from "this operation
    failed", and so the reason survives the trip through
    :class:`~spawnkit.processes.MonitoredThread` with the failing worker named.
    """


def is_oom_error(exc: BaseException | None) -> bool:
    """Whether ``exc`` (or anything it was raised from) reports memory exhaustion.

    The ``__cause__``/``__context__`` chain is followed because the error is routinely re-wrapped
    before anything gets to classify it — a supervisor's own exception is raised ``from`` the
    collector thread's, and a queue read re-raises what the feeder thread hit.

    :param exc: the exception to classify; ``None`` (nothing was recorded) is never an OOM.
    :return: True if this is memory exhaustion and no retry can clear it.
    """
    seen: set[int] = set()
    current = exc
    # The seen-set is what guarantees termination, so there is no depth cap. An earlier version had
    # one, and it was a blind spot rather than a safeguard: an OOM wrapped more times than the cap
    # classified as "not an OOM" and got retried forever, which is the exact failure this module
    # exists to break. Chains are short in practice and each link costs a set lookup.
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_oom_directly(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def raise_if_oom(exc: BaseException, context: str) -> None:
    """Turn an out-of-memory error into a run-ending :class:`OutOfMemoryAbortError`; else do nothing.

    For code running in the **main process**, where an exception can still propagate to something
    that ends the run. Call it as the first statement of a broad ``except`` block, so the handler
    keeps swallowing everything it was written for and stops swallowing the one error it must not.

    :param exc: the caught exception.
    :param context: what was running, for the log line and the abort message.
    :raises OutOfMemoryAbortError: if ``exc`` is an out-of-memory error.

    Examples
    --------
    >>> from spawnkit import raise_if_oom
    >>> try:                                    # doctest: +SKIP
    ...     step()
    ... except Exception as exc:
    ...     raise_if_oom(exc, "training step")  # re-raises only on OOM
    ...     log.warning("step failed, continuing: %s", exc)
    """
    if not is_oom_error(exc):
        return
    log.error("FATAL: out of memory in %s: %s. Ending the run - OOM is not retryable.", context, exc)
    raise OutOfMemoryAbortError(f"out of memory in {context}: {exc}") from exc


def abort_worker_on_oom(exc: BaseException, context: str) -> None:
    """Exit this **worker process** with :data:`OOM_EXIT_CODE` if ``exc`` is an OOM; else do nothing.

    A worker process has no way to raise into its parent, so the exit status is the channel: the
    parent's supervisor reads it back through :func:`process_oom_reason`, ends the run, and declines
    to restart the worker into the same exhausted node.

    ``os._exit`` rather than ``sys.exit``: the point of failure is often inside a queue read or a
    device allocation, and interpreter shutdown from there can block on the very queue feeder or
    allocator that just failed. Log handlers are flushed first so the diagnosis survives the hard
    exit; anything registered with ``atexit`` does not, which is the deliberate cost of exiting now.

    :param exc: the caught exception.
    :param context: what was running, for the log line.
    """
    if not is_oom_error(exc):
        return
    log.error(
        "FATAL: out of memory in %s: %s. Exiting this worker with code %d so the run stops.",
        context,
        exc,
        OOM_EXIT_CODE,
    )
    flush_handlers()
    os._exit(OOM_EXIT_CODE)


def process_oom_reason(process: BaseProcess | None, label: str) -> str | None:
    """Why ``label`` died of memory exhaustion — or ``None`` if it is alive or died some other way.

    For the **parent** side of a worker process. Two exit statuses count:

    * :data:`OOM_EXIT_CODE`, from the worker's own :func:`abort_worker_on_oom`.
    * ``-SIGKILL``, which on a live run means the kernel OOM-killer. It is indistinguishable from a
      deliberate ``SIGKILL``, so only call this while the run is *supposed* to be running — during
      shutdown :func:`~spawnkit.processes.graceful_shutdown` kills stragglers itself and every one
      of them would look like an OOM.

    :param process: the child to inspect; ``None`` (never started) is never a fault.
    :param label: how the process is named in the returned reason.
    :return: a one-line reason, or ``None``.
    """
    if process is None:
        return None
    try:
        exit_code = process.exitcode
    except (ValueError, AssertionError, OSError):  # a closed handle, or a foreign process
        return None

    if exit_code == OOM_EXIT_CODE:
        return f"{label} ran out of memory (exit code {OOM_EXIT_CODE})"
    if exit_code == -signal.SIGKILL:
        return f"{label} was killed by SIGKILL, which on a running node means the OS OOM-killer"
    return None


def thread_oom_reason(thread: threading.Thread | None, label: str) -> str | None:
    """Why ``label`` died of memory exhaustion — or ``None`` if it is alive or died some other way.

    The thread counterpart of :func:`process_oom_reason`, reading the exception a
    :class:`~spawnkit.processes.MonitoredThread` recorded. Unlike a process exit code, that record
    is written *before* the thread's stop event is set, so a supervisor can still find it after its
    own loop has already noticed the stop and broken out.

    :param thread: the thread to inspect; ``None`` (never started) is never a fault.
    :param label: how the thread is named in the returned reason.
    :return: a one-line reason, or ``None``.
    """
    if thread is None:
        return None
    error = getattr(thread, "exception", None)
    if not is_oom_error(error):
        return None
    return f"{label} ran out of memory: {error}"


def _is_oom_directly(exc: BaseException) -> bool:
    """Whether this one exception (ignoring its cause chain) reports memory exhaustion."""
    if isinstance(exc, (MemoryError, OutOfMemoryAbortError)):
        return True
    if isinstance(exc, OSError) and exc.errno == errno.ENOMEM:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MESSAGE_MARKERS)
