"""Signal handling: turn Ctrl+C and ``SIGTERM`` into one cleanup, and a second one into an exit.

A long run gets stopped in three ways — an impatient operator, a scheduler's time limit, a closed
terminal — and all three arrive as signals. Without a handler the default action kills the parent
and leaves its children running, orphaned and still holding GPUs, which on a shared node is the
failure other people notice before you do.

:func:`register_shutdown_signals` makes all three run the same cleanup. The second signal is treated
differently on purpose: cleanup that hangs (a queue feeder flushing to a dead consumer, a worker
ignoring ``SIGTERM``) has to remain interruptible, or the operator's only remaining option is
``SIGKILL`` on the parent — which orphans exactly the children the cleanup existed to stop.
"""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable
from types import FrameType

from spawnkit._log import get_logger

log = get_logger(__name__)

_SHUTDOWN_SIGNALS = ("SIGINT", "SIGTERM", "SIGHUP")
"""Signals that mean "wind down". ``SIGHUP`` is absent on Windows and skipped there."""


def register_shutdown_signals(
    cleanup_fn: Callable[[], None],
    exit_code: int = 0,
) -> None:
    """Run ``cleanup_fn`` once on ``SIGINT``/``SIGTERM``/``SIGHUP``; exit hard on a second signal.

    The first signal runs the cleanup and leaves through ``sys.exit``, so ``finally`` blocks and
    ``atexit`` hooks still run. A second signal — meaning the operator is watching a cleanup that
    is not finishing — skips all of that with ``os._exit``: at that point the priority is releasing
    the node, and anything still pending is what was hanging.

    A cleanup that raises is logged and does not stop the exit. Half-finished cleanup plus an exit
    beats a traceback and a process that stays up holding its workers.

    :param cleanup_fn: the shutdown routine; must be safe to call from a signal handler and, ideally,
        idempotent — the handler guarantees one call, but your own code paths may not.
    :param exit_code: status for the clean path. Leave at 0 when a signal is an expected end to the
        run; set non-zero if your scheduler should record signalled runs as failures.

    Examples
    --------
    >>> from spawnkit import register_shutdown_signals
    >>> register_shutdown_signals(lambda: print("stopping workers"))   # doctest: +SKIP
    """
    state = {"cleaning_up": False}

    def handle(signum: int, _frame: FrameType | None) -> None:
        if state["cleaning_up"]:
            log.warning("Second signal %d received during cleanup, forcing exit", signum)
            os._exit(1)

        state["cleaning_up"] = True
        log.info("Received signal %d, shutting down (send it again to force-quit)", signum)
        try:
            cleanup_fn()
        except Exception as exc:
            log.error("Error during cleanup: %s", exc, exc_info=True)
        finally:
            sys.exit(exit_code)

    for name in _SHUTDOWN_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, handle)
