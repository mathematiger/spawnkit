"""The library's logger, and the rule for using it.

``spawnkit`` logs through the standard library under a single ``spawnkit`` logger name, with a
:class:`~logging.NullHandler` attached. That is the library convention for a reason: an application
that has configured its own logging keeps that configuration, and one that has not sees nothing
rather than a stream of records it never asked for.

Nothing here configures the root logger, sets a level, or adds a stream handler. To see the
messages, the *application* opts in::

    import contextlib
import logging
    logging.basicConfig(level=logging.INFO)

Every module takes its own child logger (``logging.getLogger(__name__)``) so a caller can silence
one tier without silencing the rest — ``logging.getLogger("spawnkit.service").setLevel(WARNING)``
quiets the inference server's throughput lines and leaves the supervisor's death reports alone.
"""

from __future__ import annotations

import contextlib
import logging

logging.getLogger("spawnkit").addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name``.

    :param name: the calling module's ``__name__``.
    :return: a child of the ``spawnkit`` logger, so application-side filtering by prefix works.
    """
    return logging.getLogger(name)


def flush_handlers() -> None:
    """Push buffered log records out before a hard exit, ignoring an already-broken stream.

    :func:`~spawnkit.oom.abort_worker_on_oom` leaves the interpreter through ``os._exit``, which
    runs no ``atexit`` hook and flushes no handler. Without this the diagnosis that explains the
    exit code is exactly the line most likely to be lost.
    """
    for handler in logging.getLogger("spawnkit").handlers + logging.getLogger().handlers:
        with contextlib.suppress(ValueError, OSError):
            handler.flush()
