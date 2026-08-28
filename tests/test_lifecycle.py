"""Unit tests for :mod:`spawnkit.lifecycle`.

The handler has one job and one escape hatch, and the tests pin both:

* the **first** signal runs the cleanup and leaves through ``sys.exit``, so ``finally`` blocks and
  ``atexit`` hooks still run — and it does that even when the cleanup itself raises, because a
  half-finished cleanup plus an exit beats a traceback and a parent that stays up holding workers;
* the **second** signal means the operator is watching a cleanup that is not finishing, and skips
  all of that with ``os._exit``.

``os._exit`` is monkeypatched throughout: called for real it would take the test session with it.
Every test restores the process's real signal handlers, since leaving one installed would break
pytest's own Ctrl+C handling for the rest of the session.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from typing import TYPE_CHECKING, NoReturn

import pytest

from spawnkit.lifecycle import register_shutdown_signals

if TYPE_CHECKING:
    from collections.abc import Iterator

SHUTDOWN_SIGNALS = [
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, name)
]
"""The signals this platform actually has; ``SIGHUP`` is absent on Windows."""


class _HardExit(BaseException):
    """Stands in for ``os._exit``, which does not return and cannot be allowed to run for real."""

    def __init__(self, code: int) -> None:
        super().__init__(f"os._exit({code})")
        self.code = code


@pytest.fixture(autouse=True)
def _restore_signal_handlers() -> Iterator[None]:
    """Put the interpreter's own handlers back, so a test cannot break Ctrl+C for the session."""
    previous = [(sig, signal.getsignal(sig)) for sig in SHUTDOWN_SIGNALS]
    yield
    for sig, handler in previous:
        signal.signal(sig, handler)


@pytest.fixture(autouse=True)
def _no_real_hard_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``os._exit`` with something observable; the real one would kill the test runner."""

    def fake_exit(code: int) -> NoReturn:
        raise _HardExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)


def test_it_registers_a_handler_for_every_shutdown_signal() -> None:
    """All three mean "wind down", and a run stopped by the one you forgot orphans its children."""
    register_shutdown_signals(lambda: None)

    handlers = {signal.getsignal(sig) for sig in SHUTDOWN_SIGNALS}
    assert len(handlers) == 1, "the three signals must share one handler"
    assert handlers != {signal.SIG_DFL}


def test_registering_does_not_run_the_cleanup() -> None:
    """It arms the handler; nothing has happened yet."""
    calls: list[str] = []
    register_shutdown_signals(lambda: calls.append("cleanup"))
    assert calls == []


@pytest.mark.parametrize("sig", SHUTDOWN_SIGNALS, ids=lambda sig: sig.name)
def test_the_first_signal_runs_the_cleanup_and_exits(sig: signal.Signals) -> None:
    """The whole contract, once per signal: cleanup, then a clean interpreter exit."""
    calls: list[str] = []
    register_shutdown_signals(lambda: calls.append("cleanup"), exit_code=3)

    with pytest.raises(SystemExit) as caught:
        signal.raise_signal(sig)

    assert calls == ["cleanup"]
    assert caught.value.code == 3


def test_the_default_exit_code_is_zero() -> None:
    """A signal is an expected end to most runs, so the default must not look like a failure."""
    register_shutdown_signals(lambda: None)

    with pytest.raises(SystemExit) as caught:
        signal.raise_signal(signal.SIGTERM)

    assert caught.value.code == 0


def test_a_non_zero_exit_code_is_passed_through() -> None:
    """A scheduler that should record signalled runs as failures gets to say so."""
    register_shutdown_signals(lambda: None, exit_code=17)

    with pytest.raises(SystemExit) as caught:
        signal.raise_signal(signal.SIGINT)

    assert caught.value.code == 17


def test_it_exits_through_sys_exit_rather_than_a_hard_exit() -> None:
    """``sys.exit`` on the clean path, so ``finally`` blocks and ``atexit`` hooks still run.

    ``_HardExit`` is what the monkeypatched ``os._exit`` raises, and seeing it here would mean the
    first signal had taken the emergency path.
    """
    register_shutdown_signals(lambda: None)

    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGTERM)


def test_a_cleanup_that_raises_is_swallowed_and_the_exit_still_happens() -> None:
    """Half-finished cleanup plus an exit beats a traceback and a parent that stays up.

    The handler's ``finally`` is what guarantees it: a cleanup that dies on its first wedged queue
    must not leave the remaining workers running.
    """

    def explode() -> None:
        raise RuntimeError("the worker pool would not stop")

    register_shutdown_signals(explode, exit_code=1)

    with pytest.raises(SystemExit) as caught:
        signal.raise_signal(signal.SIGTERM)

    assert caught.value.code == 1


def test_a_cleanup_that_raises_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The cleanup error is swallowed, not hidden — it is still the operator's only clue."""
    caplog.set_level(logging.INFO, logger="spawnkit")

    def explode() -> None:
        raise RuntimeError("the worker pool would not stop")

    register_shutdown_signals(explode)

    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGTERM)

    assert any("the worker pool would not stop" in record.getMessage() for record in caplog.records)


def test_the_first_signal_is_logged_with_the_force_quit_hint(caplog: pytest.LogCaptureFixture) -> None:
    """The operator has to be told that a second signal is available, at the moment they need it."""
    caplog.set_level(logging.INFO, logger="spawnkit")
    register_shutdown_signals(lambda: None)

    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGINT)

    assert any("again" in record.getMessage() for record in caplog.records)


def test_the_second_signal_hard_exits_instead_of_cleaning_up_again() -> None:
    """Cleanup that hangs has to stay interruptible.

    Otherwise the operator's only remaining option is ``SIGKILL`` on the parent, which orphans
    exactly the children the cleanup existed to stop.
    """
    calls: list[str] = []
    register_shutdown_signals(lambda: calls.append("cleanup"))

    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGINT)

    with pytest.raises(_HardExit) as caught:
        signal.raise_signal(signal.SIGINT)

    assert caught.value.code == 1
    assert calls == ["cleanup"], "the cleanup must not be started a second time"


def test_the_second_signal_may_be_a_different_one() -> None:
    """Ctrl+C followed by the scheduler's SIGTERM is the ordinary way this happens."""
    register_shutdown_signals(lambda: None)

    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGINT)
    with pytest.raises(_HardExit):
        signal.raise_signal(signal.SIGTERM)


def test_the_second_signal_is_logged_as_a_forced_exit(caplog: pytest.LogCaptureFixture) -> None:
    """The hard exit runs no ``atexit`` hook, so this line is the last thing the run says."""
    caplog.set_level(logging.INFO, logger="spawnkit")
    register_shutdown_signals(lambda: None)

    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGTERM)
    with pytest.raises(_HardExit):
        signal.raise_signal(signal.SIGTERM)

    assert any("forcing exit" in record.getMessage() for record in caplog.records)


def test_a_second_registration_replaces_the_first() -> None:
    """The handler is per-registration state, so re-arming gives a fresh cleanup its one call."""
    calls: list[str] = []
    register_shutdown_signals(lambda: calls.append("first"))
    register_shutdown_signals(lambda: calls.append("second"))

    with pytest.raises(SystemExit):
        signal.raise_signal(signal.SIGTERM)

    assert calls == ["second"]


def test_the_handler_takes_the_two_arguments_the_signal_module_passes() -> None:
    """Signals are delivered as ``(signum, frame)``; a handler with any other shape never runs."""
    handler = signal.getsignal(signal.SIGTERM)
    register_shutdown_signals(lambda: None)
    installed = signal.getsignal(signal.SIGTERM)
    assert installed is not handler
    assert callable(installed)

    with pytest.raises(SystemExit):
        installed(int(signal.SIGTERM), sys._getframe())
