"""Unit tests for :mod:`spawnkit.oom` — classifying and acting on memory exhaustion.

The classifier is the load-bearing part: everything else in the module is a one-line policy applied
to its verdict. So it is tested against the *verbatim* messages real runs produced
(``VERBATIM_OOM_MESSAGES``) rather than against invented ones, and against the near misses that must
stay non-fatal — an ordinary worker error must not end a run because someone wrote "memory" in it.
"""

from __future__ import annotations

import errno
import logging
import multiprocessing
import signal
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any, cast

import pytest

from spawnkit.oom import (
    OOM_EXIT_CODE,
    OutOfMemoryAbortError,
    abort_worker_on_oom,
    is_oom_error,
    process_oom_reason,
    raise_if_oom,
    thread_oom_reason,
)
from spawnkit.processes import MonitoredThread

if TYPE_CHECKING:
    from collections.abc import Callable

_CONTEXT = multiprocessing.get_context("spawn")
_JOIN_TIMEOUT_S = 30.0

VERBATIM_OOM_MESSAGES = [
    # A queue read in a data-collector thread, logged 1795 times in three seconds before anyone
    # noticed the run had stopped producing.
    "unable to mmap 716 bytes from file <filename not specified>: Cannot allocate memory (12)",
    # The same shape, from a worker process rather than a thread.
    "unable to mmap 656 bytes from file <filename not specified>: Cannot allocate memory (12)",
    # The CUDA shape.
    "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 39.39 GiB total capacity)",
    # The host-allocator shape.
    "[enforce fail at alloc_cpu.cpp:75] DefaultCPUAllocator: not enough memory: you tried to "
    "allocate 8589934592 bytes.",
]
"""Messages copied out of real run logs. Every one of them must classify as an OOM."""

NON_OOM_MESSAGES = [
    "simulation step diverged",
    "unable to open shared memory object </torch_123> in read-write mode",
    "CUDA error: device-side assert triggered",
    "the memory layout of this tensor is not contiguous",
]
"""Near misses. A run that dies on one of these has a bug to fix, not a node to blame."""


# ── is_oom_error ──────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("message", VERBATIM_OOM_MESSAGES)
def test_verbatim_messages_are_classified_as_oom(message: str) -> None:
    """Every out-of-memory message these runs actually produced is recognised."""
    assert is_oom_error(RuntimeError(message))


@pytest.mark.parametrize("message", NON_OOM_MESSAGES)
def test_ordinary_errors_are_not_classified_as_oom(message: str) -> None:
    """A failure that merely mentions memory stays a normal, swallowable worker error."""
    assert not is_oom_error(RuntimeError(message))


def test_the_markers_are_matched_case_insensitively() -> None:
    """The same condition arrives capitalised from one allocator and lower-case from another."""
    assert is_oom_error(RuntimeError("CANNOT ALLOCATE MEMORY"))
    assert is_oom_error(RuntimeError("Out Of Memory"))


def test_memory_error_and_enomem_oserror_are_oom() -> None:
    """The two cases that carry the diagnosis in their type rather than their text."""
    assert is_oom_error(MemoryError())
    assert is_oom_error(OSError(errno.ENOMEM, "Cannot allocate memory"))


def test_enospc_oserror_is_not_oom() -> None:
    """A full disk is a different failure; only ENOMEM is claimed here."""
    assert not is_oom_error(OSError(errno.ENOSPC, "No space left on device"))


def test_none_is_not_an_oom() -> None:
    """"Nothing was recorded" is the common case for a live worker, and is never a fault."""
    assert not is_oom_error(None)


def test_oom_is_found_through_the_cause_chain() -> None:
    """A supervisor wraps the collector's exception in its own; the verdict must survive the wrap."""
    inner = RuntimeError(VERBATIM_OOM_MESSAGES[0])
    wrapper = ValueError("collector thread crashed; no further data can reach the consumer")
    wrapper.__cause__ = inner
    assert is_oom_error(wrapper)


def test_oom_is_found_through_an_implicit_context() -> None:
    """A handler that raises *during* the OOM leaves the diagnosis only in ``__context__``."""
    inner = MemoryError()
    wrapper = ValueError("failed to report the failure")
    wrapper.__context__ = inner
    assert is_oom_error(wrapper)


def test_a_cause_cycle_terminates() -> None:
    """A self-referencing chain must return, not spin — the depth bound is what guarantees it."""
    first = ValueError("a")
    second = ValueError("b")
    first.__cause__ = second
    second.__cause__ = first
    assert not is_oom_error(first)


def test_a_chain_deeper_than_the_bound_gives_up_rather_than_walking_forever() -> None:
    """A deeply wrapped OOM is still an OOM: the walk is bounded by the seen-set, not by a depth cap.

    There used to be a ten-link cap here, and it was a blind spot rather than a safeguard - an OOM
    wrapped more times than the cap classified as "not an OOM" and got retried forever, which is the
    exact failure this module exists to break.
    """
    deepest: BaseException = MemoryError()
    for index in range(20):
        wrapper = ValueError(f"layer {index}")
        wrapper.__cause__ = deepest
        deepest = wrapper
    assert is_oom_error(deepest)


# ── raise_if_oom ──────────────────────────────────────────────────────────────────────────────────
def test_raise_if_oom_converts_an_oom_and_keeps_the_original_as_cause() -> None:
    """The abort names the failing worker; the original error stays reachable for the log."""
    original = RuntimeError(VERBATIM_OOM_MESSAGES[0])
    with pytest.raises(OutOfMemoryAbortError, match="the collector's queue read") as caught:
        raise_if_oom(original, "the collector's queue read")
    assert caught.value.__cause__ is original


def test_raise_if_oom_is_a_no_op_for_every_other_error() -> None:
    """The broad handlers it is dropped into must keep swallowing what they were written for."""
    raise_if_oom(RuntimeError("simulation step diverged"), "the collector's queue read")


def test_raise_if_oom_logs_the_diagnosis(caplog: pytest.LogCaptureFixture) -> None:
    """The exception ends the run; the log line is what says why, and it names the context."""
    caplog.set_level(logging.INFO, logger="spawnkit")
    with pytest.raises(OutOfMemoryAbortError, match="a worker step"):
        raise_if_oom(MemoryError(), "a worker step")
    assert any("a worker step" in record.getMessage() for record in caplog.records)


def test_raise_if_oom_logs_nothing_for_an_ordinary_error(caplog: pytest.LogCaptureFixture) -> None:
    """A swallowed error stays the caller's business; this must not add noise to every handler."""
    caplog.set_level(logging.INFO, logger="spawnkit")
    raise_if_oom(RuntimeError("simulation step diverged"), "a worker step")
    assert caplog.records == []


def test_out_of_memory_abort_is_itself_an_oom() -> None:
    """So a second handler further up the stack re-raises rather than swallowing the abort."""
    assert is_oom_error(OutOfMemoryAbortError("out of memory in a worker"))


def test_out_of_memory_abort_is_a_runtime_error() -> None:
    """Callers that already catch RuntimeError around a run keep working."""
    assert issubclass(OutOfMemoryAbortError, RuntimeError)


# ── abort_worker_on_oom ───────────────────────────────────────────────────────────────────────────
def test_abort_worker_on_oom_is_a_no_op_for_every_other_error() -> None:
    """Proven in-process: a non-OOM must not reach the ``os._exit`` at all."""
    abort_worker_on_oom(RuntimeError("simulation step diverged"), "a worker")


def test_abort_worker_on_oom_exits_the_process_with_the_oom_code() -> None:
    """The parent's only channel out of a worker process is its exit status."""
    source = (
        "from spawnkit import abort_worker_on_oom\n"
        f"abort_worker_on_oom(RuntimeError({VERBATIM_OOM_MESSAGES[0]!r}), 'test worker')\n"
        "raise AssertionError('abort_worker_on_oom returned instead of exiting')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    assert completed.returncode == OOM_EXIT_CODE, completed.stderr


def test_the_oom_exit_code_is_seventeen() -> None:
    """Pinned: the parent reads this number back, and it must sit outside every other exit status.

    0 and 1 come from a normal or crashing interpreter, and a killed child reports ``-signal``.
    """
    assert OOM_EXIT_CODE == 17


# ── process_oom_reason ────────────────────────────────────────────────────────────────────────────
def _exit_with_oom_code() -> None:
    """Child entry point: exit the way :func:`abort_worker_on_oom` would."""
    sys.exit(OOM_EXIT_CODE)


def _exit_cleanly() -> None:
    """Child entry point: finish normally."""
    return


def _idle_forever() -> None:
    """Child entry point: stay alive until the parent kills it."""
    import time

    while True:
        time.sleep(0.05)


def _spawn(target: Callable[[], None]) -> multiprocessing.Process:
    """Start ``target`` in a spawned child; the cast is the ``SpawnProcess`` stub split."""
    process = cast("multiprocessing.Process", _CONTEXT.Process(target=target))
    process.start()
    return process


def test_process_oom_reason_reads_back_the_oom_exit_code() -> None:
    """The parent side of the exit-status channel."""
    process = _spawn(_exit_with_oom_code)
    try:
        process.join(timeout=_JOIN_TIMEOUT_S)
        reason = process_oom_reason(process, "Worker-3")
        assert reason is not None
        assert "Worker-3" in reason
        assert str(OOM_EXIT_CODE) in reason
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=_JOIN_TIMEOUT_S)


def test_process_oom_reason_treats_sigkill_as_the_oom_killer() -> None:
    """The kernel OOM-killer gives a worker no chance to run a handler; -9 is the only trace."""
    process = _spawn(_idle_forever)
    try:
        process.kill()
        process.join(timeout=_JOIN_TIMEOUT_S)
        reason = process_oom_reason(process, "Worker-1")
        assert reason is not None
        assert "OOM-killer" in reason
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=_JOIN_TIMEOUT_S)


def test_process_oom_reason_ignores_a_cleanly_exited_worker() -> None:
    """An ordinary exit is not a fault, however carefully the parent is watching."""
    process = _spawn(_exit_cleanly)
    try:
        process.join(timeout=_JOIN_TIMEOUT_S)
        assert process_oom_reason(process, "Worker-0") is None
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=_JOIN_TIMEOUT_S)


def test_process_oom_reason_ignores_a_missing_worker() -> None:
    """A worker slot the run never filled has nothing to diagnose."""
    assert process_oom_reason(None, "Worker-0") is None


class _FakeExitProcess:
    """A process handle with a fixed exit code; enough for the codes that are awkward to produce."""

    def __init__(self, exitcode: int | None) -> None:
        self.exitcode = exitcode


class _ClosedProcess:
    """A process handle that was already released, so reading its exit code raises."""

    @property
    def exitcode(self) -> int:
        raise ValueError("process object is closed")


@pytest.mark.parametrize(
    "exitcode",
    [None, 0, 1, OOM_EXIT_CODE - 1, -signal.SIGTERM],
    ids=["still running", "clean exit", "crashed", "another exit code", "terminated"],
)
def test_only_the_two_oom_statuses_are_reported(exitcode: int | None) -> None:
    """Every other status is somebody else's problem, and must read as ``None``."""
    assert process_oom_reason(cast("Any", _FakeExitProcess(exitcode)), "Worker-2") is None


def test_a_closed_handle_is_not_reported_as_an_oom() -> None:
    """Shutdown closes handles, and the diagnosis pass must survive being run after it."""
    assert process_oom_reason(cast("Any", _ClosedProcess()), "Worker-2") is None


# ── thread_oom_reason ─────────────────────────────────────────────────────────────────────────────
def test_thread_oom_reason_reads_a_monitored_threads_recorded_exception() -> None:
    """The thread counterpart: MonitoredThread records the exception, this classifies it."""
    thread = MonitoredThread(target=_raise_oom, name="DataCollector")
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)
    reason = thread_oom_reason(thread, "DataCollector")
    assert reason is not None
    assert "DataCollector" in reason


def test_thread_oom_reason_ignores_a_thread_that_died_of_something_else() -> None:
    """An ordinary crash is the ``thread_crash_reason`` path, not this one."""
    thread = MonitoredThread(target=_raise_ordinary_error, name="DataCollector")
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)
    assert thread_oom_reason(thread, "DataCollector") is None


def test_thread_oom_reason_ignores_a_missing_thread() -> None:
    """A role this run does not have cannot have run out of memory."""
    assert thread_oom_reason(None, "DataCollector") is None


def test_thread_oom_reason_ignores_a_plain_thread() -> None:
    """A bare ``threading.Thread`` records nothing, so it can never be diagnosed."""
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join(timeout=_JOIN_TIMEOUT_S)
    assert thread_oom_reason(thread, "DataCollector") is None


def _raise_oom() -> None:
    """Thread body: fail the way an exhausted node makes a queue read fail."""
    raise RuntimeError(VERBATIM_OOM_MESSAGES[0])


def _raise_ordinary_error() -> None:
    """Thread body: fail in a way that is the caller's problem, not the node's."""
    raise ValueError("simulation step diverged")
