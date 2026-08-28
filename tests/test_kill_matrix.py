"""One real worker killed five different ways, and the verdict the supervisor reaches for each.

Everything else in this suite uses doubles, and for the per-tick policies that is the right trade.
This file cannot: what it tests is the *reading of an exit status*, and an exit status is produced
by the operating system, not by a test. A double that returns ``-9`` from ``exitcode`` proves the
branch is wired up; only a real ``SIGKILL`` proves ``-9`` is what a killed child actually leaves
behind.

The five modes, and why each verdict is the one that keeps a run honest:

===============  =================================  ==============================================
mode             how the worker ends                verdict
===============  =================================  ==============================================
oom exit         its own ``OOM_EXIT_CODE``          raises — a retry meets the same exhausted node
sigkill          the kernel's OOM-killer            raises — indistinguishable from the above
crash            an exception escapes it            critical death: stop, report, exit cleanly
clean exit       returns while still needed         critical death: leaving early is still leaving
silent stall     it is alive and doing nothing      no verdict at all
===============  =================================  ==============================================

The last row is the one that constrains the rest. A supervisor that invents a death when it cannot
see progress ends runs that were merely slow — a long checkpoint write, a queue that filled — and
that is worse than the failure it was guarding against, because it happens to healthy runs. This
module answers "is it still there", nothing more, and the stall test is what holds it to that.

Every pool is spawned inside a context manager that terminates and joins in ``finally``, so a
failing assertion cannot leave workers behind on the machine.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import signal
import time
from typing import TYPE_CHECKING, cast

import pytest

from spawnkit import OOM_EXIT_CODE, OutOfMemoryAbortError, WorkerSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from multiprocessing.context import SpawnContext
    from multiprocessing.synchronize import Event as MPEvent

    from conftest import MonitorDriver

pytestmark = pytest.mark.timeout(60)

BYSTANDERS = 2
"""Workers spawned alongside the victim, so a verdict has to name the one that actually died."""

WORKER_LIFETIME_S = 30.0
"""How long an idle worker waits to be released before exiting on its own.

A backstop, not a timeout the tests wait on: if the parent dies without running its cleanup, the
children still go away instead of outliving the test session.
"""

EXIT_WAIT_S = 15.0
"""How long the parent waits for a worker it has already killed to be reaped."""


def idle_worker(release: MPEvent) -> None:
    """Sit still until released — a worker that is alive and has nothing to report."""
    release.wait(timeout=WORKER_LIFETIME_S)


def oom_exit_worker(release: MPEvent) -> None:
    """Exit the way a worker that diagnosed its own memory exhaustion does.

    ``os._exit`` rather than ``sys.exit``: the real call site is inside an allocation or a queue
    read, where interpreter shutdown can block on the very thing that just failed. The exit status
    is the only channel a child has for telling its parent what happened.
    """
    del release
    os._exit(OOM_EXIT_CODE)


def crash_worker(release: MPEvent) -> None:
    """Die of an ordinary bug: an exception escapes, and the interpreter exits non-zero."""
    del release
    message = "worker hit an unrecoverable error"
    raise RuntimeError(message)


def clean_exit_worker(release: MPEvent) -> None:
    """Return normally, and so exit zero, while the run still needs this worker."""
    del release


def spawn_worker(
    ctx: SpawnContext,
    target: Callable[[MPEvent], None],
    release: MPEvent,
    name: str,
) -> multiprocessing.Process:
    """Build one spawn-mode worker handle, ready to be started.

    The cast is not papering over anything at runtime: a context's process class and
    ``multiprocessing.Process`` are the same class, but the type stubs model them as siblings of a
    common base rather than as a subclass, so a handle from ``get_context("spawn")`` does not type
    as the ``multiprocessing.Process`` the supervisor's annotations name.

    :param ctx: the spawn context the whole pool shares.
    :param target: the worker body.
    :param release: the event every worker in the pool waits on.
    :param name: how this worker is named in its verdict.
    :return: the process handle.
    """
    worker = ctx.Process(target=target, args=(release,), name=name, daemon=True)
    return cast("multiprocessing.Process", worker)


@contextlib.contextmanager
def worker_pool(victim_target: Callable[[MPEvent], None]) -> Iterator[list[multiprocessing.Process]]:
    """Spawn one victim plus :data:`BYSTANDERS` idle workers, and clean all of them up.

    :param victim_target: the body the first worker runs; the rest idle.
    :yield: the processes, victim first.
    """
    ctx = multiprocessing.get_context("spawn")
    release = ctx.Event()
    workers = [spawn_worker(ctx, victim_target, release, "victim")]
    workers += [
        spawn_worker(ctx, idle_worker, release, f"bystander-{index}") for index in range(BYSTANDERS)
    ]

    try:
        for worker in workers:
            worker.start()
        yield workers
    finally:
        release.set()
        for worker in workers:
            with contextlib.suppress(ValueError, AssertionError, OSError):
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5.0)


def wait_for_exit(worker: multiprocessing.Process) -> int:
    """Wait until ``worker`` has been reaped and return its exit status.

    The supervisor is only asked for a verdict once the death has actually happened, so that a test
    cannot pass or fail on how quickly the operating system got round to it.

    :param worker: the process to wait for.
    :return: the exit status the parent can read back.
    """
    deadline = time.monotonic() + EXIT_WAIT_S
    while time.monotonic() < deadline:
        if worker.exitcode is not None:
            return worker.exitcode
        time.sleep(0.02)
    pytest.fail(f"{worker.name} was still running {EXIT_WAIT_S}s after it should have died")


def specs_for(workers: list[multiprocessing.Process]) -> list[WorkerSpec]:
    """Describe a pool as critical, non-restartable workers — the plainest policy there is."""
    return [WorkerSpec(worker.name, worker, critical=True) for worker in workers]


def test_a_worker_that_exits_with_the_oom_code_ends_the_run(watch_driver: MonitorDriver) -> None:
    """The worker's own diagnosis, read back from the only channel a dead child has left.

    It has to escape as an exception rather than be reported: a run that stops cleanly on an OOM is
    recorded by the scheduler as a run that finished, and the next thing anyone does with it is
    wonder why it produced no output.
    """
    with worker_pool(oom_exit_worker) as workers:
        assert wait_for_exit(workers[0]) == OOM_EXIT_CODE

        with pytest.raises(OutOfMemoryAbortError, match="victim") as raised:
            watch_driver.run(specs_for(workers), require_producers=False)

    assert "bystander" not in str(raised.value), "the verdict named a worker that was still healthy"


def test_a_sigkilled_worker_is_reported_as_an_oom(watch_driver: MonitorDriver) -> None:
    """A worker killed outright never runs a handler, so the signal is the entire diagnosis.

    On a shared node that is how out-of-memory failures usually arrive: the kernel picks the largest
    process and kills it, and nothing in the run gets a chance to say so. The parent cannot tell that
    apart from a deliberate ``SIGKILL`` — and while the run is supposed to be running, treating it as
    the OOM-killer is the reading that does not lose data.
    """
    with worker_pool(idle_worker) as workers:
        victim_pid = workers[0].pid
        assert victim_pid is not None, "the victim never started, so nothing was killed"

        os.kill(victim_pid, signal.SIGKILL)
        assert wait_for_exit(workers[0]) == -signal.SIGKILL

        with pytest.raises(OutOfMemoryAbortError, match="victim"):
            watch_driver.run(specs_for(workers), require_producers=False)


def test_a_crashed_worker_stops_the_run_without_raising(watch_driver: MonitorDriver) -> None:
    """An ordinary bug: the run stops and says which worker it was, and that is a complete ending.

    Nothing is raised, because there is nothing here that a caller has to be prevented from
    mistaking for success — the stop was requested, the reason was logged, and the exception would
    only obscure a diagnosis that has already been made.
    """
    with worker_pool(crash_worker) as workers:
        assert wait_for_exit(workers[0]) == 1

        watch_driver.run(specs_for(workers), require_producers=False)

    assert watch_driver.stop_event.is_set()
    assert watch_driver.ticks == 0, "the watch completed a pass instead of returning on the death"


def test_a_worker_that_exits_zero_is_still_a_critical_death(watch_driver: MonitorDriver) -> None:
    """A critical worker that finished early is as gone as one that crashed.

    This is the case a liveness check based on "did it fail" misses entirely: the exit status is
    clean, nothing was logged, and the run carries on waiting for output from a process that has
    already decided it was done.
    """
    with worker_pool(clean_exit_worker) as workers:
        assert wait_for_exit(workers[0]) == 0

        watch_driver.run(specs_for(workers), require_producers=False)

    assert watch_driver.stop_event.is_set()


def test_a_worker_doing_nothing_is_not_a_death(
    watch_driver: MonitorDriver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The supervisor watches liveness, and refuses to guess at anything else.

    A worker can be alive and unproductive for entirely ordinary reasons — a long checkpoint write,
    a queue that filled, a slow first batch. Inventing a death from the absence of progress would
    end healthy runs, which is a worse failure than the one being guarded against because it happens
    to the runs that were working.
    """
    with worker_pool(idle_worker) as workers, caplog.at_level(logging.WARNING, logger="spawnkit"):
        watch_driver.run(specs_for(workers), require_producers=False)

    assert watch_driver.ticks == watch_driver.max_ticks, "the watch ended early on a healthy pool"
    assert caplog.messages == []
