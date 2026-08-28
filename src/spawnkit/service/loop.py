"""The shared skeleton every GPU worker process runs: build a model, then resync / collect / handle.

A worker that owns a device model has the same lifecycle whatever it serves: build one private copy
of the network, then loop { periodically resync weights from the trainer's shared copy; pull a unit
of work off a queue; process it } until told to stop. These free functions are that skeleton, so a
new worker writes only its own build / sync / collect / handle steps.

The two policies worth knowing about, because both were failures first:

* **A model that will not build stops the run**, and stops it *non-zero*. Setting a stop event alone
  ends the run at exit status 0, and "the model did not fit in VRAM" must not be recorded as success.
* **A failed weight sync opens a bounded fast-retry burst**, not an unbounded one. A sync racing the
  trainer's write succeeds on the next attempt; an allocation failure never does, and retrying it
  forever just serves stale weights while logging. The burst is capped and then normal cadence
  resumes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from spawnkit._log import get_logger
from spawnkit.oom import abort_worker_on_oom

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

log = get_logger(__name__)

T = TypeVar("T")

QUEUE_STOP = None
"""What an orchestrator puts on a request queue to stop the worker reading it."""

STOP = object()
"""``collect_fn`` saw :data:`QUEUE_STOP` — leave the loop."""

IDLE = object()
"""``collect_fn`` found nothing in its poll window — keep looping, there is no work."""

_MAX_FAST_SYNC_RETRIES = 5
"""How many immediate retries a failed weight sync gets before reverting to scheduled cadence."""


def build_model_or_stop(build_fn: Callable[[], T], stop_event: Event, name: str) -> T | None:
    """Run ``build_fn``; on any exception log it, set ``stop_event``, and return ``None``.

    A worker that cannot build its private model can make no progress, so a failure here stops the
    whole run rather than spinning a dead process. An out-of-memory failure additionally exits
    non-zero through :func:`~spawnkit.oom.abort_worker_on_oom` — running out of VRAM at startup is
    the commonest way this fails, and it must be reported as the failure it is rather than as a
    clean, early finish.

    :param build_fn: builds and returns the worker's device-local model.
    :param stop_event: the run's shared stop flag, set on failure.
    :param name: this worker's name, for the log line.
    :return: the built model, or ``None`` if it could not be built.
    """
    try:
        return build_fn()
    except Exception as exc:
        abort_worker_on_oom(exc, f"{name} model build")
        # log.error, not log.exception: abort_worker_on_oom above has already exited the process
        # for the case whose traceback would matter, and a build failure's message is self-explanatory.
        log.error("[%s] failed to build model: %s; stopping", name, exc)  # noqa: TRY400
        stop_event.set()
        return None


def maybe_sync_weights(
    sync_fn: Callable[[], None],
    iters: int,
    interval: int,
    name: str,
    *,
    force: bool = False,
) -> bool:
    """Refresh weights on schedule (every ``interval`` iterations) or immediately when ``force``d.

    :param sync_fn: copies the current weights into this worker's model.
    :param iters: how many work items this worker has handled.
    :param interval: sync every this many iterations.
    :param name: this worker's name, for the log line.
    :param force: sync now regardless of the schedule (used by the fast-retry burst).
    :return: ``True`` only if a sync was *attempted and failed*, so the caller can fast-retry.
        ``False`` when it was skipped (off-schedule, not forced) or succeeded.
    """
    if not force and iters % interval != 0:
        return False
    try:
        sync_fn()
    except Exception as exc:
        # The fast-retry burst is for a sync racing the trainer's write. Retrying an allocation
        # failure just spends the burst and then serves stale weights forever, so OOM exits here.
        abort_worker_on_oom(exc, f"{name} weight sync")
        log.warning("[%s] weight sync failed: %s", name, exc)
        return True
    return False


def run_worker_loop(
    stop_event: Event,
    sync_interval: int,
    sync_fn: Callable[[], None],
    collect_fn: Callable[[], Any],
    handle_fn: Callable[[Any], None],
    name: str,
) -> None:
    """Drive the shared resync / collect / handle loop until stopped.

    ``sync_fn`` and ``handle_fn`` bind their model by closure, which is what keeps these signatures
    model-agnostic. The sync runs *before* ``collect_fn``, so a retry interleaves with serving and
    never blocks it.

    :param stop_event: the run's shared stop flag; checked every iteration.
    :param sync_interval: iterations between scheduled weight syncs.
    :param sync_fn: copies current weights into this worker's model.
    :param collect_fn: returns :data:`STOP` to leave the loop, :data:`IDLE` when nothing arrived, or
        a work item to pass to ``handle_fn``.
    :param handle_fn: processes one work item. Must not raise — a worker loop that dies takes its
        clients' pending requests with it.
    :param name: this worker's name, for the log lines.
    """
    iters = 0
    fast_retries_left = 0
    while not stop_event.is_set():
        force = fast_retries_left > 0  # invariant: force is True <=> we are mid fast-retry burst
        sync_failed = maybe_sync_weights(sync_fn, iters, sync_interval, name, force=force)
        if sync_failed:
            fast_retries_left = fast_retries_left - 1 if force else _MAX_FAST_SYNC_RETRIES
        elif force:
            fast_retries_left = 0  # a forced retry succeeded -> end the burst early

        work = collect_fn()
        if work is STOP:
            break
        if work is IDLE:
            continue

        handle_fn(work)
        iters += 1
    log.info("[%s] exiting", name)
