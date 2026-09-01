"""Optional accounting instrumentation for the service benchmark, applied strictly from outside.

``spawnkit`` depends on numpy and, for the service tier, torch. It must never depend on a profiler:
a library that imports one forces the choice on every application that installs it, and the whole
point of the batched service is that it is cheap to adopt. So nothing under ``src/spawnkit`` is
instrumented, and nothing under it ever will be.

The measurement therefore lives here, in the benchmark, as thin subclasses that override the private
steps of the client and the service. That is the honest cost of the constraint and it is worth
stating: three of the overrides below re-implement a handful of lines from
:mod:`spawnkit.service.batched` in order to put a phase boundary between them, so they have to be
kept in step with it. The alternative — a profiler hook inside the library — is the thing being
avoided.

Two instruments are used, and they answer different questions:

* :func:`~lineprofiler.accounting.phase` measures how long a *step* took, per process. That gives the
  service's own breakdown (wait / collate / forward / split / reply) and the client's (submit /
  await / decode).
* ``trace_begin`` / ``trace_mark`` / ``trace_end`` stamp one *request* as it crosses processes, and
  the offline decomposition turns those stamps into the segments of a round trip. Phases cannot do
  this: the transit between two processes belongs to neither of them, and is exactly the interval a
  per-process phase table cannot see.

Nothing here is imported unless ``--profile`` is passed.
"""

from __future__ import annotations

from typing import Any

import torch

from spawnkit.service import BatchedInferenceService, ServiceClient, SharedRows


def _require_rows(rows: SharedRows | None, rpc_name: str, side: str) -> SharedRows:
    """Return the shared-memory buffers, or say which side was built without them.

    The profiled subclasses below *override* the shared-transport methods, which means they also
    replace the base classes' own guards — so without this the None case reaches an attribute
    lookup instead of the clear error the library raises everywhere else.
    """
    if rows is None:
        msg = f"rpc {rpc_name!r} declares the shared-memory transport but this {side} has no shared_rows"
        raise RuntimeError(msg)
    return rows

ROUNDTRIP = "roundtrip"
"""Lifecycle channel carrying one request's checkpoints across the client and service processes."""

RESPONSE = "response"
"""Signal channel for the service -> client response hop, so the timeline can draw the arrow."""


def request_key(client_id: int, request_id: int) -> str:
    """Identify one round trip across processes.

    The client id alone is not unique — a client issues hundreds of calls — and the request id alone
    is not either, since every client numbers its own from 1. Only the pair identifies a round trip,
    and the lifecycle decomposition silently produces nonsense segments if two requests share a key.

    :param client_id: the issuing client's index.
    :param request_id: that client's own monotonic counter.
    :return: the lifecycle key.
    """
    return f"{client_id}:{request_id}"


def build_profiler(run_dir: str, run_id: str, role: str) -> Any:
    """Construct an enabled, tracing profiler for one process of the benchmark.

    Imported lazily so that a default benchmark run never touches the profiler package.

    :param run_dir: where worker snapshots and trace sidecars are written.
    :param run_id: shared across every process of one configuration, so the merge treats them as one
        attempt rather than as several competing ones.
    :param role: what this process does, as the report groups by it.
    :return: the constructed ``Profiler``.
    """
    try:
        from lineprofiler.accounting import Profiler
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on an unpublished package
        msg = (
            "--profile needs the `lineprofiler` accounting profiler, which is not on PyPI and is "
            "not a declared dependency. Every published number comes from a run without it; the "
            "flag exists to split a round trip into its segments during development. Drop --profile "
            "to run the benchmark."
        )
        raise RuntimeError(msg) from exc

    # Phases are opened with sync=False throughout: only the service holds a device, and the timed
    # regions are deliberately submission-side. A phase that needs a device sync asks for it itself.
    return Profiler(
        run_dir=run_dir,
        run_id=run_id,
        role=role,
        enabled=True,
        trace=True,
        install=True,
        # 1 Hz rather than the 30 s default. The trace's link ring holds capacity/20 = 10,000
        # checkpoints and drops the oldest silently once full; the service stamps three per request,
        # so a flush interval long enough to accumulate more than that would lose lifecycles and
        # the loss would look like a shorter run rather than an error. Verify with `dropped_links`.
        snapshot_interval_s=1.0,
    )


class ProfiledServiceClient(ServiceClient):
    """A :class:`~spawnkit.service.client.ServiceClient` that records where a round trip goes.

    Splits the client half of a call into submit / await / decode, and stamps the two lifecycle
    checkpoints only the client can stamp: the moment the request was handed to the queue, and the
    moment its answer came back.

    :param profiler: the process's profiler.
    :param args: forwarded to :class:`~spawnkit.service.client.ServiceClient`.
    :param kwargs: forwarded to :class:`~spawnkit.service.client.ServiceClient`.
    """

    def __init__(self, profiler: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._prof = profiler

    def call(self, rpc_name: str, payload: Any) -> Any:
        """Run one call inside a ``roundtrip`` phase, so its parts nest under one parent."""
        with self._prof.phase("roundtrip"):
            return super().call(rpc_name, payload)

    def _call_queued(self, rpc: Any, payload: Any) -> Any:
        """Submit through the queue, timing each half of the round trip separately."""
        prof = self._prof
        self._counter += 1
        key = request_key(self.client_id, self._counter)
        with prof.phase("submit"):
            prof.trace_begin(ROUNDTRIP, key)
            self._req_q.put((self.client_id, self._counter, rpc.name, payload))
        with prof.phase("await"):
            result = self._await(self._counter)
        prof.trace_end(ROUNDTRIP, key)
        prof.wait_on(RESPONSE, key)
        with prof.phase("decode"):
            return dict(result)

    def _call_shared(self, rpc: Any, payload: Any) -> Any:
        """Submit through shared memory, timing the row write and the row read separately."""
        prof = self._prof
        rows = _require_rows(self._rows, rpc.name, "client")
        with prof.phase("write_row"):
            rows.write_request(self.client_id, payload)
        self._counter += 1
        key = request_key(self.client_id, self._counter)
        with prof.phase("submit"):
            prof.trace_begin(ROUNDTRIP, key)
            self._req_q.put((self.client_id, self._counter, rpc.name, None))
        with prof.phase("await"):
            self._await(self._counter)
        prof.trace_end(ROUNDTRIP, key)
        prof.wait_on(RESPONSE, key)
        with prof.phase("read_row"):
            return {name: tensor.numpy() for name, tensor in rows.read_response(self.client_id).items()}


class ProfiledService(BatchedInferenceService):
    """A :class:`~spawnkit.service.batched.BatchedInferenceService` that reports its own breakdown.

    Owns its profiler rather than receiving one, because the object is pickled to a spawned process
    and a profiler holds threads and an open file. :meth:`run` is where the child actually starts.

    :param profile_run: ``(run_dir, run_id)`` for the profiler this process will build.
    :param args: forwarded to :class:`~spawnkit.service.batched.BatchedInferenceService`.
    :param kwargs: forwarded to :class:`~spawnkit.service.batched.BatchedInferenceService`.
    """

    def __init__(self, profile_run: tuple[str, str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._profile_run = profile_run
        self._prof: Any = None

    def run(self) -> None:
        """Build this process's profiler, then serve as usual."""
        run_dir, run_id = self._profile_run
        with build_profiler(run_dir, run_id, "service") as profiler:
            self._prof = profiler
            super().run()

    def _collect_batch(self) -> Any:
        """Time the wait for work, and stamp every request in the batch as admitted."""
        with self._prof.phase("wait_for_batch"):
            batch = super()._collect_batch()
        if isinstance(batch, list):
            for client_id, request_id, _name, _payload in batch:
                self._prof.trace_mark(ROUNDTRIP, request_key(client_id, request_id), "admitted")
        return batch

    def _handle_batch(self, model: Any, batch: list[Any], stats: Any) -> None:
        """Wrap one batch's service so its steps nest under a phase carrying the batch size."""
        with self._prof.phase("handle_batch"):
            self._prof.count("requests_per_forward", len(batch))
            super()._handle_batch(model, batch, stats)

    def _serve_queued(self, model: Any, rpc: Any, requests: list[Any]) -> None:
        """Collate, forward, split and reply — the same steps as the base class, timed apart."""
        prof = self._prof
        payloads = [payload for (_cid, _rid, _name, payload) in requests]
        with prof.phase("collate"):
            batched = rpc.collate(payloads)
        with prof.phase("forward"):
            outputs = self._forward(rpc, model, batched)
        self._mark_all(requests, "computed")
        with prof.phase("split"):
            responses = rpc.split(outputs, [rpc.rows_in(payload) for payload in payloads])
        with prof.phase("reply"):
            for (client_id, request_id, _name, _payload), response in zip(requests, responses, strict=True):
                self._resp_qs[client_id].put((request_id, response))
                prof.signal(RESPONSE, request_key(client_id, request_id))
        self._mark_all(requests, "replied")

    def _serve_shared(self, model: Any, rpc: Any, requests: list[Any]) -> None:
        """Gather, forward, scatter and signal — the same steps as the base class, timed apart."""
        prof = self._prof
        rows = _require_rows(self._rows, rpc.name, "service")
        client_ids = torch.tensor([cid for (cid, _rid, _name, _payload) in requests], dtype=torch.long)
        with prof.phase("gather"):
            tensors = rows.gather_request(client_ids, self._device)
        with prof.phase("forward"):
            outputs = self._forward(rpc, model, tensors)
        self._mark_all(requests, "computed")
        with prof.phase("scatter"):
            rows.scatter_response(client_ids, outputs)
        with prof.phase("reply"):
            for client_id, request_id, _name, _payload in requests:
                self._resp_qs[client_id].put((request_id, None))
                prof.signal(RESPONSE, request_key(client_id, request_id))
        self._mark_all(requests, "replied")

    def _mark_all(self, requests: list[Any], checkpoint: str) -> None:
        """Stamp one lifecycle checkpoint for every request in the batch."""
        for client_id, request_id, _name, _payload in requests:
            self._prof.trace_mark(ROUNDTRIP, request_key(client_id, request_id), checkpoint)
