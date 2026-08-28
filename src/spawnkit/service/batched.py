"""One process owns the device model and serves every worker's calls, batched into one forward.

The problem it solves is VRAM, and then latency. Give each of N workers its own copy of the model
and you pay N x (weights + CUDA context) — on a large model or a large N, that is the constraint that
decides how many workers you can run at all. One service process owning the model makes VRAM 1x
regardless of N, and the workers become what they should have been: CPU processes that do the
environment work and ask a question now and then.

Batching is what makes that not a slowdown. Small forwards are dominated by launch overhead rather
than arithmetic — a measured example ran flat at 2.13 ms from batch 1 to batch 512 — so serving
sixteen workers' requests in one forward costs roughly what serving one costs, and
:class:`~spawnkit.service.cudagraph.GraphRunner` then takes that same forward to 0.39 ms.

Two knobs decide the latency/throughput trade, and they point in opposite directions:

* ``max_batch`` caps how many requests one forward may serve.
* ``batch_window_ms`` sets a *latency floor*: the service waits up to this long for more requests
  before running. Zero (the default) means an opportunistic drain with no floor — take whatever has
  already arrived and go. A positive window raises requests-per-forward at the cost of up to that
  much extra round-trip latency on every call, which is the right trade only when the log below
  shows requests-per-forward sitting near 1 while workers are waiting.

Read the periodic throughput line before turning either. ``requests per forward`` near ``max_batch``
means the workers arrive together and batching is working; near 1 means they arrive one at a time and
the batching is buying nothing.
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from spawnkit._log import get_logger
from spawnkit.oom import abort_worker_on_oom
from spawnkit.service.cudagraph import GraphRunner
from spawnkit.service.loop import IDLE, QUEUE_STOP, STOP, build_model_or_stop, run_worker_loop
from spawnkit.service.rpc import Rpc

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext
    from multiprocessing.process import BaseProcess
    from multiprocessing.synchronize import Event

    from spawnkit.service.shared import SharedRows

log = get_logger(__name__)

_STATS_LOG_INTERVAL_S = 10.0
"""How often the service logs batch-fill statistics."""

_FIRST_REQUEST_POLL_S = 0.5
"""How long the service blocks for a first request before checking the stop event again."""

Request = tuple[int, int, str, Any]
"""What travels on the request queue: ``(client_id, request_id, rpc_name, payload)``."""


class BatchFillStats:
    """Windowed per-RPC counters for the service's periodic throughput log.

    Requests-per-forward is the number to read. Near ``max_batch`` means clients arrive together and
    batching is filling well; near 1 means they arrive one at a time and batching is buying nothing —
    at which point a positive ``batch_window_ms`` is the lever, and only then.

    Durations are deliberately absent. Timing the forward from inside a single-threaded server tells
    you when it ran, not how long the device took, and a profiler attached from outside answers that
    question properly without the service paying for a synchronise on its hot path.

    :param name: the service's name, for the log line.
    """

    def __init__(self, name: str = "BatchedInferenceService") -> None:
        self._name = name
        self._requests: dict[str, int] = {}
        self._forwards: dict[str, int] = {}
        self._max_batch_seen = 0
        self._last_log = time.monotonic()

    def record(self, per_rpc_counts: dict[str, int], batch_size: int) -> None:
        """Fold one processed batch into the running window.

        :param per_rpc_counts: how many requests each RPC served in this batch.
        :param batch_size: total requests in the batch.
        """
        for name, count in per_rpc_counts.items():
            self._requests[name] = self._requests.get(name, 0) + count
            self._forwards[name] = self._forwards.get(name, 0) + (1 if count else 0)
        self._max_batch_seen = max(self._max_batch_seen, batch_size)

    def maybe_log(self, max_batch_capacity: int) -> None:
        """Log throughput once per interval, if any forward ran, then reset the window.

        :param max_batch_capacity: the configured cap, shown beside the largest batch actually seen.
        """
        now = time.monotonic()
        elapsed = now - self._last_log
        if elapsed < _STATS_LOG_INTERVAL_S or not any(self._forwards.values()):
            return

        parts = [
            f"{name} {self._requests[name]} reqs / {self._forwards[name]} fwd "
            f"= {self._requests[name] / max(self._forwards[name], 1):.1f} per forward"
            for name in sorted(self._forwards)
            if self._forwards[name]
        ]
        log.info(
            "[%s] %.0fs: %s | max batch=%d/%d",
            self._name,
            elapsed,
            " | ".join(parts),
            self._max_batch_seen,
            max_batch_capacity,
        )
        self._requests.clear()
        self._forwards.clear()
        self._max_batch_seen = 0
        self._last_log = now


class ModuleReplica:
    """Build and resync a device-local copy of a shared :class:`torch.nn.Module`.

    The common case, packaged: a trainer holds one module in shared memory and updates it in place;
    the service wants its own device copy, refreshed periodically. Both methods are bound methods of
    a picklable object, which is what lets them cross the ``spawn`` boundary — a lambda cannot.

    ``load_state_dict`` copies **in place**, which is also what keeps a captured CUDA graph valid
    across a resync (see :mod:`spawnkit.service.cudagraph`). Do not replace this with an assignment
    that rebinds parameters to new tensors.

    :param shared_module: the trainer's module, already in shared memory
        (``module.share_memory()``).

    Examples
    --------
    >>> replica = ModuleReplica(net)                                  # doctest: +SKIP
    >>> service = BatchedInferenceService(                            # doctest: +SKIP
    ...     build_fn=replica.build, sync_fn=replica.sync, rpcs=[step], ...
    ... )
    """

    def __init__(self, shared_module: torch.nn.Module) -> None:
        self.shared_module = shared_module

    def build(self, device: torch.device) -> torch.nn.Module:
        """Return an eval-mode device copy of the shared module."""
        import copy

        model = copy.deepcopy(self.shared_module).to(device)
        model.eval()
        return model

    def sync(self, model: torch.nn.Module) -> None:
        """Refresh ``model``'s parameters in place from the shared module."""
        model.load_state_dict(self.shared_module.state_dict())


class BatchedInferenceService:
    """Owns the device model and serves batched calls to N clients, in a process of its own.

    **Not a** :class:`multiprocessing.Process` **subclass, deliberately.** Subclassing it binds the
    service to the platform's *default* start method, which on Linux is ``fork`` — and forking a
    parent that has already initialised torch is a deadlock waiting to happen: the child inherits the
    intra-op thread pool's bookkeeping but not its threads, and hangs on its first parallel region
    with no error and no traceback. (Measured while building this package: a forked service sat
    inside ``Linear.forward`` forever while every client waited on a response that never came.) The
    same fork also copies the parent's CUDA state, which is unsupported.

    So the service is a plain picklable object, and :meth:`start` creates the process from a context
    that defaults to ``spawn``. Pass your own context if your application has already chosen one.

    :param build_fn: ``(device) -> model``, run inside this process. Must be picklable, so a
        module-level function or a bound method of a picklable object — see :class:`ModuleReplica`.
    :param rpcs: the registry. Every name a client may call must appear here.
    :param request_queue: the queue every client submits to.
    :param response_queues: one queue per client, indexed by client id.
    :param stop_event: the run's shared stop flag.
    :param device: where the model lives. Several services can shard onto different devices by each
        being given its own.
    :param sync_fn: ``(model) -> None``, refreshing weights from the trainer. ``None`` disables
        resyncing, which is right for a frozen model.
    :param sync_interval: iterations between scheduled weight syncs.
    :param max_batch: most requests served per forward. ``0`` uses the client count. Note this counts
        *requests*, not rows: a client that batches several states into one request contributes one.
    :param batch_window_ms: latency floor for batch fill. ``0`` keeps the opportunistic drain.
    :param shared_rows: buffers for RPCs that declare the shared-memory transport. Must be the same
        object every client received.
    :param graph_rpcs: names of RPCs to accelerate with CUDA-graph replay. Only RPCs that implement
        :meth:`~spawnkit.service.rpc.Rpc.to_tensors` are eligible; others are ignored with a warning.
    :param max_graph_rows: widest batch to capture a graph for.
    :param name: the process name, used in every log line.
    """

    def __init__(
        self,
        build_fn: Callable[[torch.device], Any],
        rpcs: Sequence[Rpc],
        request_queue: Any,
        response_queues: Sequence[Any],
        stop_event: Event,
        device: str | torch.device = "cuda:0",
        sync_fn: Callable[[Any], None] | None = None,
        sync_interval: int = 50,
        max_batch: int = 0,
        batch_window_ms: float = 0.0,
        shared_rows: SharedRows | None = None,
        graph_rpcs: Sequence[str] = (),
        max_graph_rows: int = 512,
        name: str = "BatchedInferenceService",
    ) -> None:
        self._name = name
        self._build_fn = build_fn
        self._sync_fn = sync_fn
        self._rpcs = {rpc.name: rpc for rpc in rpcs}
        self._req_q = request_queue
        self._resp_qs = list(response_queues)
        self._stop = stop_event
        self._device = torch.device(str(device))
        self._sync_interval = max(1, int(sync_interval))
        self._max_batch = int(max_batch) or len(self._resp_qs)
        self._batch_window_s = max(0.0, float(batch_window_ms)) / 1000.0
        self._rows = shared_rows
        self._graph_rpcs = tuple(graph_rpcs)
        self._max_graph_rows = int(max_graph_rows)
        self._runners: dict[str, GraphRunner] = {}

    @property
    def name(self) -> str:
        """The service's name, as it appears in every log line."""
        return self._name

    def start(self, ctx: BaseContext | None = None, daemon: bool = True) -> BaseProcess:
        """Spawn the service process and return its handle.

        :param ctx: the multiprocessing context. Defaults to ``spawn``, which is the only start
            method safe for a process that will initialise a device — see the class docstring.
        :param daemon: whether the service dies with its parent. ``True`` is right for a service
            nothing outlives; set ``False`` if you shut it down explicitly and want the parent to
            wait.
        :return: the started process, ready to hand to
            :class:`~spawnkit.monitor.WorkerSpec` as a critical worker.
        """
        context = ctx or multiprocessing.get_context("spawn")
        # typeshed declares Process on each concrete context rather than on BaseContext, though every
        # context defines it; annotating the parameter as the union of concrete contexts would make
        # the signature unreadable for no gain.
        process = context.Process(target=self.run, name=self._name, daemon=daemon)  # type: ignore[attr-defined]
        process.start()
        return process

    def run(self) -> None:
        """Serve batched requests until the stop event is set or a stop sentinel arrives.

        Called in the service process by :meth:`start`. Run it directly only if you are placing the
        service yourself, and only in a process that has not already initialised torch's thread pool.
        """
        model = build_model_or_stop(lambda: self._build_fn(self._device), self._stop, self._name)
        if model is None:
            return

        self._build_graph_runners(model)
        log.info("[%s] up on %s (max_batch=%d)", self._name, self._device, self._max_batch)
        stats = BatchFillStats(self._name)

        run_worker_loop(
            stop_event=self._stop,
            sync_interval=self._sync_interval,
            sync_fn=lambda: self._sync(model),
            collect_fn=self._collect_batch,
            handle_fn=lambda batch: self._handle_batch(model, batch, stats),
            name=self._name,
        )

    def _sync(self, model: Any) -> None:
        """Refresh the model's weights, or do nothing when the service was given no ``sync_fn``."""
        if self._sync_fn is not None:
            self._sync_fn(model)

    def _build_graph_runners(self, model: Any) -> None:
        """Create one :class:`GraphRunner` per eligible RPC named in ``graph_rpcs``."""
        for rpc_name in self._graph_rpcs:
            rpc = self._rpcs.get(rpc_name)
            if rpc is None:
                log.warning("[%s] graph_rpcs names unknown rpc %r; ignoring", self._name, rpc_name)
                continue
            if not rpc.graph_output_fields:
                log.warning(
                    "[%s] rpc %r declares no graph output fields (it does not expose its inputs as "
                    "tensors); serving it eagerly",
                    self._name,
                    rpc_name,
                )
                continue
            self._runners[rpc_name] = GraphRunner(
                fn=_ModelCall(rpc, model, self._device),
                device=self._device,
                output_fields=rpc.graph_output_fields,
                max_rows=self._max_graph_rows,
            )

    def _collect_batch(self) -> object | list[Request]:
        """Block briefly for the first request, then drain more up to ``max_batch``.

        With ``batch_window_ms == 0`` the drain is a non-blocking burst: take what has already
        arrived and go. With a positive window the service instead *blocks* for more requests until
        the window elapses, so each forward serves more clients — at the cost of up to that much
        extra latency on every call.

        :return: :data:`STOP` if the stop sentinel arrived, :data:`IDLE` on an empty poll window,
            otherwise the pending requests.
        """
        try:
            first = self._req_q.get(timeout=_FIRST_REQUEST_POLL_S)
        except queue_mod.Empty:
            return IDLE
        if first is QUEUE_STOP:
            return STOP

        batch: list[Request] = [first]
        deadline = time.monotonic() + self._batch_window_s if self._batch_window_s > 0 else 0.0
        while len(batch) < self._max_batch:
            try:
                if self._batch_window_s > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    item = self._req_q.get(timeout=remaining)
                else:
                    item = self._req_q.get_nowait()
            except queue_mod.Empty:
                break
            if item is QUEUE_STOP:
                self._stop.set()
                break
            batch.append(item)
        return batch

    def _handle_batch(self, model: Any, batch: list[Request], stats: BatchFillStats) -> None:
        """Serve one collected batch, ending the run if it fails."""
        try:
            counts = self._process_batch(model, batch)
        except Exception as exc:
            # Exit non-zero rather than merely setting the stop event: a batch that would not fit is
            # the run's fatal condition, not a clean end to it.
            abort_worker_on_oom(exc, f"{self._name} batch")
            log.error("[%s] batch failed: %s; stopping the run", self._name, exc, exc_info=True)
            self._stop.set()
            return
        stats.record(counts, len(batch))
        stats.maybe_log(self._max_batch)

    @torch.inference_mode()
    def _process_batch(self, model: Any, batch: list[Request]) -> dict[str, int]:
        """Run one batched forward per RPC present in ``batch``; return per-RPC request counts.

        ``inference_mode`` rather than ``no_grad``: it additionally skips autograd's version-counter
        bookkeeping. Safe because every output is copied to host memory before it reaches anything
        that could need autograd.

        :param model: the device-local model.
        :param batch: the collected requests.
        :return: ``{rpc_name: request_count}``.
        """
        grouped: dict[str, list[Request]] = {}
        for request in batch:
            grouped.setdefault(request[2], []).append(request)

        for rpc_name, requests in grouped.items():
            rpc = self._rpcs[rpc_name]
            if rpc.shared is not None:
                self._serve_shared(model, rpc, requests)
            else:
                self._serve_queued(model, rpc, requests)
        return {name: len(requests) for name, requests in grouped.items()}

    def _serve_queued(self, model: Any, rpc: Rpc, requests: list[Request]) -> None:
        """Collate, forward, split, and send each client its own rows."""
        payloads = [payload for (_cid, _rid, _name, payload) in requests]
        batched = rpc.collate(payloads)
        outputs = self._forward(rpc, model, batched)
        responses = rpc.split(outputs, [rpc.rows_in(payload) for payload in payloads])
        for (client_id, request_id, _name, _payload), response in zip(requests, responses, strict=True):
            self._resp_qs[client_id].put((request_id, response))

    def _serve_shared(self, model: Any, rpc: Rpc, requests: list[Request]) -> None:
        """Gather the batch's rows from shared memory, forward, scatter back, signal each client."""
        if self._rows is None:
            msg = f"rpc {rpc.name!r} declares the shared-memory transport but the service has no shared_rows"
            raise RuntimeError(msg)
        client_ids = torch.tensor([cid for (cid, _rid, _name, _payload) in requests], dtype=torch.long)
        tensors = self._rows.gather_request(client_ids, self._device)
        outputs = self._forward(rpc, model, tensors)
        self._rows.scatter_response(client_ids, outputs)
        for client_id, request_id, _name, _payload in requests:
            self._resp_qs[client_id].put((request_id, None))  # signal only; the data is in the row

    def _forward(self, rpc: Rpc, model: Any, batched: Any) -> Any:
        """Run the RPC, replaying from a CUDA graph when one covers this batch.

        Graph replay is bit-identical to the eager call and several times faster; the runner returns
        ``None`` whenever it cannot serve the batch, and the eager path then runs unchanged. The
        graph's outputs alias its static buffers, which is safe because the caller copies them to
        host memory before the next batch is served.
        """
        runner = self._runners.get(rpc.name)
        if runner is None:
            return rpc.call(model, batched, self._device)

        tensors = rpc.to_tensors(batched, self._device)
        if tensors is None:
            return rpc.call(model, batched, self._device)
        graphed = runner(*tensors)
        if graphed is not None:
            return graphed
        # Pass the tensors rather than the original batch: the device move already happened, and
        # as_tensor/.to on an on-device tensor is a no-op, so this costs nothing and avoids a repeat.
        return rpc.call(model, tensors, self._device)


class _ModelCall:
    """The callable a :class:`GraphRunner` captures: one RPC's forward, with model and device bound.

    A small class rather than a lambda because the runner keeps a reference to it and a named type
    makes the resulting stack traces legible. It is never pickled — the service builds it inside its
    own process, after the model exists.
    """

    def __init__(self, rpc: Rpc, model: Any, device: torch.device) -> None:
        self._rpc = rpc
        self._model = model
        self._device = device

    def __call__(self, *tensors: torch.Tensor) -> Any:
        """Run the RPC's forward on already-batched, already-placed tensors."""
        return self._rpc.call(self._model, tensors, self._device)
