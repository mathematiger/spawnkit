"""The client side of the batched service: submit one call, block for its answer.

One object per worker process, holding no weights and no device. It puts a request on the shared
request queue and blocks on its own private response queue until the matching answer arrives.

**One in-flight request per client.** :meth:`ServiceClient.call` is synchronous, which is what makes
the invariant hold, and a great deal depends on it: the shared-memory transport can give each client
exactly one row without locking, and the ordering guard below can be an equality check rather than a
correlation table. The guard is not decoration — if a response ever arrives with the wrong id, every
subsequent answer this client receives belongs to a different request, and the results are wrong
rather than missing. Better to stop.

The blocking wait is bounded and re-checks the run's stop event, so a client whose service has died
raises instead of waiting out the job's wall clock — the failure this whole package exists to
prevent, in miniature.
"""

from __future__ import annotations

import queue as queue_mod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from spawnkit.service.rpc import Response, Rpc

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event

    from spawnkit.service.shared import SharedRows

_RESPONSE_POLL_S = 1.0
"""How long a single blocking read waits before re-checking the stop event."""


class ServiceClient:
    """Forwards inference calls to a :class:`~spawnkit.service.batched.BatchedInferenceService`.

    :param client_id: this worker's index. Indexes both ``response_queues`` in the service and, on
        the shared-memory transport, the client's row. Must be unique and less than the number of
        response queues the service was given.
    :param request_queue: the queue shared by every client, drained by the service.
    :param response_queue: this client's own queue. Private on purpose — a shared response queue
        would need correlation ids and would let one slow client's answer sit behind another's.
    :param rpcs: the same RPC registry the service was given. The client needs it to know which
        calls use the shared-memory transport.
    :param stop_event: the run's stop flag. When set while this client is waiting, the wait raises
        instead of continuing — a service that has died will never answer.
    :param shared_rows: the shared-memory buffers, when any RPC declares them. Must be the *same*
        object the service received, allocated once in the parent before either spawned.
    :param response_timeout: seconds per blocking read before re-checking ``stop_event``.

    Examples
    --------
    >>> client = ServiceClient(0, req_q, resp_qs[0], rpcs, stop)      # doctest: +SKIP
    >>> out = client.call("step", (hidden, action))                   # doctest: +SKIP
    >>> out["value"].shape                                            # doctest: +SKIP
    (1, 51)
    """

    def __init__(
        self,
        client_id: int,
        request_queue: Any,
        response_queue: Any,
        rpcs: Sequence[Rpc],
        stop_event: Event | None = None,
        shared_rows: SharedRows | None = None,
        response_timeout: float = _RESPONSE_POLL_S,
    ) -> None:
        self.client_id = int(client_id)
        self._req_q = request_queue
        self._resp_q = response_queue
        self._rpcs = {rpc.name: rpc for rpc in rpcs}
        self._stop = stop_event
        self._rows = shared_rows
        self._timeout = response_timeout
        self._counter = 0

    def call(self, rpc_name: str, payload: Any) -> Response:
        """Run one batched call on the service and return this request's own rows.

        :param rpc_name: the name of a registered RPC.
        :param payload: what that RPC's ``collate`` expects for one request. On the shared-memory
            transport this is instead a ``{field: tensor}`` mapping matching the RPC's
            :class:`~spawnkit.service.shared.SharedRowSpec`.
        :return: ``{output_field: array}`` for this request's rows.
        :raises KeyError: if ``rpc_name`` is not registered.
        :raises RuntimeError: if the service stopped while this call was waiting, or if a response
            arrived with an id that does not match the request.
        """
        rpc = self._rpcs[rpc_name]
        if rpc.shared is not None:
            return self._call_shared(rpc, payload)
        return self._call_queued(rpc, payload)

    def _call_queued(self, rpc: Rpc, payload: Any) -> Response:
        """Submit through the queue, carrying the payload; the response carries the result."""
        self._counter += 1
        self._req_q.put((self.client_id, self._counter, rpc.name, payload))
        result = self._await(self._counter)
        return dict(result)

    def _call_shared(self, rpc: Rpc, payload: Any) -> Response:
        """Submit through shared memory: write our row, send only the id, read our row back."""
        if self._rows is None:
            msg = (
                f"rpc {rpc.name!r} declares the shared-memory transport but this client was "
                "constructed without shared_rows"
            )
            raise RuntimeError(msg)
        self._rows.write_request(self.client_id, payload)
        self._counter += 1
        self._req_q.put((self.client_id, self._counter, rpc.name, None))
        self._await(self._counter)
        # read_response clones, so turning the clone into numpy shares memory with a buffer only this
        # caller holds - the response row itself is free to be overwritten by our next call.
        return {name: tensor.numpy() for name, tensor in self._rows.read_response(self.client_id).items()}

    def _await(self, request_id: int) -> Any:
        """Block until this client's matching response arrives; return its payload.

        :param request_id: the id we are waiting for.
        :return: the response payload (``None`` on the shared-memory transport, where the data is in
            the client's row rather than on the wire).
        :raises RuntimeError: on a stopped service, or on an id mismatch.
        """
        while True:
            try:
                response_id, result = self._resp_q.get(timeout=self._timeout)
            except queue_mod.Empty:
                if self._stop is not None and self._stop.is_set():
                    msg = "inference service stopped while a client was awaiting a response"
                    raise RuntimeError(msg) from None
                continue
            if response_id != request_id:
                msg = (
                    f"[client {self.client_id}] response id {response_id} != request id "
                    f"{request_id}: responses arrived out of order "
                    "(invariant: one in-flight request per client)"
                )
                raise RuntimeError(msg)
            return result


def as_torch(response: Response) -> dict[str, Any]:
    """Convert a response's arrays to torch tensors, without copying.

    A convenience for callers that want tensors back. ``torch.from_numpy`` shares memory with the
    array, and the arrays in a response are already private to this caller, so this is free.

    :param response: what :meth:`ServiceClient.call` returned.
    :return: the same fields as tensors.
    """
    import torch

    return {name: torch.from_numpy(np.ascontiguousarray(array)) for name, array in response.items()}
