"""The client's two jobs: get the answer back, and refuse to accept the wrong one.

Everything here runs against fake queues. The client holds no weights, no device and no process — it
puts a tuple on one queue and reads a tuple off another — so a real service would add seconds per
test and cover nothing this file is about.

What it *is* about is the pair of failures that a synchronous client can have, and both are worse
than a crash:

* **A response arrives with the wrong id.** The invariant is one in-flight request per client, so an
  id mismatch means the stream has slipped by one and every subsequent answer this client receives
  belongs to a different request. Nothing is missing, everything is wrong, and the run keeps going.
  The guard turns that into a ``RuntimeError``.
* **The answer never arrives.** A service that died leaves its clients blocked on a queue that no one
  will ever write to, and a job then burns its wall clock producing nothing. The wait is bounded and
  re-checks the run's stop event, so the client raises instead.
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from spawnkit.service.client import ServiceClient, as_torch
from spawnkit.service.rpc import TensorRpc
from spawnkit.service.shared import SharedRows, SharedRowSpec

pytestmark = pytest.mark.timeout(30)

HIDDEN = 3
"""Width of the fake request row."""

VALUES = 2
"""Width of the fake response row."""

FAST_TIMEOUT_S = 0.01
"""Response poll used throughout, so a test that waits at all waits for milliseconds."""

EMPTY = object()
"""Scripted "nothing had arrived yet" — the fake queue raises ``queue.Empty`` for this entry."""


class FakeRequestQueue:
    """Records what the client submitted. There is no service on the other end, and none is needed."""

    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def put(self, item: Any) -> None:
        """Record one submission."""
        self.submitted.append(item)


class FakeResponseQueue:
    """Hands out a scripted sequence of responses; an exhausted script never answers again."""

    def __init__(self, script: list[Any] | None = None) -> None:
        self._script = list(script or [])
        self.reads = 0

    def get(self, timeout: float | None = None) -> Any:
        """Return the next scripted response, or behave like a queue nothing has been put on."""
        del timeout
        self.reads += 1
        if not self._script:
            raise queue_mod.Empty
        item = self._script.pop(0)
        if item is EMPTY:
            raise queue_mod.Empty
        return item


@pytest.fixture
def scale_rpc() -> TensorRpc:
    """A plain queue-transport RPC."""
    return TensorRpc("scale", method="scale", input_axes=(0,), output_fields=("value",))


@pytest.fixture
def shared_spec() -> SharedRowSpec:
    """The row layout for the shared-memory transport."""
    return SharedRowSpec(
        request={"hidden": ((HIDDEN,), torch.float32)},
        response={"value": ((VALUES,), torch.float32)},
    )


@pytest.fixture
def shared_rpc(shared_spec: SharedRowSpec) -> TensorRpc:
    """The same call, declared onto the shared-memory transport."""
    return TensorRpc(
        "shared_scale",
        method="scale",
        input_axes=(0,),
        output_fields=("value",),
        shared=shared_spec,
    )


def test_a_successful_call_returns_the_response_fields(scale_rpc: TensorRpc) -> None:
    """The queue transport hands back ``{field: ndarray}`` — the field names are the wire contract."""
    expected = np.array([[1.0, 2.0]], dtype=np.float32)
    request_queue = FakeRequestQueue()
    client = ServiceClient(0, request_queue, FakeResponseQueue([(1, {"value": expected})]), [scale_rpc])

    result = client.call("scale", (np.zeros((1, HIDDEN), dtype=np.float32),))

    assert set(result) == {"value"}
    assert isinstance(result["value"], np.ndarray)
    np.testing.assert_array_equal(result["value"], expected)


def test_the_queue_transport_carries_the_payload(scale_rpc: TensorRpc) -> None:
    """``(client_id, request_id, rpc_name, payload)``, with ids that count from one."""
    payload = (np.zeros((1, HIDDEN), dtype=np.float32),)
    request_queue = FakeRequestQueue()
    script: list[Any] = [(1, {"value": np.zeros((1, VALUES))}), (2, {"value": np.zeros((1, VALUES))})]
    client = ServiceClient(5, request_queue, FakeResponseQueue(script), [scale_rpc])

    client.call("scale", payload)
    client.call("scale", payload)

    assert [(cid, rid, name) for cid, rid, name, _ in request_queue.submitted] == [
        (5, 1, "scale"),
        (5, 2, "scale"),
    ]
    assert request_queue.submitted[0][3] is payload


def test_a_response_id_that_does_not_match_the_request_raises(scale_rpc: TensorRpc) -> None:
    """The stream has slipped by one: every later answer would belong to a different request."""
    client = ServiceClient(
        3,
        FakeRequestQueue(),
        FakeResponseQueue([(99, {"value": np.zeros((1, VALUES))})]),
        [scale_rpc],
    )

    with pytest.raises(RuntimeError, match=r"one in-flight request per client") as caught:
        client.call("scale", (np.zeros((1, HIDDEN), dtype=np.float32),))

    message = str(caught.value)
    assert "out of order" in message
    assert "client 3" in message


def test_the_client_keeps_waiting_while_the_run_is_healthy(scale_rpc: TensorRpc) -> None:
    """An empty poll window is not a failure — it is the normal case for a service under load."""
    stop_event = multiprocessing.Event()
    response_queue = FakeResponseQueue([EMPTY, EMPTY, (1, {"value": np.zeros((1, VALUES))})])
    client = ServiceClient(
        0,
        FakeRequestQueue(),
        response_queue,
        [scale_rpc],
        stop_event,
        response_timeout=FAST_TIMEOUT_S,
    )

    result = client.call("scale", (np.zeros((1, HIDDEN), dtype=np.float32),))

    assert response_queue.reads == 3
    assert "value" in result
    assert not stop_event.is_set()


def test_a_stopped_service_raises_rather_than_blocking_forever(scale_rpc: TensorRpc) -> None:
    """The failure this package exists to prevent, in miniature: no answer, and no waiting for one."""
    stop_event = multiprocessing.Event()
    stop_event.set()
    client = ServiceClient(
        0,
        FakeRequestQueue(),
        FakeResponseQueue(),
        [scale_rpc],
        stop_event,
        response_timeout=FAST_TIMEOUT_S,
    )

    with pytest.raises(RuntimeError, match=r"inference service stopped"):
        client.call("scale", (np.zeros((1, HIDDEN), dtype=np.float32),))


def test_the_shared_transport_sends_an_identifier_and_no_payload(
    shared_rpc: TensorRpc,
    shared_spec: SharedRowSpec,
) -> None:
    """Write the row, put ``(client_id, request_id, name, None)``, read the row back."""
    rows = SharedRows(num_clients=4, spec=shared_spec)
    rows.response["value"][2] = torch.tensor([4.0, 5.0])
    request_queue = FakeRequestQueue()
    client = ServiceClient(
        2,
        request_queue,
        FakeResponseQueue([(1, None)]),
        [shared_rpc],
        shared_rows=rows,
    )

    result = client.call("shared_scale", {"hidden": torch.tensor([1.0, 2.0, 3.0])})

    assert request_queue.submitted == [(2, 1, "shared_scale", None)]
    torch.testing.assert_close(rows.request["hidden"][2], torch.tensor([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(result["value"], np.array([[4.0, 5.0]], dtype=np.float32))


def test_the_shared_transport_returns_a_copy_the_next_call_cannot_overwrite(
    shared_rpc: TensorRpc,
    shared_spec: SharedRowSpec,
) -> None:
    """The response row is reused immediately; what the caller holds must not be that row."""
    rows = SharedRows(num_clients=2, spec=shared_spec)
    rows.response["value"][0] = torch.tensor([1.0, 2.0])
    client = ServiceClient(
        0,
        FakeRequestQueue(),
        FakeResponseQueue([(1, None)]),
        [shared_rpc],
        shared_rows=rows,
    )

    result = client.call("shared_scale", {"hidden": torch.zeros(HIDDEN)})
    rows.response["value"][0] = torch.tensor([-9.0, -9.0])

    np.testing.assert_array_equal(result["value"], np.array([[1.0, 2.0]], dtype=np.float32))


def test_a_shared_rpc_without_shared_rows_says_exactly_that(shared_rpc: TensorRpc) -> None:
    """A client built without the buffers cannot serve this call, and the message must say why."""
    client = ServiceClient(0, FakeRequestQueue(), FakeResponseQueue(), [shared_rpc])

    with pytest.raises(RuntimeError, match=r"constructed without shared_rows"):
        client.call("shared_scale", {"hidden": torch.zeros(HIDDEN)})


def test_an_unregistered_rpc_name_raises_key_error(scale_rpc: TensorRpc) -> None:
    """A typo in a call name fails at the client, before anything reaches the service."""
    client = ServiceClient(0, FakeRequestQueue(), FakeResponseQueue(), [scale_rpc])

    with pytest.raises(KeyError, match="nope"):
        client.call("nope", (np.zeros((1, HIDDEN), dtype=np.float32),))


def test_as_torch_round_trips_a_response_without_copying() -> None:
    """The arrays in a response are already private to this caller, so the view is free and safe."""
    array = np.arange(6, dtype=np.float32).reshape(2, 3)

    tensors = as_torch({"value": array})

    assert isinstance(tensors["value"], torch.Tensor)
    assert np.shares_memory(tensors["value"].numpy(), array)
    np.testing.assert_array_equal(tensors["value"].numpy(), array)
