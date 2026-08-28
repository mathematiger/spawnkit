"""One pre-allocated row per client, and the four ways that arrangement can quietly corrupt data.

Every failure this module can have is silent. A write that touches the wrong row, a gather that
reorders the batch, a scatter that puts client 3's answer in client 0's slot — none of them raise,
none of them change a shape, and all of them produce a run that trains on somebody else's inference.
So the tests here assert on *values in identified rows*, and the neighbouring rows are checked for
having been left alone.

``test_read_response_hands_out_a_clone_not_a_view`` is the one worth reading twice. The row is reused
by the client's next call, so a view handed to a caller that keeps it — into a search tree, a replay
buffer — changes under that caller some milliseconds later. The clone is the fix; this test is what
notices if it is ever optimised away.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch

from spawnkit.service.shared import SharedRows, SharedRowSpec

CPU = torch.device("cpu")

NUM_CLIENTS = 4
"""Enough rows that "the other rows are untouched" is a real assertion."""

HIDDEN = 3
"""Width of the request row."""

VALUES = 2
"""Width of the response row."""


@pytest.fixture
def spec() -> SharedRowSpec:
    """One request field and two response fields, one of which the model may not produce."""
    return SharedRowSpec(
        request={"hidden": ((HIDDEN,), torch.float32), "action": ((1,), torch.long)},
        response={"value": ((VALUES,), torch.float32), "optional": ((1,), torch.float32)},
    )


@pytest.fixture
def rows(spec: SharedRowSpec) -> SharedRows:
    """The allocated buffers, as the parent process allocates them before anything spawns."""
    return SharedRows(num_clients=NUM_CLIENTS, spec=spec)


def payload_for(client_id: int, *, flat: bool = True) -> dict[str, torch.Tensor]:
    """A request payload whose every entry identifies the client that wrote it."""
    hidden = torch.full((HIDDEN,), float(client_id) + 1.0)
    return {
        "hidden": hidden if flat else hidden.unsqueeze(0),
        "action": torch.tensor([client_id], dtype=torch.long),
    }


def test_allocation_is_one_shared_row_per_client(rows: SharedRows) -> None:
    """``[num_clients, *shape]`` per field, in shared memory, or the spawn handoff gives each
    process a private copy that never agrees with the others.
    """
    assert rows.request["hidden"].shape == (NUM_CLIENTS, HIDDEN)
    assert rows.request["action"].shape == (NUM_CLIENTS, 1)
    assert rows.response["value"].shape == (NUM_CLIENTS, VALUES)
    assert rows.request["action"].dtype == torch.long

    for buffer in (*rows.request.values(), *rows.response.values()):
        assert buffer.is_shared()


def test_write_request_accepts_a_flat_row_and_a_batch_of_one(rows: SharedRows) -> None:
    """Clients disagree about ``[H]`` versus ``[1, H]`` and neither is wrong."""
    rows.write_request(1, payload_for(1, flat=True))
    rows.write_request(2, payload_for(2, flat=False))

    torch.testing.assert_close(rows.request["hidden"][1], torch.full((HIDDEN,), 2.0))
    torch.testing.assert_close(rows.request["hidden"][2], torch.full((HIDDEN,), 3.0))


def test_write_request_touches_only_the_addressed_row(rows: SharedRows) -> None:
    """A client owns exactly its own row; writing it must not disturb a neighbour's in-flight call."""
    rows.write_request(2, payload_for(2))

    untouched = [index for index in range(NUM_CLIENTS) if index != 2]
    for index in untouched:
        torch.testing.assert_close(rows.request["hidden"][index], torch.zeros(HIDDEN))
        torch.testing.assert_close(rows.request["action"][index], torch.zeros(1, dtype=torch.long))


def test_write_request_names_a_missing_declared_field(rows: SharedRows) -> None:
    """A payload short of a declared field fails by name, not by writing a stale row."""
    with pytest.raises(KeyError, match="action"):
        rows.write_request(0, {"hidden": torch.zeros(HIDDEN)})


def test_read_response_hands_out_a_clone_not_a_view(rows: SharedRows) -> None:
    """The row is reused by the next call; a view would change under a caller that kept it."""
    rows.response["value"][1] = torch.tensor([7.0, 8.0])

    first = rows.read_response(1)
    rows.response["value"][1] = torch.tensor([-1.0, -2.0])

    torch.testing.assert_close(first["value"], torch.tensor([[7.0, 8.0]]))
    torch.testing.assert_close(rows.read_response(1)["value"], torch.tensor([[-1.0, -2.0]]))


def test_read_response_shapes_the_row_like_a_single_row_queue_response(rows: SharedRows) -> None:
    """``[1, *field_shape]``, so a caller cannot tell the two transports apart by shape."""
    response = rows.read_response(0)

    assert response["value"].shape == (1, VALUES)
    assert response["optional"].shape == (1, 1)


def test_gather_request_collects_rows_in_the_given_client_order(rows: SharedRows) -> None:
    """The batch's order is the service's contract with ``scatter_response``; it must be preserved."""
    for client_id in range(NUM_CLIENTS):
        rows.write_request(client_id, payload_for(client_id))

    order = torch.tensor([3, 0, 2], dtype=torch.long)
    hidden, action = rows.gather_request(order, CPU)

    assert hidden.shape == (3, HIDDEN)
    torch.testing.assert_close(hidden[:, 0], torch.tensor([4.0, 1.0, 3.0]))
    torch.testing.assert_close(action.reshape(-1), torch.tensor([3, 0, 2], dtype=torch.long))


def test_scatter_response_writes_each_row_back_to_its_own_client(rows: SharedRows) -> None:
    """Out-of-order ids are the normal case — a batch is whatever arrived, not a contiguous range."""
    order = torch.tensor([3, 0, 2], dtype=torch.long)
    outputs = SimpleNamespace(
        value=torch.tensor([[30.0, 31.0], [0.0, 1.0], [20.0, 21.0]]),
        optional=torch.tensor([[3.0], [0.0], [2.0]]),
    )

    rows.scatter_response(order, outputs)

    torch.testing.assert_close(rows.read_response(3)["value"], torch.tensor([[30.0, 31.0]]))
    torch.testing.assert_close(rows.read_response(0)["value"], torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(rows.read_response(2)["value"], torch.tensor([[20.0, 21.0]]))
    # Client 1 was not in the batch and must not have been written.
    torch.testing.assert_close(rows.read_response(1)["value"], torch.zeros(1, VALUES))


def test_scatter_response_reshapes_a_flat_head_into_the_row_layout(rows: SharedRows) -> None:
    """A head returning ``[n]`` and one returning ``[n, 1]`` both land correctly."""
    order = torch.tensor([0, 1], dtype=torch.long)
    outputs = SimpleNamespace(
        value=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        optional=torch.tensor([5.0, 6.0]),
    )

    rows.scatter_response(order, outputs)

    torch.testing.assert_close(rows.read_response(0)["optional"], torch.tensor([[5.0]]))
    torch.testing.assert_close(rows.read_response(1)["optional"], torch.tensor([[6.0]]))


def test_scatter_response_leaves_a_field_the_model_did_not_produce(rows: SharedRows) -> None:
    """An optional head that stayed silent leaves the row at what it held, and raises nothing."""
    rows.response["optional"][0] = torch.tensor([9.0])
    order = torch.tensor([0], dtype=torch.long)

    rows.scatter_response(order, SimpleNamespace(value=torch.tensor([[1.0, 2.0]]), optional=None))

    torch.testing.assert_close(rows.read_response(0)["value"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(rows.read_response(0)["optional"], torch.tensor([[9.0]]))


def test_the_spec_preserves_the_declared_field_order(spec: SharedRowSpec) -> None:
    """``gather_request`` returns tensors positionally, so the declaration order is the contract."""
    assert list(spec.request) == ["hidden", "action"]

    rows = SharedRows(num_clients=1, spec=spec)
    gathered = rows.gather_request(torch.zeros(1, dtype=torch.long), CPU)

    assert gathered[0].shape == (1, HIDDEN)
    assert gathered[1].shape == (1, 1)
