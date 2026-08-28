"""What an RPC promises: N payloads in, one forward, N responses back to the right clients.

The single most important property here is the one in ``test_split_gives_each_request_back_its_own``:
``split`` must cut the batched output at the boundaries ``collate`` glued it together at, in the same
order, for requests that own *different* numbers of rows. When that is wrong the service does not
crash and no shape mismatches — every client simply receives somebody else's answer, which looks
like a model that has stopped learning rather than like a transport bug. So the round trip is
asserted on values, not on shapes.

The rest of the file pins the properties that make the class-based design work at all: per-axis
concatenation (an LSTM state batches on axis 1 while the inputs beside it batch on axis 0), the
``None`` output field that is dropped rather than sent, and picklability — under ``spawn`` the
registry is pickled into the service process, and a closure-based design could not have been.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from spawnkit.service.rpc import Rpc, TensorRpc
from spawnkit.service.shared import SharedRowSpec

CPU = torch.device("cpu")

HIDDEN = 4
"""Width of the fake hidden state used by the round-trip tests."""

LAYERS = 2
"""Recurrent layer count, so the axis-1 state is genuinely rank 3."""


class ScalingModel:
    """A stand-in model: every output is a fixed multiple of its input, so results stay traceable."""

    def scale(self, features: Any) -> SimpleNamespace:
        """Return the input scaled, plus an optional head that is always absent."""
        return SimpleNamespace(value=features * 10, absent=None)

    def recurrent(self, features: Any, state: Any) -> SimpleNamespace:
        """Return an axis-1 state and a rank-1 per-row scalar from the same call."""
        return SimpleNamespace(state=state + 1, done=features[:, 0] * 100)


def scale_features(model: Any, features: Any) -> SimpleNamespace:
    """Module-level callable ``method``: picklable where a lambda would not be."""
    return model.scale(features)


def rows_of(count: int, offset: int) -> np.ndarray:
    """Build ``count`` rows whose every entry identifies the request that sent them."""
    base = np.arange(count * HIDDEN, dtype=np.float32).reshape(count, HIDDEN)
    return base + offset * 1000.0


@pytest.fixture
def scale_rpc() -> TensorRpc:
    """The plain case: one array in on axis 0, one named field out on axis 0."""
    return TensorRpc("scale", method="scale", input_axes=(0,), output_fields=("value", "absent"))


def test_collate_concatenates_each_position_on_its_own_axis() -> None:
    """Per-axis is the whole point: the state batches on axis 1, the features on axis 0."""
    rpc = TensorRpc(
        "recurrent",
        method="recurrent",
        input_axes=(0, 1),
        output_fields=("state",),
        row_axis=1,
    )
    first = (rows_of(2, 1), np.zeros((LAYERS, 2, HIDDEN), dtype=np.float32))
    second = (rows_of(3, 2), np.ones((LAYERS, 3, HIDDEN), dtype=np.float32))

    features, state = rpc.collate([first, second])

    assert features.shape == (5, HIDDEN)
    assert state.shape == (LAYERS, 5, HIDDEN)
    np.testing.assert_array_equal(features[:2], first[0])
    np.testing.assert_array_equal(features[2:], second[0])
    np.testing.assert_array_equal(state[:, :2], first[1])
    np.testing.assert_array_equal(state[:, 2:], second[1])


def test_collate_rejects_a_payload_whose_arity_disagrees_with_the_axes(scale_rpc: TensorRpc) -> None:
    """One axis was declared, two arrays arrived: say so rather than concatenating the wrong pair."""
    payload = (rows_of(1, 0), rows_of(1, 1))

    with pytest.raises(ValueError, match=r"2 arrays but 1 axes were declared"):
        scale_rpc.collate([payload])


def test_split_gives_each_request_back_its_own(scale_rpc: TensorRpc) -> None:
    """Requests of 1, 3 and 2 rows each get their own values back, in submission order.

    This is the misattribution failure. Assert on the values, because every wrong answer here has
    exactly the right shape.
    """
    counts = [1, 3, 2]
    payloads = [(rows_of(count, index),) for index, count in enumerate(counts)]

    batched = scale_rpc.collate(payloads)
    outputs = ScalingModel().scale(torch.as_tensor(batched[0]))
    responses = scale_rpc.split(outputs, [scale_rpc.rows_in(payload) for payload in payloads])

    assert len(responses) == len(counts)
    for payload, response in zip(payloads, responses, strict=True):
        np.testing.assert_allclose(response["value"], payload[0] * 10)


def test_split_cuts_at_the_row_boundaries_it_was_given(scale_rpc: TensorRpc) -> None:
    """The same property stated directly on the slicing, without a model in the way."""
    value = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    responses = scale_rpc.split(SimpleNamespace(value=value, absent=None), [1, 3, 2])

    np.testing.assert_array_equal(responses[0]["value"], value[0:1].numpy())
    np.testing.assert_array_equal(responses[1]["value"], value[1:4].numpy())
    np.testing.assert_array_equal(responses[2]["value"], value[4:6].numpy())


def test_split_drops_an_absent_output_field(scale_rpc: TensorRpc) -> None:
    """An optional head that produced nothing costs nothing: the key is missing, not ``None``."""
    outputs = SimpleNamespace(value=torch.zeros(2, 2), absent=None)

    responses = scale_rpc.split(outputs, [2])

    assert "absent" in scale_rpc.output_fields
    assert set(responses[0]) == {"value"}


def test_row_axis_falls_back_to_axis_zero_for_a_field_of_lower_rank() -> None:
    """A rank-3 state (batch on axis 1) and a rank-1 per-row scalar leave the same call correctly."""
    rpc = TensorRpc(
        "recurrent",
        method="recurrent",
        input_axes=(0, 1),
        output_fields=("state", "done"),
        row_axis=1,
    )
    payloads = [
        (rows_of(count, index), np.full((LAYERS, count, HIDDEN), index, dtype=np.float32))
        for index, count in enumerate([1, 3, 2])
    ]

    batched = rpc.collate(payloads)
    outputs = rpc.call(ScalingModel(), batched, CPU)
    responses = rpc.split(outputs, [rpc.rows_in(payload) for payload in payloads])

    for payload, response in zip(payloads, responses, strict=True):
        # state: rank 3, sliced on the declared row_axis of 1.
        np.testing.assert_allclose(response["state"], payload[1] + 1)
        # done: rank 1, too low for row_axis, so it falls back to axis 0.
        np.testing.assert_allclose(response["done"], payload[0][:, 0] * 100)


def test_rows_in_reports_a_multi_row_requests_true_count(scale_rpc: TensorRpc) -> None:
    """A client that batched four states into one call owns four rows of the answer."""
    assert scale_rpc.rows_in((rows_of(4, 0),)) == 4
    assert scale_rpc.rows_in((rows_of(1, 0),)) == 1


def test_rows_in_reads_the_first_arrays_declared_axis() -> None:
    """When the first argument batches on axis 1, that is the axis the row count comes from."""
    rpc = TensorRpc("state_only", method="scale", input_axes=(1,), output_fields=("value",), row_axis=1)

    assert rpc.rows_in((np.zeros((LAYERS, 3, HIDDEN), dtype=np.float32),)) == 3


def test_an_rpc_survives_the_pickle_into_the_service_process(scale_rpc: TensorRpc) -> None:
    """Under ``spawn`` the registry is pickled, and the copy has to still work — not merely exist."""
    revived = pickle.loads(pickle.dumps(scale_rpc))

    payloads = [(rows_of(2, 0),), (rows_of(3, 1),)]
    batched = revived.collate(payloads)
    outputs = revived.call(ScalingModel(), batched, CPU)
    responses = revived.split(outputs, [revived.rows_in(payload) for payload in payloads])

    assert revived.name == scale_rpc.name
    assert revived.input_axes == scale_rpc.input_axes
    for payload, response in zip(payloads, responses, strict=True):
        np.testing.assert_allclose(response["value"], payload[0] * 10)


def test_a_callable_method_survives_the_pickle_too() -> None:
    """A module-level function is accepted as ``method`` and still pickles; a lambda would not."""
    rpc = TensorRpc("scale", method=scale_features, input_axes=(0,), output_fields=("value",))

    revived = pickle.loads(pickle.dumps(rpc))
    outputs = revived.call(ScalingModel(), (rows_of(2, 0),), CPU)

    np.testing.assert_allclose(outputs.value.numpy(), rows_of(2, 0) * 10)


def test_the_base_class_refuses_the_three_steps_it_cannot_guess() -> None:
    """``Rpc`` is a contract, not a default: collate, call and split must be written."""
    rpc = Rpc("bare")

    with pytest.raises(NotImplementedError):
        rpc.collate([])
    with pytest.raises(NotImplementedError):
        rpc.call(object(), None, CPU)
    with pytest.raises(NotImplementedError):
        rpc.split(None, [])


def test_the_base_class_is_ineligible_for_graph_replay_rather_than_broken() -> None:
    """No tensor view of its inputs means no capture — served eagerly, which is correct."""
    rpc = Rpc("bare")

    assert rpc.to_tensors(object(), CPU) is None
    assert rpc.graph_output_fields == ()
    assert rpc.rows_in(object()) == 1


def test_a_tensor_rpc_exposes_its_inputs_and_outputs_to_the_graph_runner(scale_rpc: TensorRpc) -> None:
    """The concrete case is eligible: tensors in call order, and the declared fields out."""
    tensors = scale_rpc.to_tensors((rows_of(2, 0),), CPU)

    assert tensors is not None
    assert len(tensors) == 1
    assert isinstance(tensors[0], torch.Tensor)
    assert scale_rpc.graph_output_fields == ("value", "absent")


def test_repr_names_the_transport(scale_rpc: TensorRpc) -> None:
    """Which transport an RPC uses is the first thing to check when a call misbehaves."""
    spec = SharedRowSpec(
        request={"features": ((HIDDEN,), torch.float32)},
        response={"value": ((HIDDEN,), torch.float32)},
    )
    shared = TensorRpc("scale", method="scale", input_axes=(0,), output_fields=("value",), shared=spec)

    assert repr(scale_rpc) == "TensorRpc(name='scale', transport='queue')"
    assert repr(shared) == "TensorRpc(name='scale', transport='shared')"
    assert repr(Rpc("bare")) == "Rpc(name='bare', transport='queue')"


def test_row_axis_is_identified_by_extent_not_by_rank() -> None:
    """Fields batching on different axes in one call must each be sliced on their own batch axis.

    The regression this pins: the axis used to be chosen by rank alone, so with ``row_axis=1`` a
    ``[batch, values]`` head - rank 2, and therefore "high enough" - was sliced along *values*. Every
    request received the whole batch and the last one silently lost columns. No exception and no
    shape error at the client, which is what made it the misattribution failure the module warns
    about rather than a crash.
    """
    rpc = TensorRpc(
        "recurrent",
        method="step",
        input_axes=(0,),
        output_fields=("state", "value"),
        row_axis=1,
    )
    outputs = SimpleNamespace(
        state=torch.zeros(2, 6, 4),                    # [layers, batch, hidden] - batches on axis 1
        value=torch.arange(30.0).reshape(6, 5),        # [batch, values]          - batches on axis 0
    )

    responses = rpc.split(outputs, [1, 3, 2])

    assert [r["state"].shape for r in responses] == [(2, 1, 4), (2, 3, 4), (2, 2, 4)]
    assert [r["value"].shape for r in responses] == [(1, 5), (3, 5), (2, 5)]
    # The values themselves, because the shapes alone were right for the wrong reason before.
    assert [float(r["value"][0, 0]) for r in responses] == [0.0, 5.0, 20.0]
