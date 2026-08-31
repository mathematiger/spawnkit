"""Misuse the service the way a real user plausibly will, and check it says so.

Every case here was found by attacking the package rather than by exercising it, and every one of
them failed silently or fatally before it was fixed. They are grouped in one file because they share
a theme: the service takes input from other processes, and *someone else's* mistake must not become
either a wrong answer or the end of everybody's run.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
from typing import Any, NamedTuple

import pytest

pytest.importorskip("torch")

import numpy as np
import torch

from spawnkit.service import (
    BatchedInferenceService,
    ServiceClient,
    SharedRows,
    SharedRowSpec,
    TensorRpc,
)


class Out(NamedTuple):
    value: torch.Tensor


class Doubler(torch.nn.Module):
    """Row-independent: each row's answer depends only on that row."""

    def infer(self, x: torch.Tensor) -> Out:
        return Out(value=x * 2.0)


class BatchMeanSubtractor(torch.nn.Module):
    """NOT row-independent: subtracts the batch mean, so a row's answer depends on its neighbours.

    Stands in for the realistic culprits — batch normalisation left in training mode, a
    normalisation over ``dim=0``, any pooling across the batch.
    """

    def infer(self, x: torch.Tensor) -> Out:
        return Out(value=x - x.mean(dim=0, keepdim=True))


class RecordingQueue:
    """A queue double that records what was put on it and yields what it was primed with."""

    def __init__(self, items: tuple[Any, ...] = ()) -> None:
        self.items = list(items)
        self.sent: list[Any] = []

    def put(self, item: Any) -> None:
        self.sent.append(item)

    def get(self, timeout: float | None = None) -> Any:
        del timeout
        if self.items:
            return self.items.pop(0)
        raise queue_mod.Empty


def _rpc(name: str = "infer", shared: SharedRowSpec | None = None) -> TensorRpc:
    return TensorRpc(name, method="infer", input_axes=(0,), output_fields=("value",), shared=shared)


# ---------------------------------------------------------------------------
# Row independence — the precondition batching rests on, and the one that fails silently
# ---------------------------------------------------------------------------


def test_a_model_that_mixes_rows_is_refused_rather_than_served() -> None:
    """Batching a row-dependent model gives every client a wrong answer and no error at all.

    Measured before this check existed: three clients sending identical inputs to a model that
    subtracts the batch mean received -1, 0 and +1 — each depending on who it happened to be batched
    with. Nothing raised, nothing logged, and every value looked perfectly plausible.
    """
    model = BatchMeanSubtractor()
    rpc = _rpc()
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[rpc],
        request_queue=RecordingQueue(),
        response_queues=[RecordingQueue(), RecordingQueue()],
        stop_event=mp.get_context("spawn").Event(),
        device="cpu",
    )
    requests = [
        (0, 1, "infer", (np.full((1, 3), 1.0, dtype=np.float32),)),
        (1, 1, "infer", (np.full((1, 3), 5.0, dtype=np.float32),)),
    ]

    with pytest.raises(RuntimeError, match="not row-independent"):
        service._process_batch(model, requests)


def test_a_row_independent_model_passes_verification_once() -> None:
    """The guard must not fire on a correct model, and must not keep paying for itself."""
    model = Doubler()
    rpc = _rpc()
    responses = [RecordingQueue(), RecordingQueue()]
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[rpc],
        request_queue=RecordingQueue(),
        response_queues=responses,
        stop_event=mp.get_context("spawn").Event(),
        device="cpu",
    )
    requests = [
        (0, 1, "infer", (np.full((1, 3), 1.0, dtype=np.float32),)),
        (1, 1, "infer", (np.full((1, 3), 5.0, dtype=np.float32),)),
    ]

    service._process_batch(model, requests)
    assert "infer" in service._verified_rpcs, "the rpc should be marked verified"

    # Each client got its own doubled row, not its neighbour's.
    assert responses[0].sent[0][1]["value"].tolist() == [[2.0, 2.0, 2.0]]
    assert responses[1].sent[0][1]["value"].tolist() == [[10.0, 10.0, 10.0]]


def test_verification_can_be_switched_off() -> None:
    """A caller who has reasoned about their model may decline the startup cost."""
    model = BatchMeanSubtractor()
    rpc = _rpc()
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[rpc],
        request_queue=RecordingQueue(),
        response_queues=[RecordingQueue(), RecordingQueue()],
        stop_event=mp.get_context("spawn").Event(),
        device="cpu",
        verify_row_independence=False,
    )
    requests = [
        (0, 1, "infer", (np.full((1, 3), 1.0, dtype=np.float32),)),
        (1, 1, "infer", (np.full((1, 3), 5.0, dtype=np.float32),)),
    ]

    service._process_batch(model, requests)


def test_the_shared_transport_checks_row_independence_too() -> None:
    """Both transports carry the same precondition, so both must check it.

    This gap was real and shipped: the queue path was guarded and the shared-memory path was not, so
    exactly the same row-dependent model the service refused over the queue was accepted — and
    served wrongly — over shared memory. Three clients sending 1, 2 and 3 got -1, 0 and +1 back.

    The shared path needs its own check rather than reusing the queue's: it never builds per-request
    payloads, because the inputs live in rows and the outputs are scattered straight back into them.
    """
    spec = SharedRowSpec(
        request={"x": ((3,), torch.float32)}, response={"value": ((3,), torch.float32)},
    )
    rows = SharedRows(2, spec)
    rows.write_request(0, {"x": torch.full((3,), 1.0)})
    rows.write_request(1, {"x": torch.full((3,), 5.0)})

    model = BatchMeanSubtractor()
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[_rpc(shared=spec)],
        request_queue=RecordingQueue(),
        response_queues=[RecordingQueue(), RecordingQueue()],
        stop_event=mp.get_context("spawn").Event(),
        device="cpu",
        shared_rows=rows,
    )

    with pytest.raises(RuntimeError, match="not row-independent"):
        service._process_batch(model, [(0, 1, "infer", None), (1, 1, "infer", None)])


def test_the_shared_transport_serves_a_row_independent_model() -> None:
    """The shared-path guard must not fire on a correct model, and must answer each client its own."""
    spec = SharedRowSpec(
        request={"x": ((3,), torch.float32)}, response={"value": ((3,), torch.float32)},
    )
    rows = SharedRows(2, spec)
    rows.write_request(0, {"x": torch.full((3,), 1.0)})
    rows.write_request(1, {"x": torch.full((3,), 5.0)})

    model = Doubler()
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[_rpc(shared=spec)],
        request_queue=RecordingQueue(),
        response_queues=[RecordingQueue(), RecordingQueue()],
        stop_event=mp.get_context("spawn").Event(),
        device="cpu",
        shared_rows=rows,
    )

    service._process_batch(model, [(0, 1, "infer", None), (1, 1, "infer", None)])

    assert rows.read_response(0)["value"].reshape(-1).tolist() == [2.0, 2.0, 2.0]
    assert rows.read_response(1)["value"].reshape(-1).tolist() == [10.0, 10.0, 10.0]


# ---------------------------------------------------------------------------
# One client's mistake must not end everybody's run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_request", "reason"),
    [
        ((7, 1, "infer", (np.zeros((1, 3), dtype=np.float32),)), "client id with no response queue"),
        ((0, 1, "nosuch", (np.zeros((1, 3), dtype=np.float32),)), "rpc name not in the registry"),
    ],
)
def test_an_unroutable_request_is_dropped_not_fatal(bad_request: Any, reason: str) -> None:
    """A request the service cannot answer is that request's fault, not the run's.

    Measured before the fix: one out-of-range client id stopped the service **and exited zero**, so
    the run reported success having served nothing afterwards — the silent clean finish this whole
    package exists to prevent, produced by the package itself.
    """
    model = Doubler()
    stop = mp.get_context("spawn").Event()
    good = RecordingQueue()
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[_rpc()],
        request_queue=RecordingQueue(),
        response_queues=[good],
        stop_event=stop,
        device="cpu",
    )

    service._handle_batch(model, [bad_request], _stats(service))

    assert not stop.is_set(), f"{reason} must not stop the run"


def test_a_good_request_is_still_served_alongside_a_bad_one() -> None:
    """Dropping the unroutable one must not drop its neighbours in the same batch."""
    model = Doubler()
    good = RecordingQueue()
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[_rpc()],
        request_queue=RecordingQueue(),
        response_queues=[good],
        stop_event=mp.get_context("spawn").Event(),
        device="cpu",
        verify_row_independence=False,
    )

    service._handle_batch(
        model,
        [
            (9, 1, "infer", (np.zeros((1, 3), dtype=np.float32),)),          # unroutable
            (0, 1, "infer", (np.full((1, 3), 3.0, dtype=np.float32),)),      # fine
        ],
        _stats(service),
    )

    assert good.sent, "the routable request was dropped along with the bad one"
    assert good.sent[0][1]["value"].tolist() == [[6.0, 6.0, 6.0]]


def test_a_model_that_raises_ends_the_run_non_zero() -> None:
    """A genuine failure must propagate, or the process exits 0 and the run looks successful."""

    class Exploding(torch.nn.Module):
        def infer(self, x: torch.Tensor) -> Out:
            msg = "the model exploded"
            raise ValueError(msg)

    model = Exploding()
    stop = mp.get_context("spawn").Event()
    service = BatchedInferenceService(
        build_fn=lambda _device: model,
        rpcs=[_rpc()],
        request_queue=RecordingQueue(),
        response_queues=[RecordingQueue()],
        stop_event=stop,
        device="cpu",
    )

    with pytest.raises(ValueError, match="exploded"):
        service._handle_batch(
            model,
            [(0, 1, "infer", (np.zeros((1, 3), dtype=np.float32),))],
            _stats(service),
        )
    assert stop.is_set(), "a fatal batch failure must also request a global stop"


# ---------------------------------------------------------------------------
# Payload shapes that silently misalign
# ---------------------------------------------------------------------------


def test_arrays_disagreeing_on_row_count_are_refused() -> None:
    """``rows_in`` reads the first array, so a disagreement splits the batch at the wrong boundary.

    Before this check, ``collate`` concatenated a 2-row array with a 5-row one without complaint and
    the response was cut at the wrong offsets — rows handed to the wrong client, no exception.
    """
    rpc = TensorRpc("pair", method="infer", input_axes=(0, 0), output_fields=("value",))

    with pytest.raises(ValueError, match="differing row counts"):
        rpc.collate([(np.zeros((2, 3), dtype=np.float32), np.zeros((5, 1), dtype=np.int64))])


def test_a_multi_row_request_on_the_shared_transport_says_why() -> None:
    """The one-row limit is structural, so the error should name it rather than reshape-fail."""
    spec = SharedRowSpec(
        request={"x": ((3,), torch.float32)}, response={"value": ((3,), torch.float32)},
    )
    rows = SharedRows(2, spec)

    with pytest.raises(ValueError, match="one row"):
        rows.write_request(0, {"x": torch.zeros(4, 3)})


def test_a_client_using_a_shared_rpc_without_buffers_is_told_so() -> None:
    """Constructing the client without ``shared_rows`` is a wiring mistake with a clear cause."""
    spec = SharedRowSpec(
        request={"x": ((3,), torch.float32)}, response={"value": ((3,), torch.float32)},
    )
    client = ServiceClient(0, RecordingQueue(), RecordingQueue(), [_rpc("s", shared=spec)], None)

    with pytest.raises(RuntimeError, match="without shared_rows"):
        client.call("s", {"x": torch.zeros(3)})


def _stats(service: BatchedInferenceService) -> Any:
    """A throwaway stats object of the type the service's batch handler expects."""
    from spawnkit.service import BatchFillStats

    return BatchFillStats(service.name)
