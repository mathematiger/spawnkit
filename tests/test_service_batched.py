"""The service end to end, in a real process, against real clients.

This is the only file in the service suite that spawns anything, and the reason is the property it
exists to check: **does a client get back what a direct eager call on the same model would have
produced?** Everything the service does — collate, device move, batched forward, split, scatter — is
plumbing between those two points, and plumbing that is subtly wrong here returns answers of exactly
the right shape belonging to somebody else. So both transports are compared against an eager call on
the very same shared module, and three concurrent clients are each checked against their own input.

Two structural properties get an assertion of their own because they were failures first:

* **The process is spawned, not forked.** Forking a parent that has already initialised torch gives
  the child the intra-op thread pool's bookkeeping without its threads; the service then sits inside
  its first ``Linear.forward`` forever, with no error and no traceback, while every client waits on
  a response that never comes.
* **The service runs on one intra-op thread.** Left at torch's default of one thread per core, the
  service and the workers oversubscribe the node between them. It does not show up on an idle
  machine and it does not show up at batch 1, which is why it is asserted here rather than left to a
  benchmark: the ``threads`` RPC reports ``torch.get_num_threads()`` from inside the service process.

One module-scoped service serves the whole file, so the spawn cost is paid once. It is shut down —
stop event, stop sentinel, bounded join, then terminate — in a ``finally``, so a failing assertion
cannot leave a process behind.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from spawnkit.service import batched as batched_module
from spawnkit.service.batched import (
    BatchedInferenceService,
    BatchFillStats,
    ModuleReplica,
)
from spawnkit.service.client import ServiceClient
from spawnkit.service.loop import QUEUE_STOP
from spawnkit.service.rpc import Rpc, TensorRpc
from spawnkit.service.shared import SharedRows, SharedRowSpec

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from multiprocessing.context import SpawnContext
    from multiprocessing.process import BaseProcess

pytestmark = pytest.mark.timeout(180)

FEATURE_WIDTH = 6
"""Input width of the toy model."""

VALUE_WIDTH = 3
"""Width of its main head."""

NUM_CLIENTS = 8
"""Response queues and shared rows allocated; more than the file uses, so ids stay readable."""

SEED = 20260828
"""Fixed so the parent's reference model and the service's copy are the same weights."""

BATCH_WINDOW_MS = 25.0
"""Long enough that concurrent clients land in one forward, short enough to be free here."""

CLIENT_POLL_S = 0.25
"""How long a client blocks before re-checking the stop event; keeps a broken run short."""

RESULT_WAIT_S = 60.0
"""Bound on every cross-process read, so no failure in this file can hang the session."""

SHUTDOWN_S = 20.0
"""Bound on each shutdown join."""

TOLERANCE = 1e-5
"""Batching changes the GEMM shape, so agreement with eager is float agreement, not bit equality."""

SHARED_SPEC = SharedRowSpec(
    request={"features": ((FEATURE_WIDTH,), torch.float32)},
    response={"value": ((VALUE_WIDTH,), torch.float32), "score": ((1,), torch.float32)},
)
"""One row per client for the shared-memory transport: features in, both heads back."""


class TinyModel(torch.nn.Module):
    """Two linear heads and a probe that reports the service process's own thread setting."""

    def __init__(self) -> None:
        super().__init__()
        self.value_head = torch.nn.Linear(FEATURE_WIDTH, VALUE_WIDTH)
        self.score_head = torch.nn.Linear(FEATURE_WIDTH, 1)

    def infer(self, features: Any) -> SimpleNamespace:
        """Return both heads for a batch of feature rows."""
        return SimpleNamespace(value=self.value_head(features), score=self.score_head(features))

    def threads(self, features: Any) -> SimpleNamespace:
        """Report ``torch.get_num_threads()`` once per row, from wherever this model is running."""
        count = float(torch.get_num_threads())
        return SimpleNamespace(threads=torch.full((int(features.shape[0]), 1), count))


def build_shared_model() -> TinyModel:
    """Build the trainer-side module the service replicates: fixed weights, in shared memory."""
    torch.manual_seed(SEED)
    model = TinyModel()
    model.eval()
    model.share_memory()
    return model


def build_rpcs() -> list[Rpc]:
    """The registry both the service and every client are given."""
    return [
        TensorRpc("infer", method="infer", input_axes=(0,), output_fields=("value", "score")),
        TensorRpc("threads", method="threads", input_axes=(0,), output_fields=("threads",)),
        TensorRpc(
            "shared_infer",
            method="infer",
            input_axes=(0,),
            output_fields=("value", "score"),
            shared=SHARED_SPEC,
        ),
    ]


def features_for(client_id: int) -> np.ndarray:
    """A single feature row that identifies the client that sent it."""
    return (np.linspace(-1.0, 1.0, FEATURE_WIDTH, dtype=np.float32) + client_id).reshape(1, FEATURE_WIDTH)


def eager_infer(model: TinyModel, features: np.ndarray) -> SimpleNamespace:
    """What a caller would have got by running the model directly, with no service in the way."""
    with torch.inference_mode():
        return model.infer(torch.as_tensor(features))


def run_queue_client(
    client_id: int,
    request_queue: Any,
    response_queue: Any,
    rpcs: Sequence[Rpc],
    stop_event: Any,
    result_queue: Any,
) -> None:
    """A whole worker process: build a client, ask one question, report the answer home."""
    client = ServiceClient(
        client_id,
        request_queue,
        response_queue,
        rpcs,
        stop_event,
        response_timeout=CLIENT_POLL_S,
    )
    response = client.call("infer", (features_for(client_id),))
    result_queue.put((client_id, response["value"], response["score"]))


@dataclass
class ServiceHarness:
    """Everything a test needs to talk to the one running service."""

    context: SpawnContext
    process: BaseProcess
    request_queue: Any
    response_queues: list[Any]
    stop_event: Any
    rpcs: list[Rpc]
    shared_rows: SharedRows
    model: TinyModel

    def client(self, client_id: int) -> ServiceClient:
        """A client for ``client_id``, running in this (the parent) process."""
        return ServiceClient(
            client_id,
            self.request_queue,
            self.response_queues[client_id],
            self.rpcs,
            self.stop_event,
            self.shared_rows,
            response_timeout=CLIENT_POLL_S,
        )


@pytest.fixture(scope="module")
def harness() -> Iterator[ServiceHarness]:
    """Start one service for the whole module, and take it down whatever happens."""
    context = multiprocessing.get_context("spawn")
    request_queue = context.Queue()
    response_queues = [context.Queue() for _ in range(NUM_CLIENTS)]
    stop_event = context.Event()
    model = build_shared_model()
    replica = ModuleReplica(model)
    shared_rows = SharedRows(num_clients=NUM_CLIENTS, spec=SHARED_SPEC)
    rpcs = build_rpcs()

    service = BatchedInferenceService(
        build_fn=replica.build,
        rpcs=rpcs,
        request_queue=request_queue,
        response_queues=response_queues,
        stop_event=stop_event,
        device="cpu",
        sync_fn=replica.sync,
        sync_interval=3,
        max_batch=NUM_CLIENTS,
        batch_window_ms=BATCH_WINDOW_MS,
        shared_rows=shared_rows,
        # An unknown name here must be reported and ignored; if it stopped the service instead,
        # every test in this file would fail rather than this one property.
        graph_rpcs=("no_such_rpc", "infer"),
        name="test-service",
    )
    process = service.start()
    try:
        yield ServiceHarness(
            context=context,
            process=process,
            request_queue=request_queue,
            response_queues=response_queues,
            stop_event=stop_event,
            rpcs=rpcs,
            shared_rows=shared_rows,
            model=model,
        )
    finally:
        stop_event.set()
        request_queue.put(QUEUE_STOP)
        process.join(timeout=SHUTDOWN_S)
        if process.is_alive():
            process.terminate()
            process.join(timeout=SHUTDOWN_S)


def test_start_spawns_the_service_rather_than_forking_it(harness: ServiceHarness) -> None:
    """A forked service inherits torch's thread-pool bookkeeping without its threads, and hangs."""
    assert isinstance(harness.process, SpawnProcess)
    assert harness.process.is_alive()


def test_the_queue_transport_agrees_with_an_eager_call(harness: ServiceHarness) -> None:
    """The whole point of the service: the same answer, computed somewhere else."""
    features = features_for(0)

    response = harness.client(0).call("infer", (features,))
    reference = eager_infer(harness.model, features)

    assert set(response) == {"value", "score"}
    assert response["value"].shape == (1, VALUE_WIDTH)
    np.testing.assert_allclose(response["value"], reference.value.numpy(), rtol=TOLERANCE, atol=TOLERANCE)
    np.testing.assert_allclose(response["score"], reference.score.numpy(), rtol=TOLERANCE, atol=TOLERANCE)


def test_a_multi_row_request_comes_back_whole(harness: ServiceHarness) -> None:
    """A client that batched four states into one call owns four rows of the answer."""
    features = np.stack([features_for(index)[0] for index in range(4)])

    response = harness.client(0).call("infer", (features,))
    reference = eager_infer(harness.model, features)

    assert response["value"].shape == (4, VALUE_WIDTH)
    np.testing.assert_allclose(response["value"], reference.value.numpy(), rtol=TOLERANCE, atol=TOLERANCE)


def test_the_shared_transport_agrees_with_an_eager_call(harness: ServiceHarness) -> None:
    """No payload on the wire, the data in a pre-allocated row — and the same answer regardless."""
    features = features_for(1)

    response = harness.client(1).call("shared_infer", {"features": torch.as_tensor(features[0])})
    reference = eager_infer(harness.model, features)

    assert response["value"].shape == (1, VALUE_WIDTH)
    np.testing.assert_allclose(response["value"], reference.value.numpy(), rtol=TOLERANCE, atol=TOLERANCE)
    np.testing.assert_allclose(response["score"], reference.score.numpy(), rtol=TOLERANCE, atol=TOLERANCE)


def test_the_two_transports_agree_with_each_other(harness: ServiceHarness) -> None:
    """A hot path moved onto shared memory must not become a different call."""
    features = features_for(2)
    client = harness.client(2)

    queued = client.call("infer", (features,))
    shared = client.call("shared_infer", {"features": torch.as_tensor(features[0])})

    np.testing.assert_allclose(shared["value"], queued["value"], rtol=TOLERANCE, atol=TOLERANCE)
    np.testing.assert_allclose(shared["score"], queued["score"], rtol=TOLERANCE, atol=TOLERANCE)


def test_each_client_in_a_batch_gets_its_own_rows(harness: ServiceHarness) -> None:
    """Three real worker processes, three distinguishable inputs, three answers that must not swap.

    This is the failure the split/scatter step exists to prevent, and it is invisible from inside a
    client: every wrong answer here has the right shape and a plausible value.
    """
    client_ids = [3, 4, 5]
    result_queue = harness.context.Queue()
    processes = [
        harness.context.Process(
            target=run_queue_client,
            args=(
                client_id,
                harness.request_queue,
                harness.response_queues[client_id],
                harness.rpcs,
                harness.stop_event,
                result_queue,
            ),
            daemon=True,
        )
        for client_id in client_ids
    ]
    for process in processes:
        process.start()

    try:
        results = {}
        for _ in client_ids:
            client_id, value, score = result_queue.get(timeout=RESULT_WAIT_S)
            results[client_id] = (value, score)
    finally:
        for process in processes:
            process.join(timeout=SHUTDOWN_S)
            if process.is_alive():
                process.terminate()
                process.join(timeout=SHUTDOWN_S)

    assert sorted(results) == client_ids
    for client_id in client_ids:
        value, score = results[client_id]
        reference = eager_infer(harness.model, features_for(client_id))
        np.testing.assert_allclose(value, reference.value.numpy(), rtol=TOLERANCE, atol=TOLERANCE)
        np.testing.assert_allclose(score, reference.score.numpy(), rtol=TOLERANCE, atol=TOLERANCE)


def test_the_service_process_runs_on_one_intra_op_thread(harness: ServiceHarness) -> None:
    """Not a throttle, a guard: the default of one is what stops the service oversubscribing the node.

    The probe runs inside the service process, so this reads the setting that is actually in force
    there rather than the parent's.
    """
    response = harness.client(0).call("threads", (features_for(0),))

    assert response["threads"].tolist() == [[1.0]]


def test_two_rpcs_in_one_batch_are_each_served_correctly(harness: ServiceHarness) -> None:
    """Concurrent calls to different RPCs are grouped per name; neither answer may leak into the other."""
    features = features_for(6)
    answers: dict[str, Any] = {}

    def call_infer() -> None:
        answers["infer"] = harness.client(6).call("infer", (features,))

    def call_threads() -> None:
        answers["threads"] = harness.client(7).call("threads", (features,))

    workers = [threading.Thread(target=call_infer), threading.Thread(target=call_threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=RESULT_WAIT_S)
        assert not worker.is_alive()

    reference = eager_infer(harness.model, features)
    np.testing.assert_allclose(answers["infer"]["value"], reference.value.numpy(), rtol=TOLERANCE, atol=TOLERANCE)
    assert answers["threads"]["threads"].tolist() == [[1.0]]


def test_an_unknown_graph_rpc_name_did_not_stop_the_service(harness: ServiceHarness) -> None:
    """The harness asked for a graph on a name that does not exist; the service must still serve."""
    assert harness.process.is_alive()

    response = harness.client(0).call("infer", (features_for(0),))

    assert response["value"].shape == (1, VALUE_WIDTH)


def test_an_unknown_graph_rpc_name_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """Ignoring it silently would leave someone wondering why their fast path never got faster."""
    service = BatchedInferenceService(
        build_fn=lambda device: build_shared_model().to(device),
        rpcs=build_rpcs(),
        request_queue=None,
        response_queues=[],
        stop_event=multiprocessing.Event(),
        device="cpu",
        graph_rpcs=("no_such_rpc",),
        name="offline-service",
    )

    with caplog.at_level(logging.WARNING, logger="spawnkit"):
        service._build_graph_runners(TinyModel())

    assert any("no_such_rpc" in message for message in caplog.messages)
    assert any("unknown rpc" in message for message in caplog.messages)


def test_a_mixed_batch_is_grouped_into_one_forward_per_rpc() -> None:
    """The grouping itself, asserted on the counts the service reports for one collected batch."""
    responses: dict[int, list[Any]] = {index: [] for index in range(3)}

    class Collector:
        def __init__(self, client_id: int) -> None:
            self._client_id = client_id

        def put(self, item: Any) -> None:
            responses[self._client_id].append(item)

    service = BatchedInferenceService(
        build_fn=lambda device: build_shared_model().to(device),
        rpcs=build_rpcs(),
        request_queue=None,
        response_queues=[Collector(index) for index in range(3)],
        stop_event=multiprocessing.Event(),
        device="cpu",
        name="offline-service",
    )
    model = build_shared_model()
    batch = [
        (0, 1, "infer", (features_for(0),)),
        (1, 1, "threads", (features_for(1),)),
        (2, 1, "infer", (features_for(2),)),
    ]

    counts = service._process_batch(model, batch)

    assert counts == {"infer": 2, "threads": 1}
    for client_id in range(3):
        assert len(responses[client_id]) == 1
    np.testing.assert_allclose(
        responses[0][0][1]["value"],
        eager_infer(model, features_for(0)).value.numpy(),
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        responses[2][0][1]["value"],
        eager_infer(model, features_for(2)).value.numpy(),
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )


def test_batch_fill_stats_accumulate_per_rpc(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requests per forward is the number to read, so it has to be the number that gets logged."""
    monkeypatch.setattr(batched_module, "_STATS_LOG_INTERVAL_S", 0.0)
    stats = BatchFillStats("svc")
    stats.record({"infer": 3}, 3)
    stats.record({"infer": 1, "threads": 2}, 3)

    with caplog.at_level(logging.INFO, logger="spawnkit"):
        stats.maybe_log(8)

    line = "\n".join(caplog.messages)
    assert "infer 4 reqs / 2 fwd = 2.0 per forward" in line
    assert "threads 2 reqs / 1 fwd = 2.0 per forward" in line
    assert "max batch=3/8" in line


def test_maybe_log_resets_the_window(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A window that was not reset would report the same requests again on the next interval."""
    monkeypatch.setattr(batched_module, "_STATS_LOG_INTERVAL_S", 0.0)
    stats = BatchFillStats("svc")
    stats.record({"infer": 2}, 2)

    with caplog.at_level(logging.INFO, logger="spawnkit"):
        stats.maybe_log(8)
        assert caplog.messages
        caplog.clear()
        stats.maybe_log(8)

    assert caplog.messages == []


def test_stats_stay_quiet_inside_the_logging_interval(caplog: pytest.LogCaptureFixture) -> None:
    """The throughput line is periodic, not per batch; a line per forward would drown the run's log."""
    stats = BatchFillStats("svc")
    stats.record({"infer": 1}, 1)

    with caplog.at_level(logging.INFO, logger="spawnkit"):
        stats.maybe_log(8)

    assert caplog.messages == []
