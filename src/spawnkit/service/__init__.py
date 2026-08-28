"""Batched GPU inference: one process owns the model, N worker processes ask it questions.

Needs the ``[torch]`` extra — ``pip install spawnkit[torch]``. The other two tiers do not.

The shape::

    worker (ServiceClient) ─ request_queue ─► BatchedInferenceService (device, batched)
                           ◄─ response_queues[client_id] ─┘

What it buys, in the order the problems bite:

* **VRAM is 1x the model, not N x.** N workers each holding their own copy plus their own CUDA
  context is what caps worker count on a large model long before compute does.
* **Batching is nearly free.** Small forwards are launch-bound, not arithmetic-bound — a measured
  example ran flat at 2.13 ms from batch 1 to batch 512 — so one forward serving sixteen workers
  costs about what one serving a single worker costs.
* **CUDA-graph replay takes that forward further**, 2.13 ms to 0.39 ms on the same model, with
  bit-identical outputs and an automatic first-use check that verifies exactly that.

Declare each remote call as an :class:`~spawnkit.service.rpc.Rpc` — :class:`TensorRpc` covers the
common "concatenate in, slice named fields out" case — and the service handles collection, batching,
device placement, scatter and the failure policy. Two transports: the default queue, and
:class:`~spawnkit.service.shared.SharedRowSpec` shared-memory rows for a hot path where pickling is
a measurable share of the round trip.
"""

from __future__ import annotations

from spawnkit.service.batched import (
    BatchedInferenceService,
    BatchFillStats,
    ModuleReplica,
    Request,
)
from spawnkit.service.client import ServiceClient, as_torch
from spawnkit.service.cudagraph import MAX_GRAPH_ROWS, GraphRunner
from spawnkit.service.loop import (
    IDLE,
    QUEUE_STOP,
    STOP,
    build_model_or_stop,
    maybe_sync_weights,
    run_worker_loop,
)
from spawnkit.service.rpc import Payload, Response, Rpc, TensorRpc
from spawnkit.service.shared import FieldSpec, SharedRows, SharedRowSpec

__all__ = [
    "IDLE",
    "MAX_GRAPH_ROWS",
    "QUEUE_STOP",
    "STOP",
    "BatchFillStats",
    "BatchedInferenceService",
    "FieldSpec",
    "GraphRunner",
    "ModuleReplica",
    "Payload",
    "Request",
    "Response",
    "Rpc",
    "ServiceClient",
    "SharedRowSpec",
    "SharedRows",
    "TensorRpc",
    "as_torch",
    "build_model_or_stop",
    "maybe_sync_weights",
    "run_worker_loop",
]
