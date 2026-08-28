"""Replay a small, fixed forward from a captured CUDA graph instead of relaunching it every call.

An inference service serving many small requests is usually not GPU-bound and never was. A measured
example: one forward of a modest network issued **145 CUDA kernels** whose combined device time was
~0.4 ms, and the call cost **2.13 ms** whether the batch held 1 row or 512 — flat, because almost
all of it was launch and Python overhead rather than arithmetic. That fixed cost is the ceiling on
throughput when clients issue sequential chains of such calls.

A CUDA graph records the launches once and replays them as a single submission. Measured on an
A100-40GB with that network: **2.13 ms -> 0.39 ms, a 5.5x speedup, with bit-identical outputs** on
every head.

Two properties make this safe rather than clever, and the first is a **precondition you must check
for your own model**:

* **The captured function must be row-independent.** Padding a short batch up to a captured bucket
  size is only sound if no operation mixes rows — elementwise ops, per-row reductions and MLPs
  qualify; batch normalisation in training mode, cross-row attention and any batch-wide reduction do
  not. :meth:`GraphRunner.__call__` verifies exactly this against an eager run the first time it
  uses each captured shape, and disables itself for the rest of the process if the check ever fails.
  Treat that verification as a safety net, not as permission to skip the reasoning.
* **Graphs follow the weights.** Capture records the parameter tensors' addresses, and a weight
  resync that uses ``load_state_dict`` copies *in place*. A synced graph therefore serves the new
  weights with no re-capture. A resync that rebinds parameters to new tensors would not — do not do
  that while a runner is live.

Anything unsupported — a capture failure, a batch wider than the largest bucket, a shape never seen
before that fails to capture — falls back to the eager path. This is an accelerator and never a
behaviour change.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import torch

from spawnkit._log import get_logger

log = get_logger(__name__)

MAX_GRAPH_ROWS = 512
"""Largest bucket captured by default.

Above this the eager path runs. The fixed per-call cost is flat in batch size, so a bucket this wide
already amortises across any realistic client count; capturing wider buys nothing and costs memory.
"""

_WARMUP_ITERATIONS = 3
"""Side-stream warm-up runs before capture, so every lazy allocation and workspace already exists."""

_Key = tuple[int, tuple[tuple[int, ...], ...], tuple[torch.dtype, ...]]
"""Cache key: padded row count, each input's per-row shape tail, and each input's dtype.

The shape tails are part of the key because two clients often disagree about a layout that is
semantically the same — one sends ``[n, 1]`` where another sends ``[n]`` — and if the function
one-hots or reshapes that argument the two are genuinely different forwards, each needing its own
capture. Discovering the tails from real traffic rather than declaring them up front means a new
client layout is handled automatically instead of silently mis-served.
"""


def _bucket_for(rows: int) -> int:
    """Round a row count up to the next power of two, so a handful of graphs cover every batch."""
    return 1 << (rows - 1).bit_length()


class GraphRunner:
    """Serve a fixed forward from CUDA graphs captured lazily per batch-size bucket.

    One graph is captured per (power-of-two row count, input layout) actually seen, each owning its
    own static input and output tensors. Buffers are small — rows x feature widths — so the whole
    cache costs single-digit MB even at the widest bucket.

    :param fn: the callable to capture, taking the batched input tensors positionally and returning
        an object whose ``output_fields`` are tensors. Must be free of Python-side control flow that
        depends on tensor *values*, and must not allocate outside the captured region.
    :param device: the CUDA device the model lives on. A non-CUDA device disables the runner, so a
        caller can construct one unconditionally.
    :param output_fields: names of the output tensors to slice and return.
    :param max_rows: the widest batch to capture; clamped to :data:`MAX_GRAPH_ROWS`.

    Notes
    -----
    :meth:`__call__` returns a :class:`~types.SimpleNamespace`, **not** your model's own output type.
    Read fields by attribute — the batteries-included
    :func:`~spawnkit.service.rpc.tensor_rpc` does exactly that, so the two compose without help.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        device: torch.device,
        output_fields: Sequence[str],
        max_rows: int = MAX_GRAPH_ROWS,
    ) -> None:
        self._fn = fn
        self._device = device
        self._fields = tuple(output_fields)
        self._max_rows = min(int(max_rows), MAX_GRAPH_ROWS)
        self._graphs: dict[_Key, _CapturedGraph] = {}
        self._verified: set[_Key] = set()
        self._enabled = device.type == "cuda"

    @property
    def enabled(self) -> bool:
        """Whether any further capture will be attempted; one failure disables the runner for good."""
        return self._enabled

    def __call__(self, *inputs: torch.Tensor) -> SimpleNamespace | None:
        """Run one batch through a graph, or return ``None`` to ask the caller for the eager path.

        The returned tensors are **views into the graph's static output buffers** and are valid only
        until the next call. Copy them to host memory (or clone) before serving the next batch; a
        batched service does that in its scatter step, which is what makes the aliasing safe there.

        :param inputs: the batched input tensors, already on the runner's device. All must share a
            row count on axis 0.
        :return: an object carrying the ``output_fields``, or ``None`` when this batch cannot be
            served from a graph.
        """
        if not self._enabled or not inputs:
            return None
        rows = int(inputs[0].shape[0])
        if rows > self._max_rows:
            return None

        key: _Key = (
            _bucket_for(rows),
            tuple(tuple(tensor.shape[1:]) for tensor in inputs),
            tuple(tensor.dtype for tensor in inputs),
        )
        captured = self._graphs.get(key)
        if captured is None:
            captured = self._capture(key)
            if captured is None:
                return None
            self._graphs[key] = captured

        try:
            for static, live in zip(captured.static_inputs, inputs, strict=True):
                static[:rows].copy_(live)
            captured.graph.replay()
        except Exception as exc:
            # A service treats an exception from its batch handler as fatal, so a replay hitting an
            # unforeseen shape must degrade to eager rather than take the run down. Disabling is
            # deliberate: a graph that failed once will fail again.
            log.warning("CUDA graph replay failed at %s (%s); falling back to eager", key, exc)
            self._enabled = False
            return None

        outputs = captured.slice_outputs(rows)
        if key not in self._verified and not self._verify(key, inputs, outputs):
            return None
        return outputs

    def _capture(self, key: _Key) -> _CapturedGraph | None:
        """Warm up on a side stream, then capture one graph for ``key``. ``None`` on failure."""
        bucket, tails, dtypes = key
        try:
            static_inputs = [
                torch.zeros(bucket, *tail, dtype=dtype, device=self._device)
                for tail, dtype in zip(tails, dtypes, strict=True)
            ]

            # The documented capture recipe: run the callable a few times on a side stream so every
            # lazy allocation and library workspace exists before the recording starts.
            stream = torch.cuda.Stream(device=self._device)
            stream.wait_stream(torch.cuda.current_stream(self._device))
            with torch.cuda.stream(stream):
                for _ in range(_WARMUP_ITERATIONS):
                    self._fn(*static_inputs)
            torch.cuda.current_stream(self._device).wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_out = self._fn(*static_inputs)
        except Exception as exc:
            log.warning(
                "CUDA graph capture failed at %d rows, layout %s (%s); "
                "falling back to eager inference for the rest of the run",
                bucket,
                tails,
                exc,
            )
            self._enabled = False
            return None

        static_out = {field: getattr(captured_out, field, None) for field in self._fields}
        log.info("Captured CUDA graph for %d rows, layout %s", bucket, tails)
        return _CapturedGraph(graph, static_inputs, static_out)

    def _verify(self, key: _Key, inputs: Sequence[torch.Tensor], graphed: SimpleNamespace) -> bool:
        """Check a freshly used bucket against an eager run; disable the runner if they disagree.

        This is the row-independence guard that padding rests on. It runs once per captured shape, on
        **real traffic** rather than synthetic input, so it exercises the batches the service actually
        sees rather than the ones a test imagined.
        """
        self._verified.add(key)
        reference = self._fn(*inputs)
        mismatched = [
            field
            for field in self._fields
            if not _tensors_equal(getattr(reference, field, None), getattr(graphed, field, None))
        ]
        if not mismatched:
            return True
        log.error(
            "CUDA graph output disagreed with eager inference on %s at %s; "
            "disabling graph replay and falling back to eager inference",
            mismatched,
            key,
        )
        self._enabled = False
        return False


class _CapturedGraph:
    """One captured graph plus the static tensors its replay reads from and writes to."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_inputs: list[torch.Tensor],
        static_out: dict[str, torch.Tensor | None],
    ) -> None:
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_out = static_out

    def slice_outputs(self, rows: int) -> SimpleNamespace:
        """Return the first ``rows`` rows of each output field, as views on the static buffers."""
        return SimpleNamespace(**{
            field: None if tensor is None else tensor[:rows] for field, tensor in self.static_out.items()
        })


def _tensors_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    """Exact equality, treating two absent fields as equal and one absent field as a mismatch."""
    if left is None or right is None:
        return left is None and right is None
    return bool(torch.equal(left, right))
