"""What the graph runner does when it cannot help, and what it does when it can.

Most of this file runs without a GPU, because most of what matters is the *refusal* path. A
:class:`~spawnkit.service.cudagraph.GraphRunner` is meant to be constructed unconditionally — a
caller should never have to ask whether capture is possible — which only works if a runner on a
device that cannot capture is inert rather than broken. So the CPU tests assert the whole behaviour:
``enabled`` is ``False``, every call returns ``None``, and the wrapped function is never invoked, so
the caller's eager path runs exactly as it would have without a runner at all.

The capture path itself needs a real device and is marked ``gpu``, which CI deselects. Those three
tests are the ones that matter on a machine that has one: a replay that agrees with eager, the
first-use verification that catches a function whose rows are *not* independent (padding a short
batch up to a captured bucket is only sound if nothing mixes rows), and the permanent disable that
follows such a mismatch — one bad capture must not be retried for the rest of the run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from spawnkit.service.cudagraph import MAX_GRAPH_ROWS, GraphRunner, _bucket_for

pytestmark = pytest.mark.timeout(120)

CPU = torch.device("cpu")

CUDA_AVAILABLE = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="a real CUDA device is needed to capture")

WIDTH = 8
"""Feature width of the captured forward."""


class RowIndependentFn:
    """A forward no operation of which mixes rows, so padding a short batch is sound."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *tensors: Any) -> SimpleNamespace:
        """Scale the input elementwise and count the invocation."""
        self.calls += 1
        return SimpleNamespace(value=tensors[0] * 2.0, offset=tensors[0] + 1.0)


class RowMixingFn:
    """A forward that reduces across the batch, which padding silently changes."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *tensors: Any) -> SimpleNamespace:
        """Add the batch mean to every row — correct eagerly, wrong once padding rows exist."""
        self.calls += 1
        features = tensors[0]
        return SimpleNamespace(value=features + features.mean(dim=0, keepdim=True))


def cpu_runner(fn: Any, max_rows: int = MAX_GRAPH_ROWS) -> GraphRunner:
    """A runner on the CPU, which is always disabled."""
    return GraphRunner(fn=fn, device=CPU, output_fields=("value", "offset"), max_rows=max_rows)


def test_a_cpu_device_disables_the_runner_and_captures_nothing() -> None:
    """The caller falls back to eager, and pays nothing for having constructed a runner."""
    fn = RowIndependentFn()
    runner = cpu_runner(fn)

    assert runner.enabled is False
    assert runner(torch.zeros(4, WIDTH)) is None
    assert runner(torch.zeros(1, WIDTH)) is None
    assert fn.calls == 0


@pytest.mark.parametrize(
    ("rows", "bucket"),
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (511, 512), (512, 512), (513, 1024)],
)
def test_bucket_for_rounds_up_to_the_next_power_of_two(rows: int, bucket: int) -> None:
    """A handful of graphs then covers every batch width the service will ever see."""
    assert _bucket_for(rows) == bucket


def test_a_batch_wider_than_max_rows_returns_none() -> None:
    """Above the widest bucket the eager path runs; capturing wider buys nothing and costs memory."""
    fn = RowIndependentFn()
    runner = cpu_runner(fn, max_rows=16)
    # The device check would return first, so enable the runner to reach the width check itself.
    # Nothing below it touches CUDA for an over-wide batch.
    runner._enabled = True

    assert runner(torch.zeros(17, WIDTH)) is None
    assert fn.calls == 0


def test_empty_inputs_return_none() -> None:
    """There is no row count to bucket and nothing to copy into a static buffer."""
    fn = RowIndependentFn()
    runner = cpu_runner(fn)
    runner._enabled = True

    assert runner() is None
    assert fn.calls == 0


def test_max_rows_is_clamped_to_the_module_cap() -> None:
    """Asking for a wider bucket than the module supports is clamped, not honoured."""
    runner = cpu_runner(RowIndependentFn(), max_rows=MAX_GRAPH_ROWS * 4)

    assert runner._max_rows == MAX_GRAPH_ROWS


@pytest.mark.gpu
@requires_cuda
def test_replay_reproduces_the_eager_result() -> None:
    """Capture, replay, and agree with eager on every declared field — bit for bit."""
    device = torch.device("cuda:0")
    fn = RowIndependentFn()
    runner = GraphRunner(fn=fn, device=device, output_fields=("value", "offset"))
    features = torch.arange(3 * WIDTH, dtype=torch.float32, device=device).reshape(3, WIDTH)

    graphed = runner(features)

    assert runner.enabled is True
    assert graphed is not None
    reference = fn(features)
    assert torch.equal(graphed.value, reference.value)
    assert torch.equal(graphed.offset, reference.offset)


@pytest.mark.gpu
@requires_cuda
def test_one_graph_serves_every_batch_width_in_its_bucket() -> None:
    """Three rows and four rows share the bucket of four, so only one capture happens."""
    device = torch.device("cuda:0")
    runner = GraphRunner(fn=RowIndependentFn(), device=device, output_fields=("value", "offset"))

    assert runner(torch.zeros(3, WIDTH, device=device)) is not None
    assert runner(torch.zeros(4, WIDTH, device=device)) is not None

    assert len(runner._graphs) == 1


@pytest.mark.gpu
@requires_cuda
def test_a_row_mixing_forward_is_caught_by_the_first_use_check_and_disabled_for_good() -> None:
    """Padding is only sound when rows are independent; the check is what makes that safe to assume.

    A batch mean over a padded bucket is not the batch mean over the real rows, so the first use of
    the capture disagrees with eager. The runner must then refuse this batch *and* every later one:
    a capture that was wrong once is wrong permanently.
    """
    device = torch.device("cuda:0")
    runner = GraphRunner(fn=RowMixingFn(), device=device, output_fields=("value",))
    features = torch.arange(3 * WIDTH, dtype=torch.float32, device=device).reshape(3, WIDTH)

    assert runner(features) is None
    assert runner.enabled is False
    assert runner(features) is None
