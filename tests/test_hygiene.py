"""Unit tests for :mod:`spawnkit.hygiene`.

The three public entry points are one-liners whose *value* is entirely in where and when they are
called, so the tests pin the contract a caller can get wrong:

* :func:`cuda_hidden_from_children` and :func:`blas_threads_pinned` must leave ``os.environ`` exactly
  as they found it, including when the body raises — they wrap ``Process.start()``, and a leaked
  empty ``CUDA_VISIBLE_DEVICES`` would silently move the parent's own GPU work onto the CPU for the
  rest of the run.
* :func:`prepare_cpu_only_worker` must pin threads for a CPU worker, leave a non-CPU worker alone,
  and reach the ``torch`` import only on the CPU path.

What cannot be tested in-process is the part that matters most: that masking works only from the
parent. torch caches the visible-device count on first query and offers no way to clear it, so a
test asserting "the child saw no GPU" would need a real spawn on a real GPU box. That is recorded in
the function's docstring as a measurement instead.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from types import ModuleType
from typing import Any, cast

import pytest

from spawnkit.hygiene import (
    BLAS_THREAD_VARS,
    CUDA_VISIBLE_DEVICES,
    blas_threads_pinned,
    cuda_hidden_from_children,
    prepare_cpu_only_worker,
)

_SPAWN_TIMEOUT_S = 60.0


# ---------------------------------------------------------------------------
# cuda_hidden_from_children
# ---------------------------------------------------------------------------


def test_the_variable_is_empty_inside_the_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty value is what a spawned child reads as "no GPUs exist for me"."""
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "0,1")
    with cuda_hidden_from_children():
        assert os.environ[CUDA_VISIBLE_DEVICES] == ""


def test_a_previous_value_is_restored_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent may own a CUDA context of its own; clobbering its device list would move its work."""
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "2,3")
    with cuda_hidden_from_children():
        pass
    assert os.environ[CUDA_VISIBLE_DEVICES] == "2,3"


def test_an_absent_variable_is_left_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restoring "" instead of removing the key would hide every GPU from the rest of the process."""
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES, raising=False)
    with cuda_hidden_from_children():
        assert os.environ[CUDA_VISIBLE_DEVICES] == ""
    assert CUDA_VISIBLE_DEVICES not in os.environ


def test_the_variable_is_restored_even_when_the_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Process.start()`` is what runs inside the block, and it can raise (fork failure, bad target).

    Without the ``finally`` the mask would leak into the parent and every worker started afterwards,
    which on a single-GPU box means the parent's own training silently continues on the CPU.
    """
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "0")
    with pytest.raises(OSError, match="cannot allocate memory"), cuda_hidden_from_children():
        raise OSError("cannot allocate memory")
    assert os.environ[CUDA_VISIBLE_DEVICES] == "0"


def test_hide_false_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lets a caller write "hide unless this worker wants a GPU" without branching around the with."""
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "0,1")
    with cuda_hidden_from_children(hide=False):
        assert os.environ[CUDA_VISIBLE_DEVICES] == "0,1"
    assert os.environ[CUDA_VISIBLE_DEVICES] == "0,1"


def test_hide_false_leaves_an_absent_variable_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The no-op path must not create the key either — that alone would hide the GPUs."""
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES, raising=False)
    with cuda_hidden_from_children(hide=False):
        assert CUDA_VISIBLE_DEVICES not in os.environ
    assert CUDA_VISIBLE_DEVICES not in os.environ


def test_nested_blocks_restore_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting a pool inside an outer mask must still hand the parent its original list back."""
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "0,1")
    with cuda_hidden_from_children(), cuda_hidden_from_children():
        assert os.environ[CUDA_VISIBLE_DEVICES] == ""
    assert os.environ[CUDA_VISIBLE_DEVICES] == "0,1"


# ---------------------------------------------------------------------------
# blas_threads_pinned
# ---------------------------------------------------------------------------


@pytest.fixture
def _clear_blas_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from a known state: none of the five variables set, all restored afterwards."""
    for name in BLAS_THREAD_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("_clear_blas_vars")
def test_every_blas_variable_is_set_inside_the_block() -> None:
    """All five, because which one is live depends on how numpy was built.

    A child that inherits four of them still fans its BLAS out through the fifth.
    """
    with blas_threads_pinned():
        assert [os.environ.get(name) for name in BLAS_THREAD_VARS] == ["1"] * len(BLAS_THREAD_VARS)


@pytest.mark.usefixtures("_clear_blas_vars")
def test_the_thread_count_is_configurable() -> None:
    """One thread per child is the default, not the only sensible answer."""
    with blas_threads_pinned(threads=4):
        assert [os.environ.get(name) for name in BLAS_THREAD_VARS] == ["4"] * len(BLAS_THREAD_VARS)


@pytest.mark.usefixtures("_clear_blas_vars")
def test_variables_that_were_absent_are_removed_again() -> None:
    """Leaving "1" behind would pin the parent's own math to a single thread for the rest of the run."""
    with blas_threads_pinned():
        pass
    assert [name for name in BLAS_THREAD_VARS if name in os.environ] == []


def test_previous_values_are_restored_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator who set these deliberately must get their values back, one by one."""
    original = {name: str(index + 2) for index, name in enumerate(BLAS_THREAD_VARS)}
    for name, value in original.items():
        monkeypatch.setenv(name, value)

    with blas_threads_pinned():
        assert os.environ[BLAS_THREAD_VARS[0]] == "1"

    assert {name: os.environ[name] for name in BLAS_THREAD_VARS} == original


@pytest.mark.usefixtures("_clear_blas_vars")
def test_a_mix_of_present_and_absent_variables_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """The two restore branches run in the same block, so they are exercised together."""
    present = BLAS_THREAD_VARS[0]
    monkeypatch.setenv(present, "8")

    with blas_threads_pinned():
        pass

    assert os.environ[present] == "8"
    assert [name for name in BLAS_THREAD_VARS[1:] if name in os.environ] == []


@pytest.mark.usefixtures("_clear_blas_vars")
def test_the_variables_are_restored_even_when_the_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Process.start()`` runs inside the block; a leaked pin would throttle the parent silently."""
    monkeypatch.setenv(BLAS_THREAD_VARS[0], "16")

    with pytest.raises(RuntimeError, match="failed to start"), blas_threads_pinned():
        raise RuntimeError("failed to start")

    assert os.environ[BLAS_THREAD_VARS[0]] == "16"
    assert [name for name in BLAS_THREAD_VARS[1:] if name in os.environ] == []


# ---------------------------------------------------------------------------
# prepare_cpu_only_worker
# ---------------------------------------------------------------------------


class _FakeTorch:
    """Stand-in for the ``torch`` module; records the thread counts it was asked for."""

    def __init__(self) -> None:
        self.thread_counts: list[int] = []

    def set_num_threads(self, threads: int) -> None:
        self.thread_counts.append(threads)


class _FakeDevice:
    """Stand-in for ``torch.device``: only its string form is consulted."""

    def __init__(self, spec: str) -> None:
        self._spec = spec

    def __str__(self) -> str:
        return self._spec


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> _FakeTorch:
    """Install a fake ``torch`` in ``sys.modules`` so the CPU path is testable without the real one."""
    fake = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", cast("ModuleType", cast("Any", fake)))
    return fake


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
def test_a_cpu_worker_is_pinned_to_one_thread(fake_torch: _FakeTorch, device: str) -> None:
    """Pin a CPU worker to one thread; torch otherwise gives every worker a thread per core.

    Measured on a 60-core node with three worker processes: leaving it unset put the trainer at
    ~4.5 s per gradient step; pinning took the same run from >150 s to 12 s.
    """
    prepare_cpu_only_worker(device)
    assert fake_torch.thread_counts == [1]


def test_the_default_device_is_cpu(fake_torch: _FakeTorch) -> None:
    """A worker body that does not know its own placement can call this with no arguments."""
    prepare_cpu_only_worker()
    assert fake_torch.thread_counts == [1]


def test_a_device_object_is_accepted_too(fake_torch: _FakeTorch) -> None:
    """Callers pass ``torch.device`` as readily as a string — both must work."""
    prepare_cpu_only_worker(cast("Any", _FakeDevice("cpu")))
    assert fake_torch.thread_counts == [1]


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:3", "mps"])
def test_a_non_cpu_worker_keeps_its_thread_pool(fake_torch: _FakeTorch, device: str) -> None:
    """A GPU worker's host-side collation does use several threads; pinning it would be a pessimisation."""
    prepare_cpu_only_worker(device)
    assert fake_torch.thread_counts == []


def test_a_non_cpu_device_does_not_import_torch_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The early return is what keeps this callable from a process that must not touch torch yet.

    Importing torch initialises CUDA bookkeeping in a worker that was placed elsewhere on purpose,
    so the check has to come before the import rather than after it.
    """
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    prepare_cpu_only_worker("cuda:0")
    assert "torch" not in sys.modules


def test_it_does_not_touch_the_cuda_visible_devices_variable(
    fake_torch: _FakeTorch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hiding the GPU is the *parent's* job, around ``Process.start()``, and this is the child's half.

    Pinned because the two halves are easy to confuse, and confusing them fails silently: the worker
    starts, works, and holds a CUDA context nobody asked it for.
    """
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "0,1")
    prepare_cpu_only_worker("cpu")
    assert os.environ[CUDA_VISIBLE_DEVICES] == "0,1"
    assert fake_torch.thread_counts == [1]


def test_the_real_torch_is_pinned_to_one_thread() -> None:
    """The same contract against the real module, where torch is installed."""
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        prepare_cpu_only_worker("cpu")
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)


# ---------------------------------------------------------------------------
# the parent/child boundary itself
# ---------------------------------------------------------------------------


def _report_visible_devices(result: Any) -> None:
    """Child entry point: report the ``CUDA_VISIBLE_DEVICES`` this process was spawned with.

    Module level and torch-free on purpose — ``spawn`` pickles the target by qualified name, and the
    whole point is to read the environment the child inherited before anything touches CUDA.
    """
    import os as child_os

    result.put(child_os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))


def _spawned_visible_devices() -> str:
    """Spawn a child through the current environment and return what it saw."""
    context = multiprocessing.get_context("spawn")
    result: Any = context.Queue()
    process = context.Process(target=_report_visible_devices, args=(result,))
    process.start()
    try:
        return cast("str", result.get(timeout=_SPAWN_TIMEOUT_S))
    finally:
        process.join(timeout=_SPAWN_TIMEOUT_S)
        if process.is_alive():
            process.kill()
            process.join(timeout=_SPAWN_TIMEOUT_S)


def test_a_spawned_child_inherits_the_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract the whole function exists for, proven across a real process boundary.

    A spawn child inherits ``os.environ`` as it stands at ``start()``, which is why masking in the
    parent works at all — and why masking from inside the child does not: torch caches the visible
    device count on its first query, and in a real worker something reaches CUDA before any
    application code runs.
    """
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "0,1")
    with cuda_hidden_from_children():
        assert _spawned_visible_devices() == ""


def test_a_child_spawned_after_the_block_sees_the_devices_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the mask does not outlive its ``with`` block, across a real process boundary.

    The next worker started may be the one that wants the GPU, and on a single-GPU box an escaped
    mask would silently move it to the CPU for the rest of the run.
    """
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "0,1")
    with cuda_hidden_from_children():
        pass
    assert _spawned_visible_devices() == "0,1"


def test_a_spawned_child_inherits_the_blas_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU counterpart, and the reason the pin belongs in the parent.

    numpy's BLAS backend reads these variables at *import*, before any application code runs in the
    child, so a child that sets them itself has already lost.
    """
    monkeypatch.setenv(BLAS_THREAD_VARS[0], "16")
    with blas_threads_pinned(threads=2):
        assert _spawned_blas_threads() == "2"
    assert os.environ[BLAS_THREAD_VARS[0]] == "16"


def _report_blas_threads(result: Any) -> None:
    """Child entry point: report the first BLAS thread-count variable this process inherited."""
    import os as child_os

    result.put(child_os.environ.get("OMP_NUM_THREADS", "<unset>"))


def _spawned_blas_threads() -> str:
    """Spawn a child through the current environment and return the thread count it saw."""
    context = multiprocessing.get_context("spawn")
    result: Any = context.Queue()
    process = context.Process(target=_report_blas_threads, args=(result,))
    process.start()
    try:
        return cast("str", result.get(timeout=_SPAWN_TIMEOUT_S))
    finally:
        process.join(timeout=_SPAWN_TIMEOUT_S)
        if process.is_alive():
            process.kill()
            process.join(timeout=_SPAWN_TIMEOUT_S)
