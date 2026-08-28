"""Process hygiene for a worker: hide the GPU from CPU-only roles, pin their thread pools.

Three functions, and the split between them is the whole content of this module:

* :func:`cuda_hidden_from_children` runs in the **parent**, around ``Process.start()``.
* :func:`blas_threads_pinned` also runs in the **parent**, around ``Process.start()``.
* :func:`prepare_cpu_only_worker` runs in the **child**, as early as it can.

Getting that backwards is the common failure and it fails *silently* — the child starts, works, and
holds resources nobody asked it for. Each docstring says why its half cannot move to the other side.

``torch`` is imported lazily inside :func:`prepare_cpu_only_worker` rather than at module scope, so
the two parent-side context managers — which need nothing but :mod:`os` — stay importable in a
process that has no torch installed and, more importantly, in one that must not touch CUDA yet.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
"""The environment variable a spawned child reads to decide which GPUs exist for it."""

BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""Every environment variable a BLAS backend might read for its thread count, read at import time.

All five, because which one is live depends on how numpy was built, and a child that inherits four
of them still fans out through the fifth.
"""


@contextmanager
def cuda_hidden_from_children(hide: bool = True) -> Iterator[None]:
    """Mask ``CUDA_VISIBLE_DEVICES`` while a CPU-only worker spawns, so it inherits no GPU.

    Pinning a worker's *tensors* to the CPU is not enough to keep it off the card. Any call that
    reaches the CUDA runtime creates that process's CUDA **primary context**, and the calls that do
    it are not the ones you would guard: a profiler synchronising the device, a library probing for
    capabilities, an ``is_available()`` deep inside a dependency. Measured on an A100-40GB:
    ``torch.cuda.is_available()`` costs nothing, ``torch.cuda.synchronize()`` costs **414 MiB of
    VRAM** in a process that owns no GPU tensor at all. Four CPU-only workers held 1.7 GB of a card
    they never computed on; thirty-two would hold ~13 GB.

    **This has to happen in the parent, around ``Process.start()``.** torch caches the visible
    device count on the first ``is_available()`` / ``device_count()`` and exposes no way to clear
    it, and in a spawned child something reaches CUDA before any application code runs — so masking
    from inside the child is already too late (measured: the child still reported
    ``is_available() == True``, and it fails with no error, only the memory). A spawn child inherits
    ``os.environ`` as it stands at ``start()``, so masking here and restoring immediately afterwards
    gives the child a GPU-free environment from its very first instruction and leaves the parent's
    own CUDA context untouched.

    A minimal ``multiprocessing`` spawn test does *not* reproduce the child-side failure — masking
    from inside a bare child works fine there. It appears once the parent has already initialised
    CUDA and pickles real objects across, which is every realistic trainer.

    :param hide: when False the block is a no-op, so a caller can express "hide unless this worker
        actually wants a GPU" without branching around the ``with``.

    Examples
    --------
    >>> import multiprocessing as mp
    >>> from spawnkit import cuda_hidden_from_children
    >>> def cpu_work() -> None: ...
    >>> with cuda_hidden_from_children():          # doctest: +SKIP
    ...     worker = mp.get_context("spawn").Process(target=cpu_work)
    ...     worker.start()
    """
    if not hide:
        yield
        return
    previous = os.environ.get(CUDA_VISIBLE_DEVICES)
    os.environ[CUDA_VISIBLE_DEVICES] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CUDA_VISIBLE_DEVICES, None)
        else:
            os.environ[CUDA_VISIBLE_DEVICES] = previous


@contextmanager
def blas_threads_pinned(threads: int = 1) -> Iterator[None]:
    """Mask the BLAS thread counts while env workers spawn, so each child gets ``threads``.

    The CPU counterpart of :func:`cuda_hidden_from_children`, and it belongs in the parent for the
    same reason: a spawn child inherits ``os.environ`` as it stood at ``start()``, and numpy's BLAS
    backend reads these variables **at import**, before any application code runs in the child.
    :func:`prepare_cpu_only_worker` pins torch's own intra-op pool from inside the child but cannot
    reach numpy's — by then the decision is made.

    :param threads: the per-child thread count to advertise. 1 is right for a pool of N workers on
        one node, where the default (one thread per core, per worker) oversubscribes by N x.
    """
    previous = {name: os.environ.get(name) for name in BLAS_THREAD_VARS}
    os.environ.update({name: str(threads) for name in BLAS_THREAD_VARS})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def prepare_cpu_only_worker(device: str | torch.device = "cpu") -> None:
    """Stop torch fanning a CPU-bound worker's intra-op threads across the whole machine.

    torch defaults to one thread per core, so N workers on a 256-core node each claim the whole box
    and starve each other and the trainer. Measured on a 60-core node with three actor processes:
    leaving this unset put the learner at ~4.5 s per gradient step, and pinning took the same run
    from >150 s to 12 s. A worker whose torch work is small-tensor collation gains nothing from
    intra-op parallelism, so the pin costs it nothing.

    The GPU half of "make this worker cheap" is :func:`cuda_hidden_from_children`, and the numpy
    half is :func:`blas_threads_pinned`; both must be applied by the parent at spawn time — see
    their docstrings for why neither can be done from here.

    :param device: the device this worker was assigned. Non-CPU devices are left alone, so this is
        safe to call unconditionally from a worker body that does not know its own placement.
    """
    if "cpu" not in str(device):
        return
    import torch

    torch.set_num_threads(1)
