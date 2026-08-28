"""Deriving each worker's RNG stream, so one seed reproduces a run across process boundaries.

``spawn`` children inherit no RNG state from the parent, so without an explicit per-worker seed every
worker draws fresh OS entropy and two runs with the same seed still diverge — in whatever the
workers sample, and in the order their results arrive. Every worker's stream is derived from one
``SeedSequence([base_seed, offset])``: independent between roles, reproducible from the seed alone.

Three entry points, and picking the wrong one is a silent correctness bug rather than an error:

* :func:`setup_seed` — the **parent**, once, before anything spawns. Also owns the torch determinism
  knobs, because they have to be set before the first kernel runs.
* :func:`seed_worker` — a **spawned child**, once, as early as it can. Reseeds the global
  torch/numpy/random streams *and* returns the worker's own generator.
* :func:`worker_rng` — an **in-process** role sharing an interpreter with the main loop. Returns the
  same stream :func:`seed_worker` would, without touching global state — which it must not do, since
  perturbing the shared streams changes the very run it is embedded in.

``offset`` is the worker's rank within its role. Give each *role* a distinct base offset (the
constants below are a starting set) so a rank-3 producer and a rank-3 evaluator cannot collide.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

EVALUATOR_SEED_OFFSET = 100_000
"""Base offset for evaluator-role workers; add the worker's rank to it."""

COLLECTOR_SEED_OFFSET = 400_000
"""Base offset for collector-role workers; add the worker's rank to it."""


def setup_seed(seed: int | None = None, deterministic: bool = False) -> tuple[int, Generator]:
    """Seed this (parent) process and set the torch determinism knobs. Returns the seed and its stream.

    Call once, before anything spawns. A ``None`` seed is resolved to a random one and **returned**,
    so the caller can record what the run actually used — a run that cannot say its own seed is not
    reproducible even if every other piece of this module is doing its job.

    ``PYTHONHASHSEED`` cannot take effect in the already-running parent interpreter; setting it here
    only reaches the ``spawn`` children, which is where hash-ordered iteration could perturb the
    order results are produced in. Run the parent with ``PYTHONHASHSEED`` preset if that matters.

    The determinism knobs are scoped to ``deterministic=True`` deliberately. Setting them for every
    process makes ordinary runs pay cuDNN's determinism tax and lose autotuning without asking for
    it; ``benchmark=True`` re-tunes per input shape, which suits a fixed batch and not a varying one.

    :param seed: the run's seed, or ``None`` to draw (and return) one.
    :param deterministic: enable torch's deterministic algorithms and single-threaded reductions.
        Costs throughput; buys bit-reproducibility.
    :return: ``(resolved_seed, generator)`` — the generator is the parent's own stream.

    Examples
    --------
    >>> from spawnkit import setup_seed
    >>> seed, rng = setup_seed(7)
    >>> seed
    7
    >>> float(rng.random()) == float(setup_seed(7)[1].random())
    True
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**16))

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch = _torch_or_none()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = not deterministic
            torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            # Intra-op threading reorders float reductions; one thread makes CPU results bit-stable.
            torch.set_num_threads(1)
            # warn_only: a nondeterministic op should not abort a run. The bit-exactness assertion
            # belongs in the caller's test, and this keeps GPU runs usable in the same mode.
            torch.use_deterministic_algorithms(mode=True, warn_only=True)

    return seed, rng


def seed_worker(base_seed: int | None, offset: int) -> Generator:
    """Deterministically seed torch/numpy/random inside a spawned worker process.

    ``spawn`` children do not inherit the parent's RNG state, so without this each worker draws fresh
    OS entropy and same-seed runs still diverge. Deriving the worker seed from
    ``SeedSequence([base_seed, offset])`` gives every worker an independent but reproducible stream.
    A ``None`` base seed keeps the fresh-entropy behaviour, for a run that does not want reproducibility.

    Returns the worker's own :class:`numpy.random.Generator`. **Pass it explicitly** to anything that
    samples: ``np.random.default_rng()`` draws OS entropy and ignores the global ``np.random.seed``
    set here, so a callee that constructs its own generator silently escapes the seeding.

    :param base_seed: the run's seed, from :func:`setup_seed`; ``None`` for fresh entropy.
    :param offset: this worker's rank, plus its role's base offset.
    :return: the worker's generator.
    """
    if base_seed is None:
        return np.random.default_rng()
    seed_sequence = np.random.SeedSequence([int(base_seed), int(offset)])
    derived = int(seed_sequence.generate_state(1)[0])
    random.seed(derived)
    np.random.seed(derived)
    torch = _torch_or_none()
    if torch is not None:
        torch.manual_seed(derived)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(derived)
    return np.random.default_rng(seed_sequence)


def worker_rng(base_seed: int, offset: int) -> Generator:
    """Derive a role's stream *without* touching the global RNG state.

    :func:`seed_worker` is the spawned-child version and also reseeds the global torch/numpy/random
    streams. An in-process caller must not do that: it would perturb the stream of the loop it shares
    an interpreter with, which is the one thing an embedded role has to leave alone.

    :param base_seed: the run's seed, from :func:`setup_seed`.
    :param offset: this role's rank, plus its base offset.
    :return: the role's generator, identical to what :func:`seed_worker` would return.
    """
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(offset)]))


def _torch_or_none() -> Any:
    """Return the :mod:`torch` module, or ``None`` when it is not installed.

    Seeding is useful without torch — numpy and :mod:`random` are seeded either way — so torch is an
    optional extra here rather than an import-time requirement.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch
