"""Unit tests for :mod:`spawnkit.seeding`.

What is pinned here is the *contract between the three entry points*, because picking the wrong one
is a silent correctness bug rather than an error:

* every role must get a **different** stream from the same base seed, or two workers explore
  identically and the run collects one trajectory N times over;
* :func:`worker_rng` must reproduce :func:`seed_worker`'s stream **without** touching the global RNG
  state, since it runs in the same interpreter as the main loop it must leave bit-identical;
* :func:`setup_seed` must *return* the seed it resolved, since a run that cannot say its own seed is
  not reproducible however careful the rest of the module is.

torch is optional here, so the tests that need it are skipped rather than required.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from spawnkit.seeding import (
    COLLECTOR_SEED_OFFSET,
    EVALUATOR_SEED_OFFSET,
    seed_worker,
    setup_seed,
    worker_rng,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_SEEDING_ENV_VARS = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")


@pytest.fixture(autouse=True)
def _isolate_global_state() -> Iterator[None]:
    """Undo the process-global effects these functions exist to have.

    ``setup_seed`` and ``seed_worker`` reseed the interpreter's shared generators and set two
    environment variables; leaking either would change how every later test in the session computes.
    """
    python_state = random.getstate()
    numpy_state: Any = np.random.get_state()
    environment = {name: os.environ.get(name) for name in _SEEDING_ENV_VARS}
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _draw(rng: np.random.Generator) -> list[float]:
    """Take a short, comparable sample from ``rng``."""
    return [float(value) for value in rng.random(4)]


# ---------------------------------------------------------------------------
# seed_worker
# ---------------------------------------------------------------------------


def test_the_same_rank_reproduces_its_stream() -> None:
    """The property every determinism claim downstream rests on."""
    assert _draw(seed_worker(1234, 3)) == _draw(seed_worker(1234, 3))


def test_different_ranks_get_different_streams() -> None:
    """Otherwise every worker explores identically and N workers collect one trajectory N times."""
    streams = [_draw(seed_worker(1234, rank)) for rank in range(4)]
    assert len({tuple(stream) for stream in streams}) == 4


def test_different_base_seeds_get_different_streams() -> None:
    """A seed sweep whose seeds collapsed onto one stream would report four runs of the same thing."""
    assert _draw(seed_worker(1, 0)) != _draw(seed_worker(2, 0))


def test_no_base_seed_means_fresh_entropy() -> None:
    """``None`` keeps the un-seeded behaviour rather than silently pinning every run to one seed."""
    assert _draw(seed_worker(None, 0)) != _draw(seed_worker(None, 0))


def test_the_role_offsets_do_not_collide_with_any_worker_rank() -> None:
    """The offsets are why an evaluator cannot accidentally replay worker 100000's stream.

    Both are far above any plausible worker count, and distinct from each other.
    """
    assert EVALUATOR_SEED_OFFSET != COLLECTOR_SEED_OFFSET
    assert min(EVALUATOR_SEED_OFFSET, COLLECTOR_SEED_OFFSET) > 10_000
    roles = [tuple(_draw(seed_worker(7, offset))) for offset in (0, 1, EVALUATOR_SEED_OFFSET, COLLECTOR_SEED_OFFSET)]
    assert len(set(roles)) == 4


def test_it_reseeds_the_global_generators_too() -> None:
    """Reseed the global generators, not just the one returned.

    A spawned child inherits no RNG state, so anything reaching for ``random`` or the legacy
    ``np.random`` functions would otherwise draw OS entropy and break same-seed reproducibility.
    """
    seed_worker(99, 0)
    first = (random.random(), float(np.random.random()))
    seed_worker(99, 0)
    assert (random.random(), float(np.random.random())) == first


def test_it_reseeds_torch_as_well() -> None:
    """torch is one more global source of drift in a worker that has it installed."""
    torch = pytest.importorskip("torch")
    seed_worker(99, 0)
    first = float(torch.rand(1))
    seed_worker(99, 0)
    assert float(torch.rand(1)) == first


# ---------------------------------------------------------------------------
# worker_rng
# ---------------------------------------------------------------------------


def test_worker_rng_reproduces_the_stream_seed_worker_hands_out() -> None:
    """An in-process role must draw the same numbers its spawned counterpart would.

    If these two drifted, a serial replay of an async run would stop being faithful and every
    determinism check built on it would be testing the wrong thing.
    """
    assert _draw(worker_rng(4321, 5)) == _draw(seed_worker(4321, 5))


def test_worker_rng_leaves_the_global_state_untouched() -> None:
    """Leave the global generators alone — the whole reason it exists next to ``seed_worker``.

    It runs in-process alongside the main loop, so reseeding the globals would perturb the very
    stream an embedded role is supposed to leave bit-identical.
    """
    random.seed(11)
    np.random.seed(11)
    expected = (random.random(), float(np.random.random()))

    random.seed(11)
    np.random.seed(11)
    worker_rng(4321, 5)

    assert (random.random(), float(np.random.random())) == expected


def test_worker_rng_separates_ranks_the_same_way() -> None:
    """The offsets have to mean the same thing on both sides of the process boundary."""
    streams = [tuple(_draw(worker_rng(7, offset))) for offset in (0, 1, EVALUATOR_SEED_OFFSET)]
    assert len(set(streams)) == 3


# ---------------------------------------------------------------------------
# setup_seed
# ---------------------------------------------------------------------------


def test_setup_seed_resolves_and_returns_a_missing_seed() -> None:
    """Resolve a missing seed and hand it back, so the caller can record what the run actually used.

    Workers derive their streams from this value; leaving it unresolved makes the run
    unreproducible *and* unrecorded, since what was used is never written down anywhere.
    """
    seed, _ = setup_seed(None)
    assert isinstance(seed, int)


def test_setup_seed_keeps_an_explicit_seed() -> None:
    """A sweep passes seed=1,2,3 and must get exactly those back."""
    seed, _ = setup_seed(17)
    assert seed == 17


def test_setup_seed_seeds_both_global_generators() -> None:
    """python and numpy each drive part of a run; missing one leaves a live source of drift."""

    def draw_after_seeding() -> tuple[float, float]:
        setup_seed(17)
        return random.random(), float(np.random.random())

    assert draw_after_seeding() == draw_after_seeding()


def test_setup_seed_seeds_torch_too() -> None:
    """The third global generator, where it is installed."""
    torch = pytest.importorskip("torch")

    def draw_after_seeding() -> float:
        setup_seed(17)
        return float(torch.rand(1))

    assert draw_after_seeding() == draw_after_seeding()


def test_setup_seed_returns_the_parent_generator_for_that_seed() -> None:
    """The returned generator is handed on as the run's random state; it must follow from the seed."""
    _, rng = setup_seed(23)
    assert _draw(rng) == _draw(np.random.default_rng(23))


def test_a_resolved_seed_reproduces_its_own_stream() -> None:
    """Recording the seed is only worth anything if replaying it gives the same stream back."""
    seed, rng = setup_seed(None)
    assert _draw(rng) == _draw(setup_seed(seed)[1])


def test_setup_seed_exports_the_hash_seed_to_children() -> None:
    """It cannot take effect in the running parent, but ``spawn`` children read it at startup.

    Hash-ordered iteration in a child perturbs the order its results are produced in, which is
    exactly the kind of drift the rest of this module exists to remove.
    """
    setup_seed(31)
    assert os.environ["PYTHONHASHSEED"] == "31"


def test_setup_seed_does_not_override_an_existing_workspace_config() -> None:
    """``setdefault``: an operator who chose a cuBLAS workspace keeps it."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    setup_seed(31)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_deterministic_mode_works_without_torch_installed() -> None:
    """torch is an optional extra, so both flags must be meaningful on a numpy-only install."""
    seed, rng = setup_seed(5, deterministic=True)
    assert seed == 5
    assert _draw(rng) == _draw(np.random.default_rng(5))


def test_deterministic_mode_pins_the_torch_determinism_knobs() -> None:
    """Pin the torch determinism knobs under ``deterministic``, and only under it.

    cuDNN autotuning reorders float reductions, so a bit-exact run needs benchmark off and
    determinism on — while an ordinary run must not pay a determinism tax it never asked for, which
    is what the second half checks.
    """
    torch = pytest.importorskip("torch")
    if not torch.backends.cudnn.enabled:
        pytest.skip("a build without cuDNN has no knobs to pin")

    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.get_num_threads(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )

    def knobs(*, deterministic: bool) -> tuple[bool, bool, bool]:
        """Return (cudnn.deterministic, cudnn.benchmark, use_deterministic_algorithms) after seeding.

        Read into locals rather than asserted in place: the torch backend attributes are descriptors
        that a type checker narrows to a literal, which makes every assertion after the first
        unreachable.
        """
        setup_seed(1, deterministic=deterministic)
        return (
            bool(torch.backends.cudnn.deterministic),
            bool(torch.backends.cudnn.benchmark),
            bool(torch.are_deterministic_algorithms_enabled()),
        )

    try:
        assert knobs(deterministic=True) == (True, False, True)
        # Only the cuDNN pair is scoped to the flag; use_deterministic_algorithms is one-way in
        # torch, which is why the restore below undoes it rather than the second call.
        assert knobs(deterministic=False)[:2] == (False, True)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=True)
        torch.set_num_threads(previous[1])
        torch.backends.cudnn.benchmark = previous[2]
        torch.backends.cudnn.deterministic = previous[3]
