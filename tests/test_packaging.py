"""What a cold reader gets: a torch-free base install, typed, with an actionable error at the edge.

These are the properties the *package* promises rather than any one module, and every one of them is
invisible to an ordinary unit test — they are only observable from outside, in an environment that
does not have the source tree on its path or torch in its site-packages.

This suite runs in exactly that environment: the development venv for this repo has **no torch
installed**, deliberately, so the base install's promise is checked on every run rather than
remembered.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

import spawnkit

TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(
    TORCH_INSTALLED,
    reason="the torch-free guarantee is only observable in an environment without torch",
)
def test_the_base_install_needs_no_torch() -> None:
    """Importing the package and using its first two tiers must not require torch.

    This is the claim that makes ``pip install spawnkit`` a two-second install, and it decays
    silently: one convenience import at the top of any tier-1 or tier-2 module would break it, and
    nothing else in the suite would notice.

    It can only be *observed* where torch is absent, so it skips in the CI job that installs torch to
    exercise the service tier. The matrix job that does not install torch is what keeps it honest —
    if that job is ever dropped, this test stops testing anything.
    """
    for module in ("hygiene", "oom", "processes", "supply", "seeding", "lifecycle", "monitor", "run"):
        importlib.import_module(f"spawnkit.{module}")


def test_every_exported_name_resolves() -> None:
    """`__all__` must not name anything the package does not actually export.

    A stale entry is a broken `from spawnkit import *` and a broken documentation build, and neither
    fails anywhere else.
    """
    missing = [name for name in spawnkit.__all__ if not hasattr(spawnkit, name)]
    assert not missing, f"__all__ names attributes that do not exist: {missing}"


def test_exports_are_unique() -> None:
    """A duplicate entry in `__all__` hides a rename that only half happened.

    Ordering is deliberately not asserted here: ruff's RUF022 already enforces it, and it uses an
    isort-style convention (constants, then classes, then functions) rather than plain alphabetical.
    Re-asserting it with a different rule would just make the two fight.
    """
    duplicates = sorted({name for name in spawnkit.__all__ if spawnkit.__all__.count(name) > 1})
    assert not duplicates, f"__all__ lists these more than once: {duplicates}"


def test_the_package_ships_a_py_typed_marker() -> None:
    """Without this file a downstream mypy silently types every spawnkit call as ``Any``.

    Silently is the problem. Nothing errors, nothing warns, and a caller who passes the wrong type to
    a public function gets no complaint from the checker they installed to catch exactly that.
    """
    marker = Path(spawnkit.__file__).parent / "py.typed"
    assert marker.is_file(), "py.typed is missing; downstream type checking would be silently disabled"


@pytest.mark.skipif(TORCH_INSTALLED, reason="the error only fires when torch is absent")
def test_importing_the_service_without_torch_names_the_extra() -> None:
    """The service tier's import error must tell the reader how to fix it.

    A bare ``No module named 'torch'`` sends someone looking for a broken environment. Naming the
    extra turns the same failure into a one-line fix, and this is the single most likely first
    contact anyone has with the boundary between the tiers.
    """
    with pytest.raises(ImportError, match=r"spawnkit\[torch\]"):
        importlib.import_module("spawnkit.service")


def test_the_version_is_a_single_source_of_truth() -> None:
    """``__version__`` and the installed distribution metadata must agree.

    They are written in two files, so they can disagree — and when they do, the release is tagged as
    one version and reports itself as another.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("spawnkit")
    except PackageNotFoundError:  # pragma: no cover - only when running from a bare source tree
        pytest.skip("spawnkit is not installed as a distribution in this environment")
    assert installed == spawnkit.__version__
