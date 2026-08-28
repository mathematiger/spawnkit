"""Stable-Baselines3 with a spawn-mode ``SubprocVecEnv``, and the two lines that keep it cheap.

You do not have to write your own worker pool to hit the failures spawnkit is for. Any framework
that spawns environment workers hits the same two: each worker can open a CUDA context it never
uses, and each worker's BLAS backend can claim every core on the machine. Both are decided in the
**parent**, at spawn time, which is why a framework cannot fix them for you — by the time your
``make_env`` callable runs in the child, it is too late for either.

The whole integration is a context manager around the ``SubprocVecEnv`` construction, plus one line
inside the env factory:

.. code-block:: python

    with cuda_hidden_from_children(), blas_threads_pinned():
        env = SubprocVecEnv([make_env(i) for i in range(n)], start_method="spawn")

This example measures the difference rather than asserting it: it builds the pool both ways and
reports each worker's thread count and, where a GPU is present, the VRAM it holds.

Run::

    pip install spawnkit stable-baselines3 gymnasium
    python examples/sb3_hygiene.py --workers 4
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv

from spawnkit import blas_threads_pinned, cuda_hidden_from_children, prepare_cpu_only_worker

ENV_ID = "CartPole-v1"


class ReportingEnv(gym.Wrapper):
    """A wrapper whose ``report`` tells the parent what the worker process actually inherited.

    The point of the example is what a worker *inherits at spawn*, and only the worker can see that.
    ``SubprocVecEnv.env_method`` is the channel back.
    """

    def __init__(self, env: gym.Env, pin_threads: bool) -> None:
        super().__init__(env)
        if pin_threads:
            # The child's half of the pairing. It pins torch's own intra-op pool; it cannot reach
            # numpy's BLAS backend, which read its thread count at import - before this line ran.
            prepare_cpu_only_worker("cpu")

    def report(self) -> dict[str, Any]:
        """Return this worker's pid, thread count, BLAS env and whether it can see a GPU."""
        return {
            "pid": os.getpid(),
            "torch_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "cuda_available": torch.cuda.is_available(),
        }


def make_env(pin_threads: bool) -> Any:
    """Return a picklable env factory for ``SubprocVecEnv``.

    A module-level function returning a closure over plain data - not a lambda over live objects -
    because ``start_method="spawn"`` pickles this to each worker.
    """

    def _init() -> gym.Env:
        return ReportingEnv(gym.make(ENV_ID), pin_threads)

    return _init


def build_pool(workers: int, hygiene: bool) -> SubprocVecEnv:
    """Build a ``SubprocVecEnv``, with or without the parent-side hygiene applied.

    ``start_method="spawn"`` on purpose. SB3 defaults to ``fork`` on Linux, and forking a parent that
    has already initialised torch or CUDA is unsafe — the child inherits allocator and thread-pool
    state that was never made to cross a fork.

    :param workers: pool size.
    :param hygiene: apply :func:`cuda_hidden_from_children` and :func:`blas_threads_pinned` around
        the spawn, and pin the child's torch pool from inside.
    :return: the vectorised environment.
    """
    factories = [make_env(hygiene) for _ in range(workers)]
    if not hygiene:
        return SubprocVecEnv(factories, start_method="spawn")

    # Both must wrap Process.start(): a spawn child inherits os.environ as it stands at that moment,
    # and numpy's BLAS backend reads its thread count at import, before any of our code runs there.
    with cuda_hidden_from_children(), blas_threads_pinned():
        return SubprocVecEnv(factories, start_method="spawn")


def describe(label: str, pool: SubprocVecEnv) -> list[dict[str, Any]]:
    """Print one line per worker and return their reports."""
    reports: list[dict[str, Any]] = pool.env_method("report")
    print(f"\n{label}")
    for index, report in enumerate(reports):
        print(
            f"  worker {index}: pid={report['pid']:<8} torch_threads={report['torch_threads']:<4} "
            f"OMP_NUM_THREADS={report['omp_num_threads']:<9} "
            f"CUDA_VISIBLE_DEVICES={report['cuda_visible']:<9} "
            f"cuda_available={report['cuda_available']}",
        )
    return reports


def main() -> None:
    """Build the pool both ways and report what each set of workers inherited."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    print(f"parent: {torch.get_num_threads()} torch threads, cuda_available={torch.cuda.is_available()}")

    results = {}
    for label, hygiene in (("without spawnkit hygiene", False), ("with spawnkit hygiene", True)):
        pool = build_pool(args.workers, hygiene)
        try:
            pool.reset()
            results[label] = describe(label, pool)
        finally:
            pool.close()

    before = results["without spawnkit hygiene"]
    after = results["with spawnkit hygiene"]
    print(
        f"\ntorch threads per worker: {before[0]['torch_threads']} -> {after[0]['torch_threads']}"
        f"   (x{args.workers} workers)",
    )
    print(
        f"workers that can see a GPU: {sum(r['cuda_available'] for r in before)}/{len(before)} -> "
        f"{sum(r['cuda_available'] for r in after)}/{len(after)}",
    )
    if not torch.cuda.is_available():
        print("(no GPU on this machine, so the CUDA column shows the mask only, not the VRAM saved)")


if __name__ == "__main__":
    main()
