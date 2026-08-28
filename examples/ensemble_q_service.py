"""K CPU workers acting in a Gymnasium env, querying one GPU ensemble-Q network through the service.

This example exists to show the service is not tied to any particular algorithm. There is no tree
search here, no planning and no learning — just the shape that recurs whenever many cheap CPU actors
need one expensive model: each worker steps its own environment, asks the service which action to
take, and repeats.

It also demonstrates the pieces in the order a real launcher uses them:

1. build the network in the parent and put it in shared memory;
2. declare the remote call as an :class:`~spawnkit.service.rpc.Rpc`;
3. start the service on the device;
4. spawn CPU-only workers inside :func:`~spawnkit.hygiene.cuda_hidden_from_children`, so none of them
   opens a CUDA context;
5. supervise them with a :class:`~spawnkit.monitor.WorkerMonitor` that treats the service as critical
   and the workers as producers;
6. shut everything down through one shared grace window.

Run::

    pip install spawnkit[torch] gymnasium
    python examples/ensemble_q_service.py --workers 4 --steps 200
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import threading
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from spawnkit import (
    WorkerMonitor,
    WorkerSpec,
    cuda_hidden_from_children,
    prepare_cpu_only_worker,
    seed_worker,
    shutdown_processes,
)
from spawnkit.service import BatchedInferenceService, ModuleReplica, ServiceClient, TensorRpc

ENV_ID = "CartPole-v1"
OBS_DIM = 4
NUM_ACTIONS = 2
ENSEMBLE = 5
ACT = "act"


class QOut(NamedTuple):
    """The forward's output fields, which the RPC reads by name.

    Any object with the declared attributes works — a NamedTuple, a dataclass, or your own type. The
    RPC never sees this class, only the names it was configured with.
    """

    q_mean: torch.Tensor
    q_std: torch.Tensor


class EnsembleQ(nn.Module):
    """An ensemble of Q heads over a shared trunk; the mean over heads picks the action.

    An ensemble rather than a single head because it makes the example's point: the expensive thing
    is worth centralising. Row-independent throughout, so it is safe to serve from a CUDA graph.
    """

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = NUM_ACTIONS, heads: int = ENSEMBLE) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(128, num_actions) for _ in range(heads))

    def q_values(self, obs: torch.Tensor) -> QOut:
        """Mean Q-values over the ensemble, plus the disagreement between heads."""
        latent = self.trunk(obs)
        stacked = torch.stack([head(latent) for head in self.heads])  # [heads, batch, actions]
        return QOut(q_mean=stacked.mean(dim=0), q_std=stacked.std(dim=0))


def env_worker(
    rank: int,
    request_queue: Any,
    response_queue: Any,
    rpc: TensorRpc,
    stop_event: Any,
    steps: int,
    seed: int,
    result_queue: Any,
) -> None:
    """Step one environment for ``steps``, choosing each action through the service.

    Module-level so it is picklable under ``spawn``. Pins its own thread pool: N workers each fanning
    torch across every core would spend the machine on collation.
    """
    prepare_cpu_only_worker("cpu")
    rng = seed_worker(seed, rank)
    client = ServiceClient(rank, request_queue, response_queue, [rpc], stop_event)

    env = gym.make(ENV_ID)
    obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    episode_return, returns = 0.0, []

    for _ in range(steps):
        if stop_event.is_set():
            break
        out = client.call(ACT, (np.asarray(obs, dtype=np.float32).reshape(1, OBS_DIM),))
        action = int(out["q_mean"].argmax())
        obs, reward, terminated, truncated, _info = env.step(action)
        episode_return += float(reward)
        if terminated or truncated:
            returns.append(episode_return)
            episode_return = 0.0
            obs, _info = env.reset()

    env.close()
    result_queue.put((rank, returns))


def main() -> None:
    """Run the example end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # spawnkit logs through the stdlib under the "spawnkit" logger and attaches only a NullHandler,
    # so an application sees nothing until it opts in. This is that opt-in.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ctx = mp.get_context("spawn")
    net = EnsembleQ()
    net.eval()
    net.share_memory()

    rpc = TensorRpc(ACT, method="q_values", input_axes=(0,), output_fields=("q_mean", "q_std"))
    replica = ModuleReplica(net)

    request_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(args.workers)]
    result_queue = ctx.Queue()
    stop_event = ctx.Event()

    service = BatchedInferenceService(
        build_fn=replica.build,
        rpcs=[rpc],
        request_queue=request_queue,
        response_queues=response_queues,
        stop_event=stop_event,
        device=args.device,
        sync_fn=replica.sync,
        sync_interval=100,
        max_batch=args.workers,
        graph_rpcs=(ACT,) if args.device.startswith("cuda") else (),
    )
    service_process = service.start(ctx)

    # The workers are CPU-only. Without this mask each opens a CUDA primary context it never uses.
    workers = []
    with cuda_hidden_from_children():
        for rank in range(args.workers):
            worker = ctx.Process(
                target=env_worker,
                args=(
                    rank, request_queue, response_queues[rank], rpc, stop_event,
                    args.steps, args.seed, result_queue,
                ),
                daemon=True,
            )
            worker.start()
            workers.append(worker)

    # Watch from a thread so the main thread can drain results; the monitor ends when the producers do.
    monitor = WorkerMonitor(
        [
            WorkerSpec("service", service_process, critical=True),
            *[WorkerSpec(f"worker-{i}", w, critical=True, producer=True) for i, w in enumerate(workers)],
        ],
        stop_event,
        check_interval=0.5,
    )
    watcher = threading.Thread(target=monitor.watch, daemon=True)
    watcher.start()

    try:
        all_returns: list[float] = []
        for _ in range(args.workers):
            _rank, returns = result_queue.get(timeout=300)
            all_returns.extend(returns)
    finally:
        stop_event.set()
        request_queue.put(None)
        shutdown_processes(
            [(f"worker_{i}", w) for i, w in enumerate(workers)] + [("service", service_process)],
        )
        watcher.join(timeout=5)

    episodes = len(all_returns)
    mean_return = sum(all_returns) / episodes if episodes else float("nan")
    print(
        f"\n{args.workers} workers x {args.steps} steps through one {args.device} service: "
        f"{episodes} episodes, mean return {mean_return:.1f}",
    )
    print("(the network is untrained - the point is the plumbing, not the score)")


if __name__ == "__main__":
    main()
