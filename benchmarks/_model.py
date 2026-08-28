"""The reference model the benchmarks serve: small, row-independent, and shaped like the real thing.

Deliberately not a toy. The numbers this repo publishes are about *launch and transport overhead*,
and that overhead only shows up honestly against a model with the properties the service assumes:

* **Small tensors, many layers.** The interesting regime is one where a forward issues far more
  kernels than it does arithmetic — that is what makes the call cost flat in batch size and what
  CUDA-graph replay collapses. A single huge matmul would be GPU-bound and would show nothing.
* **Row-independent.** No cross-row operation anywhere: MLPs, elementwise activations, and reductions
  over the feature axis only. This is the precondition CUDA-graph padding rests on, so the reference
  model must satisfy it or the benchmark measures an unsound configuration.
* **Several output heads of different widths.** The scatter step's cost depends on how much comes
  back, not just on how much went in, and a single-output model would hide that.

Nothing here is trained or meaningful; only its shape and cost matter.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


class Outputs(NamedTuple):
    """One forward's heads.

    :param hidden_state: the next latent, same width as the input latent.
    :param reward: a categorical distribution over a small support.
    :param policy: a distribution over the action space.
    :param value: a categorical distribution over a value support.
    """

    hidden_state: torch.Tensor
    reward: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor


class ReferenceNet(nn.Module):
    """A dynamics-plus-prediction network: ``(hidden, action) -> (hidden, reward, policy, value)``.

    :param hidden_dim: latent width.
    :param num_actions: action-space size; also the one-hot width fed to the trunk.
    :param support: width of the reward and value distributions.
    :param depth: residual blocks in the trunk. More blocks means more kernel launches at the same
        arithmetic cost, which is exactly the regime being measured.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_actions: int = 60,
        support: int = 51,
        depth: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.support = support
        self.trunk_in = nn.Linear(hidden_dim + num_actions, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(depth)
        )
        self.reward_head = nn.Linear(hidden_dim, support)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, support)

    def step(self, hidden: torch.Tensor, action: torch.Tensor) -> Outputs:
        """One recurrent step. ``action`` may be ``[n]`` or ``[n, 1]``; both are real client layouts."""
        one_hot = torch.nn.functional.one_hot(
            action.reshape(-1).to(torch.long), num_classes=self.num_actions,
        ).to(hidden.dtype)
        x = self.trunk_in(torch.cat([hidden, one_hot], dim=1))
        for block in self.blocks:
            x = torch.nn.functional.silu(x + block(x))
        # Min-max over the feature axis, not the batch axis: normalising across rows would couple
        # them and quietly invalidate CUDA-graph padding.
        lo = x.amin(dim=1, keepdim=True)
        hi = x.amax(dim=1, keepdim=True)
        latent = (x - lo) / (hi - lo + 1e-5)
        return Outputs(
            hidden_state=latent,
            reward=torch.softmax(self.reward_head(latent), dim=1),
            policy=torch.softmax(self.policy_head(latent), dim=1),
            value=torch.softmax(self.value_head(latent), dim=1),
        )

    def forward(self, hidden: torch.Tensor, action: torch.Tensor) -> Outputs:
        """Alias for :meth:`step`, so the module is callable in the usual way."""
        return self.step(hidden, action)


def build_reference_net(hidden_dim: int = 128, num_actions: int = 60, depth: int = 6) -> ReferenceNet:
    """Return an eval-mode :class:`ReferenceNet` with its weights in shared memory.

    Shared memory because the service benchmark hands the same module to a spawned service process,
    which is how the real deployment works: the trainer owns the module and updates it in place.

    :param hidden_dim: latent width.
    :param num_actions: action-space size.
    :param depth: residual blocks in the trunk.
    :return: the module, in ``eval()`` mode and shared.
    """
    net = ReferenceNet(hidden_dim=hidden_dim, num_actions=num_actions, depth=depth)
    net.eval()
    net.share_memory()
    return net
