# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from .ppo import PPO


class PPOCaT(PPO):
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).

    Support the Constraints as Terminations (CaT) approach with minimal modifications:
    - store "cstr_probs" in transitions during env step processing.
    - apply "cstr_probs" during the computation of returns and advantages.

    This implementation assumes that the rewards themselves are already scaled down
    depending on constraint violations during sample collection.
    """

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Record the constraint probability factor that affects the discounted
        # sum of rewards depending on constraint violations.
        self.transition.cstr_dones = extras["cstr_probs"]

        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            next_is_not_terminal *= 1.0 - st.cstr_dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)
