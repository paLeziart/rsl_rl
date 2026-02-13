# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


"""
# normalize before MSE
z_s = F.normalize(z_s, dim=1)
z_t = F.normalize(z_t, dim=1)

loss = F.mse_loss(z_s, z_t)

# Try out
loss = MSE(norm(zs), norm(zt)) + k * MSE(norm(hs), norm(ht)) with k = 0.1

"""
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticTeacherStudent(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_obs_normalization: bool = False,
        student_obs_normalization: bool = False,
        teacher_hidden_dims: tuple[int] | list[int] = [128, 64, 32],
        student_hidden_dims: tuple[int] | list[int] = [128, 64, 32],
        latent_dim: int = 32,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_proprio_obs = 0
        for obs_group in obs_groups["proprio_unoised"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_proprio_obs += obs[obs_group].shape[-1]

        num_privileged_obs = 0
        for obs_group in obs_groups["privileged"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_privileged_obs += obs[obs_group].shape[-1]

        num_proprio_hist_obs = 0
        for obs_group in obs_groups["proprio_history"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_proprio_hist_obs += obs[obs_group].shape[-1]

        num_teacher_obs = num_proprio_obs + num_privileged_obs
        num_student_obs = num_proprio_hist_obs
        num_actor_obs = num_proprio_obs + latent_dim
        num_critic_obs = num_proprio_obs + num_privileged_obs

        # Teacher encoder
        self.teacher = MLP(num_teacher_obs, latent_dim, teacher_hidden_dims, activation)
        print(f"Teacher encoder MLP: {self.teacher}")

        # Teacher observation normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # Student encoder
        self.student = MLP(num_student_obs, latent_dim, student_hidden_dims, activation)
        print(f"Student encoder MLP: {self.student}")

        # Student observation normalization
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # Actor
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, switch: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_actor_obs(obs, switch)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, switch: torch.Tensor | None = None) -> torch.Tensor:
        obs = self.get_actor_obs(obs, switch)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs_proprio = self.get_proprio_obs(obs)
        obs_privileged = self.get_privileged_obs(obs)
        obs = self.critic_obs_normalizer(torch.cat([obs_proprio, obs_privileged], dim=-1))
        return self.critic(obs)

    def get_proprio_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["proprio_unoised"]]
        return torch.cat(obs_list, dim=-1)

    def get_proprio_obs_noised(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_proprio_hist_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["proprio_history"]]
        return torch.cat(obs_list, dim=-1)

    def get_privileged_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["privileged"]]
        return torch.cat(obs_list, dim=-1)

    def get_actor_obs(self, obs: TensorDict, switch: torch.Tensor | None = None) -> torch.Tensor:
        obs_proprio = self.get_proprio_obs(obs)
        obs_proprio_noised = self.get_proprio_obs_noised(obs)
        obs_proprio_hist = self.get_proprio_hist_obs(obs)
        obs_privileged = self.get_privileged_obs(obs)

        # TODO: Test switch split in obs_student/obs_teacher
        obs_student = self.student_obs_normalizer(obs_proprio_hist)
        obs_teacher = self.teacher_obs_normalizer(torch.cat([obs_proprio, obs_privileged], dim=-1))

        latent_student = self.student(obs_student)  # torch.nn.functional.normalize(self.student(obs_student), dim=-1)
        latent_teacher = self.teacher(obs_teacher)  # torch.nn.functional.normalize(self.teacher(obs_teacher), dim=-1)

        if switch is not None:
            # Mixed actor input: do not backprop into student
            obs = torch.cat([obs_proprio_noised, latent_teacher], dim=-1)
            obs[switch[:, 0]] = (torch.cat([obs_proprio_noised, latent_student], dim=-1)[switch[:, 0]]).detach()
        else:
            # obs = torch.cat([obs_proprio_noised, latent_teacher], dim=-1).detach()
            obs = torch.cat([obs_proprio_noised, latent_student], dim=-1).detach()
        return obs

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_latent_student(self, obs: TensorDict) -> torch.Tensor:
        obs_proprio_hist = self.get_proprio_hist_obs(obs)
        obs_student = self.student_obs_normalizer(obs_proprio_hist)
        return self.student(obs_student)
        # return torch.nn.functional.normalize(self.student(obs_student), dim=-1)

    def get_latent_teacher(self, obs: TensorDict) -> torch.Tensor:
        obs_proprio = self.get_proprio_obs(obs)
        obs_privileged = self.get_privileged_obs(obs)
        obs_teacher = self.teacher_obs_normalizer(torch.cat([obs_proprio, obs_privileged], dim=-1))
        return self.teacher(obs_teacher)
        # return torch.nn.functional.normalize(self.teacher(obs_teacher), dim=-1)

    def update_normalization(self, obs: TensorDict, switch: torch.Tensor) -> None:
        if self.teacher_obs_normalization:
            obs_proprio = self.get_proprio_obs(obs)
            obs_privileged = self.get_privileged_obs(obs)
            self.teacher_obs_normalizer.update(torch.cat([obs_proprio, obs_privileged], dim=-1))
        if self.student_obs_normalization:
            obs_proprio_hist = self.get_proprio_hist_obs(obs)
            self.student_obs_normalizer.update(obs_proprio_hist)
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs, switch)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            obs_proprio = self.get_proprio_obs(obs)
            obs_privileged = self.get_privileged_obs(obs)
            self.critic_obs_normalizer.update(torch.cat([obs_proprio, obs_privileged], dim=-1))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
