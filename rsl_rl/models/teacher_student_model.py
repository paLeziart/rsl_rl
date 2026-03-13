# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from collections import defaultdict
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState, StudentEncoder, TeacherEncoder
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class TeacherStudentModel(nn.Module):
    """TeacherStudent-based neural model.

    This model uses a simple multi-layer perceptron (MLP) to process 1D observation groups. Observations can be
    normalized before being passed to the MLP. The output of the model can be either deterministic or
    stochastic, in which case a distribution module is used to sample the outputs.
    """

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: dict[str, str],
        output_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        encoder_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        obs_normalization: dict[str, bool] = defaultdict(bool),
        distribution_cfg: dict | None = None,
        encoder_layernorm: bool = False,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            latent_dim: Dimension of the latent space (output of the encoders).
            hidden_dims: Hidden dimensions of the actor MLP.
            encoder_dims: Hidden dimensions of the teacher-student encoders.
            activation: Activation function of the MLPs.
            obs_normalization: Whether to normalize the observations before feeding them to the actor-teacher-student.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            encoder_layernorm: Whether to include a LayerNorm layer at the end of the teacher-student encoders.
        """
        super().__init__()

        # Resolve observation groups and dimensions
        self.obs_groups, obs_dim = {}, {}
        for name in ["actor", "teacher", "student"]:
            print(obs_groups)
            self.obs_groups[name], obs_dim[name] = self._get_obs_dim(obs, obs_groups, obs_set[name])

        self.obs_groups["actor"] = (*self.obs_groups["actor"], "latent")
        obs_dim["actor"] += latent_dim

        # Observation normalization
        # Could use a dict but then we would have to overload some nn.Module methods
        self.obs_normalization, self.obs_normalizer = obs_normalization, {}
        self.actor_normalizer = (
            EmpiricalNormalization(obs_dim["actor"]) if obs_normalization["actor"] else torch.nn.Identity()
        )
        self.teacher_normalizer = (
            EmpiricalNormalization(obs_dim["teacher"]) if obs_normalization["teacher"] else torch.nn.Identity()
        )
        self.student_normalizer = (
            EmpiricalNormalization(obs_dim["student"]) if obs_normalization["student"] else torch.nn.Identity()
        )

        # TEMPORARY FIX
        print("\033[91m== Manually adding distribution to actor == \033[0m")
        distribution_cfg = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        # MLPs
        self.actor = MLP(obs_dim["actor"], actor_output_dim, hidden_dims, activation)
        # self.teacher = MLP(obs_dim["teacher"], latent_dim, encoder_dims, activation, last_layernorm=encoder_layernorm)
        # self.student = MLP(obs_dim["student"], latent_dim, encoder_dims, activation, last_layernorm=encoder_layernorm)

        self.teacher = TeacherEncoder(obs_dim["teacher"], encoder_dims, latent_dim)
        self.student = StudentEncoder([12, 12, 76, 76, 76, 12, 12], 4, 64, latent_dim, False)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        alpha: float = 0.0,
        with_latent_norm: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the Teacher-Student model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        # teacher_latent = self.teacher(self.get_observations(obs, "teacher", masks, hidden_state))
        # obs["latent"] = teacher_latent
        # print(alpha)

        #print("Alpha: ", alpha)

        # Get MLP input latent
        if alpha < 1:
            if with_latent_norm:
                teacher_latent, preLN_teacher_norm = self.teacher(self.get_observations(obs, "teacher", masks, hidden_state), with_norm=with_latent_norm)
            else:
                teacher_latent = self.teacher(self.get_observations(obs, "teacher", masks, hidden_state), with_norm=with_latent_norm)
        else:
            preLN_teacher_norm = 0

        if alpha > 0:
            if with_latent_norm:
                student_latent, preLN_student_norm = self.student(self.get_observations(obs, "student", masks, hidden_state), with_norm=with_latent_norm)
            else:
                student_latent = self.student(self.get_observations(obs, "student", masks, hidden_state), with_norm=with_latent_norm)
        else:
            preLN_student_norm = 0

        if alpha == 0.0:
            obs["latent"] = teacher_latent
        elif alpha == 1.0:
            obs["latent"] = student_latent
        else:
            obs["latent"] = alpha * student_latent + (1 - alpha) * teacher_latent

        # teacher_obs = self.get_observations(obs, "teacher", masks, hidden_state)
        # student_obs = self.get_observations(obs, "student", masks, hidden_state)
        # if switch is not None:
        #     # Mixed actor input: do not backprop into student
        #     obs["latent"] = self.teacher(teacher_obs)
        #     obs["latent"][switch[:, 0]] = (self.student(student_obs)[switch[:, 0]]).detach()
        # else:
        #     # Only student for deployment
        #     print("\033[92m= Latent space only computed by student = \033[0m")
        #     obs["latent"] = self.student(student_obs)

        # obs["latent"] = self.teacher(teacher_obs)
        # obs["latent"] *= 0.0

        actor_obs = self.get_observations(obs, "actor", masks, hidden_state)
        # MLP forward pass
        mlp_output = self.actor(actor_obs)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                if not with_latent_norm:
                    return self.distribution.sample()
                else:
                    return self.distribution.sample(), preLN_teacher_norm, preLN_student_norm
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_observations(
        self, obs: TensorDict, name: str, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        # Select and concatenate observations
        obs_list = [obs[obs_group] for obs_group in self.obs_groups[name]]
        latent = torch.cat(obs_list, dim=-1)
        # Normalize observations
        # This would be better with a dict but we'd have to overload some nn.module functions to act on dict
        if name == "actor":
            latent = self.actor_normalizer(latent)
        elif name == "student":
            latent = self.student_normalizer(latent)
        elif name == "teacher":
            latent = self.teacher_normalizer(latent)
        else:
            raise NotImplementedError
        return latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for MLP)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach therecurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchTeacherStudentModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxTeacherStudentModel(self, verbose)

    def update_normalization(self, obs: TensorDict, switch: torch.Tensor | None = None) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if any(self.obs_normalization.values()):
            # Select and concatenate observations
            obs_list = [obs[obs_group] for obs_group in self.obs_groups["teacher"]]
            teacher_obs = torch.cat(obs_list, dim=-1)
            obs_list = [obs[obs_group] for obs_group in self.obs_groups["student"]]
            student_obs = torch.cat(obs_list, dim=-1)

            # Update the normalizer parameters
            if self.obs_normalization["teacher"]:
                self.teacher_normalizer.update(teacher_obs)  # type: ignore
            if self.obs_normalization["student"]:
                self.student_normalizer.update(student_obs)  # type: ignore

            # No need to go further if actor obs are not normalized
            if not self.obs_normalization["actor"]:
                return

            # Normalize observations
            teacher_obs = self.teacher_normalizer(teacher_obs)
            # student_obs = self.student_normalizer(student_obs)
            # if switch is not None:
            #     # Mixed actor input: do not backprop into student
            #     obs["latent"] = self.teacher(teacher_obs)
            #     obs["latent"][switch[:, 0]] = (self.student(student_obs)[switch[:, 0]]).detach()
            # else:
            #     # Only student for deployment
            #     # print("\033[92m= Update actor normalizer with student latent only = \033[0m")
            #     obs["latent"] = self.student(student_obs)

            obs["latent"] = self.teacher(teacher_obs)
            # obs["latent"] *= 0.0

            # Select and concatenate observations (for actor)
            obs_list = [obs[obs_group] for obs_group in self.obs_groups["actor"]]
            mlp_obs = torch.cat(obs_list, dim=-1)
            # Update the normalizer parameters (for actor)
            self.actor_normalizer.update(mlp_obs)  # type: ignore

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchTeacherStudentModel(nn.Module):
    """Exportable MLP model for JIT."""

    def __init__(self, model: TeacherStudentModel) -> None:
        """Create a TorchScript-friendly copy of an MLPModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxTeacherStudentModel(nn.Module):
    """Exportable MLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: TeacherStudentModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(model.actor)
        self.actor_normalizer = copy.deepcopy(model.actor_normalizer)
        self.student = copy.deepcopy(model.student)
        self.student_normalizer = copy.deepcopy(model.student_normalizer)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        # self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        raise NotImplementedError("\033[91mTodo ONNX export for TeacherStudent\033[0m")
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        raise NotImplementedError("\033[91mTodo ONNX export for TeacherStudent\033[0m")
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]

