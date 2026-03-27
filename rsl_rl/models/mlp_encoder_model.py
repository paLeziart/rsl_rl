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

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState, StudentEncoder
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class MLPEncoderModel(nn.Module):
    """MLP-based neural model with an encoder for privileged information.

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
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        encoder_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        obs_normalization: dict[str, bool] = defaultdict(bool),
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the actor MLP.
            encoder_dims: Hidden dimensions of the teacher-student encoders.
            activation: Activation function of the MLPs.
            obs_normalization: Whether to normalize the observations before feeding them to the actor-teacher-student.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
        """
        super().__init__()

        # Resolve observation groups and dimensions
        self.obs_groups, obs_dim = {}, {}
        for name in ["actor", "encoder", "privileged"]:
            print(obs_groups)
            self.obs_groups[name], obs_dim[name] = self._get_obs_dim(obs, obs_groups, obs_set[name])

        _, latent_dim = self._get_obs_dim(obs, obs_groups, "privileged")

        self.obs_groups["actor"] = (*self.obs_groups["actor"], "latent")
        obs_dim["actor"] += latent_dim

        # Observation normalization
        # Could use a dict but then we would have to overload some nn.Module methods
        self.obs_normalization, self.obs_normalizer = obs_normalization, {}
        self.actor_normalizer = (
            EmpiricalNormalization(obs_dim["actor"]) if obs_normalization["actor"] else torch.nn.Identity()
        )
        self.encoder_normalizer = (
            EmpiricalNormalization(obs_dim["encoder"]) if obs_normalization["encoder"] else torch.nn.Identity()
        )
        self.privileged_normalizer = EmpiricalNormalization(obs_dim["privileged"])

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        # MLP
        self.actor = MLP(obs_dim["actor"], actor_output_dim, hidden_dims, activation)
        self.encoder = StudentEncoder([30, 30, 190, 190, 190, 30, 30], 10, 128, latent_dim, False)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Encoded latent
        # print("= PreNorm")

        # print("===")
        # print(obs["actor_history"].shape)
        # print(obs.keys())
        # print(obs["actor_history"][0, -34:])

        obs["latent"] = self.encoder(self.get_observations(obs, "encoder", masks, hidden_state))

        # obs["latent"] = self.get_observations(obs, "privileged", masks, hidden_state)

        # print("LATENT")
        # print(obs["latent"][0])

        obs["latent_denormed"] = self.privileged_normalizer.inverse(obs["latent"])

        print("Estim", obs["latent_denormed"][0])

        obs_list = [obs[obs_group] for obs_group in self.obs_groups["privileged"]]
        privileged = torch.cat(obs_list, dim=-1)
        print("Privi", privileged[0])
        

        # MLP forward pass
        mlp_output = self.actor((self.get_observations(obs, "actor", masks, hidden_state)).detach())
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
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
        elif name == "encoder":
            latent = self.encoder_normalizer(latent)
        elif name == "privileged":
            latent = self.privileged_normalizer(latent)
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
        return _TorchMLPEncoderModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPEncoderModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if any(self.obs_normalization.values()):

            obs_list = [obs[obs_group] for obs_group in self.obs_groups["privileged"]]
            priv_obs = torch.cat(obs_list, dim=-1)
            self.privileged_normalizer.update(priv_obs)

            # Select and concatenate observations
            obs_list = [obs[obs_group] for obs_group in self.obs_groups["encoder"]]
            encoder_obs = torch.cat(obs_list, dim=-1)

            # Update the normalizer parameters
            if self.obs_normalization["encoder"]:
                self.encoder_normalizer.update(encoder_obs)  # type: ignore

            # No need to go further if actor obs are not normalized
            if not self.obs_normalization["actor"]:
                return

            # Normalize encoder observations
            obs["latent"] = self.encoder(self.encoder_normalizer(encoder_obs))

            # Select and concatenate observations (for actor)
            obs_list = [obs[obs_group] for obs_group in self.obs_groups["actor"]]
            actor_obs = torch.cat(obs_list, dim=-1)
            # Update the normalizer parameters (for actor)
            self.actor_normalizer.update(actor_obs)  # type: ignore

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

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim


class _TorchMLPEncoderModel(nn.Module):
    """Exportable MLP model for JIT."""

    def __init__(self, model: MLPEncoderModel) -> None:
        """Create a TorchScript-friendly copy of an MLPEncoderModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.actor = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        out = self.actor(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxMLPEncoderModel(nn.Module):
    """Exportable MLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: MLPEncoderModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPEncoderModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.actor = copy.deepcopy(model.actor)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        raise NotImplementedError("\033[91mTodo ONNX export for MLPEncoder\033[0m")
        x = self.obs_normalizer(x)
        out = self.actor(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        raise NotImplementedError("\033[91mTodo ONNX export for MLPEncoder\033[0m")
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
