# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class TeacherEncoder(nn.Module):
    """Teacher encoder network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        latent_dim: int,
        activation: str = "swish",
    ) -> None:
        """Initialize the Teacher network.

        Args:
            input_dim: Dimension of the input.
            hidden_dims: Dimensions of the hidden layers.
            latent_dim: Dimension of the latent vector that will be given to the actor.
            activation: Activation function.
        """
        super().__init__()

        activation_mod = resolve_nn_activation(activation)
        layers, last = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), activation_mod]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(last, latent_dim)
        self.out_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, with_norm: bool = False) -> torch.Tensor:
        """Forward pass."""
        h = self.backbone(x)
        z = self.proj(h)
        if not with_norm:
            return self.out_norm(z)
        else:
            return self.out_norm(z), torch.linalg.vector_norm(z, dim=-1).mean()


def unstack_history_from_segments(
    x_flat: torch.Tensor, obs_dims: tuple[int, ...] | list[int], history_length: int = 4, newest_first: bool = True
):
    """
    x_flat: [B, sum(obs_dims)], each entry is T * d_m
    obs_dims: e.g., [12, 12, 76, 76, 76, 12, 12] with T=4
    Returns: x_seq [B, T, F], per_frame_dims list
    """
    B = x_flat.size(0)
    per_frame_dims = []
    for dim_T in obs_dims:
        assert dim_T % history_length == 0, f"Segment {dim_T} not divisible by T={history_length}"
        per_frame_dims.append(dim_T // history_length)
    F = sum(per_frame_dims)

    slices_per_mod = []
    offset = 0
    for dim_T, d in zip(obs_dims, per_frame_dims):
        seg = x_flat[:, offset : offset + dim_T]  # [B, T*d]
        offset += dim_T
        chunks = [seg[:, i * d : (i + 1) * d] for i in range(history_length)]  # list of [B, d]
        if not newest_first:
            chunks = chunks[::-1]  # make index 0 be "now"
        slices_per_mod.append(chunks)

    frames = []
    for t in range(history_length):  # t=0 is "now"
        frame_t = torch.cat([slices_per_mod[m][t] for m in range(len(per_frame_dims))], dim=-1)  # [B, F]
        frames.append(frame_t)
    x_seq = torch.stack(frames, dim=1)  # [B, T, F]
    return x_seq, per_frame_dims


class StudentEncoder(nn.Module):
    """Student encoder network."""

    def __init__(
        self,
        obs_dims: tuple[int, ...] | list[int],
        history_length: int = 4,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        newest_first: bool = True,
    ) -> torch.Tensor:
        """Initialize the Teacher network.

        Args:
            obs_dims: Dimensions of the different kinds of data in the obs vector.
            history_length: Number of history samples.
            hidden_dim: Dimensions of the hidden layer.
            latent_dim: Dimension of the latent vector that will be given to the actor.
            newest_first: Whether the history is in q_t, q_t-1, q_t-2 order or the opposite.
        """
        super().__init__()
        self.obs_dims = obs_dims
        self.history_length = history_length
        self.newest_first = newest_first
        per_frame_dims = [d // history_length for d in obs_dims]
        self.feat_per_frame = sum(per_frame_dims)

        self.gru = nn.GRU(
            input_size=self.feat_per_frame,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # self.test = nn.Linear(690, latent_dim)
        # self.out_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, with_norm: bool = False) -> torch.Tensor:
        """Forward pass."""

        # print("= Before reshape")
        # print(x.shape)
        # print(x[0, -34:])

        # return self.test(x)

        # Reshape flat [B, T x F] input into [B, T, F]
        x_seq, _ = unstack_history_from_segments(
            x, self.obs_dims, history_length=self.history_length, newest_first=self.newest_first
        )  # [B, T, F]

        # print("= Forward")
        # print(x_seq[0:1, :, -3:])

        out, _ = self.gru(x_seq)  # [B, T, H]
        h_last = out[:, -1]  # summarize up to current time
        z = self.proj(h_last)
        # print(z[0, -3:])
        # quit()
        return z

        if not with_norm:
            return self.out_norm(z)
        else:
            return self.out_norm(z), torch.linalg.vector_norm(z, dim=-1).mean()
