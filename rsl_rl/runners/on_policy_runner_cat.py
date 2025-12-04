# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from rsl_rl.algorithms import PPOCaT
from rsl_rl.env import VecEnv

from .on_policy_runner import OnPolicyRunner


class OnPolicyRunnerCaT(OnPolicyRunner):
    """On-policy runner for training and evaluation of actor-critic methods.

    Child class of OnPolicyRunner to support Constraints as Terminations (CaT).
    """

    alg: PPOCaT
    """The actor-critic algorithm with Constraints as Terminations."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device

        assert "CaT" in self.cfg["algorithm"]["class_name"]

        super().__init__(env, train_cfg, log_dir, device)
