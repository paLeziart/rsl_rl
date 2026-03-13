# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch

from rsl_rl.algorithms import PPOCaTTeacherStudent
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan

from .on_policy_runner_cat import OnPolicyRunnerCaT


class OnPolicyRunnerCaTTeacherStudent(OnPolicyRunnerCaT):
    """On-policy runner for reinforcement learning algorithms.

    Child class of OnPolicyRunner to support Constraints as Terminations (CaT).
    """

    alg: PPOCaTTeacherStudent
    """The actor-critic algorithm with Constraints as Terminations."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        assert "CaT" in self.cfg["algorithm"]["class_name"]
        assert "TeacherStudent" in self.cfg["algorithm"]["class_name"]

        super().__init__(env, train_cfg, log_dir, device)

        assert type(self.alg) is PPOCaTTeacherStudent

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        switch = torch.zeros(self.env.num_envs, 1, dtype=torch.bool, device=self.device)
        for it in range(start_it, total_it):
            start = time.time()
            if it % 20 == 0:
                off = 1000
                num_switch = int(self.env.num_envs * max(0.0, min(1.0, (it - off) / (total_it - 3000))))
                print("= NEW NUM OF STUDENTS: ", num_switch)

            # Refresh teacher-student mix and network optimizer
            alpha = self.alg.update_training_mix(it, **self.cfg["algorithm"])

            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs, alpha=alpha)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    extras["switch"] = switch  # Store who controlled the envs (either student or teacher)
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
