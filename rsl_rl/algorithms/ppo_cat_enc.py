# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict
from typing import Any

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPEncoderModel, MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from .ppo_cat import PPOCaT


class PPOCaTEncoder(PPOCaT):
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347).

    Support the Constraints as Terminations (CaT) -> See PPOCaT.
    Support a Teacher Student scheme with progressive switch from Teacher to Student.
    """

    def __init__(
        self,
        actor: MLPEncoderModel,
        critic: MLPModel,
        storage: RolloutStorage,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        **kwargs: Any,
    ) -> None:

        super().__init__(actor, critic, storage, learning_rate=learning_rate, **kwargs)

        self.learning_phase = ""

        # Create the main PPO optimizer, including the teacher
        # self.optimizer = resolve_optimizer(optimizer)(
        #     chain(self.actor.distribution.parameters(),
        #           self.actor.actor.parameters(),
        #           self.actor.actor_normalizer.parameters(),
        #           self.critic.parameters(),
        #           self.actor.teacher.parameters(),
        #           self.actor.teacher_normalizer.parameters()), lr=learning_rate
        # )  # type: ignore

        # # Create the optimizer for the student
        # self.student_optimizer = resolve_optimizer(optimizer)(
        #     chain(self.actor.student.parameters(), self.actor.student_normalizer.parameters()), lr=learning_rate
        # )  # type: ignore

        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore

    def compute_latent(self, obs: TensorDict, switch: torch.Tensor) -> None:
        """Compute the latent space information and store it in obs["latent"]."""
        # Mixed actor input: do not backprop into student
        raise NotImplementedError
        latent_obs = self.teacher(obs)
        latent_obs[switch[:, 0]] = (self.student(obs)[switch[:, 0]]).detach()
        obs["latent"] = latent_obs * 0.0 + 0.1

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_actor_kl = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_encoder_loss = 0
        mean_student_kl_loss = 0
        mean_teacher_norm = 0
        mean_student_norm = 0

        mean_pre_teacher_norm = 0
        mean_pre_student_norm = 0
        mean_mu_mismatch = 0.0

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            #print("= LOG PROB: ")
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                actor_kl = torch.mean(kl)

            # Compute KL divergence and adapt the learning rate
            if self.learning_phase != "B" and self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            #print("ratio_mean", ratio.mean().item(), "ratio_min", ratio.min().item(), "ratio_max", ratio.max().item())


            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            privileged_predicted = batch.observations["latent"]  # type: ignore
            privileged_truth = self.actor.get_observations(batch.observations,
                                                           "privileged",
                                                           batch.masks,
                                                           batch.hidden_states[0])
            
            
            #self.critic.get_latent(
            #    batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0]
            #)[:, -16:]

            # print("Predicted")
            # print(encoder_latent[0])
            # print("Truth")
            # print(critic_latent[0])

            mseloss = torch.nn.MSELoss()
            encoder_loss = mseloss(privileged_predicted, privileged_truth.detach())
            loss += encoder_loss

            perDim = torch.mean(torch.square(privileged_predicted - privileged_truth), dim=0)
            # print(perDim)
            #denormed_encoder = self.critic.obs_normalizer.inverse(privileged_predicted)
            #denormed_critic = batch.observations["critic"][:, -20:]
            # print("====")
            # print(denormed_critic[0])
            # print(denormed_encoder[0])
            # print(encoder_latent[0] - critic_latent[0])


            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                    # Compute number of augmentations per sample
                    num_aug = int(batch.observations.batch_size[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())  # , detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()

            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_actor_kl += actor_kl.item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # Encoder loss
            mean_encoder_loss += encoder_loss.item()
            # if self.learning_phase != "C":
            #     mean_encoder_loss += student_latent_loss.item()
            #     mean_student_kl_loss += student_kl.item() / student_distribution_params[1].shape[-1]

            # mean_pre_teacher_norm += pre_teacher_norm.item()
            # mean_pre_student_norm += pre_student_norm.item()
            # mean_mu_mismatch += mu_mismatch.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_actor_kl /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_encoder_loss /= num_updates
        mean_student_kl_loss /= num_updates
        mean_teacher_norm /= num_updates
        mean_student_norm /= num_updates
        mean_pre_teacher_norm /= num_updates
        mean_pre_student_norm /= num_updates
        mean_mu_mismatch /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "actor_kl": mean_actor_kl,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["encoder"] = mean_encoder_loss
        # loss_dict["student_kl"] = mean_student_kl_loss
        # loss_dict["teacher_norm"] = mean_teacher_norm
        # loss_dict["student_norm"] = mean_student_norm
        # loss_dict["mean_pre_teacher_norm"] = mean_pre_teacher_norm
        # loss_dict["mean_pre_student_norm"] = mean_pre_student_norm
        # loss_dict["mu_mismatch"] = mean_mu_mismatch

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        # saved_dict["student_optimizer_state_dict"] = self.student_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "encoder": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
            # self.student_optimizer.load_state_dict(loaded_dict["student_optimizer_state_dict"])
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPEncoderModel:
        """Get the policy model."""
        # TODO: Properly return actor/teacher/student
        return self.actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOCaTEncoder:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPOCaTEncoder] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPEncoderModel] = resolve_callable(cfg["actor_encoder"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic", "encoder"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        obs_sets = {"actor": "actor", "encoder": "encoder", "privileged": "privileged"}
        actor: MLPEncoderModel = actor_class(
            obs, cfg["obs_groups"], obs_sets, env.num_actions, **cfg["actor_encoder"]
        ).to(device)
        print(f"Actor - Teacher - Student Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl-CaT", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPOCaTEncoder = alg_class(
            actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel

    def cosine_alpha(self, step: int, warmup_steps: int, ramp_steps: int) -> float:
        """Piecewise cosine schedule for alpha across 3 phases.

        Phase A (warmup):      alpha = 0
        Phase B (distillation):alpha in [0,1] via cosine ramp
        Phase C (finetune):    alpha = 1

        Args:
            step:           global step (>= 0)
            warmup_steps:   number of steps to keep alpha=0 at start (Phase A).
            ramp_steps:     number of steps for cosine ramp 0->1 (Phase B).
            finetune_steps: alpha=1 after warmup+ramp (Phase C).
        """
        assert step >= 0, "step must be >= 0"
        assert warmup_steps >= 0 and ramp_steps >= 1, "warmup>=0 and ramp>=1"

        # Phase A: alpha = 0
        if step < warmup_steps:
            return 0.0

        # Phase B: cosine ramp from 0 -> 1
        t = (step - warmup_steps) / float(ramp_steps)
        if t < 1.0:
            # Cosine goes from 0 to 1 smoothly (with min value at 0.05 for gradients)
            return max(0.05, 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t))))

        # Phase C: alpha = 1
        return 1.0

    def freeze_module(self, module: torch.nn.Module) -> None:
        """Freeze a Torch module by disabling their need for gradient."""
        for p in module.parameters():
            p.requires_grad = False
        # module.eval()

    def unfreeze_module(self, module: torch.nn.Module) -> None:
        """Unfreeze a Torch module by enabling their need for gradient."""
        for p in module.parameters():
            p.requires_grad = True
        # module.train()

    def update_training_mix(self, step: int, optimizer: str, learning_rate: float, **kwargs: Any) -> float:

        new_alpha = self.cosine_alpha(step, warmup_steps=1500, ramp_steps=1500)
        refresh_optimizer = False
        if self.learning_phase == "":
            print("\033[91m== ENTER PHASE A == \033[0m")
            self.learning_phase = "A"
            # Phase A: Warmup by learning teacher + actor.
            # Unfreeze everything then freeze student only.
            self.unfreeze_module(self.actor)
            self.freeze_module(self.actor.student)
            self.freeze_module(self.actor.student_normalizer)
            refresh_optimizer = True
        elif self.learning_phase == "A" and step == 1000:  # new_alpha > 0:
            print("\033[91m== ENTER PHASE B == \033[0m")
            self.learning_phase = "B"
            # Phase B: Learning student with frozen actor and teacher.
            # Freeze everything then unfreeze student only.
            self.freeze_module(self.actor)
            self.unfreeze_module(self.actor.student)
            self.unfreeze_module(self.actor.student_normalizer)
            # self.unfreeze_module(self.actor.distribution)
            refresh_optimizer = True
        elif self.learning_phase == "B" and new_alpha == 1.0:
            print("\033[91m== ENTER PHASE C == \033[0m")
            self.learning_phase = "C"
            # Phase C: Finetuning student with unfrozen actor.
            # Unfreeze everything then freeze teacher only.
            self.unfreeze_module(self.actor)
            self.freeze_module(self.actor.teacher)
            self.freeze_module(self.actor.teacher_normalizer)
            refresh_optimizer = True

        if refresh_optimizer:
            params = []
            teacher_params = [p for p in self.actor.teacher.parameters() if p.requires_grad]
            if len(teacher_params) > 0:
                params.append({"params": teacher_params, "lr": 3e-4, "weight_decay": 1e-5})
            other_modules = [
                self.actor.distribution,
                self.actor.actor,
                self.actor.actor_normalizer,
                self.critic,
                self.actor.teacher_normalizer,
                self.actor.student,
                self.actor.student_normalizer,
            ]
            other_params = []
            for module in other_modules:
                other_params += [p for p in module.parameters() if p.requires_grad]
            params.append({"params": other_params, "lr": 3e-4})

            self.optimizer = resolve_optimizer(optimizer)(params)

            quit()

            # Create the optimizer
            # self.optimizer = resolve_optimizer(optimizer)([
            #     {
            #         "params": chain(
            #             self.actor.distribution.parameters(),
            #             self.actor.actor.parameters(),
            #             self.actor.actor_normalizer.parameters(),
            #             self.critic.parameters(),
            #             self.actor.teacher_normalizer.parameters(),
            #             self.actor.student.parameters(),
            #             self.actor.student_normalizer.parameters(),
            #         ),
            #         "lr": 3e-4,
            #     },
            #     {"params": self.actor.teacher.parameters(), "lr": 3e-4, "weight_decay": 1e-5},
            # ])


            # self.optimizer = resolve_optimizer(optimizer)(
            #     chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
            # )  # type: ignore

        print(f"\034[91m== PHASE {self.learning_phase} {new_alpha} == \034[0m")

        self.alpha = new_alpha
        return self.alpha
