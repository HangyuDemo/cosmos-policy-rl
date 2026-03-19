#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GRPO-style RL fine-tuning entrypoint for Cosmos Policy.

This script assumes you already have a pretrained Cosmos Policy checkpoint.
It fine-tunes the model by:

1. Sampling multiple candidates for the same input state.
2. Scoring those candidates with a group-relative reward.
3. Updating the model with a diffusion-compatible surrogate objective that
   focuses only on the action and keypoint trajectory slots.

This is intentionally a practical GRPO-style scaffold for a diffusion policy.
Unlike an autoregressive language model, the diffusion backbone does not expose
exact token log-probabilities, so we use the denoising energy on the sampled
action/trajectory targets as a surrogate policy score.
"""

from __future__ import annotations

import argparse
import os
import traceback
from typing import Dict

import torch
from loguru import logger as logging
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.config import Config, load_config, pretty_print_overrides
from cosmos_policy._src.imaginaire.lazy_config import LazyConfig, instantiate
from cosmos_policy._src.imaginaire.serialization import to_yaml
from cosmos_policy._src.imaginaire.utils import distributed, misc
from cosmos_policy._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_policy._src.imaginaire.utils.launch import log_reproducible_setup
from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence, extract_value_from_latent_sequence


def extract_keypoint_trajectory_from_latent_sequence(
    output_latent: torch.Tensor, trajectory_shape: tuple[int, int], trajectory_indices: torch.Tensor
) -> torch.Tensor:
    """Decode the trajectory latent frame back into a `(chunk_size, 2)` trajectory."""
    batch_indices = torch.arange(output_latent.shape[0], device=output_latent.device)
    trajectory_latent_frame = output_latent[batch_indices, :, trajectory_indices, :, :]
    batch_size = trajectory_latent_frame.shape[0]
    flat_trajectory_latent = trajectory_latent_frame.reshape(batch_size, -1)
    num_latent_elements = flat_trajectory_latent.shape[1]
    num_trajectory_elements = trajectory_shape[0] * trajectory_shape[1]
    assert num_trajectory_elements <= num_latent_elements, (
        f"Trajectory shape {trajectory_shape} requires {num_trajectory_elements} elements, "
        f"but the latent only has {num_latent_elements} elements."
    )
    num_trajectory_chunks = num_latent_elements // num_trajectory_elements
    all_trajectory_chunks = flat_trajectory_latent[:, : num_trajectory_chunks * num_trajectory_elements].reshape(
        batch_size, num_trajectory_chunks, num_trajectory_elements
    )
    all_trajectory_chunks = all_trajectory_chunks.reshape(
        batch_size, num_trajectory_chunks, trajectory_shape[0], trajectory_shape[1]
    )
    return torch.mean(all_trajectory_chunks, dim=1)


def maybe_compute_text_embeddings(model, data_batch: Dict[str, torch.Tensor]) -> None:
    if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")


def repeat_batch_for_candidates(data_batch: Dict[str, torch.Tensor], repeats: int) -> Dict[str, torch.Tensor]:
    """Repeat batch-major tensors so all candidate samples can be rescored in one forward pass."""
    batch_size = data_batch["actions"].shape[0]
    repeated: Dict[str, torch.Tensor] = {}
    for key, value in data_batch.items():
        if not torch.is_tensor(value):
            repeated[key] = value
            continue
        if value.ndim > 0 and value.shape[0] == batch_size:
            repeated[key] = value.repeat_interleave(repeats, dim=0)
        else:
            repeated[key] = value
    return repeated


def sample_candidate_group(
    model,
    data_batch: Dict[str, torch.Tensor],
    group_size: int,
    num_steps: int,
    seed_base: int,
    use_variance_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample multiple outputs for the same input state.

    Returns:
        action_candidates: (B, G, chunk_size, action_dim)
        trajectory_candidates: (B, G, chunk_size, 2)
        value_candidates: (B, G)
    """
    batch_size = data_batch["actions"].shape[0]
    action_shape = tuple(data_batch["actions"].shape[1:])
    trajectory_shape = tuple(data_batch["keypoint_trajectory"].shape[1:])

    action_candidates = []
    trajectory_candidates = []
    value_candidates = []

    for group_idx in range(group_size):
        latent_samples = model.generate_samples_from_batch(
            data_batch,
            n_sample=batch_size,
            num_steps=num_steps,
            seed=seed_base + group_idx,
            is_negative_prompt=False,
            use_variance_scale=use_variance_scale,
        )

        actions = extract_action_chunk_from_latent_sequence(
            latent_samples,
            action_shape=action_shape,
            action_indices=data_batch["action_latent_idx"],
        ).to(torch.float32)

        trajectories = extract_keypoint_trajectory_from_latent_sequence(
            latent_samples,
            trajectory_shape=trajectory_shape,
            trajectory_indices=data_batch["keypoint_trajectory_latent_idx"],
        ).to(torch.float32)

        values = extract_value_from_latent_sequence(latent_samples, data_batch["value_latent_idx"])
        values = torch.clamp((values + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)

        action_candidates.append(actions)
        trajectory_candidates.append(trajectories)
        value_candidates.append(values)

    action_candidates = torch.stack(action_candidates, dim=1)
    trajectory_candidates = torch.stack(trajectory_candidates, dim=1)
    value_candidates = torch.stack(value_candidates, dim=1)
    return action_candidates, trajectory_candidates, value_candidates


def compute_candidate_rewards(
    data_batch: Dict[str, torch.Tensor],
    action_candidates: torch.Tensor,
    trajectory_candidates: torch.Tensor,
    value_candidates: torch.Tensor,
    action_reward_weight: float,
    trajectory_reward_weight: float,
    value_reward_weight: float,
) -> Dict[str, torch.Tensor]:
    """
    Group-relative reward built around action + trajectory, with optional value shaping.
    """
    gt_actions = data_batch["actions"].to(torch.float32).unsqueeze(1)
    gt_trajectory = data_batch["keypoint_trajectory"].to(torch.float32).unsqueeze(1)

    action_mse = ((action_candidates - gt_actions) ** 2).mean(dim=(2, 3))
    trajectory_mse = ((trajectory_candidates - gt_trajectory) ** 2).mean(dim=(2, 3))

    reward = (
        -action_reward_weight * action_mse
        - trajectory_reward_weight * trajectory_mse
        + value_reward_weight * value_candidates
    )

    return {
        "reward": reward,
        "action_mse": action_mse,
        "trajectory_mse": trajectory_mse,
    }


def normalize_group_advantages(reward: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    reward_mean = reward.mean(dim=1, keepdim=True)
    reward_std = reward.std(dim=1, keepdim=True, unbiased=False)
    return (reward - reward_mean) / (reward_std + eps)


def compute_action_trajectory_policy_energy(
    model,
    repeated_batch: Dict[str, torch.Tensor],
    sampled_actions: torch.Tensor,
    sampled_trajectories: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the denoising energy for the sampled action/trajectory targets.

    Lower energy means the current policy likes that sample more.
    """
    _, x0_B_C_T_H_W, condition = model.get_data_and_condition(repeated_batch)
    sigma_B_T, epsilon_B_C_T_H_W = model.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)
    x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = model.broadcast_split_for_model_parallelsim(
        x0_B_C_T_H_W,
        condition,
        epsilon_B_C_T_H_W,
        sigma_B_T,
    )

    output_batch, _, _, _ = model.compute_loss_with_epsilon_and_sigma(
        x0_B_C_T_H_W,
        condition,
        epsilon_B_C_T_H_W,
        sigma_B_T,
        action_chunk=sampled_actions,
        action_indices=repeated_batch["action_latent_idx"],
        keypoint_trajectory=sampled_trajectories,
        keypoint_trajectory_indices=repeated_batch["keypoint_trajectory_latent_idx"],
        proprio=repeated_batch["proprio"],
        current_proprio_indices=repeated_batch["current_proprio_latent_idx"],
        future_proprio=repeated_batch["future_proprio"],
        future_proprio_indices=repeated_batch["future_proprio_latent_idx"],
        future_wrist_image_indices=repeated_batch["future_wrist_image_latent_idx"],
        future_wrist_image2_indices=(
            repeated_batch["future_wrist_image2_latent_idx"]
            if "future_wrist_image2_latent_idx" in repeated_batch
            else None
        ),
        future_image_indices=repeated_batch["future_image_latent_idx"],
        future_image2_indices=(
            repeated_batch["future_image2_latent_idx"] if "future_image2_latent_idx" in repeated_batch else None
        ),
        rollout_data_mask=repeated_batch["rollout_data_mask"],
        world_model_sample_mask=repeated_batch["world_model_sample_mask"],
        value_function_sample_mask=repeated_batch["value_function_sample_mask"],
        value_function_return=repeated_batch["value_function_return"],
        value_indices=repeated_batch["value_latent_idx"],
    )

    per_frame_energy = output_batch["edm_loss_per_frame"]
    batch_indices = torch.arange(per_frame_energy.shape[0], device=per_frame_energy.device)
    action_energy = per_frame_energy[batch_indices, repeated_batch["action_latent_idx"]]
    trajectory_energy = per_frame_energy[batch_indices, repeated_batch["keypoint_trajectory_latent_idx"]]
    return action_energy + trajectory_energy


def launch(config: Config, args: argparse.Namespace) -> None:
    with distributed_init():
        distributed.init()

    if parallel_state.get_data_parallel_world_size() != 1:
        raise NotImplementedError(
            "train_rl.py currently supports single-process RL fine-tuning only. "
            "Use one GPU/process for this stage."
        )

    config.validate()
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    log_reproducible_setup(config, args)

    with model_init():
        model = instantiate(config.model)

    model = model.to("cuda", memory_format=config.trainer.memory_format)  # type: ignore
    model.on_train_start(config.trainer.memory_format)
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=args.use_grad_scaler)
    iteration = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    model.train()

    with data_loader_init():
        dataset = instantiate(config.dataloader_train.dataset)
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            seed=0,
        )
        dataloader_train = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=config.dataloader_train.drop_last,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
            pin_memory_device=config.dataloader_train.pin_memory_device,
            timeout=config.dataloader_train.timeout,
        )

    logging.info(
        "Starting GRPO-style RL fine-tuning "
        f"(group_size={args.group_size}, num_steps={args.num_steps}, aux_supervised_weight={args.aux_supervised_weight})"
    )

    epoch = 0
    end_training = False
    while not end_training:
        dataloader_train.sampler.set_epoch(epoch)
        for data_batch in dataloader_train:
            if iteration >= config.trainer.max_iter:
                end_training = True
                break

            data_batch = misc.to(data_batch, device="cuda")
            maybe_compute_text_embeddings(model, data_batch)

            with torch.no_grad():
                action_candidates, trajectory_candidates, value_candidates = sample_candidate_group(
                    model=model,
                    data_batch=data_batch,
                    group_size=args.group_size,
                    num_steps=args.num_steps,
                    seed_base=args.seed + iteration * 1000,
                    use_variance_scale=args.use_variance_scale,
                )
                reward_dict = compute_candidate_rewards(
                    data_batch=data_batch,
                    action_candidates=action_candidates,
                    trajectory_candidates=trajectory_candidates,
                    value_candidates=value_candidates,
                    action_reward_weight=args.action_reward_weight,
                    trajectory_reward_weight=args.trajectory_reward_weight,
                    value_reward_weight=args.value_reward_weight,
                )
                advantages = normalize_group_advantages(reward_dict["reward"])
                if args.advantage_clip is not None:
                    advantages = torch.clamp(advantages, -args.advantage_clip, args.advantage_clip)

            repeated_batch = repeat_batch_for_candidates(data_batch, args.group_size)
            sampled_actions = action_candidates.reshape(-1, *action_candidates.shape[2:])
            sampled_trajectories = trajectory_candidates.reshape(-1, *trajectory_candidates.shape[2:])

            optimizer.zero_grad(set_to_none=True)
            policy_energy = compute_action_trajectory_policy_energy(
                model=model,
                repeated_batch=repeated_batch,
                sampled_actions=sampled_actions,
                sampled_trajectories=sampled_trajectories,
            ).reshape(data_batch["actions"].shape[0], args.group_size)

            # Maximize advantage-weighted surrogate log-probability, approximated here by minimizing
            # the denoising energy of good samples and increasing it for bad samples.
            rl_loss = torch.mean(advantages.detach() * policy_energy)
            total_loss = rl_loss

            supervised_loss = None
            if args.aux_supervised_weight > 0:
                _, supervised_loss = model.training_step(data_batch, iteration)
                total_loss = total_loss + args.aux_supervised_weight * supervised_loss

            grad_scaler.scale(total_loss).backward()
            grad_scaler.unscale_(optimizer)
            if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            iteration += 1

            if iteration % args.log_iter == 0:
                reward_mean = reward_dict["reward"].mean().item()
                reward_best = reward_dict["reward"].max(dim=1).values.mean().item()
                best_action_mse = reward_dict["action_mse"].min(dim=1).values.mean().item()
                best_trajectory_mse = reward_dict["trajectory_mse"].min(dim=1).values.mean().item()
                msg = (
                    f"[iter {iteration}] total_loss={total_loss.item():.6f} "
                    f"rl_loss={rl_loss.item():.6f} "
                    f"reward_mean={reward_mean:.6f} reward_best={reward_best:.6f} "
                    f"best_action_mse={best_action_mse:.6f} best_trajectory_mse={best_trajectory_mse:.6f}"
                )
                if supervised_loss is not None:
                    msg += f" supervised_loss={supervised_loss.item():.6f}"
                logging.info(msg)

            if iteration % config.checkpoint.save_iter == 0:
                trainer.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)

        epoch += 1

    if iteration % config.checkpoint.save_iter != 0:
        trainer.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)

    logging.success("Done with GRPO-style RL fine-tuning.")
    trainer.checkpointer.finalize()
    distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO-style RL fine-tuning for Cosmos Policy")
    parser.add_argument("--config", help="Path to the config file", required=False)
    parser.add_argument("--group-size", type=int, default=4, help="Number of sampled candidates per input state")
    parser.add_argument("--num-steps", type=int, default=20, help="Diffusion denoising steps per candidate")
    parser.add_argument("--seed", type=int, default=1, help="Base sampling seed")
    parser.add_argument("--use-variance-scale", action="store_true", help="Use variance scaling for more sample diversity")
    parser.add_argument(
        "--action-reward-weight",
        type=float,
        default=1.0,
        help="Weight for the action imitation reward term (-MSE to dataset action chunk)",
    )
    parser.add_argument(
        "--trajectory-reward-weight",
        type=float,
        default=1.0,
        help="Weight for the trajectory imitation reward term (-MSE to dataset trajectory)",
    )
    parser.add_argument(
        "--value-reward-weight",
        type=float,
        default=0.1,
        help="Weight for predicted value used as reward shaping",
    )
    parser.add_argument(
        "--aux-supervised-weight",
        type=float,
        default=0.05,
        help="Optional supervised anchor weight for stability",
    )
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=5.0,
        help="Clip normalized group advantages to this absolute value",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--log-iter", type=int, default=10, help="Logging frequency")
    parser.add_argument("--use-grad-scaler", action="store_true", help="Enable GradScaler for RL fine-tuning")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--dryrun", action="store_true", help="Do a dry run without training")
    args = parser.parse_args()

    config = load_config(args.config, args.opts, enable_one_logger=True)

    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error("to_yaml failed, falling back to LazyConfig.save_yaml:")
            logging.error(f"Traceback: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        launch(config, args)
