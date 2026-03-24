from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.rollout import evaluate_rollouts
from gnn3.models.packet_mamba import PacketMambaModel, compute_losses
from gnn3.train.config import ExperimentConfig, hidden_corridor_config_for_split


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _setup_distributed(device_name: str) -> tuple[torch.device, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif is_distributed:
        device = torch.device("cpu")
    else:
        device = _resolve_device(device_name)

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return device, rank, local_rank, world_size, is_distributed


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "branch", "--show-current"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def _hardware_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "cuda_devices": 0}
    return {
        "device": str(device),
        "cuda_devices": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device.index or 0),
    }


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _selection_score(val_metrics: dict[str, float], rollout_metrics: Any, config: ExperimentConfig) -> float:
    regret_score = 1.0 / (1.0 + max(float(rollout_metrics.average_regret), 0.0))
    tail_regret_score = 1.0 / (1.0 + max(float(rollout_metrics.p95_regret), 0.0))
    deadline_score = 1.0 / (1.0 + max(float(rollout_metrics.average_deadline_violations), 0.0))
    miss_score = 1.0 - max(min(float(rollout_metrics.deadline_miss_rate), 1.0), 0.0)
    return (
        config.train.selection_val_next_hop_weight * float(val_metrics["next_hop_accuracy"])
        + config.train.selection_rollout_solved_weight * float(rollout_metrics.solved_rate)
        + config.train.selection_rollout_next_hop_weight * float(rollout_metrics.next_hop_accuracy)
        + config.train.selection_rollout_regret_weight * regret_score
        + config.train.selection_rollout_tail_regret_weight * tail_regret_score
        + config.train.selection_rollout_miss_weight * miss_score
        + config.train.selection_rollout_deadline_weight * deadline_score
    )


@torch.no_grad()
def evaluate_decision_dataset(
    model: PacketMambaModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    final_step_only: bool,
    value_weight: float,
    route_weight: float,
    deadline_bce_weight: float,
    slack_loss_weight: float,
    quantile_loss_weight: float,
    selection_soft_target_weight: float,
    selection_soft_target_temperature: float,
    selection_soft_target_on_time_bonus: float,
    quantiles: tuple[float, ...],
    verifier_aux_last_k_steps: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {
        "loss": 0.0,
        "next_hop_loss": 0.0,
        "value_loss": 0.0,
        "route_loss": 0.0,
        "selection_soft_target_loss": 0.0,
        "on_time_loss": 0.0,
        "slack_loss": 0.0,
        "quantile_loss": 0.0,
        "next_hop_accuracy": 0.0,
        "selection_accuracy": 0.0,
        "value_mae": 0.0,
        "value_rmse": 0.0,
        "slack_mae": 0.0,
        "on_time_brier": 0.0,
        "quantile_median_mae": 0.0,
    }
    steps = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch)
        losses = compute_losses(
            output,
            batch,
            final_step_only=final_step_only,
            value_weight=value_weight,
            route_weight=route_weight,
            deadline_bce_weight=deadline_bce_weight,
            slack_weight=slack_loss_weight,
            quantile_weight=quantile_loss_weight,
            selection_soft_target_weight=selection_soft_target_weight,
            selection_soft_target_temperature=selection_soft_target_temperature,
            selection_soft_target_on_time_bonus=selection_soft_target_on_time_bonus,
            quantiles=quantiles,
            verifier_aux_last_k_steps=verifier_aux_last_k_steps,
        )
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        value_error = output["values"] - batch["cost_to_go"]
        totals["value_mae"] += float(value_error.abs().mean().detach().cpu())
        totals["value_rmse"] += float(torch.sqrt((value_error.square()).mean()).detach().cpu())
        valid_mask = batch["candidate_mask"] & batch["node_mask"]
        if valid_mask.any() and output["candidate_on_time_logits"] is not None:
            on_time_prob = torch.sigmoid(output["candidate_on_time_logits"][valid_mask])
            on_time_target = batch["candidate_on_time"][valid_mask]
            totals["on_time_brier"] += float(((on_time_prob - on_time_target) ** 2).mean().detach().cpu())
            totals["slack_mae"] += float(
                (output["candidate_slack"][valid_mask] - batch["candidate_slack"][valid_mask])
                .abs()
                .mean()
                .detach()
                .cpu()
            )
            median_index = len(quantiles) // 2
            totals["quantile_median_mae"] += float(
                (
                    output["candidate_cost_quantiles"][valid_mask][:, median_index]
                    - batch["candidate_cost_to_go"][valid_mask]
                )
                .abs()
                .mean()
                .detach()
                .cpu()
            )
        steps += 1
    if was_training:
        model.train()
    return {key: value / max(steps, 1) for key, value in totals.items()}


def _device_placement(device: torch.device, world_size: int) -> str:
    if device.type != "cuda":
        return str(device)
    if world_size > 1:
        return ",".join(f"cuda:{rank}" for rank in range(world_size))
    return str(device)


def _rollout_metrics_to_dict(rollout_metrics: Any) -> dict[str, float]:
    return {
        "solved_rate": float(rollout_metrics.solved_rate),
        "next_hop_accuracy": float(rollout_metrics.next_hop_accuracy),
        "average_regret": float(rollout_metrics.average_regret),
        "p95_regret": float(rollout_metrics.p95_regret),
        "worst_regret": float(rollout_metrics.worst_regret),
        "average_deadline_violations": float(rollout_metrics.average_deadline_violations),
        "deadline_miss_rate": float(rollout_metrics.deadline_miss_rate),
        "p95_deadline_violations": float(rollout_metrics.p95_deadline_violations),
        "priority_delivered_regret": float(rollout_metrics.priority_delivered_regret),
        "average_oracle_cost": float(rollout_metrics.average_oracle_cost),
        "average_model_cost": float(rollout_metrics.average_model_cost),
    }


def train_experiment(config: ExperimentConfig) -> dict[str, Any] | None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device, rank, local_rank, world_size, is_distributed = _setup_distributed(config.train.device)
    is_main_process = rank == 0
    train_hidden_cfg = hidden_corridor_config_for_split(config.benchmark, "train")
    val_hidden_cfg = hidden_corridor_config_for_split(config.benchmark, "val")
    test_hidden_cfg = hidden_corridor_config_for_split(config.benchmark, "test")
    train_dataset = HiddenCorridorDecisionDataset(
        config=train_hidden_cfg,
        num_episodes=config.benchmark.train_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )
    val_dataset = HiddenCorridorDecisionDataset(
        config=val_hidden_cfg,
        num_episodes=config.benchmark.val_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )
    test_dataset = HiddenCorridorDecisionDataset(
        config=test_hidden_cfg,
        num_episodes=config.benchmark.test_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        if is_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.train.num_workers,
        collate_fn=collate_decisions,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        val_dataset,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_decisions,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_decisions,
        pin_memory=device.type == "cuda",
    )
    train_manifest = train_dataset.manifest()
    val_manifest = val_dataset.manifest()
    test_manifest = test_dataset.manifest()

    model = PacketMambaModel(config.model).to(device)
    if config.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if is_main_process else None
    metrics_path = output_dir / "metrics.jsonl"
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "dataset_manifests.json"
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    use_autocast = device.type == "cuda" and config.train.bf16
    eval_model = model.module if isinstance(model, DDP) else model

    if is_main_process:
        manifest_payload = {
            "train": train_manifest,
            "val": val_manifest,
            "test": test_manifest,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        metadata = {
            "experiment": asdict(config),
            "git_commit": _git_commit(),
            "git_branch": _git_branch(),
            "run_stage": config.stage,
            "run_notes": config.notes,
            "device_placement": _device_placement(device, world_size),
            "manifest_path": str(manifest_path),
            "manifest_hashes": {
                "train": train_manifest["manifest_hash"],
                "val": val_manifest["manifest_hash"],
                "test": test_manifest["manifest_hash"],
            },
            "split_config_seeds": {
                "train": train_hidden_cfg.seed,
                "val": val_hidden_cfg.seed,
                "test": test_hidden_cfg.seed,
            },
            "hardware": {
                **_hardware_summary(device),
                "rank": rank,
                "world_size": world_size,
            },
            "runtime_flags": {
                "bf16": config.train.bf16,
                "compile": config.train.compile,
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            },
            "train_decisions": len(train_dataset),
            "val_decisions": len(val_dataset),
            "test_decisions": len(test_dataset),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_metric = float("-inf")
    start_time = time.time()
    step = 0
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_totals = {
            "loss": 0.0,
            "next_hop_loss": 0.0,
            "value_loss": 0.0,
            "route_loss": 0.0,
            "selection_soft_target_loss": 0.0,
            "on_time_loss": 0.0,
            "slack_loss": 0.0,
            "quantile_loss": 0.0,
            "next_hop_accuracy": 0.0,
            "selection_accuracy": 0.0,
        }
        for batch in train_loader:
            step += 1
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                output = model(batch)
                losses = compute_losses(
                    output,
                    batch,
                    final_step_only=config.model.final_step_only_loss,
                    value_weight=config.train.value_weight,
                    route_weight=config.train.route_weight,
                    deadline_bce_weight=config.train.deadline_bce_weight,
                    slack_weight=config.train.slack_loss_weight,
                    quantile_weight=config.train.quantile_loss_weight,
                    selection_soft_target_weight=config.train.selection_soft_target_weight,
                    selection_soft_target_temperature=config.train.selection_soft_target_temperature,
                    selection_soft_target_on_time_bonus=config.train.selection_soft_target_on_time_bonus,
                    quantiles=config.model.quantile_levels,
                    verifier_aux_last_k_steps=config.model.verifier_aux_last_k_steps,
                )
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            optimizer.step()

            for key in epoch_totals:
                epoch_totals[key] += float(losses[key].detach().cpu())

        train_metrics = {key: value / max(len(train_loader), 1) for key, value in epoch_totals.items()}
        if is_distributed:
            dist.barrier()

        if is_main_process:
            val_metrics = evaluate_decision_dataset(
                eval_model,
                eval_loader,
                device=device,
                final_step_only=config.model.final_step_only_loss,
                value_weight=config.train.value_weight,
                route_weight=config.train.route_weight,
                deadline_bce_weight=config.train.deadline_bce_weight,
                slack_loss_weight=config.train.slack_loss_weight,
                quantile_loss_weight=config.train.quantile_loss_weight,
                selection_soft_target_weight=config.train.selection_soft_target_weight,
                selection_soft_target_temperature=config.train.selection_soft_target_temperature,
                selection_soft_target_on_time_bonus=config.train.selection_soft_target_on_time_bonus,
                quantiles=config.model.quantile_levels,
                verifier_aux_last_k_steps=config.model.verifier_aux_last_k_steps,
            )
            rollout_metrics = evaluate_rollouts(
                eval_model,
                val_dataset.episodes[: config.train.rollout_eval_episodes],
                device=device,
                config=val_hidden_cfg,
            )
            selection_score = _selection_score(val_metrics, rollout_metrics, config)

            epoch_record = {
                "epoch": epoch,
                "elapsed_seconds": time.time() - start_time,
                "selection_score": selection_score,
                "manifest_hashes": {
                    "train": train_manifest["manifest_hash"],
                    "val": val_manifest["manifest_hash"],
                    "test": test_manifest["manifest_hash"],
                },
                "train": train_metrics,
                "val": val_metrics,
                "rollout": _rollout_metrics_to_dict(rollout_metrics),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record) + "\n")

            assert writer is not None
            for split_name, split_metrics in (("train", train_metrics), ("val", val_metrics)):
                for key, value in split_metrics.items():
                    writer.add_scalar(f"{split_name}/{key}", value, epoch)
            writer.add_scalar("rollout/solved_rate", rollout_metrics.solved_rate, epoch)
            writer.add_scalar("rollout/average_regret", rollout_metrics.average_regret, epoch)
            writer.add_scalar("rollout/p95_regret", rollout_metrics.p95_regret, epoch)
            writer.add_scalar("rollout/deadline_miss_rate", rollout_metrics.deadline_miss_rate, epoch)
            writer.add_scalar(
                "rollout/priority_delivered_regret",
                rollout_metrics.priority_delivered_regret,
                epoch,
            )
            writer.add_scalar("selection/score", selection_score, epoch)

            if selection_score > best_metric:
                best_metric = selection_score
                torch.save(
                    {
                        "model": eval_model.state_dict(),
                        "config": asdict(config),
                        "epoch": epoch,
                        "selection_score": selection_score,
                    },
                    best_checkpoint,
                )

        if is_distributed:
            dist.barrier()

    if not is_main_process:
        if is_distributed:
            dist.destroy_process_group()
        return None

    selected_epoch = None
    selected_selection_score = None
    if best_checkpoint.exists():
        checkpoint_payload = torch.load(best_checkpoint, map_location=device)
        eval_model.load_state_dict(checkpoint_payload["model"])
        selected_epoch = checkpoint_payload.get("epoch")
        selected_selection_score = checkpoint_payload.get("selection_score")

    test_metrics = evaluate_decision_dataset(
        eval_model,
        test_loader,
        device=device,
        final_step_only=config.model.final_step_only_loss,
        value_weight=config.train.value_weight,
        route_weight=config.train.route_weight,
        deadline_bce_weight=config.train.deadline_bce_weight,
        slack_loss_weight=config.train.slack_loss_weight,
        quantile_loss_weight=config.train.quantile_loss_weight,
        selection_soft_target_weight=config.train.selection_soft_target_weight,
        selection_soft_target_temperature=config.train.selection_soft_target_temperature,
        selection_soft_target_on_time_bonus=config.train.selection_soft_target_on_time_bonus,
        quantiles=config.model.quantile_levels,
        verifier_aux_last_k_steps=config.model.verifier_aux_last_k_steps,
    )
    test_rollout = evaluate_rollouts(
        eval_model,
        test_dataset.episodes[: config.train.rollout_eval_episodes],
        device=device,
        config=test_hidden_cfg,
    )
    if writer is not None:
        writer.close()

    elapsed_seconds = time.time() - start_time
    gpu_hours = 0.0
    if device.type == "cuda":
        gpu_hours = elapsed_seconds * world_size / 3600.0

    summary = {
        "experiment": config.name,
        "seed": config.seed,
        "stage": config.stage,
        "notes": config.notes,
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "device_placement": _device_placement(device, world_size),
        "manifest_hashes": {
            "train": train_manifest["manifest_hash"],
            "val": val_manifest["manifest_hash"],
            "test": test_manifest["manifest_hash"],
        },
        "split_config_seeds": {
            "train": train_hidden_cfg.seed,
            "val": val_hidden_cfg.seed,
            "test": test_hidden_cfg.seed,
        },
        "best_metric": best_metric,
        "selected_epoch": selected_epoch,
        "selected_selection_score": selected_selection_score,
        "world_size": world_size,
        "elapsed_seconds": elapsed_seconds,
        "gpu_hours": gpu_hours,
        "test": test_metrics,
        "test_rollout": _rollout_metrics_to_dict(test_rollout),
        "output_dir": str(output_dir),
        "bucket": config.bucket,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if is_distributed:
        dist.destroy_process_group()
    return summary
