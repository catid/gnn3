from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.rollout import evaluate_rollouts
from gnn3.models.packet_mamba import PacketMambaModel, compute_losses
from gnn3.train.config import ExperimentConfig


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


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


@torch.no_grad()
def evaluate_decision_dataset(
    model: PacketMambaModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    final_step_only: bool,
    value_weight: float,
    route_weight: float,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {
        "loss": 0.0,
        "next_hop_loss": 0.0,
        "value_loss": 0.0,
        "route_loss": 0.0,
        "next_hop_accuracy": 0.0,
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
        )
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu())
        steps += 1
    if was_training:
        model.train()
    return {key: value / max(steps, 1) for key, value in totals.items()}


def train_experiment(config: ExperimentConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = _resolve_device(config.train.device)
    train_dataset = HiddenCorridorDecisionDataset(
        config=config.benchmark.hidden_corridor,
        num_episodes=config.benchmark.train_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )
    val_dataset = HiddenCorridorDecisionDataset(
        config=config.benchmark.hidden_corridor,
        num_episodes=config.benchmark.val_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )
    test_dataset = HiddenCorridorDecisionDataset(
        config=config.benchmark.hidden_corridor,
        num_episodes=config.benchmark.test_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
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

    model = PacketMambaModel(config.model).to(device)
    if config.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    metrics_path = output_dir / "metrics.jsonl"
    metadata_path = output_dir / "metadata.json"
    best_checkpoint = output_dir / "checkpoints" / "best.pt"
    use_autocast = device.type == "cuda" and config.train.bf16

    metadata = {
        "experiment": asdict(config),
        "git_commit": _git_commit(),
        "hardware": _hardware_summary(device),
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
        epoch_totals = {
            "loss": 0.0,
            "next_hop_loss": 0.0,
            "value_loss": 0.0,
            "route_loss": 0.0,
            "next_hop_accuracy": 0.0,
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
                )
            losses["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
            optimizer.step()

            for key in epoch_totals:
                epoch_totals[key] += float(losses[key].detach().cpu())

        train_metrics = {key: value / max(len(train_loader), 1) for key, value in epoch_totals.items()}
        val_metrics = evaluate_decision_dataset(
            model,
            eval_loader,
            device=device,
            final_step_only=config.model.final_step_only_loss,
            value_weight=config.train.value_weight,
            route_weight=config.train.route_weight,
        )
        rollout_metrics = evaluate_rollouts(
            model,
            val_dataset.episodes[: config.train.rollout_eval_episodes],
            device=device,
            config=config.benchmark.hidden_corridor,
        )

        epoch_record = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - start_time,
            "train": train_metrics,
            "val": val_metrics,
            "rollout": {
                "solved_rate": rollout_metrics.solved_rate,
                "next_hop_accuracy": rollout_metrics.next_hop_accuracy,
                "average_regret": rollout_metrics.average_regret,
                "average_deadline_violations": rollout_metrics.average_deadline_violations,
                "priority_delivered_regret": rollout_metrics.priority_delivered_regret,
            },
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

        for split_name, split_metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in split_metrics.items():
                writer.add_scalar(f"{split_name}/{key}", value, epoch)
        writer.add_scalar("rollout/solved_rate", rollout_metrics.solved_rate, epoch)
        writer.add_scalar("rollout/average_regret", rollout_metrics.average_regret, epoch)
        writer.add_scalar("rollout/priority_delivered_regret", rollout_metrics.priority_delivered_regret, epoch)

        score = val_metrics["next_hop_accuracy"] + rollout_metrics.solved_rate
        if score > best_metric:
            best_metric = score
            torch.save({"model": model.state_dict(), "config": asdict(config), "epoch": epoch}, best_checkpoint)

    test_metrics = evaluate_decision_dataset(
        model,
        test_loader,
        device=device,
        final_step_only=config.model.final_step_only_loss,
        value_weight=config.train.value_weight,
        route_weight=config.train.route_weight,
    )
    test_rollout = evaluate_rollouts(
        model,
        test_dataset.episodes[: config.train.rollout_eval_episodes],
        device=device,
        config=config.benchmark.hidden_corridor,
    )
    writer.close()

    elapsed_seconds = time.time() - start_time
    gpu_hours = 0.0
    if device.type == "cuda":
        gpu_hours = elapsed_seconds / 3600.0

    summary = {
        "best_metric": best_metric,
        "elapsed_seconds": elapsed_seconds,
        "gpu_hours": gpu_hours,
        "test": test_metrics,
        "test_rollout": {
            "solved_rate": test_rollout.solved_rate,
            "next_hop_accuracy": test_rollout.next_hop_accuracy,
            "average_regret": test_rollout.average_regret,
            "average_deadline_violations": test_rollout.average_deadline_violations,
            "priority_delivered_regret": test_rollout.priority_delivered_regret,
        },
        "output_dir": str(output_dir),
        "bucket": config.bucket,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
