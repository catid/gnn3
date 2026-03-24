#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.rollout import evaluate_rollouts
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import (
    ExperimentConfig,
    hidden_corridor_config_for_split,
    load_experiment_config,
)
from gnn3.train.trainer import _resolve_device, evaluate_decision_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment configs to sweep.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV path.")
    parser.add_argument("--output-json", help="Optional JSON summary path.")
    parser.add_argument("--output-plot", help="Optional PNG plot path.")
    return parser.parse_args()


def evaluate_config(config: ExperimentConfig, checkpoint_path: Path) -> dict[str, object]:
    device = _resolve_device(config.train.device)
    test_hidden_cfg = hidden_corridor_config_for_split(config.benchmark, "test")
    dataset = HiddenCorridorDecisionDataset(
        config=test_hidden_cfg,
        num_episodes=config.benchmark.test_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        collate_fn=collate_decisions,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])

    decision_metrics = evaluate_decision_dataset(
        model,
        loader,
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
        path_soft_target_weight=config.train.path_soft_target_weight,
        path_soft_target_temperature=config.train.path_soft_target_temperature,
        path_soft_target_on_time_bonus=config.train.path_soft_target_on_time_bonus,
        selection_pairwise_weight=config.train.selection_pairwise_weight,
        selection_pairwise_temperature=config.train.selection_pairwise_temperature,
        selection_pairwise_on_time_bonus=config.train.selection_pairwise_on_time_bonus,
        selection_pairwise_slack_bonus=config.train.selection_pairwise_slack_bonus,
        selection_pairwise_margin=config.train.selection_pairwise_margin,
        selection_feasible_target_weight=config.train.selection_feasible_target_weight,
        selection_slack_critical_weight=config.train.selection_slack_critical_weight,
        selection_slack_critical_scale=config.train.selection_slack_critical_scale,
        quantiles=config.model.quantile_levels,
        verifier_aux_last_k_steps=config.model.verifier_aux_last_k_steps,
    )
    rollout_metrics = evaluate_rollouts(
        model,
        dataset.episodes[: config.train.rollout_eval_episodes],
        device=device,
        config=test_hidden_cfg,
    )

    hidden_cfg = test_hidden_cfg
    return {
        "experiment": config.name,
        "stage": config.stage,
        "bucket": config.bucket,
        "notes": config.notes,
        "device": config.train.device,
        "num_communities": hidden_cfg.num_communities,
        "tree_depth_min": hidden_cfg.tree_depth_min,
        "tree_depth_max": hidden_cfg.tree_depth_max,
        "branching_factor": hidden_cfg.branching_factor,
        "packets_max": hidden_cfg.packets_max,
        "queue_low": hidden_cfg.community_base_queue[0],
        "queue_high": hidden_cfg.community_base_queue[1],
        "queue_penalty": hidden_cfg.queue_penalty,
        "capacity_penalty": hidden_cfg.capacity_penalty,
        "urgency_penalty": hidden_cfg.urgency_penalty,
        "decision_loss": decision_metrics["loss"],
        "test_next_hop_accuracy": decision_metrics["next_hop_accuracy"],
        "value_mae": decision_metrics["value_mae"],
        "value_rmse": decision_metrics["value_rmse"],
        "rollout_solved_rate": rollout_metrics.solved_rate,
        "rollout_next_hop_accuracy": rollout_metrics.next_hop_accuracy,
        "average_regret": rollout_metrics.average_regret,
        "p95_regret": rollout_metrics.p95_regret,
        "worst_regret": rollout_metrics.worst_regret,
        "deadline_violations": rollout_metrics.average_deadline_violations,
        "deadline_miss_rate": rollout_metrics.deadline_miss_rate,
        "p95_deadline_violations": rollout_metrics.p95_deadline_violations,
        "priority_delivered_regret": rollout_metrics.priority_delivered_regret,
    }


def maybe_plot(df: pd.DataFrame, output_plot: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].bar(df["experiment"], df["test_next_hop_accuracy"], color="#1f77b4")
    axes[0].set_title("OOD Next-Hop Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(df["experiment"], df["rollout_solved_rate"], color="#2ca02c")
    axes[1].set_title("OOD Solved Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(df["experiment"], df["average_regret"], color="#d62728")
    axes[2].set_title("OOD Average Regret")
    axes[2].tick_params(axis="x", rotation=20)

    axes[3].bar(df["experiment"], df["deadline_miss_rate"], color="#9467bd")
    axes[3].set_title("OOD Deadline Miss Rate")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=160)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    rows = [evaluate_config(load_experiment_config(path), checkpoint_path) for path in args.configs]
    df = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.output_plot:
        maybe_plot(df, Path(args.output_plot))

    print(df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
