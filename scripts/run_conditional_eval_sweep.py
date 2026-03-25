#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.deployment import DeploymentDecision, round4_verifier_risk_switch
from gnn3.eval.rollout import evaluate_rollouts
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import (
    ExperimentConfig,
    hidden_corridor_config_for_split,
    load_experiment_config,
)
from gnn3.train.trainer import _resolve_device, evaluate_decision_dataset


@dataclass(frozen=True)
class LoadedDeploymentModel:
    label: str
    config: ExperimentConfig
    checkpoint_path: Path
    model: PacketMambaModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, help="Training config for the default checkpoint.")
    parser.add_argument("--base-checkpoint", required=True, help="Checkpoint for the default deployment path.")
    parser.add_argument("--verifier-config", required=True, help="Training config for the verifier checkpoint.")
    parser.add_argument("--verifier-checkpoint", required=True, help="Checkpoint for the verifier deployment path.")
    parser.add_argument("--eval-configs", nargs="+", required=True, help="Experiment configs to evaluate.")
    parser.add_argument(
        "--policy",
        default="round4_verifier_risk_switch",
        choices=("round4_verifier_risk_switch", "always_base", "always_verifier"),
        help="Deployment rule to apply across eval configs.",
    )
    parser.add_argument("--device", default=None, help="Override evaluation device (defaults to each eval config device).")
    parser.add_argument("--output-csv", required=True, help="Destination CSV path.")
    parser.add_argument("--output-json", help="Optional JSON summary path.")
    parser.add_argument("--output-plot", help="Optional PNG plot path.")
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, device: torch.device, label: str) -> LoadedDeploymentModel:
    config = load_experiment_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    return LoadedDeploymentModel(
        label=label,
        config=config,
        checkpoint_path=Path(checkpoint_path),
        model=model,
    )


def _policy_decision(policy: str, eval_config: ExperimentConfig) -> DeploymentDecision:
    hidden_cfg = hidden_corridor_config_for_split(eval_config.benchmark, "test")
    if policy == "always_base":
        return DeploymentDecision(variant="base", rule="always_base", reasons=("forced_base",))
    if policy == "always_verifier":
        return DeploymentDecision(variant="verifier", rule="always_verifier", reasons=("forced_verifier",))
    return round4_verifier_risk_switch(hidden_cfg)


def _evaluate_config(
    eval_config: ExperimentConfig,
    deployment: LoadedDeploymentModel,
    *,
    device: torch.device,
    decision: DeploymentDecision,
) -> dict[str, object]:
    test_hidden_cfg = hidden_corridor_config_for_split(eval_config.benchmark, "test")
    dataset = HiddenCorridorDecisionDataset(
        config=test_hidden_cfg,
        num_episodes=eval_config.benchmark.test_episodes,
        curriculum_levels=eval_config.benchmark.curriculum_levels,
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_config.train.eval_batch_size,
        shuffle=False,
        collate_fn=collate_decisions,
    )
    decision_metrics = evaluate_decision_dataset(
        deployment.model,
        loader,
        device=device,
        final_step_only=deployment.config.model.final_step_only_loss,
        value_weight=deployment.config.train.value_weight,
        route_weight=deployment.config.train.route_weight,
        deadline_bce_weight=deployment.config.train.deadline_bce_weight,
        slack_loss_weight=deployment.config.train.slack_loss_weight,
        quantile_loss_weight=deployment.config.train.quantile_loss_weight,
        selection_soft_target_weight=deployment.config.train.selection_soft_target_weight,
        selection_soft_target_temperature=deployment.config.train.selection_soft_target_temperature,
        selection_soft_target_on_time_bonus=deployment.config.train.selection_soft_target_on_time_bonus,
        path_soft_target_weight=deployment.config.train.path_soft_target_weight,
        path_soft_target_temperature=deployment.config.train.path_soft_target_temperature,
        path_soft_target_on_time_bonus=deployment.config.train.path_soft_target_on_time_bonus,
        selection_pairwise_weight=deployment.config.train.selection_pairwise_weight,
        selection_pairwise_temperature=deployment.config.train.selection_pairwise_temperature,
        selection_pairwise_on_time_bonus=deployment.config.train.selection_pairwise_on_time_bonus,
        selection_pairwise_slack_bonus=deployment.config.train.selection_pairwise_slack_bonus,
        selection_pairwise_margin=deployment.config.train.selection_pairwise_margin,
        selection_feasible_target_weight=deployment.config.train.selection_feasible_target_weight,
        selection_slack_critical_weight=deployment.config.train.selection_slack_critical_weight,
        selection_slack_critical_scale=deployment.config.train.selection_slack_critical_scale,
        quantiles=deployment.config.model.quantile_levels,
        verifier_aux_last_k_steps=deployment.config.model.verifier_aux_last_k_steps,
        planner_cost_weight=deployment.config.train.planner_cost_weight,
        planner_on_time_weight=deployment.config.train.planner_on_time_weight,
    )
    rollout_metrics = evaluate_rollouts(
        deployment.model,
        dataset.episodes[: eval_config.train.rollout_eval_episodes],
        device=device,
        config=test_hidden_cfg,
    )
    return {
        "experiment": eval_config.name,
        "stage": eval_config.stage,
        "bucket": eval_config.bucket,
        "notes": eval_config.notes,
        "device": str(device),
        "deployment_variant": deployment.label,
        "deployment_rule": decision.rule,
        "deployment_reasons": "|".join(decision.reasons),
        "selected_model_experiment": deployment.config.name,
        "selected_checkpoint": str(deployment.checkpoint_path),
        "num_communities": test_hidden_cfg.num_communities,
        "tree_depth_min": test_hidden_cfg.tree_depth_min,
        "tree_depth_max": test_hidden_cfg.tree_depth_max,
        "branching_factor": test_hidden_cfg.branching_factor,
        "packets_max": test_hidden_cfg.packets_max,
        "queue_low": test_hidden_cfg.community_base_queue[0],
        "queue_high": test_hidden_cfg.community_base_queue[1],
        "queue_penalty": test_hidden_cfg.queue_penalty,
        "capacity_penalty": test_hidden_cfg.capacity_penalty,
        "urgency_penalty": test_hidden_cfg.urgency_penalty,
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
    axes[0].set_title("Conditional Next-Hop Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(df["experiment"], df["rollout_solved_rate"], color="#2ca02c")
    axes[1].set_title("Conditional Solved Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(df["experiment"], df["average_regret"], color="#d62728")
    axes[2].set_title("Conditional Average Regret")
    axes[2].tick_params(axis="x", rotation=20)

    axes[3].bar(df["experiment"], df["deadline_miss_rate"], color="#9467bd")
    axes[3].set_title("Conditional Deadline Miss Rate")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_plot, dpi=160)


def main() -> None:
    args = parse_args()
    first_eval_config = load_experiment_config(args.eval_configs[0])
    device = _resolve_device(args.device or first_eval_config.train.device)
    base = _load_model(args.base_config, args.base_checkpoint, device, "Multiheavy")
    verifier = _load_model(args.verifier_config, args.verifier_checkpoint, device, "Multiheavy+VerifierPathReranker")

    rows: list[dict[str, object]] = []
    for path in args.eval_configs:
        eval_config = load_experiment_config(path)
        decision = _policy_decision(args.policy, eval_config)
        deployment = verifier if decision.variant == "verifier" else base
        rows.append(_evaluate_config(eval_config, deployment, device=device, decision=decision))

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
