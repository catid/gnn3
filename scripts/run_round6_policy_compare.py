#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.policy_analysis import collect_decision_prediction_rows, collect_episode_policy_rows
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round6_policy_compare",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, *, device_override: str | None = None) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def _plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(summary_df["suite"], summary_df["action_agreement"], color="#1f77b4")
    axes[0].set_title("Action Agreement")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(summary_df["suite"], summary_df["improves_base_failures"], color="#2ca02c")
    axes[1].set_title("Improves Base Failures")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(summary_df["suite"], summary_df["breaks_base_successes"], color="#d62728")
    axes[2].set_title("Breaks Base Successes")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    base_model, base_device = _load_model(args.base_config, args.base_checkpoint, device_override=args.device)
    candidate_model, candidate_device = _load_model(
        args.candidate_config,
        args.candidate_checkpoint,
        device_override=args.device,
    )

    decision_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )

        base_decision_df = pd.DataFrame(
            collect_decision_prediction_rows(base_model, list(dataset), device=base_device, suite=suite_config.name)
        ).rename(
            columns={
                "predicted_next_hop": "base_predicted_next_hop",
                "target_match": "base_target_match",
            }
        )
        cand_decision_df = pd.DataFrame(
            collect_decision_prediction_rows(candidate_model, list(dataset), device=candidate_device, suite=suite_config.name)
        ).rename(
            columns={
                "predicted_next_hop": "candidate_predicted_next_hop",
                "target_match": "candidate_target_match",
            }
        )
        decision_df = base_decision_df.merge(
            cand_decision_df[
                [
                    "suite",
                    "decision_index",
                    "candidate_predicted_next_hop",
                    "candidate_target_match",
                ]
            ],
            on=["suite", "decision_index"],
            how="inner",
        )
        decision_df["action_agreement"] = (
            decision_df["base_predicted_next_hop"] == decision_df["candidate_predicted_next_hop"]
        )
        decision_df["hard_feasible_case"] = decision_df["any_feasible_candidate"] & (
            decision_df["best_candidate_slack_ratio"] <= 0.05
        )
        decision_frames.append(decision_df)

        base_episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                base_model,
                dataset.episodes,
                device=base_device,
                config=hidden_cfg,
                suite=suite_config.name,
            )
        ).rename(
            columns={
                "regret": "base_regret",
                "deadline_miss": "base_deadline_miss",
                "next_hop_accuracy": "base_next_hop_accuracy",
                "solved": "base_solved",
            }
        )
        cand_episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                candidate_model,
                dataset.episodes,
                device=candidate_device,
                config=hidden_cfg,
                suite=suite_config.name,
            )
        ).rename(
            columns={
                "regret": "candidate_regret",
                "deadline_miss": "candidate_deadline_miss",
                "next_hop_accuracy": "candidate_next_hop_accuracy",
                "solved": "candidate_solved",
            }
        )
        episode_df = base_episode_df.merge(
            cand_episode_df[
                [
                    "suite",
                    "episode_index",
                    "candidate_regret",
                    "candidate_deadline_miss",
                    "candidate_next_hop_accuracy",
                    "candidate_solved",
                ]
            ],
            on=["suite", "episode_index"],
            how="inner",
        )
        episode_df["candidate_improves"] = (
            (episode_df["candidate_regret"] + 1e-6 < episode_df["base_regret"])
            | (episode_df["base_deadline_miss"] & ~episode_df["candidate_deadline_miss"])
        )
        episode_df["candidate_breaks"] = (
            (~episode_df["base_deadline_miss"] & episode_df["candidate_deadline_miss"])
            | (~episode_df["candidate_solved"] & episode_df["base_solved"])
        )
        episode_frames.append(episode_df)

        base_failures = episode_df["base_deadline_miss"] | (~episode_df["base_solved"])
        base_successes = (~episode_df["base_deadline_miss"]) & episode_df["base_solved"]
        hard_decisions = decision_df["hard_feasible_case"]
        summary_rows.append(
            {
                "suite": suite_config.name,
                "action_agreement": float(decision_df["action_agreement"].mean()),
                "overall_disagreement": float((~decision_df["action_agreement"]).mean()),
                "hard_feasible_disagreement": (
                    float((~decision_df.loc[hard_decisions, "action_agreement"]).mean()) if hard_decisions.any() else 0.0
                ),
                "improves_base_failures": (
                    float(episode_df.loc[base_failures, "candidate_improves"].mean()) if base_failures.any() else 0.0
                ),
                "breaks_base_successes": (
                    float(episode_df.loc[base_successes, "candidate_breaks"].mean()) if base_successes.any() else 0.0
                ),
                "base_failure_rate": float(base_failures.mean()),
                "candidate_failure_rate": float((episode_df["candidate_deadline_miss"] | ~episode_df["candidate_solved"]).mean()),
            }
        )

    decision_df = pd.concat(decision_frames, ignore_index=True)
    episode_df = pd.concat(episode_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("suite").reset_index(drop=True)

    decision_csv = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    episode_csv = output_prefix.with_name(output_prefix.name + "_episodes.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    plot_png = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    decision_df.to_csv(decision_csv, index=False)
    episode_df.to_csv(episode_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _plot_summary(summary_df, plot_png)
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
