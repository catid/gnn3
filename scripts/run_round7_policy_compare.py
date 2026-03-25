#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.hard_feasible import annotate_hard_feasible
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
    parser.add_argument("--base-selection-strategy", default="final")
    parser.add_argument("--candidate-selection-strategy", default="final")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round7_policy_compare",
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
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].bar(summary_df["suite"], summary_df["overall_disagreement"], color="#ff7f0e")
    axes[0].set_title("Overall Disagreement")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(summary_df["suite"], summary_df["large_gap_hard_feasible_disagreement"], color="#d62728")
    axes[1].set_title("Large-Gap Hard-Feasible Disagreement")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(summary_df["suite"], summary_df["improves_base_failures"], color="#2ca02c")
    axes[2].set_title("Improves Base Failures")
    axes[2].tick_params(axis="x", rotation=25)

    axes[3].bar(summary_df["suite"], summary_df["breaks_base_successes"], color="#9467bd")
    axes[3].set_title("Breaks Base Successes")
    axes[3].tick_params(axis="x", rotation=25)

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

    thresholds = None
    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )

        base_decision_df = pd.DataFrame(
            collect_decision_prediction_rows(
                base_model,
                list(dataset),
                device=base_device,
                suite=suite_config.name,
                selection_strategy=args.base_selection_strategy,
            )
        ).rename(
            columns={
                "predicted_next_hop": "base_predicted_next_hop",
                "target_match": "base_target_match",
                "predicted_continuation_gap": "base_continuation_gap",
                "strictly_suboptimal": "base_strictly_suboptimal",
            }
        )
        candidate_decision_df = pd.DataFrame(
            collect_decision_prediction_rows(
                candidate_model,
                list(dataset),
                device=candidate_device,
                suite=suite_config.name,
                selection_strategy=args.candidate_selection_strategy,
            )
        ).rename(
            columns={
                "predicted_next_hop": "candidate_predicted_next_hop",
                "target_match": "candidate_target_match",
                "predicted_continuation_gap": "candidate_continuation_gap",
                "strictly_suboptimal": "candidate_strictly_suboptimal",
            }
        )
        episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                base_model,
                dataset.episodes,
                device=base_device,
                config=hidden_cfg,
                suite=suite_config.name,
                selection_strategy=args.base_selection_strategy,
            )
        ).rename(
            columns={
                "regret": "base_regret",
                "deadline_miss": "base_deadline_miss",
                "next_hop_accuracy": "base_next_hop_accuracy",
                "solved": "base_solved",
            }
        )
        episode_df = episode_df.merge(
            pd.DataFrame(
                collect_episode_policy_rows(
                    candidate_model,
                    dataset.episodes,
                    device=candidate_device,
                    config=hidden_cfg,
                    suite=suite_config.name,
                    selection_strategy=args.candidate_selection_strategy,
                )
            ).rename(
                columns={
                    "regret": "candidate_regret",
                    "deadline_miss": "candidate_deadline_miss",
                    "next_hop_accuracy": "candidate_next_hop_accuracy",
                    "solved": "candidate_solved",
                }
            )[
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
        episode_frames.append(episode_df)

        merged_decision_df = base_decision_df.merge(
            candidate_decision_df[
                [
                    "suite",
                    "episode_index",
                    "decision_index",
                    "candidate_predicted_next_hop",
                    "candidate_target_match",
                    "candidate_continuation_gap",
                    "candidate_strictly_suboptimal",
                ]
            ],
            on=["suite", "episode_index", "decision_index"],
            how="inner",
        )
        annotate_df = merged_decision_df.assign(
            predicted_next_hop=merged_decision_df["base_predicted_next_hop"],
            target_match=merged_decision_df["base_target_match"],
            predicted_continuation_gap=merged_decision_df["base_continuation_gap"],
            strictly_suboptimal=merged_decision_df["base_strictly_suboptimal"],
        )
        annotated_df, thresholds = annotate_hard_feasible(
            annotate_df,
            episode_df.rename(
                columns={
                    "base_regret": "regret",
                    "base_deadline_miss": "deadline_miss",
                    "base_solved": "solved",
                }
            ),
            thresholds=thresholds,
        )
        annotated_df["action_agreement"] = annotated_df["base_predicted_next_hop"] == annotated_df["candidate_predicted_next_hop"]
        decision_frames.append(annotated_df)

        base_failures = episode_df["base_deadline_miss"] | (~episode_df["base_solved"])
        base_successes = (~episode_df["base_deadline_miss"]) & episode_df["base_solved"]
        decision_large_gap = annotated_df["large_gap_hard_feasible_case"]
        summary_rows.append(
            {
                "suite": suite_config.name,
                "overall_disagreement": float((~annotated_df["action_agreement"]).mean()),
                "hard_feasible_disagreement": (
                    float((~annotated_df.loc[annotated_df["hard_feasible_case"], "action_agreement"]).mean())
                    if annotated_df["hard_feasible_case"].any()
                    else 0.0
                ),
                "large_gap_hard_feasible_disagreement": (
                    float((~annotated_df.loc[decision_large_gap, "action_agreement"]).mean()) if decision_large_gap.any() else 0.0
                ),
                "improves_base_failures": (
                    float(
                        (
                            (episode_df.loc[base_failures, "candidate_regret"] + 1e-6 < episode_df.loc[base_failures, "base_regret"])
                            | (
                                episode_df.loc[base_failures, "base_deadline_miss"]
                                & ~episode_df.loc[base_failures, "candidate_deadline_miss"]
                            )
                        ).mean()
                    )
                    if base_failures.any()
                    else 0.0
                ),
                "breaks_base_successes": (
                    float(
                        (
                            (
                                ~episode_df.loc[base_successes, "base_deadline_miss"]
                                & episode_df.loc[base_successes, "candidate_deadline_miss"]
                            )
                            | (
                                ~episode_df.loc[base_successes, "candidate_solved"]
                                & episode_df.loc[base_successes, "base_solved"]
                            )
                        ).mean()
                    )
                    if base_successes.any()
                    else 0.0
                ),
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
    json_path.write_text(
        json.dumps(
            {
                "gap_threshold": thresholds.gap_threshold if thresholds is not None else None,
                "gap_ratio_threshold": thresholds.gap_ratio_threshold if thresholds is not None else None,
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
