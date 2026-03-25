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
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round7_hard_feasible_action_gap",
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


def _plot_gap_panels(decision_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].hist(decision_df["oracle_action_gap"], bins=30, color="#1f77b4", alpha=0.9)
    axes[0, 0].set_title("Oracle Action Gap Distribution")
    axes[0, 0].set_xlabel("Gap")

    hard_slice = decision_df[decision_df["large_gap_hard_feasible_case"]]
    if not hard_slice.empty:
        axes[0, 1].hist(hard_slice["predicted_continuation_gap"], bins=20, color="#d62728", alpha=0.9)
    axes[0, 1].set_title("Large-Gap Hard-Feasible Continuation Gap")
    axes[0, 1].set_xlabel("Baseline Continuation Gap")

    gap_df = summary_df[summary_df["group"] == "gap_bucket"]
    axes[1, 0].bar(gap_df["bucket"], gap_df["baseline_error_rate"], color="#ff7f0e")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Baseline Error Rate by Gap Bucket")

    hard_suite = (
        decision_df.groupby("suite", as_index=False)
        .agg(
            large_gap_hard_feasible_decisions=("large_gap_hard_feasible_case", "sum"),
            large_gap_hard_feasible_error_rate=(
                "baseline_error",
                lambda x: float(x[hard_slice.index.intersection(x.index)].mean()) if len(hard_slice.index.intersection(x.index)) else 0.0,
            ),
        )
        .sort_values("suite")
    )
    axes[1, 1].bar(hard_suite["suite"], hard_suite["large_gap_hard_feasible_decisions"], color="#2ca02c")
    axes[1, 1].set_title("Large-Gap Hard-Feasible Decisions by Suite")
    axes[1, 1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _summarize_group(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        frame.groupby(group, as_index=False)
        .agg(
            decisions=("decision_index", "count"),
            baseline_error_rate=("baseline_error", "mean"),
            strict_suboptimal_rate=("strictly_suboptimal", "mean"),
            predicted_miss_rate=("predicted_on_time", lambda x: float((~x.astype(bool)).mean())),
            mean_oracle_action_gap=("oracle_action_gap", "mean"),
            mean_continuation_gap=("predicted_continuation_gap", "mean"),
            large_gap_hard_feasible_fraction=("large_gap_hard_feasible_case", "mean"),
        )
        .rename(columns={group: "bucket"})
        .assign(group=group)
    )


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)
    decision_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    suite_rows: list[dict[str, object]] = []

    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                model,
                dataset.episodes,
                device=device,
                config=hidden_cfg,
                suite=suite_config.name,
            )
        )
        decision_df = pd.DataFrame(
            collect_decision_prediction_rows(
                model,
                list(dataset),
                device=device,
                suite=suite_config.name,
            )
        )
        decision_frames.append(decision_df)
        episode_frames.append(episode_df)
        suite_rows.append(
            {
                "suite": suite_config.name,
                "config_path": suite_config_path,
                "manifest_hash": dataset.manifest()["manifest_hash"],
                "episode_count": len(dataset.episodes),
                "decision_count": len(dataset),
            }
        )

    decision_df = pd.concat(decision_frames, ignore_index=True)
    episode_df = pd.concat(episode_frames, ignore_index=True)
    decision_df, thresholds = annotate_hard_feasible(decision_df, episode_df)

    summary_df = pd.concat(
        [
            _summarize_group(decision_df, "gap_bucket"),
            _summarize_group(decision_df, "slack_band"),
            _summarize_group(decision_df, "packet_band"),
            _summarize_group(decision_df, "load_band"),
            _summarize_group(decision_df, "depth_band"),
        ],
        ignore_index=True,
    )
    hard_manifest = decision_df.loc[
        decision_df["large_gap_hard_feasible_case"],
        [
            "suite",
            "episode_index",
            "decision_index",
            "packet_count",
            "max_tree_depth",
            "mean_queue",
            "slack_band",
            "load_band",
            "gap_bucket",
            "oracle_action_gap",
            "oracle_action_gap_ratio",
            "predicted_next_hop",
            "target_next_hop",
            "predicted_continuation_gap",
            "strictly_suboptimal",
        ],
    ].copy()
    mistakes_df = (
        decision_df.loc[decision_df["strictly_suboptimal"]]
        .groupby(["gap_bucket", "slack_band"], as_index=False)
        .agg(mistakes=("decision_index", "count"))
        .sort_values(["gap_bucket", "slack_band"])
    )

    decision_csv = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    hard_manifest_csv = output_prefix.with_name(output_prefix.name + "_large_gap_manifest.csv")
    mistakes_csv = output_prefix.with_name(output_prefix.name + "_mistakes.csv")
    suite_csv = output_prefix.with_name(output_prefix.name + "_suite_meta.csv")
    plot_png = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    decision_df.to_csv(decision_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    hard_manifest.to_csv(hard_manifest_csv, index=False)
    mistakes_df.to_csv(mistakes_csv, index=False)
    pd.DataFrame(suite_rows).to_csv(suite_csv, index=False)
    _plot_gap_panels(decision_df, summary_df, plot_png)
    json_path.write_text(
        json.dumps(
            {
                "suite_meta": suite_rows,
                "gap_thresholds": {
                    "gap_threshold": thresholds.gap_threshold,
                    "gap_ratio_threshold": thresholds.gap_ratio_threshold,
                },
                "summary": summary_df.to_dict(orient="records"),
                "large_gap_hard_feasible_count": int(hard_manifest.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
