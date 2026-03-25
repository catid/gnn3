#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.oracle_analysis import audit_oracle_deadlines, audits_to_rows
from gnn3.eval.policy_analysis import collect_decision_prediction_rows, collect_episode_policy_rows
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, help="Training config used for the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--suite-configs", nargs="+", required=True, help="Configs defining the evaluation suites.")
    parser.add_argument("--device", help="Optional device override, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round6_regime_audit",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    parser.add_argument("--selection-strategy", default="final")
    parser.add_argument(
        "--include-decisions",
        action="store_true",
        help="Write per-decision policy rows. Disabled by default because the regime summary is episode-driven.",
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


def _episode_audit_frame(packet_df: pd.DataFrame) -> pd.DataFrame:
    packet_df = packet_df.copy()
    packet_df["initial_slack_ratio"] = packet_df["oracle_initial_slack"] / packet_df["packet_deadline"].clip(lower=1e-6)
    return (
        packet_df.groupby(["suite", "episode_index"], as_index=False)
        .agg(
            feasible_packet_fraction=("has_on_time_feasible_route", "mean"),
            any_feasible_packet=("has_on_time_feasible_route", "max"),
            all_feasible_packet=("has_on_time_feasible_route", "min"),
            oracle_packet_miss_rate=("oracle_deadline_missed", "mean"),
            min_initial_slack=("oracle_initial_slack", "min"),
            mean_initial_slack=("oracle_initial_slack", "mean"),
            min_initial_slack_ratio=("initial_slack_ratio", "min"),
            mean_initial_slack_ratio=("initial_slack_ratio", "mean"),
        )
        .sort_values(["suite", "episode_index"])
        .reset_index(drop=True)
    )


def _slack_band(value: float) -> str:
    if value < 0.02:
        return "critical"
    if value < 0.05:
        return "very_tight"
    if value < 0.10:
        return "tight"
    if value < 0.20:
        return "moderate"
    return "loose"


def _packet_band(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if value == 4:
        return "4"
    return "5+"


def _plot_regime_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    slack_df = summary_df[summary_df["group"] == "slack_band"].copy()
    axes[0].bar(slack_df["bucket"], slack_df["deadline_miss_rate"], color="#d62728")
    axes[0].set_title("Miss Rate by Slack Band")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)

    packet_df = summary_df[summary_df["group"] == "packet_band"].copy()
    axes[1].bar(packet_df["bucket"], packet_df["average_regret"], color="#1f77b4")
    axes[1].set_title("Average Regret by Packet Count")
    axes[1].tick_params(axis="x", rotation=25)

    depth_df = summary_df[summary_df["group"] == "depth_band"].copy()
    axes[2].bar(depth_df["bucket"], depth_df["p95_regret"], color="#9467bd")
    axes[2].set_title("p95 Regret by Depth")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _summarize_group(df: pd.DataFrame, group: str, column: str) -> pd.DataFrame:
    return (
        df.groupby([group], as_index=False)
        .agg(
            episodes=("episode_index", "count"),
            feasible_episode_fraction=("any_feasible_packet", "mean"),
            solved_rate=("solved", "mean"),
            average_regret=("regret", "mean"),
            p95_regret=("regret", lambda x: float(pd.Series(x).quantile(0.95))),
            deadline_miss_rate=("deadline_miss", "mean"),
        )
        .rename(columns={group: "bucket"})
        .assign(group=column)
    )


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)
    episode_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    packet_rows: list[dict[str, object]] = []
    suite_meta: list[dict[str, object]] = []

    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        episode_rows.extend(
            collect_episode_policy_rows(
                model,
                dataset.episodes,
                device=device,
                config=hidden_cfg,
                suite=suite_config.name,
                selection_strategy=args.selection_strategy,
            )
        )
        if args.include_decisions:
            decision_rows.extend(
                collect_decision_prediction_rows(
                    model,
                    list(dataset),
                    device=device,
                    suite=suite_config.name,
                    selection_strategy=args.selection_strategy,
                )
            )
        audits = audits_to_rows(audit_oracle_deadlines(dataset.episodes, config=hidden_cfg))
        for row in audits:
            row["suite"] = suite_config.name
            packet_rows.append(row)
        suite_meta.append(
            {
                "suite": suite_config.name,
                "config_path": suite_config_path,
                "manifest_hash": dataset.manifest()["manifest_hash"],
            }
        )

    episode_df = pd.DataFrame(episode_rows)
    decision_df = pd.DataFrame(decision_rows)
    packet_df = pd.DataFrame(packet_rows)
    audit_df = _episode_audit_frame(packet_df)
    episode_df = episode_df.merge(audit_df, on=["suite", "episode_index"], how="left")
    episode_df = episode_df.merge(pd.DataFrame(suite_meta), on="suite", how="left")

    mean_queue_ranks = episode_df["mean_queue"].rank(method="average", pct=True)
    hub_gap_ranks = episode_df["hub_asymmetry"].rank(method="average", pct=True)
    episode_df["slack_band"] = episode_df["min_initial_slack_ratio"].map(_slack_band)
    episode_df["packet_band"] = episode_df["packet_count"].map(_packet_band)
    episode_df["load_band"] = pd.cut(
        mean_queue_ranks,
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["low_load", "mid_load", "high_load"],
        include_lowest=True,
    ).astype(str)
    episode_df["depth_band"] = episode_df["max_tree_depth"].astype(str)
    episode_df["hub_gap_band"] = pd.cut(
        hub_gap_ranks,
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["low_gap", "mid_gap", "high_gap"],
        include_lowest=True,
    ).astype(str)

    summary_df = pd.concat(
        [
            _summarize_group(episode_df, "slack_band", "slack_band"),
            _summarize_group(episode_df, "packet_band", "packet_band"),
            _summarize_group(episode_df, "load_band", "load_band"),
            _summarize_group(episode_df, "depth_band", "depth_band"),
            _summarize_group(episode_df, "hub_gap_band", "hub_gap_band"),
        ],
        ignore_index=True,
    )

    episode_csv = output_prefix.with_name(output_prefix.name + "_episodes.csv")
    decision_csv = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    packet_csv = output_prefix.with_name(output_prefix.name + "_packets.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    plot_png = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    episode_df.to_csv(episode_csv, index=False)
    if not decision_df.empty:
        decision_df.to_csv(decision_csv, index=False)
    packet_df.to_csv(packet_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    _plot_regime_summary(summary_df, plot_png)
    json_path.write_text(
        json.dumps(
            {
                "suite_meta": suite_meta,
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
