#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.oracle_analysis import audit_oracle_deadlines, audits_to_rows
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment configs to audit.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/oracle_deadline_audit_round4",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _suite_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("suite", as_index=False)
        .agg(
            packet_count=("packet_index", "count"),
            mean_packet_count=("packet_count", "mean"),
            mean_tree_depth=("max_tree_depth", "mean"),
            oracle_deadline_miss_rate=("oracle_deadline_missed", "mean"),
            on_time_feasible_fraction=("has_on_time_feasible_route", "mean"),
            oracle_initial_slack_p50=("oracle_initial_slack", "median"),
            oracle_initial_slack_p95=("oracle_initial_slack", lambda x: float(pd.Series(x).quantile(0.95))),
            oracle_cost_p50=("oracle_realized_cost", "median"),
            oracle_cost_p95=("oracle_realized_cost", lambda x: float(pd.Series(x).quantile(0.95))),
        )
        .sort_values("suite")
        .reset_index(drop=True)
    )
    episode_view = (
        df.groupby(["suite", "episode_index"], as_index=False)
        .agg(
            packet_count=("packet_index", "count"),
            all_packets_feasible=("has_on_time_feasible_route", "min"),
            any_packet_feasible=("has_on_time_feasible_route", "max"),
            oracle_missed_any=("oracle_deadline_missed", "max"),
        )
        .groupby("suite", as_index=False)
        .agg(
            fully_feasible_episode_fraction=("all_packets_feasible", "mean"),
            any_feasible_episode_fraction=("any_packet_feasible", "mean"),
            oracle_episode_miss_rate=("oracle_missed_any", "mean"),
        )
    )
    return summary.merge(episode_view, on="suite", how="left")


def _breakdown(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(["suite", group_col], as_index=False)
        .agg(
            packet_count=("packet_index", "count"),
            oracle_deadline_miss_rate=("oracle_deadline_missed", "mean"),
            on_time_feasible_fraction=("has_on_time_feasible_route", "mean"),
            oracle_initial_slack_mean=("oracle_initial_slack", "mean"),
            oracle_cost_mean=("oracle_realized_cost", "mean"),
        )
        .sort_values(["suite", group_col])
        .reset_index(drop=True)
    )


def _plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(summary_df["suite"], summary_df["on_time_feasible_fraction"], color="#2ca02c")
    axes[0].set_title("On-Time Feasible Fraction")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(summary_df["suite"], summary_df["oracle_deadline_miss_rate"], color="#d62728")
    axes[1].set_title("Oracle Deadline Miss Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(summary_df["suite"], summary_df["oracle_initial_slack_p50"], color="#1f77b4")
    axes[2].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[2].set_title("Median Oracle Initial Slack")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)


def _plot_slack_hist(df: pd.DataFrame, output_path: Path) -> None:
    suites = list(df["suite"].drop_duplicates())
    fig, axes = plt.subplots(len(suites), 1, figsize=(10, 3.2 * max(len(suites), 1)), squeeze=False)
    for axis, suite in zip(axes[:, 0], suites, strict=True):
        suite_df = df[df["suite"] == suite]
        axis.hist(suite_df["oracle_initial_slack"], bins=24, color="#1f77b4", alpha=0.8)
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_title(f"{suite} oracle initial slack")
        axis.set_xlabel("deadline - oracle_cost")
        axis.set_ylabel("packets")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    packet_rows: list[dict[str, object]] = []
    suite_meta: list[dict[str, object]] = []
    for config_path in args.configs:
        config = load_experiment_config(config_path)
        hidden_cfg = hidden_corridor_config_for_split(config.benchmark, args.split)
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=getattr(config.benchmark, f"{args.split}_episodes"),
            curriculum_levels=config.benchmark.curriculum_levels,
        )
        audits = audits_to_rows(audit_oracle_deadlines(dataset.episodes, config=hidden_cfg))
        for row in audits:
            row["suite"] = config.name
            row["config_path"] = config_path
            row["split"] = args.split
            row["manifest_hash"] = dataset.manifest()["manifest_hash"]
            packet_rows.append(row)
        suite_meta.append(
            {
                "suite": config.name,
                "config_path": config_path,
                "split": args.split,
                "manifest_hash": dataset.manifest()["manifest_hash"],
            }
        )

    packet_df = pd.DataFrame(packet_rows)
    summary_df = _suite_summary(packet_df).merge(pd.DataFrame(suite_meta), on="suite", how="left")
    traffic_df = _breakdown(packet_df, "packet_count")
    depth_df = _breakdown(packet_df, "max_tree_depth")

    packet_csv = output_prefix.with_name(output_prefix.name + "_packets.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    traffic_csv = output_prefix.with_name(output_prefix.name + "_traffic_breakdown.csv")
    depth_csv = output_prefix.with_name(output_prefix.name + "_depth_breakdown.csv")
    summary_png = output_prefix.with_name(output_prefix.name + "_summary.png")
    slack_png = output_prefix.with_name(output_prefix.name + "_slack_hist.png")
    json_path = output_prefix.with_suffix(".json")

    packet_df.to_csv(packet_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    traffic_df.to_csv(traffic_csv, index=False)
    depth_df.to_csv(depth_csv, index=False)
    _plot_summary(summary_df, summary_png)
    _plot_slack_hist(packet_df, slack_png)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_df.to_dict(orient="records"),
                "traffic_breakdown": traffic_df.to_dict(orient="records"),
                "depth_breakdown": depth_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
