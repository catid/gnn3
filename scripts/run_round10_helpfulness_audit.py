#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gnn3.eval.compute_helpfulness import (
    annotate_frontier_slices,
    collect_frontier_predictions,
    load_frontier_config,
    load_model,
    load_suite_records,
    merge_helpfulness_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--compute-model-config", required=True)
    parser.add_argument("--compute-checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--frontier-json", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument("--gap-epsilon", type=float, default=0.05)
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round10_helpfulness",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "hard_near_tie": frame["hard_near_tie_intersection_case"],
        "stable_near_tie": frame["stable_near_tie_case"],
        "high_headroom_near_tie": frame["high_headroom_near_tie_case"],
        "baseline_error_near_tie": frame["baseline_error_hard_near_tie_case"],
        "large_gap_control": frame["large_gap_hard_feasible_case"],
    }


def _summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for suite, suite_frame in frame.groupby("suite", sort=False):
        for slice_name, mask in _slice_map(suite_frame).items():
            target = suite_frame.loc[mask]
            helpful = target["helpful_compute"] if len(target) else pd.Series(dtype=bool)
            harmful = target["harmful_compute"] if len(target) else pd.Series(dtype=bool)
            neutral = target["neutral_compute"] if len(target) else pd.Series(dtype=bool)
            rows.append(
                {
                    "suite": suite,
                    "slice": slice_name,
                    "decisions": len(target),
                    "action_changed": float(target["action_changed"].mean()) if len(target) else 0.0,
                    "helpful_rate": float(helpful.mean()) if len(target) else 0.0,
                    "harmful_rate": float(harmful.mean()) if len(target) else 0.0,
                    "neutral_rate": float(neutral.mean()) if len(target) else 0.0,
                    "mean_delta_regret": float(target["delta_regret"].mean()) if len(target) else 0.0,
                    "p95_delta_regret": float(target["delta_regret"].quantile(0.95)) if len(target) else 0.0,
                    "mean_delta_miss": float(target["delta_miss"].mean()) if len(target) else 0.0,
                    "base_target_match": float(target["base_target_match"].mean()) if len(target) else 0.0,
                    "compute_target_match": float(target["compute_target_match"].mean()) if len(target) else 0.0,
                    "baseline_error_recovery": float(target["compute_recovers_baseline_error"].mean()) if len(target) else 0.0,
                    "baseline_success_break": float(target["compute_breaks_baseline_success"].mean()) if len(target) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _stability_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for suite, suite_frame in frame.groupby("suite", sort=False):
        source_groups = {
            "hard_near_tie": suite_frame["hard_near_tie_intersection_case"],
            "stable_near_tie": suite_frame["stable_near_tie_case"],
            "high_headroom_near_tie": suite_frame["high_headroom_near_tie_case"],
        }
        for source_name, mask in source_groups.items():
            target = suite_frame.loc[mask]
            if target.empty:
                rows.append(
                    {
                        "suite": suite,
                        "source_family": source_name,
                        "decisions": 0,
                        "helpful_share": 0.0,
                        "harmful_share": 0.0,
                        "mean_delta_regret": 0.0,
                        "action_change_share": 0.0,
                    }
                )
                continue
            rows.append(
                {
                    "suite": suite,
                    "source_family": source_name,
                    "decisions": len(target),
                    "helpful_share": float(target["helpful_compute"].mean()),
                    "harmful_share": float(target["harmful_compute"].mean()),
                    "mean_delta_regret": float(target["delta_regret"].mean()),
                    "action_change_share": float(target["action_changed"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot_helpfulness(summary_df: pd.DataFrame, output_path: Path) -> None:
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(hard["suite"], hard["helpful_rate"], color="#2ca02c", label="helpful")
    axes[0].bar(hard["suite"], hard["harmful_rate"], bottom=hard["helpful_rate"], color="#d62728", label="harmful")
    axes[0].set_title("Hard Near-Tie Helpful vs Harmful")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()

    axes[1].bar(hard["suite"], hard["mean_delta_regret"], color="#1f77b4")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Hard Near-Tie Mean Delta Regret")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frontier = load_frontier_config(args.frontier_json)
    base_model, device, _base_config = load_model(
        args.base_model_config,
        args.base_checkpoint,
        device_override=args.device,
    )
    compute_model, _compute_device, _compute_config = load_model(
        args.compute_model_config,
        args.compute_checkpoint,
        device_override=str(device),
    )

    all_frames: list[pd.DataFrame] = []
    episode_rows: list[dict[str, object]] = []
    suite_meta: list[dict[str, object]] = []
    for suite_config_path in args.suite_configs:
        suite_config, dataset, records = load_suite_records(suite_config_path)
        base_decisions, base_episodes = collect_frontier_predictions(
            base_model,
            dataset,
            records,
            device=device,
            suite_name=suite_config.name,
        )
        compute_decisions, compute_episodes = collect_frontier_predictions(
            compute_model,
            dataset,
            records,
            device=device,
            suite_name=suite_config.name,
        )
        annotated = annotate_frontier_slices(
            base_decisions,
            base_episodes,
            records,
            frontier=frontier,
        )
        merged = merge_helpfulness_labels(
            annotated,
            compute_decisions,
            gap_epsilon=args.gap_epsilon,
        )
        all_frames.append(merged)

        base_episode_frame = base_episodes.rename(
            columns={
                "regret": "base_regret",
                "deadline_miss": "base_deadline_miss",
                "next_hop_accuracy": "base_next_hop_accuracy",
            }
        )
        compute_episode_frame = compute_episodes.rename(
            columns={
                "regret": "compute_regret",
                "deadline_miss": "compute_deadline_miss",
                "next_hop_accuracy": "compute_next_hop_accuracy",
            }
        )
        episode_merge = base_episode_frame.merge(
            compute_episode_frame[
                [
                    "suite",
                    "episode_index",
                    "compute_regret",
                    "compute_deadline_miss",
                    "compute_next_hop_accuracy",
                ]
            ],
            on=["suite", "episode_index"],
            how="inner",
        )
        episode_merge["delta_regret"] = episode_merge["compute_regret"] - episode_merge["base_regret"]
        episode_merge["delta_deadline_miss"] = (
            episode_merge["compute_deadline_miss"].astype(int) - episode_merge["base_deadline_miss"].astype(int)
        )
        episode_rows.extend(episode_merge.to_dict(orient="records"))
        suite_meta.append(
            {
                "suite": suite_config.name,
                "config_path": str(suite_config_path),
                "decisions": len(records),
                "episodes": len(dataset.episodes),
            }
        )

    merged_df = pd.concat(all_frames, ignore_index=True)
    summary_df = _summary_rows(merged_df)
    stability_df = _stability_rows(merged_df)
    episodes_df = pd.DataFrame(episode_rows)
    prefix = output_prefix.with_name(output_prefix.name)
    merged_df.to_csv(prefix.with_name(prefix.name + "_decisions.csv"), index=False)
    summary_df.to_csv(prefix.with_name(prefix.name + "_summary.csv"), index=False)
    stability_df.to_csv(prefix.with_name(prefix.name + "_stability.csv"), index=False)
    episodes_df.to_csv(prefix.with_name(prefix.name + "_episodes.csv"), index=False)
    _plot_helpfulness(summary_df, prefix.with_name(prefix.name + "_summary.png"))
    prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "suite_meta": suite_meta,
                "summary": summary_df.to_dict(orient="records"),
                "stability": stability_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
