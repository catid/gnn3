#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gnn3.eval.precision_correction import (
    annotate_stable_positive_pack,
    load_decision_frames,
    safe_rate,
    signature_overlap_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-csvs", nargs="+", required=True)
    parser.add_argument("--min-regret-gain", type=float, default=0.10)
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round11_teacher_bank",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _with_teacher_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    aliased = frame.copy()
    aliased["teacher_policy"] = "compute5"
    aliased["teacher_next_hop"] = aliased["compute_predicted_next_hop"]
    aliased["teacher_target_match"] = aliased["compute_target_match"]
    aliased["teacher_predicted_continuation_gap"] = aliased["compute_predicted_continuation_gap"]
    aliased["teacher_predicted_on_time"] = aliased["compute_predicted_on_time"]
    aliased["teacher_predicted_cost_to_go"] = aliased["compute_predicted_cost_to_go"]
    aliased["teacher_model_margin"] = aliased["compute_model_margin"]
    return aliased


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "hard_near_tie": frame["hard_near_tie_intersection_case"],
        "stable_near_tie": frame["stable_near_tie_case"],
        "high_headroom_near_tie": frame["high_headroom_near_tie_case"],
        "baseline_error_near_tie": frame["baseline_error_hard_near_tie_case"],
        "stable_positive_pack": frame["stable_positive_teacher_case"],
        "unstable_positive_pack": frame["unstable_positive_teacher_case"],
        "harmful_pack": frame["harmful_teacher_case"],
        "large_gap_control": frame["large_gap_hard_feasible_case"],
    }


def _summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (seed, suite), suite_frame in frame.groupby(["seed", "suite"], sort=True):
        for slice_name, mask in _slice_map(suite_frame).items():
            target = suite_frame.loc[mask]
            rows.append(
                {
                    "seed": int(seed),
                    "suite": suite,
                    "slice": slice_name,
                    "decisions": len(target),
                    "teacher_disagreement": safe_rate(target["action_changed"]),
                    "helpful_rate": safe_rate(target["helpful_compute"]),
                    "harmful_rate": safe_rate(target["harmful_compute"]),
                    "stable_positive_rate": safe_rate(target["stable_positive_teacher_case"]),
                    "unstable_positive_rate": safe_rate(target["unstable_positive_teacher_case"]),
                    "base_target_match": safe_rate(target["base_target_match"]),
                    "teacher_target_match": safe_rate(target["teacher_target_match"]),
                    "recovery_rate": safe_rate(target["compute_recovers_baseline_error"]),
                    "baseline_success_break": safe_rate(target["compute_breaks_baseline_success"]),
                    "mean_delta_regret": float(target["delta_regret"].mean()) if len(target) else 0.0,
                    "p95_delta_regret": float(target["delta_regret"].quantile(0.95)) if len(target) else 0.0,
                    "mean_delta_miss": float(target["delta_miss"].mean()) if len(target) else 0.0,
                    "mean_teacher_gain": float(target["teacher_regret_gain"].mean()) if len(target) else 0.0,
                }
            )
    overall = frame.copy()
    overall["suite"] = "all"
    overall["seed"] = -1
    rows.extend(_summary_rows_by_group(overall, group_cols=["suite"]))
    return pd.DataFrame(rows)


def _summary_rows_by_group(frame: pd.DataFrame, *, group_cols: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_values, group_frame in frame.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_payload = dict(zip(group_cols, group_values, strict=True))
        for slice_name, mask in _slice_map(group_frame).items():
            target = group_frame.loc[mask]
            rows.append(
                {
                    **group_payload,
                    "slice": slice_name,
                    "decisions": len(target),
                    "teacher_disagreement": safe_rate(target["action_changed"]),
                    "helpful_rate": safe_rate(target["helpful_compute"]),
                    "harmful_rate": safe_rate(target["harmful_compute"]),
                    "stable_positive_rate": safe_rate(target["stable_positive_teacher_case"]),
                    "unstable_positive_rate": safe_rate(target["unstable_positive_teacher_case"]),
                    "base_target_match": safe_rate(target["base_target_match"]),
                    "teacher_target_match": safe_rate(target["teacher_target_match"]),
                    "recovery_rate": safe_rate(target["compute_recovers_baseline_error"]),
                    "baseline_success_break": safe_rate(target["compute_breaks_baseline_success"]),
                    "mean_delta_regret": float(target["delta_regret"].mean()) if len(target) else 0.0,
                    "p95_delta_regret": float(target["delta_regret"].quantile(0.95)) if len(target) else 0.0,
                    "mean_delta_miss": float(target["delta_miss"].mean()) if len(target) else 0.0,
                    "mean_teacher_gain": float(target["teacher_regret_gain"].mean()) if len(target) else 0.0,
                }
            )
    return rows


def _seed_pack_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, seed_frame in frame.groupby("seed", sort=True):
        hard = seed_frame.loc[seed_frame["hard_near_tie_intersection_case"]]
        stable_positive = seed_frame.loc[seed_frame["stable_positive_teacher_case"]]
        rows.append(
            {
                "seed": int(seed),
                "decisions": len(seed_frame),
                "hard_near_tie_decisions": len(hard),
                "stable_positive_decisions": len(stable_positive),
                "stable_positive_share_of_hard_near_tie": len(stable_positive) / max(len(hard), 1),
                "stable_positive_mean_gain": float(stable_positive["teacher_regret_gain"].mean()) if len(stable_positive) else 0.0,
                "stable_positive_mean_miss_gain": float(stable_positive["teacher_miss_gain"].mean()) if len(stable_positive) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot_sizes(seed_summary: pd.DataFrame, overlap_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(seed_summary["seed"].astype(str), seed_summary["stable_positive_decisions"], color="#2ca02c")
    axes[0].set_title("Stable-Positive Decisions by Seed")
    axes[0].set_xlabel("Seed")

    if overlap_df.empty:
        axes[1].text(0.5, 0.5, "No overlap pairs", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        labels = overlap_df["left_group"] + " vs " + overlap_df["right_group"]
        axes[1].bar(labels, overlap_df["signature_jaccard"], color="#1f77b4")
        axes[1].set_title("Stable-Positive Signature Jaccard")
        axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    teacher_bank = load_decision_frames(args.decision_csvs)
    teacher_bank = _with_teacher_aliases(teacher_bank)
    teacher_bank = annotate_stable_positive_pack(
        teacher_bank,
        min_regret_gain=args.min_regret_gain,
    )

    summary_df = _summary_rows(teacher_bank)
    seed_summary_df = _seed_pack_rows(teacher_bank)
    seed_overlap_df = signature_overlap_rows(
        teacher_bank,
        subset_col="stable_positive_teacher_case",
        group_col="seed",
    )
    suite_overlap_df = signature_overlap_rows(
        teacher_bank,
        subset_col="stable_positive_teacher_case",
        group_col="suite",
    )

    decisions_csv = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    seed_summary_csv = output_prefix.with_name(output_prefix.name + "_seed_summary.csv")
    seed_overlap_csv = output_prefix.with_name(output_prefix.name + "_seed_overlap.csv")
    suite_overlap_csv = output_prefix.with_name(output_prefix.name + "_suite_overlap.csv")
    stable_manifest_csv = output_prefix.with_name(output_prefix.name + "_stable_positive_manifest.csv")
    harmful_manifest_csv = output_prefix.with_name(output_prefix.name + "_harmful_manifest.csv")
    png_path = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    teacher_bank.to_csv(decisions_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    seed_summary_df.to_csv(seed_summary_csv, index=False)
    seed_overlap_df.to_csv(seed_overlap_csv, index=False)
    suite_overlap_df.to_csv(suite_overlap_csv, index=False)
    teacher_bank.loc[teacher_bank["stable_positive_teacher_case"]].to_csv(stable_manifest_csv, index=False)
    teacher_bank.loc[teacher_bank["harmful_teacher_case"]].to_csv(harmful_manifest_csv, index=False)

    for seed, seed_frame in teacher_bank.groupby("seed", sort=True):
        seed_frame.to_csv(
            output_prefix.with_name(output_prefix.name + f"_seed{int(seed)}_decisions.csv"),
            index=False,
        )

    _plot_sizes(seed_summary_df, seed_overlap_df, png_path)
    json_path.write_text(
        json.dumps(
            {
                "decision_csvs": args.decision_csvs,
                "min_regret_gain": args.min_regret_gain,
                "seed_summary": seed_summary_df.to_dict(orient="records"),
                "seed_overlap": seed_overlap_df.to_dict(orient="records"),
                "suite_overlap": suite_overlap_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
