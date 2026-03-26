#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gnn3.eval.precision_correction import (
    annotate_stable_positive_pack,
    build_source_signature,
    load_decision_frames,
    safe_rate,
    signature_overlap_rows,
    teacher_effect_labels,
)

KEY_COLS = ["suite", "episode_index", "decision_index"]
SAFE_TEACHERS = {
    "compute5",
    "gated_pairwise",
    "triggered_full_compute",
    "triggered_top2_compute",
    "fixed_final",
    "fixed_middle",
    "margin050",
    "margin100",
    "risktight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-decision-csvs", nargs="+", required=True)
    parser.add_argument("--selective-decisions-csv")
    parser.add_argument("--subset-distill-decisions-csv")
    parser.add_argument("--aux-policy-csvs", nargs="*", default=[])
    parser.add_argument("--min-regret-gain", type=float, default=0.25)
    parser.add_argument("--strict-regret-gain", type=float, default=0.50)
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round12_teacher_bank",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _output(path_prefix: Path, suffix: str) -> Path:
    return path_prefix.with_name(path_prefix.name + suffix)


def _load_base(paths: list[str], *, min_regret_gain: float) -> pd.DataFrame:
    base = load_decision_frames(paths)
    base = base.sort_values(["seed", "suite", "episode_index", "decision_index"]).reset_index(drop=True)
    base = annotate_stable_positive_pack(base, min_regret_gain=min_regret_gain)
    base["signature_fine"] = build_source_signature(base)
    base["signature_coarse"] = build_source_signature(base, include_suite=False, include_critical_packet=False)
    return base


def _teacher_rows_from_base(base: pd.DataFrame) -> pd.DataFrame:
    labels = teacher_effect_labels(
        base_target_match=base["base_target_match"].to_numpy(copy=True),
        teacher_target_match=base["compute_target_match"].to_numpy(copy=True),
        delta_regret=base["delta_regret"].to_numpy(copy=True),
        delta_miss=base["delta_miss"].to_numpy(copy=True),
        action_changed=base["action_changed"].to_numpy(copy=True),
        baseline_error_hard_near_tie_case=base["baseline_error_hard_near_tie_case"].to_numpy(copy=True),
    )
    return pd.DataFrame(
        {
            **{col: base[col].to_numpy(copy=True) for col in KEY_COLS},
            "seed": base["seed"].to_numpy(copy=True),
            "teacher_name": "compute5",
            "teacher_scope": "cross_seed_compute",
            "teacher_action": base["compute_predicted_next_hop"].to_numpy(copy=True),
            "teacher_target_match": base["compute_target_match"].to_numpy(copy=True),
            "teacher_delta_regret": base["delta_regret"].to_numpy(copy=True),
            "teacher_delta_miss": base["delta_miss"].to_numpy(copy=True),
            "teacher_regret_gain": base["teacher_regret_gain"].to_numpy(copy=True),
            "teacher_miss_gain": base["teacher_miss_gain"].to_numpy(copy=True),
            "action_changed": base["action_changed"].to_numpy(copy=True),
            "helpful": labels["helpful"].to_numpy(copy=True),
            "harmful": labels["harmful"].to_numpy(copy=True),
            "neutral": labels["neutral"].to_numpy(copy=True),
            "recovers_baseline_error": labels["recovers_baseline_error"].to_numpy(copy=True),
            "breaks_baseline_success": labels["breaks_baseline_success"].to_numpy(copy=True),
            "safe_teacher": True,
        }
    )


def _from_selective(base: pd.DataFrame, path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[pd.DataFrame] = []
    for policy, policy_frame in frame.groupby("policy", sort=True):
        teacher_name = str(policy)
        merged = base[[*KEY_COLS, "seed", "base_predicted_next_hop", "base_target_match", "baseline_error_hard_near_tie_case"]].merge(
            policy_frame[
                [*KEY_COLS, "predicted_next_hop_policy", "policy_target_match", "delta_regret_policy", "delta_miss_policy", "disagreement"]
            ],
            on=KEY_COLS,
            how="inner",
        )
        labels = teacher_effect_labels(
            base_target_match=merged["base_target_match"].to_numpy(copy=True),
            teacher_target_match=merged["policy_target_match"].to_numpy(copy=True),
            delta_regret=merged["delta_regret_policy"].to_numpy(copy=True),
            delta_miss=merged["delta_miss_policy"].to_numpy(copy=True),
            action_changed=merged["disagreement"].astype(bool).to_numpy(copy=True),
            baseline_error_hard_near_tie_case=merged["baseline_error_hard_near_tie_case"].to_numpy(copy=True),
        )
        rows.append(
            pd.DataFrame(
                {
                    **{col: merged[col].to_numpy(copy=True) for col in KEY_COLS},
                    "seed": merged["seed"].to_numpy(copy=True),
                    "teacher_name": teacher_name,
                    "teacher_scope": "selective_compute",
                    "teacher_action": merged["predicted_next_hop_policy"].to_numpy(copy=True),
                    "teacher_target_match": merged["policy_target_match"].to_numpy(copy=True),
                    "teacher_delta_regret": merged["delta_regret_policy"].to_numpy(copy=True),
                    "teacher_delta_miss": merged["delta_miss_policy"].to_numpy(copy=True),
                    "teacher_regret_gain": (-merged["delta_regret_policy"]).to_numpy(copy=True),
                    "teacher_miss_gain": (-merged["delta_miss_policy"]).to_numpy(copy=True),
                    "action_changed": merged["disagreement"].astype(bool).to_numpy(copy=True),
                    "helpful": labels["helpful"].to_numpy(copy=True),
                    "harmful": labels["harmful"].to_numpy(copy=True),
                    "neutral": labels["neutral"].to_numpy(copy=True),
                    "recovers_baseline_error": labels["recovers_baseline_error"].to_numpy(copy=True),
                    "breaks_baseline_success": labels["breaks_baseline_success"].to_numpy(copy=True),
                    "safe_teacher": teacher_name in SAFE_TEACHERS,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _from_subset_distill(base: pd.DataFrame, path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[pd.DataFrame] = []
    for variant, variant_frame in frame.groupby("variant", sort=True):
        merged = base[[*KEY_COLS, "seed", "base_predicted_next_hop", "base_target_match", "baseline_error_hard_near_tie_case"]].merge(
            variant_frame[
                [*KEY_COLS, "student_predicted_next_hop", "student_target_match", "delta_regret_student", "delta_miss_student", "disagreement"]
            ],
            on=KEY_COLS,
            how="inner",
        )
        labels = teacher_effect_labels(
            base_target_match=merged["base_target_match"].to_numpy(copy=True),
            teacher_target_match=merged["student_target_match"].to_numpy(copy=True),
            delta_regret=merged["delta_regret_student"].to_numpy(copy=True),
            delta_miss=merged["delta_miss_student"].to_numpy(copy=True),
            action_changed=merged["disagreement"].astype(bool).to_numpy(copy=True),
            baseline_error_hard_near_tie_case=merged["baseline_error_hard_near_tie_case"].to_numpy(copy=True),
        )
        rows.append(
            pd.DataFrame(
                {
                    **{col: merged[col].to_numpy(copy=True) for col in KEY_COLS},
                    "seed": merged["seed"].to_numpy(copy=True),
                    "teacher_name": str(variant),
                    "teacher_scope": "subset_distill",
                    "teacher_action": merged["student_predicted_next_hop"].to_numpy(copy=True),
                    "teacher_target_match": merged["student_target_match"].to_numpy(copy=True),
                    "teacher_delta_regret": merged["delta_regret_student"].to_numpy(copy=True),
                    "teacher_delta_miss": merged["delta_miss_student"].to_numpy(copy=True),
                    "teacher_regret_gain": (-merged["delta_regret_student"]).to_numpy(copy=True),
                    "teacher_miss_gain": (-merged["delta_miss_student"]).to_numpy(copy=True),
                    "action_changed": merged["disagreement"].astype(bool).to_numpy(copy=True),
                    "helpful": labels["helpful"].to_numpy(copy=True),
                    "harmful": labels["harmful"].to_numpy(copy=True),
                    "neutral": labels["neutral"].to_numpy(copy=True),
                    "recovers_baseline_error": labels["recovers_baseline_error"].to_numpy(copy=True),
                    "breaks_baseline_success": labels["breaks_baseline_success"].to_numpy(copy=True),
                    "safe_teacher": str(variant) in SAFE_TEACHERS,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _teacher_name_from_aux_path(path: str) -> str:
    stem = Path(path).stem
    prefix = "round9_compute_policy_seed314_deeper_packets6_"
    suffix = "_decisions"
    name = stem
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def _from_aux_policies(base: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        teacher_name = _teacher_name_from_aux_path(path)
        frame = pd.read_csv(path)
        merged = base[
            [*KEY_COLS, "seed", "base_predicted_next_hop", "base_target_match", "base_predicted_continuation_gap", "base_predicted_on_time", "baseline_error_hard_near_tie_case"]
        ].merge(
            frame[
                [*KEY_COLS, "predicted_next_hop", "target_match", "predicted_continuation_gap", "predicted_on_time", "disagreement"]
            ],
            on=KEY_COLS,
            how="inner",
        )
        delta_regret = merged["predicted_continuation_gap"] - merged["base_predicted_continuation_gap"]
        delta_miss = (~merged["predicted_on_time"].astype(bool)).astype(int) - (~merged["base_predicted_on_time"].astype(bool)).astype(int)
        labels = teacher_effect_labels(
            base_target_match=merged["base_target_match"].to_numpy(copy=True),
            teacher_target_match=merged["target_match"].to_numpy(copy=True),
            delta_regret=delta_regret.to_numpy(copy=True),
            delta_miss=delta_miss.to_numpy(copy=True),
            action_changed=merged["disagreement"].astype(bool).to_numpy(copy=True),
            baseline_error_hard_near_tie_case=merged["baseline_error_hard_near_tie_case"].to_numpy(copy=True),
        )
        rows.append(
            pd.DataFrame(
                {
                    **{col: merged[col].to_numpy(copy=True) for col in KEY_COLS},
                    "seed": merged["seed"].to_numpy(copy=True),
                    "teacher_name": teacher_name,
                    "teacher_scope": "seed314_aux_compute",
                    "teacher_action": merged["predicted_next_hop"].to_numpy(copy=True),
                    "teacher_target_match": merged["target_match"].to_numpy(copy=True),
                    "teacher_delta_regret": delta_regret.to_numpy(copy=True),
                    "teacher_delta_miss": delta_miss.to_numpy(copy=True),
                    "teacher_regret_gain": (-delta_regret).to_numpy(copy=True),
                    "teacher_miss_gain": (-delta_miss).to_numpy(copy=True),
                    "action_changed": merged["disagreement"].astype(bool).to_numpy(copy=True),
                    "helpful": labels["helpful"].to_numpy(copy=True),
                    "harmful": labels["harmful"].to_numpy(copy=True),
                    "neutral": labels["neutral"].to_numpy(copy=True),
                    "recovers_baseline_error": labels["recovers_baseline_error"].to_numpy(copy=True),
                    "breaks_baseline_success": labels["breaks_baseline_success"].to_numpy(copy=True),
                    "safe_teacher": teacher_name in SAFE_TEACHERS,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _aggregate_state(group: pd.DataFrame) -> pd.Series:
    available = len(group)
    positive = group.loc[group["helpful"]]
    harmful = group.loc[group["harmful"]]
    safe_group = group.loc[group["safe_teacher"]]
    safe_positive = safe_group.loc[safe_group["helpful"]]
    safe_harmful = safe_group.loc[safe_group["harmful"]]

    best_safe_name = "none"
    best_safe_action = -1
    best_safe_gain = 0.0
    best_safe_target_match = 0.0
    best_safe_helpful = False
    best_safe_delta_regret = 0.0
    best_safe_delta_miss = 0.0
    if not safe_positive.empty:
        best_safe = safe_positive.sort_values(
            ["teacher_regret_gain", "teacher_target_match", "teacher_miss_gain"],
            ascending=[False, False, False],
        ).iloc[0]
        best_safe_name = str(best_safe["teacher_name"])
        best_safe_action = int(best_safe["teacher_action"])
        best_safe_gain = float(best_safe["teacher_regret_gain"])
        best_safe_target_match = float(best_safe["teacher_target_match"])
        best_safe_helpful = True
        best_safe_delta_regret = float(best_safe["teacher_delta_regret"])
        best_safe_delta_miss = float(best_safe["teacher_delta_miss"])

    helpful_action_counts: Counter[int] = Counter(int(value) for value in safe_positive.loc[safe_positive["action_changed"], "teacher_action"])
    committee_action = -1
    committee_support = 0
    committee_gain = 0.0
    if helpful_action_counts:
        committee_action, committee_support = helpful_action_counts.most_common(1)[0]
        committee_rows = safe_positive.loc[safe_positive["teacher_action"] == committee_action]
        committee_gain = float(committee_rows["teacher_regret_gain"].mean()) if len(committee_rows) else 0.0

    positive_names = ",".join(sorted(positive["teacher_name"].astype(str).unique()))
    harmful_names = ",".join(sorted(harmful["teacher_name"].astype(str).unique()))
    safe_positive_names = ",".join(sorted(safe_positive["teacher_name"].astype(str).unique()))
    safe_harmful_names = ",".join(sorted(safe_harmful["teacher_name"].astype(str).unique()))

    return pd.Series(
        {
            "available_teacher_count": available,
            "positive_teacher_count": len(positive),
            "harmful_teacher_count": len(harmful),
            "safe_positive_teacher_count": len(safe_positive),
            "safe_harmful_teacher_count": len(safe_harmful),
            "positive_teacher_names": positive_names,
            "harmful_teacher_names": harmful_names,
            "safe_positive_teacher_names": safe_positive_names,
            "safe_harmful_teacher_names": safe_harmful_names,
            "best_safe_teacher_name": best_safe_name,
            "best_safe_teacher_action": best_safe_action,
            "best_safe_teacher_gain": best_safe_gain,
            "best_safe_teacher_target_match": best_safe_target_match,
            "best_safe_teacher_helpful": best_safe_helpful,
            "best_safe_teacher_delta_regret": best_safe_delta_regret,
            "best_safe_teacher_delta_miss": best_safe_delta_miss,
            "committee_action": committee_action,
            "committee_support": int(committee_support),
            "committee_mean_gain": committee_gain,
        }
    )


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "hard_near_tie": frame["hard_near_tie_intersection_case"],
        "stable_near_tie": frame["stable_near_tie_case"],
        "high_headroom_near_tie": frame["high_headroom_near_tie_case"],
        "baseline_error_near_tie": frame["baseline_error_hard_near_tie_case"],
        "round11_stable_positive": frame["stable_positive_teacher_case"],
        "stable_positive_v2": frame["stable_positive_v2_case"],
        "stable_positive_v2_committee": frame["stable_positive_v2_committee_case"],
        "stable_positive_v2_strict": frame["stable_positive_v2_strict_case"],
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
                    "base_target_match": safe_rate(target["base_target_match"]),
                    "round11_teacher_gain": float(target["teacher_regret_gain"].mean()) if len(target) else 0.0,
                    "v2_best_teacher_gain": float(target["best_safe_teacher_gain"].mean()) if len(target) else 0.0,
                    "v2_best_teacher_target_match": safe_rate(target["best_safe_teacher_target_match"]),
                    "positive_teacher_count_mean": float(target["positive_teacher_count"].mean()) if len(target) else 0.0,
                    "safe_harmful_teacher_rate": safe_rate(target["safe_harmful_teacher_count"] > 0),
                    "committee_support_mean": float(target["committee_support"].mean()) if len(target) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _teacher_summary_rows(teacher_rows: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keyed = base[
        [*KEY_COLS, "seed", "hard_near_tie_intersection_case", "stable_near_tie_case", "high_headroom_near_tie_case", "baseline_error_hard_near_tie_case", "large_gap_hard_feasible_case"]
    ]
    merged = keyed.merge(teacher_rows, on=[*KEY_COLS, "seed"], how="left")
    for (teacher_name, seed), group in merged.groupby(["teacher_name", "seed"], sort=True):
        for slice_name, mask in {
            "overall": pd.Series([True] * len(group), index=group.index),
            "hard_near_tie": group["hard_near_tie_intersection_case"],
            "high_headroom_near_tie": group["high_headroom_near_tie_case"],
            "baseline_error_near_tie": group["baseline_error_hard_near_tie_case"],
            "large_gap_control": group["large_gap_hard_feasible_case"],
        }.items():
            target = group.loc[mask]
            rows.append(
                {
                    "teacher_name": str(teacher_name),
                    "seed": int(seed),
                    "slice": slice_name,
                    "decisions": len(target),
                    "disagreement": safe_rate(target["action_changed"]),
                    "helpful_rate": safe_rate(target["helpful"]),
                    "harmful_rate": safe_rate(target["harmful"]),
                    "recovery_rate": safe_rate(target["recovers_baseline_error"]),
                    "break_rate": safe_rate(target["breaks_baseline_success"]),
                    "mean_teacher_gain": float(target["teacher_regret_gain"].mean()) if len(target) else 0.0,
                    "mean_teacher_miss_gain": float(target["teacher_miss_gain"].mean()) if len(target) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _pack_sensitivity_rows(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gain in thresholds:
        for support in (1, 2):
            mask = (
                frame["hard_near_tie_intersection_case"]
                & frame["stable_near_tie_case"]
                & frame["best_safe_teacher_helpful"]
                & (frame["best_safe_teacher_gain"] >= gain)
                & (frame["safe_harmful_teacher_count"] == 0)
                & (frame["committee_support"] >= support)
                & (frame["high_headroom_near_tie_case"] | frame["baseline_error_hard_near_tie_case"])
            )
            target = frame.loc[mask]
            rows.append(
                {
                    "min_regret_gain": gain,
                    "committee_support": support,
                    "decisions": len(target),
                    "share_of_hard_near_tie": len(target) / max(int(frame["hard_near_tie_intersection_case"].sum()), 1),
                    "mean_gain": float(target["best_safe_teacher_gain"].mean()) if len(target) else 0.0,
                    "coarse_signature_count": int(target["signature_coarse"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _plot(seed_summary: pd.DataFrame, teacher_summary: pd.DataFrame, sensitivity_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(seed_summary["seed"].astype(str), seed_summary["round11_pack"], color="#1f77b4", alpha=0.8, label="round11")
    axes[0].bar(seed_summary["seed"].astype(str), seed_summary["stable_positive_v2"], color="#2ca02c", alpha=0.8, label="v2")
    axes[0].set_title("Stable-Positive Size by Seed")
    axes[0].legend()

    overall = teacher_summary.loc[teacher_summary["slice"] == "overall"].copy()
    for teacher_name, group in overall.groupby("teacher_name", sort=False):
        axes[1].plot(group["seed"], group["helpful_rate"], marker="o", label=teacher_name)
    axes[1].set_title("Teacher Helpful Rate by Seed")
    axes[1].set_xlabel("Seed")
    axes[1].legend(fontsize=8)

    for support, group in sensitivity_df.groupby("committee_support", sort=True):
        axes[2].plot(group["min_regret_gain"], group["decisions"], marker="o", label=f"support>={support}")
    axes[2].set_title("Stable-Positive-v2 Sensitivity")
    axes[2].set_xlabel("Min regret gain")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    base = _load_base(args.compute_decision_csvs, min_regret_gain=0.10)
    teacher_frames = [_teacher_rows_from_base(base)]
    if args.selective_decisions_csv:
        selective = _from_selective(base, args.selective_decisions_csv)
        if not selective.empty:
            teacher_frames.append(selective)
    if args.subset_distill_decisions_csv:
        subset = _from_subset_distill(base, args.subset_distill_decisions_csv)
        if not subset.empty:
            teacher_frames.append(subset)
    if args.aux_policy_csvs:
        aux = _from_aux_policies(base, args.aux_policy_csvs)
        if not aux.empty:
            teacher_frames.append(aux)

    teacher_rows = pd.concat(teacher_frames, ignore_index=True).sort_values(["seed", "suite", "episode_index", "decision_index", "teacher_name"])
    support = teacher_rows.groupby(KEY_COLS, sort=False).apply(_aggregate_state).reset_index()
    decisions = base.merge(support, on=KEY_COLS, how="left")
    decisions["stable_positive_v2_case"] = (
        decisions["hard_near_tie_intersection_case"]
        & decisions["stable_near_tie_case"]
        & decisions["best_safe_teacher_helpful"].fillna(False)
        & (decisions["best_safe_teacher_gain"].fillna(0.0) >= args.min_regret_gain)
        & (decisions["safe_harmful_teacher_count"].fillna(0) == 0)
        & (decisions["high_headroom_near_tie_case"] | decisions["baseline_error_hard_near_tie_case"])
    )
    decisions["stable_positive_v2_committee_case"] = decisions["stable_positive_v2_case"] & (decisions["committee_support"].fillna(0) >= 2)
    decisions["stable_positive_v2_strict_case"] = decisions["stable_positive_v2_committee_case"] & (
        decisions["best_safe_teacher_gain"].fillna(0.0) >= args.strict_regret_gain
    )
    decisions["unstable_positive_v2_case"] = (
        decisions["hard_near_tie_intersection_case"]
        & decisions["best_safe_teacher_helpful"].fillna(False)
        & (~decisions["stable_positive_v2_case"])
    )
    decisions["harmful_teacher_bank_case"] = decisions["safe_harmful_teacher_count"].fillna(0) > 0

    summary_df = _summary_rows(decisions)
    teacher_summary_df = _teacher_summary_rows(teacher_rows, base)
    sensitivity_df = _pack_sensitivity_rows(decisions, [0.10, 0.25, 0.50, 0.75])

    overlap_rows: list[pd.DataFrame] = []
    for subset_col, subset_name in [
        ("stable_positive_teacher_case", "round11"),
        ("stable_positive_v2_case", "v2"),
        ("stable_positive_v2_committee_case", "v2_committee"),
        ("stable_positive_v2_strict_case", "v2_strict"),
    ]:
        fine_source = decisions.copy()
        fine_source["stable_positive_signature"] = fine_source["signature_fine"]
        fine_df = signature_overlap_rows(
            fine_source,
            subset_col=subset_col,
            group_col="seed",
        )
        fine_df["signature_type"] = "fine"
        fine_df["subset"] = subset_name
        coarse_source = decisions.copy()
        coarse_source["stable_positive_signature"] = coarse_source["signature_coarse"]
        coarse_df = signature_overlap_rows(
            coarse_source,
            subset_col=subset_col,
            group_col="seed",
        )
        coarse_df["signature_type"] = "coarse"
        coarse_df["subset"] = subset_name
        overlap_rows.extend([fine_df, coarse_df])
    overlap_df = pd.concat(overlap_rows, ignore_index=True) if overlap_rows else pd.DataFrame()

    seed_summary = (
        decisions.groupby("seed", as_index=False)
        .agg(
            decisions=("decision_index", "count"),
            hard_near_tie=("hard_near_tie_intersection_case", "sum"),
            round11_pack=("stable_positive_teacher_case", "sum"),
            stable_positive_v2=("stable_positive_v2_case", "sum"),
            stable_positive_v2_committee=("stable_positive_v2_committee_case", "sum"),
            stable_positive_v2_strict=("stable_positive_v2_strict_case", "sum"),
            harmful_teacher_bank=("harmful_teacher_bank_case", "sum"),
        )
        .sort_values("seed")
    )

    decisions_csv = _output(output_prefix, "_decisions.csv")
    teachers_csv = _output(output_prefix, "_teachers.csv")
    summary_csv = _output(output_prefix, "_summary.csv")
    teacher_summary_csv = _output(output_prefix, "_teacher_summary.csv")
    sensitivity_csv = _output(output_prefix, "_sensitivity.csv")
    overlap_csv = _output(output_prefix, "_seed_overlap.csv")
    seed_summary_csv = _output(output_prefix, "_seed_summary.csv")
    stable_manifest_csv = _output(output_prefix, "_stable_positive_v2_manifest.csv")
    committee_manifest_csv = _output(output_prefix, "_stable_positive_v2_committee_manifest.csv")
    harmful_manifest_csv = _output(output_prefix, "_harmful_teacher_bank_manifest.csv")
    png_path = _output(output_prefix, "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    decisions.to_csv(decisions_csv, index=False)
    teacher_rows.to_csv(teachers_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    teacher_summary_df.to_csv(teacher_summary_csv, index=False)
    sensitivity_df.to_csv(sensitivity_csv, index=False)
    overlap_df.to_csv(overlap_csv, index=False)
    seed_summary.to_csv(seed_summary_csv, index=False)
    decisions.loc[decisions["stable_positive_v2_case"]].to_csv(stable_manifest_csv, index=False)
    decisions.loc[decisions["stable_positive_v2_committee_case"]].to_csv(committee_manifest_csv, index=False)
    decisions.loc[decisions["harmful_teacher_bank_case"]].to_csv(harmful_manifest_csv, index=False)
    _plot(seed_summary, teacher_summary_df, sensitivity_df, png_path)

    json_path.write_text(
        json.dumps(
            {
                "compute_decision_csvs": args.compute_decision_csvs,
                "selective_decisions_csv": args.selective_decisions_csv,
                "subset_distill_decisions_csv": args.subset_distill_decisions_csv,
                "aux_policy_csvs": args.aux_policy_csvs,
                "min_regret_gain": args.min_regret_gain,
                "strict_regret_gain": args.strict_regret_gain,
                "seed_summary": seed_summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(seed_summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
