#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnn3.eval.precision_correction import top_fraction_mask
from gnn3.eval.round13_prototype import (
    KEY_COLS,
    ROUND13_FRONTIER_BUDGETS,
    aggregate_split_summary,
    evaluate_variant_scores,
    load_heldout_metadata,
    load_unique_scores,
)


@dataclass(frozen=True)
class VariantSpec:
    label: str
    variant: str
    decisions_csv: str
    reference_budget: float
    priority: float


ARCHIVED_VARIANTS = (
    VariantSpec(
        label="memory_agree",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/prototype_memory_agreement_blend_defer_decisions.csv",
        reference_budget=0.25,
        priority=4.0,
    ),
    VariantSpec(
        label="sharp_negative",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        reference_budget=0.75,
        priority=3.0,
    ),
    VariantSpec(
        label="negative_tail",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_negative_tail_support_agreement_mixture_defer_decisions.csv",
        reference_budget=1.00,
        priority=2.0,
    ),
    VariantSpec(
        label="branchwise_max",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        reference_budget=1.50,
        priority=1.0,
    ),
)

RERUN1_VARIANTS = tuple(
    VariantSpec(
        label=spec.label,
        variant=spec.variant,
        decisions_csv=f"reports/plots/round13_{Path(spec.decisions_csv).name.replace('prototype_', '').replace('_defer_decisions.csv', '')}_defer_decisions.csv",
        reference_budget=spec.reference_budget,
        priority=spec.priority,
    )
    for spec in ARCHIVED_VARIANTS
)

RERUN2_VARIANTS = (
    VariantSpec(
        label="memory_agree",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/round13_rerun2_memory_agreement_blend_defer_decisions.csv",
        reference_budget=0.25,
        priority=4.0,
    ),
    VariantSpec(
        label="sharp_negative",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        reference_budget=0.75,
        priority=3.0,
    ),
    VariantSpec(
        label="negative_tail",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_negative_tail_support_agreement_mixture_defer_decisions.csv",
        reference_budget=1.00,
        priority=2.0,
    ),
    VariantSpec(
        label="branchwise_max",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        reference_budget=1.50,
        priority=1.0,
    ),
)

SOURCES = {
    "archived": {
        "variants": ARCHIVED_VARIANTS,
        "frontier_summary_csv": "reports/plots/round13_prototype_frontier_summary.csv",
        "leaders_csv": "reports/plots/round13_prototype_frontier_leaders.csv",
    },
    "rerun1": {
        "variants": RERUN1_VARIANTS,
        "frontier_summary_csv": "reports/plots/round13_prototype_frontier_rerun1_summary.csv",
        "leaders_csv": "reports/plots/round13_prototype_frontier_rerun1_leaders.csv",
    },
    "rerun2": {
        "variants": RERUN2_VARIANTS,
        "frontier_summary_csv": "reports/plots/round13_prototype_frontier_rerun2_summary.csv",
        "leaders_csv": "reports/plots/round13_prototype_frontier_rerun2_leaders.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-metadata",
        nargs="+",
        default=[
            "reports/plots/round11_feature_cache_seed315_metadata.csv",
            "reports/plots/round11_feature_cache_seed316_metadata.csv",
        ],
    )
    parser.add_argument(
        "--teacher-bank-decisions-csv",
        default="reports/plots/round12_teacher_bank_decisions.csv",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(ROUND13_FRONTIER_BUDGETS),
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round13_hierarchical_dispatcher",
    )
    return parser.parse_args()


def _reference_threshold(scores: np.ndarray, budget_pct: float) -> float:
    values = np.asarray(scores, dtype=float)
    selected = top_fraction_mask(values, budget_pct) & (values > 0.0)
    if not selected.any():
        return float("inf")
    return float(np.min(values[selected]))


def _load_joined_scores(
    metadata_by_split: dict[str, pd.DataFrame],
    specs: tuple[VariantSpec, ...],
) -> dict[str, pd.DataFrame]:
    merged_by_split: dict[str, pd.DataFrame] = {}
    score_tables = {
        spec.label: load_unique_scores(spec.decisions_csv, variant=spec.variant).rename(columns={"score": f"score_{spec.label}"})
        for spec in specs
    }
    heldout_metadata = (
        pd.concat(
            [frame.assign(__split_name=split) for split, frame in metadata_by_split.items()],
            ignore_index=True,
        )
        if metadata_by_split
        else pd.DataFrame()
    )
    for split, metadata in metadata_by_split.items():
        merged = metadata.copy()
        for spec in specs:
            candidates = (split, f"{split}_metadata", split.removesuffix("_metadata"))
            scores = score_tables[spec.label]
            available = scores["split"].astype(str)
            if set(available.unique()) == {"heldout"}:
                heldout_scores = heldout_metadata.merge(
                    scores.loc[scores["split"] == "heldout", [*KEY_COLS, f"score_{spec.label}"]],
                    on=KEY_COLS,
                    how="inner",
                )
                split_scores = heldout_scores.loc[
                    heldout_scores["__split_name"] == split,
                    [*KEY_COLS, f"score_{spec.label}"],
                ].copy()
            else:
                matched_name = next((name for name in candidates if (available == name).any()), None)
                if matched_name is None:
                    raise ValueError(f"No scores found for split {split!r} and label {spec.label!r}.")
                split_scores = scores.loc[scores["split"] == matched_name, [*KEY_COLS, f"score_{spec.label}"]].copy()
            merged = merged.merge(split_scores, on=KEY_COLS, how="inner")
        merged_by_split[split] = merged
    return merged_by_split


def _dispatcher_scores(
    merged_by_split: dict[str, pd.DataFrame],
    specs: tuple[VariantSpec, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[pd.DataFrame] = []
    family_rows: list[pd.DataFrame] = []
    for split, frame in merged_by_split.items():
        candidate_scores: list[np.ndarray] = []
        candidate_labels: list[str] = []
        for spec in specs:
            col = f"score_{spec.label}"
            raw = frame[col].to_numpy(copy=True)
            threshold = _reference_threshold(raw, spec.reference_budget)
            if np.isinf(threshold):
                margin = np.full_like(raw, -1e9, dtype=float)
            else:
                margin = raw - threshold
                margin = np.where(margin >= 0.0, margin + spec.priority, -1e9)
            candidate_scores.append(margin)
            candidate_labels.append(spec.label)
        stacked = np.vstack(candidate_scores)
        best_index = np.argmax(stacked, axis=0)
        best_score = stacked[best_index, np.arange(stacked.shape[1])]
        best_score = np.where(best_score > -1e8, best_score, -1e9)
        chosen_label = np.array([candidate_labels[idx] for idx in best_index], dtype=object)

        score_frame = frame[KEY_COLS].copy()
        score_frame["split"] = split
        score_frame["score"] = best_score
        score_rows.append(score_frame)

        family_frame = frame[KEY_COLS].copy()
        family_frame["split"] = split
        family_frame["dispatcher_family"] = chosen_label
        family_frame["dispatcher_score"] = best_score
        family_rows.append(family_frame)
    return pd.concat(score_rows, ignore_index=True), pd.concat(family_rows, ignore_index=True)


def _leader_lookup(path: str) -> pd.DataFrame:
    leaders = pd.read_csv(path)
    return leaders[["budget_pct", "leader_variant", "stable_positive_recall", "hard_target_match", "overall_mean_delta_regret"]]


def _build_family_mix(
    dispatcher_decisions: pd.DataFrame,
    dispatcher_families: pd.DataFrame,
    *,
    budgets: list[float],
    source_name: str,
) -> pd.DataFrame:
    merged = dispatcher_decisions.merge(dispatcher_families, on=[*KEY_COLS, "split"], how="left")
    selected = merged.loc[merged["selected"]].copy()
    if selected.empty:
        return pd.DataFrame(columns=["source_name", "budget_pct", "dispatcher_family", "selected_count", "selected_share"])
    mix = (
        selected.groupby(["budget_pct", "dispatcher_family"], as_index=False)
        .size()
        .rename(columns={"size": "selected_count"})
    )
    totals = mix.groupby("budget_pct")["selected_count"].transform("sum").clip(lower=1)
    mix["selected_share"] = mix["selected_count"] / totals
    mix["source_name"] = source_name
    return mix


def _comparison_table(
    dispatcher_summary: pd.DataFrame,
    leader_table: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    stable = dispatcher_summary.loc[
        dispatcher_summary["slice"] == "stable_positive_v2",
        ["budget_pct", "stable_positive_recall"],
    ].rename(columns={"stable_positive_recall": "dispatcher_stable_positive_recall"})
    hard = dispatcher_summary.loc[
        dispatcher_summary["slice"] == "hard_near_tie",
        ["budget_pct", "system_target_match", "mean_delta_regret"],
    ].rename(
        columns={
            "system_target_match": "dispatcher_hard_target_match",
            "mean_delta_regret": "dispatcher_hard_mean_delta_regret",
        }
    )
    overall = dispatcher_summary.loc[
        dispatcher_summary["slice"] == "overall",
        ["budget_pct", "mean_delta_regret", "p95_delta_regret", "mean_delta_miss"],
    ].rename(
        columns={
            "mean_delta_regret": "dispatcher_overall_mean_delta_regret",
            "p95_delta_regret": "dispatcher_overall_p95_delta_regret",
            "mean_delta_miss": "dispatcher_overall_mean_delta_miss",
        }
    )
    merged = leader_table.merge(stable, on="budget_pct", how="left").merge(hard, on="budget_pct", how="left").merge(
        overall,
        on="budget_pct",
        how="left",
    )
    merged["source_name"] = source_name
    merged["stable_positive_recall_gain"] = (
        merged["dispatcher_stable_positive_recall"] - merged["stable_positive_recall"]
    )
    merged["hard_target_match_gain"] = merged["dispatcher_hard_target_match"] - merged["hard_target_match"]
    merged["overall_mean_delta_regret_gain"] = (
        merged["dispatcher_overall_mean_delta_regret"] - merged["overall_mean_delta_regret"]
    )
    return merged


def _plot(comparison: pd.DataFrame, mix: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for source_name, frame in comparison.groupby("source_name", sort=False):
        axes[0].plot(frame["budget_pct"], frame["stable_positive_recall_gain"], marker="o", label=f"{source_name}: recall gain")
        axes[0].plot(frame["budget_pct"], frame["hard_target_match_gain"], marker="s", linestyle="--", label=f"{source_name}: hard target gain")
        axes[1].plot(
            frame["budget_pct"],
            frame["overall_mean_delta_regret_gain"],
            marker="o",
            label=source_name,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Dispatcher vs Best Single: Target-Slice Gains")
    axes[0].set_xlabel("Budget %")
    axes[0].set_ylabel("Gain")
    axes[1].set_title("Dispatcher vs Best Single: Overall Mean Delta Regret Gain")
    axes[1].set_xlabel("Budget %")
    axes[1].set_ylabel("Dispatcher - Leader")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    if mix.empty:
        return
    mix_plot = output_path.with_name(output_path.stem.replace("_comparison", "_family_mix") + output_path.suffix)
    fig, axes = plt.subplots(len(mix["source_name"].unique()), 1, figsize=(12, 4 * len(mix["source_name"].unique())), squeeze=False)
    for axis, (source_name, frame) in zip(axes.ravel(), mix.groupby("source_name", sort=False), strict=False):
        pivot = frame.pivot(index="budget_pct", columns="dispatcher_family", values="selected_share").fillna(0.0)
        pivot.plot(kind="bar", stacked=True, ax=axis, colormap="tab20")
        axis.set_title(f"{source_name}: dispatcher family mix")
        axis.set_xlabel("Budget %")
        axis.set_ylabel("Selected share")
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(mix_plot, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metadata_by_split = load_heldout_metadata(
        args.eval_metadata,
        teacher_bank_decisions_csv=args.teacher_bank_decisions_csv,
    )

    summary_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    mix_frames: list[pd.DataFrame] = []
    static_rows: list[dict[str, object]] = []

    for source_name, config in SOURCES.items():
        joined = _load_joined_scores(metadata_by_split, config["variants"])
        dispatcher_scores, dispatcher_families = _dispatcher_scores(joined, config["variants"])
        per_split, dispatcher_decisions = evaluate_variant_scores(
            metadata_by_split,
            dispatcher_scores,
            family="round13_dispatcher",
            variant=f"round13_{source_name}_score_band_dispatcher",
            budgets=args.budgets,
        )
        aggregate = aggregate_split_summary(per_split)
        aggregate["source_name"] = source_name
        summary_frames.append(aggregate)

        leader_table = _leader_lookup(config["leaders_csv"])
        leader_table["source_name"] = source_name
        static_rows.extend(leader_table.to_dict(orient="records"))
        comparison_frames.append(_comparison_table(aggregate, leader_table, source_name=source_name))
        mix_frames.append(_build_family_mix(dispatcher_decisions, dispatcher_families, budgets=args.budgets, source_name=source_name))

    summary = pd.concat(summary_frames, ignore_index=True)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    family_mix = pd.concat(mix_frames, ignore_index=True)
    static_ladder = pd.DataFrame(static_rows)

    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    comparison_path = output_prefix.with_name(output_prefix.name + "_comparison.csv")
    family_mix_path = output_prefix.with_name(output_prefix.name + "_family_mix.csv")
    static_path = output_prefix.with_name(output_prefix.name + "_static_ladder.csv")
    plot_path = output_prefix.with_name(output_prefix.name + "_comparison.png")
    json_path = output_prefix.with_suffix(".json")

    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    family_mix.to_csv(family_mix_path, index=False)
    static_ladder.to_csv(static_path, index=False)
    _plot(comparison, family_mix, plot_path)
    json_path.write_text(
        json.dumps(
            {
                "sources": list(SOURCES),
                "reference_bands": {spec.label: spec.reference_budget for spec in ARCHIVED_VARIANTS},
                "comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(comparison.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
