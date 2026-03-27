#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gnn3.eval.precision_correction import build_source_signature, jaccard
from gnn3.eval.round13_prototype import (
    KEY_COLS,
    ROUND13_FRONTIER_BUDGETS,
    load_and_evaluate_variant,
    load_heldout_metadata,
)


@dataclass(frozen=True)
class SourceSpec:
    source_run: str
    model_family: str
    variant: str
    decisions_csv: str
    budgets: tuple[float, ...]


DEFAULT_SOURCES = (
    SourceSpec(
        source_run="archived",
        model_family="memory_agree",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/prototype_memory_agreement_blend_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="archived",
        model_family="sharp_negative",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="archived",
        model_family="negative_tail",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_negative_tail_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="archived",
        model_family="branchwise_max",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun1",
        model_family="memory_agree",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/round13_memory_agreement_blend_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun1",
        model_family="sharp_negative",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun1",
        model_family="negative_tail",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_negative_tail_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun1",
        model_family="branchwise_max",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun2",
        model_family="memory_agree",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/round13_rerun2_memory_agreement_blend_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun2",
        model_family="sharp_negative",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun2",
        model_family="negative_tail",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_negative_tail_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
    SourceSpec(
        source_run="rerun2",
        model_family="branchwise_max",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        budgets=ROUND13_FRONTIER_BUDGETS,
    ),
)


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
        "--output-prefix",
        default="reports/plots/round13_stable_positive_v3",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _combined_metadata(metadata_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(metadata_by_split.values(), ignore_index=True)
    dup_count = int(combined.duplicated(KEY_COLS).sum())
    if dup_count:
        raise ValueError(f"Held-out metadata keys are not unique across splits: {dup_count} duplicates.")
    return combined


def _selected_events(metadata_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    for spec in DEFAULT_SOURCES:
        _per_split, _aggregate, decisions = load_and_evaluate_variant(
            metadata_by_split,
            decisions_csv=spec.decisions_csv,
            variant=spec.variant,
            family=spec.source_run,
            budgets=spec.budgets,
        )
        chosen = decisions.loc[decisions["selected"]].copy()
        chosen["source_run"] = spec.source_run
        chosen["model_family"] = spec.model_family
        chosen["source_id"] = spec.source_run + ":" + spec.model_family
        selected_frames.append(chosen)
    return pd.concat(selected_frames, ignore_index=True)


def _classify_candidates(manifest: pd.DataFrame) -> pd.DataFrame:
    teacher_positive = (
        manifest["stable_positive_v2_case"].astype(bool)
        | manifest["stable_positive_v2_committee_case"].astype(bool)
        | (manifest["best_safe_teacher_gain"].astype(float) > 0.0)
        | (
            manifest["best_safe_teacher_target_match"].astype(bool)
            & (~manifest["base_target_match"].astype(bool))
        )
    )
    teacher_harmful = manifest["harmful_teacher_bank_case"].astype(bool)
    hard_near_tie = manifest["hard_near_tie_intersection_case"].astype(bool)
    candidate_pool = manifest["selection_events"].astype(int) > 0

    stable_positive_v3 = candidate_pool & hard_near_tie & (~teacher_harmful) & teacher_positive & (
        manifest["stable_positive_v2_case"].astype(bool)
        | (
            (manifest["model_family_count"].astype(int) >= 2)
            & (manifest["source_run_count"].astype(int) >= 2)
        )
        | (manifest["source_id_count"].astype(int) >= 3)
    )
    unstable_positive = candidate_pool & teacher_positive & (~stable_positive_v3)
    useful_hard_negative = candidate_pool & hard_near_tie & (
        teacher_harmful
        | ((manifest["best_safe_teacher_gain"].astype(float) <= 0.0) & (~teacher_positive))
    ) & (
        (manifest["source_run_count"].astype(int) >= 2)
        | (manifest["model_family_count"].astype(int) >= 2)
    )
    dead_noisy_positive = candidate_pool & (~stable_positive_v3) & (~unstable_positive) & (~useful_hard_negative)
    category = pd.Series("not_selected", index=manifest.index, dtype=object)
    category.loc[dead_noisy_positive] = "dead_noisy_positive"
    category.loc[useful_hard_negative] = "useful_hard_negative"
    category.loc[unstable_positive] = "unstable_positive"
    category.loc[stable_positive_v3] = "stable_positive_v3"

    return manifest.assign(
        candidate_pool_case=candidate_pool,
        stable_positive_v3_case=stable_positive_v3,
        unstable_positive_case=unstable_positive,
        useful_hard_negative_case=useful_hard_negative,
        dead_noisy_positive_case=dead_noisy_positive,
        v3_category=category,
    ).copy()


def _plot(summary_df: pd.DataFrame, source_family_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(summary_df["category"], summary_df["decisions"], color=["#2ca02c", "#ffbf00", "#d62728", "#7f7f7f"])
    axes[0].set_title("Round13 Stable-Positive-v3 Candidate Split")
    axes[0].tick_params(axis="x", rotation=20)

    pivot = source_family_df.pivot(index="model_family", columns="category", values="decisions").fillna(0.0)
    pivot.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Source Families by Mined Category")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _candidate_key_set(frame: pd.DataFrame) -> list[str]:
    unique = frame[KEY_COLS].drop_duplicates().astype(str)
    return sorted(unique["suite"] + ":" + unique["episode_index"] + ":" + unique["decision_index"])


def _pairwise_overlap(frame: pd.DataFrame, *, group_col: str) -> pd.DataFrame:
    groups = []
    for value, group in frame.groupby(group_col, sort=True):
        groups.append((str(value), _candidate_key_set(group)))
    rows: list[dict[str, object]] = []
    for index, (left_name, left_keys) in enumerate(groups):
        for right_name, right_keys in groups[index + 1 :]:
            rows.append(
                {
                    "group_col": group_col,
                    "left_group": left_name,
                    "right_group": right_name,
                    "left_size": len(left_keys),
                    "right_size": len(right_keys),
                    "candidate_jaccard": jaccard(left_keys, right_keys),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metadata_by_split = load_heldout_metadata(
        args.eval_metadata,
        teacher_bank_decisions_csv=args.teacher_bank_decisions_csv,
    )
    combined_meta = _combined_metadata(metadata_by_split)
    combined_meta["stable_positive_signature_round13"] = build_source_signature(combined_meta, include_suite=True)

    selected_events = _selected_events(metadata_by_split)
    selected_events = selected_events.merge(combined_meta, on=KEY_COLS, how="inner")
    selected_events["category"] = selected_events["source_run"] + ":" + selected_events["model_family"]

    aggregate = (
        selected_events.groupby(KEY_COLS, as_index=False)
        .agg(
            selection_events=("source_id", "count"),
            source_run_count=("source_run", "nunique"),
            model_family_count=("model_family", "nunique"),
            source_id_count=("source_id", "nunique"),
            budget_count=("budget_pct", "nunique"),
            max_budget_pct=("budget_pct", "max"),
            max_score=("score", "max"),
            mean_score=("score", "mean"),
            source_runs=("source_run", lambda x: "|".join(sorted(set(x)))),
            model_families=("model_family", lambda x: "|".join(sorted(set(x)))),
            source_ids=("source_id", lambda x: "|".join(sorted(set(x)))),
        )
    )
    manifest = combined_meta.merge(aggregate, on=KEY_COLS, how="left")
    numeric_fill = {
        "selection_events": 0,
        "source_run_count": 0,
        "model_family_count": 0,
        "source_id_count": 0,
        "budget_count": 0,
        "max_budget_pct": 0.0,
        "max_score": 0.0,
        "mean_score": 0.0,
    }
    for col, value in numeric_fill.items():
        manifest[col] = manifest[col].fillna(value)
    for col in ["source_runs", "model_families", "source_ids"]:
        manifest[col] = manifest[col].fillna("")
    manifest = _classify_candidates(manifest)

    category_summary = (
        manifest.loc[manifest["candidate_pool_case"]]
        .groupby("v3_category", as_index=False)
        .agg(
            decisions=("decision_index", "count"),
            mean_best_safe_teacher_gain=("best_safe_teacher_gain", "mean"),
            mean_selection_events=("selection_events", "mean"),
            mean_model_family_count=("model_family_count", "mean"),
            mean_source_run_count=("source_run_count", "mean"),
        )
        .rename(columns={"v3_category": "category"})
        .sort_values("category")
    )
    source_family_summary = (
        selected_events[[*KEY_COLS, "model_family"]]
        .drop_duplicates()
        .merge(
            manifest[[*KEY_COLS, "v3_category"]],
            on=KEY_COLS,
            how="inner",
        )
        .groupby(["model_family", "v3_category"], as_index=False)
        .agg(decisions=("decision_index", "count"))
        .rename(columns={"v3_category": "category"})
    )
    overlap_frame = pd.concat(
        [
            _pairwise_overlap(selected_events, group_col="model_family"),
            _pairwise_overlap(selected_events, group_col="source_run"),
            _pairwise_overlap(selected_events.assign(budget_band=selected_events["budget_pct"].map(lambda x: f"{x:.2f}")), group_col="budget_band"),
        ],
        ignore_index=True,
    )

    manifest_path = output_prefix.with_name(output_prefix.name + "_manifest.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    source_path = output_prefix.with_name(output_prefix.name + "_source_family_summary.csv")
    overlap_path = output_prefix.with_name(output_prefix.name + "_overlap.csv")
    selected_path = output_prefix.with_name(output_prefix.name + "_selected_events.csv")
    plot_path = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    manifest.to_csv(manifest_path, index=False)
    category_summary.to_csv(summary_path, index=False)
    source_family_summary.to_csv(source_path, index=False)
    overlap_frame.to_csv(overlap_path, index=False)
    selected_events.to_csv(selected_path, index=False)
    _plot(category_summary, source_family_summary, plot_path)
    json_path.write_text(
        json.dumps(
            {
                "sources": [spec.__dict__ for spec in DEFAULT_SOURCES],
                "summary": category_summary.to_dict(orient="records"),
                "stable_positive_v2_total": int(manifest["stable_positive_v2_case"].sum()),
                "stable_positive_v3_total": int(manifest["stable_positive_v3_case"].sum()),
                "new_v3_total": int((manifest["stable_positive_v3_case"] & ~manifest["stable_positive_v2_case"]).sum()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(category_summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
