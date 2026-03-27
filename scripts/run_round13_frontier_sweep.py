#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gnn3.eval.round13_prototype import (
    ROUND13_FRONTIER_BUDGETS,
    load_and_evaluate_variant,
    load_heldout_metadata,
)


@dataclass(frozen=True)
class VariantSpec:
    family: str
    variant: str
    decisions_csv: str
    label: str


ARCHIVED_VARIANTS = (
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/prototype_memory_agreement_blend_defer_decisions.csv",
        label="memory_agree_blend_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        label="sharp_negative_tail_support_agree_mix_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_negative_tail_support_agreement_mixture_defer_decisions.csv",
        label="negative_tail_support_agree_mix_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        label="branchwise_max_negative_cleanup_support_agree_mix_hybrid",
    ),
)

ROUND13_RERUN_VARIANTS = (
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/round13_memory_agreement_blend_defer_decisions.csv",
        label="memory_agree_blend_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        label="sharp_negative_tail_support_agree_mix_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_negative_tail_support_agreement_mixture_defer_decisions.csv",
        label="negative_tail_support_agree_mix_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        label="branchwise_max_negative_cleanup_support_agree_mix_hybrid",
    ),
)

ROUND13_RERUN2_VARIANTS = (
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_memory_agree_blend_hybrid",
        decisions_csv="reports/plots/round13_rerun2_memory_agreement_blend_defer_decisions.csv",
        label="memory_agree_blend_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_sharp_negative_tail_support_agreement_mixture_defer_decisions.csv",
        label="sharp_negative_tail_support_agree_mix_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_negative_tail_support_agreement_mixture_defer_decisions.csv",
        label="negative_tail_support_agree_mix_hybrid",
    ),
    VariantSpec(
        family="round13_shortlist",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        decisions_csv="reports/plots/round13_rerun2_branchwise_max_negative_cleanup_support_agreement_mixture_defer_decisions.csv",
        label="branchwise_max_negative_cleanup_support_agree_mix_hybrid",
    ),
)

DECISION_SOURCES = {
    "archived": ARCHIVED_VARIANTS,
    "round13": ROUND13_RERUN_VARIANTS,
    "round13_rerun2": ROUND13_RERUN2_VARIANTS,
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
        default="reports/plots/round13_prototype_frontier",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    parser.add_argument(
        "--decision-source",
        choices=sorted(DECISION_SOURCES),
        default="archived",
        help="Which shortlist decision exports to evaluate.",
    )
    parser.add_argument(
        "--write-decisions",
        action="store_true",
        help="Also write the concatenated per-budget decision export. Off by default to avoid large artifacts.",
    )
    return parser.parse_args()


def _pairwise_dominance(
    budget_panel: pd.DataFrame,
    *,
    budget_pct: float,
) -> pd.DataFrame:
    records = []
    variants = sorted(budget_panel["variant"].unique())
    stable = budget_panel.loc[budget_panel["slice"] == "stable_positive_v2"].set_index("variant")
    hard = budget_panel.loc[budget_panel["slice"] == "hard_near_tie"].set_index("variant")
    overall = budget_panel.loc[budget_panel["slice"] == "overall"].set_index("variant")
    large = budget_panel.loc[budget_panel["slice"] == "large_gap_control"].set_index("variant")
    for left in variants:
        for right in variants:
            if left == right:
                continue
            left_vector = (
                stable.at[left, "stable_positive_recall"],
                hard.at[left, "system_target_match"],
                -overall.at[left, "mean_delta_regret"],
                -overall.at[left, "p95_delta_regret"],
                -hard.at[left, "mean_delta_regret"],
                -large.at[left, "mean_delta_regret"],
            )
            right_vector = (
                stable.at[right, "stable_positive_recall"],
                hard.at[right, "system_target_match"],
                -overall.at[right, "mean_delta_regret"],
                -overall.at[right, "p95_delta_regret"],
                -hard.at[right, "mean_delta_regret"],
                -large.at[right, "mean_delta_regret"],
            )
            dominates = all(
                left_value >= right_value - 1e-12
                for left_value, right_value in zip(left_vector, right_vector, strict=True)
            ) and any(
                left_value > right_value + 1e-12
                for left_value, right_value in zip(left_vector, right_vector, strict=True)
            )
            records.append(
                {
                    "budget_pct": float(budget_pct),
                    "left_variant": left,
                    "right_variant": right,
                    "dominates": bool(dominates),
                    "left_stable_positive_recall": stable.at[left, "stable_positive_recall"],
                    "right_stable_positive_recall": stable.at[right, "stable_positive_recall"],
                    "left_hard_target_match": hard.at[left, "system_target_match"],
                    "right_hard_target_match": hard.at[right, "system_target_match"],
                    "left_overall_mean_delta_regret": overall.at[left, "mean_delta_regret"],
                    "right_overall_mean_delta_regret": overall.at[right, "mean_delta_regret"],
                }
            )
    return pd.DataFrame(records)


def _budget_leaders(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget_pct, group in summary.groupby("budget_pct", sort=True):
        stable = group.loc[group["slice"] == "stable_positive_v2", ["variant", "stable_positive_recall"]].set_index("variant")
        hard = group.loc[group["slice"] == "hard_near_tie", ["variant", "system_target_match", "mean_delta_regret"]].set_index(
            "variant"
        )
        overall = group.loc[group["slice"] == "overall", ["variant", "mean_delta_regret", "p95_delta_regret"]].set_index(
            "variant"
        )
        joined = stable.join(hard).join(overall, rsuffix="_overall")
        joined = joined.reset_index()
        efficient = joined.sort_values(
            ["stable_positive_recall", "system_target_match", "mean_delta_regret_overall"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(
            {
                "budget_pct": float(budget_pct),
                "leader_variant": efficient["variant"],
                "stable_positive_recall": efficient["stable_positive_recall"],
                "hard_target_match": efficient["system_target_match"],
                "hard_mean_delta_regret": efficient["mean_delta_regret"],
                "overall_mean_delta_regret": efficient["mean_delta_regret_overall"],
                "overall_p95_delta_regret": efficient["p95_delta_regret"],
            }
        )
    return pd.DataFrame(rows)


def _plot_frontier(summary: pd.DataFrame, output_path: Path) -> None:
    hard = summary.loc[
        summary["slice"] == "hard_near_tie",
        ["variant", "budget_pct", "system_target_match"],
    ].copy()
    stable = summary.loc[summary["slice"] == "stable_positive_v2"].copy()
    overall = summary.loc[summary["slice"] == "overall", ["variant", "budget_pct", "mean_delta_regret"]].rename(
        columns={"mean_delta_regret": "overall_mean_delta_regret"}
    )
    merged = hard.merge(stable[["variant", "budget_pct", "stable_positive_recall"]], on=["variant", "budget_pct"]).merge(
        overall, on=["variant", "budget_pct"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for variant, group in merged.groupby("variant", sort=False):
        axes[0].plot(group["budget_pct"], group["stable_positive_recall"], marker="o", label=variant)
        axes[1].plot(group["budget_pct"], group["system_target_match"], marker="o", label=variant)
    axes[0].set_title("Stable-Positive-v2 Recall by Matched Budget")
    axes[0].set_xlabel("Budget %")
    axes[0].set_ylabel("Recall")
    axes[1].set_title("Hard Near-Tie Target Match by Matched Budget")
    axes[1].set_xlabel("Budget %")
    axes[1].set_ylabel("Target Match")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dominance(summary: pd.DataFrame, output_path: Path) -> None:
    hard = summary.loc[summary["slice"] == "hard_near_tie", ["variant", "budget_pct", "system_target_match"]].copy()
    stable = summary.loc[summary["slice"] == "stable_positive_v2", ["variant", "budget_pct", "stable_positive_recall"]].copy()
    merged = hard.merge(stable, on=["variant", "budget_pct"])
    variants = list(merged["variant"].drop_duplicates())
    budgets = list(sorted(merged["budget_pct"].unique()))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for row_idx, variant in enumerate(variants):
        group = merged.loc[merged["variant"] == variant].set_index("budget_pct")
        for col_idx, budget in enumerate(budgets):
            if budget not in group.index:
                continue
            recall = group.at[budget, "stable_positive_recall"]
            hard_match = group.at[budget, "system_target_match"]
            ax.scatter(col_idx, row_idx, s=250 * max(recall, 0.05), c=hard_match, cmap="viridis", vmin=0.90, vmax=0.91)
            ax.text(col_idx, row_idx, f"{recall:.2f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(range(len(budgets)), [f"{budget:.2f}" for budget in budgets], rotation=25)
    ax.set_yticks(range(len(variants)), variants)
    ax.set_xlabel("Budget %")
    ax.set_title("Shortlist Dominance Map: bubble size = stable-positive recall, color = hard target match")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metadata_by_split = load_heldout_metadata(
        args.eval_metadata,
        teacher_bank_decisions_csv=args.teacher_bank_decisions_csv,
    )

    per_split_frames: list[pd.DataFrame] = []
    aggregate_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    variant_meta: list[dict[str, object]] = []
    for spec in DECISION_SOURCES[args.decision_source]:
        per_split, aggregate, decisions = load_and_evaluate_variant(
            metadata_by_split,
            decisions_csv=spec.decisions_csv,
            variant=spec.variant,
            family=spec.family,
            budgets=args.budgets,
        )
        per_split["label"] = spec.label
        aggregate["label"] = spec.label
        decisions["label"] = spec.label
        per_split_frames.append(per_split)
        aggregate_frames.append(aggregate)
        decision_frames.append(decisions)
        variant_meta.append(
            {
                "label": spec.label,
                "variant": spec.variant,
                "decisions_csv": spec.decisions_csv,
            }
        )

    per_split_summary = pd.concat(per_split_frames, ignore_index=True)
    aggregate_summary = pd.concat(aggregate_frames, ignore_index=True)
    decisions = pd.concat(decision_frames, ignore_index=True)

    dominance_frames = []
    for budget_pct, group in aggregate_summary.groupby("budget_pct", sort=True):
        dominance_frames.append(_pairwise_dominance(group, budget_pct=float(budget_pct)))
    dominance = pd.concat(dominance_frames, ignore_index=True)
    leaders = _budget_leaders(aggregate_summary)

    per_split_path = output_prefix.with_name(output_prefix.name + "_per_split_summary.csv")
    aggregate_path = output_prefix.with_name(output_prefix.name + "_summary.csv")
    dominance_path = output_prefix.with_name(output_prefix.name + "_dominance.csv")
    leaders_path = output_prefix.with_name(output_prefix.name + "_leaders.csv")
    frontier_plot = output_prefix.with_name(output_prefix.name + "_frontier.png")
    dominance_plot = output_prefix.with_name(output_prefix.name + "_dominance.png")
    json_path = output_prefix.with_suffix(".json")

    per_split_summary.to_csv(per_split_path, index=False)
    aggregate_summary.to_csv(aggregate_path, index=False)
    if args.write_decisions:
        decisions_path = output_prefix.with_name(output_prefix.name + "_decisions.csv")
        decisions.to_csv(decisions_path, index=False)
    dominance.to_csv(dominance_path, index=False)
    leaders.to_csv(leaders_path, index=False)
    _plot_frontier(aggregate_summary, frontier_plot)
    _plot_dominance(aggregate_summary, dominance_plot)
    json_path.write_text(
        json.dumps(
            {
                "budgets": args.budgets,
                "decision_source": args.decision_source,
                "variants": variant_meta,
                "leaders": leaders.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(leaders.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
