#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class AblationSpec:
    group: str
    label: str
    variant: str
    summary_csv: str


DEFAULT_SPECS = (
    AblationSpec(
        group="max_timing",
        label="global_max_after_mix",
        variant="prototype_negative_cleanup_max_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_negative_cleanup_max_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="max_timing",
        label="branchwise_max_before_mix",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="max_timing",
        label="branchwise_lift_before_mix",
        variant="prototype_branchwise_lift_negative_cleanup_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branchwise_lift_negative_cleanup_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="branch_source",
        label="shared_only",
        variant="prototype_shared_sharp_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_shared_sharp_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="branch_source",
        label="dual_only",
        variant="prototype_dual_sharp_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_dual_sharp_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="branch_source",
        label="shared_plus_dual_branchwise_max",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="negative_cleanup_source",
        label="fixed_negative_tail",
        variant="prototype_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="negative_cleanup_source",
        label="sharp_negative_tail",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_sharp_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="negative_cleanup_source",
        label="mass_negative_tail",
        variant="prototype_mass_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_mass_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="negative_cleanup_source",
        label="sharp_plus_mass_negative_tail",
        variant="prototype_sharp_mass_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_sharp_mass_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="negative_cleanup_source",
        label="branchwise_max_union",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="support_agree_mix",
        label="memory_agree_blend",
        variant="prototype_memory_agree_blend_hybrid",
        summary_csv="reports/plots/prototype_memory_agreement_blend_defer_summary.csv",
    ),
    AblationSpec(
        group="support_agree_mix",
        label="support_weighted_agree_mix",
        variant="prototype_support_weighted_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_support_weighted_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="support_agree_mix",
        label="sharp_negative_support_agree_mix",
        variant="prototype_sharp_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_sharp_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="support_agree_mix",
        label="branchwise_max_support_agree_mix",
        variant="prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="calibration",
        label="branch_calibrated_sharp",
        variant="prototype_branch_calibrated_sharp_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branch_calibrated_sharp_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="calibration",
        label="learned_gate_sharp",
        variant="prototype_learned_gate_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_learned_gate_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="calibration",
        label="branch_strength_sharp",
        variant="prototype_branch_strength_sharp_negative_tail_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branch_strength_sharp_negative_tail_support_agreement_mixture_defer_summary.csv",
    ),
    AblationSpec(
        group="calibration",
        label="branchwise_margin_max",
        variant="prototype_branchwise_margin_max_negative_cleanup_support_agree_mix_hybrid",
        summary_csv="reports/plots/prototype_branchwise_margin_max_negative_cleanup_support_agreement_mixture_defer_summary.csv",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0.75, 1.00, 1.50, 2.00],
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round13_branchwise_ablation",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _normalize_summary(path: str, *, variant: str) -> pd.DataFrame:
    summary = pd.read_csv(path)
    summary = summary.loc[summary["variant"] == variant].copy()
    if summary.empty:
        raise ValueError(f"No rows found for {variant!r} in {path}.")
    target_col = "target_match_rate" if "target_match_rate" in summary.columns else "system_target_match"
    summary = summary.rename(columns={target_col: "system_target_match"})
    return (
        summary.groupby(["variant", "budget_pct", "slice"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            defer_precision=("defer_precision", "mean"),
            system_target_match=("system_target_match", "mean"),
            mean_delta_regret=("mean_delta_regret", "mean"),
            mean_delta_miss=("mean_delta_miss", "mean"),
        )
    )


def _build_table(budgets: list[float]) -> pd.DataFrame:
    frames = []
    for spec in DEFAULT_SPECS:
        frame = _normalize_summary(spec.summary_csv, variant=spec.variant)
        frame = frame.loc[frame["budget_pct"].isin(budgets)].copy()
        frame["group"] = spec.group
        frame["label"] = spec.label
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    stable = merged.loc[merged["slice"] == "stable_positive_v2", ["group", "label", "budget_pct", "coverage"]].rename(
        columns={"coverage": "stable_positive_recall"}
    )
    hard = merged.loc[
        merged["slice"] == "hard_near_tie",
        ["group", "label", "budget_pct", "system_target_match", "mean_delta_regret"],
    ].rename(columns={"mean_delta_regret": "hard_mean_delta_regret"})
    overall = merged.loc[
        merged["slice"] == "overall",
        ["group", "label", "budget_pct", "mean_delta_regret", "mean_delta_miss"],
    ].rename(
        columns={
            "mean_delta_regret": "overall_mean_delta_regret",
            "mean_delta_miss": "overall_mean_delta_miss",
        }
    )
    table = stable.merge(hard, on=["group", "label", "budget_pct"]).merge(overall, on=["group", "label", "budget_pct"])
    return table.sort_values(["group", "budget_pct", "label"]).reset_index(drop=True)


def _group_best(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, budget_pct), frame in table.groupby(["group", "budget_pct"], sort=True):
        best = frame.sort_values(
            ["stable_positive_recall", "system_target_match", "overall_mean_delta_regret"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(
            {
                "group": group,
                "budget_pct": float(budget_pct),
                "best_label": best["label"],
                "stable_positive_recall": best["stable_positive_recall"],
                "system_target_match": best["system_target_match"],
                "overall_mean_delta_regret": best["overall_mean_delta_regret"],
            }
        )
    return pd.DataFrame(rows)


def _plot(table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    metrics = [
        ("max_timing", axes[0, 0], "overall_mean_delta_regret", "Max Timing", "Overall Mean Delta Regret"),
        ("branch_source", axes[0, 1], "stable_positive_recall", "Branch Source", "Stable-Positive-v2 Recall"),
        (
            "negative_cleanup_source",
            axes[1, 0],
            "system_target_match",
            "Negative Cleanup Source",
            "Hard Near-Tie Target Match",
        ),
        ("support_agree_mix", axes[1, 1], "overall_mean_delta_regret", "Support / Agree / Mix", "Overall Mean Delta Regret"),
        ("calibration", axes[2, 0], "overall_mean_delta_regret", "Calibration", "Overall Mean Delta Regret"),
    ]
    for group, axis, metric, title, ylabel in metrics:
        subset = table.loc[table["group"] == group]
        if subset.empty:
            continue
        pivot = subset.pivot(index="budget_pct", columns="label", values=metric)
        pivot.plot(ax=axis, marker="o")
        axis.set_title(title)
        axis.set_xlabel("Budget %")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes[2, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    table = _build_table(args.budgets)
    best = _group_best(table)

    table_path = output_prefix.with_name(output_prefix.name + "_table.csv")
    best_path = output_prefix.with_name(output_prefix.name + "_best.csv")
    plot_path = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    table.to_csv(table_path, index=False)
    best.to_csv(best_path, index=False)
    _plot(table, plot_path)
    json_path.write_text(
        json.dumps(
            {
                "budgets": args.budgets,
                "best": best.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(best.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
