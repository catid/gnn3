#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gnn3.eval.precision_correction import (
    ULTRALOW_COVERAGE_BUDGETS,
    decision_augmented_features,
    top_fraction_mask,
)

KEY_COLS = ["suite", "episode_index", "decision_index"]
VARIANTS = ("knn_v2", "prototype_committee", "margin_retrieval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-caches", nargs="+", required=True)
    parser.add_argument("--eval-metadata", nargs="+", required=True)
    parser.add_argument("--teacher-bank-decisions-csv", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--coverage-budgets",
        nargs="+",
        type=float,
        default=list(ULTRALOW_COVERAGE_BUDGETS),
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round12_retrieval_defer",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _load(cache_path: str, metadata_path: str) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    cache = torch.load(cache_path, map_location="cpu")
    meta = pd.read_csv(metadata_path)
    if int(cache["decision_features"].size(0)) != len(meta):
        raise ValueError(f"Cache rows do not match metadata rows for {cache_path}")
    return cache, meta


def _attach_teacher_bank(meta: pd.DataFrame, teacher_bank: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [*KEY_COLS, "stable_positive_v2_case", "stable_positive_v2_committee_case", "harmful_teacher_bank_case"]
    merged = meta.merge(teacher_bank[merge_cols], on=KEY_COLS, how="left")
    for col in ["stable_positive_v2_case", "stable_positive_v2_committee_case", "harmful_teacher_bank_case"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    return merged


def _normalize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _kmeans(points: torch.Tensor, clusters: int, *, iterations: int = 12) -> torch.Tensor:
    if points.size(0) <= clusters:
        return points.clone()
    centroids = points[:clusters].clone()
    for _ in range(iterations):
        distances = torch.cdist(points, centroids)
        assign = distances.argmin(dim=1)
        updated = []
        for index in range(clusters):
            members = points[assign == index]
            updated.append(members.mean(dim=0) if len(members) else centroids[index])
        centroids = torch.stack(updated, dim=0)
    return centroids


def _feature_space(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> torch.Tensor:
    return decision_augmented_features(cache, metadata).float()


def _scores(
    train_x: torch.Tensor,
    train_meta: pd.DataFrame,
    eval_x: torch.Tensor,
    eval_meta: pd.DataFrame,
    *,
    k: int,
) -> dict[str, np.ndarray]:
    train_norm, eval_norm = _normalize(train_x, eval_x)

    positive = train_meta["stable_positive_v2_case"].to_numpy(copy=True).astype(bool)
    committee = train_meta["stable_positive_v2_committee_case"].to_numpy(copy=True).astype(bool)
    if not positive.any():
        raise ValueError("No stable_positive_v2_case rows available for retrieval defer.")

    pos_points = train_norm[positive]
    k = max(1, min(k, int(pos_points.size(0))))
    pos_dist = torch.cdist(eval_norm, pos_points)
    knn_score = (-pos_dist.topk(k=k, dim=1, largest=False).values.mean(dim=1)).numpy()

    committee_points = train_norm[committee] if committee.any() else pos_points
    centroids = _kmeans(committee_points, min(4, int(committee_points.size(0))))
    prototype_score = (-torch.cdist(eval_norm, centroids).min(dim=1).values).numpy()

    margin = -eval_meta["base_model_margin"].to_numpy(copy=True)
    slack = -eval_meta["best_candidate_slack_ratio"].to_numpy(copy=True)
    hybrid = (
        (knn_score - knn_score.mean()) / max(knn_score.std(), 1e-6)
        + 0.75 * (margin - margin.mean()) / max(margin.std(), 1e-6)
        + 0.25 * (slack - slack.mean()) / max(slack.std(), 1e-6)
    )
    return {
        "knn_v2": knn_score,
        "prototype_committee": prototype_score,
        "margin_retrieval": hybrid,
    }


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "stable_positive_v2": frame["stable_positive_v2_case"],
        "stable_positive_v2_committee": frame["stable_positive_v2_committee_case"],
        "hard_near_tie": frame["hard_near_tie_intersection_case"],
        "high_headroom_near_tie": frame["high_headroom_near_tie_case"],
        "baseline_error_near_tie": frame["baseline_error_hard_near_tie_case"],
        "large_gap_control": frame["large_gap_hard_feasible_case"],
    }


def _evaluate_budget(
    meta: pd.DataFrame,
    scores: np.ndarray,
    *,
    budget_pct: float,
    variant: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = top_fraction_mask(scores, budget_pct)
    decision_frame = meta[KEY_COLS].copy()
    decision_frame["variant"] = variant
    decision_frame["split"] = split
    decision_frame["budget_pct"] = budget_pct
    decision_frame["score"] = scores
    decision_frame["selected"] = selected

    rows: list[dict[str, object]] = []
    for slice_name, mask in _slice_map(meta).items():
        target = meta.loc[mask].copy()
        idx = target.index.to_numpy()
        selected_target = selected[idx]
        selected_count = max(int(selected_target.sum()), 1)
        combined_target_match = np.where(
            selected_target,
            target["compute_target_match"].to_numpy(copy=True),
            target["base_target_match"].to_numpy(copy=True),
        )
        delta_regret = np.where(selected_target, target["delta_regret"].to_numpy(copy=True), 0.0)
        delta_miss = np.where(selected_target, target["delta_miss"].to_numpy(copy=True), 0.0)
        rows.append(
            {
                "variant": variant,
                "split": split,
                "budget_pct": budget_pct,
                "slice": slice_name,
                "decisions": len(target),
                "coverage": float(selected_target.mean()) if len(target) else 0.0,
                "defer_precision": float(
                    np.logical_and(selected_target, target["stable_positive_v2_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "false_positive_rate": float(
                    np.logical_and(selected_target, ~target["stable_positive_v2_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "harmful_selection_rate": float(
                    np.logical_and(selected_target, target["harmful_teacher_bank_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "correction_rate": float(
                    np.logical_and(selected_target, target["compute_recovers_baseline_error"].to_numpy(copy=True)).mean()
                )
                if len(target)
                else 0.0,
                "new_error_rate": float(
                    np.logical_and(selected_target, target["compute_breaks_baseline_success"].to_numpy(copy=True)).mean()
                )
                if len(target)
                else 0.0,
                "base_target_match": float(target["base_target_match"].mean()) if len(target) else 0.0,
                "system_target_match": float(combined_target_match.mean()) if len(target) else 0.0,
                "mean_delta_regret": float(delta_regret.mean()) if len(target) else 0.0,
                "mean_delta_miss": float(delta_miss.mean()) if len(target) else 0.0,
            }
        )
    return pd.DataFrame(rows), decision_frame


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    stable = summary_df.loc[summary_df["slice"] == "stable_positive_v2"].copy()
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for variant, group in stable.groupby("variant", sort=False):
        axes[0].plot(group["budget_pct"], group["defer_precision"], marker="o", label=variant)
    for variant, group in hard.groupby("variant", sort=False):
        axes[1].plot(group["budget_pct"], group["mean_delta_regret"], marker="o", label=variant)
    axes[0].set_title("Retrieval Precision vs Coverage")
    axes[1].set_title("Retrieval Hard Near-Tie Delta Regret")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    for axis in axes:
        axis.set_xlabel("Budget %")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.eval_caches) != len(args.eval_metadata):
        raise ValueError("--eval-caches and --eval-metadata must have the same length")
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    teacher_bank = pd.read_csv(args.teacher_bank_decisions_csv)
    teacher_bank = teacher_bank[[*KEY_COLS, "stable_positive_v2_case", "stable_positive_v2_committee_case", "harmful_teacher_bank_case"]].copy()

    train_cache, train_meta = _load(args.train_cache, args.train_metadata)
    train_meta = _attach_teacher_bank(train_meta, teacher_bank)
    train_x = _feature_space(train_cache, train_meta)

    summary_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, meta_path)
        meta = _attach_teacher_bank(meta, teacher_bank)
        eval_x = _feature_space(cache, meta)
        score_map = _scores(train_x, train_meta, eval_x, meta, k=args.k)
        split_name = Path(cache_path).stem
        for variant in VARIANTS:
            for budget_pct in args.coverage_budgets:
                summary_df, decision_df = _evaluate_budget(
                    meta,
                    score_map[variant],
                    budget_pct=budget_pct,
                    variant=variant,
                    split=split_name,
                )
                summary_rows.append(summary_df)
                decision_rows.append(decision_df)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    decisions_df = pd.concat(decision_rows, ignore_index=True)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    decisions_df.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "coverage_budgets": args.coverage_budgets,
                "k": args.k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
