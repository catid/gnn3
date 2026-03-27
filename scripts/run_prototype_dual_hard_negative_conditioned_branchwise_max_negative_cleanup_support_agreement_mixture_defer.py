#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gnn3.eval.precision_correction import (
    ULTRALOW_COVERAGE_BUDGETS,
    decision_augmented_features,
    margin_only_features,
    margin_regime_features,
    top_fraction_mask,
)
from gnn3.models.prototype_defer import (
    BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
)

KEY_COLS = ["suite", "episode_index", "decision_index"]
VARIANTS = (
    "prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix",
    "prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-caches", nargs="+", required=True)
    parser.add_argument("--eval-metadata", nargs="+", required=True)
    parser.add_argument("--teacher-bank-decisions-csv", required=True)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--prototype-dim", type=int, default=32)
    parser.add_argument("--positive-prototypes", type=int, default=8)
    parser.add_argument("--negative-prototypes", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--tail-margin", type=float, default=0.5)
    parser.add_argument("--tail-shrink-scale", type=float, default=2.0)
    parser.add_argument("--shared-tail-shrink-scale", type=float, default=2.0)
    parser.add_argument("--dual-tail-shrink-scale", type=float, default=2.0)
    parser.add_argument("--sharpness-center", type=float, default=0.75)
    parser.add_argument("--sharpness-scale", type=float, default=4.0)
    parser.add_argument("--min-kept-negatives", type=int, default=2)
    parser.add_argument("--search-budgets", nargs="+", type=float, default=[0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--coverage-budgets",
        nargs="+",
        type=float,
        default=list(ULTRALOW_COVERAGE_BUDGETS),
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/prototype_dual_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agreement_mixture_defer",
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
    merge_cols = [
        *KEY_COLS,
        "stable_positive_v2_case",
        "harmful_teacher_bank_case",
        "best_safe_teacher_gain",
        "committee_support",
    ]
    merged = meta.merge(teacher_bank[merge_cols], on=KEY_COLS, how="left")
    merged["stable_positive_v2_case"] = merged["stable_positive_v2_case"].fillna(False).astype(bool)
    merged["harmful_teacher_bank_case"] = merged["harmful_teacher_bank_case"].fillna(False).astype(bool)
    merged["best_safe_teacher_gain"] = merged["best_safe_teacher_gain"].fillna(0.0).astype(float)
    merged["committee_support"] = merged["committee_support"].fillna(0).astype(int)
    return merged


def _feature_sets(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    headroom = torch.as_tensor(metadata[["base_predicted_continuation_gap"]].to_numpy(copy=True), dtype=torch.float32)
    risk = torch.cat([margin_only_features(metadata), margin_regime_features(metadata), headroom], dim=1)
    return decision_augmented_features(cache, metadata).float(), risk.float()


def _normalize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _labels(meta: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(meta["stable_positive_v2_case"].astype(int).to_numpy(copy=True), dtype=torch.float32)


def _weights(meta: pd.DataFrame, target: torch.Tensor) -> torch.Tensor:
    neutral = (~meta["stable_positive_v2_case"].to_numpy(copy=True)) & (~meta["harmful_teacher_bank_case"].to_numpy(copy=True))
    weight = np.full(len(meta), 0.20, dtype=np.float32)
    weight[neutral] = 0.15
    weight[meta["harmful_teacher_bank_case"].to_numpy(copy=True)] = 2.0
    target_mask = target.numpy().astype(bool)
    weight[target_mask] = (
        2.5
        + meta["best_safe_teacher_gain"].clip(lower=0.0).to_numpy(copy=True)
        + 0.25 * meta["committee_support"].clip(lower=0, upper=4).to_numpy(copy=True)
    )[target_mask]
    return torch.as_tensor(weight, dtype=torch.float32)


def _fit_head(
    train_features: torch.Tensor,
    train_risk: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    hard_negative_mask: torch.Tensor,
    *,
    variant: str,
    epochs: int,
    lr: float,
    prototype_dim: int,
    positive_prototypes: int,
    negative_prototypes: int,
    hidden_dim: int,
    tail_margin: float,
    tail_shrink_scale: float,
    shared_tail_shrink_scale: float,
    dual_tail_shrink_scale: float,
    sharpness_center: float,
    sharpness_scale: float,
    device: torch.device,
) -> BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead:
    use_risk = variant.endswith("_hybrid")
    model = BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=train_features.size(1),
        risk_dim=train_risk.size(1),
        prototype_dim=prototype_dim,
        positive_prototypes=positive_prototypes,
        negative_prototypes=negative_prototypes,
        hidden_dim=hidden_dim,
        use_risk_branch=use_risk,
        tail_margin=tail_margin,
        tail_shrink_scale=tail_shrink_scale,
        shared_tail_shrink_scale=shared_tail_shrink_scale,
        dual_tail_shrink_scale=dual_tail_shrink_scale,
        sharpness_center=sharpness_center,
        sharpness_scale=sharpness_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = train_features.to(device)
    risk = train_risk.to(device)
    y = target.to(device)
    w = weights.to(device)
    hard_negative = hard_negative_mask.to(device)
    positive = y > 0.5
    positive_count = float(positive.sum().item())
    negative_count = float((~positive).sum().item())
    pos_weight = torch.tensor(negative_count / max(positive_count, 1.0), device=device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, risk if use_risk else None)
        per_row = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
        reg = model.regularization(x, positive_mask=positive, hard_negative_mask=hard_negative)
        support_reg = model.support_regularization()
        tail_reg = model.tail_regularization()
        loss = (per_row * w).mean() + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()
    return model.cpu()


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "stable_positive_v2": frame["stable_positive_v2_case"],
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
    selected = top_fraction_mask(scores, budget_pct) & (scores > 0.0)
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
                "target_match_rate": float(combined_target_match.mean()) if len(target) else 0.0,
                "base_target_match_rate": float(target["base_target_match"].mean()) if len(target) else 0.0,
                "compute_target_match_rate": float(target["compute_target_match"].mean()) if len(target) else 0.0,
                "mean_delta_regret": float(delta_regret.mean()) if len(target) else 0.0,
                "mean_delta_miss": float(delta_miss.mean()) if len(target) else 0.0,
            }
        )
    return pd.DataFrame(rows), decision_frame


def _search_metrics(summary: pd.DataFrame) -> dict[str, float]:
    def _slice_mean(slice_name: str, column: str) -> float:
        frame = summary[summary["slice"] == slice_name]
        if frame.empty:
            return 0.0
        return float(frame[column].mean())

    return {
        "avg_stable_recall": _slice_mean("stable_positive_v2", "coverage"),
        "avg_hard_match": _slice_mean("hard_near_tie", "target_match_rate"),
        "avg_hard_delta_regret": _slice_mean("hard_near_tie", "mean_delta_regret"),
        "avg_overall_delta_regret": _slice_mean("overall", "mean_delta_regret"),
        "avg_harmful_rate": _slice_mean("overall", "harmful_selection_rate"),
        "avg_false_positive_rate": _slice_mean("overall", "false_positive_rate"),
    }


def _shared_negative_logits(
    model: BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    features: torch.Tensor,
) -> torch.Tensor:
    shared_encoded = model.encode_shared(features)
    scale = model.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
    neg = model._shared_banks()[1]
    return scale * shared_encoded @ neg.T + model._bounded_support(model.shared_negative_support)


def _dual_negative_logits(
    model: BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    features: torch.Tensor,
) -> torch.Tensor:
    dual_neg_encoded = model.encode_dual_negative(features)
    scale = model.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
    neg = model._dual_banks()[1]
    return scale * dual_neg_encoded @ neg.T + model._bounded_support(model.dual_negative_support)


def _prototype_utility(
    neg_logits: torch.Tensor,
    *,
    positive_mask: torch.Tensor,
    hard_negative_mask: torch.Tensor,
) -> torch.Tensor:
    stable_score = neg_logits[positive_mask].mean(dim=0) if bool(positive_mask.any()) else torch.zeros(neg_logits.size(1))
    harmful_score = (
        neg_logits[hard_negative_mask].mean(dim=0) if bool(hard_negative_mask.any()) else torch.zeros(neg_logits.size(1))
    )
    return harmful_score - stable_score


def _search_keep_masks(
    base_model: BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    train_features: torch.Tensor,
    train_risk: torch.Tensor,
    train_meta: pd.DataFrame,
    *,
    args: argparse.Namespace,
    variant: str,
) -> tuple[HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead, pd.DataFrame]:
    use_risk = variant.endswith("_hybrid")
    positive_mask = torch.as_tensor(train_meta["stable_positive_v2_case"].to_numpy(copy=True), dtype=torch.bool)
    hard_negative_mask = torch.as_tensor(train_meta["harmful_teacher_bank_case"].to_numpy(copy=True), dtype=torch.bool)
    train_features = train_features.cpu()
    train_risk = train_risk.cpu()

    with torch.no_grad():
        dual_utility = _prototype_utility(
            _dual_negative_logits(base_model, train_features),
            positive_mask=positive_mask,
            hard_negative_mask=hard_negative_mask,
        )
    dual_order = torch.argsort(dual_utility, descending=True)

    search_rows: list[dict[str, object]] = []
    best_key: tuple[float, float, float, float, float, float] | None = None
    best_masks: tuple[torch.Tensor, torch.Tensor] | None = None

    keep_counts = range(args.min_kept_negatives, args.negative_prototypes + 1)
    shared_mask = torch.ones(args.negative_prototypes, dtype=torch.bool)
    for dual_keep_count in keep_counts:
        dual_mask = torch.zeros(args.negative_prototypes, dtype=torch.bool)
        dual_mask[dual_order[:dual_keep_count]] = True

        search_model = HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
            feature_dim=train_features.size(1),
            risk_dim=train_risk.size(1),
            prototype_dim=args.prototype_dim,
            positive_prototypes=args.positive_prototypes,
            negative_prototypes=args.negative_prototypes,
            hidden_dim=args.hidden_dim,
            use_risk_branch=use_risk,
            tail_margin=args.tail_margin,
            tail_shrink_scale=args.tail_shrink_scale,
            shared_tail_shrink_scale=args.shared_tail_shrink_scale,
            dual_tail_shrink_scale=args.dual_tail_shrink_scale,
            sharpness_center=args.sharpness_center,
            sharpness_scale=args.sharpness_scale,
        )
        search_model.load_state_dict(base_model.state_dict(), strict=False)
        search_model.set_negative_keep_masks(shared_keep_mask=shared_mask, dual_keep_mask=dual_mask)

        with torch.no_grad():
            scores = search_model(train_features, train_risk if use_risk else None).cpu().numpy()
        search_summary_frames: list[pd.DataFrame] = []
        for budget in args.search_budgets:
            summary, _ = _evaluate_budget(
                train_meta,
                scores,
                budget_pct=budget,
                variant=variant,
                split="train_search",
            )
            search_summary_frames.append(summary)
        search_summary = pd.concat(search_summary_frames, ignore_index=True)
        metrics = _search_metrics(search_summary)
        row = {
            "variant": variant,
            "shared_keep_count": int(shared_mask.sum().item()),
            "dual_keep_count": dual_keep_count,
            "shared_utility_min": float("nan"),
            "dual_utility_min": float(dual_utility[dual_mask].min().item()),
            **metrics,
        }
        search_rows.append(row)
        key = (
            metrics["avg_stable_recall"],
            metrics["avg_hard_match"],
            -metrics["avg_hard_delta_regret"],
            -metrics["avg_overall_delta_regret"],
            -metrics["avg_harmful_rate"],
            -metrics["avg_false_positive_rate"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_masks = (shared_mask.clone(), dual_mask.clone())

    assert best_masks is not None
    final_model = HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=train_features.size(1),
        risk_dim=train_risk.size(1),
        prototype_dim=args.prototype_dim,
        positive_prototypes=args.positive_prototypes,
        negative_prototypes=args.negative_prototypes,
        hidden_dim=args.hidden_dim,
        use_risk_branch=use_risk,
        tail_margin=args.tail_margin,
        tail_shrink_scale=args.tail_shrink_scale,
        shared_tail_shrink_scale=args.shared_tail_shrink_scale,
        dual_tail_shrink_scale=args.dual_tail_shrink_scale,
        sharpness_center=args.sharpness_center,
        sharpness_scale=args.sharpness_scale,
    )
    final_model.load_state_dict(base_model.state_dict(), strict=False)
    final_model.set_negative_keep_masks(shared_keep_mask=best_masks[0], dual_keep_mask=best_masks[1])
    search_frame = pd.DataFrame(search_rows).sort_values(
        by=[
            "avg_stable_recall",
            "avg_hard_match",
            "avg_hard_delta_regret",
            "avg_overall_delta_regret",
            "avg_harmful_rate",
            "avg_false_positive_rate",
        ],
        ascending=[False, False, True, True, True, True],
        ignore_index=True,
    )
    return final_model, search_frame


def _fit_and_score(
    train_cache: dict[str, torch.Tensor],
    train_meta: pd.DataFrame,
    eval_caches: list[dict[str, torch.Tensor]],
    eval_meta: list[pd.DataFrame],
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features_raw, train_risk = _feature_sets(train_cache, train_meta)
    train_features, _ = _normalize(train_features_raw, train_features_raw)
    train_target = _labels(train_meta)
    train_weights = _weights(train_meta, train_target)
    train_hard_negative = torch.as_tensor(train_meta["harmful_teacher_bank_case"].to_numpy(copy=True), dtype=torch.bool)

    summary_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    keep_frames: list[pd.DataFrame] = []
    search_frames: list[pd.DataFrame] = []
    device = torch.device(args.device)

    for variant in VARIANTS:
        base_model = _fit_head(
            train_features,
            train_risk,
            train_target,
            train_weights,
            train_hard_negative,
            variant=variant,
            epochs=args.epochs,
            lr=args.lr,
            prototype_dim=args.prototype_dim,
            positive_prototypes=args.positive_prototypes,
            negative_prototypes=args.negative_prototypes,
            hidden_dim=args.hidden_dim,
            tail_margin=args.tail_margin,
            tail_shrink_scale=args.tail_shrink_scale,
            shared_tail_shrink_scale=args.shared_tail_shrink_scale,
            dual_tail_shrink_scale=args.dual_tail_shrink_scale,
            sharpness_center=args.sharpness_center,
            sharpness_scale=args.sharpness_scale,
            device=device,
        )
        model, search_frame = _search_keep_masks(base_model, train_features, train_risk, train_meta, args=args, variant=variant)
        search_frames.append(search_frame)
        keep_frames.append(pd.DataFrame([{"variant": variant, **model.keep_summary()}]))
        use_risk = variant.endswith("_hybrid")
        for cache, meta in zip(eval_caches, eval_meta, strict=True):
            eval_features_raw, eval_risk = _feature_sets(cache, meta)
            _, eval_features = _normalize(train_features_raw, eval_features_raw)
            with torch.no_grad():
                scores = model(eval_features, eval_risk if use_risk else None).cpu().numpy()
            split_name = Path(meta.attrs["metadata_path"]).stem
            for budget in args.coverage_budgets:
                summary, decisions = _evaluate_budget(meta, scores, budget_pct=budget, variant=variant, split=split_name)
                summary_frames.append(summary)
                decision_frames.append(decisions)

    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(decision_frames, ignore_index=True),
        pd.concat(keep_frames, ignore_index=True),
        pd.concat(search_frames, ignore_index=True),
    )


def _plot(summary: pd.DataFrame, output_prefix: Path) -> None:
    pivot = summary[summary["slice"] == "hard_near_tie"].pivot_table(
        index="budget_pct",
        columns="variant",
        values="target_match_rate",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(ax=ax, marker="o")
    ax.set_xlabel("Budget (%)")
    ax.set_ylabel("Hard near-tie target match")
    ax.set_title("Dual-only hard-negative-conditioned branchwise-max negative-cleanup support agreement mixture")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_summary.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    train_cache, train_meta = _load(args.train_cache, args.train_metadata)
    teacher_bank = pd.read_csv(args.teacher_bank_decisions_csv)
    train_meta = _attach_teacher_bank(train_meta, teacher_bank)
    train_meta.attrs["metadata_path"] = args.train_metadata

    eval_caches: list[dict[str, torch.Tensor]] = []
    eval_meta: list[pd.DataFrame] = []
    for cache_path, metadata_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, metadata_path)
        meta = _attach_teacher_bank(meta, teacher_bank)
        meta.attrs["metadata_path"] = metadata_path
        eval_caches.append(cache)
        eval_meta.append(meta)

    summary, decisions, keep_stats, search_summary = _fit_and_score(train_cache, train_meta, eval_caches, eval_meta, args=args)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    decisions.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    keep_stats.to_csv(output_prefix.with_name(output_prefix.name + "_keep_stats.csv"), index=False)
    search_summary.to_csv(output_prefix.with_name(output_prefix.name + "_search_summary.csv"), index=False)
    _plot(summary, output_prefix)

    overall = (
        summary[summary["slice"] == "overall"]
        .groupby(["variant", "budget_pct"], as_index=False)
        .agg(
            coverage=("coverage", "mean"),
            target_match_rate=("target_match_rate", "mean"),
            mean_delta_regret=("mean_delta_regret", "mean"),
            mean_delta_miss=("mean_delta_miss", "mean"),
        )
    )
    output = {
        "variants": VARIANTS,
        "budgets": args.coverage_budgets,
        "overall": overall.to_dict(orient="records"),
        "selected_keep": keep_stats.to_dict(orient="records"),
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
