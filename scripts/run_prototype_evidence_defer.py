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
from gnn3.models.prototype_defer import EvidencePrototypeDeferHead

KEY_COLS = ["suite", "episode_index", "decision_index"]
VARIANTS = ("prototype_evidence", "prototype_evidence_hybrid")


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
    parser.add_argument("--evidence-topk", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--coverage-budgets",
        nargs="+",
        type=float,
        default=list(ULTRALOW_COVERAGE_BUDGETS),
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/prototype_evidence_defer",
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
    weight[target.numpy().astype(bool)] = (
        2.5
        + meta["best_safe_teacher_gain"].clip(lower=0.0).to_numpy(copy=True)
        + 0.25 * meta["committee_support"].clip(lower=0, upper=4).to_numpy(copy=True)
    )[target.numpy().astype(bool)]
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
    evidence_topk: int,
    hidden_dim: int,
    device: torch.device,
) -> EvidencePrototypeDeferHead:
    use_risk = variant != "prototype_evidence"
    model = EvidencePrototypeDeferHead(
        feature_dim=train_features.size(1),
        risk_dim=train_risk.size(1),
        prototype_dim=prototype_dim,
        positive_prototypes=positive_prototypes,
        negative_prototypes=negative_prototypes,
        evidence_topk=evidence_topk,
        hidden_dim=hidden_dim,
        use_risk_branch=use_risk,
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
        loss = (per_row * w).mean() + 0.1 * reg
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
    axes[0].set_title("Prototype-Evidence Precision vs Coverage")
    axes[1].set_title("Prototype-Evidence Hard Near-Tie Delta Regret")
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
    teacher_bank = teacher_bank[
        [
            *KEY_COLS,
            "stable_positive_v2_case",
            "harmful_teacher_bank_case",
            "best_safe_teacher_gain",
            "committee_support",
        ]
    ].copy()

    train_cache, train_meta = _load(args.train_cache, args.train_metadata)
    train_meta = _attach_teacher_bank(train_meta, teacher_bank)
    train_features, train_risk = _feature_sets(train_cache, train_meta)

    eval_payloads = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, meta_path)
        meta = _attach_teacher_bank(meta, teacher_bank)
        eval_payloads.append((Path(cache_path).stem, meta, *_feature_sets(cache, meta)))

    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    summary_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        train_norm, _ = _normalize(train_features, train_features)
        risk_train_norm, _ = _normalize(train_risk, train_risk)
        target = _labels(train_meta)
        weights = _weights(train_meta, target)
        hard_negative = torch.as_tensor(train_meta["harmful_teacher_bank_case"].to_numpy(copy=True), dtype=torch.bool)
        model = _fit_head(
            train_norm,
            risk_train_norm,
            target,
            weights,
            hard_negative,
            variant=variant,
            epochs=args.epochs,
            lr=args.lr,
            prototype_dim=args.prototype_dim,
            positive_prototypes=args.positive_prototypes,
            negative_prototypes=args.negative_prototypes,
            evidence_topk=args.evidence_topk,
            hidden_dim=args.hidden_dim,
            device=device,
        )
        for split_name, eval_meta, eval_features, eval_risk in eval_payloads:
            _, eval_norm = _normalize(train_features, eval_features)
            _, risk_norm_eval = _normalize(train_risk, eval_risk)
            with torch.no_grad():
                scores = model(eval_norm, risk_norm_eval if variant != "prototype_evidence" else None).squeeze(-1).numpy()
            for budget_pct in args.coverage_budgets:
                summary_df, decision_df = _evaluate_budget(
                    eval_meta,
                    scores,
                    budget_pct=budget_pct,
                    variant=variant,
                    split=split_name,
                )
                summary_rows.append(summary_df)
                decision_rows.append(decision_df)

    summary_df = pd.concat(summary_rows, ignore_index=True)
    decisions_df = pd.concat(decision_rows, ignore_index=True)
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    decisions_csv = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    summary_df.to_csv(summary_csv, index=False)
    decisions_df.to_csv(decisions_csv, index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "coverage_budgets": args.coverage_budgets,
                "train_cache": args.train_cache,
                "train_metadata": args.train_metadata,
                "eval_caches": args.eval_caches,
                "eval_metadata": args.eval_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
