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
    DEFAULT_COVERAGE_BUDGETS,
    candidate_pair_features,
    decision_augmented_features,
    top2_candidate_indices,
    top_fraction_mask,
)

VARIANTS = ("frozen", "candidate_conditioned")
SCOPES = ("broad", "narrow")


class ComparatorHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-caches", nargs="+", required=True)
    parser.add_argument("--eval-metadata", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--coverage-budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_COVERAGE_BUDGETS),
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round11_top2_comparator",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _load(cache_path: str, metadata_path: str) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    cache = torch.load(cache_path, map_location="cpu")
    meta = pd.read_csv(metadata_path)
    if int(cache["decision_features"].size(0)) != len(meta):
        raise ValueError(f"Cache rows do not match metadata rows for {cache_path}")
    return cache, meta


def _feature_sets(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> dict[str, torch.Tensor]:
    valid_mask = cache["candidate_mask"].bool()
    first, second = top2_candidate_indices(cache["base_selection_scores"], valid_mask)
    batch_index = torch.arange(valid_mask.size(0))
    scalar = torch.stack(
        [
            cache["base_selection_scores"][batch_index, first].float(),
            cache["base_selection_scores"][batch_index, second].float(),
            cache["candidate_cost_to_go"][batch_index, first].float(),
            cache["candidate_cost_to_go"][batch_index, second].float(),
            cache["candidate_on_time"][batch_index, first].float(),
            cache["candidate_on_time"][batch_index, second].float(),
        ],
        dim=1,
    )
    return {
        "frozen": torch.cat([decision_augmented_features(cache, metadata), scalar], dim=1),
        "candidate_conditioned": candidate_pair_features(cache, metadata),
    }


def _labels_and_weights(cache: dict[str, torch.Tensor], metadata: pd.DataFrame, *, scope: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = cache["candidate_mask"].bool()
    top1, top2 = top2_candidate_indices(cache["base_selection_scores"], valid_mask)
    teacher_action = torch.as_tensor(metadata["teacher_next_hop"].to_numpy(copy=True), dtype=torch.long)
    top2_available = top2 != top1
    stable_positive = torch.as_tensor(metadata["stable_positive_teacher_case"].astype(int).to_numpy(copy=True), dtype=torch.bool)
    harmful = torch.as_tensor(metadata["harmful_teacher_case"].astype(int).to_numpy(copy=True), dtype=torch.bool)
    base_correct = torch.as_tensor(metadata["base_target_match"].astype(int).to_numpy(copy=True), dtype=torch.bool)
    high_value = torch.as_tensor(
        (
            metadata["high_headroom_near_tie_case"].astype(bool)
            | metadata["baseline_error_hard_near_tie_case"].astype(bool)
        ).to_numpy(copy=True),
        dtype=torch.bool,
    )

    labels = torch.full((len(metadata),), fill_value=2, dtype=torch.long)
    flip_label = stable_positive & top2_available & (teacher_action == top2)
    keep_label = base_correct | (harmful & top2_available & (teacher_action == top2))
    labels[keep_label] = 0
    labels[flip_label] = 1

    weights = torch.ones((len(metadata),), dtype=torch.float32)
    if scope == "narrow":
        weights = torch.where(high_value | flip_label | harmful, torch.ones_like(weights), 0.2 * torch.ones_like(weights))
    top1_np = top1.cpu()
    top2_np = top2.cpu()
    return labels, weights, top1_np, top2_np


def _normalize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _fit_model(
    train_x: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> ComparatorHead:
    model = ComparatorHead(train_x.size(1), hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = train_x.to(device)
    y = labels.to(device)
    w = weights.to(device)
    class_counts = torch.bincount(labels, minlength=3).float().clamp(min=1.0)
    class_weight = (class_counts.sum() / class_counts).to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        per_row = F.cross_entropy(logits, y, reduction="none", weight=class_weight)
        loss = (per_row * w).mean()
        loss.backward()
        optimizer.step()
    return model.cpu()


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "stable_positive_pack": frame["stable_positive_teacher_case"],
        "hard_near_tie": frame["hard_near_tie_intersection_case"],
        "stable_near_tie": frame["stable_near_tie_case"],
        "high_headroom_near_tie": frame["high_headroom_near_tie_case"],
        "baseline_error_near_tie": frame["baseline_error_hard_near_tie_case"],
        "large_gap_control": frame["large_gap_hard_feasible_case"],
    }


def _evaluate_budget(
    cache: dict[str, torch.Tensor],
    metadata: pd.DataFrame,
    flip_scores: np.ndarray,
    top1: torch.Tensor,
    top2: torch.Tensor,
    *,
    budget_pct: float,
    variant: str,
    scope: str,
    split: str,
) -> pd.DataFrame:
    selected = top_fraction_mask(flip_scores, budget_pct) & (flip_scores > 0.0)
    predicted = top1.clone()
    predicted[selected] = top2[selected]
    batch_index = torch.arange(predicted.size(0))
    predicted_cost = cache["candidate_cost_to_go"][batch_index, predicted].float().numpy()
    predicted_on_time = (cache["candidate_on_time"][batch_index, predicted] > 0.5).numpy()
    target_next_hop = cache["target_next_hop"].long().numpy()
    predicted_np = predicted.numpy()

    rows: list[dict[str, object]] = []
    for slice_name, mask in _slice_map(metadata).items():
        target = metadata.loc[mask].copy()
        idx = target.index.to_numpy()
        selected_target = selected[idx]
        target_match = predicted_np[idx] == target_next_hop[idx]
        delta_regret = (predicted_cost[idx] - target["best_candidate_cost"].to_numpy(copy=True)) - target[
            "base_predicted_continuation_gap"
        ].to_numpy(copy=True)
        delta_miss = (~predicted_on_time[idx]).astype(int) - (~target["base_predicted_on_time"].to_numpy(copy=True)).astype(int)
        selected_count = max(int(selected_target.sum()), 1)
        rows.append(
            {
                "variant": variant,
                "scope": scope,
                "split": split,
                "budget_pct": budget_pct,
                "slice": slice_name,
                "decisions": len(target),
                "coverage": float(selected_target.mean()) if len(target) else 0.0,
                "flip_precision": float(
                    np.logical_and(selected_target, target["stable_positive_teacher_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "false_flip_rate": float(
                    np.logical_and(selected_target, target["harmful_teacher_case"].to_numpy(copy=True)).sum() / selected_count
                ),
                "correction_rate": float(
                    (
                        np.logical_and(selected_target, ~target["base_target_match"].to_numpy(copy=True))
                        & target_match
                    ).mean()
                )
                if len(target)
                else 0.0,
                "new_error_rate": float(
                    (
                        np.logical_and(selected_target, target["base_target_match"].to_numpy(copy=True))
                        & (~target_match)
                    ).mean()
                )
                if len(target)
                else 0.0,
                "base_target_match": float(target["base_target_match"].mean()) if len(target) else 0.0,
                "policy_target_match": float(target_match.mean()) if len(target) else 0.0,
                "mean_delta_regret": float(delta_regret.mean()) if len(target) else 0.0,
                "mean_delta_miss": float(delta_miss.mean()) if len(target) else 0.0,
                "selected_disagreement": float(selected_target.mean()) if len(target) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    stable = summary_df.loc[summary_df["slice"] == "stable_positive_pack"].copy()
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (variant, scope), group in stable.groupby(["variant", "scope"], sort=False):
        axes[0].plot(group["budget_pct"], group["flip_precision"], marker="o", label=f"{variant}:{scope}")
    for (variant, scope), group in hard.groupby(["variant", "scope"], sort=False):
        axes[1].plot(group["budget_pct"], group["mean_delta_regret"], marker="o", label=f"{variant}:{scope}")
    axes[0].set_title("Comparator Precision vs Coverage")
    axes[1].set_title("Comparator Hard Near-Tie Delta Regret")
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

    train_cache, train_meta = _load(args.train_cache, args.train_metadata)
    train_sets = _feature_sets(train_cache, train_meta)
    train_labels_by_scope = {scope: _labels_and_weights(train_cache, train_meta, scope=scope) for scope in SCOPES}

    eval_payloads = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, meta_path)
        labels, weights, top1, top2 = _labels_and_weights(cache, meta, scope="broad")
        eval_payloads.append((Path(cache_path).stem, cache, meta, _feature_sets(cache, meta), top1, top2))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        for scope in SCOPES:
            labels, weights, _train_top1, _train_top2 = train_labels_by_scope[scope]
            train_x = train_sets[variant].float()
            train_x_norm, _same = _normalize(train_x, train_x)
            model = _fit_model(
                train_x_norm,
                labels,
                weights,
                epochs=args.epochs,
                lr=args.lr,
                hidden_dim=args.hidden_dim,
                device=device,
            )
            for split_name, cache, meta, eval_sets, top1, top2 in eval_payloads:
                _train_norm, eval_x = _normalize(train_x, eval_sets[variant].float())
                with torch.no_grad():
                    logits = model(eval_x).cpu()
                probs = torch.softmax(logits, dim=-1).numpy()
                flip_scores = probs[:, 1] - np.maximum(probs[:, 0], probs[:, 2])
                for budget_pct in args.coverage_budgets:
                    rows.append(
                        _evaluate_budget(
                            cache,
                            meta,
                            flip_scores,
                            top1,
                            top2,
                            budget_pct=budget_pct,
                            variant=variant,
                            scope=scope,
                            split=split_name,
                        )
                    )

    summary_df = pd.concat(rows, ignore_index=True)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "coverage_budgets": args.coverage_budgets,
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
