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
    decision_augmented_features,
    margin_only_features,
    margin_regime_features,
    top_fraction_mask,
)

VARIANTS = ("linear", "mlp", "margin_regime")


class GateHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *, mlp: bool) -> None:
        super().__init__()
        if mlp:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = torch.nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


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
        default="reports/plots/round11_defer_gate",
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
    headroom = torch.as_tensor(metadata[["base_predicted_continuation_gap"]].to_numpy(copy=True), dtype=torch.float32)
    margin_regime = torch.cat([margin_only_features(metadata), margin_regime_features(metadata), headroom], dim=1)
    decision = decision_augmented_features(cache, metadata)
    return {
        "linear": decision,
        "mlp": decision,
        "margin_regime": margin_regime,
    }


def _normalize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _fit_gate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    variant: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> GateHead:
    mlp = variant == "mlp"
    model = GateHead(train_x.size(1), hidden_dim, mlp=mlp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = train_x.to(device)
    y = train_y.to(device)
    positive = float(train_y.sum().item())
    negative = float((1.0 - train_y).sum().item())
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
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
    meta: pd.DataFrame,
    scores: np.ndarray,
    *,
    budget_pct: float,
    variant: str,
    split: str,
) -> pd.DataFrame:
    selected = top_fraction_mask(scores, budget_pct)
    rows: list[dict[str, object]] = []
    for slice_name, mask in _slice_map(meta).items():
        target = meta.loc[mask].copy()
        selected_target = selected[target.index.to_numpy()]
        if len(target):
            effective_disagreement = selected_target & target["action_changed"].to_numpy(copy=True)
            delta_regret = np.where(selected_target, target["delta_regret"].to_numpy(copy=True), 0.0)
            delta_miss = np.where(selected_target, target["delta_miss"].to_numpy(copy=True), 0.0)
            combined_target_match = np.where(
                selected_target,
                target["teacher_target_match"].to_numpy(copy=True),
                target["base_target_match"].to_numpy(copy=True),
            )
            selected_count = max(int(selected_target.sum()), 1)
            precision = float(
                np.logical_and(selected_target, target["stable_positive_teacher_case"].to_numpy(copy=True)).sum() / selected_count
            )
            false_positive = float(
                np.logical_and(selected_target, target["harmful_teacher_case"].to_numpy(copy=True)).sum() / selected_count
            )
        else:
            effective_disagreement = np.zeros((0,), dtype=bool)
            delta_regret = np.zeros((0,), dtype=float)
            delta_miss = np.zeros((0,), dtype=float)
            combined_target_match = np.zeros((0,), dtype=float)
            precision = 0.0
            false_positive = 0.0
        rows.append(
            {
                "variant": variant,
                "split": split,
                "budget_pct": budget_pct,
                "slice": slice_name,
                "decisions": len(target),
                "coverage": float(selected_target.mean()) if len(target) else 0.0,
                "selected_disagreement": float(effective_disagreement.mean()) if len(target) else 0.0,
                "defer_precision": precision,
                "false_positive_rate": false_positive,
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
    return pd.DataFrame(rows)


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    stable = summary_df.loc[summary_df["slice"] == "stable_positive_pack"].copy()
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for variant, group in stable.groupby("variant", sort=False):
        axes[0].plot(group["budget_pct"], group["defer_precision"], marker="o", label=variant)
    for variant, group in hard.groupby("variant", sort=False):
        axes[1].plot(group["budget_pct"], group["mean_delta_regret"], marker="o", label=variant)
    axes[0].set_title("Stable-Positive Precision vs Coverage")
    axes[0].set_xlabel("Budget %")
    axes[1].set_title("Hard Near-Tie Delta Regret vs Coverage")
    axes[1].set_xlabel("Budget %")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    for axis in axes:
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
    train_y = torch.as_tensor(train_meta["stable_positive_teacher_case"].astype(int).to_numpy(copy=True), dtype=torch.float32)

    eval_payloads = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, meta_path)
        eval_payloads.append((Path(cache_path).stem, cache, meta, _feature_sets(cache, meta)))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        train_x = train_sets[variant].float()
        model = _fit_gate(
            train_x,
            train_y,
            variant=variant,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device=device,
        )
        for split_name, _cache, eval_meta, eval_sets in eval_payloads:
            _train_x, eval_x = _normalize(train_x, eval_sets[variant].float())
            with torch.no_grad():
                scores = torch.sigmoid(model(eval_x).squeeze(-1)).numpy()
            for budget_pct in args.coverage_budgets:
                rows.append(
                    _evaluate_budget(
                        eval_meta,
                        scores,
                        budget_pct=budget_pct,
                        variant=variant,
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
