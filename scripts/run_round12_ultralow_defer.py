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

VARIANTS = ("linear", "mlp", "margin_regime")
KEY_COLS = ["suite", "episode_index", "decision_index"]


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
    parser.add_argument("--teacher-bank-decisions-csv", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--coverage-budgets",
        nargs="+",
        type=float,
        default=list(ULTRALOW_COVERAGE_BUDGETS),
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round12_ultralow_defer",
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
    merge_cols = [*KEY_COLS, "stable_positive_v2_case", "stable_positive_v2_committee_case", "harmful_teacher_bank_case", "best_safe_teacher_gain", "committee_support"]
    merged = meta.merge(teacher_bank[merge_cols], on=KEY_COLS, how="left")
    for col in ["stable_positive_v2_case", "stable_positive_v2_committee_case", "harmful_teacher_bank_case"]:
        merged[col] = merged[col].fillna(False).astype(bool)
    merged["best_safe_teacher_gain"] = merged["best_safe_teacher_gain"].fillna(0.0).astype(float)
    merged["committee_support"] = merged["committee_support"].fillna(0).astype(int)
    return merged


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
    train_weight: torch.Tensor,
    *,
    variant: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> GateHead:
    model = GateHead(train_x.size(1), hidden_dim, mlp=(variant == "mlp")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = train_x.to(device)
    y = train_y.to(device)
    w = train_weight.to(device)
    positive = float(train_y.sum().item())
    negative = float((1.0 - train_y).sum().item())
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
        loss = (loss * w).mean()
        loss.backward()
        optimizer.step()
    return model.cpu()


def _slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "stable_positive_v2": frame["stable_positive_v2_case"],
        "stable_positive_v2_committee": frame["stable_positive_v2_committee_case"],
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = top_fraction_mask(scores, budget_pct)
    decision_frame = meta[KEY_COLS].copy()
    decision_frame["variant"] = variant
    decision_frame["split"] = split
    decision_frame["budget_pct"] = budget_pct
    decision_frame["score"] = scores
    decision_frame["selected"] = selected
    decision_frame["stable_positive_v2_case"] = meta["stable_positive_v2_case"].to_numpy(copy=True)
    decision_frame["harmful_teacher_bank_case"] = meta["harmful_teacher_bank_case"].to_numpy(copy=True)

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
                "selected_disagreement": float(np.logical_and(selected_target, target["action_changed"].to_numpy(copy=True)).mean())
                if len(target)
                else 0.0,
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
    axes[0].set_title("Stable-Positive-v2 Precision vs Coverage")
    axes[1].set_title("Hard Near-Tie Delta Regret vs Coverage")
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
    teacher_bank = teacher_bank[[*KEY_COLS, "stable_positive_v2_case", "stable_positive_v2_committee_case", "harmful_teacher_bank_case", "best_safe_teacher_gain", "committee_support"]].copy()

    train_cache, train_meta = _load(args.train_cache, args.train_metadata)
    train_meta = _attach_teacher_bank(train_meta, teacher_bank)
    train_sets = _feature_sets(train_cache, train_meta)
    train_y = torch.as_tensor(train_meta["stable_positive_v2_case"].astype(int).to_numpy(copy=True), dtype=torch.float32)
    train_weight = torch.as_tensor(
        (
            1.0
            + train_meta["best_safe_teacher_gain"].clip(lower=0.0).to_numpy(copy=True)
            + 0.25 * train_meta["committee_support"].clip(lower=0, upper=4).to_numpy(copy=True)
        ).astype(float),
        dtype=torch.float32,
    )

    eval_payloads = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, meta_path)
        meta = _attach_teacher_bank(meta, teacher_bank)
        eval_payloads.append((Path(cache_path).stem, cache, meta, _feature_sets(cache, meta)))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    summary_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        train_x = train_sets[variant].float()
        model = _fit_gate(
            train_x,
            train_y,
            train_weight,
            variant=variant,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device=device,
        )
        for split_name, _cache, eval_meta, eval_sets in eval_payloads:
            _train_norm, eval_norm = _normalize(train_x, eval_sets[variant].float())
            with torch.no_grad():
                scores = torch.sigmoid(model(eval_norm).squeeze(-1)).numpy()
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
    _plot(summary_df.loc[summary_df["split"].isin([Path(path).stem for path in args.eval_caches])], output_prefix.with_name(output_prefix.name + "_summary.png"))
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
