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


class GateHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-caches", nargs="+", required=True)
    parser.add_argument("--eval-metadata", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--base-steps", type=float, default=3.0)
    parser.add_argument("--compute-steps", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round10_selective_compute",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _load(cache_path: str, metadata_path: str) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    cache = torch.load(cache_path, map_location="cpu")
    meta = pd.read_csv(metadata_path)
    return cache, meta


def _decision_features(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> torch.Tensor:
    margin = torch.as_tensor(metadata["base_model_margin"].to_numpy(copy=True), dtype=torch.float32)[:, None]
    regime = torch.as_tensor(
        metadata[["best_candidate_slack_ratio", "packet_count", "mean_queue", "max_tree_depth"]].to_numpy(copy=True),
        dtype=torch.float32,
    )
    regime[:, 1] = regime[:, 1] / 8.0
    regime[:, 2] = regime[:, 2] / 10.0
    regime[:, 3] = regime[:, 3] / 6.0
    return torch.cat([cache["decision_features"].float(), margin, regime], dim=1)


def _fit_gate(
    train_x: torch.Tensor,
    helpful: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> tuple[GateHead, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    x = ((train_x - mean) / std).to(device)
    y = helpful.float().to(device)
    gate = GateHead(train_x.size(1), hidden_dim).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = gate(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return gate.cpu(), mean, std


def _choose_threshold(probs: np.ndarray, helpful: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1e9
    for threshold in np.linspace(0.1, 0.9, 17):
        triggered = probs >= threshold
        tp = float(np.logical_and(triggered, helpful == 1).sum())
        fp = float(np.logical_and(triggered, helpful == 0).sum())
        score = tp - 0.75 * fp
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _top2_mask(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~valid_mask, -1e9)
    top2 = masked.topk(k=min(2, masked.size(1)), dim=-1).indices
    top2_mask = torch.zeros_like(valid_mask)
    batch_index = torch.arange(valid_mask.size(0))[:, None]
    top2_mask[batch_index, top2] = True
    return top2_mask


def _evaluate_policy(
    policy_name: str,
    gate_probs: torch.Tensor,
    threshold: float,
    cache: dict[str, torch.Tensor],
    metadata: pd.DataFrame,
    *,
    base_steps: float,
    compute_steps: float,
) -> pd.DataFrame:
    valid_mask = cache["candidate_mask"].bool()
    base_scores = cache["base_selection_scores"].float().masked_fill(~valid_mask, -1e9)
    compute_scores = cache["compute_selection_scores"].float().masked_fill(~valid_mask, -1e9)
    trigger = gate_probs >= threshold
    if policy_name == "triggered_top2_compute":
        top2 = _top2_mask(base_scores, valid_mask)
        compute_choice = compute_scores.argmax(dim=1)
        accept = top2[torch.arange(top2.size(0)), compute_choice]
        trigger = trigger & accept
    scores = torch.where(trigger[:, None], compute_scores, base_scores)
    predicted = scores.argmax(dim=1)

    rows = metadata.copy()
    rows["policy"] = policy_name
    rows["triggered"] = trigger.numpy()
    rows["predicted_next_hop_policy"] = predicted.numpy()
    rows["policy_target_match"] = (predicted == cache["target_next_hop"].long()).numpy()
    rows["disagreement"] = predicted.numpy() != rows["base_predicted_next_hop"].to_numpy(copy=True)
    batch_index = torch.arange(predicted.size(0))
    rows["policy_continuation_gap"] = (
        cache["candidate_cost_to_go"][batch_index, predicted].float().numpy()
        - rows["best_candidate_cost"].to_numpy(copy=True)
    )
    rows["policy_on_time"] = (cache["candidate_on_time"][batch_index, predicted] > 0.5).numpy()
    rows["delta_regret_policy"] = rows["policy_continuation_gap"] - rows["base_predicted_continuation_gap"]
    rows["delta_miss_policy"] = (~rows["policy_on_time"]).astype(int) - (~rows["base_predicted_on_time"]).astype(int)
    rows["correction"] = (~rows["base_target_match"]) & rows["policy_target_match"]
    rows["new_error"] = rows["base_target_match"] & (~rows["policy_target_match"])
    rows["average_outer_steps"] = base_steps + trigger.float().mean().item() * (compute_steps - base_steps)
    rows["compute_multiplier"] = rows["average_outer_steps"] / max(base_steps, 1e-6)
    return rows


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    result_rows: list[dict[str, object]] = []
    slices = {
        "overall": pd.Series([True] * len(rows)),
        "hard_near_tie": rows["hard_near_tie_intersection_case"],
        "stable_near_tie": rows["stable_near_tie_case"],
        "high_headroom_near_tie": rows["high_headroom_near_tie_case"],
        "baseline_error_near_tie": rows["baseline_error_hard_near_tie_case"],
        "large_gap_control": rows["large_gap_hard_feasible_case"],
    }
    for (policy, suite), suite_frame in rows.groupby(["policy", "suite"], sort=False):
        for slice_name, mask in slices.items():
            target = suite_frame.loc[mask.loc[suite_frame.index]]
            result_rows.append(
                {
                    "policy": policy,
                    "suite": suite,
                    "slice": slice_name,
                    "decisions": len(target),
                    "trigger_rate": float(target["triggered"].mean()) if len(target) else 0.0,
                    "disagreement": float(target["disagreement"].mean()) if len(target) else 0.0,
                    "correction_rate": float(target["correction"].mean()) if len(target) else 0.0,
                    "new_error_rate": float(target["new_error"].mean()) if len(target) else 0.0,
                    "base_target_match": float(target["base_target_match"].mean()) if len(target) else 0.0,
                    "policy_target_match": float(target["policy_target_match"].mean()) if len(target) else 0.0,
                    "mean_delta_regret": float(target["delta_regret_policy"].mean()) if len(target) else 0.0,
                    "mean_delta_miss": float(target["delta_miss_policy"].mean()) if len(target) else 0.0,
                    "average_outer_steps": float(target["average_outer_steps"].iloc[0]) if len(target) else 0.0,
                    "compute_multiplier": float(target["compute_multiplier"].iloc[0]) if len(target) else 0.0,
                }
            )
    return pd.DataFrame(result_rows)


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for policy, group in hard.groupby("policy", sort=False):
        axes[0].plot(group["suite"], group["policy_target_match"], marker="o", label=policy)
        axes[1].plot(group["suite"], group["average_outer_steps"], marker="o", label=policy)
    axes[0].set_title("Hard Near-Tie Target Match")
    axes[1].set_title("Average Outer Steps")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
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
    train_x = _decision_features(train_cache, train_meta)
    helpful = torch.as_tensor(
        (train_meta["helpful_compute"] & train_meta["hard_near_tie_intersection_case"]).to_numpy(copy=True),
        dtype=torch.float32,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gate, mean, std = _fit_gate(
        train_x,
        helpful,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    with torch.no_grad():
        train_probs = torch.sigmoid(gate((train_x - mean) / std)).numpy()
    threshold = _choose_threshold(train_probs, helpful.numpy().astype(int))

    all_rows: list[pd.DataFrame] = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, meta_path)
        eval_x = _decision_features(cache, meta)
        with torch.no_grad():
            probs = torch.sigmoid(gate((eval_x - mean) / std)).float()
        for policy_name in ("triggered_full_compute", "triggered_top2_compute"):
            all_rows.append(
                _evaluate_policy(
                    policy_name,
                    probs,
                    threshold,
                    cache,
                    meta,
                    base_steps=args.base_steps,
                    compute_steps=args.compute_steps,
                )
            )

    decision_df = pd.concat(all_rows, ignore_index=True)
    summary_df = _summary(decision_df)
    decision_df.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
