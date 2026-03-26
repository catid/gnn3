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

TASKS = {
    "helpful": "helpful_compute",
    "harmful": "harmful_compute",
    "action_changed": "action_changed",
    "oracle_correct_flip": "compute_recovers_baseline_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--eval-caches", nargs="+", required=True)
    parser.add_argument("--eval-metadata", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round10_helpfulness_probe",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _load_cache(cache_path: str, metadata_path: str) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    cache = torch.load(cache_path, map_location="cpu")
    metadata = pd.read_csv(metadata_path)
    if int(cache["decision_features"].size(0)) != len(metadata):
        raise ValueError(f"Cache rows do not match metadata rows for {cache_path}")
    return cache, metadata


def _masked_top2(scores: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = scores.masked_fill(~valid_mask, -1e9)
    topk = masked.topk(k=min(2, masked.size(1)), dim=-1)
    first = topk.indices[:, 0]
    second = topk.indices[:, 1] if topk.indices.size(1) > 1 else topk.indices[:, 0]
    return first, second


def _decision_aug_features(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> torch.Tensor:
    base_scores = cache["base_selection_scores"]
    valid_mask = cache["candidate_mask"].bool()
    masked = base_scores.masked_fill(~valid_mask, -1e9)
    topk = masked.topk(k=min(2, masked.size(1)), dim=-1).values
    margin = (topk[:, 0] - topk[:, 1]).clamp_min(0.0) if topk.size(1) > 1 else topk[:, 0].clamp_min(0.0)
    numeric = torch.as_tensor(
        metadata[["best_candidate_slack_ratio", "packet_count", "mean_queue", "max_tree_depth"]].to_numpy(copy=True),
        dtype=torch.float32,
    )
    numeric[:, 1] = numeric[:, 1] / 8.0
    numeric[:, 2] = numeric[:, 2] / 10.0
    numeric[:, 3] = numeric[:, 3] / 6.0
    return torch.cat([cache["decision_features"].float(), margin[:, None], numeric], dim=1)


def _candidate_conditioned_features(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> torch.Tensor:
    valid_mask = cache["candidate_mask"].bool()
    first, second = _masked_top2(cache["base_selection_scores"], valid_mask)
    batch_index = torch.arange(cache["candidate_features"].size(0))
    first_feat = cache["candidate_features"][batch_index, first]
    second_feat = cache["candidate_features"][batch_index, second]
    meta = torch.as_tensor(
        metadata[["best_candidate_slack_ratio", "packet_count", "mean_queue", "max_tree_depth"]].to_numpy(copy=True),
        dtype=torch.float32,
    )
    meta[:, 1] = meta[:, 1] / 8.0
    meta[:, 2] = meta[:, 2] / 10.0
    meta[:, 3] = meta[:, 3] / 6.0
    return torch.cat(
        [
            cache["decision_features"].float(),
            first_feat.float(),
            second_feat.float(),
            (first_feat - second_feat).float(),
            meta,
        ],
        dim=1,
    )


def _margin_features(metadata: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(metadata[["base_model_margin"]].to_numpy(copy=True), dtype=torch.float32)


def _margin_plus_regime_features(metadata: pd.DataFrame) -> torch.Tensor:
    frame = metadata[["base_model_margin", "best_candidate_slack_ratio", "packet_count", "mean_queue", "max_tree_depth"]].copy()
    frame["packet_count"] = frame["packet_count"] / 8.0
    frame["mean_queue"] = frame["mean_queue"] / 10.0
    frame["max_tree_depth"] = frame["max_tree_depth"] / 6.0
    return torch.as_tensor(frame.to_numpy(copy=True), dtype=torch.float32)


def _feature_sets(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> dict[str, torch.Tensor]:
    return {
        "linear": _decision_aug_features(cache, metadata),
        "mlp": _decision_aug_features(cache, metadata),
        "candidate_conditioned": _candidate_conditioned_features(cache, metadata),
        "margin_only": _margin_features(metadata),
        "margin_plus_regime": _margin_plus_regime_features(metadata),
    }


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    negatives = int((1 - labels).sum())
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    pos_ranks = ranks[labels == 1].astype(np.float64) + 1.0
    return float((pos_ranks.sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    precision = np.cumsum(labels_sorted) / np.arange(1, len(labels_sorted) + 1)
    return float((precision * labels_sorted).sum() / positives)


def _binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = probs >= 0.5
    tp = float(np.logical_and(preds, labels == 1).sum())
    fp = float(np.logical_and(preds, labels == 0).sum())
    fn = float(np.logical_and(~preds, labels == 1).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "auroc": _auroc(labels, probs),
        "average_precision": _average_precision(labels, probs),
        "brier": float(np.mean((probs - labels) ** 2)),
        "precision_at_050": precision,
        "recall_at_050": recall,
        "trigger_rate_at_050": float(preds.mean()),
    }


def _fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    variant: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    if variant in {"linear", "margin_only", "margin_plus_regime"}:
        model = torch.nn.Linear(train_x.size(1), 1).to(device)
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.size(1), hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = train_x.to(device)
    y = train_y.to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return model.cpu()


def _normalize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _eval_variant(
    variant: str,
    train_features: torch.Tensor,
    eval_features: torch.Tensor,
    train_labels: torch.Tensor,
    eval_labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> dict[str, float]:
    train_x, eval_x = _normalize(train_features, eval_features)
    model = _fit_probe(
        train_x,
        train_labels,
        variant=variant,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        device=device,
    )
    with torch.no_grad():
        probs = torch.sigmoid(model(eval_x).squeeze(-1)).numpy()
    return _binary_metrics(eval_labels.numpy().astype(np.int64), probs.astype(np.float64))


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    helpful = summary_df.loc[summary_df["task"] == "helpful"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for split, group in helpful.groupby("split", sort=False):
        axes[0].plot(group["variant"], group["auroc"], marker="o", label=split)
        axes[1].plot(group["variant"], group["average_precision"], marker="o", label=split)
    axes[0].set_title("Helpfulness AUROC")
    axes[1].set_title("Helpfulness AP")
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

    train_cache, train_meta = _load_cache(args.train_cache, args.train_metadata)
    train_sets = _feature_sets(train_cache, train_meta)
    eval_payloads = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load_cache(cache_path, meta_path)
        eval_payloads.append((Path(cache_path).stem, cache, meta, _feature_sets(cache, meta)))

    rows: list[dict[str, object]] = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for variant, train_features in train_sets.items():
        for task_name, column in TASKS.items():
            train_labels = torch.as_tensor(train_meta[column].astype(int).to_numpy(copy=True), dtype=torch.float32)
            for split_name, _cache, eval_meta, eval_sets in eval_payloads:
                eval_labels = torch.as_tensor(eval_meta[column].astype(int).to_numpy(copy=True), dtype=torch.float32)
                metrics = _eval_variant(
                    variant,
                    train_features.float(),
                    eval_sets[variant].float(),
                    train_labels,
                    eval_labels,
                    epochs=args.epochs,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    device=device,
                )
                rows.append(
                    {
                        "variant": variant,
                        "task": task_name,
                        "split": split_name,
                        **metrics,
                    }
                )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
