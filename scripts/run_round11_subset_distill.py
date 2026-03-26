#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

VARIANTS = (
    "pairwise",
    "kl",
    "residual",
    "gated_pairwise",
)


class CandidateStudent(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


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
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round11_subset_distill",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _load(cache_path: str, metadata_path: str) -> tuple[dict[str, torch.Tensor], pd.DataFrame]:
    cache = torch.load(cache_path, map_location="cpu")
    meta = pd.read_csv(metadata_path)
    if int(cache["decision_features"].size(0)) != len(meta):
        raise ValueError(f"Cache rows do not match metadata rows for {cache_path}")
    return cache, meta


def _masked_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~mask, -1e9)


def _normalize(train: torch.Tensor, eval_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (train - mean) / std, (eval_value - mean) / std


def _teacher_actions(metadata: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(metadata["teacher_next_hop"].to_numpy(copy=True), dtype=torch.long)


def _train_gate(
    decision_features: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> tuple[GateHead, torch.Tensor, torch.Tensor]:
    train_x = decision_features.float()
    norm_x, _same = _normalize(train_x, train_x)
    gate = GateHead(norm_x.size(1), hidden_dim).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    x = norm_x.to(device)
    y = positive_mask.float().to(device)
    positive = float(positive_mask.sum().item())
    negative = float((1.0 - positive_mask).sum().item())
    pos_weight = torch.tensor(negative / max(positive, 1.0), device=device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = gate(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    return gate.cpu(), train_x.mean(dim=0, keepdim=True), train_x.std(dim=0, keepdim=True).clamp(min=1e-6)


def _candidate_loss(
    variant: str,
    logits: torch.Tensor,
    base_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
    teacher_action: torch.Tensor,
    base_action: torch.Tensor,
    helpful: torch.Tensor,
    helpful_weight: torch.Tensor,
) -> torch.Tensor:
    logits = _masked_scores(logits, mask)
    base_scores = _masked_scores(base_scores, mask)
    teacher_scores = _masked_scores(teacher_scores, mask)
    helpful_idx = helpful.nonzero(as_tuple=False).squeeze(-1)
    non_help_idx = (~helpful).nonzero(as_tuple=False).squeeze(-1)
    losses: list[torch.Tensor] = []

    if "pairwise" in variant and helpful_idx.numel() > 0:
        teacher_margin = logits[helpful_idx, teacher_action[helpful_idx]] - logits[helpful_idx, base_action[helpful_idx]]
        margin_loss = F.relu(0.5 - teacher_margin)
        ce_loss = F.cross_entropy(logits[helpful_idx], teacher_action[helpful_idx], reduction="none")
        losses.append((margin_loss + ce_loss) * helpful_weight[helpful_idx])
    elif "kl" in variant and helpful_idx.numel() > 0:
        teacher_prob = F.softmax(teacher_scores[helpful_idx] / 0.75, dim=-1)
        kl = F.kl_div(F.log_softmax(logits[helpful_idx], dim=-1), teacher_prob, reduction="none").sum(dim=-1)
        losses.append(kl * helpful_weight[helpful_idx])
    elif "residual" in variant and helpful_idx.numel() > 0:
        corrected = base_scores[helpful_idx] + logits[helpful_idx]
        ce = F.cross_entropy(corrected, teacher_action[helpful_idx], reduction="none")
        losses.append(ce * helpful_weight[helpful_idx])

    if non_help_idx.numel() > 0:
        if "residual" in variant:
            losses.append(0.02 * logits[non_help_idx].pow(2).mean(dim=-1))
        else:
            anchor = F.cross_entropy(logits[non_help_idx], base_action[non_help_idx], reduction="none")
            losses.append(0.40 * anchor)

    if not losses:
        return logits.new_tensor(0.0)
    return torch.cat(losses).mean()


def _fit_student(
    train_cache: dict[str, torch.Tensor],
    train_meta: pd.DataFrame,
    *,
    variant: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    device: torch.device,
) -> dict[str, object]:
    candidate_x = train_cache["candidate_features"].float()
    candidate_mean = candidate_x.view(-1, candidate_x.size(-1)).mean(dim=0, keepdim=True)
    candidate_std = candidate_x.view(-1, candidate_x.size(-1)).std(dim=0, keepdim=True).clamp(min=1e-6)
    candidate_x = ((candidate_x - candidate_mean[:, None, :]) / candidate_std[:, None, :]).to(device)

    decision_x = train_cache["decision_features"].float()
    helpful_mask = torch.as_tensor(
        train_meta["stable_positive_teacher_case"].astype(int).to_numpy(copy=True),
        dtype=torch.bool,
    )
    gate_model = None
    gate_mean = None
    gate_std = None
    if variant.startswith("gated_"):
        gate_model, gate_mean, gate_std = _train_gate(
            decision_x,
            helpful_mask.float(),
            epochs=max(100, epochs // 2),
            lr=lr,
            hidden_dim=max(hidden_dim // 2, 32),
            device=device,
        )

    model = CandidateStudent(candidate_x.size(-1), hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    base_scores = train_cache["base_selection_scores"].float().to(device)
    teacher_scores = train_cache["compute_selection_scores"].float().to(device)
    mask = train_cache["candidate_mask"].bool().to(device)
    teacher_action = _teacher_actions(train_meta).to(device)
    base_action = base_scores.masked_fill(~mask, -1e9).argmax(dim=1)
    helpful_weight = torch.as_tensor(
        (1.0 + train_meta["teacher_regret_gain"].clip(lower=0.0).to_numpy(copy=True)).astype(float),
        dtype=torch.float32,
        device=device,
    )
    helpful = helpful_mask.to(device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(candidate_x.view(-1, candidate_x.size(-1))).view(candidate_x.size(0), candidate_x.size(1))
        loss = _candidate_loss(
            variant,
            logits,
            base_scores,
            teacher_scores,
            mask,
            teacher_action,
            base_action,
            helpful,
            helpful_weight,
        )
        loss.backward()
        optimizer.step()
    return {
        "model": model.cpu(),
        "candidate_mean": candidate_mean,
        "candidate_std": candidate_std,
        "gate_model": gate_model,
        "gate_mean": gate_mean,
        "gate_std": gate_std,
    }


def _apply_student(
    payload: dict[str, object],
    cache: dict[str, torch.Tensor],
    *,
    variant: str,
) -> torch.Tensor:
    model: CandidateStudent = payload["model"]  # type: ignore[assignment]
    candidate_features = cache["candidate_features"].float()
    candidate_x = (candidate_features - payload["candidate_mean"][:, None, :]) / payload["candidate_std"][:, None, :]  # type: ignore[index]
    logits = model(candidate_x.view(-1, candidate_x.size(-1))).view(candidate_x.size(0), candidate_x.size(1))
    base_scores = cache["base_selection_scores"].float()
    mask = cache["candidate_mask"].bool()
    if variant.startswith("gated_"):
        gate_model: GateHead = payload["gate_model"]  # type: ignore[assignment]
        gate_x = (cache["decision_features"].float() - payload["gate_mean"]) / payload["gate_std"]  # type: ignore[index]
        gate = torch.sigmoid(gate_model(gate_x)).clamp(0.0, 1.0)
    else:
        gate = torch.ones((candidate_x.size(0),), dtype=torch.float32)

    if "residual" in variant:
        scores = base_scores + gate[:, None] * logits
    else:
        scores = base_scores + gate[:, None] * (logits - base_scores)
    return _masked_scores(scores, mask)


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


def _decision_metrics(predicted: torch.Tensor, cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> pd.DataFrame:
    predicted = predicted.cpu()
    batch_index = torch.arange(predicted.size(0))
    target = cache["target_next_hop"].long()
    predicted_cost = cache["candidate_cost_to_go"][batch_index, predicted].float()
    predicted_on_time = cache["candidate_on_time"][batch_index, predicted] > 0.5

    rows = metadata.copy()
    rows["student_predicted_next_hop"] = predicted.numpy()
    rows["student_target_match"] = (predicted == target).numpy()
    rows["disagreement"] = predicted.numpy() != rows["base_predicted_next_hop"].to_numpy(copy=True)
    rows["correction"] = (~rows["base_target_match"]) & rows["student_target_match"]
    rows["new_error"] = rows["base_target_match"] & (~rows["student_target_match"])
    rows["student_continuation_gap"] = predicted_cost.numpy() - rows["best_candidate_cost"].to_numpy(copy=True)
    rows["delta_regret_student"] = rows["student_continuation_gap"] - rows["base_predicted_continuation_gap"]
    rows["delta_miss_student"] = (~predicted_on_time.numpy()).astype(int) - (~rows["base_predicted_on_time"]).astype(int)
    return rows


def _summary(decision_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, split, suite), suite_frame in decision_rows.groupby(["variant", "split", "suite"], sort=False):
        for slice_name, mask in _slice_map(suite_frame).items():
            target = suite_frame.loc[mask]
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "suite": suite,
                    "slice": slice_name,
                    "decisions": len(target),
                    "disagreement": float(target["disagreement"].mean()) if len(target) else 0.0,
                    "correction_rate": float(target["correction"].mean()) if len(target) else 0.0,
                    "new_error_rate": float(target["new_error"].mean()) if len(target) else 0.0,
                    "base_target_match": float(target["base_target_match"].mean()) if len(target) else 0.0,
                    "student_target_match": float(target["student_target_match"].mean()) if len(target) else 0.0,
                    "mean_delta_regret": float(target["delta_regret_student"].mean()) if len(target) else 0.0,
                    "mean_delta_miss": float(target["delta_miss_student"].mean()) if len(target) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    stable = summary_df.loc[summary_df["slice"] == "stable_positive_pack"].copy()
    hard = summary_df.loc[summary_df["slice"] == "hard_near_tie"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for variant, group in stable.groupby("variant", sort=False):
        axes[0].plot(group["split"], group["correction_rate"], marker="o", label=variant)
    for variant, group in hard.groupby("variant", sort=False):
        axes[1].plot(group["split"], group["mean_delta_regret"], marker="o", label=variant)
    axes[0].set_title("Stable-Positive Correction Rate")
    axes[1].set_title("Hard Near-Tie Delta Regret")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
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
    eval_payloads = []
    for cache_path, meta_path in zip(args.eval_caches, args.eval_metadata, strict=True):
        cache, meta = _load(cache_path, meta_path)
        eval_payloads.append((Path(cache_path).stem, cache, meta))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        payload = _fit_student(
            train_cache,
            train_meta,
            variant=variant,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device=device,
        )
        for split_name, cache, meta in eval_payloads:
            predicted = _apply_student(payload, cache, variant=variant).argmax(dim=1)
            decision_rows = _decision_metrics(predicted, cache, meta)
            decision_rows["variant"] = variant
            decision_rows["split"] = split_name
            rows.append(decision_rows)

    decisions_df = pd.concat(rows, ignore_index=True)
    summary_df = _summary(decisions_df)
    decisions_df.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
