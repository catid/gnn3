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

from gnn3.models.prototype_defer import (
    DualTeacherRebuiltNegativeBankBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
)
from run_prototype_branchwise_max_negative_cleanup_support_agreement_mixture_defer import (
    _attach_teacher_bank,
    _evaluate_budget,
    _feature_sets,
    _fit_head,
    _labels,
    _load,
    _normalize,
    _weights,
)

VARIANTS = (
    "prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix",
    "prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agree_mix_hybrid",
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
    parser.add_argument("--positive-overlap-penalty", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--coverage-budgets",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/prototype_dual_teacher_rebuilt_negative_bank_branchwise_max_negative_cleanup_support_agreement_mixture_defer",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _teacher_candidate_weights(meta: pd.DataFrame) -> torch.Tensor:
    harmful = meta["harmful_teacher_bank_case"].to_numpy(copy=True)
    base = (
        1.0
        + meta["best_safe_teacher_gain"].clip(lower=0.0).to_numpy(copy=True)
        + 0.25 * meta["committee_support"].clip(lower=0, upper=4).to_numpy(copy=True)
    ).astype(np.float32)
    return torch.as_tensor(base[harmful], dtype=torch.float32)


def _teacher_rebuild_dual_bank(
    *,
    harmful_encoded: torch.Tensor,
    stable_encoded: torch.Tensor,
    harmful_weights: torch.Tensor,
    learned_bank: torch.Tensor,
    positive_overlap_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    num_prototypes = int(learned_bank.size(0))
    if harmful_encoded.numel() == 0:
        zero_support = torch.zeros(num_prototypes, dtype=learned_bank.dtype)
        return learned_bank.clone(), zero_support, {
            "harmful_candidates": 0,
            "selected_candidates": 0,
            "padded_from_learned": num_prototypes,
            "avg_positive_overlap": 0.0,
            "total_cluster_weight": 0.0,
        }

    harmful_encoded = F.normalize(harmful_encoded, dim=-1)
    learned_bank = F.normalize(learned_bank, dim=-1)
    if stable_encoded.numel():
        stable_encoded = F.normalize(stable_encoded, dim=-1)
        positive_overlap = torch.clamp(harmful_encoded @ stable_encoded.T, min=0.0).amax(dim=1)
    else:
        positive_overlap = torch.zeros(harmful_encoded.size(0), dtype=harmful_encoded.dtype)

    utility = harmful_weights * (1.0 - positive_overlap_penalty * positive_overlap).clamp(min=0.05)
    selected: list[int] = []
    available = torch.ones(harmful_encoded.size(0), dtype=torch.bool)
    rounds = min(num_prototypes, harmful_encoded.size(0))
    for _ in range(rounds):
        if selected:
            selected_bank = harmful_encoded[selected]
            novelty = (1.0 - (harmful_encoded @ selected_bank.T).amax(dim=1)).clamp(min=0.0)
            score = utility * novelty
        else:
            score = utility.clone()
        score = score.masked_fill(~available, float("-inf"))
        idx = int(torch.argmax(score).item())
        selected.append(idx)
        available[idx] = False

    seed_bank = harmful_encoded[selected]
    assignment = (harmful_encoded @ seed_bank.T).argmax(dim=1)
    rebuilt_bank: list[torch.Tensor] = []
    rebuilt_support: list[torch.Tensor] = []
    cluster_masses: list[float] = []
    cluster_overlaps: list[float] = []
    for cluster_idx in range(len(selected)):
        mask = assignment == cluster_idx
        if bool(mask.any()):
            weight = harmful_weights[mask]
            centroid = (harmful_encoded[mask] * weight.unsqueeze(1)).sum(dim=0)
            proto = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)
            mass = float(weight.sum().item())
            overlap = float(positive_overlap[mask].mean().item())
        else:
            proto = seed_bank[cluster_idx]
            mass = float(harmful_weights[selected[cluster_idx]].item())
            overlap = float(positive_overlap[selected[cluster_idx]].item())
        rebuilt_bank.append(proto)
        rebuilt_support.append(torch.tensor(np.log(max(mass, 1e-3)), dtype=learned_bank.dtype))
        cluster_masses.append(mass)
        cluster_overlaps.append(overlap)

    padding = num_prototypes - len(rebuilt_bank)
    if padding > 0:
        for proto in learned_bank[:padding]:
            rebuilt_bank.append(proto)
            rebuilt_support.append(torch.tensor(0.0, dtype=learned_bank.dtype))
            cluster_masses.append(0.0)
            cluster_overlaps.append(0.0)

    return (
        torch.stack(rebuilt_bank[:num_prototypes], dim=0),
        torch.stack(rebuilt_support[:num_prototypes], dim=0),
        {
            "harmful_candidates": int(harmful_encoded.size(0)),
            "selected_candidates": len(selected),
            "padded_from_learned": max(padding, 0),
            "avg_positive_overlap": float(np.mean(cluster_overlaps)) if cluster_overlaps else 0.0,
            "total_cluster_weight": float(np.sum(cluster_masses)) if cluster_masses else 0.0,
        },
    )


def _rebuild_model(
    base_model: DualTeacherRebuiltNegativeBankBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    train_features: torch.Tensor,
    train_meta: pd.DataFrame,
    *,
    args: argparse.Namespace,
    variant: str,
) -> tuple[
    DualTeacherRebuiltNegativeBankBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    pd.DataFrame,
]:
    harmful_mask = torch.as_tensor(train_meta["harmful_teacher_bank_case"].to_numpy(copy=True), dtype=torch.bool)
    stable_mask = torch.as_tensor(train_meta["stable_positive_v2_case"].to_numpy(copy=True), dtype=torch.bool)
    with torch.no_grad():
        dual_negative_encoded = base_model.encode_dual_negative(train_features[harmful_mask])
        dual_stable_encoded = base_model.encode_dual_negative(train_features[stable_mask])
        _, learned_dual_neg = base_model._dual_banks()
        harmful_weights = _teacher_candidate_weights(train_meta)
        rebuilt_dual, rebuilt_dual_support, stats = _teacher_rebuild_dual_bank(
            harmful_encoded=dual_negative_encoded,
            stable_encoded=dual_stable_encoded,
            harmful_weights=harmful_weights,
            learned_bank=learned_dual_neg,
            positive_overlap_penalty=args.positive_overlap_penalty,
        )

    rebuilt_model = DualTeacherRebuiltNegativeBankBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=train_features.size(1),
        risk_dim=base_model.risk_branch[0].in_features if base_model.risk_branch is not None else 0,
        prototype_dim=args.prototype_dim,
        positive_prototypes=args.positive_prototypes,
        negative_prototypes=args.negative_prototypes,
        hidden_dim=args.hidden_dim,
        use_risk_branch=variant.endswith("_hybrid"),
        tail_margin=args.tail_margin,
        tail_shrink_scale=args.tail_shrink_scale,
        shared_tail_shrink_scale=args.shared_tail_shrink_scale,
        dual_tail_shrink_scale=args.dual_tail_shrink_scale,
        sharpness_center=args.sharpness_center,
        sharpness_scale=args.sharpness_scale,
    )
    rebuilt_model.load_state_dict(base_model.state_dict(), strict=False)
    rebuilt_model.set_rebuilt_dual_negative_bank(
        dual_negative_prototypes=rebuilt_dual,
        dual_negative_support=rebuilt_dual_support,
    )
    summary_rows = [
        {"variant": variant, "bank": "dual_negative", **stats},
        {"variant": variant, **rebuilt_model.rebuild_summary()},
    ]
    return rebuilt_model, pd.DataFrame(summary_rows)


def _fit_and_score(
    train_cache: dict[str, torch.Tensor],
    train_meta: pd.DataFrame,
    eval_caches: list[dict[str, torch.Tensor]],
    eval_meta: list[pd.DataFrame],
    *,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features_raw, train_risk = _feature_sets(train_cache, train_meta)
    train_target = _labels(train_meta)
    train_weights = _weights(train_meta, train_target)
    train_hard_negative = torch.as_tensor(train_meta["harmful_teacher_bank_case"].to_numpy(copy=True), dtype=torch.bool)

    summary_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    rebuild_frames: list[pd.DataFrame] = []
    device = torch.device(args.device)

    for variant in VARIANTS:
        base_variant = (
            "prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid"
            if variant.endswith("_hybrid")
            else "prototype_branchwise_max_negative_cleanup_support_agree_mix"
        )
        base_model = _fit_head(
            train_features_raw,
            train_risk,
            train_target,
            train_weights,
            train_hard_negative,
            variant=base_variant,
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
        model, rebuild_frame = _rebuild_model(base_model, train_features_raw, train_meta, args=args, variant=variant)
        rebuild_frames.append(rebuild_frame)
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
        pd.concat(rebuild_frames, ignore_index=True),
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
    ax.set_title("Dual teacher-rebuilt negative-bank branchwise-max support agreement mixture")
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

    summary, decisions, rebuild = _fit_and_score(train_cache, train_meta, eval_caches, eval_meta, args=args)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    decisions.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    rebuild.to_csv(output_prefix.with_name(output_prefix.name + "_rebuild.csv"), index=False)
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
        "rebuild": rebuild.to_dict(orient="records"),
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
