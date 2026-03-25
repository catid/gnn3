#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from gnn3.eval.near_tie import NearTieCritic
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pt", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--variant", choices=["scalar_q", "risk_multi", "pairwise_rank", "late_unfreeze"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gate-margin", type=float, default=0.10)
    parser.add_argument("--adapter-dim", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _suite_tensors(payload: dict[str, dict[str, torch.Tensor]], suite: str) -> dict[str, torch.Tensor]:
    return {key: value.float() if value.dtype.is_floating_point else value for key, value in payload[suite].items()}


def _decision_summary(frame: pd.DataFrame, *, critic_scores: torch.Tensor, gate_margin: float) -> pd.DataFrame:
    rows = []
    for (suite, episode_index, decision_index), group in frame.groupby(["suite", "episode_index", "decision_index"], sort=False):
        scores = critic_scores[group.index.to_numpy(copy=True)]
        gate = bool(float(group["model_margin"].iloc[0]) <= gate_margin)
        chosen_row = group.iloc[int(torch.argmax(scores).item())] if gate else group.loc[group["is_predicted"] == 1].iloc[0]
        base_row = group.loc[group["is_predicted"] == 1].iloc[0]
        rows.append(
            {
                "suite": suite,
                "episode_index": int(episode_index),
                "decision_index": int(decision_index),
                "hard_near_tie_intersection_case": bool(group["hard_near_tie_intersection_case"].iloc[0]),
                "baseline_error_hard_near_tie_case": bool(group["baseline_error_hard_near_tie_case"].iloc[0]),
                "gate_triggered": gate,
                "action_agreement": int(chosen_row["candidate_node"]) == int(base_row["candidate_node"]),
                "base_target_match": bool(base_row["is_target"]),
                "candidate_target_match": bool(chosen_row["is_target"]),
                "base_regret_delta": float(base_row["regret_delta"]),
                "candidate_regret_delta": float(chosen_row["regret_delta"]),
                "base_miss": float(base_row["miss"]),
                "candidate_miss": float(chosen_row["miss"]),
            }
        )
    return pd.DataFrame(rows)


def _decision_metrics(decision_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "overall": pd.Series([True] * len(decision_df)),
        "hard_near_tie": decision_df["hard_near_tie_intersection_case"],
        "hard_near_tie_error": decision_df["baseline_error_hard_near_tie_case"],
    }
    for suite, suite_df in decision_df.groupby("suite", sort=False):
        for label, mask in masks.items():
            frame = suite_df.loc[mask.loc[suite_df.index]]
            if frame.empty:
                rows.append(
                    {
                        "suite": suite,
                        "slice": label,
                        "decisions": 0,
                        "trigger_rate": 0.0,
                        "disagreement": 0.0,
                        "base_target_match": 0.0,
                        "candidate_target_match": 0.0,
                        "correction_rate": 0.0,
                        "new_error_rate": 0.0,
                        "net_corrected": 0.0,
                        "regret_delta": 0.0,
                        "miss_delta": 0.0,
                    }
                )
                continue
            corrected = (~frame["base_target_match"]) & frame["candidate_target_match"]
            new_errors = frame["base_target_match"] & (~frame["candidate_target_match"])
            base_wrong = (~frame["base_target_match"]).sum()
            base_right = frame["base_target_match"].sum()
            rows.append(
                {
                    "suite": suite,
                    "slice": label,
                    "decisions": len(frame),
                    "trigger_rate": float(frame["gate_triggered"].mean()),
                    "disagreement": float((~frame["action_agreement"]).mean()),
                    "base_target_match": float(frame["base_target_match"].mean()),
                    "candidate_target_match": float(frame["candidate_target_match"].mean()),
                    "correction_rate": float(corrected.mean()) if len(frame) else 0.0,
                    "new_error_rate": float(new_errors.mean()) if len(frame) else 0.0,
                    "net_corrected": float(corrected.sum() - new_errors.sum()),
                    "regret_delta": float((frame["candidate_regret_delta"] - frame["base_regret_delta"]).mean()),
                    "miss_delta": float((frame["candidate_miss"] - frame["base_miss"]).mean()),
                    "baseline_error_recovery": float(corrected.sum() / max(int(base_wrong), 1)),
                    "baseline_success_break": float(new_errors.sum() / max(int(base_right), 1)),
                }
            )
    return pd.DataFrame(rows)


def _train_scalar_or_risk(
    *,
    train_x: torch.Tensor,
    train_cost: torch.Tensor,
    train_miss: torch.Tensor,
    train_tail: torch.Tensor,
    train_mask: torch.Tensor,
    risk_heads: bool,
    epochs: int,
    lr: float,
    hidden_dim: int,
    adapter_dim: int = 0,
    warmup_epochs: int = 0,
) -> NearTieCritic:
    critic = NearTieCritic(
        train_x.size(1),
        hidden_dim=hidden_dim,
        risk_heads=risk_heads,
        adapter_dim=adapter_dim,
    ).to(train_x.device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)
    mask = train_mask.bool()
    adapter_params = []
    if critic.adapter_down is not None and critic.adapter_up is not None:
        adapter_params = list(critic.adapter_down.parameters()) + list(critic.adapter_up.parameters())
        for parameter in adapter_params:
            parameter.requires_grad = False
    for _ in range(epochs):
        if adapter_params and _ == max(int(warmup_epochs), 0):
            for parameter in adapter_params:
                parameter.requires_grad = True
        optimizer.zero_grad(set_to_none=True)
        output = critic(train_x)
        loss = F.smooth_l1_loss(output["pred_cost"][mask], train_cost[mask])
        if risk_heads:
            loss = loss + F.binary_cross_entropy_with_logits(output["pred_miss_logit"][mask], train_miss[mask])
            loss = loss + F.smooth_l1_loss(output["pred_tail"][mask], train_tail[mask])
        loss.backward()
        optimizer.step()
    return critic


def _train_pairwise(
    *,
    train_x: torch.Tensor,
    train_cost: torch.Tensor,
    train_mask: torch.Tensor,
    epochs: int,
    lr: float,
    hidden_dim: int,
) -> NearTieCritic:
    critic = NearTieCritic(train_x.size(1), hidden_dim=hidden_dim, risk_heads=False).to(train_x.device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        output = critic(train_x)
        pred_cost = output["pred_cost"]
        losses = []
        for decision_id in train_mask.unique(sorted=False).tolist():
            mask = train_mask == decision_id
            if int(mask.sum()) < 2:
                continue
            costs = train_cost[mask]
            scores = -pred_cost[mask]
            best_index = int(costs.argmin().item())
            target_margin = costs - costs[best_index]
            pair_loss = F.cross_entropy(scores[None, :], torch.tensor([best_index], device=train_x.device))
            pair_loss = pair_loss + 0.05 * target_margin.mean()
            losses.append(pair_loss)
        if not losses:
            break
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    return critic


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    payload = torch.load(args.dataset_pt, map_location="cpu")
    metadata = pd.read_csv(args.metadata_csv)

    train_frames = []
    eval_frames = []
    train_features = []
    train_cost = []
    train_miss = []
    train_tail = []
    train_decision_ids = []
    for suite, suite_df in metadata.groupby("suite", sort=False):
        tensors = _suite_tensors(payload, suite)
        suite_df = suite_df.reset_index(drop=True)
        split = str(suite_df["split"].iloc[0])
        if split == "train":
            train_frames.append(suite_df)
            train_features.append(tensors["features"])
            train_cost.append(tensors["cost"])
            train_miss.append(tensors["miss"])
            train_tail.append(tensors["tail"])
            decision_ids = torch.as_tensor(
                [hash((suite, int(r.episode_index), int(r.decision_index))) for r in suite_df.itertuples(index=False)],
                dtype=torch.long,
            )
            train_decision_ids.append(decision_ids)
        else:
            eval_frames.append((suite, suite_df, tensors))

    train_frame = pd.concat(train_frames, ignore_index=True)
    train_x = torch.cat(train_features, dim=0)
    train_cost_t = torch.cat(train_cost, dim=0)
    train_miss_t = torch.cat(train_miss, dim=0)
    train_tail_t = torch.cat(train_tail, dim=0)
    train_mask = torch.as_tensor(train_frame["hard_near_tie_intersection_case"].to_numpy(copy=True), dtype=torch.bool)

    feature_mean = train_x.mean(dim=0, keepdim=True)
    feature_std = train_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_x = (train_x - feature_mean) / feature_std
    train_x = train_x.to(device)
    train_cost_t = train_cost_t.to(device)
    train_miss_t = train_miss_t.to(device)
    train_tail_t = train_tail_t.to(device)
    train_mask = train_mask.to(device)

    if args.variant == "scalar_q":
        critic = _train_scalar_or_risk(
            train_x=train_x,
            train_cost=train_cost_t,
            train_miss=train_miss_t,
            train_tail=train_tail_t,
            train_mask=train_mask,
            risk_heads=False,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
        )
    elif args.variant == "risk_multi":
        critic = _train_scalar_or_risk(
            train_x=train_x,
            train_cost=train_cost_t,
            train_miss=train_miss_t,
            train_tail=train_tail_t,
            train_mask=train_mask,
            risk_heads=True,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
        )
    elif args.variant == "late_unfreeze":
        critic = _train_scalar_or_risk(
            train_x=train_x,
            train_cost=train_cost_t,
            train_miss=train_miss_t,
            train_tail=train_tail_t,
            train_mask=train_mask,
            risk_heads=True,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            adapter_dim=args.adapter_dim,
            warmup_epochs=args.warmup_epochs,
        )
    else:
        critic = _train_pairwise(
            train_x=train_x,
            train_cost=train_cost_t,
            train_mask=torch.cat(train_decision_ids, dim=0).to(device),
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
        )

    decision_frames: list[pd.DataFrame] = []
    summary_rows: list[pd.DataFrame] = []
    for _suite, suite_df, tensors in eval_frames:
        suite_x = ((tensors["features"] - feature_mean) / feature_std).to(device)
        with torch.no_grad():
            output = critic(suite_x)
            if args.variant == "risk_multi":
                scores = -(output["pred_cost"]) + 2.0 * F.logsigmoid(-output["pred_miss_logit"]) - 0.5 * output["pred_tail"]
            else:
                scores = -output["pred_cost"]
        suite_frame = suite_df.copy()
        suite_frame["critic_score"] = scores.detach().cpu().numpy()
        suite_frame["cost"] = tensors["cost"].cpu().numpy()
        suite_frame["miss"] = tensors["miss"].cpu().numpy()
        suite_frame["tail"] = tensors["tail"].cpu().numpy()
        suite_frame["regret_delta"] = tensors["regret_delta"].cpu().numpy()
        decision_df = _decision_summary(suite_frame, critic_scores=scores.detach().cpu(), gate_margin=args.gate_margin)
        decision_frames.append(decision_df)
        summary_rows.append(_decision_metrics(decision_df))

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    decisions_df = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()

    torch.save(
        {
            "variant": args.variant,
            "critic_state_dict": critic.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "hidden_dim": args.hidden_dim,
            "risk_heads": args.variant in {"risk_multi", "late_unfreeze"},
            "adapter_dim": args.adapter_dim if args.variant == "late_unfreeze" else 0,
            "gate_margin": args.gate_margin,
        },
        output_dir / "critic.pt",
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    decisions_df.to_csv(output_dir / "decisions.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "dataset_pt": args.dataset_pt,
                "metadata_csv": args.metadata_csv,
                "epochs": args.epochs,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "gate_margin": args.gate_margin,
                "adapter_dim": args.adapter_dim if args.variant == "late_unfreeze" else 0,
                "warmup_epochs": args.warmup_epochs if args.variant == "late_unfreeze" else 0,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
