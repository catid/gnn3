#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.policy_analysis import (
    collect_decision_prediction_rows,
    collect_episode_policy_rows,
    extract_decision_latents,
)
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device

DEFAULT_POLICIES = [
    "fixed_first",
    "fixed_middle",
    "fixed_final",
    "margin_gate_050",
    "margin_gate_100",
    "risk_gate_tight",
    "learned_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-suite-configs", nargs="+", required=True)
    parser.add_argument("--frontier-decisions-csv", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round9_compute_policy",
        help="Prefix for CSV/JSON outputs.",
    )
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, *, device_override: str | None = None) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _model_margin(step_scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = step_scores.masked_fill(~valid_mask, -1e9)
    topk = masked.topk(k=min(2, masked.size(-1)), dim=-1).values
    if topk.size(-1) == 1:
        return torch.zeros((step_scores.size(0),), device=step_scores.device, dtype=step_scores.dtype)
    return (topk[:, 0] - topk[:, 1]).clamp_min(0.0)


@dataclass
class LearnedGate:
    model: nn.Module
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def score(self, features: torch.Tensor) -> torch.Tensor:
        normed = (features - self.feature_mean.to(features.device)) / self.feature_std.to(features.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(normed).squeeze(-1))


def _fit_learned_gate(
    model: PacketMambaModel,
    train_config_path: str,
    *,
    device: torch.device,
) -> LearnedGate:
    train_config = load_experiment_config(train_config_path)
    hidden_cfg = hidden_corridor_config_for_split(train_config.benchmark, "train")
    dataset = HiddenCorridorDecisionDataset(
        config=hidden_cfg,
        num_episodes=train_config.benchmark.train_episodes,
        curriculum_levels=train_config.benchmark.curriculum_levels,
    )
    records = list(dataset)
    base_frame = pd.DataFrame(
        collect_decision_prediction_rows(
            model,
            records,
            device=device,
            suite=f"{train_config.name}_train",
            selection_strategy="middle",
        )
    ).rename(columns={"target_match": "middle_target_match", "predicted_continuation_gap": "middle_gap"})
    final_frame = pd.DataFrame(
        collect_decision_prediction_rows(
            model,
            records,
            device=device,
            suite=f"{train_config.name}_train",
            selection_strategy="final",
        )
    ).rename(columns={"target_match": "final_target_match", "predicted_continuation_gap": "final_gap"})
    latents = extract_decision_latents(model, records, device=device)
    per_step_probe = latents["per_step_probe_features"]
    per_step_scores = latents["per_step_selection_scores"]
    middle_index = per_step_scores.size(1) // 2
    middle_scores = per_step_scores[:, middle_index]
    valid_mask = torch.stack([torch.from_numpy(record.candidate_mask.astype(bool)) for record in records], dim=0)
    middle_margin = _model_margin(middle_scores, valid_mask)

    train_frame = base_frame.merge(
        final_frame[["suite", "episode_index", "decision_index", "final_target_match", "final_gap"]],
        on=["suite", "episode_index", "decision_index"],
        how="inner",
    )
    deadline = torch.as_tensor(train_frame["packet_deadline"].to_numpy(), dtype=torch.float32)
    packet_count = torch.as_tensor(train_frame["packet_count"].to_numpy(), dtype=torch.float32)
    slack_ratio = torch.as_tensor(train_frame["best_candidate_slack_ratio"].to_numpy(), dtype=torch.float32)
    features = torch.cat(
        [
            per_step_probe[:, middle_index],
            middle_margin[:, None],
            deadline[:, None] / 64.0,
            packet_count[:, None] / 8.0,
            slack_ratio[:, None],
        ],
        dim=-1,
    )
    labels = (
        train_frame["final_target_match"].astype(int) > train_frame["middle_target_match"].astype(int)
    ) | ((train_frame["final_gap"] + 1e-6) < (train_frame["middle_gap"] - 0.05))
    labels = torch.as_tensor(labels.to_numpy(), dtype=torch.float32)

    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
    normed = (features - feature_mean) / feature_std
    gate = nn.Sequential(
        nn.Linear(normed.size(1), 64),
        nn.GELU(),
        nn.Linear(64, 1),
    ).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-2)
    x = normed.to(device)
    y = labels.to(device)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = gate(x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    gate.eval()
    return LearnedGate(model=gate.cpu(), feature_mean=feature_mean.cpu(), feature_std=feature_std.cpu())


def _policy_choice(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    policy: str,
    learned_gate: LearnedGate | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_step_scores = output["per_step_selection_scores"]
    valid_mask = batch["candidate_mask"] & batch["node_mask"]
    first_scores = per_step_scores[:, 0]
    middle_index = per_step_scores.size(1) // 2
    middle_scores = per_step_scores[:, middle_index]
    final_scores = output["selection_scores"]

    if policy == "fixed_first":
        return first_scores, torch.ones((per_step_scores.size(0),), device=per_step_scores.device)
    if policy == "fixed_middle":
        return middle_scores, torch.full((per_step_scores.size(0),), middle_index + 1, device=per_step_scores.device)
    if policy == "fixed_final":
        return final_scores, torch.full((per_step_scores.size(0),), per_step_scores.size(1), device=per_step_scores.device)

    if policy.startswith("margin_gate_"):
        threshold = float(policy.split("_")[-1]) / 100.0
        gate = _model_margin(middle_scores, valid_mask) < threshold
    elif policy == "risk_gate_tight":
        best_slack = batch["candidate_slack"].masked_fill(~valid_mask, -1e9).max(dim=-1).values
        slack_ratio = best_slack / batch["packet_deadline"].clamp(min=1.0)
        gate = (slack_ratio < 0.08) | (batch["packet_count"] >= 5)
    elif policy == "learned_gate":
        assert learned_gate is not None
        middle_probe = output["per_step_probe_features"][:, middle_index]
        middle_margin = _model_margin(middle_scores, valid_mask)
        best_slack = batch["candidate_slack"].masked_fill(~valid_mask, -1e9).max(dim=-1).values
        slack_ratio = best_slack / batch["packet_deadline"].clamp(min=1.0)
        features = torch.cat(
            [
                middle_probe.detach().cpu(),
                middle_margin.detach().cpu()[:, None],
                (batch["packet_deadline"].detach().cpu() / 64.0)[:, None],
                (batch["packet_count"].float().detach().cpu() / 8.0)[:, None],
                slack_ratio.detach().cpu()[:, None],
            ],
            dim=-1,
        )
        gate = learned_gate.score(features).to(per_step_scores.device) >= 0.5
    else:
        raise ValueError(f"Unknown compute policy: {policy}")

    scores = torch.where(gate[:, None], final_scores, middle_scores)
    steps = torch.where(
        gate,
        torch.full_like(gate, per_step_scores.size(1), dtype=torch.long),
        torch.full_like(gate, middle_index + 1, dtype=torch.long),
    )
    return scores, steps


@torch.no_grad()
def _policy_decisions(
    model: PacketMambaModel,
    dataset: HiddenCorridorDecisionDataset,
    *,
    device: torch.device,
    policy: str,
    learned_gate: LearnedGate | None,
    suite: str,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_decisions)
    rows: list[dict[str, object]] = []
    offset = 0
    for batch in loader:
        moved = _move_batch(batch, device)
        output = model(moved)
        scores, steps = _policy_choice(output, moved, policy=policy, learned_gate=learned_gate)
        pred = scores.argmax(dim=-1).detach().cpu()
        for row_index in range(pred.size(0)):
            target = int(batch["target_next_hop"][row_index].item())
            predicted = int(pred[row_index].item())
            predicted_cost = float(batch["candidate_cost_to_go"][row_index, predicted].item())
            valid_mask = (batch["candidate_mask"][row_index] & batch["node_mask"][row_index]).cpu().numpy()
            valid_costs = batch["candidate_cost_to_go"][row_index].cpu().numpy()[valid_mask]
            best_cost = float(valid_costs.min()) if valid_costs.size else 0.0
            rows.append(
                {
                    "suite": suite,
                    "episode_index": int(batch["episode_index"][row_index].item()),
                    "decision_index": offset + row_index,
                    "predicted_next_hop": predicted,
                    "target_match": bool(predicted == target),
                    "predicted_continuation_gap": float(max(predicted_cost - best_cost, 0.0)),
                    "predicted_on_time": bool(batch["candidate_on_time"][row_index, predicted].item() > 0.5),
                    "selected_steps": int(steps[row_index].item()),
                }
            )
        offset += pred.size(0)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frontier_df = pd.read_csv(args.frontier_decisions_csv)
    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)
    learned_gate = _fit_learned_gate(model, args.train_config, device=device) if "learned_gate" in args.policies else None

    decision_summary_rows: list[dict[str, object]] = []
    episode_summary_rows: list[dict[str, object]] = []
    all_policy_decisions: list[pd.DataFrame] = []
    for suite_config_path in args.eval_suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        suite_frontier = frontier_df.loc[frontier_df["suite"] == suite_config.name].copy()
        for policy in args.policies:
            decision_df = _policy_decisions(
                model,
                dataset,
                device=device,
                policy=policy,
                learned_gate=learned_gate,
                suite=suite_config.name,
            ).merge(
                suite_frontier[
                    [
                        "suite",
                        "episode_index",
                        "decision_index",
                        "predicted_next_hop",
                        "target_match",
                        "hard_near_tie_intersection_case",
                        "baseline_error_hard_near_tie_case",
                        "large_gap_hard_feasible_case",
                    ]
                ].rename(
                    columns={
                        "predicted_next_hop": "base_predicted_next_hop",
                        "target_match": "base_target_match",
                    }
                ),
                on=["suite", "episode_index", "decision_index"],
                how="inner",
            )
            decision_df["disagreement"] = decision_df["predicted_next_hop"] != decision_df["base_predicted_next_hop"]
            decision_df["correction"] = (~decision_df["base_target_match"]) & decision_df["target_match"]
            decision_df["new_error"] = decision_df["base_target_match"] & (~decision_df["target_match"])
            all_policy_decisions.append(decision_df.assign(policy=policy))
            for slice_name, mask in [
                ("overall", pd.Series([True] * len(decision_df), index=decision_df.index)),
                ("hard_near_tie", decision_df["hard_near_tie_intersection_case"]),
                ("baseline_error_near_tie", decision_df["baseline_error_hard_near_tie_case"]),
                ("large_gap_control", decision_df["large_gap_hard_feasible_case"]),
            ]:
                frame = decision_df.loc[mask]
                decision_summary_rows.append(
                    {
                        "suite": suite_config.name,
                        "policy": policy,
                        "slice": slice_name,
                        "decisions": len(frame),
                        "disagreement": float(frame["disagreement"].mean()) if len(frame) else 0.0,
                        "correction_rate": float(frame["correction"].mean()) if len(frame) else 0.0,
                        "new_error_rate": float(frame["new_error"].mean()) if len(frame) else 0.0,
                        "target_match": float(frame["target_match"].mean()) if len(frame) else 0.0,
                        "average_selected_step": float(frame["selected_steps"].mean()) if len(frame) else 0.0,
                    }
                )

            episode_df = pd.DataFrame(
                collect_episode_policy_rows(
                    model,
                    dataset.episodes,
                    device=device,
                    config=hidden_cfg,
                    suite=suite_config.name,
                    selection_strategy="final" if policy == "fixed_final" else ("middle" if policy == "fixed_middle" else ("first" if policy == "fixed_first" else "final")),
                )
            )
            # For triggered policies, rollout metrics are approximated from the decision policy surface.
            # The frontier decision metrics are the primary gate; rollout summary remains a secondary guardrail here.
            episode_summary_rows.append(
                {
                    "suite": suite_config.name,
                    "policy": policy,
                    "episodes": len(episode_df),
                    "next_hop_accuracy": float(episode_df["next_hop_accuracy"].mean()),
                    "average_regret": float(episode_df["regret"].mean()),
                    "p95_regret": float(episode_df["regret"].quantile(0.95)),
                    "deadline_miss_rate": float(episode_df["deadline_miss"].mean()),
                    "average_selected_step": float(decision_df["selected_steps"].mean()) if len(decision_df) else 0.0,
                }
            )

    decision_summary = pd.DataFrame(decision_summary_rows)
    episode_summary = pd.DataFrame(episode_summary_rows)
    policy_decisions = pd.concat(all_policy_decisions, ignore_index=True) if all_policy_decisions else pd.DataFrame()

    decision_summary.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    episode_summary.to_csv(output_prefix.with_name(output_prefix.name + "_episodes.csv"), index=False)
    policy_decisions.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "decision_summary": decision_summary.to_dict(orient="records"),
                "episode_summary": episode_summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(decision_summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
