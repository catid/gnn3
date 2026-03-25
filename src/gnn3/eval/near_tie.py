from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class NearTieBatchStats:
    oracle_gap: torch.Tensor
    oracle_gap_ratio: torch.Tensor
    model_margin: torch.Tensor
    slack_ratio: torch.Tensor
    mean_queue: torch.Tensor
    max_path_length: torch.Tensor
    any_feasible: torch.Tensor
    hard_near_tie: torch.Tensor


def valid_candidate_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["candidate_mask"] & batch["node_mask"]


def model_margin(selection_scores: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    valid_mask = valid_candidate_mask(batch)
    masked_scores = selection_scores.masked_fill(~valid_mask, -1e9)
    topk = masked_scores.topk(k=min(2, masked_scores.size(1)), dim=-1).values
    if topk.size(1) == 1:
        return torch.zeros((selection_scores.size(0),), device=selection_scores.device, dtype=selection_scores.dtype)
    return (topk[:, 0] - topk[:, 1]).clamp_min(0.0)


def oracle_gap_stats(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = valid_candidate_mask(batch)
    feasible_mask = valid_mask & (batch["candidate_on_time"] > 0.5)
    reference_mask = torch.where(feasible_mask.any(dim=-1, keepdim=True), feasible_mask, valid_mask)
    masked_costs = batch["candidate_cost_to_go"].masked_fill(~reference_mask, 1e9)
    topk = masked_costs.topk(k=min(2, masked_costs.size(1)), largest=False, dim=-1).values
    best_cost = topk[:, 0]
    if topk.size(1) == 1:
        second_cost = topk[:, 0]
    else:
        second_cost = topk[:, 1]
    gap = (second_cost - best_cost).clamp_min(0.0)
    gap_ratio = gap / best_cost.abs().clamp_min(1e-6)
    return best_cost, gap, gap_ratio


def near_tie_batch_stats(
    batch: dict[str, torch.Tensor],
    selection_scores: torch.Tensor,
    *,
    oracle_gap_threshold: float,
    slack_ratio_threshold: float,
    packet_min: int,
    mean_queue_threshold: float = 2.5,
    path_length_threshold: int = 5,
) -> NearTieBatchStats:
    valid_mask = valid_candidate_mask(batch)
    any_feasible = (valid_mask & (batch["candidate_on_time"] > 0.5)).any(dim=-1)
    _best_cost, gap, gap_ratio = oracle_gap_stats(batch)
    best_slack = batch["candidate_slack"].masked_fill(~valid_mask, -1e9).max(dim=-1).values
    slack_ratio = best_slack / batch["packet_deadline"].clamp(min=1.0)
    mean_queue = (batch["node_features"][..., 0] * 10.0).sum(dim=-1) / batch["node_mask"].sum(dim=-1).clamp(min=1)
    max_path_length = batch["candidate_path_mask"].sum(dim=-1).amax(dim=-1)
    hard_condition_count = (
        (slack_ratio <= slack_ratio_threshold).long()
        + (batch["packet_count"] >= packet_min).long()
        + (mean_queue >= mean_queue_threshold).long()
        + (max_path_length >= path_length_threshold).long()
    )
    hard_near_tie = any_feasible & (gap <= oracle_gap_threshold) & (hard_condition_count >= 2)
    return NearTieBatchStats(
        oracle_gap=gap,
        oracle_gap_ratio=gap_ratio,
        model_margin=model_margin(selection_scores, batch),
        slack_ratio=slack_ratio,
        mean_queue=mean_queue,
        max_path_length=max_path_length,
        any_feasible=any_feasible,
        hard_near_tie=hard_near_tie,
    )


def build_candidate_feature_tensor(
    batch: dict[str, torch.Tensor],
    *,
    probe_features: torch.Tensor,
    selection_scores: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    batch_size, max_nodes = batch["candidate_mask"].shape
    batch_index = torch.arange(batch_size, device=selection_scores.device)
    current_edges = batch["edge_features"][batch_index, batch["current_node"]]
    path_lengths = batch["candidate_path_mask"].sum(dim=-1, keepdim=True).float() / max(float(max_nodes), 1.0)
    deadline = (batch["packet_deadline"] / 20.0)[:, None, None]
    priority = (batch["packet_priority"] / 3.0)[:, None, None]
    packet_count = batch["packet_count"].float()[:, None, None] / 8.0
    margin = model_margin(selection_scores, batch)[:, None, None]
    value_expand = values[:, None, None]
    probe_expand = probe_features[:, None, :].expand(-1, max_nodes, -1)
    return torch.cat(
        [
            probe_expand,
            batch["node_features"],
            current_edges,
            batch["candidate_path_features"],
            selection_scores.unsqueeze(-1),
            path_lengths,
            deadline.expand(-1, max_nodes, -1),
            priority.expand(-1, max_nodes, -1),
            packet_count.expand(-1, max_nodes, -1),
            margin.expand(-1, max_nodes, -1),
            value_expand.expand(-1, max_nodes, -1),
        ],
        dim=-1,
    )


class NearTieCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        risk_heads: bool = False,
        adapter_dim: int = 0,
    ) -> None:
        super().__init__()
        self.risk_heads = risk_heads
        self.adapter_dim = max(int(adapter_dim), 0)
        if self.adapter_dim > 0:
            self.adapter_down = nn.Linear(input_dim, self.adapter_dim, bias=False)
            self.adapter_up = nn.Linear(self.adapter_dim, input_dim, bias=False)
        else:
            self.adapter_down = None
            self.adapter_up = None
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cost_head = nn.Linear(hidden_dim, 1)
        self.miss_head = nn.Linear(hidden_dim, 1) if risk_heads else None
        self.tail_head = nn.Linear(hidden_dim, 1) if risk_heads else None

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.adapter_down is not None and self.adapter_up is not None:
            features = features + self.adapter_up(F.gelu(self.adapter_down(features)))
        hidden = self.backbone(features)
        outputs = {
            "pred_cost": F.softplus(self.cost_head(hidden).squeeze(-1)),
        }
        if self.miss_head is not None and self.tail_head is not None:
            outputs["pred_miss_logit"] = self.miss_head(hidden).squeeze(-1)
            outputs["pred_tail"] = F.softplus(self.tail_head(hidden).squeeze(-1))
        return outputs


def critic_selection_scores(
    critic_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    miss_weight: float,
    tail_weight: float,
) -> torch.Tensor:
    deadline_scale = batch["packet_deadline"][:, None].clamp(min=1.0)
    score = -(critic_output["pred_cost"] / deadline_scale)
    if "pred_miss_logit" in critic_output:
        score = score + miss_weight * F.logsigmoid(-critic_output["pred_miss_logit"])
    if "pred_tail" in critic_output:
        score = score - tail_weight * critic_output["pred_tail"]
    return score.masked_fill(~valid_candidate_mask(batch), -1e9)


def critic_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    best_cost, _gap, _ratio = oracle_gap_stats(batch)
    deadline_scale = batch["packet_deadline"][:, None].clamp(min=1.0)
    tail_target = F.relu(-batch["candidate_slack"] / deadline_scale)
    return {
        "cost": batch["candidate_cost_to_go"] / deadline_scale,
        "miss": 1.0 - batch["candidate_on_time"],
        "tail": tail_target,
        "regret_delta": (batch["candidate_cost_to_go"] - best_cost[:, None]).clamp_min(0.0) / deadline_scale,
    }
