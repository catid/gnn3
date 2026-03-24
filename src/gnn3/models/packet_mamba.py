from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from gnn3.data.hidden_corridor import ROLE_NAMES, ROLE_TO_ID


def batched_index_select(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    expanded_index = index[:, None, None].expand(-1, 1, x.size(-1))
    return torch.gather(x, 1, expanded_index).squeeze(1)


def gather_ordered(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, order[..., None].expand(-1, -1, x.size(-1)))


def scatter_ordered(x: torch.Tensor, order: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros((x.size(0), num_nodes, x.size(-1)), device=x.device, dtype=x.dtype)
    out.scatter_(1, order[..., None].expand(-1, -1, x.size(-1)), x)
    return out


@dataclass(frozen=True)
class PacketMambaConfig:
    node_feature_dim: int = 13
    edge_feature_dim: int = 4
    d_model: int = 128
    d_state: int = 16
    inner_layers: int = 2
    outer_steps: int = 1
    dropout: float = 0.1
    router_variant: str = "local"
    role_conditioned: bool = False
    shared_transition: bool = False
    final_step_only_loss: bool = True
    detach_warmup: bool = False
    penultimate_grad_prob: float = 0.0
    history_read: bool = False
    history_read_mode: str = "dense_nodes"
    max_history_steps: int = 8
    deadline_head: bool = False
    risk_aware_scoring: bool = False
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    on_time_score_weight: float = 0.75
    slack_score_weight: float = 0.25
    tail_score_weight: float = 0.5
    verifier_aux_last_k_steps: int = 1
    hazard_memory: bool = False
    hazard_memory_dim: int = 16
    path_reranker: bool = False
    path_reranker_weight: float = 1.0
    path_reranker_bound: float = 0.0
    path_reranker_traffic_gate: bool = False
    path_reranker_gate_bias: float = 1.25
    path_reranker_gate_sharpness: float = 4.0
    path_reranker_packet_weight: float = 0.35
    path_verifier_filter: bool = False
    path_verifier_hard_mask: bool = True
    path_verifier_slack_margin: float = 0.0
    path_verifier_penalty: float = 4.0


class DenseEdgeMixer(nn.Module):
    def __init__(self, d_model: int, edge_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.src_proj = nn.Linear(d_model, d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model + edge_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.out_proj = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.norm(x)
        send = self.src_proj(normed).unsqueeze(1).expand(-1, x.size(1), -1, -1)
        recv = normed.unsqueeze(2).expand(-1, -1, x.size(1), -1)
        gate_input = torch.cat([recv, send, edge_features], dim=-1)
        gates = torch.sigmoid(self.gate_mlp(gate_input))
        edge_mask = adjacency.unsqueeze(-1) & node_mask.unsqueeze(1).unsqueeze(-1)
        gated = torch.where(edge_mask, gates * send, torch.zeros_like(send))
        degree = adjacency.sum(dim=-1, keepdim=True).clamp(min=1)
        messages = gated.sum(dim=2) / degree
        update = self.out_proj(torch.cat([x, messages], dim=-1))
        return x + self.dropout(update)


class SelectiveRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        variant: str,
        role_conditioned: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.role_conditioned = role_conditioned or variant == "hetero"
        self.norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.sender_proj = nn.Linear(d_model, d_model)
        self.payload_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.read_gate = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.role_embeddings = nn.Embedding(len(ROLE_NAMES), d_model)

    def _source_mask(self, node_roles: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        mask = node_mask
        if self.variant == "memory_hubs":
            memory_roles = torch.isin(
                node_roles,
                torch.tensor(
                    [ROLE_TO_ID["hub"], ROLE_TO_ID["monitor"]],
                    device=node_roles.device,
                ),
            )
            mask = mask & memory_roles
        return mask

    def forward(
        self,
        x: torch.Tensor,
        *,
        node_roles: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.variant == "local":
            zero = x.new_tensor(0.0)
            return x, {"read_entropy": zero, "monitor_inflow": zero, "hub_inflow": zero}

        normed = self.norm(x)
        role_bias = self.role_embeddings(node_roles) if self.role_conditioned else 0.0
        query = self.query_proj(normed + role_bias)
        key = self.key_proj(normed + role_bias)
        sender = self.sender_proj(normed + role_bias)
        payload = self.payload_proj(normed)

        read_logits = torch.einsum("bid,bjd->bij", query, key) / math.sqrt(x.size(-1))
        forward_logits = torch.einsum("bid,bjd->bij", role_bias + sender, role_bias + sender)
        if self.variant == "selective_read":
            logits = read_logits
        elif self.variant == "selective_forward":
            logits = forward_logits
        else:
            logits = read_logits + 0.5 * forward_logits

        source_mask = self._source_mask(node_roles, node_mask)
        pair_mask = node_mask[:, :, None] & source_mask[:, None, :]
        eye = torch.eye(x.size(1), device=x.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask | (eye & node_mask[:, :, None] & node_mask[:, None, :])
        logits = logits.masked_fill(~pair_mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(pair_mask, weights, torch.zeros_like(weights))
        message = torch.einsum("bij,bjd->bid", weights, payload)
        gate = torch.sigmoid(self.read_gate(torch.cat([x, message], dim=-1)))
        update = self.out_proj(gate * message)

        role_hub = node_roles == ROLE_TO_ID["hub"]
        role_monitor = node_roles == ROLE_TO_ID["monitor"]
        weight_eps = 1e-8
        entropy = -(weights * (weights + weight_eps).log()).sum(dim=-1)
        monitor_inflow = weights[..., role_monitor.any(dim=0)] if role_monitor.any() else None
        hub_inflow = weights[..., role_hub.any(dim=0)] if role_hub.any() else None
        diagnostics = {
            "read_entropy": entropy.mean(),
            "monitor_inflow": monitor_inflow.mean() if monitor_inflow is not None else x.new_tensor(0.0),
            "hub_inflow": hub_inflow.mean() if hub_inflow is not None else x.new_tensor(0.0),
            "weights": weights,
        }
        return x + self.dropout(update), diagnostics


class Mamba3LikeScan(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.delta_proj = nn.Linear(d_model, d_model)
        self.b_proj = nn.Linear(d_model, d_model * d_state)
        self.c_proj = nn.Linear(d_model, d_model * d_state)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.skip = nn.Parameter(torch.ones(d_model))
        self.log_a = nn.Parameter(torch.zeros(d_model, d_state))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        normed = self.norm(x)
        delta = F.softplus(self.delta_proj(normed)) + 1e-3
        b_term = self.b_proj(normed).view(batch_size, seq_len, d_model, self.d_state)
        c_term = self.c_proj(normed).view(batch_size, seq_len, d_model, self.d_state)
        gate = torch.sigmoid(self.gate_proj(normed))
        a = -F.softplus(self.log_a) - 1e-4
        state = x.new_zeros((batch_size, d_model, self.d_state))
        outputs: list[torch.Tensor] = []

        for step in range(seq_len):
            mask_t = sequence_mask[:, step].view(batch_size, 1, 1)
            alpha = torch.exp(delta[:, step].unsqueeze(-1) * a.unsqueeze(0))
            beta = (1.0 - alpha) / (-a.unsqueeze(0) + 1e-4)
            updated = alpha * state + beta * b_term[:, step]
            state = torch.where(mask_t, updated, state)
            y_t = (c_term[:, step] * state).sum(dim=-1) + self.skip * normed[:, step]
            outputs.append(self.out_proj(gate[:, step] * y_t))

        out = torch.stack(outputs, dim=1)
        return x + self.dropout(out)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ff(self.norm(x)))


class OuterHistoryReader(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_history_steps: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.round_embedding = nn.Embedding(max_history_steps, d_model)
        self.summary_type_embedding = nn.Embedding(3, d_model)
        self.dropout = nn.Dropout(dropout)

    def _read_dense_nodes(
        self,
        current_state: torch.Tensor,
        history_states: list[torch.Tensor],
        *,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, num_nodes, d_model = current_state.shape
        history_tensor = torch.stack(history_states[-self.round_embedding.num_embeddings :], dim=1)
        history_steps = history_tensor.size(1)
        step_ids = torch.arange(history_steps, device=current_state.device)
        step_embed = self.round_embedding(step_ids)[None, :, None, :]
        history_norm = self.norm(history_tensor)
        memory = history_norm + step_embed
        memory = memory.view(batch_size, history_steps * num_nodes, d_model)

        query = self.query_proj(self.norm(current_state))
        key = self.key_proj(memory)
        value = self.value_proj(memory)

        logits = torch.einsum("bid,bmd->bim", query, key) / math.sqrt(d_model)
        history_mask = node_mask[:, None, :].expand(-1, history_steps, -1).reshape(batch_size, 1, -1)
        pair_mask = node_mask[:, :, None] & history_mask
        logits = logits.masked_fill(~pair_mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(pair_mask, weights, torch.zeros_like(weights))
        message = torch.einsum("bim,bmd->bid", weights, value)
        gate = torch.sigmoid(self.gate_proj(torch.cat([current_state, message], dim=-1)))
        update = self.out_proj(gate * message)

        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        latest_slice = slice((history_steps - 1) * num_nodes, history_steps * num_nodes)
        latest_share = weights[..., latest_slice].sum(dim=-1).mean()
        diagnostics = {
            "history_read_entropy": entropy,
            "history_latest_round_share": latest_share,
            "history_hub_bank_share": current_state.new_tensor(0.0),
            "history_monitor_bank_share": current_state.new_tensor(0.0),
            "history_global_bank_share": current_state.new_tensor(0.0),
        }
        return current_state + self.dropout(update), diagnostics

    @staticmethod
    def _masked_mean(states: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        summary = torch.einsum("bhnd,bhn->bhd", states, weights) / denom
        valid = weights.sum(dim=-1) > 0
        return summary, valid

    def _read_summary_bank(
        self,
        current_state: torch.Tensor,
        history_states: list[torch.Tensor],
        *,
        node_mask: torch.Tensor,
        node_roles: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, _num_nodes, d_model = current_state.shape
        history_tensor = torch.stack(history_states[-self.round_embedding.num_embeddings :], dim=1)
        history_steps = history_tensor.size(1)
        history_norm = self.norm(history_tensor)

        expanded_mask = node_mask[:, None, :].expand(-1, history_steps, -1)
        hub_mask = expanded_mask & (node_roles[:, None, :] == ROLE_TO_ID["hub"])
        monitor_mask = expanded_mask & (node_roles[:, None, :] == ROLE_TO_ID["monitor"])
        global_mask = expanded_mask

        hub_summary, hub_valid = self._masked_mean(history_norm, hub_mask)
        monitor_summary, monitor_valid = self._masked_mean(history_norm, monitor_mask)
        global_summary, global_valid = self._masked_mean(history_norm, global_mask)

        summary_tensor = torch.stack([hub_summary, monitor_summary, global_summary], dim=2)
        token_valid = torch.stack([hub_valid, monitor_valid, global_valid], dim=-1)

        step_ids = torch.arange(history_steps, device=current_state.device)
        round_embed = self.round_embedding(step_ids)[None, :, None, :]
        type_embed = self.summary_type_embedding(torch.arange(3, device=current_state.device))[None, None, :, :]
        memory = summary_tensor + round_embed + type_embed
        memory = memory.view(batch_size, history_steps * 3, d_model)
        memory_mask = token_valid.view(batch_size, history_steps * 3)

        query = self.query_proj(self.norm(current_state))
        key = self.key_proj(memory)
        value = self.value_proj(memory)

        logits = torch.einsum("bid,bmd->bim", query, key) / math.sqrt(d_model)
        pair_mask = node_mask[:, :, None] & memory_mask[:, None, :]
        logits = logits.masked_fill(~pair_mask, -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(pair_mask, weights, torch.zeros_like(weights))
        message = torch.einsum("bim,bmd->bid", weights, value)
        gate = torch.sigmoid(self.gate_proj(torch.cat([current_state, message], dim=-1)))
        update = self.out_proj(gate * message)

        weights_by_type = weights.view(batch_size, current_state.size(1), history_steps, 3)
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        latest_share = weights_by_type[:, :, -1, :].sum(dim=-1).mean()
        diagnostics = {
            "history_read_entropy": entropy,
            "history_latest_round_share": latest_share,
            "history_hub_bank_share": weights_by_type[..., 0].sum(dim=-1).mean(),
            "history_monitor_bank_share": weights_by_type[..., 1].sum(dim=-1).mean(),
            "history_global_bank_share": weights_by_type[..., 2].sum(dim=-1).mean(),
        }
        return current_state + self.dropout(update), diagnostics

    def forward(
        self,
        current_state: torch.Tensor,
        history_states: list[torch.Tensor],
        *,
        node_mask: torch.Tensor,
        node_roles: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not history_states:
            zero = current_state.new_tensor(0.0)
            return current_state, {
                "history_read_entropy": zero,
                "history_latest_round_share": zero,
                "history_hub_bank_share": zero,
                "history_monitor_bank_share": zero,
                "history_global_bank_share": zero,
            }

        if self.mode == "summary_bank":
            return self._read_summary_bank(
                current_state,
                history_states,
                node_mask=node_mask,
                node_roles=node_roles,
            )
        return self._read_dense_nodes(current_state, history_states, node_mask=node_mask)


class PacketMambaLayer(nn.Module):
    def __init__(self, config: PacketMambaConfig) -> None:
        super().__init__()
        self.edge_mixer = DenseEdgeMixer(config.d_model, config.edge_feature_dim, config.dropout)
        self.router = SelectiveRouter(
            config.d_model,
            variant=config.router_variant,
            role_conditioned=config.role_conditioned,
            dropout=config.dropout,
        )
        self.scan = Mamba3LikeScan(config.d_model, config.d_state, config.dropout)
        self.merge_gate = nn.Linear(3 * config.d_model, config.d_model)
        self.ffn = FeedForwardBlock(config.d_model, config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
        node_roles: torch.Tensor,
        order_current: torch.Tensor,
        order_destination: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.edge_mixer(x, edge_features, adjacency, node_mask)
        x, router_diag = self.router(x, node_roles=node_roles, node_mask=node_mask)
        current_scan = self.scan(gather_ordered(x, order_current), sequence_mask)
        destination_scan = self.scan(gather_ordered(x, order_destination), sequence_mask)
        current_out = scatter_ordered(current_scan, order_current, x.size(1))
        destination_out = scatter_ordered(destination_scan, order_destination, x.size(1))
        merge = torch.sigmoid(self.merge_gate(torch.cat([x, current_out, destination_out], dim=-1)))
        x = x + merge * current_out + (1.0 - merge) * destination_out
        x = self.ffn(x)
        return x, router_diag


class PacketMambaModel(nn.Module):
    def __init__(self, config: PacketMambaConfig) -> None:
        super().__init__()
        self.config = config
        self.node_in = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), config.d_model)
        self.community_embedding = nn.Embedding(5, config.d_model)
        self.layers = nn.ModuleList([PacketMambaLayer(config) for _ in range(config.inner_layers)])
        self.shared_layers = self.layers if config.shared_transition else None
        self.step_layers = (
            nn.ModuleList(
                [nn.ModuleList([PacketMambaLayer(config) for _ in range(config.inner_layers)]) for _ in range(config.outer_steps)]
            )
            if not config.shared_transition
            else None
        )
        self.commit_gate = nn.Linear(2 * config.d_model, config.d_model)
        self.history_reader = (
            OuterHistoryReader(config.d_model, config.dropout, config.max_history_steps, config.history_read_mode)
            if config.history_read
            else None
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        action_input_dim = 3 * config.d_model + config.edge_feature_dim
        self.action_head = nn.Sequential(
            nn.Linear(action_input_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.route_head = nn.Sequential(
            nn.Linear(3 * config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )
        self.deadline_head = (
            nn.Sequential(
                nn.Linear(action_input_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if config.deadline_head
            else None
        )
        self.slack_head = (
            nn.Sequential(
                nn.Linear(action_input_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if config.deadline_head
            else None
        )
        self.quantile_head = (
            nn.Sequential(
                nn.Linear(action_input_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, len(config.quantile_levels)),
            )
            if config.deadline_head
            else None
        )
        self.path_reranker_head = (
            nn.Sequential(
                nn.Linear(4 * config.d_model + 5, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if config.path_reranker
            else None
        )
        self.hazard_encoder = (
            nn.Sequential(
                nn.Linear(7, config.hazard_memory_dim),
                nn.GELU(),
                nn.Linear(config.hazard_memory_dim, config.d_model),
            )
            if config.hazard_memory
            else None
        )

    def _hazard_summary(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node_mask = batch["node_mask"].float()
        denom = node_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        queue_mean = (batch["node_features"][..., 0] * node_mask).sum(dim=-1, keepdim=True) / denom
        blocked_mean = (batch["node_features"][..., 2] * node_mask).sum(dim=-1, keepdim=True) / denom
        hidden_edge_mask = batch["edge_features"][..., 3]
        residual_ratio = batch["edge_features"][..., 1]
        hidden_edge_denom = hidden_edge_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        hidden_pressure = (((1.0 - residual_ratio) * hidden_edge_mask).sum(dim=(1, 2), keepdim=True) / hidden_edge_denom)
        hidden_pressure = hidden_pressure.squeeze(-1)
        candidate_degree = batch["candidate_mask"].float().sum(dim=-1, keepdim=True) / batch["node_mask"].sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1)
        packet_priority = (batch["packet_priority"] / 3.0).unsqueeze(-1)
        packet_deadline = (batch["packet_deadline"] / 64.0).unsqueeze(-1)
        packet_count = (batch["packet_count"].float() / 8.0).unsqueeze(-1)
        summary = torch.cat(
            [
                queue_mean,
                blocked_mean,
                hidden_pressure,
                candidate_degree,
                packet_priority,
                packet_deadline,
                packet_count,
            ],
            dim=-1,
        )
        return summary

    def _transition_layers(self, step: int) -> nn.ModuleList:
        if self.config.shared_transition:
            return self.layers
        assert self.step_layers is not None
        return self.step_layers[step]

    def _path_reranker_gate(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        path_features = batch["candidate_path_features"]
        if not self.config.path_reranker_traffic_gate:
            return torch.ones_like(path_features[..., 0])

        # This deterministic fallback keeps the reranker active on short, low-queue,
        # high-capacity paths, but damps it under the exact high-stress regimes where
        # the combined recipe became unstable in deeper/heavier OOD suites.
        path_length = path_features[..., 0]
        mean_queue = path_features[..., 1]
        max_queue = path_features[..., 2]
        residual_ratio = path_features[..., 3].clamp(min=0.0, max=1.0)
        structural_bonus = path_features[..., 4].clamp(min=0.0, max=1.0)
        packet_pressure = (batch["packet_count"].float() / 8.0)[:, None]
        urgency_pressure = (
            batch["packet_priority"] / batch["packet_deadline"].clamp(min=1.0)
        )[:, None]
        risk_score = (
            path_length
            + 0.5 * mean_queue
            + max_queue
            + (1.0 - residual_ratio)
            + self.config.path_reranker_packet_weight * packet_pressure
            + 0.25 * urgency_pressure
            - 0.25 * structural_bonus
        )
        return torch.sigmoid(
            (self.config.path_reranker_gate_bias - risk_score)
            * self.config.path_reranker_gate_sharpness
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = batch["node_features"].device
        node_mask = batch["node_mask"]
        lengths = node_mask.sum(dim=-1)
        sequence_mask = torch.arange(batch["node_features"].size(1), device=device)[None, :] < lengths[:, None]

        community_ids = batch["node_communities"].clamp(min=-1) + 1
        x = self.node_in(batch["node_features"])
        x = x + self.role_embedding(batch["node_roles"]) + self.community_embedding(community_ids)
        slow_state = torch.zeros_like(x)
        per_step_logits: list[torch.Tensor] = []
        per_step_values: list[torch.Tensor] = []
        per_step_on_time_logits: list[torch.Tensor] = []
        per_step_slack: list[torch.Tensor] = []
        per_step_cost_quantiles: list[torch.Tensor] = []
        settling: list[torch.Tensor] = []
        history_states: list[torch.Tensor] = []
        diagnostics: dict[str, torch.Tensor] = {}

        for outer_step in range(self.config.outer_steps):
            prev_state = x
            transition_input = x + slow_state
            if self.hazard_encoder is not None:
                hazard_summary = self.hazard_encoder(self._hazard_summary(batch))
                hazard_summary = hazard_summary[:, None, :]
                hazard_mask = (
                    (batch["node_roles"] == ROLE_TO_ID["hub"]) | (batch["node_roles"] == ROLE_TO_ID["monitor"])
                ).unsqueeze(-1)
                transition_input = transition_input + hazard_mask * hazard_summary
            router_diag_last: dict[str, torch.Tensor] = {}
            for layer in self._transition_layers(outer_step):
                transition_input, router_diag_last = layer(
                    transition_input,
                    edge_features=batch["edge_features"],
                    adjacency=batch["adjacency"],
                    node_mask=batch["node_mask"],
                    node_roles=batch["node_roles"],
                    order_current=batch["order_current"],
                    order_destination=batch["order_destination"],
                    sequence_mask=sequence_mask,
                )
            commit_gate = torch.sigmoid(self.commit_gate(torch.cat([slow_state, transition_input], dim=-1)))
            slow_state = slow_state + commit_gate * (transition_input - slow_state)
            x = transition_input

            if self.training and self.config.detach_warmup and outer_step < self.config.outer_steps - 1:
                keep_penultimate = (
                    outer_step == self.config.outer_steps - 2
                    and self.config.penultimate_grad_prob > 0.0
                    and torch.rand((), device=device) < self.config.penultimate_grad_prob
                )
                if not keep_penultimate:
                    x = x.detach()
                    slow_state = slow_state.detach()

            readout_state = x + slow_state
            if self.history_reader is not None:
                readout_state, history_diag = self.history_reader(
                    readout_state,
                    history_states,
                    node_mask=batch["node_mask"],
                    node_roles=batch["node_roles"],
                )
                router_diag_last = {**router_diag_last, **history_diag}

            readout = self.final_norm(readout_state)
            readout_outputs = self._readout(readout, batch)
            per_step_logits.append(readout_outputs["node_logits"])
            per_step_values.append(readout_outputs["values"])
            if self.config.deadline_head:
                per_step_on_time_logits.append(readout_outputs["candidate_on_time_logits"])
                per_step_slack.append(readout_outputs["candidate_slack"])
                per_step_cost_quantiles.append(readout_outputs["candidate_cost_quantiles"])
            settling.append((readout - prev_state).norm(dim=-1).mean())
            diagnostics = router_diag_last
            history_states.append(readout.detach())

        stacked_logits = torch.stack(per_step_logits, dim=1)
        stacked_values = torch.stack(per_step_values, dim=1)
        final_outputs = readout_outputs
        return {
            "node_logits": final_outputs["node_logits"],
            "selection_scores": final_outputs["selection_scores"],
            "per_step_logits": stacked_logits,
            "values": final_outputs["values"],
            "per_step_values": stacked_values,
            "route_logits": final_outputs["route_logits"],
            "candidate_on_time_logits": final_outputs["candidate_on_time_logits"],
            "candidate_slack": final_outputs["candidate_slack"],
            "candidate_cost_quantiles": final_outputs["candidate_cost_quantiles"],
            "path_scores": final_outputs["path_scores"],
            "path_reranker_gate": final_outputs["path_reranker_gate"],
            "per_step_candidate_on_time_logits": (
                torch.stack(per_step_on_time_logits, dim=1) if per_step_on_time_logits else None
            ),
            "per_step_candidate_slack": torch.stack(per_step_slack, dim=1) if per_step_slack else None,
            "per_step_candidate_cost_quantiles": (
                torch.stack(per_step_cost_quantiles, dim=1) if per_step_cost_quantiles else None
            ),
            "settling_curve": torch.stack(settling, dim=0),
            "diagnostics": diagnostics,
        }

    def _readout(
        self,
        state: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size, num_nodes, _ = state.shape
        current_repr = batched_index_select(state, batch["current_node"])
        destination_repr = batched_index_select(state, batch["destination_node"])

        current_expand = current_repr[:, None, :].expand(-1, num_nodes, -1)
        destination_expand = destination_repr[:, None, :].expand(-1, num_nodes, -1)
        batch_index = torch.arange(batch_size, device=state.device)
        edge_context = batch["edge_features"][batch_index, batch["current_node"]]

        action_input = torch.cat([current_expand, destination_expand, state, edge_context], dim=-1)
        node_logits = self.action_head(action_input).squeeze(-1)
        invalid = ~(batch["candidate_mask"] & batch["node_mask"])
        node_logits = node_logits.masked_fill(invalid, -1e9)

        value_input = torch.cat([current_repr, destination_repr], dim=-1)
        values = self.value_head(value_input).squeeze(-1)

        route_input = torch.cat([current_expand, destination_expand, state], dim=-1)
        route_logits = self.route_head(route_input).squeeze(-1)
        route_logits = route_logits.masked_fill(~batch["node_mask"], -1e9)
        candidate_on_time_logits = None
        candidate_slack = None
        candidate_cost_quantiles = None
        path_scores = None
        path_reranker_gate = None
        path_verifier_penalty = None
        path_verifier_keep_mask = None
        selection_scores = node_logits
        if self.deadline_head is not None and self.slack_head is not None and self.quantile_head is not None:
            candidate_on_time_logits = self.deadline_head(action_input).squeeze(-1)
            candidate_slack = self.slack_head(action_input).squeeze(-1)
            candidate_cost_quantiles = self.quantile_head(action_input)

            candidate_invalid = ~(batch["candidate_mask"] & batch["node_mask"])
            candidate_on_time_logits = candidate_on_time_logits.masked_fill(candidate_invalid, -1e9)
            candidate_slack = candidate_slack.masked_fill(candidate_invalid, 0.0)
            candidate_cost_quantiles = candidate_cost_quantiles.masked_fill(candidate_invalid.unsqueeze(-1), 0.0)

            if self.config.risk_aware_scoring:
                deadline_scale = batch["packet_deadline"][:, None].clamp(min=1.0)
                on_time_bonus = self.config.on_time_score_weight * F.logsigmoid(candidate_on_time_logits)
                slack_bonus = self.config.slack_score_weight * torch.tanh(candidate_slack / deadline_scale)
                tail_penalty = self.config.tail_score_weight * (
                    candidate_cost_quantiles[..., -1] / deadline_scale
                )
                selection_scores = node_logits + on_time_bonus + slack_bonus - tail_penalty
                selection_scores = selection_scores.masked_fill(candidate_invalid, -1e9)
        if self.path_reranker_head is not None:
            safe_path_nodes = batch["candidate_path_nodes"].clamp(min=0)
            path_state = torch.gather(
                state[:, None, :, :].expand(-1, num_nodes, -1, -1),
                2,
                safe_path_nodes.unsqueeze(-1).expand(-1, -1, -1, state.size(-1)),
            )
            path_mask = batch["candidate_path_mask"].unsqueeze(-1)
            path_denom = path_mask.float().sum(dim=2).clamp(min=1.0)
            path_mean = (path_state * path_mask.float()).sum(dim=2) / path_denom
            path_valid = batch["candidate_path_mask"].any(dim=-1) & batch["candidate_mask"] & batch["node_mask"]
            path_input = torch.cat(
                [
                    current_expand,
                    destination_expand,
                    state,
                    path_mean,
                    batch["candidate_path_features"],
                ],
                dim=-1,
            )
            path_scores = self.path_reranker_head(path_input).squeeze(-1)
            if self.config.path_reranker_bound > 0.0:
                path_scores = self.config.path_reranker_bound * torch.tanh(path_scores)
            path_reranker_gate = self._path_reranker_gate(batch)
            path_scores = path_scores * path_reranker_gate
            path_scores = path_scores.masked_fill(~path_valid, 0.0)
            selection_scores = selection_scores + self.config.path_reranker_weight * path_scores
            selection_scores = selection_scores.masked_fill(~path_valid, -1e9)

        if self.config.path_verifier_filter:
            candidate_valid = batch["candidate_mask"] & batch["node_mask"]
            verified_on_time = batch["candidate_on_time"] > 0.5
            verified_slack = batch["candidate_slack"]
            deadline_scale = batch["packet_deadline"][:, None].clamp(min=1.0)
            verifier_penalty = self.config.path_verifier_penalty * F.relu(
                (self.config.path_verifier_slack_margin - verified_slack) / deadline_scale
            )
            verifier_penalty = verifier_penalty.masked_fill(~candidate_valid, 0.0)
            selection_scores = selection_scores - verifier_penalty

            feasible_candidates = verified_on_time & candidate_valid
            has_feasible = feasible_candidates.any(dim=-1, keepdim=True)
            path_verifier_keep_mask = candidate_valid & (~has_feasible | feasible_candidates)
            if self.config.path_verifier_hard_mask:
                selection_scores = selection_scores.masked_fill(~path_verifier_keep_mask, -1e9)
            else:
                path_verifier_keep_mask = candidate_valid

            path_verifier_penalty = verifier_penalty
            selection_scores = selection_scores.masked_fill(~candidate_valid, -1e9)

        return {
            "node_logits": node_logits,
            "selection_scores": selection_scores,
            "values": values,
            "route_logits": route_logits,
            "candidate_on_time_logits": candidate_on_time_logits,
            "candidate_slack": candidate_slack,
            "candidate_cost_quantiles": candidate_cost_quantiles,
            "path_scores": path_scores,
            "path_reranker_gate": path_reranker_gate,
            "path_verifier_penalty": path_verifier_penalty,
            "path_verifier_keep_mask": path_verifier_keep_mask,
        }


def _pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...],
) -> torch.Tensor:
    losses = []
    for index, quantile in enumerate(quantiles):
        error = target - prediction[:, index]
        losses.append(torch.maximum((quantile - 1.0) * error, quantile * error))
    return torch.stack(losses, dim=-1).mean()


def _candidate_aux_losses(
    *,
    on_time_logits: torch.Tensor,
    slack_prediction: torch.Tensor,
    quantile_prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    quantiles: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = batch["candidate_mask"] & batch["node_mask"]
    if not valid_mask.any():
        zero = on_time_logits.new_tensor(0.0)
        return zero, zero, zero
    on_time_loss = F.binary_cross_entropy_with_logits(
        on_time_logits[valid_mask],
        batch["candidate_on_time"][valid_mask],
    )
    slack_loss = F.smooth_l1_loss(
        slack_prediction[valid_mask],
        batch["candidate_slack"][valid_mask],
    )
    quantile_loss = _pinball_loss(
        quantile_prediction[valid_mask],
        batch["candidate_cost_to_go"][valid_mask],
        quantiles,
    )
    return on_time_loss, slack_loss, quantile_loss


def _selection_soft_target_loss(
    *,
    selection_scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    temperature: float,
    on_time_bonus: float,
) -> torch.Tensor:
    valid_mask = batch["candidate_mask"] & batch["node_mask"]
    temperature = max(float(temperature), 1e-3)
    target_logits = -batch["candidate_cost_to_go"] / temperature
    if on_time_bonus != 0.0:
        target_logits = target_logits + on_time_bonus * batch["candidate_on_time"]
    target_logits = target_logits.masked_fill(~valid_mask, float("-inf"))
    target_probs = torch.softmax(target_logits, dim=-1)
    log_probs = torch.log_softmax(selection_scores, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _path_soft_target_loss(
    *,
    path_scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    temperature: float,
    on_time_bonus: float,
) -> torch.Tensor:
    valid_mask = batch["candidate_mask"] & batch["node_mask"]
    if not valid_mask.any():
        return path_scores.new_tensor(0.0)
    temperature = max(float(temperature), 1e-3)
    target_logits = -batch["candidate_cost_to_go"] / temperature
    if on_time_bonus != 0.0:
        target_logits = target_logits + on_time_bonus * batch["candidate_on_time"]
    target_logits = target_logits.masked_fill(~valid_mask, float("-inf"))
    target_probs = torch.softmax(target_logits, dim=-1)
    masked_path_scores = path_scores.masked_fill(~valid_mask, -1e9)
    log_probs = torch.log_softmax(masked_path_scores, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _selection_pairwise_ranking_loss(
    *,
    selection_scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
    temperature: float,
    on_time_bonus: float,
    slack_bonus: float,
    margin: float,
) -> torch.Tensor:
    valid_mask = batch["candidate_mask"] & batch["node_mask"]
    temperature = max(float(temperature), 1e-3)
    target_quality = -batch["candidate_cost_to_go"] / temperature
    if on_time_bonus != 0.0:
        target_quality = target_quality + on_time_bonus * batch["candidate_on_time"]
    if slack_bonus != 0.0:
        target_quality = target_quality + slack_bonus * torch.tanh(batch["candidate_slack"] / temperature)

    pair_mask = valid_mask[:, :, None] & valid_mask[:, None, :]
    target_gap = target_quality[:, :, None] - target_quality[:, None, :]
    preferred_mask = pair_mask & (target_gap > 1e-6)
    if not preferred_mask.any():
        return selection_scores.new_tensor(0.0)

    score_gap = selection_scores[:, :, None] - selection_scores[:, None, :]
    pair_losses = F.softplus(float(margin) - score_gap)
    return pair_losses[preferred_mask].mean()


def _selection_feasible_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    valid_mask = batch["candidate_mask"] & batch["node_mask"]
    feasible_mask = valid_mask & (batch["candidate_on_time"] > 0.5)
    feasible_cost = batch["candidate_cost_to_go"].masked_fill(~feasible_mask, 1e9)
    feasible_target = feasible_cost.argmin(dim=-1)
    has_feasible = feasible_mask.any(dim=-1)
    return torch.where(has_feasible, feasible_target, batch["target_next_hop"])


def _selection_slack_critical_weights(
    *,
    batch: dict[str, torch.Tensor],
    scale: float,
) -> torch.Tensor:
    valid_mask = batch["candidate_mask"] & batch["node_mask"]
    best_slack = batch["candidate_slack"].masked_fill(~valid_mask, -1e6).max(dim=-1).values
    scale = max(float(scale), 1e-3)
    return 1.0 + torch.sigmoid(-best_slack / scale)


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    final_step_only: bool,
    value_weight: float = 0.2,
    route_weight: float = 0.1,
    deadline_bce_weight: float = 0.0,
    slack_weight: float = 0.0,
    quantile_weight: float = 0.0,
    selection_soft_target_weight: float = 0.0,
    selection_soft_target_temperature: float = 1.0,
    selection_soft_target_on_time_bonus: float = 0.0,
    path_soft_target_weight: float = 0.0,
    path_soft_target_temperature: float = 1.0,
    path_soft_target_on_time_bonus: float = 0.0,
    selection_pairwise_weight: float = 0.0,
    selection_pairwise_temperature: float = 1.0,
    selection_pairwise_on_time_bonus: float = 0.0,
    selection_pairwise_slack_bonus: float = 0.0,
    selection_pairwise_margin: float = 0.0,
    selection_feasible_target_weight: float = 0.0,
    selection_slack_critical_weight: float = 0.0,
    selection_slack_critical_scale: float = 1.0,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    verifier_aux_last_k_steps: int = 1,
) -> dict[str, torch.Tensor]:
    target = batch["target_next_hop"]
    feasible_target_loss = output["selection_scores"].new_tensor(0.0)
    if final_step_only:
        logits = output["selection_scores"]
        ce_loss = F.cross_entropy(logits, target, reduction="none")
        if selection_slack_critical_weight > 0.0:
            slack_weights = _selection_slack_critical_weights(batch=batch, scale=selection_slack_critical_scale)
            ce_loss = ce_loss * (1.0 + selection_slack_critical_weight * (slack_weights - 1.0))
        ce_loss = ce_loss.mean()
        value_loss = F.mse_loss(output["values"], batch["cost_to_go"])
    else:
        repeated_target = target[:, None].expand(-1, output["per_step_logits"].size(1)).reshape(-1)
        ce_loss = F.cross_entropy(
            output["per_step_logits"].reshape(-1, output["per_step_logits"].size(-1)),
            repeated_target,
            reduction="none",
        )
        if selection_slack_critical_weight > 0.0:
            slack_weights = _selection_slack_critical_weights(batch=batch, scale=selection_slack_critical_scale)
            repeated_weights = slack_weights[:, None].expand(-1, output["per_step_logits"].size(1)).reshape(-1)
            ce_loss = ce_loss * (1.0 + selection_slack_critical_weight * (repeated_weights - 1.0))
        ce_loss = ce_loss.mean()
        repeated_value = batch["cost_to_go"][:, None].expand_as(output["per_step_values"])
        value_loss = F.mse_loss(output["per_step_values"], repeated_value)

    route_logits = output["route_logits"][batch["node_mask"]]
    route_targets = batch["route_relevance"][batch["node_mask"]]
    route_loss = F.binary_cross_entropy_with_logits(route_logits, route_targets)
    selection_soft_target_loss = output["selection_scores"].new_tensor(0.0)
    path_soft_target_loss = output["selection_scores"].new_tensor(0.0)
    selection_pairwise_loss = output["selection_scores"].new_tensor(0.0)
    on_time_loss = output["selection_scores"].new_tensor(0.0)
    slack_loss = output["selection_scores"].new_tensor(0.0)
    quantile_loss = output["selection_scores"].new_tensor(0.0)
    if selection_soft_target_weight > 0.0:
        selection_soft_target_loss = _selection_soft_target_loss(
            selection_scores=output["selection_scores"],
            batch=batch,
            temperature=selection_soft_target_temperature,
            on_time_bonus=selection_soft_target_on_time_bonus,
        )
    if path_soft_target_weight > 0.0 and output.get("path_scores") is not None:
        path_soft_target_loss = _path_soft_target_loss(
            path_scores=output["path_scores"],
            batch=batch,
            temperature=path_soft_target_temperature,
            on_time_bonus=path_soft_target_on_time_bonus,
        )
    if selection_pairwise_weight > 0.0:
        selection_pairwise_loss = _selection_pairwise_ranking_loss(
            selection_scores=output["selection_scores"],
            batch=batch,
            temperature=selection_pairwise_temperature,
            on_time_bonus=selection_pairwise_on_time_bonus,
            slack_bonus=selection_pairwise_slack_bonus,
            margin=selection_pairwise_margin,
        )
    if selection_feasible_target_weight > 0.0:
        feasible_target = _selection_feasible_target(batch)
        feasible_target_loss = F.cross_entropy(output["selection_scores"], feasible_target)
    if deadline_bce_weight > 0.0 or slack_weight > 0.0 or quantile_weight > 0.0:
        per_step_on_time = output.get("per_step_candidate_on_time_logits")
        per_step_slack = output.get("per_step_candidate_slack")
        per_step_quantiles = output.get("per_step_candidate_cost_quantiles")
        if (
            verifier_aux_last_k_steps > 1
            and per_step_on_time is not None
            and per_step_slack is not None
            and per_step_quantiles is not None
        ):
            last_k = min(verifier_aux_last_k_steps, per_step_on_time.size(1))
            aux_losses = [
                _candidate_aux_losses(
                    on_time_logits=per_step_on_time[:, step_index],
                    slack_prediction=per_step_slack[:, step_index],
                    quantile_prediction=per_step_quantiles[:, step_index],
                    batch=batch,
                    quantiles=quantiles,
                )
                for step_index in range(per_step_on_time.size(1) - last_k, per_step_on_time.size(1))
            ]
            on_time_loss = torch.stack([entry[0] for entry in aux_losses]).mean()
            slack_loss = torch.stack([entry[1] for entry in aux_losses]).mean()
            quantile_loss = torch.stack([entry[2] for entry in aux_losses]).mean()
        else:
            on_time_loss, slack_loss, quantile_loss = _candidate_aux_losses(
                on_time_logits=output["candidate_on_time_logits"],
                slack_prediction=output["candidate_slack"],
                quantile_prediction=output["candidate_cost_quantiles"],
                batch=batch,
                quantiles=quantiles,
            )
    total_loss = (
        ce_loss
        + value_weight * value_loss
        + route_weight * route_loss
        + selection_soft_target_weight * selection_soft_target_loss
        + path_soft_target_weight * path_soft_target_loss
        + selection_pairwise_weight * selection_pairwise_loss
        + selection_feasible_target_weight * feasible_target_loss
        + deadline_bce_weight * on_time_loss
        + slack_weight * slack_loss
        + quantile_weight * quantile_loss
    )
    with torch.no_grad():
        accuracy = (output["selection_scores"].argmax(dim=-1) == target).float().mean()
        selection_accuracy = (output["selection_scores"].argmax(dim=-1) == target).float().mean()
    return {
        "loss": total_loss,
        "next_hop_loss": ce_loss.detach(),
        "value_loss": value_loss.detach(),
        "route_loss": route_loss.detach(),
        "selection_soft_target_loss": selection_soft_target_loss.detach(),
        "path_soft_target_loss": path_soft_target_loss.detach(),
        "selection_pairwise_loss": selection_pairwise_loss.detach(),
        "selection_feasible_target_loss": feasible_target_loss.detach(),
        "on_time_loss": on_time_loss.detach(),
        "slack_loss": slack_loss.detach(),
        "quantile_loss": quantile_loss.detach(),
        "next_hop_accuracy": accuracy.detach(),
        "selection_accuracy": selection_accuracy.detach(),
    }
