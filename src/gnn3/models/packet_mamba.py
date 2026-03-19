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
    max_history_steps: int = 8


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
    def __init__(self, d_model: int, dropout: float, max_history_steps: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(2 * d_model, d_model)
        self.round_embedding = nn.Embedding(max_history_steps, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        current_state: torch.Tensor,
        history_states: list[torch.Tensor],
        *,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not history_states:
            zero = current_state.new_tensor(0.0)
            return current_state, {
                "history_read_entropy": zero,
                "history_latest_round_share": zero,
            }

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
        }
        return current_state + self.dropout(update), diagnostics


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
            OuterHistoryReader(config.d_model, config.dropout, config.max_history_steps)
            if config.history_read
            else None
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.action_head = nn.Sequential(
            nn.Linear(3 * config.d_model + config.edge_feature_dim, config.d_model),
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

    def _transition_layers(self, step: int) -> nn.ModuleList:
        if self.config.shared_transition:
            return self.layers
        assert self.step_layers is not None
        return self.step_layers[step]

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
        settling: list[torch.Tensor] = []
        history_states: list[torch.Tensor] = []
        diagnostics: dict[str, torch.Tensor] = {}

        for outer_step in range(self.config.outer_steps):
            prev_state = x
            transition_input = x + slow_state
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
                )
                router_diag_last = {**router_diag_last, **history_diag}

            readout = self.final_norm(readout_state)
            node_logits, values, route_logits = self._readout(readout, batch)
            per_step_logits.append(node_logits)
            per_step_values.append(values)
            settling.append((readout - prev_state).norm(dim=-1).mean())
            diagnostics = router_diag_last
            history_states.append(readout.detach())

        stacked_logits = torch.stack(per_step_logits, dim=1)
        stacked_values = torch.stack(per_step_values, dim=1)
        route_logits = route_logits
        return {
            "node_logits": stacked_logits[:, -1],
            "per_step_logits": stacked_logits,
            "values": stacked_values[:, -1],
            "per_step_values": stacked_values,
            "route_logits": route_logits,
            "settling_curve": torch.stack(settling, dim=0),
            "diagnostics": diagnostics,
        }

    def _readout(
        self,
        state: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return node_logits, values, route_logits


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    final_step_only: bool,
    value_weight: float = 0.2,
    route_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    target = batch["target_next_hop"]
    if final_step_only:
        logits = output["node_logits"]
        ce_loss = F.cross_entropy(logits, target)
        value_loss = F.mse_loss(output["values"], batch["cost_to_go"])
    else:
        repeated_target = target[:, None].expand(-1, output["per_step_logits"].size(1)).reshape(-1)
        ce_loss = F.cross_entropy(output["per_step_logits"].reshape(-1, output["per_step_logits"].size(-1)), repeated_target)
        repeated_value = batch["cost_to_go"][:, None].expand_as(output["per_step_values"])
        value_loss = F.mse_loss(output["per_step_values"], repeated_value)

    route_logits = output["route_logits"][batch["node_mask"]]
    route_targets = batch["route_relevance"][batch["node_mask"]]
    route_loss = F.binary_cross_entropy_with_logits(route_logits, route_targets)
    total_loss = ce_loss + value_weight * value_loss + route_weight * route_loss
    with torch.no_grad():
        accuracy = (output["node_logits"].argmax(dim=-1) == target).float().mean()
    return {
        "loss": total_loss,
        "next_hop_loss": ce_loss.detach(),
        "value_loss": value_loss.detach(),
        "route_loss": route_loss.detach(),
        "next_hop_accuracy": accuracy.detach(),
    }
