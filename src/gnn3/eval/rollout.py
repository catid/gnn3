from __future__ import annotations

from dataclasses import dataclass

import torch

from gnn3.data.hidden_corridor import (
    DecisionRecord,
    EpisodeRecord,
    HiddenCorridorConfig,
    _apply_transition,
    _edge_cost,
    collate_decisions,
    make_decision_record,
    shortest_path,
)
from gnn3.models.packet_mamba import PacketMambaModel


@dataclass(frozen=True)
class RolloutMetrics:
    solved_rate: float
    next_hop_accuracy: float
    average_regret: float
    average_deadline_violations: float
    priority_delivered_regret: float


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _predict_next_hop(
    model: PacketMambaModel,
    record: DecisionRecord,
    device: torch.device,
) -> int:
    batch = _move_batch(collate_decisions([record]), device)
    output = model(batch)
    return int(output["node_logits"].argmax(dim=-1).item())


def _rollout_episode(
    episode: EpisodeRecord,
    model: PacketMambaModel | None,
    *,
    device: torch.device,
    config: HiddenCorridorConfig,
) -> tuple[bool, int, int, float, int]:
    working_graph = episode.graph.copy()
    ordered_packets = sorted(
        enumerate(episode.packets),
        key=lambda item: (-item[1].priority, item[1].deadline),
    )

    total_steps = 0
    correct_steps = 0
    total_cost = 0.0
    deadline_violations = 0
    delivered_priority = 0.0
    max_steps = working_graph.num_nodes * config.max_steps_multiplier
    solved = True

    for packet_index, packet in ordered_packets:
        current = packet.source
        remaining_deadline = packet.deadline
        steps = 0
        while current != packet.destination and steps < max_steps:
            path, path_cost = shortest_path(
                working_graph,
                packet,
                start=current,
                remaining_deadline=remaining_deadline,
                config=config,
            )
            if len(path) < 2:
                solved = False
                break
            oracle_next_hop = path[1]
            if model is None:
                chosen_next_hop = oracle_next_hop
            else:
                record = make_decision_record(
                    working_graph,
                    packet,
                    current_node=current,
                    target_next_hop=oracle_next_hop,
                    cost_to_go=path_cost,
                    route_nodes=path,
                    packet_index=packet_index,
                    packet_count=len(episode.packets),
                    curriculum_level="eval",
                )
                chosen_next_hop = _predict_next_hop(model, record, device)
            total_steps += 1
            correct_steps += int(chosen_next_hop == oracle_next_hop)
            if not working_graph.adj[current, chosen_next_hop]:
                solved = False
                break
            transition_cost = _edge_cost(
                working_graph,
                packet,
                current,
                chosen_next_hop,
                remaining_deadline=remaining_deadline,
                config=config,
            )
            total_cost += transition_cost
            remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
            _apply_transition(working_graph, current, chosen_next_hop, packet)
            current = chosen_next_hop
            steps += 1

        if current != packet.destination:
            solved = False
        if remaining_deadline <= 0.0:
            deadline_violations += 1
        else:
            delivered_priority += packet.priority

    return solved, total_steps, correct_steps, total_cost, deadline_violations, int(delivered_priority)


@torch.no_grad()
def evaluate_rollouts(
    model: PacketMambaModel,
    episodes: list[EpisodeRecord],
    *,
    device: torch.device,
    config: HiddenCorridorConfig,
) -> RolloutMetrics:
    oracle_totals: list[float] = []
    model_totals: list[float] = []
    solved = 0
    total_steps = 0
    correct_steps = 0
    deadline_violations = 0
    oracle_priority = 0
    model_priority = 0

    was_training = model.training
    model.eval()
    for episode in episodes:
        _, _, _, oracle_cost, _oracle_violations, oracle_delivered = _rollout_episode(
            episode,
            None,
            device=device,
            config=config,
        )
        episode_solved, steps, correct, model_cost, model_violations, model_delivered = _rollout_episode(
            episode,
            model,
            device=device,
            config=config,
        )
        solved += int(episode_solved)
        total_steps += steps
        correct_steps += correct
        oracle_totals.append(oracle_cost)
        model_totals.append(model_cost)
        deadline_violations += model_violations
        oracle_priority += oracle_delivered
        model_priority += model_delivered

    if was_training:
        model.train()

    count = max(len(episodes), 1)
    return RolloutMetrics(
        solved_rate=solved / count,
        next_hop_accuracy=correct_steps / max(total_steps, 1),
        average_regret=sum(model_totals) / count - sum(oracle_totals) / count,
        average_deadline_violations=deadline_violations / count,
        priority_delivered_regret=(oracle_priority - model_priority) / count,
    )
