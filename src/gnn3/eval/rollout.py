from __future__ import annotations

from dataclasses import dataclass

import torch

from gnn3.data.hidden_corridor import (
    DecisionListDataset,
    DecisionRecord,
    EpisodeRecord,
    HiddenCorridorConfig,
    _apply_transition,
    _edge_cost,
    collate_decisions,
    make_decision_record,
    shortest_path,
)
from gnn3.eval.step_policy import select_step_scores
from gnn3.models.packet_mamba import PacketMambaModel


@dataclass(frozen=True)
class RolloutMetrics:
    solved_rate: float
    next_hop_accuracy: float
    average_regret: float
    p95_regret: float
    worst_regret: float
    average_deadline_violations: float
    deadline_miss_rate: float
    p95_deadline_violations: float
    priority_delivered_regret: float
    average_oracle_cost: float
    average_model_cost: float


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _predict_next_hop(
    model: PacketMambaModel,
    record: DecisionRecord,
    device: torch.device,
    *,
    selection_strategy: str = "final",
) -> int:
    batch = _move_batch(collate_decisions([record]), device)
    output = model(batch)
    scores = select_step_scores(output, batch["candidate_mask"], strategy=selection_strategy)
    return int(scores.argmax(dim=-1).item())


def _rollout_episode(
    episode: EpisodeRecord,
    model: PacketMambaModel | None,
    *,
    device: torch.device,
    config: HiddenCorridorConfig,
    selection_strategy: str = "final",
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
                    config=config,
                    current_node=current,
                    target_next_hop=oracle_next_hop,
                    cost_to_go=path_cost,
                    route_nodes=path,
                    packet_index=packet_index,
                    packet_count=len(episode.packets),
                    curriculum_level="eval",
                    include_candidate_targets=False,
                )
                chosen_next_hop = _predict_next_hop(
                    model,
                    record,
                    device,
                    selection_strategy=selection_strategy,
                )
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
def collect_policy_decisions(
    model: PacketMambaModel,
    episodes: list[EpisodeRecord],
    *,
    device: torch.device,
    config: HiddenCorridorConfig,
    selection_strategy: str = "final",
) -> DecisionListDataset:
    decisions: list[DecisionRecord] = []
    was_training = model.training
    model.eval()
    for episode in episodes:
        working_graph = episode.graph.copy()
        ordered_packets = sorted(
            enumerate(episode.packets),
            key=lambda item: (-item[1].priority, item[1].deadline),
        )
        max_steps = working_graph.num_nodes * config.max_steps_multiplier
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
                    break
                oracle_next_hop = path[1]
                record = make_decision_record(
                    working_graph,
                    packet,
                    config=config,
                    current_node=current,
                    target_next_hop=oracle_next_hop,
                    cost_to_go=path_cost,
                    route_nodes=path,
                    packet_index=packet_index,
                    packet_count=len(episode.packets),
                    curriculum_level=episode.curriculum_level,
                )
                decisions.append(record)
                chosen_next_hop = _predict_next_hop(
                    model,
                    record,
                    device,
                    selection_strategy=selection_strategy,
                )
                if not working_graph.adj[current, chosen_next_hop]:
                    break
                transition_cost = _edge_cost(
                    working_graph,
                    packet,
                    current,
                    chosen_next_hop,
                    remaining_deadline=remaining_deadline,
                    config=config,
                )
                remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
                _apply_transition(working_graph, current, chosen_next_hop, packet)
                current = chosen_next_hop
                steps += 1
    if was_training:
        model.train()
    return DecisionListDataset(decisions)


@torch.no_grad()
def evaluate_rollouts(
    model: PacketMambaModel,
    episodes: list[EpisodeRecord],
    *,
    device: torch.device,
    config: HiddenCorridorConfig,
    selection_strategy: str = "final",
) -> RolloutMetrics:
    oracle_costs: list[float] = []
    model_costs: list[float] = []
    deadline_counts: list[float] = []
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
            selection_strategy=selection_strategy,
        )
        solved += int(episode_solved)
        total_steps += steps
        correct_steps += correct
        oracle_costs.append(oracle_cost)
        model_costs.append(model_cost)
        deadline_counts.append(float(model_violations))
        deadline_violations += model_violations
        oracle_priority += oracle_delivered
        model_priority += model_delivered

    if was_training:
        model.train()

    count = max(len(episodes), 1)
    regrets = [model - oracle for model, oracle in zip(model_costs, oracle_costs, strict=True)]
    regret_tensor = torch.tensor(regrets, dtype=torch.float32) if regrets else torch.tensor([0.0], dtype=torch.float32)
    deadline_tensor = (
        torch.tensor(deadline_counts, dtype=torch.float32) if deadline_counts else torch.tensor([0.0], dtype=torch.float32)
    )
    return RolloutMetrics(
        solved_rate=solved / count,
        next_hop_accuracy=correct_steps / max(total_steps, 1),
        average_regret=float(regret_tensor.mean().item()),
        p95_regret=float(torch.quantile(regret_tensor, 0.95).item()),
        worst_regret=float(regret_tensor.max().item()),
        average_deadline_violations=deadline_violations / count,
        deadline_miss_rate=float((deadline_tensor > 0).float().mean().item()),
        p95_deadline_violations=float(torch.quantile(deadline_tensor, 0.95).item()),
        priority_delivered_regret=(oracle_priority - model_priority) / count,
        average_oracle_cost=sum(oracle_costs) / count,
        average_model_cost=sum(model_costs) / count,
    )
