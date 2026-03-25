from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from gnn3.data.hidden_corridor import (
    DecisionRecord,
    EpisodeRecord,
    GraphState,
    HiddenCorridorConfig,
    _apply_transition,
    _edge_cost,
    collate_decisions,
    shortest_path,
)
from gnn3.eval.rollout import _move_batch, _predict_next_hop, _selection_scores_from_output
from gnn3.models.packet_mamba import PacketMambaModel


@dataclass(frozen=True)
class EpisodePolicyRow:
    suite: str
    episode_index: int
    packet_count: int
    community_count: int
    max_tree_depth: int
    hub_asymmetry: float
    mean_queue: float
    p90_queue: float
    max_queue: float
    mean_deadline: float
    min_deadline: float
    solved: bool
    total_steps: int
    correct_steps: int
    next_hop_accuracy: float
    oracle_cost: float
    model_cost: float
    regret: float
    deadline_violations: int
    deadline_miss: bool
    oracle_priority_delivered: float
    model_priority_delivered: float
    priority_delivered_regret: float


@dataclass(frozen=True)
class DecisionPredictionRow:
    suite: str
    episode_index: int
    decision_index: int
    packet_count: int
    packet_priority: float
    packet_deadline: float
    candidate_degree: int
    best_candidate_slack: float
    best_candidate_slack_ratio: float
    feasible_candidate_fraction: float
    any_feasible_candidate: bool
    max_candidate_path_length: int
    target_next_hop: int
    predicted_next_hop: int
    best_candidate_cost: float
    second_best_candidate_cost: float
    oracle_action_gap: float
    oracle_action_gap_ratio: float
    predicted_cost_to_go: float
    predicted_slack: float
    predicted_on_time: bool
    predicted_continuation_gap: float
    strictly_suboptimal: bool
    target_match: bool


def _max_tree_depth(graph: GraphState) -> int:
    max_depth = 0
    for community in np.unique(graph.node_communities):
        if int(community) < 0:
            continue
        roots = np.where((graph.node_roles == 2) & (graph.node_communities == community))[0]
        if roots.size == 0:
            continue
        root = int(roots[0])
        distances = np.full((graph.num_nodes,), fill_value=-1, dtype=np.int64)
        queue = [root]
        distances[root] = 0
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for neighbor in np.flatnonzero(graph.adj[node]):
                if distances[int(neighbor)] != -1:
                    continue
                distances[int(neighbor)] = distances[node] + 1
                queue.append(int(neighbor))
        community_leaves = np.where((graph.node_roles == 0) & (graph.node_communities == community))[0]
        if community_leaves.size:
            max_depth = max(max_depth, int(distances[community_leaves].max()))
    return max_depth


def _hub_asymmetry(graph: GraphState) -> float:
    roots = np.where(graph.node_roles == 2)[0]
    hubs = np.where(graph.node_roles == 3)[0]
    if roots.size == 0 or hubs.size < 2:
        return 0.0
    hub_means: list[float] = []
    for hub in hubs:
        latencies = [float(graph.edge_effective_latency[root, hub]) for root in roots if graph.adj[root, hub]]
        if latencies:
            hub_means.append(float(np.mean(latencies)))
    if len(hub_means) < 2:
        return 0.0
    return float(max(hub_means) - min(hub_means))


def _episode_static_features(episode: EpisodeRecord) -> dict[str, Any]:
    graph = episode.graph
    queues = graph.node_queue.astype(np.float32)
    communities = graph.node_communities[graph.node_communities >= 0]
    deadlines = [packet.deadline for packet in episode.packets]
    return {
        "packet_count": len(episode.packets),
        "community_count": len(np.unique(communities)) if communities.size else 0,
        "max_tree_depth": _max_tree_depth(graph),
        "hub_asymmetry": _hub_asymmetry(graph),
        "mean_queue": float(np.mean(queues)) if queues.size else 0.0,
        "p90_queue": float(np.quantile(queues, 0.9)) if queues.size else 0.0,
        "max_queue": float(np.max(queues)) if queues.size else 0.0,
        "mean_deadline": float(np.mean(deadlines)) if deadlines else 0.0,
        "min_deadline": float(np.min(deadlines)) if deadlines else 0.0,
    }


def _rollout_episode_detail(
    episode: EpisodeRecord,
    model: PacketMambaModel | None,
    *,
    device: torch.device,
    config: HiddenCorridorConfig,
    selection_strategy: str,
) -> dict[str, Any]:
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
            if len(path) < 2 or not math.isfinite(path_cost):
                solved = False
                break
            oracle_next_hop = path[1]
            if model is None:
                chosen_next_hop = oracle_next_hop
            else:
                from gnn3.data.hidden_corridor import make_decision_record

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

    return {
        "solved": solved,
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "next_hop_accuracy": correct_steps / max(total_steps, 1),
        "total_cost": total_cost,
        "deadline_violations": deadline_violations,
        "delivered_priority": delivered_priority,
    }


@torch.no_grad()
def collect_episode_policy_rows(
    model: PacketMambaModel,
    episodes: list[EpisodeRecord],
    *,
    device: torch.device,
    config: HiddenCorridorConfig,
    suite: str,
    selection_strategy: str = "final",
) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        oracle = _rollout_episode_detail(
            episode,
            None,
            device=device,
            config=config,
            selection_strategy="final",
        )
        model_detail = _rollout_episode_detail(
            episode,
            model,
            device=device,
            config=config,
            selection_strategy=selection_strategy,
        )
        row = EpisodePolicyRow(
            suite=suite,
            episode_index=episode_index,
            solved=bool(model_detail["solved"]),
            total_steps=int(model_detail["total_steps"]),
            correct_steps=int(model_detail["correct_steps"]),
            next_hop_accuracy=float(model_detail["next_hop_accuracy"]),
            oracle_cost=float(oracle["total_cost"]),
            model_cost=float(model_detail["total_cost"]),
            regret=float(model_detail["total_cost"] - oracle["total_cost"]),
            deadline_violations=int(model_detail["deadline_violations"]),
            deadline_miss=bool(model_detail["deadline_violations"] > 0),
            oracle_priority_delivered=float(oracle["delivered_priority"]),
            model_priority_delivered=float(model_detail["delivered_priority"]),
            priority_delivered_regret=float(oracle["delivered_priority"] - model_detail["delivered_priority"]),
            **_episode_static_features(episode),
        )
        rows.append(asdict(row))
    if was_training:
        model.train()
    return rows


@torch.no_grad()
def collect_decision_prediction_rows(
    model: PacketMambaModel,
    records: list[DecisionRecord],
    *,
    device: torch.device,
    suite: str,
    selection_strategy: str = "final",
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch = _move_batch(collate_decisions(batch_records), device)
        output = model(batch)
        scores = _selection_scores_from_output(output, batch, selection_strategy=selection_strategy)
        predicted = scores.argmax(dim=-1).detach().cpu().tolist()
        for offset, record in enumerate(batch_records):
            valid_mask = record.candidate_mask.astype(bool)
            feasible = record.candidate_on_time[valid_mask] > 0.5
            feasible_fraction = float(feasible.mean()) if feasible.size else 0.0
            if feasible.any():
                reference_costs = record.candidate_cost_to_go[valid_mask][feasible]
            else:
                reference_costs = record.candidate_cost_to_go[valid_mask]
            sorted_costs = np.sort(reference_costs.astype(np.float32)) if reference_costs.size else np.zeros((0,), dtype=np.float32)
            if sorted_costs.size:
                best_cost = float(sorted_costs[0])
                second_best_cost = float(sorted_costs[1]) if sorted_costs.size > 1 else float(sorted_costs[0])
            else:
                best_cost = 0.0
                second_best_cost = 0.0
            predicted_next_hop = int(predicted[offset])
            predicted_cost = float(record.candidate_cost_to_go[predicted_next_hop])
            predicted_slack = float(record.candidate_slack[predicted_next_hop])
            predicted_on_time = bool(record.candidate_on_time[predicted_next_hop] > 0.5)
            continuation_gap = predicted_cost - best_cost
            if valid_mask.any():
                best_slack = float(record.candidate_slack[valid_mask].max())
                max_path_length = int(record.candidate_path_mask[valid_mask].sum(axis=-1).max())
            else:
                best_slack = 0.0
                max_path_length = 0
            row = DecisionPredictionRow(
                suite=suite,
                episode_index=int(record.episode_index),
                decision_index=start + offset,
                packet_count=int(record.packet_count),
                packet_priority=float(record.packet_priority),
                packet_deadline=float(record.packet_deadline),
                candidate_degree=int(valid_mask.sum()),
                best_candidate_slack=best_slack,
                best_candidate_slack_ratio=best_slack / max(float(record.packet_deadline), 1e-6),
                feasible_candidate_fraction=feasible_fraction,
                any_feasible_candidate=bool(feasible.any()) if feasible.size else False,
                max_candidate_path_length=max_path_length,
                target_next_hop=int(record.target_next_hop),
                predicted_next_hop=predicted_next_hop,
                best_candidate_cost=best_cost,
                second_best_candidate_cost=second_best_cost,
                oracle_action_gap=max(second_best_cost - best_cost, 0.0),
                oracle_action_gap_ratio=max(second_best_cost - best_cost, 0.0) / max(abs(best_cost), 1e-6),
                predicted_cost_to_go=predicted_cost,
                predicted_slack=predicted_slack,
                predicted_on_time=predicted_on_time,
                predicted_continuation_gap=continuation_gap,
                strictly_suboptimal=bool(continuation_gap > 1e-6),
                target_match=bool(predicted_next_hop == int(record.target_next_hop)),
            )
            rows.append(asdict(row))
    if was_training:
        model.train()
    return rows


@torch.no_grad()
def extract_probe_features(
    model: PacketMambaModel,
    records: list[DecisionRecord],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    feature_chunks: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch = _move_batch(collate_decisions(batch_records), device)
        output = model(batch)
        feature_chunks.append(output["probe_features"].detach().cpu())
    if was_training:
        model.train()
    if not feature_chunks:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(feature_chunks, dim=0)
