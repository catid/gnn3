from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gnn3.data.hidden_corridor import (
    EpisodeRecord,
    GraphState,
    HiddenCorridorConfig,
    PacketSpec,
    _apply_transition,
    _edge_cost,
    shortest_path,
)


@dataclass(frozen=True)
class OraclePacketAudit:
    episode_index: int
    packet_index: int
    packet_order_rank: int
    packet_count: int
    curriculum_level: str
    max_tree_depth: int
    packet_priority: float
    packet_deadline: float
    reference_cost: float
    oracle_realized_cost: float
    oracle_remaining_deadline: float
    oracle_initial_slack: float
    has_on_time_feasible_route: bool
    oracle_deadline_missed: bool


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


def _oracle_packet_rollout(
    graph: GraphState,
    packet: PacketSpec,
    *,
    config: HiddenCorridorConfig,
) -> tuple[float, float]:
    current = packet.source
    remaining_deadline = packet.deadline
    realized_cost = 0.0
    steps = 0
    max_steps = graph.num_nodes * config.max_steps_multiplier
    while current != packet.destination and steps < max_steps:
        path, _path_cost = shortest_path(
            graph,
            packet,
            start=current,
            remaining_deadline=remaining_deadline,
            config=config,
        )
        if len(path) < 2:
            break
        next_hop = path[1]
        transition_cost = _edge_cost(
            graph,
            packet,
            current,
            next_hop,
            remaining_deadline=remaining_deadline,
            config=config,
        )
        realized_cost += transition_cost
        remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
        _apply_transition(graph, current, next_hop, packet)
        current = next_hop
        steps += 1
    return realized_cost, remaining_deadline


def audit_oracle_deadlines(
    episodes: list[EpisodeRecord],
    *,
    config: HiddenCorridorConfig,
) -> list[OraclePacketAudit]:
    audits: list[OraclePacketAudit] = []
    for episode_index, episode in enumerate(episodes):
        working_graph = episode.graph.copy()
        ordered_packets = sorted(
            enumerate(episode.packets),
            key=lambda item: (-item[1].priority, item[1].deadline),
        )
        max_tree_depth = _max_tree_depth(episode.graph)
        for packet_order_rank, (packet_index, packet) in enumerate(ordered_packets):
            _path, reference_cost = shortest_path(
                working_graph,
                packet,
                start=packet.source,
                remaining_deadline=packet.deadline,
                config=config,
            )
            realized_cost, remaining_deadline = _oracle_packet_rollout(
                working_graph,
                packet,
                config=config,
            )
            audits.append(
                OraclePacketAudit(
                    episode_index=episode_index,
                    packet_index=packet_index,
                    packet_order_rank=packet_order_rank,
                    packet_count=len(episode.packets),
                    curriculum_level=episode.curriculum_level,
                    max_tree_depth=max_tree_depth,
                    packet_priority=packet.priority,
                    packet_deadline=packet.deadline,
                    reference_cost=reference_cost,
                    oracle_realized_cost=realized_cost,
                    oracle_remaining_deadline=remaining_deadline,
                    oracle_initial_slack=packet.deadline - reference_cost,
                    has_on_time_feasible_route=reference_cost <= packet.deadline,
                    oracle_deadline_missed=remaining_deadline <= 0.0,
                )
            )
    return audits


def audits_to_rows(audits: list[OraclePacketAudit]) -> list[dict[str, object]]:
    return [asdict(audit) for audit in audits]
