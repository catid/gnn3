from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

ROLE_NAMES = ("leaf", "internal", "root", "hub", "monitor")
ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLE_NAMES)}


@dataclass(frozen=True)
class PacketSpec:
    source: int
    destination: int
    priority: float
    deadline: float
    size: float = 1.0


@dataclass
class GraphState:
    adj: np.ndarray
    node_roles: np.ndarray
    node_communities: np.ndarray
    node_queue: np.ndarray
    monitor_signal: np.ndarray
    edge_nominal_latency: np.ndarray
    edge_effective_latency: np.ndarray
    edge_capacity: np.ndarray
    edge_residual_capacity: np.ndarray
    edge_is_fast: np.ndarray
    edge_is_hidden_corridor: np.ndarray
    leaf_nodes_by_community: list[list[int]]

    def copy(self) -> GraphState:
        return GraphState(
            adj=self.adj.copy(),
            node_roles=self.node_roles.copy(),
            node_communities=self.node_communities.copy(),
            node_queue=self.node_queue.copy(),
            monitor_signal=self.monitor_signal.copy(),
            edge_nominal_latency=self.edge_nominal_latency.copy(),
            edge_effective_latency=self.edge_effective_latency.copy(),
            edge_capacity=self.edge_capacity.copy(),
            edge_residual_capacity=self.edge_residual_capacity.copy(),
            edge_is_fast=self.edge_is_fast.copy(),
            edge_is_hidden_corridor=self.edge_is_hidden_corridor.copy(),
            leaf_nodes_by_community=[list(nodes) for nodes in self.leaf_nodes_by_community],
        )

    @property
    def num_nodes(self) -> int:
        return int(self.adj.shape[0])


@dataclass(frozen=True)
class DecisionRecord:
    node_features: np.ndarray
    edge_features: np.ndarray
    node_roles: np.ndarray
    node_communities: np.ndarray
    adjacency: np.ndarray
    current_node: int
    source_node: int
    destination_node: int
    order_current: np.ndarray
    order_destination: np.ndarray
    candidate_mask: np.ndarray
    target_next_hop: int
    cost_to_go: float
    candidate_cost_to_go: np.ndarray
    candidate_slack: np.ndarray
    candidate_on_time: np.ndarray
    candidate_path_nodes: np.ndarray
    candidate_path_mask: np.ndarray
    candidate_path_features: np.ndarray
    route_relevance: np.ndarray
    packet_priority: float
    packet_deadline: float
    packet_index: int
    packet_count: int
    curriculum_level: str


@dataclass(frozen=True)
class EpisodeRecord:
    graph: GraphState
    packets: tuple[PacketSpec, ...]
    oracle_total_cost: float
    decision_count: int
    curriculum_level: str


@dataclass(frozen=True)
class HiddenCorridorConfig:
    seed: int = 0
    num_communities: int = 4
    tree_depth_min: int = 2
    tree_depth_max: int = 3
    branching_factor: int = 2
    packets_min: int = 1
    packets_max: int = 4
    dynamic_capacities: bool = True
    community_base_queue: tuple[float, float] = (0.0, 4.0)
    train_curriculum_levels: tuple[str, ...] = (
        "single_static",
        "single_dynamic",
        "multi_dynamic",
    )
    monitor_noise_std: float = 0.05
    queue_penalty: float = 0.35
    capacity_penalty: float = 2.5
    urgency_penalty: float = 1.2
    deadline_mode: str = "uniform"
    deadline_min: float = 7.0
    deadline_max: float = 18.0
    deadline_slack_ratio_range: tuple[float, float] = (0.1, 0.3)
    deadline_slack_abs_range: tuple[float, float] = (0.5, 2.0)
    deadline_reference_budget: float = 1_000_000.0
    max_steps_multiplier: int = 4


def _path_summary_features(graph: GraphState, path: list[int]) -> np.ndarray:
    if len(path) < 2:
        return np.zeros((5,), dtype=np.float32)
    edges = list(itertools.pairwise(path))
    residual_ratios = []
    hidden_flags = []
    fast_flags = []
    for u, v in edges:
        capacity = max(float(graph.edge_capacity[u, v]), 1e-3)
        residual_ratios.append(float(graph.edge_residual_capacity[u, v]) / capacity)
        hidden_flags.append(float(graph.edge_is_hidden_corridor[u, v]))
        fast_flags.append(float(graph.edge_is_fast[u, v]))
    queues = [float(graph.node_queue[node]) for node in path]
    roles = [ROLE_NAMES[int(graph.node_roles[node])] for node in path]
    hub_monitor_fraction = sum(role in {"hub", "monitor"} for role in roles) / len(roles)
    return np.asarray(
        [
            (len(path) - 1) / max(graph.num_nodes, 1),
            float(np.mean(queues)) / 10.0,
            float(np.max(queues)) / 10.0,
            float(np.mean(residual_ratios)),
            0.5 * float(np.mean(hidden_flags)) + 0.25 * float(np.mean(fast_flags)) + 0.25 * hub_monitor_fraction,
        ],
        dtype=np.float32,
    )


def _connect_undirected(
    adj: np.ndarray,
    nominal_latency: np.ndarray,
    effective_latency: np.ndarray,
    capacity: np.ndarray,
    residual_capacity: np.ndarray,
    is_fast: np.ndarray,
    is_hidden_corridor: np.ndarray,
    u: int,
    v: int,
    *,
    nominal: float,
    effective: float,
    cap: float,
    fast: bool = False,
    hidden_corridor: bool = False,
) -> None:
    for src, dst in ((u, v), (v, u)):
        adj[src, dst] = True
        nominal_latency[src, dst] = nominal
        effective_latency[src, dst] = effective
        capacity[src, dst] = cap
        residual_capacity[src, dst] = cap
        is_fast[src, dst] = fast
        is_hidden_corridor[src, dst] = hidden_corridor


def _sample_tree_depth(rng: np.random.Generator, config: HiddenCorridorConfig) -> int:
    return int(rng.integers(config.tree_depth_min, config.tree_depth_max + 1))


def build_hidden_corridor_graph(
    rng: np.random.Generator,
    config: HiddenCorridorConfig,
    *,
    dynamic_capacities: bool | None = None,
) -> GraphState:
    dynamic_capacities = config.dynamic_capacities if dynamic_capacities is None else dynamic_capacities

    node_roles: list[int] = []
    node_communities: list[int] = []
    leaf_nodes_by_community: list[list[int]] = [[] for _ in range(config.num_communities)]

    def add_node(role: str, community: int) -> int:
        node_roles.append(ROLE_TO_ID[role])
        node_communities.append(community)
        return len(node_roles) - 1

    hub_a = add_node("hub", -1)
    hub_b = add_node("hub", -1)
    community_roots: list[int] = []
    community_monitors: list[int] = []

    tree_edges: list[tuple[int, int]] = []
    fast_edges: list[tuple[int, int, float, float]] = []
    slow_edges: list[tuple[int, int, float, float]] = []

    for community in range(config.num_communities):
        root = add_node("root", community)
        monitor = add_node("monitor", community)
        community_roots.append(root)
        community_monitors.append(monitor)
        tree_edges.append((root, monitor))

        frontier = [root]
        depth = _sample_tree_depth(rng, config)
        for level in range(depth):
            next_frontier: list[int] = []
            child_role = "leaf" if level == depth - 1 else "internal"
            for parent in frontier:
                for _ in range(config.branching_factor):
                    child = add_node(child_role, community)
                    tree_edges.append((parent, child))
                    next_frontier.append(child)
                    if child_role == "leaf":
                        leaf_nodes_by_community[community].append(child)
            frontier = next_frontier

        base_fast_latency = float(rng.uniform(0.8, 1.4))
        fast_multiplier = float(rng.uniform(0.8, 4.5))
        fast_blocked = float(rng.uniform(0.0, 1.0) < 0.35)
        fast_effective = base_fast_latency * (fast_multiplier + 3.0 * fast_blocked)
        fast_edges.append((root, hub_a, base_fast_latency, fast_effective))

        slow_nominal = float(rng.uniform(1.8, 3.2))
        slow_effective = slow_nominal * float(rng.uniform(0.95, 1.2))
        slow_edges.append((root, hub_b, slow_nominal, slow_effective))

        community_queue = float(rng.uniform(*config.community_base_queue))
        monitor_signal = np.zeros((1, 3), dtype=np.float32)
        monitor_signal[0, 0] = fast_multiplier + rng.normal(0.0, config.monitor_noise_std)
        monitor_signal[0, 1] = fast_blocked
        monitor_signal[0, 2] = community_queue / 8.0

    num_nodes = len(node_roles)
    adj = np.zeros((num_nodes, num_nodes), dtype=bool)
    nominal_latency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    effective_latency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    capacity = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    residual_capacity = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    is_fast = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    is_hidden_corridor = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    node_queue = np.zeros((num_nodes,), dtype=np.float32)
    monitor_signal = np.zeros((num_nodes, 3), dtype=np.float32)

    _connect_undirected(
        adj,
        nominal_latency,
        effective_latency,
        capacity,
        residual_capacity,
        is_fast,
        is_hidden_corridor,
        hub_a,
        hub_b,
        nominal=1.9,
        effective=2.1,
        cap=8.0,
    )
    for community in range(config.num_communities):
        nodes_in_community = [idx for idx, comm in enumerate(node_communities) if comm == community]
        base_queue = float(rng.uniform(*config.community_base_queue))
        for node_id in nodes_in_community:
            role_name = ROLE_NAMES[node_roles[node_id]]
            noise = float(rng.uniform(0.0, 1.0))
            node_queue[node_id] = base_queue + noise * (1.0 if role_name != "hub" else 0.5)

        root = community_roots[community]
        monitor = community_monitors[community]
        hidden_multiplier = float(rng.uniform(0.8, 4.5))
        blocked = float(rng.uniform(0.0, 1.0) < 0.35)
        monitor_signal[monitor, 0] = hidden_multiplier + rng.normal(0.0, config.monitor_noise_std)
        monitor_signal[monitor, 1] = blocked
        monitor_signal[monitor, 2] = node_queue[root] / 8.0

    for u, v in tree_edges:
        role_u = ROLE_NAMES[node_roles[u]]
        role_v = ROLE_NAMES[node_roles[v]]
        nominal = float(rng.uniform(0.8, 1.6))
        effective = nominal * float(rng.uniform(0.9, 1.1))
        cap = float(rng.uniform(3.0, 7.0) if dynamic_capacities else rng.uniform(6.0, 10.0))
        if role_u == "monitor" or role_v == "monitor":
            nominal = 0.6
            effective = 0.6
            cap = 6.0
        _connect_undirected(
            adj,
            nominal_latency,
            effective_latency,
            capacity,
            residual_capacity,
            is_fast,
            is_hidden_corridor,
            u,
            v,
            nominal=nominal,
            effective=effective,
            cap=cap,
        )
    for _community, (root, hub, nominal, effective) in enumerate(fast_edges):
        cap = float(rng.uniform(2.0, 5.0) if dynamic_capacities else rng.uniform(5.0, 8.0))
        _connect_undirected(
            adj,
            nominal_latency,
            effective_latency,
            capacity,
            residual_capacity,
            is_fast,
            is_hidden_corridor,
            root,
            hub,
            nominal=nominal,
            effective=effective,
            cap=cap,
            fast=True,
            hidden_corridor=True,
        )

    for root, hub, nominal, effective in slow_edges:
        cap = float(rng.uniform(5.0, 8.0) if dynamic_capacities else rng.uniform(7.0, 10.0))
        _connect_undirected(
            adj,
            nominal_latency,
            effective_latency,
            capacity,
            residual_capacity,
            is_fast,
            is_hidden_corridor,
            root,
            hub,
            nominal=nominal,
            effective=effective,
            cap=cap,
        )

    return GraphState(
        adj=adj,
        node_roles=np.asarray(node_roles, dtype=np.int64),
        node_communities=np.asarray(node_communities, dtype=np.int64),
        node_queue=node_queue,
        monitor_signal=monitor_signal,
        edge_nominal_latency=nominal_latency,
        edge_effective_latency=effective_latency,
        edge_capacity=capacity,
        edge_residual_capacity=residual_capacity,
        edge_is_fast=is_fast,
        edge_is_hidden_corridor=is_hidden_corridor,
        leaf_nodes_by_community=leaf_nodes_by_community,
    )


def sample_packets(
    rng: np.random.Generator,
    graph: GraphState,
    config: HiddenCorridorConfig,
    *,
    packet_count: int,
) -> tuple[PacketSpec, ...]:
    packets: list[PacketSpec] = []
    for _ in range(packet_count):
        src_comm = int(rng.integers(0, config.num_communities))
        dst_comm = int(rng.integers(0, config.num_communities - 1))
        if dst_comm >= src_comm:
            dst_comm += 1
        source = int(rng.choice(graph.leaf_nodes_by_community[src_comm]))
        destination = int(rng.choice(graph.leaf_nodes_by_community[dst_comm]))
        priority = float(rng.integers(1, 4))
        deadline = float(rng.uniform(config.deadline_min, config.deadline_max))
        packets.append(PacketSpec(source=source, destination=destination, priority=priority, deadline=deadline))
    if config.deadline_mode == "oracle_calibrated":
        return calibrate_packet_deadlines(rng, graph, tuple(packets), config)
    return tuple(packets)


def calibrate_packet_deadlines(
    rng: np.random.Generator,
    graph: GraphState,
    packets: tuple[PacketSpec, ...],
    config: HiddenCorridorConfig,
) -> tuple[PacketSpec, ...]:
    working_graph = graph.copy()
    calibrated = list(packets)
    ordered_indices = sorted(range(len(calibrated)), key=lambda idx: (-calibrated[idx].priority, idx))
    max_steps = working_graph.num_nodes * config.max_steps_multiplier

    for packet_index in ordered_indices:
        packet = calibrated[packet_index]
        reference_packet = PacketSpec(
            source=packet.source,
            destination=packet.destination,
            priority=packet.priority,
            deadline=config.deadline_reference_budget,
            size=packet.size,
        )
        _reference_path, reference_cost = shortest_path(
            working_graph,
            reference_packet,
            start=packet.source,
            remaining_deadline=reference_packet.deadline,
            config=config,
        )
        if not math.isfinite(reference_cost):
            continue

        slack_ratio = float(rng.uniform(*config.deadline_slack_ratio_range))
        slack_abs = float(rng.uniform(*config.deadline_slack_abs_range))
        provisional_deadline = reference_cost * (1.0 + slack_ratio) + slack_abs
        provisional_packet = PacketSpec(
            source=packet.source,
            destination=packet.destination,
            priority=packet.priority,
            deadline=provisional_deadline,
            size=packet.size,
        )
        _provisional_path, calibrated_cost = shortest_path(
            working_graph,
            provisional_packet,
            start=packet.source,
            remaining_deadline=provisional_deadline,
            config=config,
        )
        if math.isfinite(calibrated_cost):
            provisional_deadline = max(
                provisional_deadline,
                calibrated_cost + 0.5 * slack_abs,
            )
        calibrated_packet = PacketSpec(
            source=packet.source,
            destination=packet.destination,
            priority=packet.priority,
            deadline=provisional_deadline,
            size=packet.size,
        )
        calibrated[packet_index] = calibrated_packet

        current = calibrated_packet.source
        remaining_deadline = calibrated_packet.deadline
        steps = 0
        while current != calibrated_packet.destination and steps < max_steps:
            path, _path_cost = shortest_path(
                working_graph,
                calibrated_packet,
                start=current,
                remaining_deadline=remaining_deadline,
                config=config,
            )
            if len(path) < 2:
                break
            next_hop = path[1]
            transition_cost = _edge_cost(
                working_graph,
                calibrated_packet,
                current,
                next_hop,
                remaining_deadline=remaining_deadline,
                config=config,
            )
            remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
            _apply_transition(working_graph, current, next_hop, calibrated_packet)
            current = next_hop
            steps += 1

    return tuple(calibrated)


def _edge_cost(
    graph: GraphState,
    packet: PacketSpec,
    u: int,
    v: int,
    *,
    remaining_deadline: float,
    config: HiddenCorridorConfig,
) -> float:
    nominal = float(graph.edge_nominal_latency[u, v])
    effective = float(graph.edge_effective_latency[u, v])
    capacity = max(float(graph.edge_capacity[u, v]), 1e-3)
    residual = float(graph.edge_residual_capacity[u, v])
    utilization = max(0.0, 1.0 - residual / capacity)
    congestion = config.queue_penalty * 0.5 * (float(graph.node_queue[u]) + float(graph.node_queue[v]))
    urgency = config.urgency_penalty * packet.priority / max(remaining_deadline, 1.0)
    return effective * (1.0 + urgency) + congestion + config.capacity_penalty * utilization + 0.25 * nominal


def shortest_path(
    graph: GraphState,
    packet: PacketSpec,
    *,
    start: int,
    remaining_deadline: float,
    config: HiddenCorridorConfig,
) -> tuple[list[int], float]:
    target = packet.destination
    dist = [math.inf] * graph.num_nodes
    prev = [-1] * graph.num_nodes
    dist[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist[node]:
            continue
        if node == target:
            break
        neighbors = np.flatnonzero(graph.adj[node])
        for neighbor in neighbors:
            edge_cost = _edge_cost(
                graph,
                packet,
                node,
                int(neighbor),
                remaining_deadline=max(remaining_deadline - cost, 1.0),
                config=config,
            )
            new_cost = cost + edge_cost
            if new_cost < dist[int(neighbor)]:
                dist[int(neighbor)] = new_cost
                prev[int(neighbor)] = node
                heapq.heappush(heap, (new_cost, int(neighbor)))

    if not math.isfinite(dist[target]):
        return [start], math.inf

    path = [target]
    cursor = target
    while cursor != start:
        cursor = prev[cursor]
        if cursor == -1:
            return [start], math.inf
        path.append(cursor)
    path.reverse()
    return path, float(dist[target])


def _apply_transition(graph: GraphState, u: int, v: int, packet: PacketSpec) -> None:
    graph.edge_residual_capacity[u, v] = max(graph.edge_residual_capacity[u, v] - packet.size, 0.0)
    graph.edge_residual_capacity[v, u] = max(graph.edge_residual_capacity[v, u] - packet.size, 0.0)
    graph.node_queue[u] += 0.15 * packet.priority
    graph.node_queue[v] += 0.25 * packet.priority


def _bfs_order(adjacency: np.ndarray, start: int) -> np.ndarray:
    distances = np.full((adjacency.shape[0],), fill_value=np.inf, dtype=np.float32)
    queue = [start]
    distances[start] = 0.0
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for neighbor in np.flatnonzero(adjacency[node]):
            if math.isinf(float(distances[int(neighbor)])):
                distances[int(neighbor)] = distances[node] + 1.0
                queue.append(int(neighbor))
    return np.argsort(distances, kind="stable")


def make_decision_record(
    graph: GraphState,
    packet: PacketSpec,
    *,
    config: HiddenCorridorConfig | None = None,
    current_node: int,
    target_next_hop: int,
    cost_to_go: float,
    route_nodes: Iterable[int],
    packet_index: int,
    packet_count: int,
    curriculum_level: str,
    include_candidate_targets: bool = True,
) -> DecisionRecord:
    del include_candidate_targets
    num_nodes = graph.num_nodes
    community_one_hot = np.zeros((num_nodes, 4), dtype=np.float32)
    valid_communities = graph.node_communities >= 0
    community_ids = graph.node_communities[valid_communities]
    community_one_hot[valid_communities, community_ids] = 1.0

    is_source = np.zeros((num_nodes, 1), dtype=np.float32)
    is_destination = np.zeros((num_nodes, 1), dtype=np.float32)
    is_current = np.zeros((num_nodes, 1), dtype=np.float32)
    is_source[packet.source, 0] = 1.0
    is_destination[packet.destination, 0] = 1.0
    is_current[current_node, 0] = 1.0

    priority_feature = np.full((num_nodes, 1), packet.priority / 3.0, dtype=np.float32)
    deadline_feature = np.full((num_nodes, 1), packet.deadline / 20.0, dtype=np.float32)
    queue_feature = (graph.node_queue[:, None] / 10.0).astype(np.float32)
    node_features = np.concatenate(
        [
            queue_feature,
            graph.monitor_signal.astype(np.float32),
            is_source,
            is_destination,
            is_current,
            priority_feature,
            deadline_feature,
            community_one_hot,
        ],
        axis=-1,
    )

    edge_features = np.stack(
        [
            graph.edge_nominal_latency / 4.0,
            np.divide(
                graph.edge_residual_capacity,
                np.maximum(graph.edge_capacity, 1e-3),
                out=np.zeros_like(graph.edge_capacity),
                where=graph.edge_capacity > 0.0,
            ),
            graph.edge_is_fast,
            graph.edge_is_hidden_corridor,
        ],
        axis=-1,
    ).astype(np.float32)

    route_relevance = np.zeros((num_nodes,), dtype=np.float32)
    for node in route_nodes:
        route_relevance[int(node)] = 1.0

    candidate_cost_to_go = np.zeros((num_nodes,), dtype=np.float32)
    candidate_slack = np.zeros((num_nodes,), dtype=np.float32)
    candidate_on_time = np.zeros((num_nodes,), dtype=np.float32)
    candidate_path_nodes = np.full((num_nodes, num_nodes), fill_value=-1, dtype=np.int64)
    candidate_path_mask = np.zeros((num_nodes, num_nodes), dtype=bool)
    candidate_path_features = np.zeros((num_nodes, 5), dtype=np.float32)
    if config is not None:
        for candidate in np.flatnonzero(graph.adj[current_node]):
            candidate = int(candidate)
            candidate_graph = graph.copy()
            transition_cost = _edge_cost(
                candidate_graph,
                packet,
                current_node,
                candidate,
                remaining_deadline=packet.deadline,
                config=config,
            )
            remaining_deadline = max(packet.deadline - transition_cost, 0.0)
            _apply_transition(candidate_graph, current_node, candidate, packet)
            if candidate == packet.destination:
                full_path = [current_node, candidate]
                path_valid = True
            else:
                downstream_path, downstream_cost = shortest_path(
                    candidate_graph,
                    packet,
                    start=candidate,
                    remaining_deadline=max(remaining_deadline, 1.0),
                    config=config,
                )
                path_valid = len(downstream_path) >= 1 and math.isfinite(downstream_cost)
                full_path = [current_node, *downstream_path] if path_valid else [current_node, candidate]
                total_cost = transition_cost + downstream_cost if path_valid else math.inf
            path_len = min(len(full_path), num_nodes)
            candidate_path_nodes[candidate, :path_len] = np.asarray(full_path[:path_len], dtype=np.int64)
            candidate_path_mask[candidate, :path_len] = path_valid
            candidate_path_features[candidate] = _path_summary_features(graph, full_path)
            if candidate == packet.destination:
                total_cost = transition_cost
            if math.isfinite(total_cost):
                candidate_cost_to_go[candidate] = float(total_cost)
                candidate_slack[candidate] = float(packet.deadline - total_cost)
                candidate_on_time[candidate] = float(total_cost <= packet.deadline)
            else:
                candidate_cost_to_go[candidate] = 1e6
                candidate_slack[candidate] = -1e6
                candidate_on_time[candidate] = 0.0

    return DecisionRecord(
        node_features=node_features,
        edge_features=edge_features,
        node_roles=graph.node_roles.copy(),
        node_communities=graph.node_communities.copy(),
        adjacency=graph.adj.copy(),
        current_node=current_node,
        source_node=packet.source,
        destination_node=packet.destination,
        order_current=_bfs_order(graph.adj, current_node),
        order_destination=_bfs_order(graph.adj, packet.destination),
        candidate_mask=graph.adj[current_node].copy(),
        target_next_hop=target_next_hop,
        cost_to_go=cost_to_go,
        candidate_cost_to_go=candidate_cost_to_go,
        candidate_slack=candidate_slack,
        candidate_on_time=candidate_on_time,
        candidate_path_nodes=candidate_path_nodes,
        candidate_path_mask=candidate_path_mask,
        candidate_path_features=candidate_path_features,
        route_relevance=route_relevance,
        packet_priority=packet.priority,
        packet_deadline=packet.deadline,
        packet_index=packet_index,
        packet_count=packet_count,
        curriculum_level=curriculum_level,
    )


def oracle_rollout(
    rng: np.random.Generator,
    graph: GraphState,
    packets: tuple[PacketSpec, ...],
    config: HiddenCorridorConfig,
    *,
    curriculum_level: str,
) -> tuple[list[DecisionRecord], EpisodeRecord]:
    del rng
    working_graph = graph.copy()
    ordered_packets = sorted(
        enumerate(packets),
        key=lambda item: (-item[1].priority, item[1].deadline),
    )
    decisions: list[DecisionRecord] = []
    total_cost = 0.0

    max_steps = working_graph.num_nodes * config.max_steps_multiplier
    for packet_index, packet in ordered_packets:
        current = packet.source
        steps = 0
        remaining_deadline = packet.deadline
        while current != packet.destination and steps < max_steps:
            path, path_cost = shortest_path(
                working_graph,
                packet,
                start=current,
                remaining_deadline=remaining_deadline,
                config=config,
            )
            if len(path) < 2 or not math.isfinite(path_cost):
                break
            next_hop = path[1]
            decisions.append(
                make_decision_record(
                    working_graph,
                    packet,
                    config=config,
                    current_node=current,
                    target_next_hop=next_hop,
                    cost_to_go=path_cost,
                    route_nodes=path,
                    packet_index=packet_index,
                    packet_count=len(packets),
                    curriculum_level=curriculum_level,
                )
            )
            transition_cost = _edge_cost(
                working_graph,
                packet,
                current,
                next_hop,
                remaining_deadline=remaining_deadline,
                config=config,
            )
            total_cost += transition_cost
            remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
            _apply_transition(working_graph, current, next_hop, packet)
            current = next_hop
            steps += 1

    return decisions, EpisodeRecord(
        graph=graph.copy(),
        packets=packets,
        oracle_total_cost=total_cost,
        decision_count=len(decisions),
        curriculum_level=curriculum_level,
    )


def curriculum_packet_range(level: str, config: HiddenCorridorConfig) -> tuple[int, bool]:
    if level == "single_static":
        return 1, False
    if level == "single_dynamic":
        return 1, True
    if level == "multi_dynamic":
        return int(config.packets_max), True
    raise ValueError(f"Unknown curriculum level: {level}")


def generate_hidden_corridor_episode(
    rng: np.random.Generator,
    config: HiddenCorridorConfig,
    *,
    curriculum_level: str,
) -> tuple[list[DecisionRecord], EpisodeRecord]:
    packet_cap, dynamic_capacities = curriculum_packet_range(curriculum_level, config)
    graph = build_hidden_corridor_graph(rng, config, dynamic_capacities=dynamic_capacities)
    if packet_cap == 1:
        packet_count = 1
    else:
        packet_count = int(rng.integers(config.packets_min, packet_cap + 1))
    packets = sample_packets(rng, graph, config, packet_count=packet_count)
    return oracle_rollout(rng, graph, packets, config, curriculum_level=curriculum_level)


class HiddenCorridorDecisionDataset(Dataset[DecisionRecord]):
    def __init__(
        self,
        *,
        config: HiddenCorridorConfig,
        num_episodes: int,
        curriculum_levels: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config
        self.curriculum_levels = curriculum_levels or config.train_curriculum_levels
        rng = np.random.default_rng(config.seed)
        decisions: list[DecisionRecord] = []
        episodes: list[EpisodeRecord] = []
        for episode_idx in range(num_episodes):
            level = self.curriculum_levels[episode_idx % len(self.curriculum_levels)]
            episode_decisions, episode = generate_hidden_corridor_episode(
                rng,
                config,
                curriculum_level=level,
            )
            decisions.extend(episode_decisions)
            episodes.append(episode)
        self._decisions = decisions
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self._decisions)

    def __getitem__(self, index: int) -> DecisionRecord:
        return self._decisions[index]

    @staticmethod
    def _graph_hash(graph: GraphState) -> str:
        hasher = hashlib.sha256()
        hasher.update(graph.adj.tobytes())
        hasher.update(graph.node_roles.tobytes())
        hasher.update(graph.node_communities.tobytes())
        hasher.update(graph.node_queue.tobytes())
        hasher.update(graph.monitor_signal.tobytes())
        hasher.update(graph.edge_nominal_latency.tobytes())
        hasher.update(graph.edge_effective_latency.tobytes())
        hasher.update(graph.edge_capacity.tobytes())
        hasher.update(graph.edge_residual_capacity.tobytes())
        hasher.update(graph.edge_is_fast.tobytes())
        hasher.update(graph.edge_is_hidden_corridor.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def _packets_hash(packets: tuple[PacketSpec, ...]) -> str:
        payload = [
            {
                "source": packet.source,
                "destination": packet.destination,
                "priority": packet.priority,
                "deadline": packet.deadline,
                "size": packet.size,
            }
            for packet in packets
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def manifest(self) -> dict[str, object]:
        episodes: list[dict[str, object]] = []
        manifest_hasher = hashlib.sha256()
        for index, episode in enumerate(self.episodes):
            entry = {
                "index": index,
                "curriculum_level": episode.curriculum_level,
                "decision_count": episode.decision_count,
                "oracle_total_cost": episode.oracle_total_cost,
                "packet_count": len(episode.packets),
                "graph_hash": self._graph_hash(episode.graph),
                "packets_hash": self._packets_hash(episode.packets),
            }
            manifest_hasher.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
            episodes.append(entry)
        return {
            "config_seed": self.config.seed,
            "episode_count": len(self.episodes),
            "decision_count": len(self._decisions),
            "curriculum_levels": list(self.curriculum_levels),
            "manifest_hash": manifest_hasher.hexdigest(),
            "episodes": episodes,
        }


def collate_decisions(records: list[DecisionRecord]) -> dict[str, torch.Tensor]:
    max_nodes = max(record.node_features.shape[0] for record in records)
    node_feature_dim = records[0].node_features.shape[-1]
    edge_feature_dim = records[0].edge_features.shape[-1]

    batch_size = len(records)
    node_features = torch.zeros((batch_size, max_nodes, node_feature_dim), dtype=torch.float32)
    edge_features = torch.zeros(
        (batch_size, max_nodes, max_nodes, edge_feature_dim),
        dtype=torch.float32,
    )
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    node_roles = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    node_communities = torch.full((batch_size, max_nodes), fill_value=-1, dtype=torch.long)
    candidate_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    route_relevance = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    order_current = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    order_destination = torch.zeros((batch_size, max_nodes), dtype=torch.long)

    current_node = torch.zeros((batch_size,), dtype=torch.long)
    source_node = torch.zeros((batch_size,), dtype=torch.long)
    destination_node = torch.zeros((batch_size,), dtype=torch.long)
    target_next_hop = torch.zeros((batch_size,), dtype=torch.long)
    cost_to_go = torch.zeros((batch_size,), dtype=torch.float32)
    candidate_cost_to_go = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    candidate_slack = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    candidate_on_time = torch.zeros((batch_size, max_nodes), dtype=torch.float32)
    candidate_path_nodes = torch.full((batch_size, max_nodes, max_nodes), fill_value=-1, dtype=torch.long)
    candidate_path_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    candidate_path_features = torch.zeros((batch_size, max_nodes, 5), dtype=torch.float32)
    packet_priority = torch.zeros((batch_size,), dtype=torch.float32)
    packet_deadline = torch.zeros((batch_size,), dtype=torch.float32)
    packet_index = torch.zeros((batch_size,), dtype=torch.long)
    packet_count = torch.zeros((batch_size,), dtype=torch.long)

    for batch_index, record in enumerate(records):
        num_nodes = record.node_features.shape[0]
        node_features[batch_index, :num_nodes] = torch.from_numpy(record.node_features)
        edge_features[batch_index, :num_nodes, :num_nodes] = torch.from_numpy(record.edge_features)
        adjacency[batch_index, :num_nodes, :num_nodes] = torch.from_numpy(record.adjacency)
        node_mask[batch_index, :num_nodes] = True
        node_roles[batch_index, :num_nodes] = torch.from_numpy(record.node_roles)
        node_communities[batch_index, :num_nodes] = torch.from_numpy(record.node_communities)
        candidate_mask[batch_index, :num_nodes] = torch.from_numpy(record.candidate_mask)
        route_relevance[batch_index, :num_nodes] = torch.from_numpy(record.route_relevance)
        order_current[batch_index, :num_nodes] = torch.from_numpy(record.order_current)
        order_destination[batch_index, :num_nodes] = torch.from_numpy(record.order_destination)
        current_node[batch_index] = record.current_node
        source_node[batch_index] = record.source_node
        destination_node[batch_index] = record.destination_node
        target_next_hop[batch_index] = record.target_next_hop
        cost_to_go[batch_index] = record.cost_to_go
        candidate_cost_to_go[batch_index, :num_nodes] = torch.from_numpy(record.candidate_cost_to_go)
        candidate_slack[batch_index, :num_nodes] = torch.from_numpy(record.candidate_slack)
        candidate_on_time[batch_index, :num_nodes] = torch.from_numpy(record.candidate_on_time)
        candidate_path_nodes[batch_index, :num_nodes, :num_nodes] = torch.from_numpy(record.candidate_path_nodes)
        candidate_path_mask[batch_index, :num_nodes, :num_nodes] = torch.from_numpy(record.candidate_path_mask)
        candidate_path_features[batch_index, :num_nodes] = torch.from_numpy(record.candidate_path_features)
        packet_priority[batch_index] = record.packet_priority
        packet_deadline[batch_index] = record.packet_deadline
        packet_index[batch_index] = record.packet_index
        packet_count[batch_index] = record.packet_count

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "adjacency": adjacency,
        "node_mask": node_mask,
        "node_roles": node_roles,
        "node_communities": node_communities,
        "candidate_mask": candidate_mask,
        "route_relevance": route_relevance,
        "order_current": order_current,
        "order_destination": order_destination,
        "current_node": current_node,
        "source_node": source_node,
        "destination_node": destination_node,
        "target_next_hop": target_next_hop,
        "cost_to_go": cost_to_go,
        "candidate_cost_to_go": candidate_cost_to_go,
        "candidate_slack": candidate_slack,
        "candidate_on_time": candidate_on_time,
        "candidate_path_nodes": candidate_path_nodes,
        "candidate_path_mask": candidate_path_mask,
        "candidate_path_features": candidate_path_features,
        "packet_priority": packet_priority,
        "packet_deadline": packet_deadline,
        "packet_index": packet_index,
        "packet_count": packet_count,
    }
