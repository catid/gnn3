#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from gnn3.data.hidden_corridor import (
    ROLE_NAMES,
    DecisionRecord,
    HiddenCorridorDecisionDataset,
    PacketSpec,
    _apply_transition,
    _edge_cost,
    collate_decisions,
    make_decision_record,
    shortest_path,
)
from gnn3.eval.rollout import _move_batch
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--frontier-decisions-csv", required=True)
    parser.add_argument("--target-col", default="hard_near_tie_intersection_case")
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round9_route_persistence",
        help="Prefix for CSV/JSON outputs.",
    )
    return parser.parse_args()


def _load_model(
    config_path: str,
    checkpoint_path: str,
    *,
    device_override: str | None = None,
) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


@dataclass
class DecisionState:
    graph: object
    ordered_packets: list[tuple[int, PacketSpec]]
    packet_cursor: int
    current_packet_index: int
    current_packet: PacketSpec
    current_node: int
    remaining_deadline: float


@torch.no_grad()
def _selection_scores(
    model: PacketMambaModel,
    record: DecisionRecord,
    *,
    device: torch.device,
) -> torch.Tensor:
    batch = _move_batch(collate_decisions([record]), device)
    output = model(batch)
    return output["selection_scores"][0].detach().cpu()


def _copy_packet(packet: PacketSpec) -> PacketSpec:
    return PacketSpec(
        source=packet.source,
        destination=packet.destination,
        priority=packet.priority,
        deadline=packet.deadline,
        size=packet.size,
    )


def _copy_state(state: DecisionState) -> DecisionState:
    return DecisionState(
        graph=state.graph.copy(),
        ordered_packets=[(packet_index, _copy_packet(packet)) for packet_index, packet in state.ordered_packets],
        packet_cursor=state.packet_cursor,
        current_packet_index=state.current_packet_index,
        current_packet=_copy_packet(state.current_packet),
        current_node=state.current_node,
        remaining_deadline=state.remaining_deadline,
    )


def _target_set(frontier_df: pd.DataFrame, suite: str, target_col: str) -> set[tuple[int, int]]:
    frame = frontier_df.loc[(frontier_df["suite"] == suite) & frontier_df[target_col]].copy()
    return {(int(row.episode_index), int(row.decision_index)) for row in frame.itertuples(index=False)}


def _first_hub_on_path(graph, path: list[int]) -> int | None:
    for node in path[1:]:
        if ROLE_NAMES[int(graph.node_roles[node])] == "hub":
            return int(node)
    return None


def _oracle_path_and_hub(
    graph,
    packet: PacketSpec,
    *,
    current_node: int,
    remaining_deadline: float,
    config,
) -> tuple[list[int], int | None]:
    path, _cost = shortest_path(
        graph,
        packet,
        start=current_node,
        remaining_deadline=remaining_deadline,
        config=config,
    )
    return path, _first_hub_on_path(graph, path)


def _make_record(
    graph,
    packet: PacketSpec,
    *,
    current_node: int,
    remaining_deadline: float,
    packet_index: int,
    packet_count: int,
    curriculum_level: str,
    config,
) -> tuple[DecisionRecord | None, int | None]:
    path, path_cost = shortest_path(
        graph,
        packet,
        start=current_node,
        remaining_deadline=remaining_deadline,
        config=config,
    )
    if len(path) < 2:
        return None, None
    oracle_next_hop = path[1]
    record = make_decision_record(
        graph,
        packet,
        config=config,
        current_node=current_node,
        target_next_hop=oracle_next_hop,
        cost_to_go=path_cost,
        route_nodes=path,
        packet_index=packet_index,
        packet_count=packet_count,
        curriculum_level=curriculum_level,
        include_candidate_targets=False,
    )
    return record, oracle_next_hop


def _advance_one_decision(
    state: DecisionState,
    chosen_next_hop: int,
    *,
    config,
) -> tuple[bool, bool]:
    if not state.graph.adj[state.current_node, chosen_next_hop]:
        return False, False
    transition_cost = _edge_cost(
        state.graph,
        state.current_packet,
        state.current_node,
        chosen_next_hop,
        remaining_deadline=state.remaining_deadline,
        config=config,
    )
    state.remaining_deadline = max(state.remaining_deadline - transition_cost, 0.0)
    _apply_transition(state.graph, state.current_node, chosen_next_hop, state.current_packet)
    state.current_node = chosen_next_hop
    reached_destination = state.current_node == state.current_packet.destination
    return True, reached_destination


def _advance_to_next_packet(state: DecisionState) -> bool:
    state.packet_cursor += 1
    if state.packet_cursor >= len(state.ordered_packets):
        return False
    packet_index, packet = state.ordered_packets[state.packet_cursor]
    state.current_packet_index = packet_index
    state.current_packet = _copy_packet(packet)
    state.current_node = packet.source
    state.remaining_deadline = packet.deadline
    return True


def _model_first_hub(
    model: PacketMambaModel,
    state: DecisionState,
    *,
    device: torch.device,
    config,
    curriculum_level: str,
    lookahead_hops: int,
) -> int | None:
    probe_state = _copy_state(state)
    for _ in range(max(lookahead_hops, 1)):
        record, _oracle_next_hop = _make_record(
            probe_state.graph,
            probe_state.current_packet,
            current_node=probe_state.current_node,
            remaining_deadline=probe_state.remaining_deadline,
            packet_index=probe_state.current_packet_index,
            packet_count=len(probe_state.ordered_packets),
            curriculum_level=curriculum_level,
            config=config,
        )
        if record is None:
            return None
        scores = _selection_scores(model, record, device=device)
        chosen_next_hop = int(scores.argmax().item())
        if ROLE_NAMES[int(probe_state.graph.node_roles[chosen_next_hop])] == "hub":
            return chosen_next_hop
        ok, reached_destination = _advance_one_decision(probe_state, chosen_next_hop, config=config)
        if not ok:
            return None
        if reached_destination:
            return None
    return None


def _oracle_persistence_rows(
    state: DecisionState,
    *,
    config,
    max_hops: int,
) -> dict[str, object]:
    oracle_state = _copy_state(state)
    initial_path, initial_hub = _oracle_path_and_hub(
        oracle_state.graph,
        oracle_state.current_packet,
        current_node=oracle_state.current_node,
        remaining_deadline=oracle_state.remaining_deadline,
        config=config,
    )
    row: dict[str, object] = {
        "initial_oracle_hub": -1 if initial_hub is None else int(initial_hub),
        "initial_oracle_path_length": max(len(initial_path) - 1, 0),
    }
    for horizon in range(1, max_hops + 1):
        if len(initial_path) < 2:
            row[f"oracle_stable_h{horizon}"] = False
            continue
        ok, reached_destination = _advance_one_decision(oracle_state, int(initial_path[1]), config=config)
        if not ok:
            row[f"oracle_stable_h{horizon}"] = False
            break
        if reached_destination:
            row[f"oracle_stable_h{horizon}"] = True
            continue
        _, next_hub = _oracle_path_and_hub(
            oracle_state.graph,
            oracle_state.current_packet,
            current_node=oracle_state.current_node,
            remaining_deadline=oracle_state.remaining_deadline,
            config=config,
        )
        row[f"oracle_stable_h{horizon}"] = initial_hub == next_hub
        initial_path, _ = _oracle_path_and_hub(
            oracle_state.graph,
            oracle_state.current_packet,
            current_node=oracle_state.current_node,
            remaining_deadline=oracle_state.remaining_deadline,
            config=config,
        )
    return row


def _model_flip_rows(
    model: PacketMambaModel,
    state: DecisionState,
    *,
    device: torch.device,
    config,
    curriculum_level: str,
    max_hops: int,
) -> dict[str, object]:
    model_state = _copy_state(state)
    initial_hub = _model_first_hub(
        model,
        model_state,
        device=device,
        config=config,
        curriculum_level=curriculum_level,
        lookahead_hops=max_hops,
    )
    row: dict[str, object] = {"initial_model_hub": -1 if initial_hub is None else int(initial_hub)}
    for horizon in range(1, max_hops + 1):
        record, _oracle_next_hop = _make_record(
            model_state.graph,
            model_state.current_packet,
            current_node=model_state.current_node,
            remaining_deadline=model_state.remaining_deadline,
            packet_index=model_state.current_packet_index,
            packet_count=len(model_state.ordered_packets),
            curriculum_level=curriculum_level,
            config=config,
        )
        if record is None:
            row[f"model_flip_h{horizon}"] = False
            continue
        scores = _selection_scores(model, record, device=device)
        chosen_next_hop = int(scores.argmax().item())
        ok, reached_destination = _advance_one_decision(model_state, chosen_next_hop, config=config)
        if not ok:
            row[f"model_flip_h{horizon}"] = False
            break
        if reached_destination:
            row[f"model_flip_h{horizon}"] = False
            break
        next_hub = _model_first_hub(
            model,
            model_state,
            device=device,
            config=config,
            curriculum_level=curriculum_level,
            lookahead_hops=max_hops,
        )
        row[f"model_flip_h{horizon}"] = (initial_hub is not None) and (next_hub is not None) and (next_hub != initial_hub)
    return row


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frontier_df = pd.read_csv(args.frontier_decisions_csv)
    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)

    decision_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        target_decisions = _target_set(frontier_df, suite_config.name, args.target_col)
        if not target_decisions:
            continue

        for episode_index, episode in enumerate(dataset.episodes):
            working_graph = episode.graph.copy()
            ordered_packets = sorted(
                enumerate(episode.packets),
                key=lambda item: (-item[1].priority, item[1].deadline),
            )
            decision_index = -1
            for packet_cursor, (packet_index, packet) in enumerate(ordered_packets):
                current = packet.source
                remaining_deadline = packet.deadline
                max_steps = working_graph.num_nodes * hidden_cfg.max_steps_multiplier
                steps = 0
                while current != packet.destination and steps < max_steps:
                    decision_index += 1
                    record, oracle_next_hop = _make_record(
                        working_graph,
                        packet,
                        current_node=current,
                        remaining_deadline=remaining_deadline,
                        packet_index=packet_index,
                        packet_count=len(episode.packets),
                        curriculum_level=episode.curriculum_level,
                        config=hidden_cfg,
                    )
                    if record is None or oracle_next_hop is None:
                        break
                    if (episode_index, decision_index) in target_decisions:
                        state = DecisionState(
                            graph=working_graph,
                            ordered_packets=ordered_packets,
                            packet_cursor=packet_cursor,
                            current_packet_index=packet_index,
                            current_packet=packet,
                            current_node=current,
                            remaining_deadline=remaining_deadline,
                        )
                        row = {
                            "suite": suite_config.name,
                            "episode_index": episode_index,
                            "decision_index": decision_index,
                            "packet_index": packet_index,
                        }
                        row.update(_oracle_persistence_rows(state, config=hidden_cfg, max_hops=args.max_hops))
                        row.update(
                            _model_flip_rows(
                                model,
                                state,
                                device=device,
                                config=hidden_cfg,
                                curriculum_level=episode.curriculum_level,
                                max_hops=args.max_hops,
                            )
                        )
                        decision_rows.append(row)

                    if not working_graph.adj[current, oracle_next_hop]:
                        break
                    transition_cost = _edge_cost(
                        working_graph,
                        packet,
                        current,
                        oracle_next_hop,
                        remaining_deadline=remaining_deadline,
                        config=hidden_cfg,
                    )
                    remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
                    _apply_transition(working_graph, current, oracle_next_hop, packet)
                    current = oracle_next_hop
                    steps += 1

        if decision_rows:
            suite_frame = pd.DataFrame([row for row in decision_rows if row["suite"] == suite_config.name])
            summary = {
                "suite": suite_config.name,
                "target_decisions": len(suite_frame),
                "oracle_hub_defined_rate": float((suite_frame["initial_oracle_hub"] >= 0).mean()),
                "model_hub_defined_rate": float((suite_frame["initial_model_hub"] >= 0).mean()),
            }
            for horizon in range(1, args.max_hops + 1):
                oracle_mask = suite_frame["initial_oracle_hub"] >= 0
                model_mask = oracle_mask & (suite_frame["initial_model_hub"] >= 0)
                stable = suite_frame[f"oracle_stable_h{horizon}"]
                flip = suite_frame[f"model_flip_h{horizon}"]
                summary[f"oracle_stable_h{horizon}"] = float(suite_frame.loc[oracle_mask, f"oracle_stable_h{horizon}"].mean()) if oracle_mask.any() else 0.0
                summary[f"model_flip_h{horizon}_given_oracle_stable"] = float(
                    suite_frame.loc[model_mask & stable, f"model_flip_h{horizon}"].mean()
                ) if (model_mask & stable).any() else 0.0
                summary[f"model_unnecessary_flip_h{horizon}"] = float(
                    (flip & stable & model_mask).mean()
                ) if len(suite_frame) else 0.0
            summary_rows.append(summary)

    if not decision_rows:
        raise SystemExit("No route-persistence rows were generated for the requested target slice.")

    decision_df = pd.DataFrame(decision_rows)
    summary_df = pd.DataFrame(summary_rows)
    decision_df.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "target_col": args.target_col,
                "max_hops": args.max_hops,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
