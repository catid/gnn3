#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from gnn3.data.hidden_corridor import (
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
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--decision-horizon", type=int, default=2)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round9_branch_teacher",
        help="Prefix for CSV/JSON outputs.",
    )
    parser.add_argument("--selection-strategy", default="final")
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, *, device_override: str | None = None) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


@dataclass
class BranchState:
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
    state: BranchState,
    chosen_next_hop: int,
    *,
    config,
) -> tuple[bool, float, bool]:
    if not state.graph.adj[state.current_node, chosen_next_hop]:
        return False, 0.0, False
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
    return True, float(transition_cost), reached_destination


def _simulate_branch(
    model: PacketMambaModel,
    state: BranchState,
    *,
    forced_next_hop: int,
    horizon: int,
    device: torch.device,
    config,
    curriculum_level: str,
) -> dict[str, float | bool | int]:
    total_cost = 0.0
    decisions = 0
    deadline_miss = False
    state = BranchState(
        graph=state.graph.copy(),
        ordered_packets=list(state.ordered_packets),
        packet_cursor=state.packet_cursor,
        current_packet_index=state.current_packet_index,
        current_packet=PacketSpec(
            source=state.current_packet.source,
            destination=state.current_packet.destination,
            priority=state.current_packet.priority,
            deadline=state.current_packet.deadline,
            size=state.current_packet.size,
        ),
        current_node=state.current_node,
        remaining_deadline=state.remaining_deadline,
    )
    ok, transition_cost, reached_destination = _advance_one_decision(state, forced_next_hop, config=config)
    if not ok:
        return {"valid": False, "decisions": 0, "cost": 0.0, "deadline_miss": True}
    decisions += 1
    total_cost += transition_cost
    deadline_miss = state.remaining_deadline <= 0.0

    while decisions < horizon:
        if reached_destination:
            state.packet_cursor += 1
            if state.packet_cursor >= len(state.ordered_packets):
                break
            packet_index, packet = state.ordered_packets[state.packet_cursor]
            state.current_packet_index = packet_index
            state.current_packet = PacketSpec(
                source=packet.source,
                destination=packet.destination,
                priority=packet.priority,
                deadline=packet.deadline,
                size=packet.size,
            )
            state.current_node = packet.source
            state.remaining_deadline = packet.deadline

        record, _oracle = _make_record(
            state.graph,
            state.current_packet,
            current_node=state.current_node,
            remaining_deadline=state.remaining_deadline,
            packet_index=state.current_packet_index,
            packet_count=len(state.ordered_packets),
            curriculum_level=curriculum_level,
            config=config,
        )
        if record is None:
            break
        scores = _selection_scores(model, record, device=device)
        chosen_next_hop = int(scores.argmax().item())
        ok, transition_cost, reached_destination = _advance_one_decision(state, chosen_next_hop, config=config)
        if not ok:
            deadline_miss = True
            break
        decisions += 1
        total_cost += transition_cost
        deadline_miss = deadline_miss or (state.remaining_deadline <= 0.0)

    return {
        "valid": True,
        "decisions": decisions,
        "cost": total_cost,
        "deadline_miss": deadline_miss,
    }


def _target_set(frontier_df: pd.DataFrame, suite: str, target_col: str) -> set[tuple[int, int]]:
    frame = frontier_df.loc[(frontier_df["suite"] == suite) & frontier_df[target_col]].copy()
    return {(int(row.episode_index), int(row.decision_index)) for row in frame.itertuples(index=False)}


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frontier_df = pd.read_csv(args.frontier_decisions_csv)
    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)

    branch_rows: list[dict[str, object]] = []
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
                    scores = _selection_scores(model, record, device=device)
                    valid_mask = torch.from_numpy(record.candidate_mask.astype(bool))
                    valid_scores = scores.masked_fill(~valid_mask, -1e9)
                    ranked = torch.topk(valid_scores, k=min(args.top_k, int(valid_mask.sum()))).indices.tolist()
                    base_choice = int(valid_scores.argmax().item())

                    if (episode_index, decision_index) in target_decisions:
                        branch_state = BranchState(
                            graph=working_graph,
                            ordered_packets=ordered_packets,
                            packet_cursor=packet_cursor,
                            current_packet_index=packet_index,
                            current_packet=packet,
                            current_node=current,
                            remaining_deadline=remaining_deadline,
                        )
                        candidate_rows = []
                        for rank, candidate in enumerate(ranked):
                            result = _simulate_branch(
                                model,
                                branch_state,
                                forced_next_hop=int(candidate),
                                horizon=args.decision_horizon,
                                device=device,
                                config=hidden_cfg,
                                curriculum_level=episode.curriculum_level,
                            )
                            if not result["valid"]:
                                continue
                            penalty = 100.0 if bool(result["deadline_miss"]) else 0.0
                            score = float(result["cost"]) + penalty
                            candidate_rows.append(
                                {
                                    "suite": suite_config.name,
                                    "episode_index": episode_index,
                                    "decision_index": decision_index,
                                    "packet_index": packet_index,
                                    "candidate_next_hop": int(candidate),
                                    "candidate_rank": rank,
                                    "base_choice": base_choice,
                                    "oracle_next_hop": int(oracle_next_hop),
                                    "teacher_score": score,
                                    "branch_cost": float(result["cost"]),
                                    "branch_deadline_miss": bool(result["deadline_miss"]),
                                    "branch_decisions": int(result["decisions"]),
                                    "base_target_match": bool(base_choice == int(oracle_next_hop)),
                                    "candidate_target_match": bool(int(candidate) == int(oracle_next_hop)),
                                }
                            )
                        if candidate_rows:
                            best_score = min(float(row["teacher_score"]) for row in candidate_rows)
                            for row in candidate_rows:
                                row["teacher_choice"] = bool(float(row["teacher_score"]) <= best_score + 1e-9)
                                branch_rows.append(row)

                    if not working_graph.adj[current, base_choice]:
                        break
                    transition_cost = _edge_cost(
                        working_graph,
                        packet,
                        current,
                        base_choice,
                        remaining_deadline=remaining_deadline,
                        config=hidden_cfg,
                    )
                    remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
                    _apply_transition(working_graph, current, base_choice, packet)
                    current = base_choice
                    steps += 1

    if not branch_rows:
        raise SystemExit("No branch-teacher rows were generated for the requested target slice.")

    branch_df = pd.DataFrame(branch_rows)
    best_df = branch_df.loc[branch_df["teacher_choice"]].copy()
    summary_rows.append(
        {
            "target_col": args.target_col,
            "top_k": args.top_k,
            "decision_horizon": args.decision_horizon,
            "branch_rows": len(branch_df),
            "teacher_decisions": len(best_df),
            "teacher_disagreement": float((best_df["candidate_next_hop"] != best_df["base_choice"]).mean()),
            "teacher_recovery": float(
                ((~best_df["base_target_match"]) & best_df["candidate_target_match"]).mean()
            ),
            "teacher_new_error": float(
                (best_df["base_target_match"] & (~best_df["candidate_target_match"])).mean()
            ),
            "teacher_target_match": float(best_df["candidate_target_match"].mean()),
        }
    )
    summary_df = pd.DataFrame(summary_rows)
    branch_df.to_csv(output_prefix.with_name(output_prefix.name + "_branches.csv"), index=False)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "teacher_rows": len(best_df),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
