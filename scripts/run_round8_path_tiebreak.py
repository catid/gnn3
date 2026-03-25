#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch

from gnn3.data.hidden_corridor import (
    HiddenCorridorDecisionDataset,
    PacketSpec,
    _apply_transition,
    _edge_cost,
    collate_decisions,
    make_decision_record,
    shortest_path,
)
from gnn3.eval.hard_feasible import annotate_hard_feasible
from gnn3.eval.near_tie import model_margin, valid_candidate_mask
from gnn3.eval.policy_analysis import collect_decision_prediction_rows, collect_episode_policy_rows
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--gate-margin", type=float, default=0.10)
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round8_path_tiebreak",
        help="Prefix for CSV/JSON outputs.",
    )
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, *, device_override: str | None = None) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def _predict_with_tiebreak(
    model: PacketMambaModel,
    record,
    graph,
    packet: PacketSpec,
    *,
    config,
    device: torch.device,
    top_k: int,
    gate_margin: float,
) -> tuple[int, bool]:
    batch = {key: value.to(device) for key, value in collate_decisions([record]).items()}
    with torch.no_grad():
        output = model(batch)
    base_scores = output["selection_scores"].squeeze(0)
    valid = valid_candidate_mask(batch).squeeze(0)
    base_choice = int(base_scores.argmax().item())
    margin = float(model_margin(output["selection_scores"], batch).item())
    if margin > gate_margin:
        return base_choice, False

    remaining_deadline = float(record.packet_deadline)
    candidate_ids = (
        base_scores.masked_fill(~valid, -1e9).topk(k=min(top_k, int(valid.sum().item())), dim=-1).indices.tolist()
    )

    best_choice = base_choice
    best_cost = float("inf")
    for candidate in candidate_ids:
        candidate = int(candidate)
        transition_cost = _edge_cost(
            graph,
            packet,
            record.current_node,
            candidate,
            remaining_deadline=remaining_deadline,
            config=config,
        )
        next_remaining_deadline = max(remaining_deadline - transition_cost, 0.0)
        if candidate == packet.destination:
            total_cost = float(transition_cost)
        else:
            next_graph = graph.copy()
            _apply_transition(next_graph, record.current_node, candidate, packet)
            path, suffix_cost = shortest_path(
                next_graph,
                packet,
                start=candidate,
                remaining_deadline=next_remaining_deadline,
                config=config,
            )
            if len(path) < 2 or not math.isfinite(suffix_cost):
                total_cost = 1e6
            else:
                total_cost = float(transition_cost + suffix_cost)
        if total_cost < best_cost:
            best_cost = total_cost
            best_choice = candidate
    return best_choice, True


def _rollout_episode_with_tiebreak(
    episode,
    model: PacketMambaModel,
    *,
    config,
    device: torch.device,
    top_k: int,
    gate_margin: float,
) -> tuple[bool, int, int, float, int, float]:
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
            chosen_next_hop, _triggered = _predict_with_tiebreak(
                model,
                record,
                working_graph,
                packet,
                config=config,
                device=device,
                top_k=top_k,
                gate_margin=gate_margin,
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

    return solved, total_steps, correct_steps, total_cost, deadline_violations, delivered_priority


def _rollout_episode_oracle(episode, *, config) -> tuple[bool, int, int, float, int, float]:
    working_graph = episode.graph.copy()
    ordered_packets = sorted(
        enumerate(episode.packets),
        key=lambda item: (-item[1].priority, item[1].deadline),
    )
    total_steps = 0
    total_cost = 0.0
    deadline_violations = 0
    delivered_priority = 0.0
    max_steps = working_graph.num_nodes * config.max_steps_multiplier
    solved = True

    for _packet_index, packet in ordered_packets:
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
            chosen_next_hop = path[1]
            total_steps += 1
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

    return solved, total_steps, total_steps, total_cost, deadline_violations, delivered_priority


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model, device = _load_model(args.base_config, args.base_checkpoint, device_override=args.device)

    suite_summary_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []

    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        records = list(dataset)
        episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                model,
                dataset.episodes,
                device=device,
                config=hidden_cfg,
                suite=suite_config.name,
            )
        )
        base_decision_df = pd.DataFrame(
            collect_decision_prediction_rows(
                model,
                records,
                device=device,
                suite=suite_config.name,
            )
        )
        base_decision_df, _thresholds = annotate_hard_feasible(base_decision_df, episode_df)
        base_decision_df["row_in_suite"] = range(len(base_decision_df))

        trigger_count = 0
        corrected = 0
        new_errors = 0
        for row_in_suite, record in enumerate(records):
            row = base_decision_df.iloc[row_in_suite]
            episode_index = int(record.episode_index)
            episode = dataset.episodes[episode_index]
            packet = episode.packets[int(record.packet_index)]
            choice, triggered = _predict_with_tiebreak(
                model,
                record,
                episode.graph.copy(),
                packet,
                config=hidden_cfg,
                device=device,
                top_k=args.top_k,
                gate_margin=args.gate_margin,
            )
            trigger_count += int(triggered)
            corrected += int((int(row.predicted_next_hop) != int(row.target_next_hop)) and (choice == int(row.target_next_hop)))
            new_errors += int((int(row.predicted_next_hop) == int(row.target_next_hop)) and (choice != int(row.target_next_hop)))
            decision_rows.append(
                {
                    "suite": suite_config.name,
                    "episode_index": episode_index,
                    "decision_index": int(row.decision_index),
                    "row_in_suite": row_in_suite,
                    "base_choice": int(row.predicted_next_hop),
                    "tiebreak_choice": choice,
                    "target_next_hop": int(row.target_next_hop),
                    "triggered": triggered,
                    "corrected": int((int(row.predicted_next_hop) != int(row.target_next_hop)) and (choice == int(row.target_next_hop))),
                    "new_error": int((int(row.predicted_next_hop) == int(row.target_next_hop)) and (choice != int(row.target_next_hop))),
                    "model_margin": float(row.model_margin),
                    "oracle_action_gap": float(row.oracle_action_gap),
                    "hard_near_tie_intersection_case": bool(row.hard_near_tie_intersection_case),
                    "baseline_error_hard_near_tie_case": bool(row.baseline_error_hard_near_tie_case),
                }
            )

        oracle_costs: list[float] = []
        tie_costs: list[float] = []
        deadline_counts: list[float] = []
        solved = 0
        total_steps = 0
        correct_steps = 0
        oracle_priority = 0.0
        tie_priority = 0.0
        for episode in dataset.episodes:
            _oracle_solved, _oracle_steps, _oracle_correct, oracle_cost, _oracle_violations, oracle_delivered = _rollout_episode_oracle(
                episode,
                config=hidden_cfg,
            )
            tie_solved, steps, correct, tie_cost, tie_violations, tie_delivered = _rollout_episode_with_tiebreak(
                episode,
                model,
                config=hidden_cfg,
                device=device,
                top_k=args.top_k,
                gate_margin=args.gate_margin,
            )
            oracle_costs.append(float(oracle_cost))
            tie_costs.append(float(tie_cost))
            deadline_counts.append(float(tie_violations))
            solved += int(tie_solved)
            total_steps += int(steps)
            correct_steps += int(correct)
            oracle_priority += float(oracle_delivered)
            tie_priority += float(tie_delivered)

        suite_decision_df = pd.DataFrame([row for row in decision_rows if row["suite"] == suite_config.name])
        hard_df = suite_decision_df.loc[suite_decision_df["hard_near_tie_intersection_case"]]
        error_df = suite_decision_df.loc[suite_decision_df["baseline_error_hard_near_tie_case"]]
        count = max(len(dataset.episodes), 1)
        regrets = torch.tensor([tie - oracle for tie, oracle in zip(tie_costs, oracle_costs, strict=True)], dtype=torch.float32)
        deadline_tensor = torch.tensor(deadline_counts, dtype=torch.float32)
        suite_summary_rows.append(
            {
                "suite": suite_config.name,
                "decisions": len(base_decision_df),
                "search_trigger_rate": trigger_count / max(len(base_decision_df), 1),
                "corrected_errors": corrected,
                "new_errors": new_errors,
                "net_corrected": corrected - new_errors,
                "hard_near_tie_disagreement": float((hard_df["base_choice"] != hard_df["tiebreak_choice"]).mean()) if len(hard_df) else 0.0,
                "hard_near_tie_correction_rate": float(hard_df["corrected"].mean()) if len(hard_df) else 0.0,
                "hard_near_tie_new_error_rate": float(hard_df["new_error"].mean()) if len(hard_df) else 0.0,
                "baseline_error_recovery": float(error_df["corrected"].mean()) if len(error_df) else 0.0,
                "solved_rate": solved / count,
                "rollout_next_hop_accuracy": correct_steps / max(total_steps, 1),
                "average_regret": float(regrets.mean().item()) if len(regrets) else 0.0,
                "p95_regret": float(torch.quantile(regrets, 0.95).item()) if len(regrets) else 0.0,
                "worst_regret": float(regrets.max().item()) if len(regrets) else 0.0,
                "deadline_miss_rate": float((deadline_tensor > 0).float().mean().item()) if len(deadline_tensor) else 0.0,
                "average_deadline_violations": float(deadline_tensor.mean().item()) if len(deadline_tensor) else 0.0,
                "priority_delivered_regret": (oracle_priority - tie_priority) / count,
                "average_oracle_cost": sum(oracle_costs) / count,
                "average_model_cost": sum(tie_costs) / count,
            }
        )

    summary_df = pd.DataFrame(suite_summary_rows)
    decisions_df = pd.DataFrame(decision_rows)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    decisions_df.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "top_k": args.top_k,
                "gate_margin": args.gate_margin,
                "summary": suite_summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
