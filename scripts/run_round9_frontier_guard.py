#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.policy_analysis import collect_decision_prediction_rows, collect_episode_policy_rows
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--frontier-decisions-csv", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument("--base-selection-strategy", default="final")
    parser.add_argument("--candidate-selection-strategy", default="final")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round9_frontier_guard",
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


def _rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _slice_rows(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("overall", pd.Series([True] * len(frame), index=frame.index)),
        ("hard_feasible", frame["hard_feasible_case"]),
        ("hard_near_tie", frame["hard_near_tie_intersection_case"]),
        ("stable_near_tie", frame["stable_near_tie_case"]),
        ("high_headroom_near_tie", frame["high_headroom_near_tie_case"]),
        ("baseline_error_near_tie", frame["baseline_error_hard_near_tie_case"]),
        ("large_gap_control", frame["large_gap_hard_feasible_case"]),
    ]


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frontier_df = pd.read_csv(args.frontier_decisions_csv)
    keep_cols = [
        "suite",
        "episode_index",
        "decision_index",
        "hard_feasible_case",
        "hard_near_tie_intersection_case",
        "stable_near_tie_case",
        "high_headroom_near_tie_case",
        "baseline_error_hard_near_tie_case",
        "large_gap_hard_feasible_case",
        "target_match",
        "strictly_suboptimal",
    ]
    frontier_df = frontier_df[keep_cols].copy()

    base_model, base_device = _load_model(args.base_config, args.base_checkpoint, device_override=args.device)
    candidate_model, candidate_device = _load_model(
        args.candidate_config,
        args.candidate_checkpoint,
        device_override=args.device,
    )

    summary_rows: list[dict[str, object]] = []
    decision_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        suite_name = suite_config.name
        suite_frontier = frontier_df.loc[frontier_df["suite"] == suite_name].copy()
        if suite_frontier.empty:
            continue

        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        records = list(dataset)
        base_df = pd.DataFrame(
            collect_decision_prediction_rows(
                base_model,
                records,
                device=base_device,
                suite=suite_name,
                selection_strategy=args.base_selection_strategy,
            )
        ).rename(
            columns={
                "predicted_next_hop": "base_predicted_next_hop",
                "target_match": "base_target_match",
                "predicted_continuation_gap": "base_continuation_gap",
                "predicted_on_time": "base_predicted_on_time",
            }
        )
        candidate_df = pd.DataFrame(
            collect_decision_prediction_rows(
                candidate_model,
                records,
                device=candidate_device,
                suite=suite_name,
                selection_strategy=args.candidate_selection_strategy,
            )
        ).rename(
            columns={
                "predicted_next_hop": "candidate_predicted_next_hop",
                "target_match": "candidate_target_match",
                "predicted_continuation_gap": "candidate_continuation_gap",
                "predicted_on_time": "candidate_predicted_on_time",
            }
        )
        merged = suite_frontier.merge(
            base_df[
                [
                    "suite",
                    "episode_index",
                    "decision_index",
                    "base_predicted_next_hop",
                    "base_target_match",
                    "base_continuation_gap",
                    "base_predicted_on_time",
                ]
            ],
            on=["suite", "episode_index", "decision_index"],
            how="inner",
        ).merge(
            candidate_df[
                [
                    "suite",
                    "episode_index",
                    "decision_index",
                    "candidate_predicted_next_hop",
                    "candidate_target_match",
                    "candidate_continuation_gap",
                    "candidate_predicted_on_time",
                ]
            ],
            on=["suite", "episode_index", "decision_index"],
            how="inner",
        )
        merged["action_agreement"] = merged["base_predicted_next_hop"] == merged["candidate_predicted_next_hop"]
        merged["correction"] = (~merged["base_target_match"]) & merged["candidate_target_match"]
        merged["new_error"] = merged["base_target_match"] & (~merged["candidate_target_match"])
        merged["anti_oracle_disagreement"] = (~merged["action_agreement"]) & (~merged["candidate_target_match"])
        merged["headroom_delta"] = merged["candidate_continuation_gap"] - merged["base_continuation_gap"]
        merged["on_time_delta"] = merged["candidate_predicted_on_time"].astype(int) - merged["base_predicted_on_time"].astype(int)
        decision_frames.append(merged)

        base_episode = pd.DataFrame(
            collect_episode_policy_rows(
                base_model,
                dataset.episodes,
                device=base_device,
                config=hidden_cfg,
                suite=suite_name,
                selection_strategy=args.base_selection_strategy,
            )
        ).rename(columns={"regret": "base_regret", "deadline_miss": "base_deadline_miss", "next_hop_accuracy": "base_next_hop_accuracy"})
        candidate_episode = pd.DataFrame(
            collect_episode_policy_rows(
                candidate_model,
                dataset.episodes,
                device=candidate_device,
                config=hidden_cfg,
                suite=suite_name,
                selection_strategy=args.candidate_selection_strategy,
            )
        ).rename(
            columns={
                "regret": "candidate_regret",
                "deadline_miss": "candidate_deadline_miss",
                "next_hop_accuracy": "candidate_next_hop_accuracy",
            }
        )
        merged_episode = base_episode.merge(
            candidate_episode[
                ["suite", "episode_index", "candidate_regret", "candidate_deadline_miss", "candidate_next_hop_accuracy"]
            ],
            on=["suite", "episode_index"],
            how="inner",
        )
        episode_frames.append(merged_episode)

        for slice_name, mask in _slice_rows(merged):
            frame = merged.loc[mask]
            summary_rows.append(
                {
                    "suite": suite_name,
                    "slice": slice_name,
                    "decisions": len(frame),
                    "disagreement": _rate(~frame["action_agreement"]),
                    "correction_rate": _rate(frame["correction"]),
                    "new_error_rate": _rate(frame["new_error"]),
                    "anti_oracle_rate": _rate(frame["anti_oracle_disagreement"]),
                    "base_target_match": _rate(frame["base_target_match"]),
                    "candidate_target_match": _rate(frame["candidate_target_match"]),
                    "headroom_delta": float(frame["headroom_delta"].mean()) if len(frame) else 0.0,
                    "on_time_delta": float(frame["on_time_delta"].mean()) if len(frame) else 0.0,
                }
            )

    if not summary_rows:
        raise SystemExit("No overlapping frontier rows were found for the requested suites.")

    summary_df = pd.DataFrame(summary_rows)
    decisions_df = pd.concat(decision_frames, ignore_index=True)
    episodes_df = pd.concat(episode_frames, ignore_index=True)

    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    decisions_df.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    episodes_df.to_csv(output_prefix.with_name(output_prefix.name + "_episodes.csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "summary": summary_df.to_dict(orient="records"),
                "overall": {
                    "hard_near_tie_disagreement": _rate(
                        ~decisions_df.loc[decisions_df["hard_near_tie_intersection_case"], "action_agreement"]
                    ),
                    "large_gap_control_target_match": _rate(
                        decisions_df.loc[decisions_df["large_gap_hard_feasible_case"], "candidate_target_match"]
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
