#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.policy_analysis import collect_decision_prediction_rows
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
    parser.add_argument(
        "--audit-decisions-csv",
        default="reports/plots/round7_hard_feasible_action_gap_decisions.csv",
    )
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round7_hard_slice_compare",
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


def _collect_predictions(
    model: PacketMambaModel,
    samples,
    *,
    device: torch.device,
    suite: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        collect_decision_prediction_rows(
            model,
            samples,
            device=device,
            suite=suite,
        )
    )


def _indexed_samples(dataset: HiddenCorridorDecisionDataset) -> list[tuple[int, int, object]]:
    indexed: list[tuple[int, int, object]] = []
    current_episode = None
    current_decision = -1
    for sample in dataset:
        episode_index = int(sample.episode_index)
        if episode_index != current_episode:
            current_episode = episode_index
            current_decision = 0
        else:
            current_decision += 1
        indexed.append((episode_index, current_decision, sample))
    return indexed


def _rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    audit_df = pd.read_csv(args.audit_decisions_csv)
    keep_cols = [
        "suite",
        "episode_index",
        "decision_index",
        "hard_feasible_case",
        "large_gap_hard_feasible_case",
        "gap_bucket",
        "slack_band",
        "packet_band",
        "load_band",
        "depth_band",
        "hard_condition_count",
        "oracle_action_gap",
        "oracle_action_gap_ratio",
        "target_match",
        "strictly_suboptimal",
    ]
    audit_df = audit_df[keep_cols].copy()

    base_model, base_device = _load_model(args.base_config, args.base_checkpoint, device_override=args.device)
    candidate_model, candidate_device = _load_model(
        args.candidate_config,
        args.candidate_checkpoint,
        device_override=args.device,
    )

    suite_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        suite_name = suite_config.name
        suite_audit = audit_df.loc[audit_df["suite"] == suite_name].copy()
        if suite_audit.empty:
            continue

        wanted = {
            (int(row.episode_index), int(row.decision_index))
            for row in suite_audit.itertuples(index=False)
        }
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        filtered_samples = [
            sample
            for episode_index, decision_index, sample in _indexed_samples(dataset)
            if (episode_index, decision_index) in wanted
        ]

        base_df = _collect_predictions(base_model, filtered_samples, device=base_device, suite=suite_name).rename(
            columns={
                "predicted_next_hop": "base_predicted_next_hop",
                "target_match": "base_target_match",
            }
        )
        candidate_df = _collect_predictions(
            candidate_model,
            filtered_samples,
            device=candidate_device,
            suite=suite_name,
        ).rename(
            columns={
                "predicted_next_hop": "candidate_predicted_next_hop",
                "target_match": "candidate_target_match",
            }
        )

        merged = suite_audit.merge(
            base_df[["suite", "episode_index", "decision_index", "base_predicted_next_hop", "base_target_match"]],
            on=["suite", "episode_index", "decision_index"],
            how="inner",
        ).merge(
            candidate_df[
                ["suite", "episode_index", "decision_index", "candidate_predicted_next_hop", "candidate_target_match"]
            ],
            on=["suite", "episode_index", "decision_index"],
            how="inner",
        )
        merged["action_agreement"] = merged["base_predicted_next_hop"] == merged["candidate_predicted_next_hop"]
        merged["hard_near_tie_case"] = merged["hard_feasible_case"] & (merged["gap_bucket"] == "near_tie")
        suite_frames.append(merged)

        summary_rows.append(
            {
                "suite": suite_name,
                "decisions": len(merged),
                "overall_disagreement": _rate(~merged["action_agreement"]),
                "hard_feasible_disagreement": _rate(~merged.loc[merged["hard_feasible_case"], "action_agreement"]),
                "large_gap_hard_feasible_disagreement": _rate(
                    ~merged.loc[merged["large_gap_hard_feasible_case"], "action_agreement"]
                ),
                "hard_near_tie_disagreement": _rate(~merged.loc[merged["hard_near_tie_case"], "action_agreement"]),
                "candidate_target_match": _rate(merged["candidate_target_match"]),
                "base_target_match": _rate(merged["base_target_match"]),
            }
        )

    if not suite_frames:
        raise SystemExit("No overlapping audited hard-slice decisions were found for the requested suites.")

    merged_df = pd.concat(suite_frames, ignore_index=True)
    decisions_csv = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    json_path = output_prefix.with_suffix(".json")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    merged_df.to_csv(decisions_csv, index=False)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "overall": {
                    "decisions": len(merged_df),
                    "overall_disagreement": _rate(~merged_df["action_agreement"]),
                    "hard_feasible_disagreement": _rate(~merged_df.loc[merged_df["hard_feasible_case"], "action_agreement"]),
                    "large_gap_hard_feasible_disagreement": _rate(
                        ~merged_df.loc[merged_df["large_gap_hard_feasible_case"], "action_agreement"]
                    ),
                    "hard_near_tie_disagreement": _rate(
                        ~merged_df.loc[merged_df["hard_near_tie_case"], "action_agreement"]
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
