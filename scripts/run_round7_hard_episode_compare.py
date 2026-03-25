#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.policy_analysis import collect_episode_policy_rows
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
        default="reports/plots/round7_hard_episode_compare",
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


def _quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else 0.0


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    audit_df = pd.read_csv(args.audit_decisions_csv)
    episode_flags = (
        audit_df.groupby(["suite", "episode_index"], as_index=False)[["hard_feasible_case", "large_gap_hard_feasible_case"]]
        .max()
        .rename(
            columns={
                "hard_feasible_case": "episode_has_hard_feasible_case",
                "large_gap_hard_feasible_case": "episode_has_large_gap_hard_feasible_case",
            }
        )
    )

    base_model, base_device = _load_model(args.base_config, args.base_checkpoint, device_override=args.device)
    candidate_model, candidate_device = _load_model(
        args.candidate_config,
        args.candidate_checkpoint,
        device_override=args.device,
    )

    summary_rows: list[dict[str, object]] = []
    episode_frames: list[pd.DataFrame] = []

    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        suite_name = suite_config.name
        suite_flags = episode_flags.loc[episode_flags["suite"] == suite_name].copy()
        if suite_flags.empty:
            continue

        hard_episode_ids = set(suite_flags.loc[suite_flags["episode_has_hard_feasible_case"], "episode_index"].astype(int).tolist())
        if not hard_episode_ids:
            continue

        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        filtered_episodes = [
            episode
            for episode_index, episode in enumerate(dataset.episodes)
            if episode_index in hard_episode_ids
        ]

        base_episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                base_model,
                filtered_episodes,
                device=base_device,
                config=hidden_cfg,
                suite=suite_name,
            )
        ).rename(
            columns={
                "regret": "base_regret",
                "deadline_miss": "base_deadline_miss",
                "next_hop_accuracy": "base_next_hop_accuracy",
                "solved": "base_solved",
            }
        )
        candidate_episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                candidate_model,
                filtered_episodes,
                device=candidate_device,
                config=hidden_cfg,
                suite=suite_name,
            )
        ).rename(
            columns={
                "regret": "candidate_regret",
                "deadline_miss": "candidate_deadline_miss",
                "next_hop_accuracy": "candidate_next_hop_accuracy",
                "solved": "candidate_solved",
            }
        )

        merged = base_episode_df.merge(
            candidate_episode_df[
                [
                    "suite",
                    "episode_index",
                    "candidate_regret",
                    "candidate_deadline_miss",
                    "candidate_next_hop_accuracy",
                    "candidate_solved",
                ]
            ],
            on=["suite", "episode_index"],
            how="inner",
        ).merge(
            suite_flags,
            on=["suite", "episode_index"],
            how="left",
        )
        episode_frames.append(merged)

        hard_df = merged.loc[merged["episode_has_hard_feasible_case"]].copy()
        large_df = merged.loc[merged["episode_has_large_gap_hard_feasible_case"]].copy()
        summary_rows.append(
            {
                "suite": suite_name,
                "hard_episode_count": len(hard_df),
                "large_gap_episode_count": len(large_df),
                "base_hard_regret": float(hard_df["base_regret"].mean()),
                "candidate_hard_regret": float(hard_df["candidate_regret"].mean()),
                "base_hard_p95_regret": _quantile(hard_df["base_regret"], 0.95),
                "candidate_hard_p95_regret": _quantile(hard_df["candidate_regret"], 0.95),
                "base_hard_miss": _rate(hard_df["base_deadline_miss"]),
                "candidate_hard_miss": _rate(hard_df["candidate_deadline_miss"]),
                "base_large_gap_regret": float(large_df["base_regret"].mean()) if len(large_df) else 0.0,
                "candidate_large_gap_regret": float(large_df["candidate_regret"].mean()) if len(large_df) else 0.0,
                "base_large_gap_miss": _rate(large_df["base_deadline_miss"]),
                "candidate_large_gap_miss": _rate(large_df["candidate_deadline_miss"]),
            }
        )

    if not episode_frames:
        raise SystemExit("No audited hard-feasible episodes were found for the requested suites.")

    merged_df = pd.concat(episode_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    episodes_csv = output_prefix.with_name(output_prefix.name + "_episodes.csv")
    json_path = output_prefix.with_suffix(".json")
    summary_df.to_csv(summary_csv, index=False)
    merged_df.to_csv(episodes_csv, index=False)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "overall": {
                    "hard_episode_count": int(merged_df["episode_has_hard_feasible_case"].sum()),
                    "large_gap_episode_count": int(merged_df["episode_has_large_gap_hard_feasible_case"].sum()),
                    "base_hard_regret": float(
                        merged_df.loc[merged_df["episode_has_hard_feasible_case"], "base_regret"].mean()
                    ),
                    "candidate_hard_regret": float(
                        merged_df.loc[merged_df["episode_has_hard_feasible_case"], "candidate_regret"].mean()
                    ),
                    "base_hard_miss": _rate(
                        merged_df.loc[merged_df["episode_has_hard_feasible_case"], "base_deadline_miss"]
                    ),
                    "candidate_hard_miss": _rate(
                        merged_df.loc[merged_df["episode_has_hard_feasible_case"], "candidate_deadline_miss"]
                    ),
                    "base_large_gap_regret": float(
                        merged_df.loc[merged_df["episode_has_large_gap_hard_feasible_case"], "base_regret"].mean()
                    ),
                    "candidate_large_gap_regret": float(
                        merged_df.loc[merged_df["episode_has_large_gap_hard_feasible_case"], "candidate_regret"].mean()
                    ),
                    "base_large_gap_miss": _rate(
                        merged_df.loc[merged_df["episode_has_large_gap_hard_feasible_case"], "base_deadline_miss"]
                    ),
                    "candidate_large_gap_miss": _rate(
                        merged_df.loc[merged_df["episode_has_large_gap_hard_feasible_case"], "candidate_deadline_miss"]
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
