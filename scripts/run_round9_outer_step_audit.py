#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.policy_analysis import collect_decision_prediction_rows, collect_episode_policy_rows
from gnn3.eval.step_policy import STEP_STRATEGIES, select_step_index
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--frontier-decisions-csv", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STEP_STRATEGIES),
        choices=STEP_STRATEGIES,
        help="Step-selection strategies to evaluate.",
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round9_outer_step_headroom",
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


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _average_selected_step(
    model: PacketMambaModel,
    dataset: HiddenCorridorDecisionDataset,
    *,
    device: torch.device,
    strategy: str,
    batch_size: int = 64,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_decisions)
    steps: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    for batch in loader:
        moved = _move_batch(batch, device)
        output = model(moved)
        step_index = select_step_index(output, moved["candidate_mask"], strategy=strategy)
        steps.append(step_index.detach().cpu().float() + 1.0)
    if was_training:
        model.train()
    if not steps:
        return 0.0
    return float(torch.cat(steps).mean().item())


def _slice_summary(frame: pd.DataFrame, strategy: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slice_name, mask in [
        ("overall", pd.Series([True] * len(frame), index=frame.index)),
        ("hard_near_tie", frame["hard_near_tie_intersection_case"]),
        ("stable_near_tie", frame["stable_near_tie_case"]),
        ("high_headroom_near_tie", frame["high_headroom_near_tie_case"]),
        ("baseline_error_near_tie", frame["baseline_error_hard_near_tie_case"]),
        ("large_gap_control", frame["large_gap_hard_feasible_case"]),
    ]:
        subset = frame.loc[mask]
        rows.append(
            {
                "strategy": strategy,
                "slice": slice_name,
                "decisions": len(subset),
                "target_match": float(subset["target_match"].mean()) if len(subset) else 0.0,
                "error_rate": float((~subset["target_match"]).mean()) if len(subset) else 0.0,
                "predicted_on_time": float(subset["predicted_on_time"].mean()) if len(subset) else 0.0,
                "mean_continuation_gap": float(subset["predicted_continuation_gap"].mean()) if len(subset) else 0.0,
            }
        )
    return rows


def _write_outputs(
    output_prefix: Path,
    decision_rows: list[dict[str, object]],
    episode_rows: list[dict[str, object]],
) -> None:
    decision_summary = pd.DataFrame(decision_rows)
    episode_summary = pd.DataFrame(episode_rows)
    decision_summary.to_csv(output_prefix.with_name(output_prefix.name + "_decisions.csv"), index=False)
    episode_summary.to_csv(output_prefix.with_name(output_prefix.name + "_episodes.csv"), index=False)
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "decision_summary": decision_summary.to_dict(orient="records"),
                "episode_summary": episode_summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frontier_df = pd.read_csv(args.frontier_decisions_csv)
    keep_cols = [
        "suite",
        "episode_index",
        "decision_index",
        "hard_near_tie_intersection_case",
        "stable_near_tie_case",
        "high_headroom_near_tie_case",
        "baseline_error_hard_near_tie_case",
        "large_gap_hard_feasible_case",
    ]
    frontier_df = frontier_df[keep_cols].copy()

    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)
    decision_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []

    for suite_config_path in args.suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        for strategy in args.strategies:
            average_step = _average_selected_step(model, dataset, device=device, strategy=strategy)
            decision_df = pd.DataFrame(
                collect_decision_prediction_rows(
                    model,
                    list(dataset),
                    device=device,
                    suite=suite_config.name,
                    selection_strategy=strategy,
                )
            ).merge(
                frontier_df,
                on=["suite", "episode_index", "decision_index"],
                how="left",
            )
            episode_df = pd.DataFrame(
                collect_episode_policy_rows(
                    model,
                    dataset.episodes,
                    device=device,
                    config=hidden_cfg,
                    suite=suite_config.name,
                    selection_strategy=strategy,
                )
            )
            for row in _slice_summary(decision_df, strategy):
                row["suite"] = suite_config.name
                row["average_selected_step"] = average_step
                decision_rows.append(row)
            episode_rows.append(
                {
                    "suite": suite_config.name,
                    "strategy": strategy,
                    "episodes": len(episode_df),
                    "average_selected_step": average_step,
                    "next_hop_accuracy": float(episode_df["next_hop_accuracy"].mean()),
                    "average_regret": float(episode_df["regret"].mean()),
                    "p95_regret": float(episode_df["regret"].quantile(0.95)),
                    "deadline_miss_rate": float(episode_df["deadline_miss"].mean()),
                    "solved_rate": float(episode_df["solved"].mean()),
                }
            )
            _write_outputs(output_prefix, decision_rows, episode_rows)

    episode_summary = pd.DataFrame(episode_rows)
    print(episode_summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
