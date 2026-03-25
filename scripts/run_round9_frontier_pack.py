#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gnn3.data.hidden_corridor import DecisionRecord, HiddenCorridorDecisionDataset
from gnn3.eval.hard_feasible import annotate_hard_feasible
from gnn3.eval.policy_analysis import collect_decision_prediction_rows, collect_episode_policy_rows
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round9_frontier_pack",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    parser.add_argument("--perturb-samples", type=int, default=64)
    parser.add_argument("--perturb-sigma", type=float, default=0.02)
    parser.add_argument("--reuse-thresholds-json", help="Optional prior frontier JSON to reuse thresholds.")
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, *, device_override: str | None = None) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def _perturb_flip_rate(record: DecisionRecord, *, samples: int, sigma: float) -> float:
    valid = record.candidate_mask.astype(bool)
    if int(valid.sum()) < 2:
        return 0.0
    costs = record.candidate_cost_to_go[valid].astype(np.float64)
    if not np.isfinite(costs).any():
        return 0.0
    best_index = int(np.argmin(costs))
    seed = (int(record.episode_index) + 1) * 10007 + (int(record.packet_index) + 1) * 97 + int(record.current_node)
    rng = np.random.default_rng(seed)
    flips = 0
    for _ in range(samples):
        noise = rng.normal(loc=0.0, scale=sigma, size=costs.shape[0])
        perturbed = costs * np.maximum(1.0 + noise, 1e-4)
        flips += int(int(np.argmin(perturbed)) != best_index)
    return float(flips / max(samples, 1))


def _slice_specs(decision_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("hard_feasible", decision_df["hard_feasible_case"]),
        ("oracle_near_tie", decision_df["oracle_near_tie_case"]),
        ("model_near_tie", decision_df["model_near_tie_case"]),
        ("hard_near_tie_intersection", decision_df["hard_near_tie_intersection_case"]),
        ("stable_near_tie", decision_df["stable_near_tie_case"]),
        ("high_headroom_near_tie", decision_df["high_headroom_near_tie_case"]),
        ("baseline_error_intersection", decision_df["baseline_error_hard_near_tie_case"]),
        ("large_gap_control", decision_df["large_gap_hard_feasible_case"]),
    ]


def _slice_summary(decision_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, mask in _slice_specs(decision_df):
        frame = decision_df.loc[mask]
        rows.append(
            {
                "slice": name,
                "decisions": len(frame),
                "episodes": int(frame[["suite", "episode_index"]].drop_duplicates().shape[0]),
                "baseline_error_rate": float(frame["baseline_error"].mean()) if len(frame) else 0.0,
                "mean_headroom": float(frame["predicted_continuation_gap"].mean()) if len(frame) else 0.0,
                "p95_headroom": float(frame["predicted_continuation_gap"].quantile(0.95)) if len(frame) else 0.0,
                "target_match": float(frame["target_match"].mean()) if len(frame) else 0.0,
                "predicted_on_time": float(frame["predicted_on_time"].mean()) if len(frame) else 0.0,
                "effective_tie_rate": float(frame["effective_tie_under_perturbation"].mean()) if len(frame) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot_pack(decision_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    hard_df = decision_df.loc[decision_df["hard_feasible_case"]]
    near_tie_df = decision_df.loc[decision_df["hard_near_tie_intersection_case"]]

    axes[0, 0].hist(hard_df["oracle_action_gap"], bins=30, color="#1f77b4", alpha=0.9)
    axes[0, 0].set_title("Hard Slice Oracle Top-2 Gap")
    axes[0, 0].set_xlabel("Gap")

    axes[0, 1].hist(near_tie_df["predicted_continuation_gap"], bins=25, color="#ff7f0e", alpha=0.9)
    axes[0, 1].set_title("Hard Near-Tie Headroom")
    axes[0, 1].set_xlabel("Continuation Gap")

    axes[1, 0].bar(summary_df["slice"], summary_df["baseline_error_rate"], color="#d62728")
    axes[1, 0].set_title("Baseline Error Rate by Slice")
    axes[1, 0].tick_params(axis="x", rotation=25)

    axes[1, 1].bar(summary_df["slice"], summary_df["mean_headroom"], color="#2ca02c")
    axes[1, 1].set_title("Mean Headroom by Slice")
    axes[1, 1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)

    decision_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    suite_meta: list[dict[str, object]] = []
    threshold_payload = None
    if args.reuse_thresholds_json:
        threshold_payload = json.loads(Path(args.reuse_thresholds_json).read_text(encoding="utf-8")).get("thresholds")

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
        decision_df = pd.DataFrame(
            collect_decision_prediction_rows(
                model,
                records,
                device=device,
                suite=suite_config.name,
            )
        )
        decision_df["row_in_suite"] = np.arange(len(decision_df), dtype=np.int64)
        decision_df["perturb_flip_rate"] = [
            _perturb_flip_rate(
                records[int(row.row_in_suite)],
                samples=args.perturb_samples,
                sigma=args.perturb_sigma,
            )
            for row in decision_df.itertuples(index=False)
        ]
        decision_frames.append(decision_df)
        episode_frames.append(episode_df)
        suite_meta.append(
            {
                "suite": suite_config.name,
                "config_path": suite_config_path,
                "manifest_hash": dataset.manifest()["manifest_hash"],
                "episodes": len(dataset.episodes),
                "decisions": len(dataset),
            }
        )

    decision_df = pd.concat(decision_frames, ignore_index=True)
    episode_df = pd.concat(episode_frames, ignore_index=True)
    thresholds = None
    if threshold_payload is not None:
        from gnn3.eval.hard_feasible import HardFeasibleThresholds

        thresholds = HardFeasibleThresholds(
            gap_threshold=float(threshold_payload["large_gap_threshold"]),
            gap_ratio_threshold=float(threshold_payload["large_gap_ratio_threshold"]),
            near_tie_gap_threshold=float(threshold_payload["near_tie_gap_threshold"]),
            model_margin_threshold=float(threshold_payload["model_margin_threshold"]),
        )
    decision_df, thresholds = annotate_hard_feasible(decision_df, episode_df, thresholds=thresholds)
    decision_df["effective_tie_under_perturbation"] = decision_df["perturb_flip_rate"] >= 0.25
    baseline_error_df = decision_df.loc[decision_df["baseline_error_hard_near_tie_case"]]
    high_headroom_threshold = float(baseline_error_df["predicted_continuation_gap"].median()) if len(baseline_error_df) else 0.0
    decision_df["stable_near_tie_case"] = (
        decision_df["hard_near_tie_intersection_case"] & (~decision_df["effective_tie_under_perturbation"])
    )
    decision_df["high_headroom_near_tie_case"] = (
        decision_df["hard_near_tie_intersection_case"] & (decision_df["predicted_continuation_gap"] >= high_headroom_threshold)
    )
    decision_df["decodable_near_tie_case"] = (
        decision_df["hard_near_tie_intersection_case"] & (decision_df["model_margin"] > thresholds.model_margin_threshold)
    )
    decision_df["weakly_decodable_near_tie_case"] = (
        decision_df["hard_near_tie_intersection_case"] & (~decision_df["decodable_near_tie_case"])
    )

    summary_df = _slice_summary(decision_df)
    suite_summary = (
        decision_df.groupby(["suite"], as_index=False)
        .agg(
            decisions=("decision_index", "count"),
            hard_near_tie_decisions=("hard_near_tie_intersection_case", "sum"),
            baseline_error_decisions=("baseline_error_hard_near_tie_case", "sum"),
            stable_near_tie_decisions=("stable_near_tie_case", "sum"),
            high_headroom_decisions=("high_headroom_near_tie_case", "sum"),
            large_gap_control_decisions=("large_gap_hard_feasible_case", "sum"),
        )
        .sort_values("suite")
    )

    prefix = output_prefix.with_name(output_prefix.name)
    decisions_csv = prefix.with_name(prefix.name + "_decisions.csv")
    summary_csv = prefix.with_name(prefix.name + "_summary.csv")
    suite_csv = prefix.with_name(prefix.name + "_suite_summary.csv")
    plot_png = prefix.with_name(prefix.name + "_summary.png")
    json_path = prefix.with_suffix(".json")

    decision_df.to_csv(decisions_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    suite_summary.to_csv(suite_csv, index=False)
    for name, mask in _slice_specs(decision_df):
        decision_df.loc[mask].to_csv(prefix.with_name(prefix.name + f"_{name}_manifest.csv"), index=False)
    decision_df.loc[decision_df["decodable_near_tie_case"]].to_csv(
        prefix.with_name(prefix.name + "_decodable_near_tie_manifest.csv"),
        index=False,
    )
    decision_df.loc[decision_df["weakly_decodable_near_tie_case"]].to_csv(
        prefix.with_name(prefix.name + "_weakly_decodable_near_tie_manifest.csv"),
        index=False,
    )
    _plot_pack(decision_df, summary_df, plot_png)

    payload = {
        "suite_meta": suite_meta,
        "thresholds": {
            "large_gap_threshold": thresholds.gap_threshold,
            "large_gap_ratio_threshold": thresholds.gap_ratio_threshold,
            "near_tie_gap_threshold": thresholds.near_tie_gap_threshold,
            "model_margin_threshold": thresholds.model_margin_threshold,
            "high_headroom_threshold": high_headroom_threshold,
        },
        "slice_summary": summary_df.to_dict(orient="records"),
        "suite_summary": suite_summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
