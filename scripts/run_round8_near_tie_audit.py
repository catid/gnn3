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
        default="reports/plots/round8_near_tie_headroom",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    parser.add_argument("--perturb-samples", type=int, default=64)
    parser.add_argument("--perturb-sigma", type=float, default=0.02)
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


def _slice_frame(decision_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("hard_feasible", decision_df["hard_feasible_case"]),
        ("oracle_near_tie", decision_df["oracle_near_tie_case"]),
        ("model_near_tie", decision_df["model_near_tie_case"]),
        ("hard_near_tie_intersection", decision_df["hard_near_tie_intersection_case"]),
        ("baseline_error_intersection", decision_df["baseline_error_hard_near_tie_case"]),
        ("large_gap_control", decision_df["large_gap_hard_feasible_case"]),
    ]
    for name, mask in specs:
        frame = decision_df.loc[mask]
        rows.append(
            {
                "slice": name,
                "decisions": len(frame),
                "episodes": int(frame[["suite", "episode_index"]].drop_duplicates().shape[0]),
                "baseline_error_rate": float(frame["baseline_error"].mean()) if len(frame) else 0.0,
                "mean_regret_headroom": float(frame.loc[frame["baseline_error"], "predicted_continuation_gap"].mean())
                if len(frame.loc[frame["baseline_error"]])
                else 0.0,
                "p95_regret_headroom": float(frame.loc[frame["baseline_error"], "predicted_continuation_gap"].quantile(0.95))
                if len(frame.loc[frame["baseline_error"]])
                else 0.0,
                "deadline_rescue_rate": float(
                    (frame["baseline_error"] & frame["any_feasible_candidate"] & (~frame["predicted_on_time"])).mean()
                )
                if len(frame)
                else 0.0,
                "effective_tie_rate": float(frame["effective_tie_under_perturbation"].mean()) if len(frame) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot_headroom(decision_df: pd.DataFrame, slice_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    hard_df = decision_df.loc[decision_df["hard_feasible_case"]]
    axes[0, 0].hist(hard_df["oracle_action_gap"], bins=30, color="#1f77b4", alpha=0.9)
    axes[0, 0].set_title("Hard Slice Oracle Top-2 Gap")
    axes[0, 0].set_xlabel("Gap")

    axes[0, 1].hist(hard_df["model_margin"], bins=30, color="#ff7f0e", alpha=0.9)
    axes[0, 1].set_title("Hard Slice Model Margin")
    axes[0, 1].set_xlabel("Margin")

    plot_df = slice_df.copy()
    axes[1, 0].bar(plot_df["slice"], plot_df["baseline_error_rate"], color="#d62728")
    axes[1, 0].set_ylim(0.0, max(0.08, float(plot_df["baseline_error_rate"].max()) * 1.2 + 0.01))
    axes[1, 0].set_title("Baseline Error Rate by Slice")
    axes[1, 0].tick_params(axis="x", rotation=25)

    near_tie_errors = decision_df.loc[decision_df["baseline_error_hard_near_tie_case"]]
    if len(near_tie_errors):
        axes[1, 1].hist(near_tie_errors["perturb_flip_rate"], bins=20, color="#2ca02c", alpha=0.9)
    axes[1, 1].set_title("Perturbation Flip Rate on Near-Tie Errors")
    axes[1, 1].set_xlabel("Flip rate")

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
    decision_df, thresholds = annotate_hard_feasible(decision_df, episode_df)
    decision_df["effective_tie_under_perturbation"] = decision_df["perturb_flip_rate"] >= 0.25
    decision_df["local_regret_if_corrected"] = decision_df["predicted_continuation_gap"].where(
        decision_df["baseline_error"], 0.0
    )

    slice_df = _slice_frame(decision_df)
    suite_summary = (
        decision_df.groupby(["suite"], as_index=False)
        .agg(
            decisions=("decision_index", "count"),
            hard_near_tie_decisions=("hard_near_tie_intersection_case", "sum"),
            hard_near_tie_errors=("baseline_error_hard_near_tie_case", "sum"),
            mean_model_margin=("model_margin", "mean"),
            mean_oracle_gap=("oracle_action_gap", "mean"),
        )
        .sort_values("suite")
    )

    decisions_csv = output_prefix.with_name(output_prefix.name + "_decisions.csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    suite_csv = output_prefix.with_name(output_prefix.name + "_suite_summary.csv")
    hard_csv = output_prefix.with_name(output_prefix.name + "_hard_manifest.csv")
    oracle_tie_csv = output_prefix.with_name(output_prefix.name + "_oracle_near_tie_manifest.csv")
    model_tie_csv = output_prefix.with_name(output_prefix.name + "_model_near_tie_manifest.csv")
    intersection_csv = output_prefix.with_name(output_prefix.name + "_intersection_manifest.csv")
    baseline_error_csv = output_prefix.with_name(output_prefix.name + "_baseline_error_manifest.csv")
    large_gap_csv = output_prefix.with_name(output_prefix.name + "_large_gap_manifest.csv")
    plot_png = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")

    decision_df.to_csv(decisions_csv, index=False)
    slice_df.to_csv(summary_csv, index=False)
    suite_summary.to_csv(suite_csv, index=False)
    decision_df.loc[decision_df["hard_feasible_case"]].to_csv(hard_csv, index=False)
    decision_df.loc[decision_df["oracle_near_tie_case"]].to_csv(oracle_tie_csv, index=False)
    decision_df.loc[decision_df["model_near_tie_case"]].to_csv(model_tie_csv, index=False)
    decision_df.loc[decision_df["hard_near_tie_intersection_case"]].to_csv(intersection_csv, index=False)
    decision_df.loc[decision_df["baseline_error_hard_near_tie_case"]].to_csv(baseline_error_csv, index=False)
    decision_df.loc[decision_df["large_gap_hard_feasible_case"]].to_csv(large_gap_csv, index=False)
    _plot_headroom(decision_df, slice_df, plot_png)

    payload = {
        "suite_meta": suite_meta,
        "thresholds": {
            "large_gap_threshold": thresholds.gap_threshold,
            "large_gap_ratio_threshold": thresholds.gap_ratio_threshold,
            "near_tie_gap_threshold": thresholds.near_tie_gap_threshold,
            "model_margin_threshold": thresholds.model_margin_threshold,
        },
        "slice_summary": slice_df.to_dict(orient="records"),
        "hard_near_tie_error_count": int(decision_df["baseline_error_hard_near_tie_case"].sum()),
        "hard_near_tie_decision_count": int(decision_df["hard_near_tie_intersection_case"].sum()),
        "hard_near_tie_effective_tie_rate": float(
            decision_df.loc[decision_df["hard_near_tie_intersection_case"], "effective_tie_under_perturbation"].mean()
        )
        if decision_df["hard_near_tie_intersection_case"].any()
        else 0.0,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(slice_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
