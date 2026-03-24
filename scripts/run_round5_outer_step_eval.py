#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.rollout import evaluate_rollouts
from gnn3.eval.step_policy import STEP_STRATEGIES, select_step_scores
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device, _rollout_metrics_to_dict

ROOT = Path("artifacts/experiments")
PLOTS = Path("reports/plots")
REPORT = Path("reports")
EVAL_BATCH_SIZE = 16

DEFAULT_EXPERIMENTS = [
    "e3_memory_hubs_rsm_round4_multiheavy_seed311",
    "e3_memory_hubs_rsm_round4_multiheavy_seed312",
    "e3_memory_hubs_rsm_round4_multiheavy_seed313",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Experiment names to evaluate.",
    )
    parser.add_argument(
        "--output-stem",
        default="round5_multiheavy_outer_step_vs_final",
        help="Output stem under reports/plots.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STEP_STRATEGIES),
        choices=STEP_STRATEGIES,
        help="Step-selection strategies to evaluate.",
    )
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        help="Override the number of rollout episodes per experiment for a faster scout.",
    )
    return parser.parse_args()


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_decision_accuracy(
    model: PacketMambaModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    strategy: str,
) -> float:
    correct = 0.0
    total = 0
    was_training = model.training
    model.eval()
    for batch in loader:
        moved = _move_batch(batch, device)
        output = model(moved)
        scores = select_step_scores(output, moved["candidate_mask"], strategy=strategy)
        pred = scores.argmax(dim=-1).cpu()
        target = batch["target_next_hop"]
        correct += float((pred == target).float().sum().item())
        total += int(target.numel())
    if was_training:
        model.train()
    return correct / max(total, 1)


def maybe_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plot_df = frame[frame["seed"] != "mean"].copy()
    means = plot_df.groupby("strategy", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].bar(means["strategy"], means["decision_next_hop_accuracy"], color="#1f77b4")
    axes[0].set_title("Decision Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(means["strategy"], means["next_hop_accuracy"], color="#2ca02c")
    axes[1].set_title("Rollout Next-Hop Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(means["strategy"], means["average_regret"], color="#d62728")
    axes[2].set_title("Average Regret")
    axes[2].tick_params(axis="x", rotation=30)

    axes[3].bar(means["strategy"], means["deadline_miss_rate"], color="#9467bd")
    axes[3].set_title("Deadline Miss Rate")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for experiment_name in args.experiments:
        config = load_experiment_config(Path("configs/experiments") / f"{experiment_name}.yaml")
        device = _resolve_device(config.train.device)
        checkpoint = torch.load(ROOT / experiment_name / "checkpoints" / "best.pt", map_location=device)
        model = PacketMambaModel(config.model).to(device)
        model.load_state_dict(checkpoint["model"])
        test_hidden_cfg = hidden_corridor_config_for_split(config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=test_hidden_cfg,
            num_episodes=config.benchmark.test_episodes,
            curriculum_levels=config.benchmark.curriculum_levels,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(config.train.eval_batch_size, EVAL_BATCH_SIZE),
            shuffle=False,
            collate_fn=collate_decisions,
        )
        rollout_episodes = args.rollout_episodes or config.train.rollout_eval_episodes
        for strategy in args.strategies:
            start = time.perf_counter()
            decision_acc = evaluate_decision_accuracy(model, loader, device=device, strategy=strategy)
            rollout = evaluate_rollouts(
                model,
                dataset.episodes[: rollout_episodes],
                device=device,
                config=test_hidden_cfg,
                selection_strategy=strategy,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "seed": config.seed,
                    "experiment": experiment_name,
                    "strategy": strategy,
                    "decision_next_hop_accuracy": decision_acc,
                    **_rollout_metrics_to_dict(rollout),
                    "gpu_hours": elapsed / 3600.0 if device.type == "cuda" else 0.0,
                }
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).sort_values(["strategy", "seed"]).reset_index(drop=True)
    means = frame.groupby("strategy", as_index=False).mean(numeric_only=True)
    means["seed"] = "mean"
    means["experiment"] = means["strategy"] + "-mean"
    output = pd.concat([frame, means[frame.columns]], ignore_index=True)
    output.to_csv(PLOTS / f"{args.output_stem}.csv", index=False)
    maybe_plot(output, PLOTS / f"{args.output_stem}.png")
    (PLOTS / f"{args.output_stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
