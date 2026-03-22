#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.rollout import evaluate_rollouts
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import load_experiment_config
from gnn3.train.trainer import _resolve_device, _rollout_metrics_to_dict, evaluate_decision_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    device = _resolve_device(config.train.device)
    dataset = HiddenCorridorDecisionDataset(
        config=config.benchmark.hidden_corridor,
        num_episodes=config.benchmark.test_episodes,
        curriculum_levels=config.benchmark.curriculum_levels,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        collate_fn=collate_decisions,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])

    decision_metrics = evaluate_decision_dataset(
        model,
        loader,
        device=device,
        final_step_only=config.model.final_step_only_loss,
        value_weight=config.train.value_weight,
        route_weight=config.train.route_weight,
    )
    rollout_metrics = evaluate_rollouts(
        model,
        dataset.episodes[: config.train.rollout_eval_episodes],
        device=device,
        config=config.benchmark.hidden_corridor,
    )
    print(
        json.dumps(
            {
                "decision": decision_metrics,
                "rollout": _rollout_metrics_to_dict(rollout_metrics),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
