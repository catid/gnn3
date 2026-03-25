#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.eval.hard_feasible import annotate_hard_feasible
from gnn3.eval.near_tie import build_candidate_feature_tensor, critic_targets, valid_candidate_mask
from gnn3.eval.policy_analysis import (
    collect_decision_prediction_rows,
    collect_episode_policy_rows,
    extract_decision_latents,
)
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-suite-configs", nargs="+", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="artifacts/round8_counterfactual_dataset",
        help="Prefix for PT/CSV/JSON outputs.",
    )
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, *, device_override: str | None = None) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def _build_dataset_payload(
    model: PacketMambaModel,
    dataset: HiddenCorridorDecisionDataset,
    *,
    device: torch.device,
    config,
    suite: str,
    split: str,
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    episode_df = pd.DataFrame(
        collect_episode_policy_rows(
            model,
            dataset.episodes,
            device=device,
            config=config,
            suite=suite,
        )
    )
    decision_df = pd.DataFrame(
        collect_decision_prediction_rows(
            model,
            list(dataset),
            device=device,
            suite=suite,
        )
    )
    decision_df, thresholds = annotate_hard_feasible(decision_df, episode_df)
    latents = extract_decision_latents(model, list(dataset), device=device)

    batch = collate_decisions(list(dataset))
    probe = latents["probe_features"]
    selection_scores = latents["selection_scores"]
    values = latents["values"]
    features = build_candidate_feature_tensor(batch, probe_features=probe, selection_scores=selection_scores, values=values)
    targets = critic_targets(batch)
    valid_mask = valid_candidate_mask(batch)
    topk_model = selection_scores.masked_fill(~valid_mask, -1e9).topk(k=min(top_k, selection_scores.size(1)), dim=-1).indices

    metadata_rows: list[dict[str, object]] = []
    feature_rows: list[torch.Tensor] = []
    cost_rows: list[torch.Tensor] = []
    miss_rows: list[torch.Tensor] = []
    tail_rows: list[torch.Tensor] = []
    regret_rows: list[torch.Tensor] = []
    mask_rows: list[bool] = []

    for decision_index, row in enumerate(decision_df.itertuples(index=False)):
        valid_candidates = valid_mask[decision_index].nonzero(as_tuple=False).squeeze(-1)
        if valid_candidates.numel() == 0:
            continue
        oracle_order = batch["candidate_cost_to_go"][decision_index, valid_candidates].argsort()
        oracle_rank = {int(valid_candidates[idx].item()): rank for rank, idx in enumerate(oracle_order.tolist())}
        model_rank = {
            int(candidate): rank
            for rank, candidate in enumerate(topk_model[decision_index].tolist())
            if bool(valid_mask[decision_index, candidate])
        }
        for candidate in valid_candidates.tolist():
            metadata_rows.append(
                {
                    "row_in_suite": len(metadata_rows),
                    "split": split,
                    "suite": suite,
                    "episode_index": int(row.episode_index),
                    "decision_index": int(row.decision_index),
                    "candidate_node": int(candidate),
                    "target_next_hop": int(row.target_next_hop),
                    "predicted_next_hop": int(row.predicted_next_hop),
                    "is_target": int(candidate == int(row.target_next_hop)),
                    "is_predicted": int(candidate == int(row.predicted_next_hop)),
                    "hard_feasible_case": bool(row.hard_feasible_case),
                    "oracle_near_tie_case": bool(row.oracle_near_tie_case),
                    "model_near_tie_case": bool(row.model_near_tie_case),
                    "hard_near_tie_intersection_case": bool(row.hard_near_tie_intersection_case),
                    "baseline_error_hard_near_tie_case": bool(row.baseline_error_hard_near_tie_case),
                    "large_gap_hard_feasible_case": bool(row.large_gap_hard_feasible_case),
                    "oracle_rank": int(oracle_rank.get(int(candidate), 99)),
                    "model_topk_rank": int(model_rank.get(int(candidate), 99)),
                    "model_margin": float(row.model_margin),
                    "oracle_action_gap": float(row.oracle_action_gap),
                    "near_tie_gap_threshold": float(thresholds.near_tie_gap_threshold),
                }
            )
            feature_rows.append(features[decision_index, candidate].detach().cpu())
            cost_rows.append(targets["cost"][decision_index, candidate].detach().cpu())
            miss_rows.append(targets["miss"][decision_index, candidate].detach().cpu())
            tail_rows.append(targets["tail"][decision_index, candidate].detach().cpu())
            regret_rows.append(targets["regret_delta"][decision_index, candidate].detach().cpu())
            mask_rows.append(bool(valid_mask[decision_index, candidate]))

    metadata_df = pd.DataFrame(metadata_rows)
    tensor_payload = {
        "features": torch.stack(feature_rows, dim=0) if feature_rows else torch.empty((0, 0), dtype=torch.float32),
        "cost": torch.stack(cost_rows, dim=0) if cost_rows else torch.empty((0,), dtype=torch.float32),
        "miss": torch.stack(miss_rows, dim=0) if miss_rows else torch.empty((0,), dtype=torch.float32),
        "tail": torch.stack(tail_rows, dim=0) if tail_rows else torch.empty((0,), dtype=torch.float32),
        "regret_delta": torch.stack(regret_rows, dim=0) if regret_rows else torch.empty((0,), dtype=torch.float32),
        "valid": torch.as_tensor(mask_rows, dtype=torch.bool),
    }
    return metadata_df, tensor_payload


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    train_config = load_experiment_config(args.model_config)
    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)

    splits: list[tuple[str, str, HiddenCorridorDecisionDataset, object]] = []
    train_hidden_cfg = hidden_corridor_config_for_split(train_config.benchmark, "train")
    train_dataset = HiddenCorridorDecisionDataset(
        config=train_hidden_cfg,
        num_episodes=train_config.benchmark.train_episodes,
        curriculum_levels=train_config.benchmark.curriculum_levels,
    )
    splits.append(("train", f"{train_config.name}_train", train_dataset, train_hidden_cfg))

    for suite_config_path in args.eval_suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        splits.append(("eval", suite_config.name, dataset, hidden_cfg))

    frame_rows: list[pd.DataFrame] = []
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    split_summary_rows: list[dict[str, object]] = []

    for split, suite_name, dataset, hidden_cfg in splits:
        metadata_df, tensor_payload = _build_dataset_payload(
            model,
            dataset,
            device=device,
            config=hidden_cfg,
            suite=suite_name,
            split=split,
            top_k=args.top_k,
        )
        frame_rows.append(metadata_df)
        tensors[suite_name] = tensor_payload
        split_summary_rows.append(
            {
                "split": split,
                "suite": suite_name,
                "candidate_rows": len(metadata_df),
                "hard_near_tie_rows": int(metadata_df["hard_near_tie_intersection_case"].sum()) if len(metadata_df) else 0,
                "baseline_error_rows": int(metadata_df["baseline_error_hard_near_tie_case"].sum()) if len(metadata_df) else 0,
            }
        )

    metadata_csv = output_prefix.with_suffix(".csv")
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    tensor_path = output_prefix.with_suffix(".pt")
    json_path = output_prefix.with_suffix(".json")

    metadata_df = pd.concat(frame_rows, ignore_index=True) if frame_rows else pd.DataFrame()
    metadata_df.to_csv(metadata_csv, index=False)
    pd.DataFrame(split_summary_rows).to_csv(summary_csv, index=False)
    torch.save(tensors, tensor_path)
    json_path.write_text(
        json.dumps(
            {
                "train_config": args.model_config,
                "checkpoint": args.checkpoint,
                "top_k": args.top_k,
                "splits": split_summary_rows,
                "tensor_path": str(tensor_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(split_summary_rows).to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
