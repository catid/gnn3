from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn3.data.hidden_corridor import DecisionRecord, collate_decisions
from gnn3.eval.hard_feasible import HardFeasibleThresholds, annotate_hard_feasible
from gnn3.eval.near_tie import build_candidate_feature_tensor
from gnn3.eval.policy_analysis import (
    collect_decision_prediction_rows,
    collect_episode_policy_rows,
)
from gnn3.eval.rollout import _move_batch
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import (
    ExperimentConfig,
    hidden_corridor_config_for_split,
    load_experiment_config,
)
from gnn3.train.trainer import _resolve_device


@dataclass(frozen=True)
class FrontierConfig:
    thresholds: HardFeasibleThresholds
    high_headroom_threshold: float
    perturb_samples: int = 64
    perturb_sigma: float = 0.02


def load_frontier_config(path: str | Path) -> FrontierConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    threshold_payload = payload["thresholds"]
    return FrontierConfig(
        thresholds=HardFeasibleThresholds(
            gap_threshold=float(threshold_payload["large_gap_threshold"]),
            gap_ratio_threshold=float(threshold_payload["large_gap_ratio_threshold"]),
            near_tie_gap_threshold=float(threshold_payload["near_tie_gap_threshold"]),
            model_margin_threshold=float(threshold_payload["model_margin_threshold"]),
        ),
        high_headroom_threshold=float(threshold_payload["high_headroom_threshold"]),
        perturb_samples=int(payload.get("perturb_samples", 64)),
        perturb_sigma=float(payload.get("perturb_sigma", 0.02)),
    )


def load_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device_override: str | None = None,
) -> tuple[PacketMambaModel, torch.device, ExperimentConfig]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device, config


def load_suite_records(suite_config_path: str | Path) -> tuple[ExperimentConfig, object, list[DecisionRecord]]:
    suite_config = load_experiment_config(suite_config_path)
    hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
    from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset

    dataset = HiddenCorridorDecisionDataset(
        config=hidden_cfg,
        num_episodes=suite_config.benchmark.test_episodes,
        curriculum_levels=suite_config.benchmark.curriculum_levels,
    )
    return suite_config, dataset, list(dataset)


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


def annotate_frontier_slices(
    decision_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    records: list[DecisionRecord],
    *,
    frontier: FrontierConfig,
) -> pd.DataFrame:
    annotated, _thresholds = annotate_hard_feasible(
        decision_df,
        episode_df,
        thresholds=frontier.thresholds,
    )
    perturb_rates = [
        _perturb_flip_rate(
            records[int(row.decision_index)],
            samples=frontier.perturb_samples,
            sigma=frontier.perturb_sigma,
        )
        for row in annotated.itertuples(index=False)
    ]
    annotated = annotated.copy()
    annotated["perturb_flip_rate"] = perturb_rates
    annotated["effective_tie_under_perturbation"] = annotated["perturb_flip_rate"] >= 0.25
    annotated["stable_near_tie_case"] = (
        annotated["hard_near_tie_intersection_case"] & (~annotated["effective_tie_under_perturbation"])
    )
    annotated["high_headroom_near_tie_case"] = (
        annotated["hard_near_tie_intersection_case"]
        & (annotated["predicted_continuation_gap"] >= frontier.high_headroom_threshold)
    )
    annotated["decodable_near_tie_case"] = (
        annotated["hard_near_tie_intersection_case"]
        & (annotated["model_margin"] > frontier.thresholds.model_margin_threshold)
    )
    annotated["weakly_decodable_near_tie_case"] = (
        annotated["hard_near_tie_intersection_case"] & (~annotated["decodable_near_tie_case"])
    )
    return annotated


def collect_frontier_predictions(
    model: PacketMambaModel,
    dataset,
    records: list[DecisionRecord],
    *,
    device: torch.device,
    suite_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_df = pd.DataFrame(
        collect_episode_policy_rows(
            model,
            dataset.episodes,
            device=device,
            config=dataset.config,
            suite=suite_name,
        )
    )
    decision_df = pd.DataFrame(
        collect_decision_prediction_rows(
            model,
            records,
            device=device,
            suite=suite_name,
        )
    )
    return decision_df, episode_df


def merge_helpfulness_labels(
    base_df: pd.DataFrame,
    compute_df: pd.DataFrame,
    *,
    gap_epsilon: float = 0.05,
) -> pd.DataFrame:
    merge_cols = ["suite", "episode_index", "decision_index"]
    base_cols = {
        "predicted_next_hop": "base_predicted_next_hop",
        "target_match": "base_target_match",
        "predicted_continuation_gap": "base_predicted_continuation_gap",
        "predicted_on_time": "base_predicted_on_time",
        "predicted_cost_to_go": "base_predicted_cost_to_go",
        "model_margin": "base_model_margin",
    }
    compute_cols = {
        "predicted_next_hop": "compute_predicted_next_hop",
        "target_match": "compute_target_match",
        "predicted_continuation_gap": "compute_predicted_continuation_gap",
        "predicted_on_time": "compute_predicted_on_time",
        "predicted_cost_to_go": "compute_predicted_cost_to_go",
        "model_margin": "compute_model_margin",
    }
    merged = base_df.rename(columns=base_cols).merge(
        compute_df[merge_cols + list(compute_cols.keys())].rename(columns=compute_cols),
        on=merge_cols,
        how="inner",
    )
    merged["action_changed"] = merged["compute_predicted_next_hop"] != merged["base_predicted_next_hop"]
    merged["delta_regret"] = merged["compute_predicted_continuation_gap"] - merged["base_predicted_continuation_gap"]
    merged["delta_cost_to_go"] = merged["compute_predicted_cost_to_go"] - merged["base_predicted_cost_to_go"]
    merged["delta_miss"] = (~merged["compute_predicted_on_time"]).astype(int) - (~merged["base_predicted_on_time"]).astype(int)

    improved_match = merged["compute_target_match"] & (~merged["base_target_match"])
    worsened_match = merged["base_target_match"] & (~merged["compute_target_match"])
    improved_gap = merged["delta_regret"] <= -gap_epsilon
    worsened_gap = merged["delta_regret"] >= gap_epsilon
    improved_miss = merged["delta_miss"] < 0
    worsened_miss = merged["delta_miss"] > 0

    helpful = improved_match | improved_miss | (merged["action_changed"] & improved_gap & (~worsened_miss))
    harmful = worsened_match | worsened_miss | (merged["action_changed"] & worsened_gap & (~improved_miss))
    helpful = helpful & (~harmful)
    neutral = (~helpful) & (~harmful)

    merged["helpful_compute"] = helpful
    merged["harmful_compute"] = harmful
    merged["neutral_compute"] = neutral
    merged["helpfulness"] = np.select(
        [helpful.to_numpy(), harmful.to_numpy()],
        ["helpful", "harmful"],
        default="neutral",
    )
    merged["compute_recovers_baseline_error"] = merged["baseline_error_hard_near_tie_case"] & improved_match
    merged["compute_breaks_baseline_success"] = (~merged["baseline_error_hard_near_tie_case"]) & worsened_match
    merged["teacher_next_hop"] = np.where(helpful, merged["compute_predicted_next_hop"], merged["base_predicted_next_hop"])
    return merged


def build_feature_cache(
    model: PacketMambaModel,
    other_model: PacketMambaModel | None,
    records: list[DecisionRecord],
    *,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    decision_feature_chunks: list[torch.Tensor] = []
    candidate_feature_chunks: list[torch.Tensor] = []
    base_score_chunks: list[torch.Tensor] = []
    compute_score_chunks: list[torch.Tensor] = []
    per_step_probe_chunks: list[torch.Tensor] = []
    per_step_score_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    candidate_mask_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    candidate_cost_chunks: list[torch.Tensor] = []
    candidate_on_time_chunks: list[torch.Tensor] = []
    candidate_slack_chunks: list[torch.Tensor] = []
    packet_count_chunks: list[torch.Tensor] = []
    packet_deadline_chunks: list[torch.Tensor] = []
    packet_priority_chunks: list[torch.Tensor] = []
    max_candidates = 0
    max_other_candidates = 0
    other_model_device = None
    if other_model is not None:
        other_model_device = next(other_model.parameters()).device
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        batch_cpu = collate_decisions(batch_records)
        batch = _move_batch(batch_cpu, device)
        output = model(batch)
        candidate_features = build_candidate_feature_tensor(
            batch,
            probe_features=output["probe_features"],
            selection_scores=output["selection_scores"],
            values=output["values"],
        ).detach().cpu()
        decision_feature_chunks.append(output["probe_features"].detach().cpu())
        per_step_probe_chunks.append(output["per_step_probe_features"].detach().cpu())
        per_step_score = output["per_step_selection_scores"].detach().cpu()
        per_step_score_chunks.append(per_step_score)
        value_chunks.append(output["values"].detach().cpu())
        base_scores = output["selection_scores"].detach().cpu()
        base_score_chunks.append(base_scores)
        candidate_feature_chunks.append(candidate_features)
        candidate_mask_chunks.append((batch["candidate_mask"] & batch["node_mask"]).detach().cpu())
        target_chunks.append(batch["target_next_hop"].detach().cpu())
        candidate_cost_chunks.append(batch["candidate_cost_to_go"].detach().cpu())
        candidate_on_time_chunks.append(batch["candidate_on_time"].detach().cpu())
        candidate_slack_chunks.append(batch["candidate_slack"].detach().cpu())
        packet_count_chunks.append(batch["packet_count"].detach().cpu())
        packet_deadline_chunks.append(batch["packet_deadline"].detach().cpu())
        packet_priority_chunks.append(batch["packet_priority"].detach().cpu())
        max_candidates = max(max_candidates, int(base_scores.size(1)))
        if other_model is not None:
            if other_model_device == device:
                other_batch = batch
            elif other_model_device is not None:
                other_batch = _move_batch(batch_cpu, other_model_device)
            else:
                other_batch = batch_cpu
            other_output = other_model(other_batch)
            other_scores = other_output["selection_scores"].detach().cpu()
            compute_score_chunks.append(other_scores)
            max_other_candidates = max(max_other_candidates, int(other_scores.size(1)))

    def _pad_score_chunks(chunks: list[torch.Tensor], width: int) -> torch.Tensor:
        padded = []
        for chunk in chunks:
            if int(chunk.size(-1)) == width:
                padded.append(chunk)
            else:
                padded.append(torch.nn.functional.pad(chunk, (0, width - int(chunk.size(-1)))))
        return torch.cat(padded, dim=0) if padded else torch.empty((0, width), dtype=torch.float32)

    def _pad_per_step_chunks(chunks: list[torch.Tensor], width: int) -> torch.Tensor:
        padded = []
        for chunk in chunks:
            if int(chunk.size(-1)) == width:
                padded.append(chunk)
            else:
                padded.append(torch.nn.functional.pad(chunk, (0, width - int(chunk.size(-1)))))
        return torch.cat(padded, dim=0) if padded else torch.empty((0, 0, width), dtype=torch.float32)

    def _pad_candidate_features(chunks: list[torch.Tensor], width: int) -> torch.Tensor:
        padded = []
        for chunk in chunks:
            if int(chunk.size(1)) == width:
                padded.append(chunk)
            else:
                pad = width - int(chunk.size(1))
                padded.append(torch.nn.functional.pad(chunk, (0, 0, 0, pad)))
        return torch.cat(padded, dim=0) if padded else torch.empty((0, width, 0), dtype=torch.float32)

    def _pad_candidate_masks(chunks: list[torch.Tensor], width: int) -> torch.Tensor:
        padded = []
        for chunk in chunks:
            if int(chunk.size(1)) == width:
                padded.append(chunk)
            else:
                pad = width - int(chunk.size(1))
                padded.append(torch.nn.functional.pad(chunk, (0, pad), value=False))
        return torch.cat(padded, dim=0) if padded else torch.empty((0, width), dtype=torch.bool)

    def _pad_candidate_values(chunks: list[torch.Tensor], width: int) -> torch.Tensor:
        padded = []
        for chunk in chunks:
            if int(chunk.size(1)) == width:
                padded.append(chunk)
            else:
                padded.append(torch.nn.functional.pad(chunk, (0, width - int(chunk.size(1)))))
        return torch.cat(padded, dim=0) if padded else torch.empty((0, width), dtype=torch.float32)

    score_width = max(max_candidates, max_other_candidates)
    return {
        "decision_features": torch.cat(decision_feature_chunks, dim=0),
        "candidate_features": _pad_candidate_features(candidate_feature_chunks, score_width),
        "base_selection_scores": _pad_score_chunks(base_score_chunks, score_width),
        "compute_selection_scores": _pad_score_chunks(compute_score_chunks, score_width)
        if compute_score_chunks
        else torch.empty((0, score_width), dtype=torch.float32),
        "per_step_probe_features": torch.cat(per_step_probe_chunks, dim=0),
        "per_step_selection_scores": _pad_per_step_chunks(per_step_score_chunks, score_width),
        "values": torch.cat(value_chunks, dim=0),
        "candidate_mask": _pad_candidate_masks(candidate_mask_chunks, score_width),
        "candidate_cost_to_go": _pad_candidate_values(candidate_cost_chunks, score_width),
        "candidate_on_time": _pad_candidate_values(candidate_on_time_chunks, score_width),
        "candidate_slack": _pad_candidate_values(candidate_slack_chunks, score_width),
        "target_next_hop": torch.cat(target_chunks, dim=0),
        "packet_count": torch.cat(packet_count_chunks, dim=0),
        "packet_deadline": torch.cat(packet_deadline_chunks, dim=0),
        "packet_priority": torch.cat(packet_priority_chunks, dim=0),
    }
