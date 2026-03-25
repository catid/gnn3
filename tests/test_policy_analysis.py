from __future__ import annotations

import torch

from gnn3.data.hidden_corridor import HiddenCorridorConfig, HiddenCorridorDecisionDataset
from gnn3.eval.policy_analysis import (
    collect_decision_prediction_rows,
    collect_episode_policy_rows,
    extract_decision_latents,
    extract_probe_features,
)
from gnn3.models.packet_mamba import PacketMambaConfig, PacketMambaModel


def test_collect_decision_prediction_rows_matches_dataset_length() -> None:
    dataset = HiddenCorridorDecisionDataset(
        config=HiddenCorridorConfig(seed=21, packets_max=2, deadline_mode="oracle_calibrated"),
        num_episodes=2,
        curriculum_levels=("single_dynamic",),
    )
    model = PacketMambaModel(PacketMambaConfig(d_model=32, d_state=8, inner_layers=1, outer_steps=1))
    rows = collect_decision_prediction_rows(model, list(dataset), device=torch.device("cpu"), suite="unit")
    assert len(rows) == len(dataset)
    assert rows[0]["suite"] == "unit"
    assert "best_candidate_slack_ratio" in rows[0]
    assert "oracle_action_gap" in rows[0]
    assert "predicted_continuation_gap" in rows[0]
    assert rows[0]["episode_index"] >= 0


def test_collect_episode_policy_rows_matches_episode_length() -> None:
    dataset = HiddenCorridorDecisionDataset(
        config=HiddenCorridorConfig(seed=22, packets_max=2, deadline_mode="oracle_calibrated"),
        num_episodes=2,
        curriculum_levels=("single_dynamic",),
    )
    model = PacketMambaModel(PacketMambaConfig(d_model=32, d_state=8, inner_layers=1, outer_steps=1))
    rows = collect_episode_policy_rows(
        model,
        dataset.episodes,
        device=torch.device("cpu"),
        config=dataset.config,
        suite="unit",
    )
    assert len(rows) == len(dataset.episodes)
    assert rows[0]["suite"] == "unit"
    assert "hub_asymmetry" in rows[0]


def test_extract_probe_features_matches_dataset_length() -> None:
    dataset = HiddenCorridorDecisionDataset(
        config=HiddenCorridorConfig(seed=23, packets_max=2, deadline_mode="oracle_calibrated"),
        num_episodes=2,
        curriculum_levels=("single_dynamic",),
    )
    model = PacketMambaModel(PacketMambaConfig(d_model=32, d_state=8, inner_layers=1, outer_steps=1))
    features = extract_probe_features(model, list(dataset), device=torch.device("cpu"))
    assert features.shape[0] == len(dataset)
    assert features.shape[1] > 32


def test_extract_decision_latents_exposes_per_step_tensors() -> None:
    dataset = HiddenCorridorDecisionDataset(
        config=HiddenCorridorConfig(seed=24, packets_max=2, deadline_mode="oracle_calibrated"),
        num_episodes=2,
        curriculum_levels=("single_dynamic",),
    )
    model = PacketMambaModel(PacketMambaConfig(d_model=32, d_state=8, inner_layers=1, outer_steps=3))
    latents = extract_decision_latents(model, list(dataset), device=torch.device("cpu"))
    assert latents["probe_features"].shape[0] == len(dataset)
    assert latents["per_step_selection_scores"].shape[:2] == (len(dataset), 3)
    assert latents["per_step_probe_features"].shape[:2] == (len(dataset), 3)


def test_extract_decision_latents_pads_variable_candidate_widths_across_batches() -> None:
    records = list(
        HiddenCorridorDecisionDataset(
            config=HiddenCorridorConfig(seed=25, packets_max=4, deadline_mode="oracle_calibrated"),
            num_episodes=4,
            curriculum_levels=("multi_dynamic",),
        )
    )
    widths = {int(record.candidate_mask.sum()) for record in records}
    assert len(widths) > 1
    model = PacketMambaModel(PacketMambaConfig(d_model=32, d_state=8, inner_layers=1, outer_steps=3))
    latents = extract_decision_latents(model, records, device=torch.device("cpu"), batch_size=1)
    assert latents["probe_features"].shape[0] == len(records)
    assert latents["per_step_selection_scores"].shape[:2] == (len(records), 3)
    assert latents["per_step_selection_scores"].shape[2] == latents["selection_scores"].shape[1]
