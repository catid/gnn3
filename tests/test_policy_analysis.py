from __future__ import annotations

import torch

from gnn3.data.hidden_corridor import HiddenCorridorConfig, HiddenCorridorDecisionDataset
from gnn3.eval.policy_analysis import (
    collect_decision_prediction_rows,
    collect_episode_policy_rows,
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
