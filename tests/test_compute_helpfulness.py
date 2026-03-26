from __future__ import annotations

import pandas as pd
import torch

from gnn3.data.hidden_corridor import HiddenCorridorConfig, HiddenCorridorDecisionDataset
from gnn3.eval.compute_helpfulness import build_feature_cache, merge_helpfulness_labels
from gnn3.models.packet_mamba import PacketMambaConfig, PacketMambaModel


def test_merge_helpfulness_labels_marks_helpful_harmful_and_neutral() -> None:
    base = pd.DataFrame(
        [
            {
                "suite": "s",
                "episode_index": 0,
                "decision_index": 0,
                "predicted_next_hop": 1,
                "target_match": False,
                "predicted_continuation_gap": 1.5,
                "predicted_on_time": True,
                "predicted_cost_to_go": 3.0,
                "model_margin": 0.1,
                "baseline_error_hard_near_tie_case": True,
            },
            {
                "suite": "s",
                "episode_index": 0,
                "decision_index": 1,
                "predicted_next_hop": 2,
                "target_match": True,
                "predicted_continuation_gap": 0.0,
                "predicted_on_time": True,
                "predicted_cost_to_go": 2.0,
                "model_margin": 0.2,
                "baseline_error_hard_near_tie_case": False,
            },
            {
                "suite": "s",
                "episode_index": 0,
                "decision_index": 2,
                "predicted_next_hop": 3,
                "target_match": True,
                "predicted_continuation_gap": 0.2,
                "predicted_on_time": True,
                "predicted_cost_to_go": 2.1,
                "model_margin": 0.3,
                "baseline_error_hard_near_tie_case": False,
            },
        ]
    )
    compute = pd.DataFrame(
        [
            {
                "suite": "s",
                "episode_index": 0,
                "decision_index": 0,
                "predicted_next_hop": 4,
                "target_match": True,
                "predicted_continuation_gap": 0.0,
                "predicted_on_time": True,
                "predicted_cost_to_go": 1.0,
                "model_margin": 0.4,
            },
            {
                "suite": "s",
                "episode_index": 0,
                "decision_index": 1,
                "predicted_next_hop": 5,
                "target_match": False,
                "predicted_continuation_gap": 1.0,
                "predicted_on_time": False,
                "predicted_cost_to_go": 3.5,
                "model_margin": 0.5,
            },
            {
                "suite": "s",
                "episode_index": 0,
                "decision_index": 2,
                "predicted_next_hop": 3,
                "target_match": True,
                "predicted_continuation_gap": 0.21,
                "predicted_on_time": True,
                "predicted_cost_to_go": 2.12,
                "model_margin": 0.6,
            },
        ]
    )

    merged = merge_helpfulness_labels(base, compute)
    assert merged["helpfulness"].tolist() == ["helpful", "harmful", "neutral"]
    assert merged["teacher_next_hop"].tolist() == [4, 2, 3]


def test_build_feature_cache_pads_candidate_widths_and_scores() -> None:
    records = list(
        HiddenCorridorDecisionDataset(
            config=HiddenCorridorConfig(seed=41, packets_max=4, deadline_mode="oracle_calibrated"),
            num_episodes=4,
            curriculum_levels=("multi_dynamic",),
        )
    )
    model = PacketMambaModel(PacketMambaConfig(d_model=32, d_state=8, inner_layers=1, outer_steps=3))
    cache = build_feature_cache(model, model, records, device=torch.device("cpu"), batch_size=1)
    assert cache["decision_features"].shape[0] == len(records)
    assert cache["candidate_features"].shape[0] == len(records)
    assert cache["base_selection_scores"].shape == cache["compute_selection_scores"].shape
    assert cache["candidate_mask"].shape == cache["base_selection_scores"].shape
    assert cache["per_step_selection_scores"].shape[0] == len(records)
