from __future__ import annotations

import pandas as pd
import torch

from gnn3.eval.precision_correction import (
    annotate_stable_positive_pack,
    build_source_signature,
    candidate_pair_features,
    signature_overlap_rows,
    teacher_effect_labels,
    top_fraction_mask,
)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "suite": ["s1", "s1", "s2"],
            "depth_load_regime": ["tight_high", "tight_high", "mid_mid"],
            "slack_band": ["critical", "critical", "tight"],
            "packet_band": ["5+", "5+", "4"],
            "load_band": ["high", "high", "mid"],
            "depth_band": ["4", "4", "3"],
            "gap_bucket": ["near_tie", "near_tie", "near_tie"],
            "critical_packet_proxy": [True, True, False],
            "high_headroom_near_tie_case": [True, False, False],
            "baseline_error_hard_near_tie_case": [True, False, True],
            "hard_near_tie_intersection_case": [True, True, True],
            "stable_near_tie_case": [True, False, True],
            "action_changed": [True, True, True],
            "helpful_compute": [True, True, False],
            "harmful_compute": [False, False, True],
            "delta_regret": [-0.5, -0.2, 0.3],
            "delta_miss": [0, 0, 1],
            "compute_recovers_baseline_error": [True, False, False],
            "seed": [314, 315, 314],
        }
    )


def test_annotate_stable_positive_pack_marks_only_stable_high_value_helpful_cases() -> None:
    annotated = annotate_stable_positive_pack(_base_frame(), min_regret_gain=0.1)
    assert annotated["stable_positive_teacher_case"].tolist() == [True, False, False]
    assert annotated["unstable_positive_teacher_case"].tolist() == [False, True, False]
    assert annotated["harmful_teacher_case"].tolist() == [False, False, True]


def test_signature_overlap_rows_reports_pairwise_jaccard() -> None:
    frame = annotate_stable_positive_pack(_base_frame(), min_regret_gain=0.1)
    overlap = signature_overlap_rows(frame, subset_col="stable_positive_teacher_case", group_col="seed")
    assert list(overlap.columns) == ["left_group", "right_group", "left_size", "right_size", "signature_jaccard"]
    assert len(overlap) == 1
    assert overlap.iloc[0]["signature_jaccard"] == 0.0


def test_top_fraction_mask_respects_requested_budget() -> None:
    scores = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.3]).numpy()
    mask = top_fraction_mask(scores, 40.0)
    assert mask.tolist() == [False, True, False, True, False]


def test_candidate_pair_features_builds_batch_aligned_tensor() -> None:
    cache = {
        "candidate_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "base_selection_scores": torch.tensor([[0.9, 0.7, -1.0], [0.5, 0.4, 0.3]]),
        "candidate_cost_to_go": torch.tensor([[1.0, 1.2, 9.9], [1.3, 1.5, 1.7]]),
        "candidate_on_time": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        "candidate_features": torch.randn(2, 3, 4),
        "decision_features": torch.randn(2, 5),
    }
    metadata = pd.DataFrame(
        {
            "base_model_margin": [0.2, 0.1],
            "best_candidate_slack_ratio": [0.3, 0.4],
            "packet_count": [5, 6],
            "mean_queue": [3.0, 4.0],
            "max_tree_depth": [4, 4],
        }
    )
    features = candidate_pair_features(cache, metadata)
    assert features.shape[0] == 2
    assert features.ndim == 2


def test_teacher_effect_labels_marks_helpful_harmful_and_recovery_flags() -> None:
    labels = teacher_effect_labels(
        base_target_match=[False, True, True, False],
        teacher_target_match=[True, False, True, False],
        delta_regret=[-0.4, 0.3, -0.02, 0.0],
        delta_miss=[0.0, 1.0, 0.0, 0.0],
        action_changed=[True, True, False, False],
        baseline_error_hard_near_tie_case=[True, False, False, True],
    )
    assert labels["helpful"].tolist() == [True, False, False, False]
    assert labels["harmful"].tolist() == [False, True, False, False]
    assert labels["neutral"].tolist() == [False, False, True, True]
    assert labels["recovers_baseline_error"].tolist() == [True, False, False, False]
    assert labels["breaks_baseline_success"].tolist() == [False, True, False, False]


def test_build_source_signature_supports_coarse_mode() -> None:
    frame = _base_frame()
    fine = build_source_signature(frame)
    coarse = build_source_signature(frame, include_suite=False, include_critical_packet=False)
    assert fine.iloc[0].startswith("s1|")
    assert not coarse.iloc[0].startswith("s1|")
    assert "True" in fine.iloc[0]
    assert fine.iloc[0] != coarse.iloc[0]
