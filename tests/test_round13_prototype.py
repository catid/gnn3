from __future__ import annotations

import pandas as pd

from gnn3.eval.round13_prototype import (
    KEY_COLS,
    aggregate_split_summary,
    load_unique_scores,
    merge_scores_by_split,
    summarize_selection,
)


def test_load_unique_scores_deduplicates_budget_rows(tmp_path) -> None:
    path = tmp_path / "decisions.csv"
    frame = pd.DataFrame(
        [
            {"suite": "s", "episode_index": 0, "decision_index": 0, "variant": "v", "split": "seed315", "budget_pct": 0.25, "score": 1.5},
            {"suite": "s", "episode_index": 0, "decision_index": 0, "variant": "v", "split": "seed315", "budget_pct": 0.50, "score": 1.5},
            {"suite": "s", "episode_index": 0, "decision_index": 1, "variant": "v", "split": "seed315", "budget_pct": 0.25, "score": 0.2},
            {"suite": "s", "episode_index": 0, "decision_index": 1, "variant": "other", "split": "seed315", "budget_pct": 0.25, "score": 0.7},
        ]
    )
    frame.to_csv(path, index=False)

    unique = load_unique_scores(path, variant="v")

    assert list(unique.columns) == [*KEY_COLS, "variant", "split", "score"]
    assert len(unique) == 2
    assert set(unique["decision_index"]) == {0, 1}


def test_summarize_selection_and_aggregate_split_summary() -> None:
    frame = pd.DataFrame(
        {
            "suite": ["s", "s"],
            "episode_index": [0, 0],
            "decision_index": [0, 1],
            "score": [1.0, 0.1],
            "stable_positive_v2_case": [True, False],
            "stable_positive_v2_committee_case": [False, False],
            "hard_near_tie_intersection_case": [True, True],
            "stable_near_tie_case": [True, True],
            "high_headroom_near_tie_case": [True, False],
            "baseline_error_hard_near_tie_case": [True, False],
            "large_gap_hard_feasible_case": [False, False],
            "harmful_teacher_bank_case": [False, False],
            "compute_recovers_baseline_error": [True, False],
            "compute_breaks_baseline_success": [False, False],
            "base_target_match": [False, True],
            "compute_target_match": [True, True],
            "delta_regret": [-0.2, 0.0],
            "delta_miss": [-1.0, 0.0],
            "base_predicted_next_hop": ["a", "b"],
            "compute_predicted_next_hop": ["c", "b"],
        }
    )
    selected = pd.Series([True, False]).to_numpy()

    split315 = summarize_selection(
        frame,
        family="f",
        variant="v",
        split="seed315",
        budget_pct=0.5,
        selected=selected,
    )
    split316 = summarize_selection(
        frame,
        family="f",
        variant="v",
        split="seed316",
        budget_pct=0.5,
        selected=selected,
    )
    aggregate = aggregate_split_summary(pd.concat([split315, split316], ignore_index=True))
    stable = aggregate.loc[aggregate["slice"] == "stable_positive_v2"].iloc[0]
    overall = aggregate.loc[aggregate["slice"] == "overall"].iloc[0]

    assert stable["stable_positive_total"] == 2
    assert stable["stable_positive_selected"] == 2
    assert stable["stable_positive_recall"] == 1.0
    assert overall["defer_precision"] == 1.0
    assert overall["system_target_match"] == 1.0
    assert overall["mean_delta_regret"] < 0.0


def test_merge_scores_by_split_accepts_combined_heldout_frame() -> None:
    meta_a = pd.DataFrame(
        {
            "suite": ["s1"],
            "episode_index": [0],
            "decision_index": [0],
        }
    )
    meta_b = pd.DataFrame(
        {
            "suite": ["s2"],
            "episode_index": [0],
            "decision_index": [0],
        }
    )
    scores = pd.DataFrame(
        {
            "split": ["heldout", "heldout"],
            "suite": ["s1", "s2"],
            "episode_index": [0, 0],
            "decision_index": [0, 0],
            "score": [1.0, 2.0],
        }
    )

    merged = merge_scores_by_split({"seed315": meta_a, "seed316": meta_b}, scores)

    assert list(merged) == ["heldout"]
    assert len(merged["heldout"]) == 2
