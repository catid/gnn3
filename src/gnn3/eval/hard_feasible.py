from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HardFeasibleThresholds:
    gap_threshold: float
    gap_ratio_threshold: float
    near_tie_gap_threshold: float
    model_margin_threshold: float


def slack_band(value: float) -> str:
    if value < 0.02:
        return "critical"
    if value < 0.05:
        return "very_tight"
    if value < 0.10:
        return "tight"
    if value < 0.20:
        return "moderate"
    return "loose"


def packet_band(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if value == 4:
        return "4"
    return "5+"


def _gap_bucket_from_thresholds(value: float, *, medium_threshold: float, large_threshold: float) -> str:
    if value <= medium_threshold:
        return "near_tie"
    if value <= large_threshold:
        return "medium_gap"
    return "large_gap"


def annotate_hard_feasible(
    decision_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    *,
    thresholds: HardFeasibleThresholds | None = None,
) -> tuple[pd.DataFrame, HardFeasibleThresholds]:
    episode_cols = [
        "suite",
        "episode_index",
        "packet_count",
        "max_tree_depth",
        "hub_asymmetry",
        "mean_queue",
        "p90_queue",
        "max_queue",
        "deadline_miss",
        "regret",
        "solved",
    ]
    merged = decision_df.merge(
        episode_df[episode_cols],
        on=["suite", "episode_index", "packet_count"],
        how="left",
        suffixes=("", "_episode"),
    ).copy()
    merged["slack_band"] = merged["best_candidate_slack_ratio"].map(slack_band)
    merged["packet_band"] = merged["packet_count"].map(packet_band)
    merged["load_percentile"] = merged["mean_queue"].rank(method="average", pct=True)
    merged["load_band"] = pd.cut(
        merged["load_percentile"],
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["low_load", "mid_load", "high_load"],
        include_lowest=True,
    ).astype(str)
    merged["depth_band"] = merged["max_tree_depth"].astype(int).astype(str)
    merged["hard_condition_count"] = (
        merged["slack_band"].isin(["critical", "very_tight"]).astype(int)
        + (merged["packet_count"] >= 5).astype(int)
        + (merged["max_tree_depth"] >= 4).astype(int)
        + (merged["load_band"] == "high_load").astype(int)
    )
    # Round-seven audit uses a score-based hard slice rather than the empty
    # four-way intersection, which proved too strict on the corrected suites.
    merged["hard_feasible_case"] = merged["any_feasible_candidate"] & (merged["hard_condition_count"] >= 2)
    hard_slice = merged.loc[merged["hard_feasible_case"]]
    if thresholds is None:
        if hard_slice.empty:
            thresholds = HardFeasibleThresholds(
                gap_threshold=0.5,
                gap_ratio_threshold=0.10,
                near_tie_gap_threshold=0.25,
                model_margin_threshold=0.10,
            )
        else:
            positive_hard_gaps = hard_slice.loc[hard_slice["oracle_action_gap"] > 0.0, "oracle_action_gap"]
            hard_margins = hard_slice.loc[hard_slice["model_margin"] > 0.0, "model_margin"]
            thresholds = HardFeasibleThresholds(
                gap_threshold=max(float(hard_slice["oracle_action_gap"].median()), 0.5),
                gap_ratio_threshold=max(float(hard_slice["oracle_action_gap_ratio"].median()), 0.10),
                near_tie_gap_threshold=max(
                    float(positive_hard_gaps.quantile(0.25)) if not positive_hard_gaps.empty else 0.25,
                    0.05,
                ),
                model_margin_threshold=max(
                    float(hard_margins.quantile(0.25)) if not hard_margins.empty else 0.10,
                    0.01,
                ),
            )
    merged["large_gap_hard_feasible_case"] = (
        merged["hard_feasible_case"]
        & (merged["oracle_action_gap"] >= thresholds.gap_threshold)
        & (merged["oracle_action_gap_ratio"] >= thresholds.gap_ratio_threshold)
    )
    merged["oracle_near_tie_case"] = (
        merged["hard_feasible_case"] & (merged["oracle_action_gap"] <= thresholds.near_tie_gap_threshold)
    )
    merged["model_near_tie_case"] = (
        merged["hard_feasible_case"] & (merged["model_margin"] <= thresholds.model_margin_threshold)
    )
    merged["hard_near_tie_intersection_case"] = merged["oracle_near_tie_case"] & merged["model_near_tie_case"]
    merged["baseline_error_hard_near_tie_case"] = merged["hard_near_tie_intersection_case"] & (
        merged["strictly_suboptimal"] | (~merged["predicted_on_time"])
    )
    positive_gaps = merged.loc[merged["oracle_action_gap"] > 0.0, "oracle_action_gap"]
    medium_threshold = float(positive_gaps.quantile(0.25)) if not positive_gaps.empty else 0.25
    large_threshold = float(positive_gaps.quantile(0.75)) if not positive_gaps.empty else 1.0
    merged["gap_bucket"] = merged["oracle_action_gap"].map(
        lambda value: _gap_bucket_from_thresholds(
            float(value),
            medium_threshold=max(medium_threshold, 0.1),
            large_threshold=max(large_threshold, max(medium_threshold, 0.1)),
        )
    )
    merged["baseline_error"] = merged["strictly_suboptimal"] | (~merged["predicted_on_time"])
    merged["depth_load_regime"] = (
        "d"
        + merged["max_tree_depth"].fillna(-1).astype(int).astype(str)
        + "_"
        + merged["load_band"].astype(str)
    )
    merged["critical_packet_proxy"] = (
        merged.groupby(["suite", "episode_index"])["best_candidate_slack_ratio"].transform("min")
        >= merged["best_candidate_slack_ratio"] - 1e-9
    )
    return merged, thresholds


def build_probe_labels(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame[["suite", "episode_index", "decision_index"]].copy()
    slack_order = {"critical": 0, "very_tight": 1, "tight": 2, "moderate": 3, "loose": 4}
    gap_order = {"near_tie": 0, "medium_gap": 1, "large_gap": 2}
    depth_load_vocab = {value: idx for idx, value in enumerate(sorted(frame["depth_load_regime"].unique().tolist()))}
    labels["slack_bucket"] = frame["slack_band"].map(slack_order).astype(int)
    labels["critical_packet_proxy"] = frame["critical_packet_proxy"].astype(int)
    labels["feasible_continuation"] = frame["any_feasible_candidate"].astype(int)
    labels["oracle_gap_bucket"] = frame["gap_bucket"].map(gap_order).astype(int)
    labels["depth_load_regime"] = frame["depth_load_regime"].map(depth_load_vocab).astype(int)
    labels["baseline_strictly_suboptimal"] = frame["strictly_suboptimal"].astype(int)
    labels["hard_near_tie_baseline_error"] = frame["baseline_error_hard_near_tie_case"].astype(int)
    labels["oracle_near_tie"] = frame["oracle_near_tie_case"].astype(int)
    return labels
