from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

DEFAULT_COVERAGE_BUDGETS = (0.25, 0.5, 1.0, 2.0, 5.0)
ULTRALOW_COVERAGE_BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)


def seed_from_text(text: str) -> int | None:
    match = re.search(r"seed(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def add_seed_column(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    seed = seed_from_text(source_name)
    annotated = frame.copy()
    annotated["source_name"] = source_name
    annotated["seed"] = -1 if seed is None else seed
    return annotated


def load_decision_frames(paths: Iterable[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frames.append(add_seed_column(frame, path))
    if not frames:
        raise ValueError("No decision CSVs were provided.")
    return pd.concat(frames, ignore_index=True)


def annotate_stable_positive_pack(
    frame: pd.DataFrame,
    *,
    min_regret_gain: float = 0.10,
) -> pd.DataFrame:
    annotated = frame.copy()
    regret_gain = (-annotated["delta_regret"]).astype(float)
    miss_gain = (-annotated["delta_miss"]).astype(float)
    action_changed = annotated["action_changed"].astype(bool)
    helpful = annotated["helpful_compute"].astype(bool)
    harmful = annotated["harmful_compute"].astype(bool)
    stable_near_tie = annotated["stable_near_tie_case"].astype(bool)
    hard_near_tie = annotated["hard_near_tie_intersection_case"].astype(bool)
    high_headroom = annotated["high_headroom_near_tie_case"].astype(bool)
    baseline_error = annotated["baseline_error_hard_near_tie_case"].astype(bool)
    stable_gain = (regret_gain >= float(min_regret_gain)) | (miss_gain > 0.0) | annotated[
        "compute_recovers_baseline_error"
    ].astype(bool)

    stable_positive = hard_near_tie & stable_near_tie & action_changed & helpful & stable_gain & (
        high_headroom | baseline_error
    )
    unstable_positive = hard_near_tie & action_changed & helpful & (~stable_positive)
    harmful_correction = hard_near_tie & action_changed & harmful
    neutral_correction = hard_near_tie & (~stable_positive) & (~unstable_positive) & (~harmful_correction)

    annotated["teacher_regret_gain"] = regret_gain
    annotated["teacher_miss_gain"] = miss_gain
    annotated["stable_positive_teacher_case"] = stable_positive
    annotated["unstable_positive_teacher_case"] = unstable_positive
    annotated["harmful_teacher_case"] = harmful_correction
    annotated["neutral_teacher_case"] = neutral_correction
    annotated["stable_positive_family"] = np.select(
        [
            stable_positive & high_headroom & baseline_error,
            stable_positive & high_headroom,
            stable_positive & baseline_error,
        ],
        [
            "high_headroom+baseline_error",
            "high_headroom",
            "baseline_error",
        ],
        default="none",
    )
    annotated["stable_positive_signature"] = build_source_signature(annotated)
    return annotated


def build_source_signature(
    frame: pd.DataFrame,
    *,
    include_suite: bool = True,
    include_critical_packet: bool = True,
) -> pd.Series:
    keys = []
    if include_suite:
        keys.append(frame["suite"].astype(str))
    keys.extend(
        [
            frame["depth_load_regime"].astype(str),
            frame["slack_band"].astype(str),
            frame["packet_band"].astype(str),
            frame["load_band"].astype(str),
            frame["depth_band"].astype(str),
            frame["gap_bucket"].astype(str),
        ]
    )
    if include_critical_packet:
        keys.append(frame["critical_packet_proxy"].astype(str))
    keys.extend(
        [
            frame["high_headroom_near_tie_case"].astype(int).astype(str),
            frame["baseline_error_hard_near_tie_case"].astype(int).astype(str),
        ]
    )
    value = keys[0]
    for key in keys[1:]:
        value = value + "|" + key
    return value


def teacher_effect_labels(
    *,
    base_target_match: Sequence[bool] | np.ndarray,
    teacher_target_match: Sequence[bool] | np.ndarray,
    delta_regret: Sequence[float] | np.ndarray,
    delta_miss: Sequence[float] | np.ndarray,
    action_changed: Sequence[bool] | np.ndarray,
    baseline_error_hard_near_tie_case: Sequence[bool] | np.ndarray | None = None,
    gap_epsilon: float = 0.05,
) -> pd.DataFrame:
    base_target_match_np = np.asarray(base_target_match, dtype=bool)
    teacher_target_match_np = np.asarray(teacher_target_match, dtype=bool)
    delta_regret_np = np.asarray(delta_regret, dtype=float)
    delta_miss_np = np.asarray(delta_miss, dtype=float)
    action_changed_np = np.asarray(action_changed, dtype=bool)
    if baseline_error_hard_near_tie_case is None:
        baseline_error_np = ~base_target_match_np
    else:
        baseline_error_np = np.asarray(baseline_error_hard_near_tie_case, dtype=bool)

    improved_match = teacher_target_match_np & (~base_target_match_np)
    worsened_match = base_target_match_np & (~teacher_target_match_np)
    improved_gap = delta_regret_np <= -float(gap_epsilon)
    worsened_gap = delta_regret_np >= float(gap_epsilon)
    improved_miss = delta_miss_np < 0.0
    worsened_miss = delta_miss_np > 0.0

    helpful = improved_match | improved_miss | (action_changed_np & improved_gap & (~worsened_miss))
    harmful = worsened_match | worsened_miss | (action_changed_np & worsened_gap & (~improved_miss))
    helpful = helpful & (~harmful)
    neutral = (~helpful) & (~harmful)

    return pd.DataFrame(
        {
            "helpful": helpful,
            "harmful": harmful,
            "neutral": neutral,
            "recovers_baseline_error": baseline_error_np & improved_match,
            "breaks_baseline_success": (~baseline_error_np) & worsened_match,
        }
    )


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    return float(len(left_set & right_set) / max(len(left_set | right_set), 1))


def signature_overlap_rows(
    frame: pd.DataFrame,
    *,
    subset_col: str,
    group_col: str = "seed",
) -> pd.DataFrame:
    groups = []
    for value, group in frame.groupby(group_col, sort=True):
        signatures = sorted(group.loc[group[subset_col].astype(bool), "stable_positive_signature"].astype(str).unique())
        groups.append((str(value), signatures))
    rows: list[dict[str, object]] = []
    for index, (left_name, left_signatures) in enumerate(groups):
        for right_name, right_signatures in groups[index + 1 :]:
            rows.append(
                {
                    "left_group": left_name,
                    "right_group": right_name,
                    "left_size": len(left_signatures),
                    "right_size": len(right_signatures),
                    "signature_jaccard": jaccard(left_signatures, right_signatures),
                }
            )
    return pd.DataFrame(rows)


def top_fraction_mask(scores: np.ndarray, coverage_pct: float) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError("scores must be a rank-1 array")
    if coverage_pct <= 0.0 or scores.size == 0:
        return np.zeros_like(scores, dtype=bool)
    target = math.ceil(scores.size * float(coverage_pct) / 100.0)
    if target <= 0:
        return np.zeros_like(scores, dtype=bool)
    if target >= scores.size:
        return np.ones_like(scores, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    keep = order[:target]
    mask = np.zeros_like(scores, dtype=bool)
    mask[keep] = True
    return mask


def safe_rate(series: pd.Series | np.ndarray | Sequence[bool]) -> float:
    if isinstance(series, pd.Series):
        if len(series) == 0:
            return 0.0
        return float(series.mean())
    values = np.asarray(series)
    if values.size == 0:
        return 0.0
    return float(values.mean())


def decision_augmented_features(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> torch.Tensor:
    margin = torch.as_tensor(metadata["base_model_margin"].to_numpy(copy=True), dtype=torch.float32)[:, None]
    regime = margin_regime_features(metadata)
    return torch.cat([cache["decision_features"].float(), margin, regime], dim=1)


def margin_regime_features(metadata: pd.DataFrame) -> torch.Tensor:
    frame = metadata[["best_candidate_slack_ratio", "packet_count", "mean_queue", "max_tree_depth"]].copy()
    frame["packet_count"] = frame["packet_count"] / 8.0
    frame["mean_queue"] = frame["mean_queue"] / 10.0
    frame["max_tree_depth"] = frame["max_tree_depth"] / 6.0
    return torch.as_tensor(frame.to_numpy(copy=True), dtype=torch.float32)


def margin_only_features(metadata: pd.DataFrame) -> torch.Tensor:
    return torch.as_tensor(metadata[["base_model_margin"]].to_numpy(copy=True), dtype=torch.float32)


def top2_candidate_indices(scores: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked = scores.masked_fill(~valid_mask, -1e9)
    topk = masked.topk(k=min(2, masked.size(1)), dim=-1)
    first = topk.indices[:, 0]
    second = topk.indices[:, 1] if topk.indices.size(1) > 1 else topk.indices[:, 0]
    return first, second


def candidate_pair_features(cache: dict[str, torch.Tensor], metadata: pd.DataFrame) -> torch.Tensor:
    valid_mask = cache["candidate_mask"].bool()
    first, second = top2_candidate_indices(cache["base_selection_scores"], valid_mask)
    batch_index = torch.arange(cache["candidate_features"].size(0))
    first_features = cache["candidate_features"][batch_index, first].float()
    second_features = cache["candidate_features"][batch_index, second].float()
    score_features = torch.stack(
        [
            cache["base_selection_scores"][batch_index, first].float(),
            cache["base_selection_scores"][batch_index, second].float(),
            cache["candidate_cost_to_go"][batch_index, first].float(),
            cache["candidate_cost_to_go"][batch_index, second].float(),
            cache["candidate_on_time"][batch_index, first].float(),
            cache["candidate_on_time"][batch_index, second].float(),
        ],
        dim=1,
    )
    return torch.cat(
        [
            cache["decision_features"].float(),
            first_features,
            second_features,
            first_features - second_features,
            score_features,
            margin_only_features(metadata),
            margin_regime_features(metadata),
        ],
        dim=1,
    )
