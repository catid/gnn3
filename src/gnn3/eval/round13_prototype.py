from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from gnn3.eval.precision_correction import top_fraction_mask

KEY_COLS = ["suite", "episode_index", "decision_index"]
ROUND13_FRONTIER_BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50)

_TEACHER_BOOL_COLS = (
    "stable_positive_v2_case",
    "stable_positive_v2_committee_case",
    "stable_positive_v2_strict_case",
    "unstable_positive_v2_case",
    "harmful_teacher_bank_case",
    "best_safe_teacher_target_match",
    "best_safe_teacher_helpful",
)
_TEACHER_FLOAT_COLS = (
    "best_safe_teacher_gain",
    "best_safe_teacher_delta_regret",
    "best_safe_teacher_delta_miss",
    "committee_mean_gain",
)
_TEACHER_INT_COLS = ("committee_support",)
_TEACHER_OBJECT_COLS = ("best_safe_teacher_name", "best_safe_teacher_action", "committee_action")
_TEACHER_MERGE_COLS = [
    *KEY_COLS,
    *_TEACHER_BOOL_COLS,
    *_TEACHER_FLOAT_COLS,
    *_TEACHER_INT_COLS,
    *_TEACHER_OBJECT_COLS,
]


def load_teacher_bank_frame(path: str | Path) -> pd.DataFrame:
    teacher_bank = pd.read_csv(path)
    available = [col for col in _TEACHER_MERGE_COLS if col in teacher_bank.columns]
    return teacher_bank[available].drop_duplicates(KEY_COLS)


def attach_teacher_bank(meta: pd.DataFrame, teacher_bank: pd.DataFrame) -> pd.DataFrame:
    merged = meta.merge(teacher_bank, on=KEY_COLS, how="left")
    for col in _TEACHER_BOOL_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)
    for col in _TEACHER_FLOAT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0).astype(float)
    for col in _TEACHER_INT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    for col in _TEACHER_OBJECT_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna("").astype(str)
    return merged


def load_heldout_metadata(
    metadata_paths: Iterable[str | Path],
    *,
    teacher_bank_decisions_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    teacher_bank = load_teacher_bank_frame(teacher_bank_decisions_csv)
    metadata_by_split: dict[str, pd.DataFrame] = {}
    for path in metadata_paths:
        path = Path(path)
        split = path.stem
        if split.endswith("_metadata"):
            split = split[: -len("_metadata")]
        metadata = pd.read_csv(path)
        metadata = attach_teacher_bank(metadata, teacher_bank)
        metadata.attrs["metadata_path"] = str(path)
        metadata.attrs["split"] = split
        metadata_by_split[split] = metadata
    if not metadata_by_split:
        raise ValueError("No metadata paths were provided.")
    return metadata_by_split


def default_slice_map(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "overall": pd.Series([True] * len(frame), index=frame.index),
        "stable_positive_v2": frame["stable_positive_v2_case"],
        "stable_positive_v2_committee": frame["stable_positive_v2_committee_case"],
        "hard_near_tie": frame["hard_near_tie_intersection_case"],
        "stable_near_tie": frame["stable_near_tie_case"],
        "high_headroom_near_tie": frame["high_headroom_near_tie_case"],
        "baseline_error_near_tie": frame["baseline_error_hard_near_tie_case"],
        "large_gap_control": frame["large_gap_hard_feasible_case"],
    }


def load_unique_scores(decisions_csv: str | Path, *, variant: str) -> pd.DataFrame:
    frame = pd.read_csv(decisions_csv, usecols=[*KEY_COLS, "variant", "split", "score"])
    frame = frame.loc[frame["variant"] == variant].copy()
    if frame.empty:
        raise ValueError(f"No rows found for variant {variant!r} in {decisions_csv}.")
    grouped = frame.groupby(["split", *KEY_COLS], as_index=False).agg(score_nunique=("score", "nunique"))
    bad = grouped.loc[grouped["score_nunique"] != 1]
    if not bad.empty:
        raise ValueError(f"Non-unique scores detected for variant {variant!r} in {decisions_csv}.")
    deduped = frame.drop_duplicates(["split", *KEY_COLS]).reset_index(drop=True)
    return deduped


def merge_scores_by_split(
    metadata_by_split: Mapping[str, pd.DataFrame],
    score_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    merged_by_split: dict[str, pd.DataFrame] = {}
    available_splits = set(score_table["split"].astype(str).unique())
    if available_splits == {"heldout"}:
        concat = pd.concat(metadata_by_split.values(), ignore_index=True)
        split_scores = score_table.loc[score_table["split"] == "heldout", [*KEY_COLS, "score"]].copy()
        merged = concat.merge(split_scores, on=KEY_COLS, how="inner")
        if len(merged) != len(concat):
            raise ValueError(
                f"Heldout score merge mismatch: expected {len(concat)} rows, got {len(merged)}."
            )
        merged_by_split["heldout"] = merged
        return merged_by_split
    for split, metadata in metadata_by_split.items():
        candidate_splits = (
            split,
            f"{split}_metadata",
            split.removesuffix("_metadata"),
        )
        matched_name = next((name for name in candidate_splits if name in available_splits), None)
        if matched_name is None:
            raise ValueError(f"No scores found for split {split!r}.")
        split_scores = score_table.loc[score_table["split"] == matched_name, [*KEY_COLS, "score"]].copy()
        if split_scores.empty:
            raise ValueError(f"No scores found for split {split!r}.")
        merged = metadata.merge(split_scores, on=KEY_COLS, how="inner")
        if len(merged) != len(metadata):
            raise ValueError(
                f"Score merge mismatch for split {split!r}: expected {len(metadata)} rows, got {len(merged)}."
            )
        merged_by_split[split] = merged
    return merged_by_split


def _p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, 0.95))


def _selection_rows(
    frame: pd.DataFrame,
    *,
    family: str,
    variant: str,
    split: str,
    budget_pct: float,
    selected: np.ndarray,
) -> pd.DataFrame:
    rows = frame[KEY_COLS].copy()
    rows["family"] = family
    rows["variant"] = variant
    rows["split"] = split
    rows["budget_pct"] = float(budget_pct)
    rows["score"] = frame["score"].to_numpy(copy=True)
    rows["selected"] = selected
    return rows


def summarize_selection(
    frame: pd.DataFrame,
    *,
    family: str,
    variant: str,
    split: str,
    budget_pct: float,
    selected: np.ndarray,
) -> pd.DataFrame:
    base_action = frame["base_predicted_next_hop"].astype(str).to_numpy(copy=True)
    compute_action = frame["compute_predicted_next_hop"].astype(str).to_numpy(copy=True)
    rows: list[dict[str, object]] = []
    for slice_name, mask in default_slice_map(frame).items():
        target = frame.loc[mask].copy()
        idx = target.index.to_numpy()
        selected_target = selected[idx]
        selected_count = int(selected_target.sum())
        selected_denom = max(selected_count, 1)

        stable_positive = target["stable_positive_v2_case"].to_numpy(copy=True).astype(bool)
        harmful = target["harmful_teacher_bank_case"].to_numpy(copy=True).astype(bool)
        recover = target["compute_recovers_baseline_error"].to_numpy(copy=True).astype(bool)
        breaks = target["compute_breaks_baseline_success"].to_numpy(copy=True).astype(bool)
        base_target = target["base_target_match"].to_numpy(copy=True).astype(bool)
        compute_target = target["compute_target_match"].to_numpy(copy=True).astype(bool)

        stable_positive_total = int(stable_positive.sum())
        stable_positive_selected = int(np.logical_and(selected_target, stable_positive).sum())
        harmful_selected = int(np.logical_and(selected_target, harmful).sum())
        false_positive_selected = int(np.logical_and(selected_target, ~stable_positive).sum())
        helpful_false_positive_selected = int(
            np.logical_and(np.logical_and(selected_target, ~stable_positive), recover).sum()
        )

        combined_target = np.where(selected_target, compute_target, base_target)
        delta_regret = np.where(selected_target, target["delta_regret"].to_numpy(copy=True), 0.0)
        delta_miss = np.where(selected_target, target["delta_miss"].to_numpy(copy=True), 0.0)
        selected_delta_regret = target.loc[selected_target, "delta_regret"].to_numpy(copy=True)
        selected_delta_miss = target.loc[selected_target, "delta_miss"].to_numpy(copy=True)
        disagreement = np.logical_and(selected_target, compute_action[idx] != base_action[idx])

        rows.append(
            {
                "family": family,
                "variant": variant,
                "split": split,
                "budget_pct": float(budget_pct),
                "slice": slice_name,
                "decisions": len(target),
                "selected_count": selected_count,
                "coverage": float(selected_target.mean()) if len(target) else 0.0,
                "stable_positive_total": stable_positive_total,
                "stable_positive_selected": stable_positive_selected,
                "stable_positive_recall": (
                    float(stable_positive_selected / max(stable_positive_total, 1))
                    if stable_positive_total
                    else 0.0
                ),
                "defer_precision": float(stable_positive_selected / selected_denom),
                "false_positive_rate": float(false_positive_selected / selected_denom),
                "harmful_selection_rate": float(harmful_selected / selected_denom),
                "false_positive_correction_rate": float(helpful_false_positive_selected / selected_denom),
                "correction_rate": float(np.logical_and(selected_target, recover).mean()) if len(target) else 0.0,
                "new_error_rate": float(np.logical_and(selected_target, breaks).mean()) if len(target) else 0.0,
                "base_target_match": float(base_target.mean()) if len(target) else 0.0,
                "system_target_match": float(combined_target.mean()) if len(target) else 0.0,
                "mean_delta_regret": float(delta_regret.mean()) if len(target) else 0.0,
                "p95_delta_regret": _p95(delta_regret),
                "selected_mean_delta_regret": float(selected_delta_regret.mean()) if selected_delta_regret.size else 0.0,
                "selected_p95_delta_regret": _p95(selected_delta_regret),
                "mean_delta_miss": float(delta_miss.mean()) if len(target) else 0.0,
                "selected_mean_delta_miss": float(selected_delta_miss.mean()) if selected_delta_miss.size else 0.0,
                "selected_disagreement": float(disagreement.mean()) if len(target) else 0.0,
                "score_mean": float(target["score"].mean()) if len(target) else 0.0,
                "score_p95": _p95(target["score"].to_numpy(copy=True)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_variant_scores(
    metadata_by_split: Mapping[str, pd.DataFrame],
    score_table: pd.DataFrame,
    *,
    family: str,
    variant: str,
    budgets: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_by_split = merge_scores_by_split(metadata_by_split, score_table)
    summary_rows: list[pd.DataFrame] = []
    decision_rows: list[pd.DataFrame] = []
    for split, merged in merged_by_split.items():
        scores = merged["score"].to_numpy(copy=True)
        for budget_pct in budgets:
            selected = top_fraction_mask(scores, float(budget_pct)) & (scores > 0.0)
            decision_rows.append(
                _selection_rows(
                    merged,
                    family=family,
                    variant=variant,
                    split=split,
                    budget_pct=float(budget_pct),
                    selected=selected,
                )
            )
            summary_rows.append(
                summarize_selection(
                    merged,
                    family=family,
                    variant=variant,
                    split=split,
                    budget_pct=float(budget_pct),
                    selected=selected,
                )
            )
    return pd.concat(summary_rows, ignore_index=True), pd.concat(decision_rows, ignore_index=True)


def aggregate_split_summary(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby(["family", "variant", "budget_pct", "slice"], as_index=False).agg(
        split_count=("split", "nunique"),
        decisions=("decisions", "sum"),
        selected_count=("selected_count", "sum"),
        stable_positive_total=("stable_positive_total", "sum"),
        stable_positive_selected=("stable_positive_selected", "sum"),
        coverage=("coverage", "mean"),
        stable_positive_recall=("stable_positive_recall", "mean"),
        defer_precision=("defer_precision", "mean"),
        false_positive_rate=("false_positive_rate", "mean"),
        harmful_selection_rate=("harmful_selection_rate", "mean"),
        false_positive_correction_rate=("false_positive_correction_rate", "mean"),
        correction_rate=("correction_rate", "mean"),
        new_error_rate=("new_error_rate", "mean"),
        base_target_match=("base_target_match", "mean"),
        system_target_match=("system_target_match", "mean"),
        mean_delta_regret=("mean_delta_regret", "mean"),
        p95_delta_regret=("p95_delta_regret", "mean"),
        selected_mean_delta_regret=("selected_mean_delta_regret", "mean"),
        selected_p95_delta_regret=("selected_p95_delta_regret", "mean"),
        mean_delta_miss=("mean_delta_miss", "mean"),
        selected_mean_delta_miss=("selected_mean_delta_miss", "mean"),
        selected_disagreement=("selected_disagreement", "mean"),
        score_mean=("score_mean", "mean"),
        score_p95=("score_p95", "mean"),
    )
    return grouped


def load_and_evaluate_variant(
    metadata_by_split: Mapping[str, pd.DataFrame],
    *,
    decisions_csv: str | Path,
    variant: str,
    family: str,
    budgets: Sequence[float] = ROUND13_FRONTIER_BUDGETS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = load_unique_scores(decisions_csv, variant=variant)
    per_split, decisions = evaluate_variant_scores(
        metadata_by_split,
        scores,
        family=family,
        variant=variant,
        budgets=budgets,
    )
    aggregate = aggregate_split_summary(per_split)
    return per_split, aggregate, decisions
