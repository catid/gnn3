#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank-decisions-csv", required=True)
    parser.add_argument("--train-seed", type=int, default=314)
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=[315, 316])
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round12_positive_mining",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _regime_signature(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["depth_load_regime"].astype(str)
        + "|"
        + frame["slack_band"].astype(str)
        + "|"
        + frame["packet_band"].astype(str)
        + "|"
        + frame["gap_bucket"].astype(str)
    )


def _mask_definitions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    helpful = frame["best_safe_teacher_helpful"].astype(bool)
    positive_gain = frame["best_safe_teacher_gain"].to_numpy(copy=True) >= 0.25
    support2 = frame["committee_support"].to_numpy(copy=True) >= 2
    return {
        "strict_v2": frame["stable_positive_v2_case"],
        "committee_v2": frame["stable_positive_v2_committee_case"],
        "high_headroom_union": frame["high_headroom_near_tie_case"] & helpful & positive_gain,
        "baseline_error_union": frame["baseline_error_hard_near_tie_case"] & helpful & positive_gain,
        "expanded_union": (
            frame["high_headroom_near_tie_case"]
            | frame["baseline_error_hard_near_tie_case"]
            | frame["stable_positive_v2_case"]
        )
        & helpful
        & positive_gain,
        "committee_expanded": (
            (
                frame["high_headroom_near_tie_case"]
                | frame["baseline_error_hard_near_tie_case"]
                | frame["stable_positive_v2_case"]
            )
            & helpful
            & positive_gain
            & support2
        ),
    }


def _selector_matches(train_frame: pd.DataFrame, heldout_frame: pd.DataFrame) -> dict[str, pd.Series]:
    selectors = {
        "signature_fine": set(train_frame["signature_fine"].astype(str)),
        "signature_coarse": set(train_frame["signature_coarse"].astype(str)),
        "regime_signature": set(_regime_signature(train_frame).astype(str)),
    }
    heldout_regime = _regime_signature(heldout_frame).astype(str)
    return {
        "signature_fine": heldout_frame["signature_fine"].astype(str).isin(selectors["signature_fine"]),
        "signature_coarse": heldout_frame["signature_coarse"].astype(str).isin(selectors["signature_coarse"]),
        "regime_signature": heldout_regime.isin(selectors["regime_signature"]),
        "coarse_plus_risk": heldout_frame["signature_coarse"].astype(str).isin(selectors["signature_coarse"])
        & (
            heldout_frame["high_headroom_near_tie_case"]
            | heldout_frame["baseline_error_hard_near_tie_case"]
            | (heldout_frame["base_model_margin"] <= float(train_frame["base_model_margin"].quantile(0.25)))
        ),
    }


def _summary_row(
    *,
    manifest: str,
    match_mode: str,
    train_count: int,
    heldout: pd.DataFrame,
    selected: pd.Series,
) -> dict[str, object]:
    heldout_positive = heldout["stable_positive_v2_case"].astype(bool)
    heldout_harmful = heldout["harmful_teacher_bank_case"].astype(bool)
    selected = selected.astype(bool)
    selected_count = int(selected.sum())
    positive_count = int(heldout_positive.sum())
    overlap = selected & heldout_positive
    harmful = selected & heldout_harmful
    return {
        "manifest": manifest,
        "match_mode": match_mode,
        "train_manifest_count": train_count,
        "heldout_decisions": len(heldout),
        "selected_count": selected_count,
        "heldout_coverage": float(selected.mean()) if len(heldout) else 0.0,
        "stable_positive_precision": float(overlap.sum() / max(selected_count, 1)),
        "stable_positive_recall": float(overlap.sum() / max(positive_count, 1)),
        "harmful_selection_rate": float(harmful.sum() / max(selected_count, 1)),
        "selected_mean_teacher_gain": float(
            heldout.loc[selected, "best_safe_teacher_gain"].mean() if selected_count else 0.0
        ),
        "selected_hard_near_tie_share": float(
            heldout.loc[selected, "hard_near_tie_intersection_case"].mean() if selected_count else 0.0
        ),
    }


def _plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot_precision = summary_df.pivot(index="manifest", columns="match_mode", values="stable_positive_precision").fillna(0.0)
    pivot_recall = summary_df.pivot(index="manifest", columns="match_mode", values="stable_positive_recall").fillna(0.0)
    pivot_precision.plot(kind="bar", ax=axes[0], rot=25)
    pivot_recall.plot(kind="bar", ax=axes[1], rot=25)
    axes[0].set_title("Held-out Stable-Positive Precision")
    axes[1].set_title("Held-out Stable-Positive Recall")
    axes[0].set_ylabel("Precision")
    axes[1].set_ylabel("Recall")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.teacher_bank_decisions_csv)
    frame["best_safe_teacher_helpful"] = frame["best_safe_teacher_helpful"].fillna(False).astype(bool)

    train = frame.loc[frame["seed"] == args.train_seed].copy()
    heldout = frame.loc[frame["seed"].isin(args.heldout_seeds)].copy()

    manifest_masks = _mask_definitions(train)
    manifest_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for manifest_name, manifest_mask in manifest_masks.items():
        manifest_frame = train.loc[manifest_mask].copy()
        if manifest_frame.empty:
            continue
        manifest_frame["manifest"] = manifest_name
        manifest_frame["mining_weight"] = np.where(
            manifest_frame["stable_positive_v2_case"],
            8.0,
            np.where(
                manifest_frame["baseline_error_hard_near_tie_case"] | manifest_frame["high_headroom_near_tie_case"],
                4.0,
                2.0,
            ),
        )
        manifest_rows.append(manifest_frame)
        selector_map = _selector_matches(manifest_frame, heldout)
        for match_mode, selected in selector_map.items():
            summary_rows.append(
                _summary_row(
                    manifest=manifest_name,
                    match_mode=match_mode,
                    train_count=len(manifest_frame),
                    heldout=heldout,
                    selected=selected,
                )
            )

    manifest_df = pd.concat(manifest_rows, ignore_index=True) if manifest_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    manifest_df.to_csv(output_prefix.with_name(output_prefix.name + "_manifest.csv"), index=False)
    summary_df.to_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), index=False)
    _plot(summary_df, output_prefix.with_name(output_prefix.name + "_summary.png"))
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "train_seed": args.train_seed,
                "heldout_seeds": args.heldout_seeds,
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
