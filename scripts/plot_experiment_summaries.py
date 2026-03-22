#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/experiments",
        help="Directory containing experiment subdirectories with summary.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/plots",
        help="Directory to write the summary CSV and PNG plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": summary_path.parent.name,
                "bucket": summary["bucket"],
                "gpu_hours": summary["gpu_hours"],
                "test_next_hop_accuracy": summary["test"]["next_hop_accuracy"],
                "value_mae": summary["test"].get("value_mae", 0.0),
                "value_rmse": summary["test"].get("value_rmse", 0.0),
                "rollout_solved_rate": summary["test_rollout"]["solved_rate"],
                "rollout_next_hop_accuracy": summary["test_rollout"]["next_hop_accuracy"],
                "average_regret": summary["test_rollout"]["average_regret"],
                "p95_regret": summary["test_rollout"].get("p95_regret", 0.0),
                "worst_regret": summary["test_rollout"].get("worst_regret", 0.0),
                "deadline_violations": summary["test_rollout"]["average_deadline_violations"],
                "deadline_miss_rate": summary["test_rollout"].get("deadline_miss_rate", 0.0),
            }
        )

    if not rows:
        raise SystemExit("No experiment summaries found.")

    df = pd.DataFrame(rows).sort_values(["bucket", "experiment"]).reset_index(drop=True)
    df.to_csv(output_dir / "experiment_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].bar(df["experiment"], df["test_next_hop_accuracy"], color=["#1f77b4" if b == "exploit" else "#ff7f0e" for b in df["bucket"]])
    axes[0].set_title("Test Next-Hop Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(df["experiment"], df["rollout_solved_rate"], color=["#1f77b4" if b == "exploit" else "#ff7f0e" for b in df["bucket"]])
    axes[1].set_title("Rollout Solved Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(df["experiment"], df["average_regret"], color=["#1f77b4" if b == "exploit" else "#ff7f0e" for b in df["bucket"]])
    axes[2].set_title("Average Regret")
    axes[2].tick_params(axis="x", rotation=25)

    axes[3].bar(df["experiment"], df["deadline_miss_rate"], color=["#1f77b4" if b == "exploit" else "#ff7f0e" for b in df["bucket"]])
    axes[3].set_title("Deadline Miss Rate")
    axes[3].set_ylim(0.0, 1.05)
    axes[3].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(output_dir / "experiment_summary.png", dpi=160)


if __name__ == "__main__":
    main()
