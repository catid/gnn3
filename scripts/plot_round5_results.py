#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("artifacts/experiments")
PLOTS = Path("reports/plots")


def _summary_frame(pairs: list[tuple[int, str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed, baseline_name, variant_name in pairs:
        for label, experiment_name in (("Multiheavy", baseline_name), ("TailSelect", variant_name)):
            summary = json.loads((ROOT / experiment_name / "summary.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "seed": seed,
                    "variant": label,
                    "experiment": experiment_name,
                    "test_next_hop_accuracy": summary["test"]["next_hop_accuracy"],
                    "average_regret": summary["test_rollout"]["average_regret"],
                    "p95_regret": summary["test_rollout"]["p95_regret"],
                    "deadline_miss_rate": summary["test_rollout"]["deadline_miss_rate"],
                    "gpu_hours": summary["gpu_hours"],
                }
            )

    frame = pd.DataFrame(rows)
    means = (
        frame.groupby("variant", as_index=False)[
            ["test_next_hop_accuracy", "average_regret", "p95_regret", "deadline_miss_rate", "gpu_hours"]
        ]
        .mean()
        .assign(seed="mean")
    )
    means["experiment"] = means["variant"].map({"Multiheavy": "Multiheavy-mean", "TailSelect": "TailSelect-mean"})
    return pd.concat([frame, means[frame.columns]], ignore_index=True)


def _plot_compare(frame: pd.DataFrame, output_name: str) -> None:
    plot_df = frame[frame["seed"] != "mean"].copy()
    baseline = plot_df[plot_df["variant"] == "Multiheavy"].reset_index(drop=True)
    variant = plot_df[plot_df["variant"] == "TailSelect"].reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    x = list(range(len(baseline)))
    width = 0.36
    labels = [str(seed) for seed in baseline["seed"]]

    axes[0].bar([idx - width / 2 for idx in x], baseline["test_next_hop_accuracy"], width=width, color="#1f77b4")
    axes[0].bar([idx + width / 2 for idx in x], variant["test_next_hop_accuracy"], width=width, color="#ff7f0e")
    axes[0].set_title("Test Next-Hop Accuracy")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)

    axes[1].bar([idx - width / 2 for idx in x], baseline["average_regret"], width=width, color="#1f77b4")
    axes[1].bar([idx + width / 2 for idx in x], variant["average_regret"], width=width, color="#ff7f0e")
    axes[1].set_title("Average Regret")
    axes[1].set_xticks(x, labels)

    axes[2].bar([idx - width / 2 for idx in x], baseline["p95_regret"], width=width, color="#1f77b4")
    axes[2].bar([idx + width / 2 for idx in x], variant["p95_regret"], width=width, color="#ff7f0e")
    axes[2].set_title("p95 Regret")
    axes[2].set_xticks(x, labels)

    axes[3].bar([idx - width / 2 for idx in x], baseline["deadline_miss_rate"], width=width, color="#1f77b4")
    axes[3].bar([idx + width / 2 for idx in x], variant["deadline_miss_rate"], width=width, color="#ff7f0e")
    axes[3].set_title("Deadline Miss Rate")
    axes[3].set_xticks(x, labels)
    axes[3].set_ylim(0.0, 1.05)

    axes[0].legend(["Multiheavy", "TailSelect"], loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS / output_name, dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    compare = _summary_frame(
        [
            (311, "e3_memory_hubs_rsm_round4_multiheavy_seed311", "e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed311"),
            (312, "e3_memory_hubs_rsm_round4_multiheavy_seed312", "e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed312"),
            (313, "e3_memory_hubs_rsm_round4_multiheavy_seed313", "e3_memory_hubs_rsm_round5_multiheavy_tail_select_seed313"),
        ]
    )
    compare.to_csv(PLOTS / "round5_multiheavy_tail_select_vs_multiheavy.csv", index=False)
    _plot_compare(compare, "round5_multiheavy_tail_select_vs_multiheavy.png")


if __name__ == "__main__":
    main()
