#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset, collate_decisions
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _move_batch, _resolve_device

PLOTS_DIR = Path("reports/plots")
ARTIFACTS_DIR = Path("artifacts/experiments")


def _summary(experiment: str) -> dict:
    path = ARTIFACTS_DIR / experiment / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed, experiment in (
        (311, "e3_memory_hubs_rsm_round4_seed311"),
        (312, "e3_memory_hubs_rsm_round4_seed312"),
        (313, "e3_memory_hubs_rsm_round4_seed313"),
    ):
        summary = _summary(experiment)
        rows.append(
            {
                "seed": seed,
                "experiment": experiment,
                "test_next_hop_accuracy": summary["test"]["next_hop_accuracy"],
                "average_regret": summary["test_rollout"]["average_regret"],
                "p95_regret": summary["test_rollout"]["p95_regret"],
                "deadline_miss_rate": summary["test_rollout"]["deadline_miss_rate"],
                "gpu_hours": summary["gpu_hours"],
            }
        )
    frame = pd.DataFrame(rows)
    mean_row = {
        "seed": "mean",
        "experiment": "E3-round4-baseline-mean",
        "test_next_hop_accuracy": frame["test_next_hop_accuracy"].mean(),
        "average_regret": frame["average_regret"].mean(),
        "p95_regret": frame["p95_regret"].mean(),
        "deadline_miss_rate": frame["deadline_miss_rate"].mean(),
        "gpu_hours": frame["gpu_hours"].sum(),
    }
    return pd.concat([frame, pd.DataFrame([mean_row])], ignore_index=True)


def _plot_baseline(frame: pd.DataFrame) -> None:
    plot_df = frame[frame["seed"] != "mean"].copy()
    plot_df["seed"] = plot_df["seed"].astype(str)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].bar(plot_df["seed"], plot_df["test_next_hop_accuracy"], color="#1f77b4")
    axes[0].set_title("E3 Test Next-Hop")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(plot_df["seed"], plot_df["average_regret"], color="#d62728")
    axes[1].set_title("E3 Average Regret")

    axes[2].bar(plot_df["seed"], plot_df["p95_regret"], color="#ff7f0e")
    axes[2].set_title("E3 p95 Regret")

    axes[3].bar(plot_df["seed"], plot_df["deadline_miss_rate"], color="#9467bd")
    axes[3].set_title("E3 Deadline Miss Rate")
    axes[3].set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "round4_e3_matched_baseline.png", dpi=160)
    plt.close(fig)


def _collect_calibration_predictions(
    *,
    model_label: str,
    model_config_path: str,
    eval_config_path: str,
    checkpoint_path: Path,
) -> pd.DataFrame:
    model_config = load_experiment_config(model_config_path)
    eval_config = load_experiment_config(eval_config_path)
    device = _resolve_device(model_config.train.device)

    test_hidden_cfg = hidden_corridor_config_for_split(eval_config.benchmark, "test")
    dataset = HiddenCorridorDecisionDataset(
        config=test_hidden_cfg,
        num_episodes=eval_config.benchmark.test_episodes,
        curriculum_levels=eval_config.benchmark.curriculum_levels,
    )
    loader = DataLoader(
        dataset,
        batch_size=model_config.train.eval_batch_size,
        shuffle=False,
        collate_fn=collate_decisions,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(model_config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, float | int | str]] = []
    median_index = len(model_config.model.quantile_levels) // 2
    for batch in loader:
        batch = _move_batch(batch, device)
        output = model(batch)
        if output["candidate_on_time_logits"] is None:
            continue
        valid_mask = batch["candidate_mask"] & batch["node_mask"]
        if not valid_mask.any():
            continue
        prob = torch.sigmoid(output["candidate_on_time_logits"][valid_mask]).detach().cpu().numpy()
        on_time = batch["candidate_on_time"][valid_mask].detach().cpu().numpy()
        slack_pred = output["candidate_slack"][valid_mask].detach().cpu().numpy()
        slack_true = batch["candidate_slack"][valid_mask].detach().cpu().numpy()
        cost_pred = output["candidate_cost_quantiles"][valid_mask][:, median_index].detach().cpu().numpy()
        cost_true = batch["candidate_cost_to_go"][valid_mask].detach().cpu().numpy()

        for idx in range(len(prob)):
            rows.append(
                {
                    "model": model_label,
                    "on_time_prob": float(prob[idx]),
                    "on_time_true": float(on_time[idx]),
                    "slack_pred": float(slack_pred[idx]),
                    "slack_true": float(slack_true[idx]),
                    "cost_pred_q50": float(cost_pred[idx]),
                    "cost_true": float(cost_true[idx]),
                }
            )
    return pd.DataFrame(rows)


def _binned_curve(
    frame: pd.DataFrame,
    *,
    model_col: str,
    pred_col: str,
    target_col: str,
    curve_name: str,
    bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    if frame.empty:
        return pd.DataFrame(columns=["model", "curve", "bin_index", "count", "mean_pred", "mean_obs"])

    for model_name, model_frame in frame.groupby(model_col):
        work = model_frame[[pred_col, target_col]].copy()
        work = work.sort_values(pred_col).reset_index(drop=True)
        work["bin_index"] = pd.qcut(work.index, q=min(bins, len(work)), labels=False, duplicates="drop")
        for bin_index, bin_frame in work.groupby("bin_index"):
            rows.append(
                {
                    "model": model_name,
                    "curve": curve_name,
                    "bin_index": int(bin_index),
                    "count": len(bin_frame),
                    "mean_pred": float(bin_frame[pred_col].mean()),
                    "mean_obs": float(bin_frame[target_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def _plot_calibration(calibration_frame: pd.DataFrame) -> None:
    on_time = calibration_frame[calibration_frame["curve"] == "on_time"]
    slack = calibration_frame[calibration_frame["curve"] == "slack"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
    for model_name, model_frame in on_time.groupby("model"):
        axes[0].plot(model_frame["mean_pred"], model_frame["mean_obs"], marker="o", label=model_name)
    axes[0].set_title("On-Time Reliability")
    axes[0].set_xlabel("Predicted on-time probability")
    axes[0].set_ylabel("Observed on-time frequency")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend()

    lower = min(float(slack["mean_pred"].min()), float(slack["mean_obs"].min()))
    upper = max(float(slack["mean_pred"].max()), float(slack["mean_obs"].max()))
    axes[1].plot([lower, upper], [lower, upper], linestyle="--", color="#666666", linewidth=1)
    for model_name, model_frame in slack.groupby("model"):
        axes[1].plot(model_frame["mean_pred"], model_frame["mean_obs"], marker="o", label=model_name)
    axes[1].set_title("Slack Calibration")
    axes[1].set_xlabel("Predicted slack")
    axes[1].set_ylabel("Observed slack")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "round4_calibration_curves.png", dpi=160)
    plt.close(fig)


def _plot_deadline_compare(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(frame["variant"], frame["test_next_hop_accuracy"], color="#1f77b4")
    axes[0].set_title("Seed311 Test Accuracy")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(frame["variant"], frame["p95_regret"], color="#d62728")
    axes[1].set_title("Seed311 p95 Regret")

    axes[2].bar(frame["variant"], frame["deadline_miss_rate"], color="#9467bd")
    axes[2].set_title("Seed311 Deadline Miss Rate")
    axes[2].set_ylim(0.0, 1.0)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "round4_deadline_p95_compare.png", dpi=160)
    plt.close(fig)


def _portfolio_frame() -> pd.DataFrame:
    rows = [
        {"experiment": "A0-oracle-audit", "bucket": "exploit", "gpu_hours": 0.0, "status": "completed"},
        {"experiment": "A1-E3-baseline", "bucket": "exploit", "gpu_hours": 0.36592718925740986, "status": "completed"},
        {"experiment": "A2-deadline-head", "bucket": "exploit", "gpu_hours": 0.11770952933364444, "status": "completed"},
        {"experiment": "A4-verifier-refine", "bucket": "exploit", "gpu_hours": 0.07911762999163734, "status": "completed"},
        {"experiment": "A3-multiheavy-followup", "bucket": "exploit", "gpu_hours": 0.22461334804693855, "status": "completed"},
        {"experiment": "A3-path-reranker", "bucket": "exploit", "gpu_hours": 0.10660791357358296, "status": "completed"},
        {"experiment": "A3-path-reranker-seed313", "bucket": "exploit", "gpu_hours": 0.028074771033393637, "status": "killed-early"},
        {"experiment": "A3-combined-multiheavy-reranker", "bucket": "exploit", "gpu_hours": 0.18660099433528052, "status": "completed"},
        {"experiment": "A2-combined-deadline-head", "bucket": "exploit", "gpu_hours": 0.06111519972483317, "status": "completed"},
        {"experiment": "A3-combined-ood-stress", "bucket": "exploit", "gpu_hours": 0.37833333333333335, "status": "completed"},
        {"experiment": "A2-multiheavy-deadline-head-scout", "bucket": "exploit", "gpu_hours": 0.1422042096985711, "status": "completed"},
        {"experiment": "B1-hazard-memory", "bucket": "explore", "gpu_hours": 0.26461411317189536, "status": "completed"},
        {"experiment": "B2-gated-reranker-seed311", "bucket": "explore", "gpu_hours": 0.060779284636179605, "status": "completed"},
        {"experiment": "B2-gated-reranker-seed312", "bucket": "explore", "gpu_hours": 0.089862193663915, "status": "completed"},
        {"experiment": "B2-gated-reranker-seed313", "bucket": "explore", "gpu_hours": 0.06544615877999199, "status": "completed"},
        {"experiment": "B2-gated-reranker-ood-seed311", "bucket": "explore", "gpu_hours": 0.0208, "status": "completed"},
        {"experiment": "B2-gated-reranker-ood-seed312", "bucket": "explore", "gpu_hours": 0.0258, "status": "completed"},
    ]
    frame = pd.DataFrame(rows)
    total = float(frame["gpu_hours"].sum())
    frame["portfolio_share"] = frame["gpu_hours"] / total if total else 0.0
    return frame


def _plot_portfolio(frame: pd.DataFrame) -> None:
    totals = frame.groupby("bucket", as_index=False)["gpu_hours"].sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#1f77b4" if bucket == "exploit" else "#ff7f0e" for bucket in frame["bucket"]]
    axes[0].bar(frame["experiment"], frame["gpu_hours"], color=colors)
    axes[0].set_title("Round4 GPU-Hours by Experiment")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].pie(
        totals["gpu_hours"],
        labels=totals["bucket"],
        autopct=lambda pct: f"{pct:.1f}%",
        colors=["#1f77b4", "#ff7f0e"],
        startangle=90,
    )
    axes[1].set_title("Round4 Exploit / Explore Split")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "portfolio_usage_round4.png", dpi=160)
    plt.close(fig)


def _variant_compare_frame(pairs: list[tuple[int, str, str]]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed, baseline_exp, variant_exp in pairs:
        for variant, experiment in (("E3", baseline_exp), ("variant", variant_exp)):
            summary = _summary(experiment)
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "experiment": experiment,
                    "test_next_hop_accuracy": summary["test"]["next_hop_accuracy"],
                    "average_regret": summary["test_rollout"]["average_regret"],
                    "p95_regret": summary["test_rollout"]["p95_regret"],
                    "deadline_miss_rate": summary["test_rollout"]["deadline_miss_rate"],
                    "gpu_hours": summary["gpu_hours"],
                }
            )
    frame = pd.DataFrame(rows)
    mean_rows: list[dict[str, float | int | str]] = []
    for variant, variant_frame in frame.groupby("variant"):
        mean_rows.append(
            {
                "seed": "mean",
                "variant": variant,
                "experiment": f"{variant}-mean",
                "test_next_hop_accuracy": variant_frame["test_next_hop_accuracy"].mean(),
                "average_regret": variant_frame["average_regret"].mean(),
                "p95_regret": variant_frame["p95_regret"].mean(),
                "deadline_miss_rate": variant_frame["deadline_miss_rate"].mean(),
                "gpu_hours": variant_frame["gpu_hours"].sum(),
            }
        )
    return pd.concat([frame, pd.DataFrame(mean_rows)], ignore_index=True)


def _plot_variant_compare(frame: pd.DataFrame, *, title_prefix: str, output_name: str) -> None:
    plot_df = frame.copy()
    plot_df["seed"] = plot_df["seed"].astype(str)
    seeds = list(dict.fromkeys(plot_df["seed"]))
    x = range(len(seeds))
    width = 0.35

    variant_names = list(dict.fromkeys(plot_df["variant"]))
    baseline_name = variant_names[0]
    variant_name = variant_names[1]
    baseline = plot_df[plot_df["variant"] == baseline_name].set_index("seed").loc[seeds]
    variant = plot_df[plot_df["variant"] == variant_name].set_index("seed").loc[seeds]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar([idx - width / 2 for idx in x], baseline["average_regret"], width=width, label=baseline_name, color="#1f77b4")
    axes[0].bar([idx + width / 2 for idx in x], variant["average_regret"], width=width, label=variant_name, color="#ff7f0e")
    axes[0].set_title(f"{title_prefix} Regret")
    axes[0].set_xticks(list(x), seeds)

    axes[1].bar([idx - width / 2 for idx in x], baseline["p95_regret"], width=width, color="#1f77b4")
    axes[1].bar([idx + width / 2 for idx in x], variant["p95_regret"], width=width, color="#ff7f0e")
    axes[1].set_title(f"{title_prefix} p95")
    axes[1].set_xticks(list(x), seeds)

    axes[2].bar([idx - width / 2 for idx in x], baseline["deadline_miss_rate"], width=width, color="#1f77b4")
    axes[2].bar([idx + width / 2 for idx in x], variant["deadline_miss_rate"], width=width, color="#ff7f0e")
    axes[2].set_title(f"{title_prefix} Deadline Miss")
    axes[2].set_xticks(list(x), seeds)
    axes[2].set_ylim(0.0, 1.0)
    axes[0].legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close(fig)


def _suite_label(experiment_name: str) -> str:
    if "branching3" in experiment_name:
        return "branching3"
    if "deeper_packets6" in experiment_name:
        return "deeper_packets6"
    if "heavy_dynamic" in experiment_name:
        return "heavy_dynamic"
    return experiment_name


def _ood_compare_frame_from_specs(
    specs: list[tuple[int, str, str]],
    *,
    baseline_label: str,
    variant_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for seed, baseline_name, variant_name in specs:
        baseline_path = PLOTS_DIR / baseline_name
        variant_path = PLOTS_DIR / variant_name
        if not baseline_path.exists() or not variant_path.exists():
            continue
        baseline_df = pd.read_csv(baseline_path).copy()
        variant_df = pd.read_csv(variant_path).copy()
        for label, frame in ((baseline_label, baseline_df), (variant_label, variant_df)):
            frame["suite"] = frame["experiment"].map(_suite_label)
            for _, row in frame.iterrows():
                rows.append(
                    {
                        "seed": seed,
                        "suite": row["suite"],
                        "variant": label,
                        "average_regret": row["average_regret"],
                        "p95_regret": row["p95_regret"],
                        "deadline_miss_rate": row["deadline_miss_rate"],
                        "rollout_next_hop_accuracy": row["rollout_next_hop_accuracy"],
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    mean_rows: list[dict[str, float | int | str]] = []
    for (suite, variant), suite_frame in frame.groupby(["suite", "variant"]):
        mean_rows.append(
            {
                "seed": "mean",
                "suite": suite,
                "variant": variant,
                "average_regret": suite_frame["average_regret"].mean(),
                "p95_regret": suite_frame["p95_regret"].mean(),
                "deadline_miss_rate": suite_frame["deadline_miss_rate"].mean(),
                "rollout_next_hop_accuracy": suite_frame["rollout_next_hop_accuracy"].mean(),
            }
        )
    for variant, variant_frame in frame.groupby("variant"):
        mean_rows.append(
            {
                "seed": "mean",
                "suite": "overall",
                "variant": variant,
                "average_regret": variant_frame["average_regret"].mean(),
                "p95_regret": variant_frame["p95_regret"].mean(),
                "deadline_miss_rate": variant_frame["deadline_miss_rate"].mean(),
                "rollout_next_hop_accuracy": variant_frame["rollout_next_hop_accuracy"].mean(),
            }
        )
    return pd.concat([frame, pd.DataFrame(mean_rows)], ignore_index=True)


def _ood_compare_frame() -> pd.DataFrame:
    return _ood_compare_frame_from_specs(
        [
            (311, "round4_multiheavy_ood_seed311.csv", "round4_multiheavy_path_reranker_ood_seed311.csv"),
            (312, "round4_multiheavy_ood_seed312.csv", "round4_multiheavy_path_reranker_ood_seed312.csv"),
            (313, "round4_multiheavy_ood_seed313.csv", "round4_multiheavy_path_reranker_ood_seed313.csv"),
        ],
        baseline_label="Multiheavy",
        variant_label="Multiheavy+PathReranker",
    )


def _plot_ood_compare(frame: pd.DataFrame, *, output_name: str) -> None:
    plot_df = frame[frame["seed"] == "mean"].copy()
    if plot_df.empty:
        return
    plot_df = plot_df[plot_df["suite"] != "overall"].copy()
    suites = list(dict.fromkeys(plot_df["suite"]))
    x = range(len(suites))
    width = 0.35
    variant_names = list(dict.fromkeys(plot_df["variant"]))
    baseline_name = variant_names[0]
    variant_name = variant_names[1]
    baseline = plot_df[plot_df["variant"] == baseline_name].set_index("suite").loc[suites]
    variant = plot_df[plot_df["variant"] == variant_name].set_index("suite").loc[suites]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar([idx - width / 2 for idx in x], baseline["average_regret"], width=width, label=baseline_name, color="#1f77b4")
    axes[0].bar([idx + width / 2 for idx in x], variant["average_regret"], width=width, label=variant_name, color="#ff7f0e")
    axes[0].set_title("OOD Regret")
    axes[0].set_xticks(list(x), suites)

    axes[1].bar([idx - width / 2 for idx in x], baseline["p95_regret"], width=width, color="#1f77b4")
    axes[1].bar([idx + width / 2 for idx in x], variant["p95_regret"], width=width, color="#ff7f0e")
    axes[1].set_title("OOD p95")
    axes[1].set_xticks(list(x), suites)

    axes[2].bar([idx - width / 2 for idx in x], baseline["deadline_miss_rate"], width=width, color="#1f77b4")
    axes[2].bar([idx + width / 2 for idx in x], variant["deadline_miss_rate"], width=width, color="#ff7f0e")
    axes[2].set_title("OOD Deadline Miss")
    axes[2].set_xticks(list(x), suites)
    axes[2].set_ylim(0.0, 1.0)
    axes[0].legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / output_name, dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = _baseline_frame()
    baseline.to_csv(PLOTS_DIR / "round4_e3_matched_baseline.csv", index=False)
    _plot_baseline(baseline)

    common_eval_config = "configs/experiments/e3_memory_hubs_rsm_round4_seed311.yaml"
    a2_predictions = _collect_calibration_predictions(
        model_label="A2",
        model_config_path="configs/experiments/a2_e3_deadline_head_round4_seed311.yaml",
        eval_config_path=common_eval_config,
        checkpoint_path=ARTIFACTS_DIR / "a2_e3_deadline_head_round4_seed311" / "checkpoints" / "best.pt",
    )
    a4_predictions = _collect_calibration_predictions(
        model_label="A4",
        model_config_path="configs/experiments/a4_e3_verifier_refine_round4_seed311.yaml",
        eval_config_path=common_eval_config,
        checkpoint_path=ARTIFACTS_DIR / "a4_e3_verifier_refine_round4_seed311" / "checkpoints" / "best.pt",
    )
    calibration_predictions = pd.concat([a2_predictions, a4_predictions], ignore_index=True)
    on_time_curve = _binned_curve(
        calibration_predictions,
        model_col="model",
        pred_col="on_time_prob",
        target_col="on_time_true",
        curve_name="on_time",
    )
    slack_curve = _binned_curve(
        calibration_predictions,
        model_col="model",
        pred_col="slack_pred",
        target_col="slack_true",
        curve_name="slack",
    )
    calibration_frame = pd.concat([on_time_curve, slack_curve], ignore_index=True)
    calibration_frame.to_csv(PLOTS_DIR / "round4_calibration_curves.csv", index=False)
    _plot_calibration(calibration_frame)

    deadline_compare = pd.read_csv(PLOTS_DIR / "round4_seed311_variant_compare.csv")
    deadline_compare.to_csv(PLOTS_DIR / "round4_deadline_p95_compare.csv", index=False)
    _plot_deadline_compare(deadline_compare)

    multiheavy = _variant_compare_frame(
        [
            (311, "e3_memory_hubs_rsm_round4_seed311", "e3_memory_hubs_rsm_round4_multiheavy_seed311"),
            (312, "e3_memory_hubs_rsm_round4_seed312", "e3_memory_hubs_rsm_round4_multiheavy_seed312"),
            (313, "e3_memory_hubs_rsm_round4_seed313", "e3_memory_hubs_rsm_round4_multiheavy_seed313"),
        ]
    )
    multiheavy["variant"] = multiheavy["variant"].replace({"variant": "Multiheavy"})
    multiheavy.loc[(multiheavy["seed"] == "mean") & (multiheavy["variant"] == "Multiheavy"), "experiment"] = "Multiheavy-mean"
    multiheavy.to_csv(PLOTS_DIR / "round4_multiheavy_vs_e3.csv", index=False)
    _plot_variant_compare(multiheavy, title_prefix="Multiheavy vs E3", output_name="round4_multiheavy_vs_e3.png")

    reranker = _variant_compare_frame(
        [
            (311, "e3_memory_hubs_rsm_round4_seed311", "a3_e3_path_reranker_round4_seed311"),
            (312, "e3_memory_hubs_rsm_round4_seed312", "a3_e3_path_reranker_round4_seed312"),
        ]
    )
    reranker["variant"] = reranker["variant"].replace({"variant": "PathReranker"})
    reranker.loc[(reranker["seed"] == "mean") & (reranker["variant"] == "PathReranker"), "experiment"] = "PathReranker-mean"
    reranker.to_csv(PLOTS_DIR / "round4_path_reranker_vs_e3.csv", index=False)
    _plot_variant_compare(reranker, title_prefix="Path Reranker vs E3", output_name="round4_path_reranker_vs_e3.png")

    combined_e3 = _variant_compare_frame(
        [
            (311, "e3_memory_hubs_rsm_round4_seed311", "a3_e3_multiheavy_path_reranker_round4_seed311"),
            (312, "e3_memory_hubs_rsm_round4_seed312", "a3_e3_multiheavy_path_reranker_round4_seed312"),
            (313, "e3_memory_hubs_rsm_round4_seed313", "a3_e3_multiheavy_path_reranker_round4_seed313"),
        ]
    )
    combined_e3["variant"] = combined_e3["variant"].replace({"variant": "Multiheavy+PathReranker"})
    combined_e3.loc[(combined_e3["seed"] == "mean") & (combined_e3["variant"] == "Multiheavy+PathReranker"), "experiment"] = "Multiheavy+PathReranker-mean"
    combined_e3.to_csv(PLOTS_DIR / "round4_multiheavy_path_reranker_vs_e3.csv", index=False)
    _plot_variant_compare(
        combined_e3,
        title_prefix="Multiheavy+Path Reranker vs E3",
        output_name="round4_multiheavy_path_reranker_vs_e3.png",
    )

    combined_multiheavy = _variant_compare_frame(
        [
            (311, "e3_memory_hubs_rsm_round4_multiheavy_seed311", "a3_e3_multiheavy_path_reranker_round4_seed311"),
            (312, "e3_memory_hubs_rsm_round4_multiheavy_seed312", "a3_e3_multiheavy_path_reranker_round4_seed312"),
            (313, "e3_memory_hubs_rsm_round4_multiheavy_seed313", "a3_e3_multiheavy_path_reranker_round4_seed313"),
        ]
    )
    combined_multiheavy["variant"] = combined_multiheavy["variant"].replace({"E3": "Multiheavy", "variant": "Multiheavy+PathReranker"})
    combined_multiheavy.loc[(combined_multiheavy["seed"] == "mean") & (combined_multiheavy["variant"] == "Multiheavy"), "experiment"] = "Multiheavy-mean"
    combined_multiheavy.loc[(combined_multiheavy["seed"] == "mean") & (combined_multiheavy["variant"] == "Multiheavy+PathReranker"), "experiment"] = "Multiheavy+PathReranker-mean"
    combined_multiheavy.to_csv(PLOTS_DIR / "round4_multiheavy_path_reranker_vs_multiheavy.csv", index=False)
    _plot_variant_compare(
        combined_multiheavy,
        title_prefix="Multiheavy+Path Reranker vs Multiheavy",
        output_name="round4_multiheavy_path_reranker_vs_multiheavy.png",
    )

    combined_deadline_head_summary = ARTIFACTS_DIR / "a2_e3_multiheavy_path_reranker_deadline_head_round4_seed311" / "summary.json"
    if combined_deadline_head_summary.exists():
        combined_deadline_head = _variant_compare_frame(
            [
                (311, "a3_e3_multiheavy_path_reranker_round4_seed311", "a2_e3_multiheavy_path_reranker_deadline_head_round4_seed311"),
            ]
        )
        combined_deadline_head["variant"] = combined_deadline_head["variant"].replace(
            {"E3": "Multiheavy+PathReranker", "variant": "Multiheavy+PathReranker+DeadlineHead"}
        )
        combined_deadline_head.loc[
            (combined_deadline_head["seed"] == "mean") & (combined_deadline_head["variant"] == "Multiheavy+PathReranker"),
            "experiment",
        ] = "Multiheavy+PathReranker-mean"
        combined_deadline_head.loc[
            (combined_deadline_head["seed"] == "mean") & (combined_deadline_head["variant"] == "Multiheavy+PathReranker+DeadlineHead"),
            "experiment",
        ] = "Multiheavy+PathReranker+DeadlineHead-mean"
        combined_deadline_head.to_csv(PLOTS_DIR / "round4_combined_deadline_head_vs_combined.csv", index=False)
        _plot_variant_compare(
            combined_deadline_head,
            title_prefix="Combined Deadline Head vs Combined",
            output_name="round4_combined_deadline_head_vs_combined.png",
        )

    multiheavy_deadline_head_summary = ARTIFACTS_DIR / "a2_e3_multiheavy_deadline_head_round4_seed311" / "summary.json"
    if multiheavy_deadline_head_summary.exists():
        multiheavy_deadline_head = _variant_compare_frame(
            [
                (311, "e3_memory_hubs_rsm_round4_multiheavy_seed311", "a2_e3_multiheavy_deadline_head_round4_seed311"),
            ]
        )
        multiheavy_deadline_head["variant"] = multiheavy_deadline_head["variant"].replace(
            {"E3": "Multiheavy", "variant": "Multiheavy+DeadlineHead"}
        )
        multiheavy_deadline_head.loc[
            (multiheavy_deadline_head["seed"] == "mean") & (multiheavy_deadline_head["variant"] == "Multiheavy"),
            "experiment",
        ] = "Multiheavy-mean"
        multiheavy_deadline_head.loc[
            (multiheavy_deadline_head["seed"] == "mean")
            & (multiheavy_deadline_head["variant"] == "Multiheavy+DeadlineHead"),
            "experiment",
        ] = "Multiheavy+DeadlineHead-mean"
        multiheavy_deadline_head.to_csv(PLOTS_DIR / "round4_multiheavy_deadline_head_vs_multiheavy.csv", index=False)
        _plot_variant_compare(
            multiheavy_deadline_head,
            title_prefix="Multiheavy Deadline Head vs Multiheavy",
            output_name="round4_multiheavy_deadline_head_vs_multiheavy.png",
        )

    gated_reranker = _variant_compare_frame(
        [
            (311, "e3_memory_hubs_rsm_round4_multiheavy_seed311", "a3_e3_multiheavy_path_reranker_gated_round4_seed311"),
            (312, "e3_memory_hubs_rsm_round4_multiheavy_seed312", "a3_e3_multiheavy_path_reranker_gated_round4_seed312"),
            (313, "e3_memory_hubs_rsm_round4_multiheavy_seed313", "a3_e3_multiheavy_path_reranker_gated_round4_seed313"),
        ]
    )
    gated_reranker["variant"] = gated_reranker["variant"].replace(
        {"E3": "Multiheavy", "variant": "Multiheavy+GatedPathReranker"}
    )
    gated_reranker.loc[
        (gated_reranker["seed"] == "mean") & (gated_reranker["variant"] == "Multiheavy"),
        "experiment",
    ] = "Multiheavy-mean"
    gated_reranker.loc[
        (gated_reranker["seed"] == "mean") & (gated_reranker["variant"] == "Multiheavy+GatedPathReranker"),
        "experiment",
    ] = "Multiheavy+GatedPathReranker-mean"
    gated_reranker.to_csv(PLOTS_DIR / "round4_multiheavy_path_reranker_gated_vs_multiheavy.csv", index=False)
    _plot_variant_compare(
        gated_reranker,
        title_prefix="Gated Path Reranker vs Multiheavy",
        output_name="round4_multiheavy_path_reranker_gated_vs_multiheavy.png",
    )

    verifier_pairs = [
        (311, "e3_memory_hubs_rsm_round4_multiheavy_seed311", "a3_e3_multiheavy_path_reranker_verifier_round4_seed311"),
        (312, "e3_memory_hubs_rsm_round4_multiheavy_seed312", "a3_e3_multiheavy_path_reranker_verifier_round4_seed312"),
        (313, "e3_memory_hubs_rsm_round4_multiheavy_seed313", "a3_e3_multiheavy_path_reranker_verifier_round4_seed313"),
    ]
    verifier_pairs = [
        pair for pair in verifier_pairs if (ARTIFACTS_DIR / pair[2] / "summary.json").exists()
    ]
    if verifier_pairs:
        verifier_reranker = _variant_compare_frame(
            verifier_pairs
        )
        verifier_reranker["variant"] = verifier_reranker["variant"].replace(
            {"E3": "Multiheavy", "variant": "Multiheavy+VerifierPathReranker"}
        )
        verifier_reranker.loc[
            (verifier_reranker["seed"] == "mean") & (verifier_reranker["variant"] == "Multiheavy"),
            "experiment",
        ] = "Multiheavy-mean"
        verifier_reranker.loc[
            (verifier_reranker["seed"] == "mean")
            & (verifier_reranker["variant"] == "Multiheavy+VerifierPathReranker"),
            "experiment",
        ] = "Multiheavy+VerifierPathReranker-mean"
        verifier_reranker.to_csv(PLOTS_DIR / "round4_multiheavy_path_reranker_verifier_vs_multiheavy.csv", index=False)
        _plot_variant_compare(
            verifier_reranker,
            title_prefix="Verifier Path Reranker vs Multiheavy",
            output_name="round4_multiheavy_path_reranker_verifier_vs_multiheavy.png",
        )

    ood_compare = _ood_compare_frame()
    if not ood_compare.empty:
        ood_compare.to_csv(PLOTS_DIR / "round4_multiheavy_path_reranker_ood_vs_multiheavy.csv", index=False)
        _plot_ood_compare(ood_compare, output_name="round4_multiheavy_path_reranker_ood_vs_multiheavy.png")

    gated_ood_compare = _ood_compare_frame_from_specs(
        [
            (311, "round4_multiheavy_ood_seed311.csv", "round4_multiheavy_path_reranker_gated_ood_seed311.csv"),
            (312, "round4_multiheavy_ood_seed312.csv", "round4_multiheavy_path_reranker_gated_ood_seed312.csv"),
        ],
        baseline_label="Multiheavy",
        variant_label="Multiheavy+GatedPathReranker",
    )
    if not gated_ood_compare.empty:
        gated_ood_compare.to_csv(PLOTS_DIR / "round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.csv", index=False)
        _plot_ood_compare(
            gated_ood_compare,
            output_name="round4_multiheavy_path_reranker_gated_ood_vs_multiheavy.png",
        )

    verifier_ood_specs = [
        (311, "round4_multiheavy_ood_seed311.csv", "round4_multiheavy_path_reranker_verifier_ood_seed311.csv"),
        (312, "round4_multiheavy_ood_seed312.csv", "round4_multiheavy_path_reranker_verifier_ood_seed312.csv"),
        (313, "round4_multiheavy_ood_seed313.csv", "round4_multiheavy_path_reranker_verifier_ood_seed313.csv"),
    ]
    verifier_ood_specs = [
        spec for spec in verifier_ood_specs if (PLOTS_DIR / spec[2]).exists()
    ]
    verifier_ood_compare = _ood_compare_frame_from_specs(
        verifier_ood_specs,
        baseline_label="Multiheavy",
        variant_label="Multiheavy+VerifierPathReranker",
    )
    if not verifier_ood_compare.empty:
        verifier_ood_compare.to_csv(
            PLOTS_DIR / "round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.csv",
            index=False,
        )
        _plot_ood_compare(
            verifier_ood_compare,
            output_name="round4_multiheavy_path_reranker_verifier_ood_vs_multiheavy.png",
        )

    portfolio = _portfolio_frame()
    portfolio.to_csv(PLOTS_DIR / "portfolio_usage_round4.csv", index=False)
    _plot_portfolio(portfolio)


if __name__ == "__main__":
    main()
