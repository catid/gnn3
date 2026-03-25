#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOTS = Path("reports/plots")
ARTIFACTS = Path("artifacts")
EXPERIMENTS = Path("artifacts/experiments")
EXPERIMENT_SUMMARY = PLOTS / "experiment_summary.csv"

ROUND8_BASELINES = [
    "e3_memory_hubs_rsm_round8_multiheavy_seed311",
    "e3_memory_hubs_rsm_round8_multiheavy_seed312",
    "e3_memory_hubs_rsm_round8_multiheavy_seed313",
]

CRITIC_VARIANTS = {
    "scalar_q": ARTIFACTS / "round8_critic_scalar_q_seed312" / "summary.csv",
    "risk_multi": ARTIFACTS / "round8_critic_risk_multi_seed312" / "summary.csv",
    "late_unfreeze": ARTIFACTS / "round8_critic_late_unfreeze_seed312" / "summary.csv",
    "late_unfreeze_gate15": ARTIFACTS / "round8_critic_late_unfreeze_gate15_seed312" / "summary.csv",
    "pairwise_rank": ARTIFACTS / "round8_critic_pairwise_seed312" / "summary.csv",
}

SEARCH_VARIANTS = {
    "scalar_q_top2d1_targeted": PLOTS / "round8_search_scalar_q_top2d1_targeted_summary.csv",
    "pairwise_top2d1_targeted": PLOTS / "round8_search_pairwise_top2d1_targeted_summary.csv",
}

SEARCH_RUNTIME_ROWS = [
    {"variant": "scalar_q_full_suite", "family": "search", "minutes_to_kill": 1456.0 / 60.0, "status": "killed_runtime"},
    {"variant": "pairwise_full_suite", "family": "search", "minutes_to_kill": 731.0 / 60.0, "status": "killed_runtime"},
    {"variant": "scalar_q_top2d1_targeted", "family": "search", "minutes_to_kill": 980.0 / 60.0, "status": "killed_runtime"},
    {"variant": "pairwise_top2d1_targeted", "family": "search", "minutes_to_kill": 980.0 / 60.0, "status": "killed_runtime"},
    {"variant": "path_tiebreak_top2_targeted", "family": "backup", "minutes_to_kill": 621.0 / 60.0, "status": "killed_runtime"},
]

PORTFOLIO_ROWS = [
    {
        "experiment_id": "A1-R8-baseline-batch",
        "bucket": "exploit",
        "actual_gpu_hours": 0.09881602830357021 + 0.06624586999416351 + 0.10116986592610677,
        "status": "completed",
        "key_result": "Fresh round-eight multiheavy guardrail batch on seeds 311/312/313 reproduced the established regret and miss band.",
    },
    {
        "experiment_id": "A2-R8-near-tie-audit",
        "bucket": "exploit",
        "actual_gpu_hours": 0.09 + 0.09 + (393.0 / 3600.0),
        "status": "completed",
        "key_result": "Three fresh near-tie audits confirmed that large-gap stays solved while the hard near-tie intersection remains the real residual error slice.",
    },
    {
        "experiment_id": "A3-R8-probe-audit",
        "bucket": "exploit",
        "actual_gpu_hours": 0.09 + (470.0 / 3600.0),
        "status": "stopped",
        "key_result": "Seed312 probe completed and supported the decision-rule bottleneck diagnosis; seed311 confirmatory probe was stopped after the round decision was already fixed.",
    },
    {
        "experiment_id": "B1-R8-counterfactual-dataset",
        "bucket": "explore",
        "actual_gpu_hours": 0.05,
        "status": "completed",
        "key_result": "Cached 25,007 all-action candidate rows, with near-tie baseline-error coverage concentrated in deeper_packets6 and heavy_dynamic.",
    },
    {
        "experiment_id": "B2-R8-critics",
        "bucket": "explore",
        "actual_gpu_hours": 0.04 + 0.04 + 0.045 + 0.045 + 0.04,
        "status": "completed",
        "key_result": "Five critic variants settled the family: pairwise was the safest direct critic, while scalar and late-unfreeze were informative but too destructive.",
    },
    {
        "experiment_id": "C1-R8-search",
        "bucket": "explore",
        "actual_gpu_hours": (1456.0 + 731.0 + 980.0 + 980.0) / 3600.0,
        "status": "killed-early",
        "key_result": "Full-suite and targeted bounded-search scouts were killed on runtime before earning promotion.",
    },
    {
        "experiment_id": "E1-R8-path-tiebreak",
        "bucket": "explore",
        "actual_gpu_hours": 621.0 / 3600.0,
        "status": "killed-early",
        "key_result": "The near-tie-only local suffix-cost tie-break backup was cheaper than bounded search but still crossed the scout runtime bar.",
    },
]

TARGET_SUITES = [
    "a1_multiheavy_ood_deeper_packets6_round8_eval",
    "a1_multiheavy_ood_heavy_dynamic_round8_eval",
]


def _weighted_mean(frame: pd.DataFrame, column: str, weight_col: str = "decisions") -> float:
    if frame.empty:
        return 0.0
    weights = frame[weight_col].clip(lower=1.0)
    return float((frame[column] * weights).sum() / weights.sum())


def _load_baseline_rows() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for experiment in ROUND8_BASELINES:
        summary_path = EXPERIMENTS / experiment / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        test = summary["test"]
        rollout = summary["test_rollout"]
        rows.append(
            {
                "experiment": experiment,
                "bucket": "exploit",
                "gpu_hours": float(summary.get("gpu_hours", 0.0)),
                "test_next_hop_accuracy": float(test["next_hop_accuracy"]),
                "value_mae": float(test.get("value_mae", 0.0)),
                "value_rmse": float(test.get("value_rmse", 0.0)),
                "rollout_solved_rate": float(rollout["solved_rate"]),
                "rollout_next_hop_accuracy": float(rollout["next_hop_accuracy"]),
                "average_regret": float(rollout["average_regret"]),
                "p95_regret": float(rollout["p95_regret"]),
                "worst_regret": float(rollout["worst_regret"]),
                "deadline_violations": float(rollout["average_deadline_violations"]),
                "deadline_miss_rate": float(rollout["deadline_miss_rate"]),
            }
        )
    return pd.DataFrame(rows)


def _load_critic_summary() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, path in CRITIC_VARIANTS.items():
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        ood_overall = summary.loc[(summary["suite"].isin(TARGET_SUITES)) & (summary["slice"] == "overall")].copy()
        hard = summary.loc[(summary["suite"].isin(TARGET_SUITES)) & (summary["slice"] == "hard_near_tie")].copy()
        error = summary.loc[(summary["suite"].isin(TARGET_SUITES)) & (summary["slice"] == "hard_near_tie_error")].copy()
        rows.append(
            {
                "variant": variant,
                "overall_target_match_delta": _weighted_mean(ood_overall, "candidate_target_match")
                - _weighted_mean(ood_overall, "base_target_match"),
                "hard_near_tie_decisions": float(hard["decisions"].sum()),
                "hard_near_tie_disagreement": _weighted_mean(hard, "disagreement"),
                "hard_near_tie_correction_rate": _weighted_mean(hard, "correction_rate"),
                "hard_near_tie_new_error_rate": _weighted_mean(hard, "new_error_rate"),
                "hard_near_tie_regret_delta": _weighted_mean(hard, "regret_delta"),
                "hard_near_tie_miss_delta": _weighted_mean(hard, "miss_delta"),
                "baseline_error_decisions": float(error["decisions"].sum()),
                "baseline_error_recovery": _weighted_mean(error, "candidate_target_match"),
                "baseline_error_regret_delta": _weighted_mean(error, "regret_delta"),
            }
        )
    return pd.DataFrame(rows)


def _load_search_summary() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, path in SEARCH_VARIANTS.items():
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        target = summary.loc[summary["suite"].isin(TARGET_SUITES)].copy()
        rows.append(
            {
                "variant": variant,
                "decisions": float(target["decisions"].sum()),
                "search_trigger_rate": _weighted_mean(target, "search_trigger_rate"),
                "net_corrected": float(target["net_corrected"].sum()),
                "hard_near_tie_disagreement": _weighted_mean(target, "hard_near_tie_disagreement"),
                "hard_near_tie_correction_rate": _weighted_mean(target, "hard_near_tie_correction_rate"),
                "hard_near_tie_new_error_rate": _weighted_mean(target, "hard_near_tie_new_error_rate"),
                "baseline_error_recovery": _weighted_mean(target, "baseline_error_recovery"),
                "average_regret": _weighted_mean(target, "average_regret"),
                "p95_regret": _weighted_mean(target, "p95_regret"),
                "worst_regret": _weighted_mean(target, "worst_regret"),
                "deadline_miss_rate": _weighted_mean(target, "deadline_miss_rate"),
                "priority_delivered_regret": _weighted_mean(target, "priority_delivered_regret"),
            }
        )
    return pd.DataFrame(rows)


def _plot_guardrail(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = df["experiment"].tolist()
    axes[0].bar(labels, df["average_regret"], color="#1f77b4")
    axes[0].set_title("Round 8 Baseline Regret")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, df["p95_regret"], color="#ff7f0e")
    axes[1].set_title("Round 8 Baseline p95 Regret")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(labels, df["deadline_miss_rate"], color="#d62728")
    axes[2].set_title("Round 8 Baseline Miss Rate")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(PLOTS / "round8_multiheavy_guardrail.png", dpi=160)
    plt.close(fig)


def _plot_critic(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = df["variant"].tolist()

    axes[0].bar(labels, df["hard_near_tie_disagreement"], color="#1f77b4")
    axes[0].set_title("Hard Near-Tie Disagreement")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, df["hard_near_tie_correction_rate"], color="#2ca02c", label="Correction")
    axes[1].bar(labels, -df["hard_near_tie_new_error_rate"], color="#d62728", label="New Error")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Correction vs New Error")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    axes[2].bar(labels, df["baseline_error_regret_delta"], color="#9467bd")
    axes[2].axhline(0.0, color="black", linewidth=1.0)
    axes[2].set_title("Error-Subset Regret Delta")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(PLOTS / "round8_critic_summary.png", dpi=160)
    plt.close(fig)


def _plot_search(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = df["variant"].tolist()

    axes[0].bar(labels, df["search_trigger_rate"], color="#1f77b4")
    axes[0].set_title("Search Trigger Rate")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(labels, df["baseline_error_recovery"], color="#2ca02c")
    axes[1].set_title("Baseline-Error Recovery")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(0.0, 1.0)

    axes[2].bar(labels, df["deadline_miss_rate"], color="#d62728")
    axes[2].set_title("Deadline Miss Rate")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(PLOTS / "round8_search_summary.png", dpi=160)
    plt.close(fig)


def _plot_search_runtime(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#d62728" if family == "search" else "#9467bd" for family in df["family"]]
    ax.bar(df["variant"], df["minutes_to_kill"], color=colors)
    ax.set_title("Round 8 Search Runtime To Kill")
    ax.set_ylabel("minutes")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOTS / "round8_search_runtime.png", dpi=160)
    plt.close(fig)


def _plot_portfolio(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    colors = ["#1f77b4" if bucket == "exploit" else "#ff7f0e" for bucket in df["bucket"]]
    axes[0].bar(df["experiment_id"], df["actual_gpu_hours"], color=colors)
    axes[0].set_title("Round 8 GPU-Hours by Task")
    axes[0].tick_params(axis="x", rotation=35)

    totals = df.groupby("bucket", as_index=False)["actual_gpu_hours"].sum()
    totals = totals.sort_values("bucket").reset_index(drop=True)
    axes[1].bar(totals["bucket"], totals["actual_gpu_hours"], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Round 8 Exploit vs Explore")
    axes[1].set_ylim(0.0, max(0.05, float(totals["actual_gpu_hours"].max()) * 1.15))

    fig.tight_layout()
    fig.savefig(PLOTS / "round8_portfolio_usage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    guardrail_df = _load_baseline_rows()
    if not guardrail_df.empty:
        guardrail_df = guardrail_df.sort_values("experiment").reset_index(drop=True)
    critic_df = _load_critic_summary()
    if not critic_df.empty:
        critic_df = critic_df.sort_values("variant").reset_index(drop=True)
    search_df = _load_search_summary()
    if not search_df.empty:
        search_df = search_df.sort_values("variant").reset_index(drop=True)
    search_runtime_df = pd.DataFrame(SEARCH_RUNTIME_ROWS)
    portfolio_df = pd.DataFrame(PORTFOLIO_ROWS)

    guardrail_df.to_csv(PLOTS / "round8_multiheavy_guardrail.csv", index=False)
    critic_df.to_csv(PLOTS / "round8_critic_summary.csv", index=False)
    if not search_df.empty:
        search_df.to_csv(PLOTS / "round8_search_summary.csv", index=False)
    search_runtime_df.to_csv(PLOTS / "round8_search_runtime.csv", index=False)
    portfolio_df.to_csv(PLOTS / "round8_portfolio_usage.csv", index=False)

    _plot_guardrail(guardrail_df)
    _plot_critic(critic_df)
    _plot_search(search_df)
    _plot_search_runtime(search_runtime_df)
    _plot_portfolio(portfolio_df)

    if EXPERIMENT_SUMMARY.exists():
        existing = pd.read_csv(EXPERIMENT_SUMMARY)
    else:
        existing = pd.DataFrame(
            columns=[
                "experiment",
                "bucket",
                "gpu_hours",
                "test_next_hop_accuracy",
                "value_mae",
                "value_rmse",
                "rollout_solved_rate",
                "rollout_next_hop_accuracy",
                "average_regret",
                "p95_regret",
                "worst_regret",
                "deadline_violations",
                "deadline_miss_rate",
            ]
        )
    existing = existing.loc[~existing["experiment"].isin(set(guardrail_df["experiment"].tolist()))].copy()
    combined = pd.concat([existing, guardrail_df], ignore_index=True)
    combined = combined.sort_values("experiment").reset_index(drop=True)
    combined.to_csv(EXPERIMENT_SUMMARY, index=False)


if __name__ == "__main__":
    main()
