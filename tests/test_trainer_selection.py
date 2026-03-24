from __future__ import annotations

from types import SimpleNamespace

from gnn3.train.config import BenchmarkConfig, ExperimentConfig, TrainConfig
from gnn3.train.trainer import _selection_score


def _experiment(train: TrainConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name="selection-test",
        bucket="exploit",
        seed=0,
        output_dir="artifacts/experiments/selection-test",
        benchmark=BenchmarkConfig(),
        model=SimpleNamespace(),  # type: ignore[arg-type]
        train=train,
    )


def _rollout(*, regret: float, p95: float, miss: float, deadline_violations: float) -> SimpleNamespace:
    return SimpleNamespace(
        average_regret=regret,
        p95_regret=p95,
        average_deadline_violations=deadline_violations,
        deadline_miss_rate=miss,
        solved_rate=1.0 - miss,
        next_hop_accuracy=0.9,
    )


def test_selection_score_defaults_match_legacy_weights() -> None:
    config = _experiment(TrainConfig())
    score = _selection_score(
        {"next_hop_accuracy": 0.8},
        _rollout(regret=1.0, p95=3.0, miss=0.25, deadline_violations=0.5),
        config,
    )
    expected = (
        0.35 * 0.8
        + 0.20 * 0.75
        + 0.10 * 0.9
        + 0.15 * (1.0 / 2.0)
        + 0.10 * (1.0 / 4.0)
        + 0.05 * 0.75
        + 0.05 * (1.0 / 1.5)
    )
    assert score == expected


def test_selection_score_can_bias_toward_tail_and_miss() -> None:
    base = TrainConfig()
    risk_biased = TrainConfig(
        selection_val_next_hop_weight=0.15,
        selection_rollout_solved_weight=0.10,
        selection_rollout_next_hop_weight=0.05,
        selection_rollout_regret_weight=0.20,
        selection_rollout_tail_regret_weight=0.25,
        selection_rollout_miss_weight=0.15,
        selection_rollout_deadline_weight=0.10,
    )
    val_metrics = {"next_hop_accuracy": 0.8}
    safe_rollout = _rollout(regret=1.0, p95=3.0, miss=0.25, deadline_violations=0.5)
    risky_rollout = _rollout(regret=0.9, p95=8.0, miss=0.5, deadline_violations=1.5)

    base_safe = _selection_score(val_metrics, safe_rollout, _experiment(base))
    base_risky = _selection_score(val_metrics, risky_rollout, _experiment(base))
    risk_safe = _selection_score(val_metrics, safe_rollout, _experiment(risk_biased))
    risk_risky = _selection_score(val_metrics, risky_rollout, _experiment(risk_biased))

    assert base_safe > base_risky
    assert risk_safe > risk_risky
    assert (risk_safe - risk_risky) > (base_safe - base_risky)
