from __future__ import annotations

from gnn3.data.hidden_corridor import HiddenCorridorConfig
from gnn3.train.config import BenchmarkConfig, hidden_corridor_config_for_split


def test_hidden_corridor_split_overrides_apply_only_to_requested_split() -> None:
    benchmark = BenchmarkConfig(
        hidden_corridor=HiddenCorridorConfig(
            seed=311,
            deadline_mode="oracle_calibrated",
            deadline_slack_ratio_range=(0.05, 0.15),
            deadline_slack_abs_range=(0.25, 1.0),
        ),
        train_hidden_corridor_overrides={
            "deadline_slack_ratio_range": (0.02, 0.08),
            "deadline_slack_abs_range": (0.1, 0.5),
        },
    )

    train_cfg = hidden_corridor_config_for_split(benchmark, "train")
    val_cfg = hidden_corridor_config_for_split(benchmark, "val")
    test_cfg = hidden_corridor_config_for_split(benchmark, "test")

    assert train_cfg.seed == 311
    assert val_cfg.seed == 10_311
    assert test_cfg.seed == 20_311

    assert train_cfg.deadline_slack_ratio_range == (0.02, 0.08)
    assert train_cfg.deadline_slack_abs_range == (0.1, 0.5)
    assert val_cfg.deadline_slack_ratio_range == (0.05, 0.15)
    assert test_cfg.deadline_slack_abs_range == (0.25, 1.0)
