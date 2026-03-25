from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from gnn3.data.hidden_corridor import HiddenCorridorConfig
from gnn3.models.packet_mamba import PacketMambaConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    train_episodes: int = 256
    val_episodes: int = 64
    test_episodes: int = 64
    curriculum_levels: tuple[str, ...] = ("single_static", "single_dynamic", "multi_dynamic")
    packets_max_eval: int = 8
    train_seed_offset: int = 0
    val_seed_offset: int = 10_000
    test_seed_offset: int = 20_000
    hidden_corridor: HiddenCorridorConfig = field(default_factory=HiddenCorridorConfig)
    train_hidden_corridor_overrides: dict[str, Any] = field(default_factory=dict)
    val_hidden_corridor_overrides: dict[str, Any] = field(default_factory=dict)
    test_hidden_corridor_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 0
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    bf16: bool = True
    compile: bool = False
    device: str = "auto"
    value_weight: float = 0.2
    route_weight: float = 0.1
    deadline_bce_weight: float = 0.0
    slack_loss_weight: float = 0.0
    quantile_loss_weight: float = 0.0
    rollout_eval_episodes: int = 16
    selection_val_next_hop_weight: float = 0.35
    selection_rollout_solved_weight: float = 0.20
    selection_rollout_next_hop_weight: float = 0.10
    selection_rollout_regret_weight: float = 0.15
    selection_rollout_tail_regret_weight: float = 0.10
    selection_rollout_miss_weight: float = 0.05
    selection_rollout_deadline_weight: float = 0.05
    selection_soft_target_weight: float = 0.0
    selection_soft_target_temperature: float = 1.0
    selection_soft_target_on_time_bonus: float = 0.0
    path_soft_target_weight: float = 0.0
    path_soft_target_temperature: float = 1.0
    path_soft_target_on_time_bonus: float = 0.0
    selection_pairwise_weight: float = 0.0
    selection_pairwise_temperature: float = 1.0
    selection_pairwise_on_time_bonus: float = 0.0
    selection_pairwise_slack_bonus: float = 0.0
    selection_pairwise_margin: float = 0.0
    selection_feasible_target_weight: float = 0.0
    selection_slack_critical_weight: float = 0.0
    selection_slack_critical_scale: float = 1.0
    planner_cost_weight: float = 0.0
    planner_on_time_weight: float = 0.0
    train_decision_sampling: str = "uniform"
    train_critical_slack_weight: float = 0.0
    train_critical_packet_weight: float = 0.0
    train_critical_infeasible_bonus: float = 0.0
    train_critical_max_multiplier: float = 1.0
    dagger_refresh_episodes: int = 0
    dagger_finetune_epochs: int = 0
    dagger_finetune_lr_scale: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    bucket: str
    seed: int
    output_dir: str
    benchmark: BenchmarkConfig
    model: PacketMambaConfig
    train: TrainConfig
    stage: str = "candidate"
    notes: str = ""


def _hidden_corridor_from_dict(data: dict[str, Any]) -> HiddenCorridorConfig:
    return HiddenCorridorConfig(**data)


def _benchmark_from_dict(data: dict[str, Any]) -> BenchmarkConfig:
    hidden_corridor = _hidden_corridor_from_dict(data.get("hidden_corridor", {}))
    return BenchmarkConfig(
        train_episodes=int(data.get("train_episodes", 256)),
        val_episodes=int(data.get("val_episodes", 64)),
        test_episodes=int(data.get("test_episodes", 64)),
        curriculum_levels=tuple(data.get("curriculum_levels", ("single_static", "single_dynamic", "multi_dynamic"))),
        packets_max_eval=int(data.get("packets_max_eval", 8)),
        train_seed_offset=int(data.get("train_seed_offset", 0)),
        val_seed_offset=int(data.get("val_seed_offset", 10_000)),
        test_seed_offset=int(data.get("test_seed_offset", 20_000)),
        hidden_corridor=hidden_corridor,
        train_hidden_corridor_overrides=_hidden_corridor_overrides_from_dict(data.get("train_hidden_corridor_overrides", {})),
        val_hidden_corridor_overrides=_hidden_corridor_overrides_from_dict(data.get("val_hidden_corridor_overrides", {})),
        test_hidden_corridor_overrides=_hidden_corridor_overrides_from_dict(data.get("test_hidden_corridor_overrides", {})),
    )


def _hidden_corridor_overrides_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, value in data.items():
        if key == "seed":
            continue
        if isinstance(value, list):
            overrides[key] = tuple(value)
        else:
            overrides[key] = value
    return overrides


def hidden_corridor_config_for_split(benchmark: BenchmarkConfig, split: str) -> HiddenCorridorConfig:
    split_offsets = {
        "train": benchmark.train_seed_offset,
        "val": benchmark.val_seed_offset,
        "test": benchmark.test_seed_offset,
    }
    split_overrides = {
        "train": benchmark.train_hidden_corridor_overrides,
        "val": benchmark.val_hidden_corridor_overrides,
        "test": benchmark.test_hidden_corridor_overrides,
    }
    if split not in split_offsets:
        raise ValueError(f"Unknown benchmark split: {split}")
    return replace(
        benchmark.hidden_corridor,
        seed=benchmark.hidden_corridor.seed + split_offsets[split],
        **split_overrides[split],
    )


def _model_from_dict(data: dict[str, Any]) -> PacketMambaConfig:
    return PacketMambaConfig(**data)


def _train_from_dict(data: dict[str, Any]) -> TrainConfig:
    return TrainConfig(**data)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        name=str(raw["name"]),
        bucket=str(raw["bucket"]),
        seed=int(raw["seed"]),
        output_dir=str(raw["output_dir"]),
        stage=str(raw.get("stage", "candidate")),
        notes=str(raw.get("notes", "")),
        benchmark=_benchmark_from_dict(raw.get("benchmark", {})),
        model=_model_from_dict(raw.get("model", {})),
        train=_train_from_dict(raw.get("train", {})),
    )
