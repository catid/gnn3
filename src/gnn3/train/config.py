from __future__ import annotations

from dataclasses import dataclass, field
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
    hidden_corridor: HiddenCorridorConfig = field(default_factory=HiddenCorridorConfig)


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
    rollout_eval_episodes: int = 16


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    bucket: str
    seed: int
    output_dir: str
    benchmark: BenchmarkConfig
    model: PacketMambaConfig
    train: TrainConfig


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
        hidden_corridor=hidden_corridor,
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
        benchmark=_benchmark_from_dict(raw.get("benchmark", {})),
        model=_model_from_dict(raw.get("model", {})),
        train=_train_from_dict(raw.get("train", {})),
    )
