from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnn3.data.hidden_corridor import DecisionRecord, HiddenCorridorConfig, critical_decision_weight
from gnn3.models.packet_mamba import PacketMambaConfig
from gnn3.train.config import BenchmarkConfig, ExperimentConfig, TrainConfig
from gnn3.train.trainer import train_experiment


def _record(*, packet_count: int, feasible_slacks: list[float] | None) -> DecisionRecord:
    num_nodes = 3
    candidate_mask = np.asarray([False, True, True], dtype=bool)
    candidate_slack = np.zeros((num_nodes,), dtype=np.float32)
    candidate_on_time = np.zeros((num_nodes,), dtype=np.float32)
    if feasible_slacks is not None:
        for index, slack in enumerate(feasible_slacks, start=1):
            candidate_slack[index] = slack
            candidate_on_time[index] = 1.0
    else:
        candidate_slack[1:] = -2.0
    return DecisionRecord(
        node_features=np.zeros((num_nodes, 13), dtype=np.float32),
        edge_features=np.zeros((num_nodes, num_nodes, 4), dtype=np.float32),
        node_roles=np.zeros((num_nodes,), dtype=np.int64),
        node_communities=np.zeros((num_nodes,), dtype=np.int64),
        adjacency=np.zeros((num_nodes, num_nodes), dtype=bool),
        current_node=0,
        source_node=0,
        destination_node=2,
        order_current=np.arange(num_nodes, dtype=np.int64),
        order_destination=np.arange(num_nodes, dtype=np.int64),
        candidate_mask=candidate_mask,
        target_next_hop=1,
        cost_to_go=1.0,
        candidate_cost_to_go=np.zeros((num_nodes,), dtype=np.float32),
        candidate_slack=candidate_slack,
        candidate_on_time=candidate_on_time,
        candidate_path_nodes=np.full((num_nodes, num_nodes), fill_value=-1, dtype=np.int64),
        candidate_path_mask=np.zeros((num_nodes, num_nodes), dtype=bool),
        candidate_path_features=np.zeros((num_nodes, 5), dtype=np.float32),
        route_relevance=np.zeros((num_nodes,), dtype=np.float32),
        packet_priority=1.0,
        packet_deadline=10.0,
        packet_index=0,
        packet_count=packet_count,
        curriculum_level="multi_dynamic",
    )


def test_critical_decision_weight_prioritizes_low_slack_and_infeasible_records() -> None:
    easy = _record(packet_count=1, feasible_slacks=[5.0, 3.0])
    low_slack = _record(packet_count=4, feasible_slacks=[0.4, 0.2])
    infeasible = _record(packet_count=4, feasible_slacks=None)

    easy_weight = critical_decision_weight(
        easy,
        packets_cap=4,
        slack_weight=2.0,
        packet_weight=1.0,
        infeasible_bonus=1.0,
        max_multiplier=4.0,
    )
    low_slack_weight = critical_decision_weight(
        low_slack,
        packets_cap=4,
        slack_weight=2.0,
        packet_weight=1.0,
        infeasible_bonus=1.0,
        max_multiplier=4.0,
    )
    infeasible_weight = critical_decision_weight(
        infeasible,
        packets_cap=4,
        slack_weight=2.0,
        packet_weight=1.0,
        infeasible_bonus=1.0,
        max_multiplier=4.0,
    )

    assert easy_weight >= 1.0
    assert low_slack_weight > easy_weight
    assert infeasible_weight >= low_slack_weight


def test_smoke_training_runs_with_critical_sampling(tmp_path: Path) -> None:
    benchmark = BenchmarkConfig(
        train_episodes=8,
        val_episodes=2,
        test_episodes=2,
        curriculum_levels=("single_static", "single_dynamic"),
        hidden_corridor=HiddenCorridorConfig(seed=21, packets_max=2),
    )
    experiment = ExperimentConfig(
        name="pytest_sampling_smoke",
        bucket="exploit",
        seed=21,
        output_dir=str(tmp_path / "pytest_sampling_smoke"),
        benchmark=benchmark,
        model=PacketMambaConfig(
            d_model=32,
            d_state=4,
            inner_layers=1,
            outer_steps=1,
            router_variant="local",
        ),
        train=TrainConfig(
            batch_size=4,
            eval_batch_size=4,
            epochs=1,
            device="cpu",
            bf16=False,
            rollout_eval_episodes=2,
            train_decision_sampling="critical",
            train_critical_slack_weight=2.0,
            train_critical_packet_weight=0.5,
            train_critical_infeasible_bonus=1.0,
            train_critical_max_multiplier=4.0,
        ),
    )
    summary = train_experiment(experiment)
    assert summary is not None
    metadata = json.loads(Path(summary["output_dir"], "metadata.json").read_text())
    assert metadata["train_sampling"]["mode"] == "critical"
    assert metadata["train_sampling"]["max_weight"] >= metadata["train_sampling"]["min_weight"] >= 1.0
