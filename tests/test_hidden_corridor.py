from __future__ import annotations

from pathlib import Path

import numpy as np

from gnn3.data.hidden_corridor import (
    ROLE_TO_ID,
    HiddenCorridorConfig,
    HiddenCorridorDecisionDataset,
    PacketSpec,
    build_hidden_corridor_graph,
    collate_decisions,
    shortest_path,
)
from gnn3.models.packet_mamba import PacketMambaConfig, PacketMambaModel, compute_losses
from gnn3.train.config import BenchmarkConfig, ExperimentConfig, TrainConfig
from gnn3.train.trainer import train_experiment


def _community_root(graph, community: int) -> int:
    matches = np.where(
        (graph.node_roles == ROLE_TO_ID["root"]) & (graph.node_communities == community)
    )[0]
    return int(matches[0])


def _hub_nodes(graph) -> tuple[int, int]:
    matches = np.where(graph.node_roles == ROLE_TO_ID["hub"])[0]
    return int(matches[0]), int(matches[1])


def test_graph_generation_has_monitors_and_hubs() -> None:
    cfg = HiddenCorridorConfig(seed=3)
    graph = build_hidden_corridor_graph(np.random.default_rng(cfg.seed), cfg)
    assert int((graph.node_roles == ROLE_TO_ID["hub"]).sum()) == 2
    assert int((graph.node_roles == ROLE_TO_ID["monitor"]).sum()) == cfg.num_communities
    assert all(graph.leaf_nodes_by_community)


def test_oracle_prefers_fast_hub_when_fast_corridor_is_good() -> None:
    cfg = HiddenCorridorConfig(seed=4)
    graph = build_hidden_corridor_graph(np.random.default_rng(cfg.seed), cfg)
    hub_fast, hub_slow = _hub_nodes(graph)
    root_a = _community_root(graph, 0)
    root_b = _community_root(graph, 1)

    graph.edge_effective_latency[root_a, hub_fast] = 0.4
    graph.edge_effective_latency[hub_fast, root_a] = 0.4
    graph.edge_effective_latency[root_b, hub_fast] = 0.4
    graph.edge_effective_latency[hub_fast, root_b] = 0.4
    graph.edge_effective_latency[root_a, hub_slow] = 6.0
    graph.edge_effective_latency[hub_slow, root_a] = 6.0
    graph.edge_effective_latency[root_b, hub_slow] = 6.0
    graph.edge_effective_latency[hub_slow, root_b] = 6.0

    packet = PacketSpec(source=root_a, destination=root_b, priority=3.0, deadline=12.0)
    path, _ = shortest_path(graph, packet, start=root_a, remaining_deadline=packet.deadline, config=cfg)
    assert path[1] == hub_fast


def test_oracle_avoids_fast_hub_when_blocked() -> None:
    cfg = HiddenCorridorConfig(seed=5)
    graph = build_hidden_corridor_graph(np.random.default_rng(cfg.seed), cfg)
    hub_fast, hub_slow = _hub_nodes(graph)
    root_a = _community_root(graph, 0)
    root_b = _community_root(graph, 1)

    graph.edge_effective_latency[root_a, hub_fast] = 25.0
    graph.edge_effective_latency[hub_fast, root_a] = 25.0
    graph.edge_effective_latency[root_b, hub_fast] = 25.0
    graph.edge_effective_latency[hub_fast, root_b] = 25.0
    graph.edge_effective_latency[root_a, hub_slow] = 2.0
    graph.edge_effective_latency[hub_slow, root_a] = 2.0
    graph.edge_effective_latency[root_b, hub_slow] = 2.0
    graph.edge_effective_latency[hub_slow, root_b] = 2.0

    packet = PacketSpec(source=root_a, destination=root_b, priority=3.0, deadline=12.0)
    path, _ = shortest_path(graph, packet, start=root_a, remaining_deadline=packet.deadline, config=cfg)
    assert path[1] == hub_slow


def test_collate_and_model_forward() -> None:
    cfg = HiddenCorridorConfig(seed=6)
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model_cfg = PacketMambaConfig(
        node_feature_dim=batch["node_features"].shape[-1],
        outer_steps=2,
        inner_layers=2,
        router_variant="selective_read",
    )
    model = PacketMambaModel(model_cfg)
    output = model(batch)
    losses = compute_losses(output, batch, final_step_only=True)
    assert output["node_logits"].shape[0] == 2
    assert output["route_logits"].shape[:2] == batch["node_mask"].shape
    assert float(losses["loss"].detach()) > 0.0


def test_history_read_forward_path() -> None:
    cfg = HiddenCorridorConfig(seed=8)
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=3,
            inner_layers=2,
            router_variant="selective_read",
            history_read=True,
            detach_warmup=True,
        )
    )
    output = model(batch)
    assert "history_read_entropy" in output["diagnostics"]


def test_history_summary_bank_forward_path() -> None:
    cfg = HiddenCorridorConfig(seed=9)
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=3,
            inner_layers=2,
            router_variant="memory_hubs",
            history_read=True,
            history_read_mode="summary_bank",
            detach_warmup=True,
        )
    )
    output = model(batch)
    assert "history_hub_bank_share" in output["diagnostics"]
    assert "history_monitor_bank_share" in output["diagnostics"]
    assert "history_global_bank_share" in output["diagnostics"]


def test_dataset_manifest_is_stable_for_same_seed() -> None:
    cfg = HiddenCorridorConfig(seed=10)
    dataset_a = HiddenCorridorDecisionDataset(config=cfg, num_episodes=4)
    dataset_b = HiddenCorridorDecisionDataset(config=cfg, num_episodes=4)
    assert dataset_a.manifest()["manifest_hash"] == dataset_b.manifest()["manifest_hash"]


def test_smoke_training_runs(tmp_path: Path) -> None:
    benchmark = BenchmarkConfig(
        train_episodes=8,
        val_episodes=2,
        test_episodes=2,
        curriculum_levels=("single_static", "single_dynamic"),
        hidden_corridor=HiddenCorridorConfig(seed=7, packets_max=2),
    )
    experiment = ExperimentConfig(
        name="pytest_smoke",
        bucket="exploit",
        seed=7,
        output_dir=str(tmp_path / "pytest_smoke"),
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
        ),
    )
    summary = train_experiment(experiment)
    assert Path(summary["output_dir"]).exists()
    assert Path(summary["output_dir"], "summary.json").exists()
    assert Path(summary["output_dir"], "dataset_manifests.json").exists()
    assert summary["stage"] == "candidate"
    assert summary["git_branch"]
    assert "device_placement" in summary
    assert "manifest_hashes" in summary
    assert "p95_regret" in summary["test_rollout"]
    assert "deadline_miss_rate" in summary["test_rollout"]
