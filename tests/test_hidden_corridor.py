from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gnn3.data.hidden_corridor import (
    ROLE_TO_ID,
    HiddenCorridorConfig,
    HiddenCorridorDecisionDataset,
    PacketSpec,
    build_hidden_corridor_graph,
    collate_decisions,
    shortest_path,
)
from gnn3.eval.oracle_analysis import audit_oracle_deadlines
from gnn3.models.packet_mamba import PacketMambaConfig, PacketMambaModel, compute_losses
from gnn3.train.config import (
    BenchmarkConfig,
    ExperimentConfig,
    TrainConfig,
    hidden_corridor_config_for_split,
)
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


def test_deadline_head_forward_and_loss() -> None:
    cfg = HiddenCorridorConfig(
        seed=13,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=2,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            deadline_head=True,
            risk_aware_scoring=True,
        )
    )
    output = model(batch)
    losses = compute_losses(
        output,
        batch,
        final_step_only=True,
        deadline_bce_weight=0.2,
        slack_weight=0.1,
        quantile_weight=0.1,
        quantiles=model.config.quantile_levels,
        verifier_aux_last_k_steps=model.config.verifier_aux_last_k_steps,
    )
    assert output["selection_scores"].shape == output["node_logits"].shape
    assert output["candidate_on_time_logits"].shape == output["node_logits"].shape
    assert output["candidate_cost_quantiles"].shape[-1] == len(model.config.quantile_levels)
    assert float(losses["on_time_loss"].detach()) >= 0.0


def test_hazard_memory_forward_path() -> None:
    cfg = HiddenCorridorConfig(
        seed=14,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=2,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            hazard_memory=True,
        )
    )
    output = model(batch)
    assert output["node_logits"].shape[0] == 2


def test_delay_mailbox_forward_path() -> None:
    cfg = HiddenCorridorConfig(
        seed=115,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=3,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            delay_mailbox=True,
            delay_mailbox_delays=(1, 2),
            delay_mailbox_target="monitor_only",
            delay_mailbox_fusion="slow_only",
        )
    )
    output = model(batch)
    assert output["per_step_selection_scores"].shape[:2] == (2, 3)
    assert output["per_step_probe_features"].shape[:2] == (2, 3)
    assert "delay_mailbox_release_mean" in output["diagnostics"]
    assert "delay_mailbox_active" in output["diagnostics"]


def test_regime_experts_forward_path() -> None:
    cfg = HiddenCorridorConfig(
        seed=114,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=2,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            regime_experts=True,
            regime_num_experts=3,
            regime_hidden_dim=32,
        )
    )
    output = model(batch)
    assert output["node_logits"].shape[0] == 2
    assert output["regime_gate_weights"].shape == (2, 3)
    assert torch.allclose(
        output["regime_gate_weights"].sum(dim=-1),
        torch.ones((2,), dtype=output["regime_gate_weights"].dtype),
        atol=1e-5,
    )


def test_candidate_path_reranker_forward_and_loss() -> None:
    cfg = HiddenCorridorConfig(
        seed=15,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=2,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            path_reranker=True,
            path_reranker_bound=1.25,
            path_reranker_traffic_gate=True,
        )
    )
    output = model(batch)
    assert output["path_scores"].shape == output["node_logits"].shape
    assert output["path_reranker_gate"].shape == output["node_logits"].shape
    assert output["selection_scores"].shape == output["node_logits"].shape
    valid_mask = batch["candidate_path_mask"].any(dim=-1) & batch["candidate_mask"] & batch["node_mask"]
    assert float(output["path_reranker_gate"][valid_mask].min().detach()) >= 0.0
    assert float(output["path_reranker_gate"][valid_mask].max().detach()) <= 1.0
    assert float(output["path_scores"][valid_mask].abs().max().detach()) <= 1.25 + 1e-5


def test_planner_decoder_forward_and_loss() -> None:
    cfg = HiddenCorridorConfig(
        seed=16,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=2,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            planner_decoder=True,
            planner_hidden_dim=64,
        )
    )
    output = model(batch)
    losses = compute_losses(
        output,
        batch,
        final_step_only=True,
        planner_cost_weight=0.2,
        planner_on_time_weight=0.1,
        path_soft_target_weight=0.1,
    )
    assert output["planner_costs"].shape == output["node_logits"].shape
    assert output["planner_on_time_logits"].shape == output["node_logits"].shape
    assert output["selection_scores"].shape == output["node_logits"].shape
    valid_mask = batch["candidate_path_mask"].any(dim=-1) & batch["candidate_mask"] & batch["node_mask"]
    assert float(output["planner_costs"][valid_mask].min().detach()) >= 0.0
    assert float(losses["planner_cost_loss"].detach()) >= 0.0
    assert float(losses["planner_on_time_loss"].detach()) >= 0.0
    assert float(losses["loss"].detach()) > 0.0


def test_candidate_path_reranker_replace_mode_uses_path_scores() -> None:
    cfg = HiddenCorridorConfig(
        seed=115,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=2,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            path_reranker=True,
            path_reranker_mode="replace",
            path_reranker_bound=1.25,
            path_reranker_traffic_gate=True,
        )
    )
    output = model(batch)
    valid_mask = batch["candidate_path_mask"].any(dim=-1) & batch["candidate_mask"] & batch["node_mask"]
    assert torch.allclose(
        output["selection_scores"][valid_mask],
        output["path_scores"][valid_mask],
        atol=1e-5,
    )
    assert (output["selection_scores"][~valid_mask] < -1e8).all()


def test_candidate_path_verifier_filter_masks_infeasible_choices() -> None:
    cfg = HiddenCorridorConfig(
        seed=16,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(config=cfg, num_episodes=2)
    batch = collate_decisions([dataset[0], dataset[1]])
    valid_candidates = (batch["candidate_mask"] & batch["node_mask"])[0].nonzero().flatten()
    target = int(batch["target_next_hop"][0].item())
    batch["candidate_on_time"][0].zero_()
    batch["candidate_slack"][0].fill_(-5.0)
    batch["candidate_on_time"][0, target] = 1.0
    batch["candidate_slack"][0, target] = 2.0
    for candidate in valid_candidates.tolist():
        if candidate != target:
            batch["candidate_slack"][0, candidate] = -3.0

    model = PacketMambaModel(
        PacketMambaConfig(
            node_feature_dim=batch["node_features"].shape[-1],
            outer_steps=2,
            inner_layers=2,
            router_variant="memory_hubs",
            detach_warmup=True,
            path_reranker=True,
            path_verifier_filter=True,
        )
    )
    output = model(batch)
    infeasible_candidates = [candidate for candidate in valid_candidates.tolist() if candidate != target]
    assert output["selection_scores"][0, target] > -1e8
    for candidate in infeasible_candidates:
        assert output["selection_scores"][0, candidate] < -1e8


def test_dataset_manifest_is_stable_for_same_seed() -> None:
    cfg = HiddenCorridorConfig(seed=10)
    dataset_a = HiddenCorridorDecisionDataset(config=cfg, num_episodes=4)
    dataset_b = HiddenCorridorDecisionDataset(config=cfg, num_episodes=4)
    assert dataset_a.manifest()["manifest_hash"] == dataset_b.manifest()["manifest_hash"]


def test_split_seed_offsets_make_val_and_test_manifests_distinct() -> None:
    benchmark = BenchmarkConfig(
        val_episodes=4,
        test_episodes=4,
        curriculum_levels=("single_dynamic",),
        hidden_corridor=HiddenCorridorConfig(seed=11, packets_max=2),
    )
    val_cfg = hidden_corridor_config_for_split(benchmark, "val")
    test_cfg = hidden_corridor_config_for_split(benchmark, "test")
    assert val_cfg.seed != test_cfg.seed

    val_dataset = HiddenCorridorDecisionDataset(
        config=val_cfg,
        num_episodes=benchmark.val_episodes,
        curriculum_levels=benchmark.curriculum_levels,
    )
    test_dataset = HiddenCorridorDecisionDataset(
        config=test_cfg,
        num_episodes=benchmark.test_episodes,
        curriculum_levels=benchmark.curriculum_levels,
    )
    assert val_dataset.manifest()["manifest_hash"] != test_dataset.manifest()["manifest_hash"]


def test_oracle_calibrated_deadlines_restore_feasible_routes() -> None:
    cfg = HiddenCorridorConfig(
        seed=12,
        deadline_mode="oracle_calibrated",
    )
    dataset = HiddenCorridorDecisionDataset(
        config=cfg,
        num_episodes=8,
        curriculum_levels=("single_dynamic", "multi_dynamic"),
    )
    audits = audit_oracle_deadlines(dataset.episodes, config=cfg)
    feasible_fraction = sum(audit.has_on_time_feasible_route for audit in audits) / len(audits)
    miss_fraction = sum(audit.oracle_deadline_missed for audit in audits) / len(audits)
    assert feasible_fraction > 0.9
    assert miss_fraction < 0.5


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
    assert "split_config_seeds" in summary
    assert "selected_epoch" in summary
    assert "selected_selection_score" in summary
    assert "p95_regret" in summary["test_rollout"]
    assert "deadline_miss_rate" in summary["test_rollout"]
    manifest_payload = json.loads(Path(summary["output_dir"], "dataset_manifests.json").read_text())
    assert manifest_payload["val"]["manifest_hash"] != manifest_payload["test"]["manifest_hash"]
