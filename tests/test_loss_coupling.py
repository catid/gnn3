from __future__ import annotations

import torch

from gnn3.models.packet_mamba import compute_losses


def _batch() -> dict[str, torch.Tensor]:
    return {
        "target_next_hop": torch.tensor([0], dtype=torch.long),
        "cost_to_go": torch.tensor([1.0], dtype=torch.float32),
        "node_mask": torch.tensor([[True, True, True]], dtype=torch.bool),
        "candidate_mask": torch.tensor([[True, True, True]], dtype=torch.bool),
        "route_relevance": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        "candidate_cost_to_go": torch.tensor([[1.0, 1.4, 4.0]], dtype=torch.float32),
        "candidate_on_time": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32),
        "candidate_slack": torch.tensor([[1.0, 0.6, -2.0]], dtype=torch.float32),
    }


def _batch_with_slack(candidate_slack: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = _batch()
    batch["candidate_slack"] = candidate_slack
    return batch


def _output(selection_scores: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "selection_scores": selection_scores,
        "path_scores": selection_scores.clone(),
        "values": torch.tensor([1.0], dtype=torch.float32),
        "route_logits": torch.tensor([[2.0, 2.0, -2.0]], dtype=torch.float32),
        "candidate_on_time_logits": None,
        "candidate_slack": None,
        "candidate_cost_quantiles": None,
        "per_step_candidate_on_time_logits": None,
        "per_step_candidate_slack": None,
        "per_step_candidate_cost_quantiles": None,
    }


def test_selection_soft_target_loss_prefers_feasible_low_cost_actions() -> None:
    batch = _batch()
    better = compute_losses(
        _output(torch.tensor([[3.0, 2.0, -1.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        selection_soft_target_weight=1.0,
        selection_soft_target_temperature=0.5,
        selection_soft_target_on_time_bonus=1.0,
    )
    worse = compute_losses(
        _output(torch.tensor([[-1.0, 0.0, 3.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        selection_soft_target_weight=1.0,
        selection_soft_target_temperature=0.5,
        selection_soft_target_on_time_bonus=1.0,
    )
    assert float(better["selection_soft_target_loss"]) < float(worse["selection_soft_target_loss"])
    assert float(better["loss"]) < float(worse["loss"])


def test_selection_pairwise_loss_prefers_feasible_low_cost_rankings() -> None:
    batch = _batch()
    better = compute_losses(
        _output(torch.tensor([[3.0, 2.0, -1.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        selection_pairwise_weight=1.0,
        selection_pairwise_temperature=1.0,
        selection_pairwise_on_time_bonus=1.0,
        selection_pairwise_slack_bonus=0.25,
        selection_pairwise_margin=0.5,
    )
    worse = compute_losses(
        _output(torch.tensor([[-1.0, 0.0, 3.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        selection_pairwise_weight=1.0,
        selection_pairwise_temperature=1.0,
        selection_pairwise_on_time_bonus=1.0,
        selection_pairwise_slack_bonus=0.25,
        selection_pairwise_margin=0.5,
    )
    assert float(better["selection_pairwise_loss"]) < float(worse["selection_pairwise_loss"])
    assert float(better["loss"]) < float(worse["loss"])


def test_path_soft_target_loss_prefers_feasible_low_cost_paths() -> None:
    batch = _batch()
    better = compute_losses(
        _output(torch.tensor([[3.0, 2.0, -1.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        path_soft_target_weight=1.0,
        path_soft_target_temperature=0.5,
        path_soft_target_on_time_bonus=1.0,
    )
    worse = compute_losses(
        _output(torch.tensor([[-1.0, 0.0, 3.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        path_soft_target_weight=1.0,
        path_soft_target_temperature=0.5,
        path_soft_target_on_time_bonus=1.0,
    )
    assert float(better["path_soft_target_loss"]) < float(worse["path_soft_target_loss"])
    assert float(better["loss"]) < float(worse["loss"])


def test_selection_feasible_target_loss_prefers_best_on_time_candidate() -> None:
    batch = _batch()
    better = compute_losses(
        _output(torch.tensor([[2.0, 3.0, -1.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        selection_feasible_target_weight=1.0,
    )
    worse = compute_losses(
        _output(torch.tensor([[-1.0, 3.0, 2.0]], dtype=torch.float32)),
        batch,
        final_step_only=True,
        selection_feasible_target_weight=1.0,
    )
    assert float(better["selection_feasible_target_loss"]) < float(worse["selection_feasible_target_loss"])
    assert float(better["loss"]) < float(worse["loss"])


def test_slack_critical_weighting_upweights_low_slack_decisions() -> None:
    logits = _output(torch.tensor([[0.3, 0.2, -1.0]], dtype=torch.float32))
    high_slack = compute_losses(
        logits,
        _batch_with_slack(torch.tensor([[2.0, 1.6, -2.0]], dtype=torch.float32)),
        final_step_only=True,
        selection_slack_critical_weight=1.0,
        selection_slack_critical_scale=0.5,
    )
    low_slack = compute_losses(
        logits,
        _batch_with_slack(torch.tensor([[0.1, -0.2, -2.0]], dtype=torch.float32)),
        final_step_only=True,
        selection_slack_critical_weight=1.0,
        selection_slack_critical_scale=0.5,
    )
    assert float(low_slack["loss"]) > float(high_slack["loss"])
