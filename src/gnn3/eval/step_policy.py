from __future__ import annotations

import torch

STEP_STRATEGIES = (
    "final",
    "first",
    "middle",
    "max_margin",
    "earliest_final_agreement",
)


def _per_step_scores(output: dict[str, torch.Tensor]) -> torch.Tensor:
    if output.get("per_step_selection_scores") is not None:
        return output["per_step_selection_scores"]
    return output["per_step_logits"]


def _masked_step_logits(output: dict[str, torch.Tensor], candidate_mask: torch.Tensor) -> torch.Tensor:
    return _per_step_scores(output).masked_fill(~candidate_mask[:, None, :], -1e9)


def _top_margin(masked_per_step_logits: torch.Tensor) -> torch.Tensor:
    topk = torch.topk(masked_per_step_logits, k=min(2, masked_per_step_logits.size(-1)), dim=-1).values
    if topk.size(-1) == 1:
        return topk[..., 0]
    return topk[..., 0] - topk[..., 1]


def select_step_index(
    output: dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
    *,
    strategy: str,
) -> torch.Tensor:
    if strategy not in STEP_STRATEGIES:
        raise ValueError(f"Unknown step policy strategy: {strategy}")

    if strategy == "final":
        last_index = _per_step_scores(output).size(1) - 1
        return torch.full(
            (candidate_mask.size(0),),
            fill_value=last_index,
            device=candidate_mask.device,
            dtype=torch.long,
        )

    masked_logits = _masked_step_logits(output, candidate_mask)

    if strategy == "first":
        return torch.zeros((candidate_mask.size(0),), device=candidate_mask.device, dtype=torch.long)
    if strategy == "middle":
        return torch.full(
            (candidate_mask.size(0),),
            fill_value=masked_logits.size(1) // 2,
            device=candidate_mask.device,
            dtype=torch.long,
        )
    if strategy == "max_margin":
        margins = _top_margin(masked_logits)
        return margins.argmax(dim=1)

    final_action = output["selection_scores"].argmax(dim=-1)
    per_step_action = masked_logits.argmax(dim=-1)
    agrees = per_step_action == final_action[:, None]
    step_ids = torch.arange(masked_logits.size(1), device=masked_logits.device)[None, :]
    fallback = masked_logits.size(1) - 1
    masked_ids = torch.where(agrees, step_ids, torch.full_like(step_ids, fallback))
    return masked_ids.min(dim=1).values


def select_step_scores(
    output: dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
    *,
    strategy: str,
) -> torch.Tensor:
    if strategy == "final":
        return output["selection_scores"]

    masked_logits = _masked_step_logits(output, candidate_mask)
    best_step = select_step_index(output, candidate_mask, strategy=strategy)
    gather_index = best_step[:, None, None].expand(-1, 1, masked_logits.size(-1))
    return torch.gather(masked_logits, 1, gather_index).squeeze(1)
