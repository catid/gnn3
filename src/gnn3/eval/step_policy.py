from __future__ import annotations

import torch

STEP_STRATEGIES = (
    "final",
    "first",
    "middle",
    "max_margin",
    "earliest_final_agreement",
)


def _masked_step_logits(per_step_logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return per_step_logits.masked_fill(~candidate_mask[:, None, :], -1e9)


def _top_margin(masked_per_step_logits: torch.Tensor) -> torch.Tensor:
    topk = torch.topk(masked_per_step_logits, k=min(2, masked_per_step_logits.size(-1)), dim=-1).values
    if topk.size(-1) == 1:
        return topk[..., 0]
    return topk[..., 0] - topk[..., 1]


def select_step_scores(
    output: dict[str, torch.Tensor],
    candidate_mask: torch.Tensor,
    *,
    strategy: str,
) -> torch.Tensor:
    if strategy not in STEP_STRATEGIES:
        raise ValueError(f"Unknown step policy strategy: {strategy}")

    if strategy == "final":
        return output["selection_scores"]

    per_step_logits = output["per_step_logits"]
    masked_logits = _masked_step_logits(per_step_logits, candidate_mask)

    if strategy == "first":
        return masked_logits[:, 0]
    if strategy == "middle":
        return masked_logits[:, masked_logits.size(1) // 2]
    if strategy == "max_margin":
        margins = _top_margin(masked_logits)
        best_step = margins.argmax(dim=1)
    else:
        final_action = output["selection_scores"].argmax(dim=-1)
        per_step_action = masked_logits.argmax(dim=-1)
        agrees = per_step_action == final_action[:, None]
        step_ids = torch.arange(masked_logits.size(1), device=masked_logits.device)[None, :]
        fallback = masked_logits.size(1) - 1
        masked_ids = torch.where(agrees, step_ids, torch.full_like(step_ids, fallback))
        best_step = masked_ids.min(dim=1).values

    gather_index = best_step[:, None, None].expand(-1, 1, masked_logits.size(-1))
    return torch.gather(masked_logits, 1, gather_index).squeeze(1)
