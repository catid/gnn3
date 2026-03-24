from __future__ import annotations

import torch

from gnn3.eval.step_policy import select_step_scores


def _output() -> dict[str, torch.Tensor]:
    per_step = torch.tensor(
        [
            [
                [4.0, 1.0, -2.0],
                [3.0, 2.0, -2.0],
                [2.0, 3.5, -2.0],
            ],
            [
                [0.1, 2.5, -3.0],
                [3.5, 0.0, -3.0],
                [3.4, 0.1, -3.0],
            ],
        ],
        dtype=torch.float32,
    )
    return {
        "selection_scores": per_step[:, -1],
        "per_step_logits": per_step,
    }


def test_select_step_scores_returns_first_and_middle_steps() -> None:
    candidate_mask = torch.tensor([[True, True, False], [True, True, False]], dtype=torch.bool)
    output = _output()
    first = select_step_scores(output, candidate_mask, strategy="first")
    middle = select_step_scores(output, candidate_mask, strategy="middle")
    assert torch.equal(first, output["per_step_logits"][:, 0].masked_fill(~candidate_mask, -1e9))
    assert torch.equal(middle, output["per_step_logits"][:, 1].masked_fill(~candidate_mask, -1e9))


def test_select_step_scores_max_margin_and_earliest_final_agreement() -> None:
    candidate_mask = torch.tensor([[True, True, False], [True, True, False]], dtype=torch.bool)
    output = _output()
    max_margin = select_step_scores(output, candidate_mask, strategy="max_margin")
    stable = select_step_scores(output, candidate_mask, strategy="earliest_final_agreement")

    # sample 0 picks step 0 by margin but step 2 by final agreement
    assert int(max_margin[0].argmax().item()) == 0
    assert int(stable[0].argmax().item()) == 1
    # sample 1 already agrees with final on step 1, so stable picks step 1
    assert int(stable[1].argmax().item()) == 0
