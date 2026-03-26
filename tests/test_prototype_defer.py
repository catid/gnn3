from __future__ import annotations

import torch
import torch.nn.functional as F

from gnn3.models.prototype_defer import PrototypeMemoryDeferHead


def test_prototype_memory_defer_forward_shape() -> None:
    head = PrototypeMemoryDeferHead(feature_dim=6, risk_dim=3, prototype_dim=4, positive_prototypes=2, negative_prototypes=2)
    logits = head(torch.randn(5, 6), torch.randn(5, 3))
    assert logits.shape == (5,)


def test_prototype_memory_defer_separates_toy_clusters() -> None:
    torch.manual_seed(7)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = PrototypeMemoryDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        loss = bce + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())
