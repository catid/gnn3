from __future__ import annotations

import torch
import torch.nn.functional as F

from gnn3.models.prototype_defer import (
    AdapterPrototypeDeferHead,
    AgreementMixturePrototypeDeferHead,
    AnchoredEvidenceAgreementPrototypeDeferHead,
    BandpassPrototypeDeferHead,
    CascadePrototypeDeferHead,
    DualProjectionPrototypeDeferHead,
    EvidenceAgreementPrototypeDeferHead,
    EvidencePrototypeDeferHead,
    GatedPrototypeDeferHead,
    InteractionPrototypeDeferHead,
    LiftPrototypeDeferHead,
    MixturePrototypeDeferHead,
    MultiscalePrototypeDeferHead,
    PrototypeMemoryDeferHead,
    PrototypeTriageDeferHead,
    SpecialistPrototypeDeferHead,
    SuppressorPrototypeDeferHead,
    SwitchPrototypeDeferHead,
)


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


def test_dual_projection_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(11)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = DualProjectionPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(13)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = MixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(23)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = AgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_cascade_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(29)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = CascadePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_switch_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(31)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SwitchPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(37)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = EvidenceAgreementPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_anchor_evidence_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(41)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = AnchoredEvidenceAgreementPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_gated_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(17)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = GatedPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_bias_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_adapter_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(19)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = AdapterPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_multiscale_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(29)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = MultiscalePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_evidence_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(31)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = EvidencePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=3,
        negative_prototypes=3,
        evidence_topk=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
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


def test_bandpass_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(37)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BandpassPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        band_width=1.0,
        band_sharpness=2.0,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_suppressor_prototype_defer_separates_positive_from_neutral_and_harmful() -> None:
    torch.manual_seed(41)
    positives = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, 1.0])
    neutrals = torch.randn(18, 2) * 0.12 + torch.tensor([-1.0, -1.0])
    harmful = torch.randn(18, 2) * 0.12 + torch.tensor([-1.0, 1.0])
    features = torch.cat([positives, neutrals, harmful], dim=0)
    risk = features.clone()
    labels = torch.cat(
        [
            torch.ones(len(positives)),
            torch.zeros(len(neutrals) + len(harmful)),
        ],
        dim=0,
    )
    positive_mask = torch.zeros(len(features), dtype=torch.bool)
    neutral_mask = torch.zeros(len(features), dtype=torch.bool)
    harmful_mask = torch.zeros(len(features), dtype=torch.bool)
    positive_mask[: len(positives)] = True
    neutral_mask[len(positives) : len(positives) + len(neutrals)] = True
    harmful_mask[len(positives) + len(neutrals) :] = True

    head = SuppressorPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        neutral_prototypes=2,
        harmful_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(
            features,
            positive_mask=positive_mask,
            neutral_negative_mask=neutral_mask,
            harmful_negative_mask=harmful_mask,
        )
        loss = bce + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    positive_mean = float(logits[: len(positives)].mean())
    negative_mean = float(logits[len(positives) :].mean())
    assert positive_mean > negative_mean


def test_interaction_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(43)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = InteractionPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        gated=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_lift_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(47)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = LiftPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        gated=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
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


def test_specialist_prototype_defer_separates_two_positive_clusters() -> None:
    torch.manual_seed(23)
    headroom = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, 1.0])
    residual = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, -1.0])
    negatives = torch.randn(24, 2) * 0.12 + torch.tensor([-1.0, -1.0])
    features = torch.cat([headroom, residual, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat(
        [
            torch.ones(len(headroom) + len(residual)),
            torch.zeros(len(negatives)),
        ],
        dim=0,
    )
    headroom_mask = torch.zeros(len(features), dtype=torch.bool)
    residual_mask = torch.zeros(len(features), dtype=torch.bool)
    negative_mask = torch.zeros(len(features), dtype=torch.bool)
    headroom_mask[: len(headroom)] = True
    residual_mask[len(headroom) : len(headroom) + len(residual)] = True
    negative_mask[len(headroom) + len(residual) :] = True

    head = SpecialistPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        headroom_prototypes=2,
        residual_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_gate=True,
        use_bias_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(
            features,
            headroom_positive_mask=headroom_mask,
            residual_positive_mask=residual_mask,
            hard_negative_mask=negative_mask,
        )
        loss = bce + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    positive_mean = float(logits[: len(headroom) + len(residual)].mean())
    negative_mean = float(logits[len(headroom) + len(residual) :].mean())
    assert positive_mean > negative_mean


def test_prototype_triage_defer_separates_three_way_clusters() -> None:
    torch.manual_seed(11)
    positives = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, 1.0])
    neutrals = torch.randn(18, 2) * 0.12 + torch.tensor([0.0, 0.0])
    harmful = torch.randn(18, 2) * 0.12 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, neutrals, harmful], dim=0)
    risk = features.clone()
    labels = torch.cat(
        [
            torch.zeros(len(positives), dtype=torch.long),
            torch.ones(len(neutrals), dtype=torch.long),
            2 * torch.ones(len(harmful), dtype=torch.long),
        ],
        dim=0,
    )

    head = PrototypeTriageDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        neutral_prototypes=2,
        harmful_prototypes=2,
        hidden_dim=8,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head.class_logits(features, risk)
        ce = F.cross_entropy(logits, labels)
        reg = head.regularization(
            features,
            positive_mask=labels == 0,
            neutral_mask=labels == 1,
            harmful_mask=labels == 2,
        )
        loss = ce + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        scores = head(features, risk)
    assert float(scores[: len(positives)].mean()) > 0.0
    assert float(scores[len(positives) :].mean()) < float(scores[: len(positives)].mean())
