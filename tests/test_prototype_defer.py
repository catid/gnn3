from __future__ import annotations

import torch
import torch.nn.functional as F

from gnn3.models.prototype_defer import (
    AdapterPrototypeDeferHead,
    AgreementMixturePrototypeDeferHead,
    AnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    AnchoredEvidenceAgreementPrototypeDeferHead,
    AsymmetricTailSupportAgreementMixturePrototypeDeferHead,
    BandpassPrototypeDeferHead,
    BranchCalibratedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    BranchStrengthNegativeCleanupLiftSupportAgreementMixturePrototypeDeferHead,
    BranchStrengthNegativeCleanupMaxSupportAgreementMixturePrototypeDeferHead,
    BranchStrengthSharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    BranchwiseLiftNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    BranchwiseMarginMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    BudgetConditionedEvidenceAgreementPrototypeDeferHead,
    BudgetConditionedMemoryBlendPrototypeDeferHead,
    CascadePrototypeDeferHead,
    ConfidentRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    ContrastiveEvidenceAgreementPrototypeDeferHead,
    DualProjectionPrototypeDeferHead,
    DualSharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    EvidenceAgreementPrototypeDeferHead,
    EvidencePrototypeDeferHead,
    FixedRescueAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    FixedTailMarginBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    FloorGatedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    FloorSharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    GatedPrototypeDeferHead,
    HardDedupBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    InteractionPrototypeDeferHead,
    JointSupportBranchwiseNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    LearnedGateNegativeTailSupportAgreementMixturePrototypeDeferHead,
    LiftPrototypeDeferHead,
    MassAwareSharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    MemoryAgreementBlendPrototypeDeferHead,
    MemoryCalibratedEvidenceAgreementPrototypeDeferHead,
    MemoryDualLiftBlendPrototypeDeferHead,
    MemoryEvidenceAgreementBlendPrototypeDeferHead,
    MixturePrototypeDeferHead,
    MultiscalePrototypeDeferHead,
    NegativeCleanupBlendSupportAgreementMixturePrototypeDeferHead,
    NegativeCleanupLiftSupportAgreementMixturePrototypeDeferHead,
    NegativeCleanupMaxSupportAgreementMixturePrototypeDeferHead,
    NegativeTailSupportAgreementMixturePrototypeDeferHead,
    PositiveLiftEvidenceAgreementPrototypeDeferHead,
    PrototypeMemoryDeferHead,
    PrototypeTriageDeferHead,
    PrunedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    RampedRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    RegimeSplitEvidenceAgreementPrototypeDeferHead,
    RegimeSplitMemoryBlendPrototypeDeferHead,
    RescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    ResidualRegimeEvidenceAgreementPrototypeDeferHead,
    RiskConditionedSupportAgreementMixturePrototypeDeferHead,
    RiskPriorRegimeMemoryBlendPrototypeDeferHead,
    RiskVetoRegimeMemoryBlendPrototypeDeferHead,
    SelectiveEvidenceAgreementPrototypeDeferHead,
    SharedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    SharpMassNegativeTailSupportAgreementMixturePrototypeDeferHead,
    SharpNegativeTailSupportAgreementMixturePrototypeDeferHead,
    SharpnessEvidenceAgreementPrototypeDeferHead,
    SoftTailBlendSupportAgreementMixturePrototypeDeferHead,
    SoftTailSupportAgreementMixturePrototypeDeferHead,
    SpecialistPrototypeDeferHead,
    SplitScaleSupportAgreementMixturePrototypeDeferHead,
    SupportedResidualRegimeEvidenceAgreementPrototypeDeferHead,
    SupportWeightedAgreementMixturePrototypeDeferHead,
    SupportWeightedEvidenceAgreementPrototypeDeferHead,
    SupportWeightedMemoryAgreementBlendPrototypeDeferHead,
    SupportWeightedPrototypeMemoryDeferHead,
    SuppressorPrototypeDeferHead,
    SwitchPrototypeDeferHead,
    TailMarginCalibratedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead,
    TeacherMarginMemoryBlendPrototypeDeferHead,
    TeacherSignalEvidenceAgreementPrototypeDeferHead,
    Top2NegativeTailSupportAgreementMixturePrototypeDeferHead,
    TopKSupportAgreementMixturePrototypeDeferHead,
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


def test_support_weighted_prototype_memory_defer_separates_toy_clusters() -> None:
    torch.manual_seed(17)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SupportWeightedPrototypeMemoryDeferHead(
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
        support_reg = head.support_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg
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


def test_support_weighted_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(38)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SupportWeightedEvidenceAgreementPrototypeDeferHead(
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
        support_reg = head.support_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_memory_calibrated_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(39)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = MemoryCalibratedEvidenceAgreementPrototypeDeferHead(
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


def test_teacher_signal_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(40)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    committee_target = labels.clone()
    gain_target = labels.clone()
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = TeacherSignalEvidenceAgreementPrototypeDeferHead(
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
        logits, committee_logits, gain_logits = head.forward_with_aux(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        committee_loss = F.binary_cross_entropy_with_logits(committee_logits, committee_target)
        gain_loss = F.binary_cross_entropy_with_logits(gain_logits, gain_target)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        loss = bce + 0.05 * (committee_loss + gain_loss) + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, committee_logits, gain_logits = head.forward_with_aux(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())
    assert float(committee_logits[: len(positives)].mean()) > float(committee_logits[len(positives) :].mean())
    assert float(gain_logits[: len(positives)].mean()) > float(gain_logits[len(positives) :].mean())


def test_budget_conditioned_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(43)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    budget = torch.tensor([[0.25, 0.5]] * len(features), dtype=torch.float32)
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BudgetConditionedEvidenceAgreementPrototypeDeferHead(
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
        logits = head(features, budget, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        loss = bce + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, budget, risk)
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


def test_positive_lift_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(47)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = PositiveLiftEvidenceAgreementPrototypeDeferHead(
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


def test_contrastive_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(53)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = ContrastiveEvidenceAgreementPrototypeDeferHead(
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


def test_sharpness_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(59)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SharpnessEvidenceAgreementPrototypeDeferHead(
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


def test_selective_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(67)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SelectiveEvidenceAgreementPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
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


def test_regime_split_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(71)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()
    regime_targets = torch.full((len(features),), -1, dtype=torch.long)
    regime_targets[: len(positives) // 2] = 0
    regime_targets[len(positives) // 2 : len(positives)] = 1

    head = RegimeSplitEvidenceAgreementPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits, regime_logits = head.forward_with_regime(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        regime_mask = regime_targets >= 0
        regime_loss = F.cross_entropy(regime_logits[regime_mask], regime_targets[regime_mask])
        reg = head.regularization(
            features,
            positive_mask=positive_mask,
            hard_negative_mask=hard_negatives,
            regime_targets=regime_targets,
        )
        loss = bce + 0.05 * regime_loss + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_residual_regime_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(73)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()
    regime_targets = torch.full((len(features),), -1, dtype=torch.long)
    regime_targets[: len(positives) // 2] = 0
    regime_targets[len(positives) // 2 : len(positives)] = 1

    head = ResidualRegimeEvidenceAgreementPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        logits, regime_logits = head.forward_with_regime(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        regime_mask = regime_targets >= 0
        regime_loss = F.cross_entropy(regime_logits[regime_mask], regime_targets[regime_mask])
        reg = head.regularization(
            features,
            positive_mask=positive_mask,
            hard_negative_mask=hard_negatives,
            regime_targets=regime_targets,
        )
        loss = bce + 0.05 * regime_loss + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_supported_residual_regime_evidence_agreement_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(79)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()
    regime_targets = torch.full((len(features),), -1, dtype=torch.long)
    regime_targets[: len(positives) // 2] = 0
    regime_targets[len(positives) // 2 : len(positives)] = 1

    head = SupportedResidualRegimeEvidenceAgreementPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        logits, regime_logits = head.forward_with_regime(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        regime_mask = regime_targets >= 0
        regime_loss = F.cross_entropy(regime_logits[regime_mask], regime_targets[regime_mask])
        reg = head.regularization(
            features,
            positive_mask=positive_mask,
            hard_negative_mask=hard_negatives,
            regime_targets=regime_targets,
        )
        loss = bce + 0.05 * regime_loss + 0.1 * reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_memory_agreement_blend_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(61)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = MemoryAgreementBlendPrototypeDeferHead(
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


def test_support_weighted_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(71)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SupportWeightedAgreementMixturePrototypeDeferHead(
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
        support_reg = head.support_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_split_scale_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(72)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SplitScaleSupportAgreementMixturePrototypeDeferHead(
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
        support_reg = head.support_regularization()
        scale_reg = head.scale_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * scale_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_topk_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(72)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = TopKSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
        pool_topk=2,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_soft_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(73)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SoftTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_soft_tail_blend_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(74)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SoftTailBlendSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(75)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = NegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_top2_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(86)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = Top2NegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_mass_aware_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(87)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = MassAwareSharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_sharp_mass_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(88)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SharpMassNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_asymmetric_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(76)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = AsymmetricTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(77)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_branch_calibrated_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(78)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BranchCalibratedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_branch_strength_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(79)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BranchStrengthSharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_branch_strength_negative_cleanup_lift_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(80)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BranchStrengthNegativeCleanupLiftSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_branch_strength_negative_cleanup_max_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(81)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BranchStrengthNegativeCleanupMaxSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(82)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_pruned_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(123)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = PrunedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        keep_reg = head.keep_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg + 0.002 * keep_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
        keep_summary = head.keep_summary()
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())
    assert keep_summary["shared_positive_keep_mean"] >= head.keep_floor
    assert keep_summary["shared_negative_keep_mean"] >= head.keep_floor


def test_hard_dedup_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(124)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = HardDedupBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
        negative_dedup_threshold=0.7,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_hard_dedup_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_drops_duplicate_negative_banks() -> None:
    head = HardDedupBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
        negative_dedup_threshold=0.8,
    )
    with torch.no_grad():
        shared_duplicate = F.normalize(torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=0)
        shared_other = F.normalize(torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=0)
        dual_duplicate = F.normalize(torch.tensor([0.0, 0.0, 1.0, 0.0]), dim=0)
        dual_other = F.normalize(torch.tensor([0.0, 0.0, 0.0, 1.0]), dim=0)
        head.shared_negative_prototypes.copy_(torch.stack([shared_duplicate, shared_duplicate, shared_other, -shared_other]))
        head.dual_negative_prototypes.copy_(torch.stack([dual_duplicate, dual_duplicate, dual_other, -dual_other]))
        head.shared_negative_support.copy_(torch.tensor([2.0, 1.0, 0.0, -1.0]))
        head.dual_negative_support.copy_(torch.tensor([2.0, 1.0, 0.0, -1.0]))

    summary = head.dedup_summary()
    assert summary["shared_negative_kept"] == 3.0
    assert summary["dual_negative_kept"] == 3.0


def test_tail_margin_calibrated_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(125)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = TailMarginCalibratedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        margin_reg = head.margin_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg + 0.01 * margin_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
        summary = head.margin_summary()
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())
    assert 0.1 <= summary["shared_fixed_margin"] <= 0.9
    assert 0.1 <= summary["dual_sharp_margin"] <= 0.9


def test_fixed_tail_margin_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(126)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = FixedTailMarginBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
        shared_fixed_margin=0.35,
        shared_sharp_margin=0.55,
        dual_fixed_margin=0.65,
        dual_sharp_margin=0.45,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
        summary = head.margin_summary()
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())
    assert summary == {
        "shared_fixed_margin": 0.35,
        "shared_sharp_margin": 0.55,
        "dual_fixed_margin": 0.65,
        "dual_sharp_margin": 0.45,
    }


def test_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(128)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = AnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_fixed_rescue_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(129)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = FixedRescueAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(130)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = RescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_confident_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(132)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = ConfidentRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_ramped_rescue_weighted_anchored_dual_lift_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(131)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = RampedRescueWeightedAnchoredDualLiftBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_hard_negative_conditioned_branchwise_max_negative_cleanup_support_agreement_mixture_prototype_defer_masks_and_separates() -> None:
    torch.manual_seed(127)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = HardNegativeConditionedBranchwiseMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    head.set_negative_keep_masks(
        shared_keep_mask=torch.tensor([True, False, True, False]),
        dual_keep_mask=torch.tensor([True, True, False, False]),
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
        summary = head.keep_summary()
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())
    assert summary == {
        "shared_negative_kept": 2.0,
        "shared_negative_total": 4.0,
        "dual_negative_kept": 2.0,
        "dual_negative_total": 4.0,
    }


def test_branchwise_lift_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(83)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BranchwiseLiftNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_branchwise_margin_max_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(84)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BranchwiseMarginMaxNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_joint_support_branchwise_negative_cleanup_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(85)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = JointSupportBranchwiseNegativeCleanupSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_learned_gate_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(86)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = LearnedGateNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_floor_gated_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(82)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = FloorGatedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_negative_cleanup_blend_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(83)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = NegativeCleanupBlendSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_negative_cleanup_max_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(84)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = NegativeCleanupMaxSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_negative_cleanup_lift_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(85)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = NegativeCleanupLiftSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_dual_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(81)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = DualSharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_shared_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(80)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SharedSharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_floor_sharp_negative_tail_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(78)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = FloorSharpNegativeTailSupportAgreementMixturePrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=4,
        negative_prototypes=4,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        support_reg = head.support_regularization()
        tail_reg = head.tail_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * tail_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_risk_conditioned_support_agreement_mixture_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(73)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = RiskConditionedSupportAgreementMixturePrototypeDeferHead(
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
        support_reg = head.support_regularization()
        dynamic_support_reg = head.dynamic_support_regularization(risk)
        loss = bce + 0.1 * reg + 0.01 * support_reg + 0.01 * dynamic_support_reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = head(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())


def test_support_weighted_memory_agreement_blend_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(83)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = SupportWeightedMemoryAgreementBlendPrototypeDeferHead(
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
        support_reg = head.support_regularization()
        loss = bce + 0.1 * reg + 0.01 * support_reg
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


def test_memory_evidence_agreement_blend_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(67)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = MemoryEvidenceAgreementBlendPrototypeDeferHead(
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


def test_memory_duallift_blend_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(71)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = MemoryDualLiftBlendPrototypeDeferHead(
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


def test_budget_conditioned_memory_blend_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(73)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = BudgetConditionedMemoryBlendPrototypeDeferHead(
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


def test_teacher_margin_memory_blend_prototype_defer_separates_toy_clusters() -> None:
    torch.manual_seed(79)
    positives = torch.randn(24, 2) * 0.15 + torch.tensor([1.0, 1.0])
    negatives = torch.randn(24, 2) * 0.15 + torch.tensor([-1.0, -1.0])
    features = torch.cat([positives, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(positives)), torch.zeros(len(negatives))], dim=0)
    gain_target = torch.cat([torch.full((len(positives),), 0.8), torch.zeros(len(negatives))], dim=0)
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(positives) :] = True
    positive_mask = labels.bool()

    head = TeacherMarginMemoryBlendPrototypeDeferHead(
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
        logits, predicted_gain = head.forward_with_gain(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        gain_loss = F.smooth_l1_loss(predicted_gain[positive_mask], gain_target[positive_mask])
        loss = bce + 0.1 * reg + 0.1 * gain_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, predicted_gain = head.forward_with_gain(features, risk)
    assert float(logits[: len(positives)].mean()) > float(logits[len(positives) :].mean())
    assert abs(float(predicted_gain[: len(positives)].mean()) - 0.8) < 0.15


def test_regime_split_memory_blend_prototype_defer_separates_positive_regimes() -> None:
    torch.manual_seed(83)
    headroom = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, 1.0])
    residual = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, -1.0])
    negatives = torch.randn(24, 2) * 0.12 + torch.tensor([-1.0, -1.0])
    features = torch.cat([headroom, residual, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(headroom) + len(residual)), torch.zeros(len(negatives))], dim=0)
    regime_labels = torch.cat(
        [
            torch.ones(len(headroom), dtype=torch.long),
            2 * torch.ones(len(residual), dtype=torch.long),
            torch.zeros(len(negatives), dtype=torch.long),
        ],
        dim=0,
    )
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(headroom) + len(residual) :] = True
    positive_mask = labels.bool()

    head = RegimeSplitMemoryBlendPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits, regime_logits = head.forward_with_regime(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        ce = F.cross_entropy(regime_logits[positive_mask], regime_labels[positive_mask])
        loss = bce + 0.1 * reg + 0.1 * ce
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, regime_logits = head.forward_with_regime(features, risk)
        regime_probs = torch.softmax(regime_logits, dim=1)
    assert float(logits[: len(headroom) + len(residual)].mean()) > float(logits[len(headroom) + len(residual) :].mean())
    assert float(regime_probs[: len(headroom), 1].mean()) > float(regime_probs[: len(headroom), 2].mean())
    assert float(regime_probs[len(headroom) : len(headroom) + len(residual), 2].mean()) > float(
        regime_probs[len(headroom) : len(headroom) + len(residual), 1].mean()
    )


def test_risk_prior_regime_memory_blend_prototype_defer_separates_positive_regimes() -> None:
    torch.manual_seed(89)
    headroom = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, 1.0])
    residual = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, -1.0])
    negatives = torch.randn(24, 2) * 0.12 + torch.tensor([-1.0, -1.0])
    features = torch.cat([headroom, residual, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(headroom) + len(residual)), torch.zeros(len(negatives))], dim=0)
    regime_labels = torch.cat(
        [
            torch.ones(len(headroom), dtype=torch.long),
            2 * torch.ones(len(residual), dtype=torch.long),
            torch.zeros(len(negatives), dtype=torch.long),
        ],
        dim=0,
    )
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(headroom) + len(residual) :] = True
    positive_mask = labels.bool()

    head = RiskPriorRegimeMemoryBlendPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits, regime_logits = head.forward_with_regime(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        ce = F.cross_entropy(regime_logits[positive_mask], regime_labels[positive_mask])
        loss = bce + 0.1 * reg + 0.1 * ce
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, regime_logits = head.forward_with_regime(features, risk)
        regime_probs = torch.softmax(regime_logits, dim=1)
    assert float(logits[: len(headroom) + len(residual)].mean()) > float(logits[len(headroom) + len(residual) :].mean())
    assert float(regime_probs[: len(headroom), 1].mean()) > float(regime_probs[: len(headroom), 2].mean())
    assert float(regime_probs[len(headroom) : len(headroom) + len(residual), 2].mean()) > float(
        regime_probs[len(headroom) : len(headroom) + len(residual), 1].mean()
    )


def test_risk_veto_regime_memory_blend_prototype_defer_separates_positive_regimes() -> None:
    torch.manual_seed(97)
    headroom = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, 1.0])
    residual = torch.randn(18, 2) * 0.12 + torch.tensor([1.0, -1.0])
    negatives = torch.randn(24, 2) * 0.12 + torch.tensor([-1.0, -1.0])
    features = torch.cat([headroom, residual, negatives], dim=0)
    risk = features.clone()
    labels = torch.cat([torch.ones(len(headroom) + len(residual)), torch.zeros(len(negatives))], dim=0)
    regime_labels = torch.cat(
        [
            torch.ones(len(headroom), dtype=torch.long),
            2 * torch.ones(len(residual), dtype=torch.long),
            torch.zeros(len(negatives), dtype=torch.long),
        ],
        dim=0,
    )
    veto_labels = torch.zeros((len(features), 2), dtype=torch.float32)
    veto_labels[: len(headroom), 0] = 1.0
    veto_labels[len(headroom) : len(headroom) + len(residual), 1] = 1.0
    hard_negatives = torch.zeros(len(features), dtype=torch.bool)
    hard_negatives[len(headroom) + len(residual) :] = True
    positive_mask = labels.bool()

    head = RiskVetoRegimeMemoryBlendPrototypeDeferHead(
        feature_dim=2,
        risk_dim=2,
        prototype_dim=4,
        positive_prototypes=2,
        negative_prototypes=2,
        hidden_dim=8,
        use_risk_branch=True,
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-2, weight_decay=1e-4)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        logits, regime_logits, veto_logits = head.forward_with_regime(features, risk)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        reg = head.regularization(features, positive_mask=positive_mask, hard_negative_mask=hard_negatives)
        ce = F.cross_entropy(regime_logits[positive_mask], regime_labels[positive_mask])
        veto = F.binary_cross_entropy_with_logits(veto_logits, veto_labels)
        loss = bce + 0.1 * reg + 0.1 * ce + 0.05 * veto
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, regime_logits, veto_logits = head.forward_with_regime(features, risk)
        regime_probs = torch.softmax(regime_logits, dim=1)
        veto_masks = torch.sigmoid(veto_logits)
    assert float(logits[: len(headroom) + len(residual)].mean()) > float(logits[len(headroom) + len(residual) :].mean())
    assert float(regime_probs[: len(headroom), 1].mean()) > float(regime_probs[: len(headroom), 2].mean())
    assert float(regime_probs[len(headroom) : len(headroom) + len(residual), 2].mean()) > float(
        regime_probs[len(headroom) : len(headroom) + len(residual), 1].mean()
    )
    assert float(veto_masks[: len(headroom), 0].mean()) > float(veto_masks[: len(headroom), 1].mean())
    assert float(veto_masks[len(headroom) : len(headroom) + len(residual), 1].mean()) > float(
        veto_masks[len(headroom) : len(headroom) + len(residual), 0].mean()
    )
