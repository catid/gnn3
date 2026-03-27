from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class PrototypeMemoryDeferHead(torch.nn.Module):
    """Tiny prototype bank for sparse defer/correct decisions.

    The head learns positive and negative prototypes in a projected frozen-feature
    space, then optionally adds a small risk branch over cheaper regime features.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _prototype_score(self, encoded: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        pos_score = torch.logsumexp(scale * encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(features)
        logits = self._prototype_score(encoded) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class DualProjectionPrototypeDeferHead(torch.nn.Module):
    """Prototype bank with separate positive and negative query projections."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.positive_feature_norm(self.positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.negative_feature_norm(self.negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.positive_prototypes, dim=-1), F.normalize(self.negative_prototypes, dim=-1)

    def _prototype_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        positive_encoded = self.encode_positive(features)
        negative_encoded = self.encode_negative(features)
        logits = self._prototype_score(positive_encoded, negative_encoded) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        pos_bank, neg_bank = self._banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            positive_encoded = self.encode_positive(pos_features)
            positive_encoded_for_negative = self.encode_negative(pos_features)
            pos_alignment = (positive_encoded @ pos_bank.T).amax(dim=1)
            neg_overlap = (positive_encoded_for_negative @ neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            negative_encoded = self.encode_negative(neg_features)
            negative_encoded_for_positive = self.encode_positive(neg_features)
            neg_alignment = (negative_encoded @ neg_bank.T).amax(dim=1)
            pos_overlap = (negative_encoded_for_positive @ pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class MixturePrototypeDeferHead(torch.nn.Module):
    """Mixture of shared-projection and dual-projection prototype geometry."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.mix_logit = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)
        mix = torch.sigmoid(self.mix_logit)
        logits = (1.0 - mix) * shared_score + mix * dual_score + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class AgreementMixturePrototypeDeferHead(torch.nn.Module):
    """Shared/dual prototype mixture gated by geometry agreement."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.agreement_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)
        gate_features = self._agreement_features(shared_score, dual_score)
        gate = torch.sigmoid(self.agreement_gate(gate_features).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class SupportWeightedAgreementMixturePrototypeDeferHead(AgreementMixturePrototypeDeferHead):
    """Agreement mixture with bounded per-prototype support weights."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
        support_scale: float = 2.0,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.support_scale = support_scale
        self.shared_positive_support = torch.nn.Parameter(torch.zeros(positive_prototypes, dtype=torch.float32))
        self.shared_negative_support = torch.nn.Parameter(torch.zeros(negative_prototypes, dtype=torch.float32))
        self.dual_positive_support = torch.nn.Parameter(torch.zeros(positive_prototypes, dtype=torch.float32))
        self.dual_negative_support = torch.nn.Parameter(torch.zeros(negative_prototypes, dtype=torch.float32))

    def _bounded_support(self, raw_support: torch.Tensor) -> torch.Tensor:
        centered = raw_support - raw_support.mean()
        return self.support_scale * torch.tanh(centered)

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T + self._bounded_support(self.shared_positive_support)
        neg_logits = scale * shared_encoded @ neg.T + self._bounded_support(self.shared_negative_support)
        return torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(neg_logits, dim=1)

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T + self._bounded_support(self.dual_positive_support)
        neg_logits = scale * negative_encoded @ neg.T + self._bounded_support(self.dual_negative_support)
        return torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(neg_logits, dim=1)

    def support_regularization(self) -> torch.Tensor:
        penalties = []
        for support in (
            self.shared_positive_support,
            self.shared_negative_support,
            self.dual_positive_support,
            self.dual_negative_support,
        ):
            penalties.append(self._bounded_support(support).abs().mean())
        return torch.stack(penalties).mean()


class EvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Agreement mixture with branch-internal prototype evidence features."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.evidence_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _dual_evidence(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _evidence_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        gate_features = self._evidence_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        gate = torch.sigmoid(self.evidence_gate(gate_features).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class TeacherSignalEvidenceAgreementPrototypeDeferHead(EvidenceAgreementPrototypeDeferHead):
    """Evidence-agreement mixture with auxiliary teacher-signal predictions."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.evidence_trunk = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
        )
        self.evidence_gate_head = torch.nn.Linear(hidden_dim, 1)
        self.committee_head = torch.nn.Linear(hidden_dim, 1)
        self.gain_head = torch.nn.Linear(hidden_dim, 1)

    def forward_with_aux(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        gate_features = self._evidence_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        hidden = self.evidence_trunk(gate_features)
        gate = torch.sigmoid(self.evidence_gate_head(hidden).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, self.committee_head(hidden).squeeze(-1), self.gain_head(hidden).squeeze(-1)

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits, _, _ = self.forward_with_aux(features, risk_features)
        return logits


class SelectiveEvidenceAgreementPrototypeDeferHead(EvidenceAgreementPrototypeDeferHead):
    """Evidence-agreement head with per-state prototype subset selection."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.shared_positive_selector = torch.nn.Linear(prototype_dim, prototype_dim, bias=False)
        self.shared_negative_selector = torch.nn.Linear(prototype_dim, prototype_dim, bias=False)
        self.dual_positive_selector = torch.nn.Linear(prototype_dim, prototype_dim, bias=False)
        self.dual_negative_selector = torch.nn.Linear(prototype_dim, prototype_dim, bias=False)
        self.selector_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))

    def _selected_bank_score(
        self,
        encoded: torch.Tensor,
        bank: torch.Tensor,
        selector: torch.nn.Linear,
        *,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(selector(encoded), dim=-1)
        selector_scale = self.selector_scale.exp().clamp(min=1.0, max=64.0)
        weights = F.softmax(selector_scale * (query @ bank.T), dim=1)
        selected = F.normalize(weights @ bank, dim=-1)
        score = scale * (encoded * selected).sum(dim=1)
        top = (scale * encoded @ bank.T).amax(dim=1)
        return score, top

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        shared_scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        dual_scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()

        shared_pos_score, shared_pos_top = self._selected_bank_score(
            shared_encoded,
            shared_pos_bank,
            self.shared_positive_selector,
            scale=shared_scale,
        )
        shared_neg_score, shared_neg_top = self._selected_bank_score(
            shared_encoded,
            shared_neg_bank,
            self.shared_negative_selector,
            scale=shared_scale,
        )
        dual_pos_score, dual_pos_top = self._selected_bank_score(
            dual_pos_encoded,
            dual_pos_bank,
            self.dual_positive_selector,
            scale=dual_scale,
        )
        dual_neg_score, dual_neg_top = self._selected_bank_score(
            dual_neg_encoded,
            dual_neg_bank,
            self.dual_negative_selector,
            scale=dual_scale,
        )

        shared_score = shared_pos_score - shared_neg_score
        dual_score = dual_pos_score - dual_neg_score
        gate_features = self._evidence_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        gate = torch.sigmoid(self.evidence_gate(gate_features).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits


class RegimeSplitEvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Separate evidence-agreement banks for headroom and residual regimes."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)

        self.headroom_shared_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.headroom_shared_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )
        self.headroom_dual_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.headroom_dual_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )

        self.residual_shared_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.residual_shared_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )
        self.residual_dual_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.residual_dual_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.headroom_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.residual_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        regime_input_dim = 8 + (risk_dim if use_risk_branch and risk_dim > 0 else 0)
        self.regime_head = torch.nn.Sequential(
            torch.nn.Linear(regime_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        self.headroom_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.residual_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self._regime_uses_risk = use_risk_branch and risk_dim > 0
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks_for_regime(self, regime: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if regime == "headroom":
            return (
                F.normalize(self.headroom_shared_positive_prototypes, dim=-1),
                F.normalize(self.headroom_shared_negative_prototypes, dim=-1),
                F.normalize(self.headroom_dual_positive_prototypes, dim=-1),
                F.normalize(self.headroom_dual_negative_prototypes, dim=-1),
            )
        if regime == "residual":
            return (
                F.normalize(self.residual_shared_positive_prototypes, dim=-1),
                F.normalize(self.residual_shared_negative_prototypes, dim=-1),
                F.normalize(self.residual_dual_positive_prototypes, dim=-1),
                F.normalize(self.residual_dual_negative_prototypes, dim=-1),
            )
        raise ValueError(f"Unknown regime: {regime}")

    def _evidence_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def _regime_features(
        self,
        headroom_score: torch.Tensor,
        residual_score: torch.Tensor,
        headroom_shared_margin: torch.Tensor,
        headroom_dual_margin: torch.Tensor,
        residual_shared_margin: torch.Tensor,
        residual_dual_margin: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = residual_score - headroom_score
        base = torch.stack(
            [
                headroom_score,
                residual_score,
                diff.abs(),
                headroom_score * residual_score,
                headroom_shared_margin,
                headroom_dual_margin,
                residual_shared_margin,
                residual_dual_margin,
            ],
            dim=1,
        )
        if self._regime_uses_risk and risk_features is not None:
            return torch.cat([base, risk_features], dim=1)
        return base

    def _regime_score(
        self,
        regime: str,
        shared_encoded: torch.Tensor,
        dual_pos_encoded: torch.Tensor,
        dual_neg_encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_pos_bank, shared_neg_bank, dual_pos_bank, dual_neg_bank = self._banks_for_regime(regime)
        shared_scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        dual_scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)

        shared_pos_logits = shared_scale * shared_encoded @ shared_pos_bank.T
        shared_neg_logits = shared_scale * shared_encoded @ shared_neg_bank.T
        dual_pos_logits = dual_scale * dual_pos_encoded @ dual_pos_bank.T
        dual_neg_logits = dual_scale * dual_neg_encoded @ dual_neg_bank.T

        shared_pos_top = shared_pos_logits.amax(dim=1)
        shared_neg_top = shared_neg_logits.amax(dim=1)
        dual_pos_top = dual_pos_logits.amax(dim=1)
        dual_neg_top = dual_neg_logits.amax(dim=1)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        gate_features = self._evidence_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        gate_net = self.headroom_gate if regime == "headroom" else self.residual_gate
        gate_bias = self.headroom_gate_bias if regime == "headroom" else self.residual_gate_bias
        gate = torch.sigmoid(gate_net(gate_features).squeeze(-1) + gate_bias)
        score = shared_score + gate * (dual_score - shared_score)
        return score, shared_pos_top - shared_neg_top, dual_pos_top - dual_neg_top

    def forward_with_regime(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        headroom_score, headroom_shared_margin, headroom_dual_margin = self._regime_score(
            "headroom",
            shared_encoded,
            dual_pos_encoded,
            dual_neg_encoded,
        )
        residual_score, residual_shared_margin, residual_dual_margin = self._regime_score(
            "residual",
            shared_encoded,
            dual_pos_encoded,
            dual_neg_encoded,
        )
        regime_logits = self.regime_head(
            self._regime_features(
                headroom_score,
                residual_score,
                headroom_shared_margin,
                headroom_dual_margin,
                residual_shared_margin,
                residual_dual_margin,
                risk_features,
            )
        )
        regime_probs = F.softmax(regime_logits, dim=1)
        logits = regime_probs[:, 0] * headroom_score + regime_probs[:, 1] * residual_score + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, regime_logits

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_regime(features, risk_features)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
        regime_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_positive_encoded = self.encode_dual_positive(features)
        dual_negative_encoded = self.encode_dual_negative(features)
        loss = features.new_tensor(0.0)

        for regime_name, regime_index in (("headroom", 0), ("residual", 1)):
            shared_pos_bank, shared_neg_bank, dual_pos_bank, dual_neg_bank = self._banks_for_regime(regime_name)
            if regime_targets is None:
                regime_positive_mask = positive_mask
            else:
                regime_positive_mask = positive_mask & (regime_targets == regime_index)
            if bool(regime_positive_mask.any()):
                regime_shared = shared_encoded[regime_positive_mask]
                regime_dual_positive = dual_positive_encoded[regime_positive_mask]
                regime_dual_negative = dual_negative_encoded[regime_positive_mask]
                shared_pos_alignment = (regime_shared @ shared_pos_bank.T).amax(dim=1)
                shared_neg_overlap = (regime_shared @ shared_neg_bank.T).amax(dim=1)
                dual_pos_alignment = (regime_dual_positive @ dual_pos_bank.T).amax(dim=1)
                dual_neg_overlap = (regime_dual_negative @ dual_neg_bank.T).amax(dim=1)
                loss = loss + (1.0 - shared_pos_alignment).mean()
                loss = loss + F.relu(shared_neg_overlap - margin).mean()
                loss = loss + (1.0 - dual_pos_alignment).mean()
                loss = loss + F.relu(dual_neg_overlap - margin).mean()

            if bool(hard_negative_mask.any()):
                hard_shared = shared_encoded[hard_negative_mask]
                hard_dual_positive = dual_positive_encoded[hard_negative_mask]
                hard_dual_negative = dual_negative_encoded[hard_negative_mask]
                shared_neg_alignment = (hard_shared @ shared_neg_bank.T).amax(dim=1)
                shared_pos_overlap = (hard_shared @ shared_pos_bank.T).amax(dim=1)
                dual_neg_alignment = (hard_dual_negative @ dual_neg_bank.T).amax(dim=1)
                dual_pos_overlap = (hard_dual_positive @ dual_pos_bank.T).amax(dim=1)
                loss = loss + (1.0 - shared_neg_alignment).mean()
                loss = loss + F.relu(shared_pos_overlap - margin).mean()
                loss = loss + (1.0 - dual_neg_alignment).mean()
                loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class ResidualRegimeEvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Evidence-agreement anchor with bounded regime-specific positive lifts."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)

        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.headroom_shared_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.headroom_shared_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )
        self.headroom_dual_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.headroom_dual_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )

        self.residual_shared_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.residual_shared_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )
        self.residual_dual_positive_prototypes = torch.nn.Parameter(
            torch.randn(positive_prototypes, prototype_dim) * scale
        )
        self.residual_dual_negative_prototypes = torch.nn.Parameter(
            torch.randn(negative_prototypes, prototype_dim) * scale
        )

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.anchor_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.anchor_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.headroom_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.headroom_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.residual_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.residual_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        regime_input_dim = 8 + (risk_dim if use_risk_branch and risk_dim > 0 else 0)
        self.regime_head = torch.nn.Sequential(
            torch.nn.Linear(regime_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self._regime_uses_risk = use_risk_branch and risk_dim > 0
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks_for_regime(self, regime: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if regime == "anchor":
            return (
                F.normalize(self.shared_positive_prototypes, dim=-1),
                F.normalize(self.shared_negative_prototypes, dim=-1),
                F.normalize(self.dual_positive_prototypes, dim=-1),
                F.normalize(self.dual_negative_prototypes, dim=-1),
            )
        if regime == "headroom":
            return (
                F.normalize(self.headroom_shared_positive_prototypes, dim=-1),
                F.normalize(self.headroom_shared_negative_prototypes, dim=-1),
                F.normalize(self.headroom_dual_positive_prototypes, dim=-1),
                F.normalize(self.headroom_dual_negative_prototypes, dim=-1),
            )
        if regime == "residual":
            return (
                F.normalize(self.residual_shared_positive_prototypes, dim=-1),
                F.normalize(self.residual_shared_negative_prototypes, dim=-1),
                F.normalize(self.residual_dual_positive_prototypes, dim=-1),
                F.normalize(self.residual_dual_negative_prototypes, dim=-1),
            )
        raise ValueError(f"Unknown regime: {regime}")

    def _evidence_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def _regime_features(
        self,
        anchor_score: torch.Tensor,
        headroom_score: torch.Tensor,
        residual_score: torch.Tensor,
        headroom_margin: torch.Tensor,
        residual_margin: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        headroom_lift = F.relu(headroom_score - anchor_score)
        residual_lift = F.relu(residual_score - anchor_score)
        base = torch.stack(
            [
                anchor_score,
                headroom_score,
                residual_score,
                headroom_lift,
                residual_lift,
                (headroom_score - residual_score).abs(),
                headroom_margin,
                residual_margin,
            ],
            dim=1,
        )
        if self._regime_uses_risk and risk_features is not None:
            return torch.cat([base, risk_features], dim=1)
        return base

    def _score_for_regime(
        self,
        regime: str,
        shared_encoded: torch.Tensor,
        dual_pos_encoded: torch.Tensor,
        dual_neg_encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_pos_bank, shared_neg_bank, dual_pos_bank, dual_neg_bank = self._banks_for_regime(regime)
        shared_scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        dual_scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)

        shared_pos_logits = shared_scale * shared_encoded @ shared_pos_bank.T
        shared_neg_logits = shared_scale * shared_encoded @ shared_neg_bank.T
        dual_pos_logits = dual_scale * dual_pos_encoded @ dual_pos_bank.T
        dual_neg_logits = dual_scale * dual_neg_encoded @ dual_neg_bank.T

        shared_pos_top = shared_pos_logits.amax(dim=1)
        shared_neg_top = shared_neg_logits.amax(dim=1)
        dual_pos_top = dual_pos_logits.amax(dim=1)
        dual_neg_top = dual_neg_logits.amax(dim=1)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        features = self._evidence_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        if regime == "anchor":
            gate = torch.sigmoid(self.anchor_gate(features).squeeze(-1) + self.anchor_bias)
        elif regime == "headroom":
            gate = torch.sigmoid(self.headroom_gate(features).squeeze(-1) + self.headroom_bias)
        else:
            gate = torch.sigmoid(self.residual_gate(features).squeeze(-1) + self.residual_bias)
        score = shared_score + gate * (dual_score - shared_score)
        margin = dual_pos_top - dual_neg_top
        return score, margin

    def forward_with_regime(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        anchor_score, _ = self._score_for_regime("anchor", shared_encoded, dual_pos_encoded, dual_neg_encoded)
        headroom_score, headroom_margin = self._score_for_regime(
            "headroom",
            shared_encoded,
            dual_pos_encoded,
            dual_neg_encoded,
        )
        residual_score, residual_margin = self._score_for_regime(
            "residual",
            shared_encoded,
            dual_pos_encoded,
            dual_neg_encoded,
        )

        regime_logits = self.regime_head(
            self._regime_features(
                anchor_score,
                headroom_score,
                residual_score,
                headroom_margin,
                residual_margin,
                risk_features,
            )
        )
        regime_probs = F.softmax(regime_logits, dim=1)
        headroom_lift = F.relu(headroom_score - anchor_score)
        residual_lift = F.relu(residual_score - anchor_score)
        logits = anchor_score + regime_probs[:, 0] * headroom_lift + regime_probs[:, 1] * residual_lift + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, regime_logits

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_regime(features, risk_features)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
        regime_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_positive_encoded = self.encode_dual_positive(features)
        dual_negative_encoded = self.encode_dual_negative(features)
        loss = features.new_tensor(0.0)

        for regime_name, regime_index in (("anchor", -1), ("headroom", 0), ("residual", 1)):
            shared_pos_bank, shared_neg_bank, dual_pos_bank, dual_neg_bank = self._banks_for_regime(regime_name)
            if regime_name == "anchor" or regime_targets is None:
                regime_positive_mask = positive_mask
            else:
                regime_positive_mask = positive_mask & (regime_targets == regime_index)
            if bool(regime_positive_mask.any()):
                regime_shared = shared_encoded[regime_positive_mask]
                regime_dual_positive = dual_positive_encoded[regime_positive_mask]
                regime_dual_negative = dual_negative_encoded[regime_positive_mask]
                shared_pos_alignment = (regime_shared @ shared_pos_bank.T).amax(dim=1)
                shared_neg_overlap = (regime_shared @ shared_neg_bank.T).amax(dim=1)
                dual_pos_alignment = (regime_dual_positive @ dual_pos_bank.T).amax(dim=1)
                dual_neg_overlap = (regime_dual_negative @ dual_neg_bank.T).amax(dim=1)
                loss = loss + (1.0 - shared_pos_alignment).mean()
                loss = loss + F.relu(shared_neg_overlap - margin).mean()
                loss = loss + (1.0 - dual_pos_alignment).mean()
                loss = loss + F.relu(dual_neg_overlap - margin).mean()

            if bool(hard_negative_mask.any()):
                hard_shared = shared_encoded[hard_negative_mask]
                hard_dual_positive = dual_positive_encoded[hard_negative_mask]
                hard_dual_negative = dual_negative_encoded[hard_negative_mask]
                shared_neg_alignment = (hard_shared @ shared_neg_bank.T).amax(dim=1)
                shared_pos_overlap = (hard_shared @ shared_pos_bank.T).amax(dim=1)
                dual_neg_alignment = (hard_dual_negative @ dual_neg_bank.T).amax(dim=1)
                dual_pos_overlap = (hard_dual_positive @ dual_pos_bank.T).amax(dim=1)
                loss = loss + (1.0 - shared_neg_alignment).mean()
                loss = loss + F.relu(shared_pos_overlap - margin).mean()
                loss = loss + (1.0 - dual_neg_alignment).mean()
                loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class SupportedResidualRegimeEvidenceAgreementPrototypeDeferHead(ResidualRegimeEvidenceAgreementPrototypeDeferHead):
    """Residual-regime evidence anchor with explicit positive support gating."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.headroom_support_gate = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.residual_support_gate = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.headroom_support_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.residual_support_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

    def _support_features(
        self,
        anchor_score: torch.Tensor,
        specialist_score: torch.Tensor,
        anchor_margin: torch.Tensor,
        specialist_margin: torch.Tensor,
    ) -> torch.Tensor:
        positive_lift = F.relu(specialist_score - anchor_score)
        margin_lift = F.relu(specialist_margin - anchor_margin)
        return torch.stack(
            [
                anchor_score,
                specialist_score,
                positive_lift,
                margin_lift,
                (specialist_score - anchor_score).abs(),
                anchor_margin,
                specialist_margin,
                anchor_score * specialist_score,
            ],
            dim=1,
        )

    def forward_with_regime(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        anchor_score, anchor_margin = self._score_for_regime("anchor", shared_encoded, dual_pos_encoded, dual_neg_encoded)
        headroom_score, headroom_margin = self._score_for_regime(
            "headroom",
            shared_encoded,
            dual_pos_encoded,
            dual_neg_encoded,
        )
        residual_score, residual_margin = self._score_for_regime(
            "residual",
            shared_encoded,
            dual_pos_encoded,
            dual_neg_encoded,
        )

        regime_logits = self.regime_head(
            self._regime_features(
                anchor_score,
                headroom_score,
                residual_score,
                headroom_margin,
                residual_margin,
                risk_features,
            )
        )
        regime_probs = F.softmax(regime_logits, dim=1)

        headroom_support = torch.sigmoid(
            self.headroom_support_gate(
                self._support_features(anchor_score, headroom_score, anchor_margin, headroom_margin)
            ).squeeze(-1)
            + self.headroom_support_bias
        )
        residual_support = torch.sigmoid(
            self.residual_support_gate(
                self._support_features(anchor_score, residual_score, anchor_margin, residual_margin)
            ).squeeze(-1)
            + self.residual_support_bias
        )

        headroom_lift = F.relu(headroom_score - anchor_score)
        residual_lift = F.relu(residual_score - anchor_score)
        logits = (
            anchor_score
            + regime_probs[:, 0] * headroom_support * headroom_lift
            + regime_probs[:, 1] * residual_support * residual_lift
            + self.bias
        )
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, regime_logits


class BudgetConditionedEvidenceAgreementPrototypeDeferHead(EvidenceAgreementPrototypeDeferHead):
    """Evidence-agreement mixture with an explicit budget-conditioning input."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.evidence_gate = torch.nn.Sequential(
            torch.nn.Linear(12, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        budget_features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        gate_features = torch.cat(
            [
                self._evidence_features(
                    shared_score,
                    dual_score,
                    shared_pos_top,
                    shared_neg_top,
                    dual_pos_top,
                    dual_neg_top,
                ),
                budget_features,
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.evidence_gate(gate_features).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits


class MemoryCalibratedEvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Evidence-agreement mixture whose gate also sees memory-anchor evidence."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.memory_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.memory_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.memory_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.memory_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.memory_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.evidence_gate = torch.nn.Sequential(
            torch.nn.Linear(15, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_memory(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.memory_feature_norm(self.memory_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _memory_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.memory_positive_prototypes, dim=-1),
            F.normalize(self.memory_negative_prototypes, dim=-1),
        )

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _memory_evidence(self, memory_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.memory_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._memory_banks()
        pos_logits = scale * memory_encoded @ pos.T
        neg_logits = scale * memory_encoded @ neg.T
        memory_score = torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(neg_logits, dim=1)
        return memory_score, pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _dual_evidence(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _evidence_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
        memory_score: torch.Tensor,
        memory_pos_top: torch.Tensor,
        memory_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        memory_margin = memory_pos_top - memory_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
                memory_score,
                memory_pos_top,
                memory_neg_top,
                shared_margin - memory_margin,
                dual_margin - memory_margin,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        memory_score, memory_pos_top, memory_neg_top = self._memory_evidence(memory_encoded)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        gate_features = self._evidence_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
            memory_score,
            memory_pos_top,
            memory_neg_top,
        )
        gate = torch.sigmoid(self.evidence_gate(gate_features).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        memory_pos_bank, memory_neg_bank = self._memory_banks()
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            memory_encoded = self.encode_memory(pos_features)
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)

            memory_pos_alignment = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            memory_neg_overlap = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_pos_alignment).mean()
            loss = loss + F.relu(memory_neg_overlap - margin).mean()
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            memory_encoded = self.encode_memory(neg_features)
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)

            memory_neg_alignment = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            memory_pos_overlap = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_neg_alignment).mean()
            loss = loss + F.relu(memory_pos_overlap - margin).mean()
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class AnchoredEvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Evidence-agreement mixture with an extra conservative score-anchor gate."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.anchor_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.anchor_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.evidence_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.evidence_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _dual_evidence(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _anchor_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _evidence_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top

        anchor_features = self._anchor_features(shared_score, dual_score)
        evidence_features = self._evidence_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        anchor_gate = torch.sigmoid(self.anchor_gate(anchor_features).squeeze(-1) + self.anchor_bias)
        evidence_gate = torch.sigmoid(self.evidence_gate(evidence_features).squeeze(-1) + self.evidence_bias)
        gate = anchor_gate * evidence_gate
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class PositiveLiftEvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Agreement mixture with a one-sided positive evidence lift."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.agreement_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.agreement_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.lift_gate = torch.nn.Sequential(
            torch.nn.Linear(8, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.lift_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _dual_evidence(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _lift_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_adv = F.relu(dual_pos_top - shared_pos_top)
        neg_adv = F.relu(shared_neg_top - dual_neg_top)
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        margin_adv = F.relu(dual_margin - shared_margin)
        positive_delta = F.relu(dual_score - shared_score)
        support = (pos_adv + neg_adv + 0.5 * margin_adv).clamp(max=4.0) / 4.0
        features = torch.stack(
            [
                shared_score,
                dual_score,
                positive_delta,
                pos_adv,
                neg_adv,
                margin_adv,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )
        return features, support

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top

        agreement = torch.sigmoid(
            self.agreement_gate(self._agreement_features(shared_score, dual_score)).squeeze(-1) + self.agreement_bias
        )
        lift_features, support = self._lift_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        lift = torch.sigmoid(self.lift_gate(lift_features).squeeze(-1) + self.lift_bias)
        dual_delta = dual_score - shared_score
        positive_delta = F.relu(dual_delta)
        logits = shared_score + agreement * dual_delta + lift * support * (1.0 - agreement) * positive_delta + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class ContrastiveEvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Agreement mixture gated by explicit shared-vs-dual evidence deltas."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.delta_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _dual_evidence(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _delta_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        score_delta = dual_score - shared_score
        abs_delta = score_delta.abs()
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        pos_delta = dual_pos_top - shared_pos_top
        neg_delta = shared_neg_top - dual_neg_top
        margin_delta = dual_margin - shared_margin
        evidence_balance = pos_delta + neg_delta
        return torch.stack(
            [
                shared_score,
                dual_score,
                score_delta,
                abs_delta,
                pos_delta,
                neg_delta,
                margin_delta,
                evidence_balance,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        gate_features = self._delta_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
        )
        gate = torch.sigmoid(self.delta_gate(gate_features).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class SharpnessEvidenceAgreementPrototypeDeferHead(torch.nn.Module):
    """Evidence-agreement mixture with prototype sharpness features."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 1 or negative_prototypes <= 1:
            raise ValueError("Sharpness head requires at least two prototypes per bank.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.sharpness_gate = torch.nn.Sequential(
            torch.nn.Linear(14, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _top_and_gap(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        top2 = torch.topk(logits, k=2, dim=1).values
        return top2[:, 0], top2[:, 0] - top2[:, 1]

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        pos_top, pos_gap = self._top_and_gap(pos_logits)
        neg_top, neg_gap = self._top_and_gap(neg_logits)
        return pos_top, neg_top, pos_gap, neg_gap

    def _dual_evidence(
        self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        pos_top, pos_gap = self._top_and_gap(pos_logits)
        neg_top, neg_gap = self._top_and_gap(neg_logits)
        return pos_top, neg_top, pos_gap, neg_gap

    def _sharpness_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
        shared_pos_gap: torch.Tensor,
        shared_neg_gap: torch.Tensor,
        dual_pos_gap: torch.Tensor,
        dual_neg_gap: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
                shared_pos_gap,
                shared_neg_gap,
                dual_pos_gap,
                dual_neg_gap,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_pos_top, shared_neg_top, shared_pos_gap, shared_neg_gap = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top, dual_pos_gap, dual_neg_gap = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top
        gate_features = self._sharpness_features(
            shared_score,
            dual_score,
            shared_pos_top,
            shared_neg_top,
            dual_pos_top,
            dual_neg_top,
            shared_pos_gap,
            shared_neg_gap,
            dual_pos_gap,
            dual_neg_gap,
        )
        gate = torch.sigmoid(self.sharpness_gate(gate_features).squeeze(-1) + self.gate_bias)
        logits = shared_score + gate * (dual_score - shared_score) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class MemoryAgreementBlendPrototypeDeferHead(torch.nn.Module):
    """Prototype-memory anchor with a one-sided agreement-mixture lift."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)

        self.memory_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.memory_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.memory_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.memory_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.memory_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.inner_agreement_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.inner_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.outer_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.outer_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_memory(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.memory_feature_norm(self.memory_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _memory_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.memory_positive_prototypes, dim=-1),
            F.normalize(self.memory_negative_prototypes, dim=-1),
        )

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _memory_score(self, memory_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.memory_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._memory_banks()
        pos_score = torch.logsumexp(scale * memory_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * memory_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _outer_features(
        self,
        memory_score: torch.Tensor,
        agreement_score: torch.Tensor,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
    ) -> torch.Tensor:
        diff = agreement_score - memory_score
        return torch.stack(
            [
                memory_score,
                agreement_score,
                F.relu(diff),
                diff.abs(),
                memory_score * agreement_score,
                shared_score,
                dual_score,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        inner_gate = torch.sigmoid(
            self.inner_agreement_gate(self._agreement_features(shared_score, dual_score)).squeeze(-1) + self.inner_gate_bias
        )
        agreement_score = shared_score + inner_gate * (dual_score - shared_score)

        outer_gate = torch.sigmoid(
            self.outer_blend_gate(self._outer_features(memory_score, agreement_score, shared_score, dual_score)).squeeze(-1)
            + self.outer_gate_bias
        )
        positive_lift = F.relu(agreement_score - memory_score)
        logits = memory_score + outer_gate * positive_lift + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        memory_pos_bank, memory_neg_bank = self._memory_banks()
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            memory_encoded = self.encode_memory(pos_features)
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)

            memory_pos_alignment = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            memory_neg_overlap = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_pos_alignment).mean()
            loss = loss + F.relu(memory_neg_overlap - margin).mean()
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            memory_encoded = self.encode_memory(neg_features)
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)

            memory_neg_alignment = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            memory_pos_overlap = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_neg_alignment).mean()
            loss = loss + F.relu(memory_pos_overlap - margin).mean()
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class SupportWeightedMemoryAgreementBlendPrototypeDeferHead(MemoryAgreementBlendPrototypeDeferHead):
    """Memory-agreement blend with bounded per-prototype support weights."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
        support_scale: float = 2.0,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.support_scale = support_scale
        self.memory_positive_support = torch.nn.Parameter(torch.zeros(positive_prototypes, dtype=torch.float32))
        self.memory_negative_support = torch.nn.Parameter(torch.zeros(negative_prototypes, dtype=torch.float32))
        self.shared_positive_support = torch.nn.Parameter(torch.zeros(positive_prototypes, dtype=torch.float32))
        self.shared_negative_support = torch.nn.Parameter(torch.zeros(negative_prototypes, dtype=torch.float32))
        self.dual_positive_support = torch.nn.Parameter(torch.zeros(positive_prototypes, dtype=torch.float32))
        self.dual_negative_support = torch.nn.Parameter(torch.zeros(negative_prototypes, dtype=torch.float32))

    def _bounded_support(self, raw_support: torch.Tensor) -> torch.Tensor:
        centered = raw_support - raw_support.mean()
        return self.support_scale * torch.tanh(centered)

    def _memory_score(self, memory_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.memory_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._memory_banks()
        pos_logits = scale * memory_encoded @ pos.T + self._bounded_support(self.memory_positive_support)
        neg_logits = scale * memory_encoded @ neg.T + self._bounded_support(self.memory_negative_support)
        return torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(neg_logits, dim=1)

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T + self._bounded_support(self.shared_positive_support)
        neg_logits = scale * shared_encoded @ neg.T + self._bounded_support(self.shared_negative_support)
        return torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(neg_logits, dim=1)

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T + self._bounded_support(self.dual_positive_support)
        neg_logits = scale * negative_encoded @ neg.T + self._bounded_support(self.dual_negative_support)
        return torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(neg_logits, dim=1)

    def support_regularization(self) -> torch.Tensor:
        penalties = []
        for support in (
            self.memory_positive_support,
            self.memory_negative_support,
            self.shared_positive_support,
            self.shared_negative_support,
            self.dual_positive_support,
            self.dual_negative_support,
        ):
            penalties.append(self._bounded_support(support).abs().mean())
        return torch.stack(penalties).mean()


class RegimeSplitMemoryBlendPrototypeDeferHead(MemoryAgreementBlendPrototypeDeferHead):
    """Memory-anchor blend with separate headroom and residual lift specialists."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.headroom_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.headroom_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.residual_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.residual_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        regime_input_dim = 7 + (risk_dim if use_risk_branch and risk_dim > 0 else 0)
        self.regime_head = torch.nn.Sequential(
            torch.nn.Linear(regime_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 3),
        )
        self._regime_uses_risk = use_risk_branch and risk_dim > 0

    def _regime_features(self, outer_features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        if self._regime_uses_risk and risk_features is not None:
            return torch.cat([outer_features, risk_features], dim=1)
        return outer_features

    def forward_with_regime(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        inner_gate = torch.sigmoid(
            self.inner_agreement_gate(self._agreement_features(shared_score, dual_score)).squeeze(-1) + self.inner_gate_bias
        )
        agreement_score = shared_score + inner_gate * (dual_score - shared_score)

        outer_features = self._outer_features(memory_score, agreement_score, shared_score, dual_score)
        base_gate = torch.sigmoid(self.outer_blend_gate(outer_features).squeeze(-1) + self.outer_gate_bias)
        headroom_gate = torch.sigmoid(self.headroom_blend_gate(outer_features).squeeze(-1) + self.headroom_gate_bias)
        residual_gate = torch.sigmoid(self.residual_blend_gate(outer_features).squeeze(-1) + self.residual_gate_bias)
        regime_logits = self.regime_head(self._regime_features(outer_features, risk_features))
        regime_probs = F.softmax(regime_logits, dim=1)
        positive_lift = F.relu(agreement_score - memory_score)
        mixed_gate = (
            regime_probs[:, 0] * base_gate
            + regime_probs[:, 1] * headroom_gate
            + regime_probs[:, 2] * residual_gate
        )
        logits = memory_score + mixed_gate * positive_lift + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, regime_logits

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_regime(features, risk_features)
        return logits


class RiskPriorRegimeMemoryBlendPrototypeDeferHead(MemoryAgreementBlendPrototypeDeferHead):
    """Memory-anchor blend with regime specialists biased by explicit risk priors."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.headroom_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.headroom_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.residual_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.residual_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.regime_head = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 3),
        )
        self.risk_prior = None
        self.risk_prior_scale = None
        if risk_dim > 0:
            self.risk_prior = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 3),
            )
            self.risk_prior_scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward_with_regime(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        inner_gate = torch.sigmoid(
            self.inner_agreement_gate(self._agreement_features(shared_score, dual_score)).squeeze(-1) + self.inner_gate_bias
        )
        agreement_score = shared_score + inner_gate * (dual_score - shared_score)

        outer_features = self._outer_features(memory_score, agreement_score, shared_score, dual_score)
        base_gate = torch.sigmoid(self.outer_blend_gate(outer_features).squeeze(-1) + self.outer_gate_bias)
        headroom_gate = torch.sigmoid(self.headroom_blend_gate(outer_features).squeeze(-1) + self.headroom_gate_bias)
        residual_gate = torch.sigmoid(self.residual_blend_gate(outer_features).squeeze(-1) + self.residual_gate_bias)
        regime_logits = self.regime_head(outer_features)
        if self.risk_prior is not None and risk_features is not None and self.risk_prior_scale is not None:
            regime_logits = regime_logits + self.risk_prior_scale * self.risk_prior(risk_features)
        regime_probs = F.softmax(regime_logits, dim=1)
        positive_lift = F.relu(agreement_score - memory_score)
        mixed_gate = (
            regime_probs[:, 0] * base_gate
            + regime_probs[:, 1] * headroom_gate
            + regime_probs[:, 2] * residual_gate
        )
        logits = memory_score + mixed_gate * positive_lift + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, regime_logits

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_regime(features, risk_features)
        return logits


class RiskVetoRegimeMemoryBlendPrototypeDeferHead(MemoryAgreementBlendPrototypeDeferHead):
    """Memory-anchor regime blend with suppressive risk veto masks on specialist lifts."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__(
            feature_dim,
            risk_dim=risk_dim,
            prototype_dim=prototype_dim,
            positive_prototypes=positive_prototypes,
            negative_prototypes=negative_prototypes,
            hidden_dim=hidden_dim,
            use_risk_branch=use_risk_branch,
        )
        self.headroom_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.headroom_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.residual_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.residual_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        regime_input_dim = 7 + (risk_dim if use_risk_branch and risk_dim > 0 else 0)
        self.regime_head = torch.nn.Sequential(
            torch.nn.Linear(regime_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 3),
        )
        self._regime_uses_risk = use_risk_branch and risk_dim > 0
        self.risk_veto = None
        if risk_dim > 0:
            self.risk_veto = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 2),
            )

    def _regime_features(self, outer_features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        if self._regime_uses_risk and risk_features is not None:
            return torch.cat([outer_features, risk_features], dim=1)
        return outer_features

    def forward_with_regime(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        inner_gate = torch.sigmoid(
            self.inner_agreement_gate(self._agreement_features(shared_score, dual_score)).squeeze(-1) + self.inner_gate_bias
        )
        agreement_score = shared_score + inner_gate * (dual_score - shared_score)

        outer_features = self._outer_features(memory_score, agreement_score, shared_score, dual_score)
        base_gate = torch.sigmoid(self.outer_blend_gate(outer_features).squeeze(-1) + self.outer_gate_bias)
        headroom_gate = torch.sigmoid(self.headroom_blend_gate(outer_features).squeeze(-1) + self.headroom_gate_bias)
        residual_gate = torch.sigmoid(self.residual_blend_gate(outer_features).squeeze(-1) + self.residual_gate_bias)
        regime_logits = self.regime_head(self._regime_features(outer_features, risk_features))
        regime_probs = F.softmax(regime_logits, dim=1)
        if self.risk_veto is not None and risk_features is not None:
            veto_logits = self.risk_veto(risk_features)
            veto_masks = torch.sigmoid(veto_logits)
        else:
            veto_logits = outer_features.new_zeros((outer_features.size(0), 2))
            veto_masks = outer_features.new_ones((outer_features.size(0), 2))
        positive_lift = F.relu(agreement_score - memory_score)
        mixed_gate = (
            regime_probs[:, 0] * base_gate
            + regime_probs[:, 1] * headroom_gate * veto_masks[:, 0]
            + regime_probs[:, 2] * residual_gate * veto_masks[:, 1]
        )
        logits = memory_score + mixed_gate * positive_lift + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, regime_logits, veto_logits

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits, _, _ = self.forward_with_regime(features, risk_features)
        return logits


class TeacherMarginMemoryBlendPrototypeDeferHead(torch.nn.Module):
    """Prototype-memory anchor with a one-sided agreement lift plus teacher-gain calibration."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)

        self.memory_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.memory_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.memory_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.memory_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.memory_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.inner_agreement_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.inner_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.outer_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.outer_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        gain_input_dim = 7 + (risk_dim if use_risk_branch and risk_dim > 0 else 0)
        self.gain_branch = torch.nn.Sequential(
            torch.nn.Linear(gain_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        self._gain_uses_risk = use_risk_branch and risk_dim > 0
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_memory(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.memory_feature_norm(self.memory_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _memory_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.memory_positive_prototypes, dim=-1),
            F.normalize(self.memory_negative_prototypes, dim=-1),
        )

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _memory_score(self, memory_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.memory_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._memory_banks()
        pos_score = torch.logsumexp(scale * memory_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * memory_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _outer_features(
        self,
        memory_score: torch.Tensor,
        agreement_score: torch.Tensor,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
    ) -> torch.Tensor:
        diff = agreement_score - memory_score
        return torch.stack(
            [
                memory_score,
                agreement_score,
                F.relu(diff),
                diff.abs(),
                memory_score * agreement_score,
                shared_score,
                dual_score,
            ],
            dim=1,
        )

    def _gain_features(self, outer_features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        if self._gain_uses_risk and risk_features is not None:
            return torch.cat([outer_features, risk_features], dim=1)
        return outer_features

    def forward_with_gain(
        self,
        features: torch.Tensor,
        risk_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        inner_gate = torch.sigmoid(
            self.inner_agreement_gate(self._agreement_features(shared_score, dual_score)).squeeze(-1) + self.inner_gate_bias
        )
        agreement_score = shared_score + inner_gate * (dual_score - shared_score)

        outer_features = self._outer_features(memory_score, agreement_score, shared_score, dual_score)
        outer_gate = torch.sigmoid(self.outer_blend_gate(outer_features).squeeze(-1) + self.outer_gate_bias)
        predicted_gain = F.softplus(self.gain_branch(self._gain_features(outer_features, risk_features)).squeeze(-1))
        gain_gate = predicted_gain / (1.0 + predicted_gain)
        positive_lift = F.relu(agreement_score - memory_score)
        logits = memory_score + outer_gate * (1.0 + 0.5 * gain_gate) * positive_lift + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits, predicted_gain

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_with_gain(features, risk_features)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        memory_pos_bank, memory_neg_bank = self._memory_banks()
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            memory_encoded = self.encode_memory(pos_features)
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)

            memory_pos_alignment = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            memory_neg_overlap = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_pos_alignment).mean()
            loss = loss + F.relu(memory_neg_overlap - margin).mean()
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            memory_encoded = self.encode_memory(neg_features)
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)

            memory_neg_alignment = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            memory_pos_overlap = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_neg_alignment).mean()
            loss = loss + F.relu(memory_pos_overlap - margin).mean()
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class MemoryEvidenceAgreementBlendPrototypeDeferHead(torch.nn.Module):
    """Prototype-memory anchor with a one-sided evidence-agreement lift."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)

        self.memory_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.memory_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.memory_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.memory_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.memory_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.inner_evidence_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.inner_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.outer_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(11, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.outer_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_memory(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.memory_feature_norm(self.memory_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _memory_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.memory_positive_prototypes, dim=-1),
            F.normalize(self.memory_negative_prototypes, dim=-1),
        )

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _memory_score(self, memory_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.memory_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._memory_banks()
        pos_score = torch.logsumexp(scale * memory_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * memory_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _dual_evidence(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _inner_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def _outer_features(
        self,
        memory_score: torch.Tensor,
        agreement_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = agreement_score - memory_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                memory_score,
                agreement_score,
                F.relu(diff),
                diff.abs(),
                memory_score * agreement_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)
        shared_score = shared_pos_top - shared_neg_top
        dual_score = dual_pos_top - dual_neg_top

        inner_gate = torch.sigmoid(
            self.inner_evidence_gate(
                self._inner_features(
                    shared_score,
                    dual_score,
                    shared_pos_top,
                    shared_neg_top,
                    dual_pos_top,
                    dual_neg_top,
                )
            ).squeeze(-1)
            + self.inner_gate_bias
        )
        agreement_score = shared_score + inner_gate * (dual_score - shared_score)

        outer_gate = torch.sigmoid(
            self.outer_blend_gate(
                self._outer_features(
                    memory_score,
                    agreement_score,
                    shared_pos_top,
                    shared_neg_top,
                    dual_pos_top,
                    dual_neg_top,
                )
            ).squeeze(-1)
            + self.outer_gate_bias
        )
        positive_lift = F.relu(agreement_score - memory_score)
        logits = memory_score + outer_gate * positive_lift + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        memory_pos_bank, memory_neg_bank = self._memory_banks()
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            memory_encoded = self.encode_memory(pos_features)
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)

            memory_pos_alignment = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            memory_neg_overlap = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_pos_alignment).mean()
            loss = loss + F.relu(memory_neg_overlap - margin).mean()
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            memory_encoded = self.encode_memory(neg_features)
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)

            memory_neg_alignment = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            memory_pos_overlap = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_neg_alignment).mean()
            loss = loss + F.relu(memory_pos_overlap - margin).mean()
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class MemoryDualLiftBlendPrototypeDeferHead(torch.nn.Module):
    """Prototype-memory anchor with score-only and evidence-aware positive lifts."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)

        self.memory_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.memory_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.memory_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.memory_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.memory_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))

        self.inner_score_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.inner_score_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.inner_evidence_gate = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.inner_evidence_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

        self.score_lift_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.score_lift_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.evidence_lift_gate = torch.nn.Sequential(
            torch.nn.Linear(11, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.evidence_lift_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_memory(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.memory_feature_norm(self.memory_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _memory_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.memory_positive_prototypes, dim=-1),
            F.normalize(self.memory_negative_prototypes, dim=-1),
        )

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _memory_score(self, memory_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.memory_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._memory_banks()
        pos_score = torch.logsumexp(scale * memory_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * memory_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _shared_evidence(self, shared_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_logits = scale * shared_encoded @ pos.T
        neg_logits = scale * shared_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _dual_evidence(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_logits = scale * positive_encoded @ pos.T
        neg_logits = scale * negative_encoded @ neg.T
        return pos_logits.amax(dim=1), neg_logits.amax(dim=1)

    def _score_agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _evidence_agreement_features(
        self,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = dual_score - shared_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def _score_outer_features(
        self,
        memory_score: torch.Tensor,
        agreement_score: torch.Tensor,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
    ) -> torch.Tensor:
        diff = agreement_score - memory_score
        return torch.stack(
            [
                memory_score,
                agreement_score,
                F.relu(diff),
                diff.abs(),
                memory_score * agreement_score,
                shared_score,
                dual_score,
            ],
            dim=1,
        )

    def _evidence_outer_features(
        self,
        memory_score: torch.Tensor,
        agreement_score: torch.Tensor,
        shared_pos_top: torch.Tensor,
        shared_neg_top: torch.Tensor,
        dual_pos_top: torch.Tensor,
        dual_neg_top: torch.Tensor,
    ) -> torch.Tensor:
        diff = agreement_score - memory_score
        shared_margin = shared_pos_top - shared_neg_top
        dual_margin = dual_pos_top - dual_neg_top
        return torch.stack(
            [
                memory_score,
                agreement_score,
                F.relu(diff),
                diff.abs(),
                memory_score * agreement_score,
                shared_pos_top,
                shared_neg_top,
                dual_pos_top,
                dual_neg_top,
                shared_margin,
                dual_margin,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)
        shared_pos_top, shared_neg_top = self._shared_evidence(shared_encoded)
        dual_pos_top, dual_neg_top = self._dual_evidence(dual_pos_encoded, dual_neg_encoded)

        score_inner_gate = torch.sigmoid(
            self.inner_score_gate(self._score_agreement_features(shared_score, dual_score)).squeeze(-1) + self.inner_score_bias
        )
        score_agreement = shared_score + score_inner_gate * (dual_score - shared_score)

        evidence_inner_gate = torch.sigmoid(
            self.inner_evidence_gate(
                self._evidence_agreement_features(
                    shared_score,
                    dual_score,
                    shared_pos_top,
                    shared_neg_top,
                    dual_pos_top,
                    dual_neg_top,
                )
            ).squeeze(-1)
            + self.inner_evidence_bias
        )
        evidence_agreement = shared_score + evidence_inner_gate * (dual_score - shared_score)

        score_lift_gate = torch.sigmoid(
            self.score_lift_gate(
                self._score_outer_features(memory_score, score_agreement, shared_score, dual_score)
            ).squeeze(-1)
            + self.score_lift_bias
        )
        evidence_lift_gate = torch.sigmoid(
            self.evidence_lift_gate(
                self._evidence_outer_features(
                    memory_score,
                    evidence_agreement,
                    shared_pos_top,
                    shared_neg_top,
                    dual_pos_top,
                    dual_neg_top,
                )
            ).squeeze(-1)
            + self.evidence_lift_bias
        )

        score_candidate = memory_score + score_lift_gate * F.relu(score_agreement - memory_score)
        evidence_candidate = memory_score + evidence_lift_gate * F.relu(evidence_agreement - memory_score)
        logits = torch.maximum(score_candidate, evidence_candidate) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        memory_pos_bank, memory_neg_bank = self._memory_banks()
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            memory_encoded = self.encode_memory(pos_features)
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)

            memory_pos_alignment = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            memory_neg_overlap = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_pos_alignment).mean()
            loss = loss + F.relu(memory_neg_overlap - margin).mean()
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            memory_encoded = self.encode_memory(neg_features)
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)

            memory_neg_alignment = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            memory_pos_overlap = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_neg_alignment).mean()
            loss = loss + F.relu(memory_pos_overlap - margin).mean()
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class BudgetConditionedMemoryBlendPrototypeDeferHead(torch.nn.Module):
    """Prototype-memory anchor with separate micro and matched-band lift gates."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)

        self.memory_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.memory_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.memory_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.memory_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.memory_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))

        self.inner_agreement_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.inner_gate_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

        self.micro_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.micro_gate_bias = torch.nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
        self.matched_blend_gate = torch.nn.Sequential(
            torch.nn.Linear(7, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.matched_gate_bias = torch.nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))

        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_memory(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.memory_feature_norm(self.memory_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _memory_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.memory_positive_prototypes, dim=-1),
            F.normalize(self.memory_negative_prototypes, dim=-1),
        )

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _memory_score(self, memory_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.memory_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._memory_banks()
        pos_score = torch.logsumexp(scale * memory_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * memory_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _outer_features(
        self,
        memory_score: torch.Tensor,
        agreement_score: torch.Tensor,
        shared_score: torch.Tensor,
        dual_score: torch.Tensor,
    ) -> torch.Tensor:
        diff = agreement_score - memory_score
        return torch.stack(
            [
                memory_score,
                agreement_score,
                F.relu(diff),
                diff.abs(),
                memory_score * agreement_score,
                shared_score,
                dual_score,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        memory_encoded = self.encode_memory(features)
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)

        memory_score = self._memory_score(memory_encoded)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        inner_gate = torch.sigmoid(
            self.inner_agreement_gate(self._agreement_features(shared_score, dual_score)).squeeze(-1) + self.inner_gate_bias
        )
        agreement_score = shared_score + inner_gate * (dual_score - shared_score)
        outer_features = self._outer_features(memory_score, agreement_score, shared_score, dual_score)
        positive_lift = F.relu(agreement_score - memory_score)

        micro_gate = torch.sigmoid(self.micro_blend_gate(outer_features).squeeze(-1) + self.micro_gate_bias)
        matched_gate = torch.sigmoid(self.matched_blend_gate(outer_features).squeeze(-1) + self.matched_gate_bias)

        micro_candidate = memory_score + micro_gate * positive_lift
        matched_candidate = memory_score + matched_gate * positive_lift
        logits = torch.maximum(micro_candidate, matched_candidate) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        memory_pos_bank, memory_neg_bank = self._memory_banks()
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            memory_encoded = self.encode_memory(pos_features)
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)

            memory_pos_alignment = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            memory_neg_overlap = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_pos_alignment).mean()
            loss = loss + F.relu(memory_neg_overlap - margin).mean()
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            memory_encoded = self.encode_memory(neg_features)
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)

            memory_neg_alignment = (memory_encoded @ memory_neg_bank.T).amax(dim=1)
            memory_pos_overlap = (memory_encoded @ memory_pos_bank.T).amax(dim=1)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)

            loss = loss + (1.0 - memory_neg_alignment).mean()
            loss = loss + F.relu(memory_pos_overlap - margin).mean()
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class AdapterPrototypeDeferHead(torch.nn.Module):
    """Prototype bank plus a tiny learned adapter in encoded feature space."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.adapter_branch = torch.nn.Sequential(
            torch.nn.Linear(prototype_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.adapter_scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _prototype_score(self, encoded: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        pos_score = torch.logsumexp(scale * encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(features)
        logits = self._prototype_score(encoded) + self.bias
        logits = logits + self.adapter_scale.tanh() * self.adapter_branch(encoded).squeeze(-1)
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class MultiscalePrototypeDeferHead(torch.nn.Module):
    """Prototype bank with a coarse centroid branch for sparse transfer."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.global_scale = torch.nn.Parameter(torch.tensor(0.75, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.positive_prototypes, dim=-1), F.normalize(self.negative_prototypes, dim=-1)

    def _prototype_score(self, encoded: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos_score = torch.logsumexp(scale * encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _centroid_score(self, encoded: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        pos_centroid = F.normalize(pos.mean(dim=0, keepdim=True), dim=-1)
        neg_centroid = F.normalize(neg.mean(dim=0, keepdim=True), dim=-1)
        return (encoded @ pos_centroid.T).squeeze(1) - (encoded @ neg_centroid.T).squeeze(1)

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neg = self._banks()
        logits = self._prototype_score(encoded, pos, neg)
        logits = logits + self.global_scale.tanh() * self._centroid_score(encoded, pos, neg) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neg = self._banks()
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            centroid_alignment = positive_encoded @ F.normalize(pos.mean(dim=0, keepdim=True), dim=-1).T
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + (1.0 - centroid_alignment.squeeze(1)).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            centroid_overlap = negative_encoded @ F.normalize(pos.mean(dim=0, keepdim=True), dim=-1).T
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()
            loss = loss + F.relu(centroid_overlap.squeeze(1) - margin).mean()

        return loss


class EvidencePrototypeDeferHead(torch.nn.Module):
    """Prototype bank with a learned readout over top-k prototype evidence."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        evidence_topk: int = 4,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")
        if evidence_topk <= 0:
            raise ValueError("evidence_topk must be positive.")

        self.evidence_topk = evidence_topk
        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        evidence_dim = min(evidence_topk, positive_prototypes) + min(evidence_topk, negative_prototypes) + 4
        self.evidence_branch = torch.nn.Sequential(
            torch.nn.Linear(evidence_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.positive_prototypes, dim=-1), F.normalize(self.negative_prototypes, dim=-1)

    def _evidence_features(self, encoded: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos_sims = scale * (encoded @ pos.T)
        neg_sims = scale * (encoded @ neg.T)
        pos_k = min(self.evidence_topk, pos_sims.size(1))
        neg_k = min(self.evidence_topk, neg_sims.size(1))
        pos_topk, _ = pos_sims.topk(pos_k, dim=1)
        neg_topk, _ = neg_sims.topk(neg_k, dim=1)
        pos_mean = pos_topk.mean(dim=1, keepdim=True)
        neg_mean = neg_topk.mean(dim=1, keepdim=True)
        top_gap = pos_topk[:, :1] - neg_topk[:, :1]
        mean_gap = pos_mean - neg_mean
        return torch.cat([pos_topk, neg_topk, pos_mean, neg_mean, top_gap, mean_gap], dim=1)

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neg = self._banks()
        evidence = self._evidence_features(encoded, pos, neg)
        logits = self.evidence_branch(evidence).squeeze(-1) + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neg = self._banks()
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class BandpassPrototypeDeferHead(torch.nn.Module):
    """Prototype bank with a residual branch active only near ambiguous scores."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        band_width: float = 1.0,
        band_sharpness: float = 2.0,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")
        if risk_dim <= 0:
            raise ValueError("risk_dim must be positive for bandpass prototype heads.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.band_center = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.band_width = torch.nn.Parameter(torch.tensor(band_width, dtype=torch.float32))
        self.band_sharpness = torch.nn.Parameter(torch.tensor(band_sharpness, dtype=torch.float32))
        self.risk_branch = torch.nn.Sequential(
            torch.nn.Linear(risk_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _prototype_score(self, encoded: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        pos_score = torch.logsumexp(scale * encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _band_gate(self, prototype_score: torch.Tensor) -> torch.Tensor:
        width = self.band_width.abs().clamp(min=0.05, max=8.0)
        sharpness = self.band_sharpness.abs().clamp(min=0.5, max=16.0)
        centered = (prototype_score - self.band_center).abs()
        return torch.sigmoid((width - centered) * sharpness)

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        prototype_score = self._prototype_score(encoded)
        band_gate = self._band_gate(prototype_score)
        risk_delta = self.risk_branch(risk_features).squeeze(-1)
        return prototype_score + band_gate * risk_delta + self.bias

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class CascadePrototypeDeferHead(torch.nn.Module):
    """Shared-anchor cascade with agreement-gated positive lift."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.agreement_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.agreement_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.lift_gate = torch.nn.Sequential(
            torch.nn.Linear(5, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.lift_bias = torch.nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _lift_features(
        self,
        shared_score: torch.Tensor,
        agreed_score: torch.Tensor,
        lift_amount: torch.Tensor,
        dual_score: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                shared_score,
                agreed_score,
                lift_amount,
                (dual_score - shared_score).abs(),
                shared_score * agreed_score,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        agreement_features = self._agreement_features(shared_score, dual_score)
        agreement = torch.sigmoid(self.agreement_gate(agreement_features).squeeze(-1) + self.agreement_bias)
        agreed_score = shared_score + agreement * (dual_score - shared_score)

        lift_amount = F.relu(agreed_score - shared_score)
        lift_features = self._lift_features(shared_score, agreed_score, lift_amount, dual_score)
        lift_gate = torch.sigmoid(self.lift_gate(lift_features).squeeze(-1) + self.lift_bias)

        logits = shared_score + lift_gate * lift_amount + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class SwitchPrototypeDeferHead(torch.nn.Module):
    """Per-state switch between shared-anchor and agreement-mixture branches."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")

        scale = 1.0 / math.sqrt(prototype_dim)
        self.shared_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.shared_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.shared_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.shared_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.dual_positive_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_positive_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_negative_feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.dual_negative_feature_norm = torch.nn.LayerNorm(prototype_dim)
        self.dual_positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.dual_negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)

        self.shared_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.dual_logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))

        self.agreement_gate = torch.nn.Sequential(
            torch.nn.Linear(4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.agreement_bias = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

        self.switch_gate = torch.nn.Sequential(
            torch.nn.Linear(6, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.switch_bias = torch.nn.Parameter(torch.tensor(-0.75, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        self.shared_risk_branch = None
        self.agree_risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.shared_risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )
            self.agree_risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode_shared(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.shared_feature_norm(self.shared_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_positive(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_positive_feature_norm(self.dual_positive_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def encode_dual_negative(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.dual_negative_feature_norm(self.dual_negative_feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _shared_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.shared_positive_prototypes, dim=-1),
            F.normalize(self.shared_negative_prototypes, dim=-1),
        )

    def _dual_banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.dual_positive_prototypes, dim=-1),
            F.normalize(self.dual_negative_prototypes, dim=-1),
        )

    def _shared_score(self, shared_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.shared_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._shared_banks()
        pos_score = torch.logsumexp(scale * shared_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * shared_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _dual_score(self, positive_encoded: torch.Tensor, negative_encoded: torch.Tensor) -> torch.Tensor:
        scale = self.dual_logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._dual_banks()
        pos_score = torch.logsumexp(scale * positive_encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * negative_encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def _agreement_features(self, shared_score: torch.Tensor, dual_score: torch.Tensor) -> torch.Tensor:
        diff = dual_score - shared_score
        return torch.stack(
            [
                shared_score,
                dual_score,
                diff.abs(),
                shared_score * dual_score,
            ],
            dim=1,
        )

    def _switch_features(self, shared_branch: torch.Tensor, agree_branch: torch.Tensor) -> torch.Tensor:
        diff = agree_branch - shared_branch
        return torch.stack(
            [
                shared_branch,
                agree_branch,
                diff,
                diff.abs(),
                shared_branch * agree_branch,
                torch.maximum(shared_branch, agree_branch),
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        shared_encoded = self.encode_shared(features)
        dual_pos_encoded = self.encode_dual_positive(features)
        dual_neg_encoded = self.encode_dual_negative(features)
        shared_score = self._shared_score(shared_encoded)
        dual_score = self._dual_score(dual_pos_encoded, dual_neg_encoded)

        agreement_features = self._agreement_features(shared_score, dual_score)
        agreement = torch.sigmoid(self.agreement_gate(agreement_features).squeeze(-1) + self.agreement_bias)
        agree_score = shared_score + agreement * (dual_score - shared_score)

        shared_branch = shared_score
        agree_branch = agree_score
        if self.shared_risk_branch is not None and self.agree_risk_branch is not None and risk_features is not None:
            shared_branch = shared_branch + self.shared_risk_branch(risk_features).squeeze(-1)
            agree_branch = agree_branch + self.agree_risk_branch(risk_features).squeeze(-1)

        switch_features = self._switch_features(shared_branch, agree_branch)
        switch = torch.sigmoid(self.switch_gate(switch_features).squeeze(-1) + self.switch_bias)
        logits = shared_branch + switch * (agree_branch - shared_branch) + self.bias
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        shared_pos_bank, shared_neg_bank = self._shared_banks()
        dual_pos_bank, dual_neg_bank = self._dual_banks()
        loss = features.new_tensor(0.0)

        if bool(positive_mask.any()):
            pos_features = features[positive_mask]
            shared_encoded = self.encode_shared(pos_features)
            dual_positive_encoded = self.encode_dual_positive(pos_features)
            dual_negative_encoded = self.encode_dual_negative(pos_features)
            shared_pos_alignment = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            shared_neg_overlap = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            dual_pos_alignment = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            dual_neg_overlap = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_pos_alignment).mean()
            loss = loss + F.relu(shared_neg_overlap - margin).mean()
            loss = loss + (1.0 - dual_pos_alignment).mean()
            loss = loss + F.relu(dual_neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            neg_features = features[hard_negative_mask]
            shared_encoded = self.encode_shared(neg_features)
            dual_positive_encoded = self.encode_dual_positive(neg_features)
            dual_negative_encoded = self.encode_dual_negative(neg_features)
            shared_neg_alignment = (shared_encoded @ shared_neg_bank.T).amax(dim=1)
            shared_pos_overlap = (shared_encoded @ shared_pos_bank.T).amax(dim=1)
            dual_neg_alignment = (dual_negative_encoded @ dual_neg_bank.T).amax(dim=1)
            dual_pos_overlap = (dual_positive_encoded @ dual_pos_bank.T).amax(dim=1)
            loss = loss + (1.0 - shared_neg_alignment).mean()
            loss = loss + F.relu(shared_pos_overlap - margin).mean()
            loss = loss + (1.0 - dual_neg_alignment).mean()
            loss = loss + F.relu(dual_pos_overlap - margin).mean()

        return loss


class SuppressorPrototypeDeferHead(torch.nn.Module):
    """Prototype bank with an explicit harmful-memory suppressor branch."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        neutral_prototypes: int = 8,
        harmful_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if min(positive_prototypes, neutral_prototypes, harmful_prototypes) <= 0:
            raise ValueError("Prototype counts must be positive.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.neutral_prototypes = torch.nn.Parameter(torch.randn(neutral_prototypes, prototype_dim) * scale)
        self.harmful_prototypes = torch.nn.Parameter(torch.randn(harmful_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.harmful_scale = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            F.normalize(self.positive_prototypes, dim=-1),
            F.normalize(self.neutral_prototypes, dim=-1),
            F.normalize(self.harmful_prototypes, dim=-1),
        )

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neutral, harmful = self._banks()
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos_score = torch.logsumexp(scale * encoded @ pos.T, dim=1)
        neutral_score = torch.logsumexp(scale * encoded @ neutral.T, dim=1)
        harmful_score = torch.logsumexp(scale * encoded @ harmful.T, dim=1)
        suppress = F.softplus(self.harmful_scale).clamp(min=0.1, max=4.0)
        logits = pos_score - neutral_score - suppress * harmful_score + self.bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        neutral_negative_mask: torch.Tensor,
        harmful_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neutral, harmful = self._banks()
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            current = encoded[positive_mask]
            pos_alignment = (current @ pos.T).amax(dim=1)
            neutral_overlap = (current @ neutral.T).amax(dim=1)
            harmful_overlap = (current @ harmful.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neutral_overlap - margin).mean()
            loss = loss + F.relu(harmful_overlap - margin).mean()

        if bool(neutral_negative_mask.any()):
            current = encoded[neutral_negative_mask]
            neutral_alignment = (current @ neutral.T).amax(dim=1)
            pos_overlap = (current @ pos.T).amax(dim=1)
            harmful_overlap = (current @ harmful.T).amax(dim=1)
            loss = loss + (1.0 - neutral_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()
            loss = loss + 0.5 * F.relu(harmful_overlap - margin).mean()

        if bool(harmful_negative_mask.any()):
            current = encoded[harmful_negative_mask]
            harmful_alignment = (current @ harmful.T).amax(dim=1)
            pos_overlap = (current @ pos.T).amax(dim=1)
            neutral_overlap = (current @ neutral.T).amax(dim=1)
            loss = loss + (1.0 - harmful_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()
            loss = loss + 0.5 * F.relu(neutral_overlap - margin).mean()

        return loss


class InteractionPrototypeDeferHead(torch.nn.Module):
    """Prototype bank with a residual branch conditioned on prototype score stats."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        gated: bool = False,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")
        if risk_dim <= 0:
            raise ValueError("risk_dim must be positive for interaction prototype heads.")

        self.gated = gated
        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        summary_dim = risk_dim + 4
        self.interaction_branch = torch.nn.Sequential(
            torch.nn.Linear(summary_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gate_branch = None
        if gated:
            self.gate_branch = torch.nn.Sequential(
                torch.nn.Linear(4, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.positive_prototypes, dim=-1), F.normalize(self.negative_prototypes, dim=-1)

    def _score_stats(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._banks()
        pos_sims = scale * (encoded @ pos.T)
        neg_sims = scale * (encoded @ neg.T)
        pos_score = torch.logsumexp(pos_sims, dim=1)
        neg_score = torch.logsumexp(neg_sims, dim=1)
        proto_score = pos_score - neg_score
        pos_max = pos_sims.max(dim=1).values
        neg_max = neg_sims.max(dim=1).values
        summary = torch.stack([proto_score, pos_max, neg_max, pos_max - neg_max], dim=1)
        return proto_score, summary

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        proto_score, summary = self._score_stats(encoded)
        residual_input = torch.cat([risk_features, summary], dim=1)
        residual = self.interaction_branch(residual_input).squeeze(-1)
        if self.gate_branch is not None:
            gate = torch.sigmoid(self.gate_branch(summary).squeeze(-1))
            residual = gate * residual
        return proto_score + residual + self.bias

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neg = self._banks()
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class LiftPrototypeDeferHead(torch.nn.Module):
    """Prototype bank with a nonnegative residual lift over the base score."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        gated: bool = False,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")
        if risk_dim <= 0:
            raise ValueError("risk_dim must be positive for lift prototype heads.")

        self.gated = gated
        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.lift_scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        summary_dim = risk_dim + 4
        self.lift_branch = torch.nn.Sequential(
            torch.nn.Linear(summary_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.gate_branch = None
        if gated:
            self.gate_branch = torch.nn.Sequential(
                torch.nn.Linear(4, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _banks(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.positive_prototypes, dim=-1), F.normalize(self.negative_prototypes, dim=-1)

    def _score_stats(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos, neg = self._banks()
        pos_sims = scale * (encoded @ pos.T)
        neg_sims = scale * (encoded @ neg.T)
        pos_score = torch.logsumexp(pos_sims, dim=1)
        neg_score = torch.logsumexp(neg_sims, dim=1)
        proto_score = pos_score - neg_score
        pos_max = pos_sims.max(dim=1).values
        neg_max = neg_sims.max(dim=1).values
        summary = torch.stack([proto_score, pos_max, neg_max, pos_max - neg_max], dim=1)
        return proto_score, summary

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        proto_score, summary = self._score_stats(encoded)
        lift_input = torch.cat([risk_features, summary], dim=1)
        lift = F.softplus(self.lift_branch(lift_input).squeeze(-1))
        if self.gate_branch is not None:
            gate = torch.sigmoid(self.gate_branch(summary).squeeze(-1))
            lift = gate * lift
        return proto_score + self.lift_scale.abs().clamp(min=0.05, max=4.0) * lift + self.bias

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos, neg = self._banks()
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class GatedPrototypeDeferHead(torch.nn.Module):
    """Prototype bank with risk-conditioned scaling of prototype evidence."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_bias_branch: bool = True,
    ) -> None:
        super().__init__()
        if positive_prototypes <= 0 or negative_prototypes <= 0:
            raise ValueError("Prototype counts must be positive.")
        if risk_dim <= 0:
            raise ValueError("risk_dim must be positive for gated prototype heads.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.prototype_bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.gate_branch = torch.nn.Sequential(
            torch.nn.Linear(risk_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.bias_branch = None
        if use_bias_branch:
            self.bias_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _prototype_score(self, encoded: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        pos_score = torch.logsumexp(scale * encoded @ pos.T, dim=1)
        neg_score = torch.logsumexp(scale * encoded @ neg.T, dim=1)
        return pos_score - neg_score

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        proto = self._prototype_score(encoded) + self.prototype_bias
        gate = 2.0 * torch.sigmoid(self.gate_branch(risk_features).squeeze(-1))
        logits = gate * proto
        if self.bias_branch is not None:
            logits = logits + self.bias_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neg = F.normalize(self.negative_prototypes, dim=-1)
        loss = encoded.new_tensor(0.0)

        if bool(positive_mask.any()):
            positive_encoded = encoded[positive_mask]
            pos_alignment = (positive_encoded @ pos.T).amax(dim=1)
            neg_overlap = (positive_encoded @ neg.T).amax(dim=1)
            loss = loss + (1.0 - pos_alignment).mean()
            loss = loss + F.relu(neg_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            negative_encoded = encoded[hard_negative_mask]
            neg_alignment = (negative_encoded @ neg.T).amax(dim=1)
            pos_overlap = (negative_encoded @ pos.T).amax(dim=1)
            loss = loss + (1.0 - neg_alignment).mean()
            loss = loss + F.relu(pos_overlap - margin).mean()

        return loss


class SpecialistPrototypeDeferHead(torch.nn.Module):
    """Prototype head with separate positive banks for distinct source families."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int,
        prototype_dim: int = 32,
        headroom_prototypes: int = 6,
        residual_prototypes: int = 6,
        negative_prototypes: int = 8,
        hidden_dim: int = 32,
        use_gate: bool = True,
        use_bias_branch: bool = True,
    ) -> None:
        super().__init__()
        if min(headroom_prototypes, residual_prototypes, negative_prototypes) <= 0:
            raise ValueError("Prototype counts must be positive.")
        if risk_dim <= 0:
            raise ValueError("risk_dim must be positive for specialist prototype heads.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.headroom_prototypes = torch.nn.Parameter(torch.randn(headroom_prototypes, prototype_dim) * scale)
        self.residual_prototypes = torch.nn.Parameter(torch.randn(residual_prototypes, prototype_dim) * scale)
        self.negative_prototypes = torch.nn.Parameter(torch.randn(negative_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.prototype_bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.use_gate = use_gate
        self.gate_branch = None
        if use_gate:
            self.gate_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 2),
            )
        self.bias_branch = None
        if use_bias_branch:
            self.bias_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 1),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def _scores(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        headroom = F.normalize(self.headroom_prototypes, dim=-1)
        residual = F.normalize(self.residual_prototypes, dim=-1)
        negative = F.normalize(self.negative_prototypes, dim=-1)
        headroom_score = torch.logsumexp(scale * encoded @ headroom.T, dim=1)
        residual_score = torch.logsumexp(scale * encoded @ residual.T, dim=1)
        negative_score = torch.logsumexp(scale * encoded @ negative.T, dim=1)
        return headroom_score, residual_score, negative_score

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(features)
        headroom_score, residual_score, negative_score = self._scores(encoded)
        positive_scores = torch.stack([headroom_score, residual_score], dim=1)
        if self.gate_branch is not None:
            mix = torch.softmax(self.gate_branch(risk_features), dim=1)
            positive_score = (mix * positive_scores).sum(dim=1)
        else:
            positive_score = positive_scores.max(dim=1).values
        logits = positive_score - negative_score + self.prototype_bias
        if self.bias_branch is not None:
            logits = logits + self.bias_branch(risk_features).squeeze(-1)
        return logits

    def regularization(
        self,
        features: torch.Tensor,
        *,
        headroom_positive_mask: torch.Tensor,
        residual_positive_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        headroom = F.normalize(self.headroom_prototypes, dim=-1)
        residual = F.normalize(self.residual_prototypes, dim=-1)
        negative = F.normalize(self.negative_prototypes, dim=-1)
        loss = encoded.new_tensor(0.0)

        if bool(headroom_positive_mask.any()):
            current = encoded[headroom_positive_mask]
            align = (current @ headroom.T).amax(dim=1)
            residual_overlap = (current @ residual.T).amax(dim=1)
            negative_overlap = (current @ negative.T).amax(dim=1)
            loss = loss + (1.0 - align).mean()
            loss = loss + F.relu(residual_overlap - margin).mean()
            loss = loss + F.relu(negative_overlap - margin).mean()

        if bool(residual_positive_mask.any()):
            current = encoded[residual_positive_mask]
            align = (current @ residual.T).amax(dim=1)
            headroom_overlap = (current @ headroom.T).amax(dim=1)
            negative_overlap = (current @ negative.T).amax(dim=1)
            loss = loss + (1.0 - align).mean()
            loss = loss + F.relu(headroom_overlap - margin).mean()
            loss = loss + F.relu(negative_overlap - margin).mean()

        if bool(hard_negative_mask.any()):
            current = encoded[hard_negative_mask]
            align = (current @ negative.T).amax(dim=1)
            headroom_overlap = (current @ headroom.T).amax(dim=1)
            residual_overlap = (current @ residual.T).amax(dim=1)
            loss = loss + (1.0 - align).mean()
            loss = loss + F.relu(headroom_overlap - margin).mean()
            loss = loss + F.relu(residual_overlap - margin).mean()

        return loss


class PrototypeTriageDeferHead(torch.nn.Module):
    """Prototype bank with explicit positive / neutral / harmful memories."""

    def __init__(
        self,
        feature_dim: int,
        *,
        risk_dim: int = 0,
        prototype_dim: int = 32,
        positive_prototypes: int = 8,
        neutral_prototypes: int = 8,
        harmful_prototypes: int = 8,
        hidden_dim: int = 32,
        use_risk_branch: bool = True,
    ) -> None:
        super().__init__()
        if min(positive_prototypes, neutral_prototypes, harmful_prototypes) <= 0:
            raise ValueError("Prototype counts must be positive.")

        self.feature_proj = torch.nn.Linear(feature_dim, prototype_dim, bias=False)
        self.feature_norm = torch.nn.LayerNorm(prototype_dim)
        scale = 1.0 / math.sqrt(prototype_dim)
        self.positive_prototypes = torch.nn.Parameter(torch.randn(positive_prototypes, prototype_dim) * scale)
        self.neutral_prototypes = torch.nn.Parameter(torch.randn(neutral_prototypes, prototype_dim) * scale)
        self.harmful_prototypes = torch.nn.Parameter(torch.randn(harmful_prototypes, prototype_dim) * scale)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.class_bias = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.risk_branch = None
        if use_risk_branch and risk_dim > 0:
            self.risk_branch = torch.nn.Sequential(
                torch.nn.Linear(risk_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 3),
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.feature_norm(self.feature_proj(features))
        return F.normalize(projected, dim=-1)

    def class_logits(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(features)
        scale = self.logit_scale.exp().clamp(min=1.0, max=64.0)
        pos = F.normalize(self.positive_prototypes, dim=-1)
        neutral = F.normalize(self.neutral_prototypes, dim=-1)
        harmful = F.normalize(self.harmful_prototypes, dim=-1)
        logits = torch.stack(
            [
                torch.logsumexp(scale * encoded @ pos.T, dim=1),
                torch.logsumexp(scale * encoded @ neutral.T, dim=1),
                torch.logsumexp(scale * encoded @ harmful.T, dim=1),
            ],
            dim=1,
        )
        logits = logits + self.class_bias
        if self.risk_branch is not None and risk_features is not None:
            logits = logits + self.risk_branch(risk_features)
        return logits

    def forward(self, features: torch.Tensor, risk_features: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.class_logits(features, risk_features)
        return logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1)

    def regularization(
        self,
        features: torch.Tensor,
        *,
        positive_mask: torch.Tensor,
        neutral_mask: torch.Tensor,
        harmful_mask: torch.Tensor,
        margin: float = 0.20,
    ) -> torch.Tensor:
        encoded = self.encode(features)
        banks = {
            "positive": F.normalize(self.positive_prototypes, dim=-1),
            "neutral": F.normalize(self.neutral_prototypes, dim=-1),
            "harmful": F.normalize(self.harmful_prototypes, dim=-1),
        }
        masks = {
            "positive": positive_mask,
            "neutral": neutral_mask,
            "harmful": harmful_mask,
        }
        loss = encoded.new_tensor(0.0)
        for name, mask in masks.items():
            if not bool(mask.any()):
                continue
            current = encoded[mask]
            align = (current @ banks[name].T).amax(dim=1)
            loss = loss + (1.0 - align).mean()
            for other_name, other_bank in banks.items():
                if other_name == name:
                    continue
                overlap = (current @ other_bank.T).amax(dim=1)
                loss = loss + F.relu(overlap - margin).mean()
        return loss
