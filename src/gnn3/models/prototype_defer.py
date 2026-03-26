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
