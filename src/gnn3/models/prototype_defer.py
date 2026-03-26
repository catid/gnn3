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
