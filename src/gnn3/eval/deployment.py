from __future__ import annotations

from dataclasses import dataclass

from gnn3.data.hidden_corridor import HiddenCorridorConfig


@dataclass(frozen=True)
class DeploymentDecision:
    variant: str
    rule: str
    reasons: tuple[str, ...]


def round4_verifier_risk_switch(config: HiddenCorridorConfig) -> DeploymentDecision:
    reasons: list[str] = []
    if config.branching_factor >= 3:
        reasons.append("branching>=3")
    if config.packets_max >= 6:
        reasons.append("packets>=6")
    if config.tree_depth_max >= 4:
        reasons.append("depth>=4")
    if config.community_base_queue[0] >= 2.5:
        reasons.append("queue_low>=2.5")
    if config.queue_penalty >= 1.0:
        reasons.append("queue_penalty>=1.0")
    if config.capacity_penalty >= 3.0:
        reasons.append("capacity_penalty>=3.0")
    if config.urgency_penalty >= 1.4:
        reasons.append("urgency_penalty>=1.4")

    if reasons:
        return DeploymentDecision(
            variant="verifier",
            rule="round4_verifier_risk_switch",
            reasons=tuple(reasons),
        )
    return DeploymentDecision(
        variant="base",
        rule="round4_verifier_risk_switch",
        reasons=("default_low_risk",),
    )

