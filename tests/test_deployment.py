from __future__ import annotations

from gnn3.data.hidden_corridor import HiddenCorridorConfig
from gnn3.eval.deployment import round4_verifier_risk_switch


def test_round4_verifier_risk_switch_stays_base_on_default_multiheavy() -> None:
    decision = round4_verifier_risk_switch(HiddenCorridorConfig())
    assert decision.variant == "base"
    assert decision.reasons == ("default_low_risk",)


def test_round4_verifier_risk_switch_activates_on_high_branching_and_packets() -> None:
    decision = round4_verifier_risk_switch(
        HiddenCorridorConfig(
            branching_factor=3,
            packets_max=6,
        )
    )
    assert decision.variant == "verifier"
    assert "branching>=3" in decision.reasons
    assert "packets>=6" in decision.reasons


def test_round4_verifier_risk_switch_activates_on_heavy_dynamic_pressure() -> None:
    decision = round4_verifier_risk_switch(
        HiddenCorridorConfig(
            community_base_queue=(2.5, 8.0),
            queue_penalty=2.5,
            capacity_penalty=3.0,
            urgency_penalty=1.4,
        )
    )
    assert decision.variant == "verifier"
    assert "queue_low>=2.5" in decision.reasons
    assert "queue_penalty>=1.0" in decision.reasons
    assert "capacity_penalty>=3.0" in decision.reasons
    assert "urgency_penalty>=1.4" in decision.reasons
