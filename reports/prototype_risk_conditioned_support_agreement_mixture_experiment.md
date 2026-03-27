# Risk-Conditioned Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch can improve if
per-prototype support weights are allowed to shift **per state** from the cheap
risk features, instead of staying static across the whole bank.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep the same shared/dual agreement gate
- keep the same static bounded support weights
- add a tiny risk-conditioned support delta for each shared/dual positive and
  negative prototype bank

This was the smallest state-conditional follow-up on top of the current
matched-band lead.

## Implementation

- New head: `RiskConditionedSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_risk_conditioned_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_risk_support_agree_mix`
  - `prototype_risk_support_agree_mix_hybrid`

The new mechanism adds:

- a small MLP from risk features to per-state support deltas
- mean-centered bounded deltas over each prototype bank
- those deltas are added on top of the static support logits before
  `logsumexp`
- an extra small regularizer discourages large dynamic support shifts

## Held-Out Result

### `prototype_risk_support_agree_mix`

Dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `0.0000`

### `prototype_risk_support_agree_mix_hybrid`

Also dead.

Best point:

- budget `2.0%`
- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- overall mean delta regret `0.0000`

One detail is worth noting:

- the branch still selected a nontrivial fraction of hard near-tie states
- at `2.0%` nominal budget it covered `6.78%` of the hard near-tie slice
- but every selected decision was effectively policy-identical on the measured
  target metrics

So the added state-conditional support path did not sharpen the live branch. It
just changed which inert states were ranked highest.

## Comparison against current leads

`prototype_support_weighted_agree_mix_hybrid @ 1.0%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0148`

`prototype_risk_support_agree_mix_hybrid @ 2.0%`

- overall coverage `2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- overall mean delta regret `0.0000`

So the dynamic support version is not merely weaker. It is completely dominated
by the static support-weighted agreement-mixture branch.

It also loses to the older `prototype_hybrid` ultra-low-coverage lead and to
the round-eleven `margin_regime` reference, because it never moves the accepted
frontier at all.

## Interpretation

This closes another branch cleanly.

Current read:

- support weighting helps as a **global bank cleanup** mechanism
- but adding per-state support modulation on top of the live static
  support-weighted agreement-mixture head does not recover additional signal
- the remaining frontier is not being solved by tiny dynamic perturbations of
  the support-weighted bank

So this is a hard negative:

- no held-out stable-positive recovery
- no hard near-tie movement
- no aggregate regret improvement

## Decision

Close:

- `prototype_risk_support_agree_mix`
- `prototype_risk_support_agree_mix_hybrid`

Keep the shortlist unchanged:

- `prototype_hybrid` for ultra-low coverage
- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_support_weighted_agree_mix_hybrid` as the primary matched-band
  branch
