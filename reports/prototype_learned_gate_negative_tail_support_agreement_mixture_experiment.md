# Prototype Learned-Gate Negative-Tail Support Agreement-Mixture Experiment

## Question

Test whether the live support-weighted agreement-mixture branch improves if the
shared and dual negative-tail cleanup gates are **learned directly** from
internal negative-bank summary features, instead of using hand-written
sharpness or tail-mass formulas.

The design goal was:

- keep the live support-weighted agreement-mixture geometry
- keep negative-bank-only cleanup
- let each branch learn its own cleanup trigger from:
  - lead negative logit
  - top-1 vs top-2 gap
  - lead-vs-tail-mass gap
- preserve the sharp-negative branch's low-coverage quality while recovering
  more recall at `1–2%` coverage

This is the first tiny learned internal gate over negative-bank summary
features.

## Implementation

- New head:
  `LearnedGateNegativeTailSupportAgreementMixturePrototypeDeferHead`
- New runner:
  `scripts/run_prototype_learned_gate_negative_tail_support_agreement_mixture_defer.py`
- Variants:
  - `prototype_learned_gate_negative_tail_support_agree_mix`
  - `prototype_learned_gate_negative_tail_support_agree_mix_hybrid`

Relative to the older sharp-negative and mass-negative branches:

- the negative-bank tail penalty is still the same soft cleanup
- but the cleanup gate is now learned
- shared and dual banks each get their own linear gate over summary features

So this is still a bank-internal cleanup change, but now the trigger is learned
instead of hand-coded.

## Held-Out Result

### `prototype_learned_gate_negative_tail_support_agree_mix`

Closed, effectively dead.

Best point:

- budget `0.10–2.00%`
- held-out `stable_positive_v2` recovery `0%`
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`
- overall mean delta regret `0.0000`

### `prototype_learned_gate_negative_tail_support_agree_mix_hybrid`

Closed positive, but not a new live lead.

At `0.10%` nominal budget:

- overall coverage `0.12%`
- held-out `stable_positive_v2` recovery `25%`
- hard near-tie target match `90.53% -> 90.60%`
- hard near-tie mean delta regret `-0.0048`
- overall target match `96.51% -> 96.63%`
- overall mean delta regret `-0.0057`

At `1.00%` nominal budget:

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery `50%`
- hard near-tie target match `90.53% -> 90.66%`
- hard near-tie mean delta regret `-0.0071`
- overall target match `96.51% -> 96.79%`
- overall mean delta regret `-0.0145`

At `1.50%` nominal budget:

- overall coverage `1.46%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.82%`
- overall mean delta regret `-0.0154`

At `2.00%` nominal budget:

- overall coverage `1.70%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- hard near-tie mean delta regret `-0.0089`
- overall target match `96.51% -> 96.86%`
- overall mean delta regret `-0.0161`

Large-gap controls stayed clean:

- large-gap target match `99.79% -> 99.90%`
- large-gap mean delta regret improved
- no harmful large-gap miss pattern appeared

## Comparison against live leads

`prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%`

- overall coverage `0.76%`
- held-out `stable_positive_v2` recovery `75%`
- hard near-tie target match `90.53% -> 90.73%`
- overall mean delta regret `-0.0144`

`prototype_learned_gate_negative_tail_support_agree_mix_hybrid @ 0.75%`

- same overall coverage `0.76%`
- only `25%` held-out `stable_positive_v2` recovery
- only `90.53% -> 90.60%`
- overall mean delta regret `-0.0130`

So the learned internal gate clearly loses to the live sharp-negative branch
below `1%`.

`prototype_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.00%`
- held-out `stable_positive_v2` recovery `100%`
- hard near-tie target match `90.53% -> 90.80%`
- overall mean delta regret `-0.0131`

`prototype_learned_gate_negative_tail_support_agree_mix_hybrid @ 1.00%`

- overall coverage `1.01%`
- held-out `stable_positive_v2` recovery only `50%`
- hard near-tie target match only `90.53% -> 90.66%`
- overall mean delta regret `-0.0145`

So the learned gate also does not preserve the fixed negative-tail branch's
high-recall lane.

`prototype_support_weighted_agree_mix_hybrid @ 1.50%`

- overall mean delta regret `-0.0158`

`prototype_learned_gate_negative_tail_support_agree_mix_hybrid @ 1.50%`

- overall mean delta regret `-0.0154`

At `2.00%`, it still trails the older higher-budget reference slightly:

- `prototype_support_weighted_agree_mix_hybrid`: `-0.0165`
- `prototype_learned_gate_negative_tail_support_agree_mix_hybrid`: `-0.0161`

## Interpretation

This is a real positive, but still dominated.

Current read:

- the learned gate can recover the full `75%` / `90.73%` frontier
- but it only does so once coverage rises to about `1.5%`
- below `1%`, it still collapses toward the old weak middle
- once it reaches the higher-budget matched band, it is still slightly weaker
  than the older support-weighted agreement-mixture reference

So learned internal gating over these simple negative-bank summaries is not
enough to improve the live frontier.

## Decision

Close:

- `prototype_learned_gate_negative_tail_support_agree_mix`
- `prototype_learned_gate_negative_tail_support_agree_mix_hybrid`

Keep the live shortlist unchanged:

- `prototype_memory_agree_blend_hybrid` for micro-budget Tier-1
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` for the best
  sub-`1%` full-band / coverage-efficient matched-band point
- `prototype_negative_tail_support_agree_mix_hybrid` for maximum held-out
  recall around `1%`
- `prototype_support_weighted_agree_mix_hybrid` as the higher-budget
  matched-band reference
