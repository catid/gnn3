# Prototype Selective Evidence Agreement Experiment

## Question

Test whether changing the prototype bank readout itself, rather than adding yet
another outer gate, can recover sparse-positive correction states inside the
live evidence-agreement family.

The design goal was:

- keep the shared-vs-dual evidence-agreement structure
- replace fixed logsumexp bank pooling with per-state prototype selection
- let each row build a selected positive and negative prototype summary before
  scoring
- preserve the same Tier-1 and Tier-2 evaluation surface as the other
  evidence-agreement follow-ups

## Implementation

- New head: `SelectiveEvidenceAgreementPrototypeDeferHead`
- New runner: `scripts/run_prototype_selective_evidence_agreement_defer.py`
- Variants:
  - `prototype_selective_evidence_agree`
  - `prototype_selective_evidence_agree_hybrid`

Each branch learns per-state selectors for:

- shared positive prototypes
- shared negative prototypes
- dual positive prototypes
- dual negative prototypes

The selected bank summaries replace the fixed pooled score path, but the
evidence-agreement gate still sees the same top-match evidence features.

## Held-Out Result

This branch is closed.

### `prototype_selective_evidence_agree`

Dead on the real target and slightly harmful overall.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`
- the only effect was one false-positive non-target selection:
  - overall coverage `0.023%`
  - overall target match `96.51% -> 96.50%`
  - false-positive rate `100%`

### `prototype_selective_evidence_agree_hybrid`

Also target-dead.

Best aggregate point:

- budget `0.10%`, already saturated through `2.00%`
- overall coverage `0.023%`
- overall target match `96.51% -> 96.52%`
- overall mean delta regret `-0.00079`

But that came only from a tiny large-gap control fix:

- large-gap control coverage `0.052%`
- large-gap control target match `99.79% -> 99.84%`
- large-gap control mean delta regret `-0.00359`

The real target never moved:

- held-out `stable_positive_v2` recovery `0%` at every budget
- hard near-tie target match unchanged at `90.53%`
- hard near-tie mean delta regret `0.0000`

## Interpretation

Changing the bank readout to per-state prototype selection did not recover the
rare correction family.

So this is another negative result for the current evidence-agreement family:

- it is worse than `prototype_hybrid` on the ultra-low-coverage frontier
- it is worse than `prototype_memory_agree_blend_hybrid` on the micro-budget
  Tier-1 point
- it is worse than `prototype_agree_mix_hybrid` on coverage-efficient
  matched-band quality
- it is worse than `prototype_evidence_agree_hybrid` on aggregate matched-band
  quality

## Decision

Close `prototype_selective_evidence_agree` and
`prototype_selective_evidence_agree_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
