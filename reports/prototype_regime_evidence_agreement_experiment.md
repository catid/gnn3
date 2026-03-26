# Prototype Regime-Split Evidence Agreement Experiment

## Question

Test whether the live evidence-agreement family should branch explicitly around
the two source families that still matter:

- `high_headroom_near_tie`
- `baseline_error_hard_near_tie`

The design goal was:

- keep the current shared-vs-dual evidence-agreement structure
- replace one generic bank family with two regime-specific bank families
- let a small regime head choose between the headroom and residual scores
- supervise that regime choice where the cached metadata already identifies a
  clean headroom or residual source-family label

## Implementation

- New head: `RegimeSplitEvidenceAgreementPrototypeDeferHead`
- New runner: `scripts/run_prototype_regime_evidence_agreement_defer.py`
- Variants:
  - `prototype_regime_evidence_agree`
  - `prototype_regime_evidence_agree_hybrid`

The head trains two separate evidence-agreement bank families:

- headroom regime banks
- residual / baseline-error regime banks

The final score is a learned mixture of those two regime scores.

## Held-Out Result

This branch is closed.

### `prototype_regime_evidence_agree`

Dead on the target and broad-false-positive off target.

- recovered `0%` of held-out `stable_positive_v2` at every budget
- hard near-tie target match stayed at `90.53%`
- hard near-tie mean delta regret stayed at `0.0000`
- it only spent more coverage on non-target states:
  - overall coverage rose to about `0.58%`
  - large-gap control coverage rose past `1%`
  - but system target match and regret stayed unchanged

### `prototype_regime_evidence_agree_hybrid`

Weak positive, but not a live lead.

Best coverage-efficient point:

- budget `0.75%`
- overall coverage `0.66%`
- held-out `stable_positive_v2` recovery `25%`
- overall target match `96.51% -> 96.63%`
- overall mean delta regret `-0.00827`

But the real trade is still weak:

- hard near-tie target match only `90.53% -> 90.60%`
- hard near-tie mean delta regret only `-0.00476`
- false-positive rate on selected hard near-tie states was about `96.9%`
- high-headroom near-tie coverage ballooned to about `38.6%`
- baseline-error near-tie coverage ballooned to about `18.8%`
- both of those regime slices were still almost entirely false positives

So the branch did find some real signal, but it was too diffuse and too
imprecise to beat the current shortlist.

## Interpretation

Source-family specialization inside the evidence-agreement family is not enough
on its own.

The regime split did recover one held-out stable-positive-v2 case, but it did
so by spending far too much coverage across the regime slices it was supposed to
sharpen.

So this is still worse than the live alternatives:

- worse than `prototype_hybrid` on the ultra-low-coverage frontier
- worse than `prototype_memory_agree_blend_hybrid` on the micro-budget Tier-1
  point
- worse than `prototype_agree_mix_hybrid` on coverage-efficient matched-band
  quality
- worse than `prototype_evidence_agree_hybrid` on aggregate matched-band
  quality

## Decision

Close `prototype_regime_evidence_agree` and
`prototype_regime_evidence_agree_hybrid`.

Keep the shortlist unchanged:

- `prototype_hybrid` remains the best ultra-low-coverage lead
- `prototype_memory_agree_blend_hybrid` remains the best micro-budget Tier-1
  follow-up
- `prototype_agree_mix_hybrid` remains the best coverage-efficient matched-band
  follow-up
- `prototype_evidence_agree_hybrid` remains the best aggregate-quality
  matched-band follow-up
