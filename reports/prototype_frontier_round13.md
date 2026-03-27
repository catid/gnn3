# Round 13 Prototype Frontier

## Scope

Wave A mapped the matched-budget frontier for the four accepted round-12/early-round-13 prototype references:

- `prototype_memory_agree_blend_hybrid`
- `prototype_sharp_negative_tail_support_agree_mix_hybrid`
- `prototype_negative_tail_support_agree_mix_hybrid`
- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`

The campaign used the canonical held-out pack, `stable_positive_v2`, and the hard near-tie pack. Outputs live in:

- `reports/plots/round13_prototype_frontier_*`
- `reports/plots/round13_prototype_frontier_rerun1_*`
- `reports/plots/round13_prototype_frontier_rerun2_*`

## Archived Frontier

The archived decision exports still show a four-lane frontier:

- `0.25%` to `0.50%`: `prototype_memory_agree_blend_hybrid`
  - `66.7%` `stable_positive_v2` recall
  - hard near-tie `90.39% -> 90.53%`
  - overall mean delta regret `-0.0049` to `-0.0093`
- `0.75%`: `prototype_sharp_negative_tail_support_agree_mix_hybrid`
  - `75.0%` `stable_positive_v2` recall
  - hard near-tie `90.53% -> 90.73%`
  - overall mean delta regret `-0.0144`
- `1.00%` to `1.50%`: `prototype_negative_tail_support_agree_mix_hybrid`
  - `100.0%` `stable_positive_v2` recall
  - hard near-tie `90.53% -> 90.80%`
  - overall mean delta regret `-0.0131` to `-0.0138`
- `2.00%+`: `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`
  - `100.0%` `stable_positive_v2` recall
  - hard near-tie only `90.53% -> 90.68%` in the saved repo outputs
  - overall mean delta regret `-0.0167`

Important repo-grounded correction:

- the saved archived CSVs do **not** support the earlier handoff claim that archived branchwise-max reaches hard near-tie `90.80%`
- in the repo outputs, `90.80%` belongs to archived `prototype_negative_tail_support_agree_mix_hybrid`
- archived branchwise-max wins on higher-budget aggregate regret, not on the best archived hard-slice target match

## Rerun Robustness

Two fresh rerun sweeps materially changed the interpretation.

### Rerun 1

Only `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` remained clearly alive across the matched budget ladder:

- `0.25%`: `50.0%` recall, hard near-tie `90.39% -> 90.45%`, overall mean delta regret `-0.0081`
- `1.00%`: `66.7%` recall, hard near-tie `90.39% -> 90.53%`, overall mean delta regret `-0.0134`
- `1.50%` to `2.00%`: `83.3%` recall, hard near-tie `90.53% -> 90.60%`, overall mean delta regret `-0.0154` to `-0.0162`

The other three archived frontier branches collapsed sharply:

- `prototype_memory_agree_blend_hybrid`: dead at every budget
- `prototype_sharp_negative_tail_support_agree_mix_hybrid`: dead at every budget
- `prototype_negative_tail_support_agree_mix_hybrid`: dead through `1.0%`, only a weak `25%` niche at `1.5%+`

Always-dominates pairs in rerun 1:

- branchwise-max dominates memory-agree at every matched budget
- negative-tail dominates memory-agree at every matched budget
- negative-tail dominates sharp-negative at every matched budget

### Rerun 2

Rerun 2 partially revived `prototype_sharp_negative_tail_support_agree_mix_hybrid`, but the only consistently leading system was again branchwise-max:

- `0.25%`: branchwise-max `50.0%` recall, hard near-tie `90.39% -> 90.45%`, overall mean delta regret `-0.0083`
- `0.75%`: branchwise-max `66.7%` recall, hard near-tie `90.39% -> 90.53%`, overall mean delta regret `-0.0129`
- `1.50%` to `2.50%`: branchwise-max `83.3%` recall, hard near-tie `90.53% -> 90.60%`, overall mean delta regret `-0.0159` to `-0.0165`

Secondary behavior:

- `prototype_sharp_negative_tail_support_agree_mix_hybrid` recovered a partial middle lane:
  - `50.0%` recall by `0.75%`
  - hard near-tie capped at `90.53% -> 90.66%`
  - overall mean delta regret `-0.0108` at `0.75%`
- `prototype_memory_agree_blend_hybrid`: dead
- `prototype_negative_tail_support_agree_mix_hybrid`: dead

Always-dominates pairs in rerun 2:

- branchwise-max dominates memory-agree at every matched budget
- negative-tail dominates memory-agree at every matched budget
- sharp-negative dominates memory-agree at every matched budget
- sharp-negative dominates negative-tail at every matched budget

## Frontier Decision

Wave A changes the accepted operating picture:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` is the only **robust** live prototype branch
- `prototype_sharp_negative_tail_support_agree_mix_hybrid` remains mechanistically informative, but it is no longer promotion-safe as a standalone deployed operating point
- `prototype_negative_tail_support_agree_mix_hybrid` remains useful as a source-family analysis reference, but it failed the rerun promotion gate
- `prototype_memory_agree_blend_hybrid` should be demoted from the active frontier pack

## Promotion Outcome

Promoted:

- `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid`
  - accepted as the robust round-13 prototype reference
  - accepted as the only single-branch candidate for any later deployment panel

Demoted from active promotion surface:

- `prototype_memory_agree_blend_hybrid`
- `prototype_sharp_negative_tail_support_agree_mix_hybrid`
- `prototype_negative_tail_support_agree_mix_hybrid`

These branches remain useful as mechanism probes, not as accepted robust operating points.
