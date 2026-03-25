# Near-Tie Search Round 8

## Scope

Round eight only opened bounded correction after the counterfactual critic family showed real hard near-tie error signal.

This track tested:

1. critic-guided bounded search on the hard OOD near-tie suites
2. a narrower local suffix-cost tie-break backup after bounded search proved too slow

The target suites were:

- `deeper_packets6`
- `heavy_dynamic`

The base policy was the fresh seed `312` round-eight `multiheavy` guardrail.

## Full-Suite Search: Killed On Runtime

The first bounded-search scouts were run on the full evaluation suite and killed before completion.

Observed wall-clock cost before kill:

- scalar-Q search: about `1456s` (`24.3m`)
- pairwise search: about `731s` (`12.2m`)

That was already too slow for a scout in this repo, so round eight narrowed the search lane to the two OOD suites that actually contain most of the hard near-tie opportunity.

## Targeted OOD Search: Still Too Slow

The narrower two-suite search scouts were then relaunched on:

- `deeper_packets6`
- `heavy_dynamic`

Even after dropping the base suite and `branching3`, both targeted runs were still too expensive for scout use:

- scalar-Q targeted search was stopped after about `16.3m`
- pairwise targeted search was stopped after about `16.3m`

Key point:

- this was still only a two-suite scout
- both GPUs stayed busy for a long time
- neither run landed quickly enough to justify expansion to candidate seeds

That is enough to treat critic-guided bounded rollout search as a runtime negative for this round.

## Backup Tie-Breaker: Also Crossed The Runtime Line

After killing bounded search, round eight tested a narrower backup:

- near-tie gate only
- top-2 local suffix-cost tie-break
- no critic rollout recursion

This was meant to answer a simpler question:

- if the ambiguity is truly local, can a very cheap planner-style tie-breaker help without paying bounded-search cost?

Result:

- the top-2 path-cost tie-breaker scout was also stopped on runtime grounds
- it crossed about `10.3m` on the same two-suite target before completing

So the backup was cheaper than bounded search, but still not cheap enough to count as a practical narrow correction path for the current round.

## Decision

Round eight does **not** promote any search-time correction branch.

The conclusions are:

1. full bounded search is too slow for this repo at scout time
2. even targeted two-suite search remains too slow to justify promotion
3. the narrower local tie-break backup is cheaper, but still crosses the runtime bar before proving enough value
4. distillation was not opened, because no search-time branch first earned promotion

This closes the search-side thesis for round eight:

- counterfactual critic signal exists
- but the runtime cost of turning that signal into bounded decision-time correction is still too high in the current implementation and problem setting
