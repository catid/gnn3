# Path Tie-Break Round 8

## Scope

Round eight opened this backup only after two facts were already established:

1. the hard remaining opportunity sits in the hard near-tie slice
2. critic-guided bounded search was too slow even on the narrowed two-suite scout

This backup tested a smaller mechanism:

- trigger only on ambiguous states
- expand only the top-2 actions
- break ties with local suffix-cost comparison
- do not replace the global constructor

## Result

The backup was cheaper than critic-guided bounded search, but still too slow for scout use on the same targeted OOD lane.

Observed runtime before kill:

- top-2 local suffix-cost tie-break scout: about `10.3m`

This was still too expensive for a supposedly cheap ambiguous-state correction path, especially since it had not yet earned promotion on completed output metrics.

## Decision

Do not promote the round-eight path tie-break backup.

Interpretation:

- the near-tie problem does admit narrower decision-time correction mechanisms than full bounded search
- but even the cheap local-planner version still carries too much runtime overhead in the current setup
- that is not a good enough trade for deployment or for expansion to matched-seed candidate runs this round
