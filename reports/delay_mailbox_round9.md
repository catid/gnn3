# Round 9 Delay Mailbox Report

## Goal

Round nine reopened explicit delayed state in the narrowest form still not
already closed by prior rounds:

- tiny residual mailbox
- fixed short delays
- minimal placement choices
- evaluated against the current `multiheavy` guardrail

The question was whether explicit delayed state can improve hard near-tie
decisions without reopening the old generic history-memory family.

## Scout Variants

Two seed314 scouts were run on the corrected round-nine benchmark:

1. `mailbox_monitor12_seed314`
   - monitor-only placement
   - delays `{1, 2}`
2. `mailbox_hubmonitor124_seed314`
   - hub + monitor placement
   - delays `{1, 2, 4}`

## Matched Seed314 Baseline

Seed314 `multiheavy` baseline rollout:

- regret `1.898`
- p95 regret `8.140`
- miss `43.75%`

## Results

### `mailbox_monitor12_seed314`

- epoch-2 rollout: regret `6.676`, p95 `20.028`, miss `68.75%`
- epoch-3 rollout: regret `7.542`, p95 `23.491`, miss `68.75%`

### `mailbox_hubmonitor124_seed314`

- epoch-2 rollout: regret `7.029`, p95 `21.113`, miss `56.25%`

## Verdict

Both plain mailbox scouts failed their early gates decisively.

What they show:

- explicit delayed state in this direct mailbox form is not a safe upgrade
- the failure is not subtle; rollout metrics were far behind the matched
  baseline almost immediately
- broader placement and a larger delay set did not rescue the family

## Round-Nine Read

The plain mailbox family is closed as a standalone branch for this round.

It should only be reconsidered if:

- a compute-triggered or teacher-guided branch first demonstrates real frontier
  gains, and
- a mailbox is tested only as a tightly scoped contingent combo rather than as
  a general replacement mechanism

Until then, round nine should spend budget on:

- offline branch-teacher supervision
- adaptive continuation / compute compression
- other compute-first mechanisms that directly target the hard near-tie slice
