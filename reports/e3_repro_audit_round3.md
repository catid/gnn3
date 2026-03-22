# E3 Repro Audit Round 3

## Question

Why did the archived-best `E3` artifact outperform the round-two replay even though the nominal config family was the same?

## Hard Evidence

- The archived-best artifact is [e3_memory_hubs_rsm](/home/catid/gnn3/artifacts/experiments/e3_memory_hubs_rsm).
- The weaker replay is [e3_memory_hubs_rsm_round2_repro](/home/catid/gnn3/artifacts/experiments/e3_memory_hubs_rsm_round2_repro).
- Both metadata files report the same nominal benchmark seed and the same visible high-level config family:
  - base seed `103`
  - `router_variant=memory_hubs`
  - `detach_warmup=true`
  - `final_step_only_loss=true`
  - `penultimate_grad_prob=0.25`
- Despite that, the recorded dataset sizes differ:
  - archived best: `train=2254`, `val=596`, `test=596`
  - round-two replay: `train=2149`, `val=541`, `test=541`
- On current live code, regenerating the base `E3` config deterministically reproduces the replay counts and stable manifest hashes:
  - `2149 / 541 / 541`
  - train manifest `32b37ce5841c5a1c2614c9c8458d5af5b5f6deedff45a3ee5e6386165a2ae6b3`
  - val manifest `0c2450faa9cb1afd9637322f1100c30daeec2c801a0edfb0d237f73c7cbd8863`
  - test manifest `0c2450faa9cb1afd9637322f1100c30daeec2c801a0edfb0d237f73c7cbd8863`
- The archived-best artifact does not contain `dataset_manifests.json`, manifest hashes, branch provenance, or a recoverable git commit. Its metadata lists `git_commit: "unknown"`.
- The archived-best run predates split-specific dataset seeds. On current pre-fix semantics, val and test with the same episode count hash identically, which means the earlier pipeline evaluated on an aliased validation/test split.

## What This Means

- The archived-best `E3` artifact is not a clean replay target.
- The replay gap is not just “training variance.” The archived-best run and the later replay were not operating on the same reconstructable dataset instance.
- The missing archived commit means there is no code-accurate reconstruction target for that artifact.
- The older val/test aliasing means earlier checkpoint selection and final test reporting were methodologically weaker than the current round-three setup.

## Secondary Contributors

- Training is still nondeterministic:
  - CUDA bf16 autocast
  - dropout
  - stochastic penultimate-step gradient inclusion
  - no deterministic-algorithm mode
- Checkpoint-selection policy changed after the archived-best run.
  - archived-best trainer used `val_next_hop_accuracy + solved_rate`
  - current trainer uses a more regret/deadline-aware blended score
  - this matters for best-checkpoint sweeps, but not enough to explain the final-summary gap on its own

## Practical Conclusion

- Treat the archived-best `E3` numbers as historical signal, not as a reproducible current-code baseline.
- Use fresh round-three matched runs with:
  - persisted manifests
  - split-specific dataset seeds
  - shared seed lists
  - shared hardware placement
- Judge `X6` against the fresh matched `E3` contender, not against the archived-best artifact.
