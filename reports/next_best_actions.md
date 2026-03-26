# Next Best Actions

1. Keep plain `multiheavy` as the default exploit policy. Round twelve still did not produce a deployable correction branch that beats it cleanly on the accepted hard near-tie frontier.
2. Keep using the round-nine/ten frontier pack and guard as the only Tier-2 promotion surface:
   - hard near-tie intersection
   - stable near-tie
   - high-headroom near-tie
   - baseline-error subset inside the intersection
   - large-gap control slice
3. Keep treating the live opportunity as a **tiny stable-positive correction** problem. Round twelve confirmed that the richer teacher bank still yields only `46` stable-positive-v2 audited decisions total, with just `4` held-out positives across seeds `315` and `316`.
4. Do not claim that richer teacher-bank construction solved the source-family problem. It improved confidence, but not coverage:
   - `compute5` remains the best safe teacher on `45 / 46` stable-positive-v2 cases
   - fine-signature overlap across seeds is still `0.0`
   - coarse-signature overlap is still only `0.0833`, `0.1667`, and `0.0`
5. Do not promote round-twelve `margin_regime`. It is still the only surviving learned defer gate, but it only matches the old behavior:
   - at `0.5%` coverage it recovers `50%` of the held-out stable-positive-v2 pack
   - hard near-tie mean delta regret is `-0.0071`
   - overall mean delta regret is only `-0.0075`
   - that does not beat the round-eleven reference
6. Keep the round-eleven `margin_regime` defer at `1–2%` coverage as the current reference operating point. It is still the strongest remembered deployment-style branch:
   - `1%`: hard near-tie delta regret `-0.0071`, overall `-0.0134`
   - `2%`: hard near-tie delta regret `-0.0089`, overall `-0.0151`
7. Remember round-twelve committee defer only as an **offline upper bound**, not a deployment policy. On held-out seeds it is cleaner but smaller:
   - about `0.25%` overall coverage
   - `50%` stable-positive-v2 recovery
   - `100%` stable-positive precision
   - hard near-tie delta regret only `-0.0029`
   - requires per-state teacher-bank knowledge
8. Keep raw retrieval / hand-built prototype defer closed, but keep a small shortlist of narrow architecture leads inside the prototype family:
   - plain prototype bank is still dead
   - committee-only prototype bank is still too weak
   - explicit positive / neutral / harmful triage prototypes are also closed
   - candidate-aware prototype defer with top-2 pair features is also closed
   - auxiliary pair-context branches on top of the prototype bank are also closed
   - temporal outer-step context hybrids are also closed
   - risk-conditioned gated prototype hybrids are also closed
   - specialist-bank prototype splits are also closed
   - shallow prototype-space adapter hybrids are also closed
   - coarse centroid / multiscale prototype hybrids are also closed
   - top-k prototype evidence readouts are also closed
   - score-band prototype residuals are also closed
   - explicit harmful-memory suppressor banks are also closed
   - score-aware prototype interaction residuals are also closed
   - positive-only prototype lift residuals are also closed
   - asymmetric dual-projection prototype defer is also closed
   - but `prototype_hybrid` recovered `75%` of held-out stable-positive-v2
   - at `0.75%` overall coverage it reached hard near-tie target match `90.53% -> 90.73%`
   - hard near-tie mean delta regret matched the round-eleven `2%` reference at `-0.0089`
   - overall mean delta regret at that point was still only `-0.0097`, so it is a lead, not a promotion
   - `prototype_memory_agree_blend_hybrid` is now the best micro-budget Tier-1 follow-up
   - at `0.25%` overall coverage it recovered `50%` of held-out stable-positive-v2 and reached the weaker but still real `90.53% -> 90.66%` hard near-tie band
   - at `0.50%` overall coverage it still held `50%` stable-positive-v2 recovery, kept the same `90.53% -> 90.66%` hard near-tie band, and beat `prototype_hybrid` at the same budget on both hard-slice regret and overall mean delta regret
   - but it capped out there and never reached the full `75%` / `90.73%` frontier band, so it is a micro-budget companion rather than a replacement lead
   - direct fusion of that memory anchor with the richer evidence-aware inner agreement gate is now also closed
   - `prototype_memory_evidence_blend_hybrid` gave back the micro-budget Tier-1 win, recovering only `25%` at `0.50–1.50%` nominal budgets and only reaching `50%` by `2.0%`, while still capping out at the weaker `90.53% -> 90.66%` hard near-tie band
   - parallel dual-lift max routing on top of the same memory anchor is also closed
   - `prototype_memory_duallift_hybrid` only reproduced the weaker `25%` held-out stable-positive-v2 / `90.53% -> 90.60%` ultra-low-coverage behavior and never challenged either the micro-budget memory-agreement lead or the matched-band agreement leads
   - `prototype_mixture_hybrid` was the first follow-up that fully caught that hard-slice band at matched higher coverage
   - `prototype_agree_mix_hybrid` now improves on it in coverage efficiency
   - at `1.5%` nominal budget it matched `75%` held-out stable-positive-v2 recovery and the same `90.53% -> 90.73%` hard near-tie band
   - it reached that matched band at about `1.05%` overall coverage instead of `1.84%`
   - its overall mean delta regret at that point was `-0.0137`, essentially matching the old mixture result while paying much less coverage
   - but at `0.75%` it only recovered `50%` of held-out stable-positive-v2 and stayed in the weaker `90.53% -> 90.66%` band
   - temporal scalar context was cleaner overall, but only recovered `50%` of held-out stable-positive-v2 and only reached hard near-tie `90.53% -> 90.66%`, so it is not the right architecture direction
   - gated rescaling of prototype evidence also capped out at `50%` held-out stable-positive-v2 recovery and the same weaker `90.53% -> 90.66%` hard near-tie band
   - specialist source-family banks also capped out at `50%` held-out stable-positive-v2 recovery, but only at materially higher coverage than `prototype_hybrid`
   - shallow prototype-space adapters also capped out at `50%` held-out stable-positive-v2 recovery and only reached the weaker `90.53% -> 90.66%` hard near-tie band
   - multiscale centroid branches improved broad overall caution but capped out at `25–50%` held-out stable-positive-v2 recovery and only reached `90.53% -> 90.66%` on hard near-tie
   - top-k prototype evidence readouts only recovered `25%` of held-out stable-positive-v2 and the hybrid version collapsed fully to baseline
   - score-band prototype residuals also capped out at `50%` held-out stable-positive-v2 recovery and only reached the weaker `90.53% -> 90.66%` hard near-tie band
   - explicit harmful-memory suppressor banks recovered `0%` of held-out stable-positive-v2 and effectively collapsed to baseline
   - score-aware interaction residuals improved aggregate overall regret but still capped out at `25%` held-out stable-positive-v2 recovery and only reached the weaker `90.53% -> 90.60%` hard near-tie band
   - positive-only lift residuals are also closed: the plain lift recovered `0%` of held-out stable-positive-v2, while the gated lift only reached `25%` recovery and still capped out at the weaker `90.53% -> 90.60%` hard near-tie band
   - asymmetric dual-projection prototype defer is also closed: it found a tiny useful niche at ultra-low coverage, but still capped out at `25%` held-out stable-positive-v2 recovery and the same weaker `90.53% -> 90.60%` hard near-tie band
   - shared-anchor cascade prototype defer is also closed: it improved aggregate regret cleanly, but capped out at `50%` held-out stable-positive-v2 recovery and still only reached the weaker `90.53% -> 90.66%` hard near-tie band
   - switch-gated branch routing is also closed: plain switch selected only inert controls, and the hybrid switch collapsed fully to baseline
   - anchor-biased evidence agreement is also closed: the plain anchored head only recovered `25%` of held-out stable-positive-v2 and only reached the weaker `90.53% -> 90.60%` hard near-tie band, while the anchored hybrid collapsed to `0%` held-out stable-positive-v2 recovery
   - one-sided positive-lift evidence agreement is also closed: the plain lift recovered `0%` of held-out stable-positive-v2, while the hybrid improved broad overall regret but still recovered only `25%` of held-out stable-positive-v2 and stayed in the weaker `90.53% -> 90.60%` hard near-tie band
   - contrastive evidence agreement is also closed: the plain contrastive gate recovered `0%` of held-out stable-positive-v2, while the hybrid reached `50%` recovery only at high coverage and still capped out at the weaker `90.53% -> 90.66%` hard near-tie band
   - sharpness-aware evidence agreement is also closed: the plain sharpness head only recovered `25%` of held-out stable-positive-v2 and still only reached the weaker `90.53% -> 90.60%` hard near-tie band, while the sharpness hybrid recovered `0%` and collapsed to baseline on the target slice
9. Do not reopen conservative student retry. Round-twelve positive mining showed that source-family expansion through training-side mining is too noisy:
   - best coarse mining recovered all held-out positives only at `5.97%` precision
   - broader mining fell to `1.77%` to `2.94%` precision
   - regime mining was worse and introduced nonzero harmful selection
10. Keep the representation diagnosis unchanged. The backbone still appears to expose most of the local signals; the open problem is still precision calibration and abstention on a tiny ambiguous subset.
11. If another round opens, bias it toward **prototype-memory hybrid defer** and **agreement-gated prototype-mixture hybrid defer** before any broader family:
   - keep the richer teacher-bank filters from round twelve
   - use the learnable prototype-memory plus risk-branch architecture as the ultra-low-coverage contender
   - use the memory-agreement blend hybrid as the micro-budget contender below roughly `0.5%` overall coverage
   - use the agreement-gated geometry-mixture head as the matched-band contender
   - use the evidence-calibrated agreement-mixture head only when aggregate-quality-at-higher-coverage matters more than coverage efficiency
   - compare against the round-eleven `margin_regime` reference at matched or lower coverage
   - preserve large-gap controls and broad feasible-suite behavior
12. If another round opens beyond that, bias it toward **richer teachers first, then ultra-low-coverage defer**:
   - more diverse safe teacher bank members
   - teacher agreement / correction-margin filters
   - explicit held-out comparison against the round-eleven reference at matched or lower coverage
   - hard false-positive penalties and large-gap preservation
   - do not spend another cycle on score-only conservative anchoring layered on top of the current evidence-aware mixture heads, because it suppressed the rare positive states instead of improving coverage efficiency
   - do not spend another cycle on one-sided positive-lift gating layered on top of the current agreement-mixture family, because it broadened safe non-target fixes without recovering the actual sparse-positive frontier
   - do not spend another cycle on simple contrastive evidence-delta gating layered on top of the current evidence-aware mixture heads, because it surfaced some real signal but still only reached the weaker `90.53% -> 90.66%` band at high coverage
   - do not spend another cycle on prototype sharpness features layered on top of the current evidence-aware mixture heads, because top-match evidence already appears to capture the useful local structure and the sharpness hybrid fully collapsed on the target slice
   - do not spend another cycle on temporal drift features unless they are used only as a secondary risk calibrator under the existing `prototype_hybrid` lead
   - do not spend another cycle on multiplicative prototype gating unless it changes the prototype bank itself rather than just reweighting the same score
   - do not spend another cycle on shared-anchor lift cascades unless they can move beyond `50%` held-out stable-positive-v2 recovery without giving back the hard-slice band
   - do not spend another cycle on branch-switch routing unless the branches themselves change, because routing between the current live branches produced no Tier-1 recovery
   - do not spend another cycle on dropping the evidence-aware agreement gate directly inside the current memory-anchor blend, because it destroyed the micro-budget Tier-1 gain and still failed to reach the full `90.73%` matched band
   - do not spend another cycle on simple max-style parallel lift routing over the current memory / score-agreement / evidence-agreement paths, because it collapsed to the weak `25%` / `90.60%` pattern already covered by `prototype_hybrid`
13. Keep the hard gate for every future branch:
   - stable-positive recovery
   - false-positive deferral rate
   - hard near-tie regret / miss delta
   - large-gap control preservation
   - runtime overhead
14. Keep `detach_warmup` mandatory in every future shortlist. That contract remains unbroken by every round since it was established.
