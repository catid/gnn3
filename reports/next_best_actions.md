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
8. Keep raw retrieval / hand-built prototype defer closed, but reopen one narrow architecture lead: a learnable `prototype_hybrid` defer head.
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
   - but `prototype_hybrid` recovered `75%` of held-out stable-positive-v2
   - at `0.75%` overall coverage it reached hard near-tie target match `90.53% -> 90.73%`
   - hard near-tie mean delta regret matched the round-eleven `2%` reference at `-0.0089`
   - overall mean delta regret at that point was still only `-0.0097`, so it is a lead, not a promotion
   - temporal scalar context was cleaner overall, but only recovered `50%` of held-out stable-positive-v2 and only reached hard near-tie `90.53% -> 90.66%`, so it is not the right architecture direction
   - gated rescaling of prototype evidence also capped out at `50%` held-out stable-positive-v2 recovery and the same weaker `90.53% -> 90.66%` hard near-tie band
   - specialist source-family banks also capped out at `50%` held-out stable-positive-v2 recovery, but only at materially higher coverage than `prototype_hybrid`
   - shallow prototype-space adapters also capped out at `50%` held-out stable-positive-v2 recovery and only reached the weaker `90.53% -> 90.66%` hard near-tie band
   - multiscale centroid branches improved broad overall caution but capped out at `25–50%` held-out stable-positive-v2 recovery and only reached `90.53% -> 90.66%` on hard near-tie
   - top-k prototype evidence readouts only recovered `25%` of held-out stable-positive-v2 and the hybrid version collapsed fully to baseline
   - score-band prototype residuals also capped out at `50%` held-out stable-positive-v2 recovery and only reached the weaker `90.53% -> 90.66%` hard near-tie band
   - explicit harmful-memory suppressor banks recovered `0%` of held-out stable-positive-v2 and effectively collapsed to baseline
   - score-aware interaction residuals improved aggregate overall regret but still capped out at `25%` held-out stable-positive-v2 recovery and only reached the weaker `90.53% -> 90.60%` hard near-tie band
9. Do not reopen conservative student retry. Round-twelve positive mining showed that source-family expansion through training-side mining is too noisy:
   - best coarse mining recovered all held-out positives only at `5.97%` precision
   - broader mining fell to `1.77%` to `2.94%` precision
   - regime mining was worse and introduced nonzero harmful selection
10. Keep the representation diagnosis unchanged. The backbone still appears to expose most of the local signals; the open problem is still precision calibration and abstention on a tiny ambiguous subset.
11. If another round opens, bias it toward **prototype-memory hybrid defer** before any broader family:
   - keep the richer teacher-bank filters from round twelve
   - use the learnable prototype-memory plus risk-branch architecture as the first architecture contender
   - compare against the round-eleven `margin_regime` reference at matched or lower coverage
   - preserve large-gap controls and broad feasible-suite behavior
12. If another round opens beyond that, bias it toward **richer teachers first, then ultra-low-coverage defer**:
   - more diverse safe teacher bank members
   - teacher agreement / correction-margin filters
   - explicit held-out comparison against the round-eleven reference at matched or lower coverage
   - hard false-positive penalties and large-gap preservation
   - do not spend another cycle on temporal drift features unless they are used only as a secondary risk calibrator under the existing `prototype_hybrid` lead
   - do not spend another cycle on multiplicative prototype gating unless it changes the prototype bank itself rather than just reweighting the same score
13. Keep the hard gate for every future branch:
   - stable-positive recovery
   - false-positive deferral rate
   - hard near-tie regret / miss delta
   - large-gap control preservation
   - runtime overhead
14. Keep `detach_warmup` mandatory in every future shortlist. That contract remains unbroken by every round since it was established.
