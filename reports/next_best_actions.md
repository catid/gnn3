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
   - direct bank-internal support weighting on top of that original prototype-memory geometry is now also closed
   - `prototype_support_weighted_hybrid` improved broad overall regret to `-0.0142`, but it only recovered `25%` of held-out stable-positive-v2 through `0.75%` coverage and only `50%` by `1.0–2.0%`
   - it still capped out at the weaker `90.53% -> 90.66%` hard near-tie band, so support weighting alone is not enough on the raw prototype-memory geometry
   - `prototype_memory_agree_blend_hybrid` is now the best micro-budget Tier-1 follow-up
   - at `0.25%` overall coverage it recovered `50%` of held-out stable-positive-v2 and reached the weaker but still real `90.53% -> 90.66%` hard near-tie band
   - at `0.50%` overall coverage it still held `50%` stable-positive-v2 recovery, kept the same `90.53% -> 90.66%` hard near-tie band, and beat `prototype_hybrid` at the same budget on both hard-slice regret and overall mean delta regret
   - but it capped out there and never reached the full `75%` / `90.73%` frontier band, so it is a micro-budget companion rather than a replacement lead
   - bank-internal support weighting inside that same memory-agreement geometry is now also alive
   - `prototype_support_weighted_memory_blend_hybrid` is the first bank-level retrieval change that materially improves the live prototype shortlist
   - at `0.75%` overall coverage it recovered `50%` of held-out stable-positive-v2, reached the weaker `90.53% -> 90.66%` band, and improved overall mean delta regret to `-0.0130`
   - at `1.50%` overall coverage it recovered `75%` of held-out stable-positive-v2, matched the full `90.53% -> 90.73%` hard near-tie band, and improved overall mean delta regret to `-0.0157`
   - at `2.00%` overall coverage it kept that same `75%` / `90.73%` frontier band and improved overall mean delta regret further to `-0.0162`
   - it proved that bank-internal support weighting is a real architecture direction, but it is no longer the top matched-band branch after the newer support-weighted agreement-mixture follow-up
   - direct fusion of that memory anchor with the richer evidence-aware inner agreement gate is now also closed
   - `prototype_memory_evidence_blend_hybrid` gave back the micro-budget Tier-1 win, recovering only `25%` at `0.50–1.50%` nominal budgets and only reaching `50%` by `2.0%`, while still capping out at the weaker `90.53% -> 90.66%` hard near-tie band
   - parallel dual-lift max routing on top of the same memory anchor is also closed
   - `prototype_memory_duallift_hybrid` only reproduced the weaker `25%` held-out stable-positive-v2 / `90.53% -> 90.60%` ultra-low-coverage behavior and never challenged either the micro-budget memory-agreement lead or the matched-band agreement leads
   - budget-conditioned calibration on top of the same memory-anchor geometry is also closed
   - `prototype_budget_memory_hybrid` recovered `0%` of held-out stable-positive-v2 at every budget and only spent coverage on broad-safe non-target states, so splitting the outer gate by budget did not recover any real frontier signal
   - direct teacher-gain supervision on top of the same memory-anchor geometry is also closed
   - `prototype_teacher_margin_memory_hybrid` improved broad-safe overall regret to `-0.0042`, but recovered `0%` of held-out stable-positive-v2 at every budget and left hard near-tie unchanged at `90.53%`, so teacher-gain regression calibrated non-target helpfulness instead of the rare correction family
   - explicit regime-split lift specialists on top of the same memory-anchor geometry are also closed
   - `prototype_regime_split_memory_hybrid` did recover `25%` of held-out stable-positive-v2 at `0.10%` overall coverage, but it still capped out at the weaker `90.53% -> 90.60%` hard near-tie band and never challenged `prototype_memory_agree_blend_hybrid` on the micro-budget frontier
   - explicit risk-prior regime mixing on top of that same memory-anchor geometry is also closed
   - `prototype_risk_prior_regime_memory_hybrid` improved broad-safe overall regret to `-0.0045`, but recovered `0%` of held-out stable-positive-v2 at every budget and left hard near-tie unchanged at `90.53%`, so injecting explicit headroom / residual priors made the branch broader instead of sharper
   - explicit risk-veto regime mixing on top of that same memory-anchor geometry is also closed
   - `prototype_risk_veto_regime_memory_hybrid` also recovered `0%` of held-out stable-positive-v2 at every budget and left hard near-tie unchanged at `90.53%`; it only found a tiny broad-safe control fix at `0.03%` overall coverage, so suppressive vetoing collapsed the already-weak regime signal back toward baseline
   - direct memory-context injection into the current evidence-agreement gate is also closed
   - `prototype_memory_calibrated_evidence_hybrid` recovered `0%` of held-out stable-positive-v2 at every budget, left hard near-tie unchanged at `90.53%`, and only redirected about `0.24%` overall coverage into broad-safe non-target states, so memory evidence did not improve ranking inside the live evidence-agreement family
   - auxiliary teacher-signal prediction inside the current evidence-agreement family is also closed
   - `prototype_teacher_signal_evidence` and `prototype_teacher_signal_evidence_hybrid` both recovered `0%` of held-out stable-positive-v2 at every budget, left hard near-tie unchanged at `90.53%`, and only surfaced tiny broad-safe control fixes, so committee/gain auxiliary heads did not sharpen the real frontier
   - explicit budget-conditioning inside the current evidence-agreement family is also closed
   - `prototype_budget_evidence_agree` and `prototype_budget_evidence_agree_hybrid` both recovered `0%` of held-out stable-positive-v2 at every budget and left hard near-tie unchanged at `90.53%`; the plain model only spent more coverage on non-target states and the hybrid collapsed to a tiny large-gap control fix
   - bank-internal support weighting inside that same evidence-agreement family is also now closed
   - `prototype_support_weighted_evidence_agree_hybrid` did eventually recover `75%` of held-out stable-positive-v2 and the full `90.53% -> 90.73%` hard near-tie band, but only at `1.50–2.00%` overall coverage and with much weaker aggregate regret (`-0.0097` to `-0.0104`) than both the older `prototype_evidence_agree_hybrid` and the newer `prototype_support_weighted_agree_mix_hybrid`
   - so support weighting is not a universal win: it helps the score-only agreement and memory-agreement geometries, but it degrades the evidence-agreement family
   - `prototype_mixture_hybrid` was the first follow-up that fully caught that hard-slice band at matched higher coverage
   - `prototype_agree_mix_hybrid` now improves on it in coverage efficiency
   - at `1.5%` nominal budget it matched `75%` held-out stable-positive-v2 recovery and the same `90.53% -> 90.73%` hard near-tie band
   - it reached that matched band at about `1.05%` overall coverage instead of `1.84%`
   - its overall mean delta regret at that point was `-0.0137`, essentially matching the old mixture result while paying much less coverage
   - but at `0.75%` it only recovered `50%` of held-out stable-positive-v2 and stayed in the weaker `90.53% -> 90.66%` band
   - bank-internal support weighting inside that same agreement-mixture geometry is now the strongest matched-band result overall
   - `prototype_support_weighted_agree_mix_hybrid @ 1.00%` already reaches `75%` held-out stable-positive-v2 recovery and the full `90.53% -> 90.73%` hard near-tie band at about `1.01%` overall coverage, with overall mean delta regret `-0.0148`
   - `prototype_support_weighted_agree_mix_hybrid @ 1.50%` keeps that same `75%` / `90.73%` frontier band at `1.52%` overall coverage and improves overall mean delta regret to `-0.0158`
   - `prototype_support_weighted_agree_mix_hybrid @ 2.00%` improves overall mean delta regret further to `-0.0165`
   - that means it now supersedes `prototype_agree_mix_hybrid` as the coverage-efficient matched-band leader, and also edges past `prototype_support_weighted_memory_blend_hybrid` and `prototype_evidence_agree_hybrid` on aggregate matched-band quality
   - soft tail suppression over that same support-weighted agreement-mixture bank is now a real positive follow-up
   - the plain soft-tail head is inert, but `prototype_soft_tail_support_agree_mix_hybrid @ 0.75%` reaches `75%` held-out stable-positive-v2 recovery and the full `90.53% -> 90.73%` hard near-tie band at `0.76%` overall coverage
   - at that same point it slightly improves on `prototype_hybrid` overall mean delta regret (`-0.0104` vs `-0.0097`) while preserving clean large-gap controls
   - that makes it the best sub-`1%` full-band architecture lead, while `prototype_support_weighted_agree_mix_hybrid` still remains the best matched-band branch overall once coverage can rise to `~1%`
   - negative-tail-only cleanup over that same support-weighted agreement-mixture bank is now also a real positive follow-up
   - the plain negative-tail head is inert, but `prototype_negative_tail_support_agree_mix_hybrid @ 1.00%` reaches `100%` held-out stable-positive-v2 recovery, improves hard near-tie target match `90.53% -> 90.80%`, and improves hard near-tie mean delta regret to `-0.0100`
   - its overall mean delta regret at that point is `-0.0131`, so it still trails `prototype_support_weighted_agree_mix_hybrid` on aggregate matched-band quality, but it is now the strongest high-recall branch around `1%` coverage
   - sharpness-gated negative-tail cleanup over that same support-weighted agreement-mixture bank is now also a real positive follow-up
   - the plain sharp-negative-tail head only surfaced a weak `50%` held-out stable-positive-v2 niche and is closed
   - but `prototype_sharp_negative_tail_support_agree_mix_hybrid @ 0.75%` reaches the full `75%` / `90.73%` frontier band at `0.76%` overall coverage with overall mean delta regret `-0.0144`
   - at `1.00%` overall coverage it keeps that same full frontier band and improves overall mean delta regret to `-0.0152`
   - that cleanly supersedes the older soft-tail branch below `1%` coverage and slightly edges the older support-weighted agreement-mixture branch at about `1%` coverage, while still trailing the pure negative-tail branch on maximum held-out recall
   - shared-branch-only sharp negative cleanup with fixed dual negative cleanup is now also closed
   - the plain shared-sharp branch only recovered `50%` of held-out stable-positive-v2 and only reached the weaker `90.53% -> 90.66%` hard-slice band
   - the hybrid fully collapsed on the target slice, recovering `0%` of held-out stable-positive-v2 through the whole budget range and leaving hard near-tie unchanged at `90.53%`
   - so the sharp-negative-tail gain is not explained by shared-only cleanup while the dual branch stays fixed
   - dual-branch-only sharp negative cleanup with fixed shared negative cleanup is now also closed
   - the plain dual-sharp branch only found a tiny `25%` held-out stable-positive-v2 niche at `0.10%` coverage and only reached the weaker `90.53% -> 90.60%` band
   - the hybrid only improved that weak niche to about `0.18%` overall coverage with the same `25%` held-out recovery and the same weaker `90.53% -> 90.60%` hard-slice band
   - so the sharp-negative-tail gain also is not explained by moving the adaptive cleanup onto only the dual branch while the shared branch stays fixed
   - fixed-floor plus sharpness-gated negative cleanup on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain floor-plus-sharp head collapsed fully to baseline
   - the hybrid recovered `0%` of held-out stable-positive-v2 through `1.0%` coverage and only `25%` by `1.5–2.0%`, while hard near-tie stayed unchanged until the weak `90.53% -> 90.60%` band at high coverage
   - so stacking the fixed negative-tail floor and the sharpness-gated extra cleanup simply over-suppressed the bank and destroyed both live negative-cleanup lanes
   - floor-gated sharp negative cleanup on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain floor-gated sharp head was mostly inert until high coverage, where it only reached a weak `25–50%` held-out stable-positive-v2 niche and only the weaker `90.53% -> 90.60%` hard-slice band
   - the hybrid recovered only `25%` of held-out stable-positive-v2 through the full budget range and also stayed capped at the weaker `90.53% -> 90.60%` hard-slice band, even when overall mean delta regret improved to `-0.0090`
   - so the sharp-negative-tail gain appears to require letting the adaptive negative cleanup truly turn down, rather than forcing a nonzero cleanup floor on every state
   - learned blending between the fixed negative-tail and sharp negative-tail cleanup paths on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain cleanup-blend head stayed mostly inert until `2.0%` overall coverage, where it only recovered `50%` of held-out stable-positive-v2 and still only reached the weaker `90.53% -> 90.60%` hard-slice band
   - the hybrid saturated immediately into a small `25%` held-out stable-positive-v2 / `90.53% -> 90.60%` niche, with best overall mean delta regret only `-0.0048`
   - so a learned per-state interpolation between the live fixed and sharp negative-cleanup paths also collapsed toward the old weak middle instead of preserving either live lane
   - max-style union between the fixed negative-tail and sharp negative-tail cleanup paths on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain cleanup-max head stayed fully dead, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid did recover the full `75%` / `90.73%` frontier band, but only at `1.5–2.0%` overall coverage and with weaker overall mean delta regret (`-0.0118` to `-0.0132`) than the existing sharp-negative and support-weighted matched-band leads
   - it also gave back the fixed negative-tail branch's `100%` held-out sparse-positive recall lane, so hard union is real signal but still clearly dominated by the current live shortlist
   - sharp-base positive-only lift from the fixed negative-tail branch on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain cleanup-lift head stayed fully dead, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid only recovered `25%` of held-out stable-positive-v2 by `0.75–1.0%` coverage and only `50%` by `1.5–2.0%`, while still capping out at the weaker `90.53% -> 90.66%` hard-slice band
   - so using the sharp-negative branch as the base and adding only a gated positive fixed-branch lift still collapses back into the old weak middle rather than preserving either live negative-cleanup lane
   - top-2-preserving negative-tail cleanup on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain top2-negative head was effectively dead, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid only recovered `25%` of held-out stable-positive-v2 through `0.50–1.50%` coverage and only `50%` by `2.0%`, while still capping out at the weaker `90.53% -> 90.66%` hard-slice band
   - so preserving the runner-up negative match and only suppressing the broader tail also weakens the live retrieval cleanup too much to preserve either the sharp-quality lane or the fixed high-recall lane
   - full-tail-mass-gated negative cleanup on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain mass-aware head was effectively dead, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid recovered `25%` of held-out stable-positive-v2 at `0.10–0.25%` overall coverage and `50%` by `0.50–2.0%`, while still capping out at the weaker `90.53% -> 90.66%` hard-slice band
   - so using full negative tail mass is enough to recreate the old `50%` / `90.66%` middle lane inside the stronger agreement-mixture family, but it still does not preserve either live negative-cleanup frontier
   - sharp-plus-mass negative-tail cleanup on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain sharp-mass head was fully dead, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid improved on the pure mass-aware gate and eventually recovered the full `75%` / `90.73%` frontier at `2.0%`, but it still only recovered `50%` through `0.75–1.5%` coverage and lost to the live sharp-negative branch below `1%`
   - so smooth OR-combining the sharpness and tail-mass gates still does not preserve either the sharp branch's low-coverage quality lane or the fixed branch's high-recall lane
   - branch-calibrated sharp negative cleanup on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain branch-calibrated sharp head was effectively dead, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid eventually recovered the full `75%` / `90.73%` frontier by `1.5%`, but it still only recovered `25%` at `0.75%` and `50%` at `1.0%`, so it lost badly to the live sharp-negative branch below `1%`
   - it also still trailed the older support-weighted agreement-mixture branch on aggregate regret once it reached the higher-budget matched band, so simple branch-specific sharpness calibration is not enough
   - learned summary-gated negative cleanup on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain learned-gate head was effectively dead, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid also only recovered `25%` at `0.75%`, `50%` at `1.0%`, and only reached the full `75%` / `90.73%` frontier by about `1.5%`, so it again lost badly to the live sharp-negative branch below `1%`
   - it also still trailed the older support-weighted agreement-mixture branch once it reached the higher-budget matched band, so simply learning the gate over lead / top-gap / tail-mass summaries is not enough
   - branch-specific negative-tail cleanup amplitude on top of that same support-weighted agreement-mixture bank is now also closed
   - the plain branch-strength head was effectively dead and slightly harmful, recovering `0%` of held-out stable-positive-v2 and regressing overall target match slightly
   - the hybrid is the strongest of the recent post-sharp branch-specific follow-ups: it recovered `50%` at `0.75–1.0%` overall coverage and reached the full `75%` / `90.73%` frontier by about `1.17%` coverage with overall mean delta regret `-0.0154`
   - that is better than the older branch-calibrated and learned-gate follow-ups, so separate branch cleanup amplitude is more useful than separate branch gate shape
   - but it still loses to the live sharp-negative branch below `1%` coverage and still does not clearly replace the older support-weighted agreement-mixture reference once coverage can rise into the higher-budget matched band, so it is not a new live lane
   - fixed-branch positive lift on top of that newer branch-strength sharp cleanup is now also closed
   - the plain branch-strength lift head was effectively dead, recovering `0%` of held-out stable-positive-v2 and only finding a tiny broad-safe control fix at `1.5–2.0%`
   - the hybrid only recovered `25%` of held-out stable-positive-v2 through `1.0%` overall coverage and only reached the full `75%` / `90.73%` frontier once coverage rose to about `1.5–2.0%`
   - at `2.0%` overall coverage it improved overall mean delta regret to `-0.0162`, but that still trails the older `prototype_support_weighted_agree_mix_hybrid` reference slightly and completely fails to preserve the fixed negative-tail branch's `100%` recall lane around `1%`
   - so fixed-cleanup lift remains the wrong fusion mechanism for this family even after improving the sharp base with branch-specific cleanup amplitude
   - max-style union on top of that newer branch-strength sharp cleanup is now also closed
   - the plain branch-strength max head collapsed fully to baseline, recovering `0%` of held-out stable-positive-v2 and leaving hard near-tie unchanged across the full budget range
   - the hybrid is better than the failed branch-strength lift: by `1.0%` overall coverage it recovered `50%` of held-out stable-positive-v2 and reached the weaker `90.53% -> 90.66%` hard-slice band
   - but it still loses to the branch-strength sharp base at matched coverage, still does not preserve the fixed negative-tail branch's `100%` / `90.80%` high-recall lane, and only reaches the full `75%` / `90.73%` frontier at `2.0%` overall coverage with overall mean delta regret `-0.0158`
   - so even with the improved branch-strength sharp base, hard max fusion remains a dominated compromise rather than a new live lane
   - branchwise max fusion on top of that newer branch-strength sharp cleanup is now alive
   - the plain branchwise-max head still collapsed fully to baseline, so the useful signal still depends on the hybrid risk path
   - but the hybrid is the first successful fusion of the improved sharp branch and the fixed negative-tail branch
   - at `1.00%` overall coverage it already reaches the full `75%` / `90.73%` frontier with overall mean delta regret `-0.0145`, only slightly behind the live sharp-negative branch at the same point
   - at `1.50%` overall coverage it keeps that same `75%` / `90.73%` frontier and edges the older `prototype_support_weighted_agree_mix_hybrid` higher-budget reference on aggregate regret (`-0.0159` vs `-0.0158`)
   - at `2.00%` overall coverage it becomes the new higher-budget max-recall lead, reaching `100%` held-out stable-positive-v2 recovery, `90.53% -> 90.80%` on hard near-tie, and overall mean delta regret `-0.0167`
   - so the live interpretation is now split by budget: keep the older sharp-negative and fixed negative-tail lanes at lower coverage, and use branchwise max as the higher-budget matched-band and higher-budget max-recall leader
   - branchwise positive-only lift on top of that same branch-strength sharp base is now also closed
   - the plain branchwise-lift head is dead-to-harmful: it recovers `0%` of held-out stable-positive-v2 and starts degrading hard near-tie once it becomes active above `0.5%`
   - the hybrid is a real improvement over the older global lift, confirming that fusion does belong inside the shared and dual branches:
     - by `1.0%` overall coverage it recovers `50%` of held-out stable-positive-v2, reaches `90.53% -> 90.66%` on hard near-tie, and improves overall mean delta regret to `-0.0149`
   - but it still loses cleanly to the live sharp-negative lane below `1%`, only reaches the full `75%` / `90.73%` frontier once coverage rises to `1.5%`, and even there it trails both the older `prototype_support_weighted_agree_mix_hybrid` higher-budget reference and the newer branchwise-max result
   - so the structural insight from the positive branchwise-max result is not just “branch-local fusion”; it is that this family still wants a hard branchwise union rather than a softer learned lift
   - branchwise margin-max fusion on that same branch-strength sharp base is now also closed
   - the plain branchwise-margin head stayed inert through `1.5%` coverage and only found a tiny `25%` held-out stable-positive-v2 niche at `2.0%`
   - the hybrid does improve materially over the closed branchwise-lift follow-up:
     - at `1.0%` overall coverage it recovers the full `75%` / `90.73%` frontier with overall mean delta regret `-0.0152`
   - but it still loses to the live sharp-negative branch below `1%`, still never preserves the branchwise-max branch's `100%` / `90.80%` higher-budget recall lane, and by `1.5%` it still trails both the older `prototype_support_weighted_agree_mix_hybrid` higher-budget reference and the newer branchwise-max result
   - so adding a learned branch margin is better than replacing branchwise max with a soft learned lift, but hard branchwise max still remains the best live fusion for this family
   - strict joint-support branchwise fusion on that same branch-strength sharp base is now also closed
   - the plain joint-support head is inert through `1.5%` coverage and only wakes up slightly at `2.0%`, where it still recovers only `25%` of held-out stable-positive-v2
   - the hybrid finds only the tiny ultra-low-coverage niche:
     - by `0.1%` coverage it recovers `25%` of held-out stable-positive-v2 and reaches the weaker `90.53% -> 90.60%` hard near-tie band
     - then it saturates completely, topping out at only `0.68%` overall coverage and never improving beyond that same `25%` / `90.60%` niche
   - so the live branchwise-max gain is not just “avoid one-branch takeovers”; requiring both branches to share the same fixed-cleanup gain is too conservative and throws away almost all of the useful recall signal
   - asymmetric positive-plus-negative tail cleanup over that same support-weighted agreement-mixture bank is now also closed
   - the plain asymmetric-tail head is inert, and `prototype_asymmetric_tail_support_agree_mix_hybrid @ 0.75%` only matches the full `75%` / `90.73%` frontier band with weaker overall mean delta regret (`-0.0085`) than the existing soft-tail branch
   - at `1.00%` overall coverage it still only recovers `75%` of held-out stable-positive-v2 and reaches overall mean delta regret `-0.0105`, so it also trails both the original support-weighted branch on aggregate quality and the negative-tail branch on held-out recall
   - so adding a smaller positive-tail penalty on top of the successful negative-tail cleanup just pulls the branch back toward the middle instead of improving either live lane
   - learned interpolation between the live support-weighted score and the new soft-tail score is now also closed
   - `prototype_soft_tail_blend_support_agree_mix_hybrid` only recovered `50%` of held-out stable-positive-v2 through `0.25–2.0%` coverage and stayed capped at the weaker `90.53% -> 90.66%` hard-slice band
   - so the soft-tail gain appears to require committing to that retrieval view directly, not blending it back toward the older score
   - explicit per-state risk-conditioned support modulation on top of that same support-weighted agreement-mixture head is now also closed
   - both `prototype_risk_support_agree_mix` variants recovered `0%` of held-out stable-positive-v2 at every budget, left hard near-tie unchanged at `90.53%`, and produced `0.0000` overall mean delta regret
   - so the live gain appears to come from static bank cleanup, not from tiny state-conditional support perturbations layered on top
   - split shared/dual positive-vs-negative temperatures on top of that same support-weighted agreement-mixture head are now also closed
   - the plain split-scale head recovered `0%` of held-out stable-positive-v2 at every budget
   - the hybrid only recovered `25%` at `1.5–2.0%` overall coverage, still stayed in the weaker `90.53% -> 90.60%` hard near-tie band, and only reached `-0.0021` overall mean delta regret
   - so the live gain does not come from giving the current support-weighted banks extra temperature freedom either
   - hard top-k pooling on top of that same support-weighted agreement-mixture head is now also closed
   - the plain top-k head only recovered `50%` of held-out stable-positive-v2 at `2.0%` overall coverage, still capped out at the weaker `90.53% -> 90.66%` hard near-tie band, and only reached `-0.0030` overall mean delta regret
   - the hybrid top-k branch recovered `0%` of held-out stable-positive-v2 at every budget and collapsed to a tiny large-gap control fix
   - so the live gain depends on soft full-bank pooling over the cleaned bank, not on hard tail truncation
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
   - use the sharp-negative-tail support-weighted agreement-mixture hybrid as the primary sub-`1%` full-band contender
   - use the negative-tail support-weighted agreement-mixture hybrid as the high-recall contender around `1%` coverage
   - use the branchwise-max negative-cleanup support-weighted agreement-mixture hybrid as the higher-budget matched-band and high-recall contender once coverage can rise to roughly `1.5–2.0%`
   - keep the original learnable prototype-memory plus risk-branch architecture as the lighter low-coverage reference behind it
   - use the memory-agreement blend hybrid as the micro-budget contender below roughly `0.5%` overall coverage
   - use the sharp-negative-tail support-weighted agreement-mixture head as the primary coverage-efficient matched-band contender around `0.75–1.0%`
   - keep the original support-weighted agreement-mixture head only as the older higher-budget matched-band reference behind the newer branchwise-max result
   - keep the older agreement-gated geometry-mixture head only as the lighter pre-support-weighting reference
   - keep the support-weighted memory-agreement blend hybrid only as a supporting bank-weighting reference behind the newer agreement-mixture result
   - keep the evidence-calibrated agreement-mixture head only as an older aggregate-quality reference behind the newer support-weighted agreement-mixture branch
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
   - do not spend another cycle on budget-conditioned outer calibration over the current memory-anchor geometry, because it recovered `0%` of held-out stable-positive-v2 and just redirected coverage toward broad-safe non-target states
   - do not spend another cycle on direct teacher-gain regression over the current memory-anchor geometry, because it also recovered `0%` of held-out stable-positive-v2 and only improved broad-safe non-target states
   - do not spend another cycle on regime-split outer lift specialists over the current memory-anchor geometry, because they surfaced only a weak `25%` held-out stable-positive niche and still failed to move beyond the weaker `90.53% -> 90.60%` band
   - do not spend another cycle on explicit risk-prior regime mixing over the current memory-anchor geometry, because it broadened residual-style non-target selections and still recovered `0%` of held-out stable-positive-v2
   - do not spend another cycle on explicit risk-veto regime mixing over the current memory-anchor geometry, because it over-suppressed the already-weak specialist signal, recovered `0%` of held-out stable-positive-v2, and collapsed back toward baseline plus a single harmless control fix
   - do not spend another cycle on feeding memory score and memory top-match context directly into the current evidence-agreement gate, because it recovered `0%` of held-out stable-positive-v2 and only redirected coverage into broad-safe non-target states
   - do not spend another cycle on auxiliary committee-support / safe-gain heads inside the current evidence-agreement family, because they also recovered `0%` of held-out stable-positive-v2 and only improved broad-safe control behavior
   - do not spend another cycle on explicit budget-conditioning inside the current evidence-agreement family, because it also recovered `0%` of held-out stable-positive-v2 and only changed broad-safe non-target selection patterns
   - do not spend another cycle on bank-internal support weighting inside the current evidence-agreement family, because it only recovered the full `75%` / `90.73%` band at `1.50–2.00%` coverage while materially degrading aggregate regret relative to the older evidence-agreement baseline and the live support-weighted agreement-mixture lead
   - do not spend another cycle on per-state prototype selection inside the current evidence-agreement family, because it also recovered `0%` of held-out stable-positive-v2, never moved hard near-tie, and only surfaced another tiny large-gap control fix in the hybrid variant
   - do not spend another cycle on regime-specialized bank families inside the current evidence-agreement family, because they only recovered `25%` of held-out stable-positive-v2 by spending very broad false-positive coverage across the headroom and baseline-error slices and still only reached the weaker `90.53% -> 90.60%` hard-slice band
   - do not spend another cycle on anchored residual regime specialists inside the current evidence-agreement family, because they still only recovered `25%` of held-out stable-positive-v2 and also stayed capped at the weaker `90.53% -> 90.60%` hard-slice band
   - do not spend another cycle on explicit positive-support gating over those anchored residual regime specialists, because it preserved the same weak `25%` held-out stable-positive-v2 recovery and slightly improved coverage-efficiency, but still spent about `75%` false-positive rate inside the targeted regime slices and still stayed capped at the weaker `90.53% -> 90.60%` hard-slice band
   - do spend future cycles on bank-internal support weighting before adding more outer gates only inside the stronger agreement/anchor geometries, because support-weighted retrieval materially improved the matched-band frontier in the memory-agreement and agreement-mixture families
   - do spend future cycles on soft bank-tail cleanup before hard truncation inside the support-weighted agreement-mixture family, because soft tail suppression preserved the full frontier band below `1%` coverage while hard top-k truncation killed it
   - do spend future cycles on targeted negative-bank cleanup inside the support-weighted agreement-mixture family when the goal is maximizing held-out sparse-positive recall around `1%` coverage, because the negative-tail branch reached `100%` held-out recovery and the strongest hard-slice band there
   - do spend future cycles on internal-bank sharpness-gated negative cleanup inside the support-weighted agreement-mixture family when the goal is improving aggregate quality around `0.75–1.0%` coverage, because the sharp-negative-tail branch preserved the full `75%` / `90.73%` frontier band and improved overall mean delta regret to `-0.0152`
   - do not spend another cycle on moving sharpness-gated negative cleanup onto only the shared branch while leaving the dual branch on fixed negative cleanup, because the plain branch stayed weak and the hybrid collapsed to `0%` held-out recovery across the full budget range
   - do not spend another cycle on moving sharpness-gated negative cleanup onto only the dual branch while leaving the shared branch on fixed negative cleanup, because both variants stayed trapped in the tiny `25%` / `90.60%` niche and never recovered either live negative-cleanup lane
   - do not spend another cycle on stacking a fixed negative-tail floor with extra sharpness-gated cleanup on top of the current support-weighted agreement-mixture head, because it collapsed to `0%` held-out recovery through `1.0%` coverage and only recovered a weak `25%` / `90.60%` niche at high coverage
   - do not spend another cycle on forcing a nonzero cleanup floor inside the current sharp-negative-tail gate, because the floor-gated variant only recovered `25%` held-out stable-positive-v2 across the full budget range and stayed trapped in the weaker `90.53% -> 90.60%` hard-slice band
   - do not spend another cycle on learned blending between the current fixed negative-tail and sharp negative-tail cleanup paths, because the cleanup-blend variant also collapsed to the weak `25%` / `90.60%` hybrid niche and never preserved either the sharp branch's `75%` / `90.73%` frontier or the fixed branch's `100%` / `90.80%` recall lane
   - do not spend another cycle on max-style union between the current fixed negative-tail and sharp negative-tail cleanup paths, because the cleanup-max variant only recovered the full `75%` / `90.73%` band at `1.5–2.0%` coverage, still lost the fixed branch's `100%` recall lane, and remained clearly weaker than the existing matched-band leads
   - do not spend another cycle on sharp-base positive-only lift from the current fixed negative-tail branch, because the cleanup-lift variant also only reached the weak `25–50%` held-out stable-positive-v2 / `90.53% -> 90.60%` band and never preserved either live negative-cleanup lane
   - do not spend another cycle on preserving the top-2 negative matches while only suppressing the broader negative tail, because the top2-negative variant also only reached the weak `25–50%` held-out stable-positive-v2 / `90.53% -> 90.60%` band and never preserved either live negative-cleanup lane
   - do not spend another cycle on replacing the sharp negative-tail gate with a pure total-tail-mass gate, because the mass-aware variant only recreated the weaker `50%` / `90.53% -> 90.66%` middle lane and still failed to preserve either the sharp branch's `75%` / `90.73%` frontier or the fixed branch's `100%` / `90.80%` recall lane
   - do not spend another cycle on smooth OR-combining the current sharpness gate and total-tail-mass gate inside the negative-tail cleanup path, because the sharp-mass variant only recovered the full `75%` / `90.73%` frontier once coverage rose toward `2%` and still lost to the live sharp-negative branch below `1%`
   - do not spend another cycle on simply learning separate shared and dual sharpness-gate centers and slopes inside the negative-tail cleanup path, because the branch-calibrated sharp variant still only recovered `25%` at `0.75%`, only `50%` at `1.0%`, and remained slightly weaker than the older support-weighted agreement-mixture reference once it finally reached the full `75%` / `90.73%` matched band
   - do not spend another cycle on simply learning an internal negative-tail gate from lead / top-gap / tail-mass summaries, because the learned-gate variant still only recovered `25%` at `0.75%`, only `50%` at `1.0%`, and remained slightly weaker than the older support-weighted agreement-mixture reference once it finally reached the full `75%` / `90.73%` matched band
   - do not spend another cycle on asymmetric positive-plus-negative tail cleanup on top of the current support-weighted agreement-mixture head, because it preserved the `75%` / `90.73%` band but lost to the soft-tail branch below `1%`, lost to the original support-weighted branch above `1%`, and gave back the negative-tail branch's `100%` held-out recall
   - do not spend another cycle on learned interpolation between the live support-weighted score and the new soft-tail score, because it collapsed back to the weaker `50%` / `90.66%` band across the whole budget range
   - do not spend another cycle on tiny risk-conditioned support modulation on top of the current support-weighted agreement-mixture head, because both variants recovered `0%` of held-out stable-positive-v2 and left the accepted frontier unchanged
   - do not spend another cycle on splitting positive and negative bank temperatures inside the current support-weighted agreement-mixture head, because the hybrid only recovered `25%` of held-out stable-positive-v2 at `1.5–2.0%` overall coverage and still stayed capped at the weaker `90.53% -> 90.60%` hard-slice band
   - do not spend another cycle on hard top-k pooling inside the current support-weighted agreement-mixture head, because the plain branch only reached `50%` held-out stable-positive-v2 recovery at `2.0%` overall coverage and the hybrid collapsed to `0%` recovery
   - do not spend another cycle on applying that same support weighting directly to the raw prototype-memory geometry, because it improved broad regret but still only reached `50%` held-out stable-positive-v2 recovery and the weaker `90.53% -> 90.66%` hard-slice band
13. Keep the hard gate for every future branch:
   - stable-positive recovery
   - false-positive deferral rate
   - hard near-tie regret / miss delta
   - large-gap control preservation
   - runtime overhead
14. Keep `detach_warmup` mandatory in every future shortlist. That contract remains unbroken by every round since it was established.

## Round 13 Update

15. Narrow the active prototype promotion surface to the one branch that survived the rerun robustness gate:
   - promote `prototype_branchwise_max_negative_cleanup_support_agree_mix_hybrid` as the only robust prototype correction reference
   - demote `prototype_memory_agree_blend_hybrid` from the active frontier pack
   - demote `prototype_sharp_negative_tail_support_agree_mix_hybrid` and `prototype_negative_tail_support_agree_mix_hybrid` from deployment-facing promotion; keep them only as mechanism probes
16. Treat the archived frontier as descriptive, not promotive:
   - archived leaders still split by budget band
   - rerun 1 and rerun 2 collapse that ladder back to branchwise-max at every matched budget
   - do not accept archived-only operating regions without a rerun confirmation
17. Keep the accepted mechanism simple:
   - support-weighted agreement-mixture banks
   - shared plus dual branch scoring
   - branch-local negative cleanup
   - hard branchwise max before the outer mix
18. Do not promote a hierarchical dispatcher:
   - the static budget ladder was only an archived convenience view
   - the score-band dispatcher lost to the best single branch on the archived frontier and lost clearly on both fresh reruns
19. Do not reopen conservative student retry:
   - `stable_positive_v3_total` stayed at `4`
   - `new_v3_total` stayed at `0`
   - there are `31` unstable positives, but they are not stable enough to support compression
20. If another round opens, bias it toward branchwise-max stabilization only:
   - prototype pruning and deduplication inside the branchwise-max family
     - but not via soft suppression-only keep masks, because the pruned
       branchwise-max follow-up kept bank means around `0.98` and a same-config
       rerun collapsed from a brief `83.3%` / `90.60%` point back to the weaker
       `66.7%` / `90.53%` lane
     - and not via static support-ranked cosine-threshold hard deduplication,
       because the hard-dedup follow-up really cut the negative banks from `8`
       to about `5` prototypes but still gave back the accepted `1.0%` and
       `2.0%` frontier positions
     - if pruning reopens again, require offline bank reconstruction,
       teacher-guided bank rebuild, or hard-negative-conditioned bank editing
   - cleanup-threshold tuning against the `71` useful hard negatives from round 13
     - but not via free branch/path-specific tail-margin deltas, because the
       tail-margin-calibrated branchwise-max follow-up left the hybrid margins
       effectively pinned at `0.498–0.500` and still gave back the accepted
       `1.0%` and `2.0%` frontier positions
     - and not via offline searched branch/path-specific fixed margins either,
       because the searched fixed-margin follow-up chose aggressive `0.2`
       shared margins on train and then collapsed to only `33.3%` held-out
       stable-positive-v2 recovery with much weaker overall regret
     - if threshold tuning reopens, require rerun-stable outer validation and
       hard-negative-conditioned calibration beyond four scalar margins
   - rerun robustness first, before any new composite or student work
