# Round 12 Positive-Mining Study

## Question

Round twelve asked whether the sparse correction family could be expanded
through training-only mining before reopening any student work.

Training-side mining manifests were constructed from seed314 using:

- strict stable-positive-v2 cases
- committee-v2 cases
- high-headroom positive union
- baseline-error positive union
- expanded union
- committee-expanded union

Held-out seeds `315 / 316` were then matched by:

- fine signature
- coarse signature
- regime signature
- coarse signature plus risk filter

## Main result

Positive mining does **not** create a clean new source family.

Fine signature matching is dead:

- held-out recall: `0.0`
- held-out precision: `0.0`

Coarse signature matching can recover the held-out positives, but only very
noisily.

Best case is `high_headroom_union` with coarse matching:

- held-out coverage: `0.77%`
- held-out stable-positive precision: `5.97%`
- held-out stable-positive recall: `100%`
- harmful selection rate: `0.0`
- selected mean teacher gain: `0.2213`

Broader coarse manifests are worse:

- precision around `1.77%` to `2.94%`
- coverage around `1.30%` to `1.57%`

Regime-signature mining is much noisier:

- recall still `50%` to `100%`
- precision only `0.26%` to `0.44%`
- harmful selection around `1.1%`

## Interpretation

Mining can recover the positives only by selecting a much larger, mostly
neutral set.

That means:

- the round-twelve source family is **recoverable by very weak coarse
  heuristics**
- but it is **not expanded into a clean trainable family**

This does not justify reopening student compression.

## Decision

Downgrade training-only positive mining as a source-family expansion tool.

What survives:

- high-headroom union coarse matching is a useful diagnostic upper bound

What does not survive:

- the thesis that mining alone creates a clean enlarged stable-positive family
- a conservative-student retry this round

## Artifacts

- `reports/plots/round12_positive_mining_manifest.csv`
- `reports/plots/round12_positive_mining_summary.csv`
- `reports/plots/round12_positive_mining_summary.png`
