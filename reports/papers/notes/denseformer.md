# DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging

- Source: https://arxiv.org/abs/2402.02622
- What is actually novel:
  Uses learned depth-weighted aggregation to improve information flow across layers.
- What is directly applicable here:
  Useful as a low-risk reference for aggregating outer-round states without fully general cross-depth attention.
- What is risky or likely hype:
  Depth averaging can blur state roles if used everywhere instead of at carefully chosen read points.
