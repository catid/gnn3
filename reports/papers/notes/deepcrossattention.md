# DeepCrossAttention: Cross-Layer Attention for Improved Deep Model Training

- Source: https://arxiv.org/abs/2502.06785
- What is actually novel:
  Applies explicit cross-layer attention over previous hidden states rather than fixed residual accumulation.
- What is directly applicable here:
  Relevant for exploration around selective reads over prior outer rounds and role-conditioned depth mixing.
- What is risky or likely hype:
  Cross-layer retrieval adds cost and can become a crutch if the local transition dynamics are weak.
