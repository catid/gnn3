# Graph-Mamba: Long-Range Graph Sequence Modeling with Selective State Spaces

- Source: https://arxiv.org/abs/2402.00789
- What is actually novel:
  Adapts selective state spaces to ordered graph sequences to improve long-range modeling.
- What is directly applicable here:
  Relevant for graph ordering choices and for deciding how much sequential bias to inject before local mixing.
- What is risky or likely hype:
  Sequence-first graph reductions can lose locality unless the ordering is strongly aligned with the routing task.
