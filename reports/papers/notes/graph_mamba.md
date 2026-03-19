# Graph Mamba: Towards Learning on Graphs with State Space Models

- Source: https://arxiv.org/abs/2402.08678
- What is actually novel:
  Frames graph-selective SSM design around neighborhood tokenization, ordering, local encoding, and bidirectional scans.
- What is directly applicable here:
  This is the most directly relevant graph-SSM paper for the baseline packet backbone design.
- What is risky or likely hype:
  Reported gains rely on careful ordering and local encoders; naive graph-to-sequence conversions underperform.
