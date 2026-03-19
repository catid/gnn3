# Mamba-3: Improved Sequence Modeling using State Space Principles

- Source: https://arxiv.org/abs/2603.15569
- What is actually novel:
  Pushes Mamba forward with improved discretization, complex-valued dynamics, and MIMO updates.
- What is directly applicable here:
  Directly motivates a reusable transition operator, bidirectional scans, and multi-input fusion in the packet model.
- What is risky or likely hype:
  The full paper is sequence-centric and hardware-kernel-dependent, so a small pure-PyTorch adaptation is safer initially.
