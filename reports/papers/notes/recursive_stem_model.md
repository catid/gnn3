# Form Follows Function: Recursive Stem Model

- Source: https://arxiv.org/abs/2603.15641
- What is actually novel:
  Changes the recursive training contract: detached warm-up, terminal loss, independent H/L growth, and settling diagnostics.
- What is directly applicable here:
  This is the most directly applicable source for outer refinement loops, curricula, and reliability signals.
- What is risky or likely hype:
  Benefits depend on having a transition operator that already makes local progress; otherwise terminal-only loss can stall.
