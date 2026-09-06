# Qwen3-1.7B continual-learning scale addendum

Status: frozen on 2026-09-06 before all Qwen3-1.7B validation and formal runs.

This addendum inserts the official Qwen3-1.7B base checkpoint between the
previously completed Qwen3-0.6B and Qwen3-4B scale points. It uses the same
data, task streams, trainable last Transformer block, methods, seeds, training
settings, endpoints, and interpretation rules specified in
`qwen_continual_scale_protocol.md`. It is a separately frozen extension rather
than a retrospective change to that protocol.

The validation matrix contains Rank-1+GD and Diagonal+GD on the disjoint
two-task stream for seeds 3407--3409 and lambdas
`{1e2, 1e3, 1e4, 1e5, 1e6}`. A boundary optimum is extended by one decade at a
time before selection is frozen. The formal matrix contains Sequential, GD,
Rank-1+GD, and Diagonal+GD on the four-task forward stream for the same three
seeds: 12 formal runs.

The combined four-size presentation must distinguish the 219M DLLM main
experiment from the last-block autoregressive Qwen3 experiments at 0.6B,
1.7B, and 4B. It must not attribute differences between 219M and Qwen3 solely
to parameter count.
