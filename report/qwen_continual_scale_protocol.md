# Qwen3 continual-learning scale protocol

Status: frozen on 2026-09-06 before all validation-grid and formal runs.

## Claim boundary

This experiment tests whether the paper's qualitative continual-learning
findings recur across two Qwen3 scales under autoregressive training. It is not
a full-parameter replication: only the final Transformer block is trainable so
Qwen3-4B, its optimizer, a frozen GD teacher, and Fisher constraints fit on one
40GB A100. It cannot identify parameter count as an isolated causal variable.

## Models and formal matrix

- Official base checkpoints: Qwen3-0.6B and Qwen3-4B.
- Methods: Sequential, GD, Rank-1+GD, and Diagonal+GD.
- Seeds: 3407, 3408, and 3409.
- Formal forward task stream: `d2p_8-11`, `p2d_12-15`, `d2p_16-19`,
  `p2d_20-23`, matching the paper's main forward stream.
- Matrix size: 2 models x 4 methods x 3 seeds = 24 formal runs.

## Shared training and evaluation

- Train the last Transformer block for 1,000 steps/task with batch size 4,
  AdamW, learning rate `5e-5`, zero weight decay, and gradient clipping at 1.
- Use mean answer-only causal cross-entropy, including EOS. Prompts are context.
- Each task has 30 training templates and 10 held-out test templates per fact.
- Primary endpoints are final average held-out answer loss and past-task
  forgetting: final loss minus the loss immediately after each task was learned,
  averaged over all but the final task. Answer-token accuracy is secondary.

## GD and Fisher penalties

- Before each later task, freeze the current model as teacher, retain 64 prompts
  per previous task balanced over facts, greedily generate at most 32 new tokens,
  and use teacher KL on generated answer tokens with temperature and weight 1.
- Estimate Fisher from the last 10 designated training templates per fact.
- Rank-1 uses the calibrated mean-gradient outer product and fitted coefficient;
  diagonal uses the empirical squared-gradient mean. Both constrain the same
  trainable parameters and accumulate one constraint per prior task.

## Lambda selection

Select Rank-1+GD and Diagonal+GD lambda separately for each model on the disjoint
two-task stream `d2p_0-1 -> p2d_2-3`. Use seeds 3407--3409 and the grid
`{1e2, 1e3, 1e4, 1e5, 1e6}`; choose the lowest mean final average held-out loss.
If a minimum lies on a grid boundary, extend by one decade in that direction
before freezing the selected values. Formal runs must reference the immutable
selection artifact.

## Interpretation

Three seeds estimate paired effects but do not support high-powered significance
claims. Compare GD minus Sequential for forgetting mitigation, Rank-1+GD minus
GD for the incremental rank-1 claim, and Rank-1+GD minus Diagonal+GD for Fisher
structure. A conclusion is cross-scale only when its paired direction is
consistent at both 0.6B and 4B.

Frozen code SHA-256 values:

- `qwen_continual_transfer.py`:
  `c8e5d5f295319544ce55788f99cc4b0a961889ef549209ecb0b87ecb8bba7984`
- `summarize_qwen_continual_transfer.py`:
  `24db5905e78f9276f5f81c52be1e114225d36b2f4121515f1a45254cfbc8f56e`
