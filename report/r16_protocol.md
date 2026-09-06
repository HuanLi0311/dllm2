# R16 native-mask DLLM protocol

**Frozen on 2026-09-05 before the confirmation and final matrices.** R14 and
R15 continual-learning outputs are invalidated in `INVALIDATED_RESULTS.md`.
They used a forced non-empty answer mask and shared the current/replay Python
random stream. R16 uses independent Bernoulli answer masks (an empty draw has
zero loss) and separate current/replay minibatch streams.

## Claim boundary

This is a controlled regularizer study on synthetic factual associations. It
does not establish broad continual-learning ability, language-skill transfer,
or a universal rank-1 Fisher structure. The experimental unit is the mean over
one optimization seed; repeated masks, templates, layers, and probes are not
independent replicates.

## Frozen selection

Facts 0--3 form the two-task validation stream. For Rank-1+GD and Diagonal+GD,
`lambda` was selected separately from `{1e3, 1e4, 1e5, 1e6, 1e7}` by the
lowest mean final average held-out DLLM loss over seeds 3407--3409. The frozen
choices are:

| Method | Selected `lambda` | Validation loss, mean +/- SEM |
|---|---:|---:|
| Rank-1+GD | `1e5` | 0.2631 +/- 0.0069 |
| Diagonal+GD | `1e3` | 0.2705 +/- 0.0192 |

The immutable validation summary is
`runs/r16_native_mask/validation_summary.json`, SHA-256
`b621509724d73059362bd095d9e67a45cdc690da68d84d0b0a37b2e33dc610e4`.
Facts 4--7 are the two-task confirmation set. They are not used for tuning.

## Shared settings

- Model: SMDM 170M checkpoint; all 219,050,496 parameters are trainable.
- Training: 1,000 steps per task, batch 4, AdamW, learning rate `5e-5`,
  gradient-norm clip 1.
- Objective: native answer-only SMDM corruption with `t ~ U(1e-3, 1)` and
  independent Bernoulli masks; empty masks contribute zero loss.
- Evaluation: 32 paired mask draws for every task/stage/method/seed.
- Fisher: 10 designated examples per fact drawn from the task's training rows.
- GD: 64 balanced retained prompts per prior task, 32 denoising steps,
  distillation weight 1. Current and replay minibatches use separate RNGs.
- Seeds: 3407, 3408, 3409; generation seed equals optimization seed.

## Main four-task matrix

The canonical stream uses facts 8--23:
`d2p_8-11`, `p2d_12-15`, `d2p_16-19`, `p2d_20-23`; reverse order is the exact
reversal. Each task has 120 optimization rows and 40 separate test rows.

The 33 runs are seven forward methods (Sequential, GD, Rank-1, Diagonal,
Rank-1+GD, Diagonal+GD, Joint) and four reverse methods (Sequential, GD,
Rank-1+GD, Diagonal+GD), each over three seeds. Reverse EWC-only and Joint runs
are omitted because they do not identify an additional claim.

## Fresh-fact replication

Because an invalid R15 pilot exposed facts 8--23 before R16, a second locked
stream uses previously unused facts 24--29:
`d2p_24-25`, `p2d_26-27`, `d2p_28-29`, in both exact orders. Each task has 60
optimization rows and 20 separate test rows. Sequential, GD, Rank-1+GD, and
Diagonal+GD are run for both orders and all three seeds (24 runs). This removes
result exposure for the core comparison, but remains the same narrow benchmark.

## Endpoints and audit

Primary performance is final average held-out DLLM loss. Loss-based past-task
forgetting is the mean, over all but the final task, of final loss minus loss
when learned. Report paired seed-level differences, mean, SEM, and wins for
Rank-1+GD versus GD and Diagonal+GD; three seeds support effect estimates, not
claims of statistical superiority. Teacher-forced answer-token accuracy,
case/whitespace-normalized exact match, target-string containment, and word
ROUGE-L are secondary diagnostics.

Locked runs must pass `--final-protocol` or `--fresh-protocol`. The strict
summarizer rejects incomplete matrices, protocol drift, mixed source,
dependency, checkpoint, tokenizer, data, validation, task-row, or replay-prompt
hashes, unbalanced replay, and non-finite outputs.

Frozen runner hashes:

- `dllm_rank1_transfer.py`: `456fd0cc12f3a2d8c2868a58e5d71002c761d2114ce7d2dd929fddea68091074`
- `dllm_rank1_multitask.py`: `53f0f1e31736a56bd7bc3fe01487c17affafc938d78b4f7527c3fae6073108d2`
