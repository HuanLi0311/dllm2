# R24 SMDM-1.14B stored-direction trace extension

**Design frozen on 2026-09-07 before any R24 run.** R24 replaces the
mechanically invalid nominal-trace R22 family; no R22 endpoint is reused or
pooled.  It is a predeclared exploratory scale extension of R23 and cannot
establish a broad model-family claim.

## Fixed question and grid

Does the R23 ordering among no-Fisher GD, corrected-trace Rank-1+GD, and
corrected-trace Diagonal+GD recur when the same two-task stream updates every
parameter of the 1.14B SMDM checkpoint?

- Model: official SMDM-1.14B (`Diff_LLaMA_1028M`), all 1,142,367,744
  parameters trainable.
- Stream, rows, objective, optimizer, Task-A training, Fisher sampling, and
  replay are identical to R23: 1,000 steps/task, batch 4, `5e-5`, 40 Fisher
  rows, and 64 replay rows balanced 16/16/16/16.
- Matching uses the actual stored float32 direction.  Let
  `s = ||u32||_2^2` and `tau = tr(D32)`, both measured from the stored
  tensors by fixed-size chunked float64 accumulation, and `lambda_D = 1000`. Freeze
  `lambda_R = 1000 * tau / (alpha * s)`, so the implemented weighted traces
  `lambda_R * alpha * s` and `lambda_D * tau` are equal.
- Task B uses canonical clip 1.  R24 does not cross scale with the no-clip
  factor because R23 already isolates clipping at 219M.
- Methods: GD, Rank-1+GD, Diagonal+GD.  Seeds: 3407, 3408, 3409.  Total:
  nine cells.
- Mechanical subset: all three methods at seed 3407.  If and only if all
  mechanical checks pass, run the other six cells unchanged irrespective of
  endpoint direction.
- Evaluation uses 32 held-out loss-mask repeats, answer-token accuracy, and
  32-step greedy generation on both tasks.

Sequential training is omitted because it does not isolate the registered
increment from Fisher structure; GD is the required shared non-Fisher
baseline.

## Endpoints and audit

The primary endpoint is paired final average held-out loss.  Past-task
forgetting and final-task loss separate stability and acquisition.  Secondary
diagnostics are answer-token accuracy, pre-clip gradient-norm maximum, clip
fraction, and weighted EWC mean/maximum.  Report all nine cells, paired
per-seed differences, mean, and SEM; do not make a superiority or significance
claim from three seeds.

R24 uses the same corrected-trace runner and fail-closed summarizer as R23,
but a separate protocol, contract, output root, and complete grid.  The runner
must enforce the 1,142,367,744 full-parameter count, 40-row Fisher, balanced
replay, actual-direction trace equality, finite full results, exact output
identity, and frozen source/input hashes.  Within-seed Task-A state, Fisher,
replay, and matched multipliers must be identical across methods.

The R21 smoke established feasibility only and contributes no endpoint.  A
complete R24 cell is budgeted at 35--45 minutes, with a 120-minute hard
wrapper timeout and all resource/failure logs retained.  Stop only for OOM,
timeout, non-finite output, failed invariant, or provenance change; never
prune or change cells based on result sign.
