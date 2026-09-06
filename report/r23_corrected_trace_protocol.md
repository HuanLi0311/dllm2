# R23 stored-direction trace matching and clipping protocol

**Design frozen on 2026-09-07 before any R23 run.** R23 replaces the
mechanically invalid nominal-trace R19 family; no R19 endpoint is reused.
R23 is an exploratory mechanism control and does not alter the locked R16
recipes.

## Question and correction

R23 asks whether the ordering among no-Fisher GD, Rank-1+GD, and Diagonal+GD
changes when their Fisher penalties have the same implemented total trace,
and whether that comparison is sensitive to clipping.

The R16 Fisher estimator stores a float32 direction `u32`.  Its long-vector
float32 normalization does not guarantee that the represented vector has
exactly unit Euclidean norm.  The implemented rank-1 matrix is
`alpha * u32 u32^T`, whose trace is therefore
`alpha * ||u32||_2^2`, not merely `alpha`.  For every cell, the runner
computes `s = ||u32||_2^2` from the stored tensor using a fixed-size chunked
float64 sum, sets `tau = tr(D)`, fixes `lambda_D = 1000`, and sets

`lambda_R = 1000 * tau / (alpha * s)`.

It records `s`, both unweighted traces, both multipliers, and both resulting
weighted traces.  Non-positive or non-finite values fail closed.  This
matches implemented total stiffness, not directional stiffness or the
penalty encountered at every optimization step.

## Frozen design

- Model: SMDM-219M, all 219,050,496 parameters trainable.
- Stream: `d2p_8-11 -> p2d_12-15` using the exact R16 rows.
- Task A: 1,000 AdamW steps, batch 4, learning rate `5e-5`, clip 1.
- Fisher: the same 40 designated Task-A rows and exact R16 answer-only
  objective/masks (`seed + 2101`); rank-1 and diagonal structures use the
  same gradients.
- Replay/GD: 64 generated Task-A rows, balanced 16/16/16/16, a frozen
  teacher, 32 diffusion steps, and distillation weight 1.
- Methods: GD, Rank-1+GD, and Diagonal+GD.
- Task-B clips: canonical 1 and effectively unclipped `1e6`.
- Seeds: 3407, 3408, and 3409; 18 total cells.
- Mechanical subset: all six method/clip cells at seed 3407.  If and only if
  hashes, R16 anchors, full-parameter count, replay balance, finite outputs,
  and corrected trace equality pass, run the remaining 12 frozen cells
  unchanged, irrespective of endpoint direction.

The primary endpoint is paired final average held-out DLLM loss.  Past-task
forgetting and final-task loss separate retention and acquisition.  Secondary
diagnostics are answer-token accuracy, pre-clip gradient norm, clip fraction,
and weighted EWC mean/maximum.  Report all cells, paired seed differences,
means, and SEM.  Three seeds do not support a superiority claim.

## Audit and stopping

One shared corrected-trace runner serves R23 and R24 but rejects methods,
clips, seeds, model sizes, parameter counts, filenames, and output roots
outside the selected frozen family.  It imports the exact R16 task, loss,
training, replay, and Fisher routines; reproduces all frozen R16 Task-A
training/Fisher anchors; checks 40 Fisher rows and 16/16/16/16 replay; binds
source/input hashes; rejects existing output; recursively rejects non-finite
results; and rechecks provenance before writing.

The CPU self-check materializes a deliberately non-unit rank-1 direction,
compares the chunked norm with a direct float64 computation, and verifies
the corrected trace against explicit rank-1 and diagonal matrices.  A
fail-closed summarizer independently recomputes endpoints and matching.

Stop only for mechanical failure, OOM, timeout, non-finite output, failed
invariant, or provenance change.  Preserve every success and failure.  Only
a complete independently summarized grid may enter the appendix, labeled
exploratory and limited to one two-task factual stream.
