# R19 penalty-gradient matching and clipping protocol

**Frozen on 2026-09-07 before any R19 run.** R19 is an exploratory mechanism
control in a new run family and does not alter the locked R16 recipes.

## Question

R16 selected rank-1 and diagonal coefficients separately, and the resulting
rank-1 penalty was much larger while the total gradient was clipped on almost
every later-task update.  R19 asks whether the comparison changes when the two
Fisher structures have the same penalty-gradient norm at a common,
method-independent displacement, and whether it is sensitive to clipping.

## Frozen design

- Model: SMDM-219M, all parameters trainable.
- Stream: the first two tasks of the R16 main forward order,
  `d2p_8-11 -> p2d_12-15`.
- Task A: exact R16 training (1,000 steps, batch 4, AdamW, `5e-5`, clip 1).
- Fisher: the same 40 designated Task-A training rows, native answer-only loss,
  mask seed `seed + 2101`; both rank-1 and diagonal structures are estimated
  from the identical gradients.
- Replay: the R16 balanced rule selects 64 Task-A prompts, exactly 16/fact;
  the same generated cache and frozen teacher are used within a run.
- Common displacement: starting from the post-Task-A state, run exactly 100
  Task-B GD-only updates with clip 1 and the canonical Task-B random streams.
  This probe is used only to calibrate penalty scale; reported training always
  restarts from the post-Task-A state.
- Matching: set each structure's coefficient so that the Euclidean norm of its
  raw EWC gradient at the common displacement is exactly 1.0.  Thus
  `lambda_R1 = 1 / (alpha * |u^T delta|)` and
  `lambda_D = 1 / ||D delta||`.  A zero/non-finite norm fails the run.
- Methods: GD, Rank-1+GD, and Diagonal+GD.
- Task-B clipping: canonical 1 and effectively-unclipped `1e6`.
- Seeds: 3407, 3408, and 3409.  The Rank-1+GD/Diagonal+GD pair at seed 3407,
  clip 1 is the mechanical pilot.  After it passes invariants, every remaining
  method/clip/seed cell runs unchanged regardless of numerical outcomes.

The primary endpoint is paired final average held-out DLLM loss; past-task
forgetting and final-task loss separate stability and acquisition.  Secondary
endpoints are answer-token accuracy and the recorded training penalty,
pre-clip gradient norm, and clip fraction.  We report all 18 cells, paired
seed differences, mean and SEM; three seeds do not support superiority claims.

## Audit and stopping

The runner imports the exact R16 answer-only loss, balanced replay selector,
training loop, and Fisher estimator.  It must reproduce the frozen R16 Task-A
training/Fisher anchors for each seed, verify 16/16/16/16 replay prompts,
verify both matched norms numerically, bind source/input hashes, reject an
existing output, and fail if provenance changes while running.  Its CPU
self-check compares the analytic rank-1 and diagonal penalty-gradient norms to
autograd on an explicit toy problem.

Stop only for a mechanical failure, OOM, non-finite quantity, failed invariant,
or provenance change.  Never stop or select cells based on result sign.  Logs
and failures remain under `runs/r19_penalty_match/`.  Only a complete,
independently summarized matrix can enter the appendix, explicitly labeled
exploratory and limited to one two-task factual stream.
