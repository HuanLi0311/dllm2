# R19 weighted-trace matching and clipping protocol

**Frozen on 2026-09-07 before any R19 run.** R19 is an exploratory mechanism
control in a new run family and does not alter the locked R16 recipes.

## Question

R16 selected rank-1 and diagonal coefficients separately, and the resulting
rank-1 penalty was much larger while the total gradient was clipped on almost
every later-task update.  R19 asks whether the comparison changes when the two
Fisher structures have the same total weighted Fisher trace, and whether it is
sensitive to clipping.

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
- Matching: anchor the scale to the independently selected R16 diagonal recipe
  `lambda_D = 1e3`.  For each paired seed let `tau = tr(D)` and let `alpha` be
  the normalized rank-1 coefficient, so `tr(alpha uu^T) = alpha`.  Freeze the
  target `kappa = 1e3 * tau` and set
  `lambda_R1 = kappa / alpha`, giving
  `lambda_R1 * alpha = lambda_D * tau = kappa`.  A zero/non-finite trace fails
  the run.  This matches total stiffness, not directional stiffness or penalty
  magnitude at every point on the optimization path.
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
verify weighted-trace equality numerically, bind source/input hashes, reject
an existing output, and fail if provenance changes while running.  Its CPU
self-check explicitly materializes toy rank-1 and diagonal matrices and checks
the matching formula.

Stop only for a mechanical failure, OOM, non-finite quantity, failed invariant,
or provenance change.  Never stop or select cells based on result sign.  Logs
and failures remain under `runs/r19_penalty_match/`.  Only a complete,
independently summarized matrix can enter the appendix, explicitly labeled
exploratory and limited to one two-task factual stream.
