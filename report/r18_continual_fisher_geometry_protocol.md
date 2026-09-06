# R18 continual-task Fisher geometry protocol

**Frozen on 2026-09-07 before inspecting any R18 result.** This is a new,
exploratory run family.  It neither changes nor pools with the locked R16
submission matrix.

## Question and claim boundary

R16's broad geometry audit used packed text and frozen checkpoints, whereas
its EWC penalty used answer-only gradients after learning a continual task.
R18 closes that measurement gap at one learned checkpoint: it evaluates the
same mean-gradient rank-1 and diagonal surrogates on disjoint gradients from
the actual first R16 continual task.  It is a selected-slice diagnostic, not a
full-parameter reconstruction claim and not an additional utility experiment.

## Frozen design

- Model and task: full-parameter SMDM-219M training on `d2p` facts 8--11.
- Training: the R16 Task-1 path exactly--1,000 AdamW updates, batch size 4,
  learning rate `5e-5`, gradient clip 1, and seed-derived native answer-mask
  stream.  There is no replay or EWC on Task 1.
- Objective: answer-only masked cross-entropy, independent Bernoulli masks
  with `t ~ U(1e-3, 1)`, empty masks allowed, and `1/t` importance weighting
  divided by answer length.  This is the same implementation imported by the
  R16 runner.
- Calibration: the 40 designated R16 Fisher rows (10 training templates per
  fact), one mask draw per row with seed `seed + 2101`.
- Test: all 40 held-out test prompts (10 per fact), one independent mask draw
  per row with seed `seed + 4101`.  Calibration and test prompt strings must
  have empty intersection or the run fails.
- Parameter slices: `transformer.h.{0,8,17}.norm_1.weight` and
  `transformer.h.0.attn.proj.weight`.
- Optimization/probe seeds: 3407, 3408, 3409.  Seed 3407 is a mechanical
  pilot.  Once it passes the checks below, seeds 3408 and 3409 run unchanged,
  regardless of the sign or size of the pilot result.

For each slice, calibration fits
`R_mu = c mu mu^T` and `D = diag(F_cal)`.  Both are scored only against
`F_test`; the primary score is `s = log(error_D / error_R_mu)`, positive when
rank-1 has lower held-out relative Frobenius error.  Report every slice and
seed, the unweighted mean over the four fixed slices, mean/SEM over the three
seed means, and slice wins.  No significance claim is planned for three
seeds.

## Mandatory checks and stopping rule

The runner must pass a CPU self-check that (i) rejects calibration/test prompt
overlap, (ii) verifies that its audited loss call is exactly the imported R16
answer-only objective, (iii) matches matrix-free rank-1 and diagonal errors to
explicit small matrices, and (iv) verifies held-out oracle ordering.  Every
GPU result must be finite, identify the post-Task-1 sampling stage, record
source/dependency/input hashes and mask seeds, and repeat the prompt-overlap
check.  A source or input change during a run fails closed.

Stop and diagnose on OOM, a non-finite quantity, failed invariant, or changed
provenance.  Numerical results never trigger early stopping or condition
selection.  All successful and failed attempts remain in the R18 directory.
Only a complete independently recomputed three-seed summary may be considered
for the appendix, and only with the exploratory/slice-limited scope above.
