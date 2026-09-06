# R22 SMDM-1.14B full-parameter matched-Fisher scale extension

**Design frozen on 2026-09-07 before any R22 run; launch awaits independent
review.** R22 is a predeclared exploratory scale extension of R19.  It does
not alter or pool with R16/R19 and cannot support a broad model-family claim.

## Fixed question and grid

Does the R19 ordering among no-Fisher GD, weighted-trace-matched Rank-1+GD,
and weighted-trace-matched Diagonal+GD recur when the same two-task continual
stream updates every parameter of the 1.14B SMDM checkpoint?

- Model: official SMDM-1.14B (`Diff_LLaMA_1028M`), all 1,142,367,744
  parameters trainable.
- Stream: `d2p_8-11 -> p2d_12-15`, identical rows and answer-only objective to
  R19.
- Task A: 1,000 AdamW steps, batch 4, learning rate `5e-5`, clip 1.
- Fisher: the same 40 designated Task-A rows, mask seed `seed+2101`; rank-1
  and diagonal use identical gradients.
- Replay/GD: 64 Task-A prompts balanced 16/16/16/16, 32 diffusion steps,
  frozen teacher, distillation weight 1.
- Fisher matching: per seed, `lambda_D=1000`, `tau=tr(D)`,
  `kappa=1000*tau`, and `lambda_R1=kappa/alpha`.  Thus
  `lambda_R1*alpha=lambda_D*tau`; a non-positive/non-finite value fails.
- Task B: 1,000 steps with clip 1.  The no-clip branch is deliberately not
  repeated at scale: R19 already isolates clipping, while the R21 smoke saw
  a Task-B pre-clip maximum of 270,336 and clipping on every step.
- Methods: GD, Rank-1+GD, Diagonal+GD.  Seeds: 3407, 3408, 3409.  Total: nine
  cells.  Seed 3407 is the three-method mechanical subset; after it passes,
  all six remaining cells run unchanged regardless of numerical direction.
- Evaluation: R19's 32 held-out loss-mask repeats, answer-token accuracy, and
  32-step greedy generation on both tasks.

Sequential training is omitted.  It does not isolate Fisher structure, while
GD is the necessary shared non-Fisher baseline for the two incremental EWC
effects.  Adding Sequential would cost three cells without answering the
registered cross-scale question.

## Endpoints and audit

The primary endpoint is paired final average held-out loss.  Past-task
forgetting and final-task loss separate retention/acquisition.  Secondary
diagnostics are answer-token accuracy, pre-clip gradient-norm maximum, clip
fraction, and weighted EWC mean/maximum.  Report all nine cells, paired
per-seed differences, mean and SEM; do not make a superiority/significance
claim from three seeds.

The new runner must reuse the exact R19 loss, task/replay selection, training,
and Fisher functions; refuse existing outputs; bind runner/protocol,
dependency, checkpoint, tokenizer, data, and task-artifact hashes; verify
40-row Fisher, 16/16/16/16 replay, matching equality, finite endpoints, and
within-seed Task-A/Fisher/replay identity.  A separate fail-closed summarizer
must recompute all primary endpoints from stage metrics.  CPU toy checks must
cover the weighted-trace formula and direct loss alias.

## Measured cost envelope

R21's 100-step worst-path smoke took 182.829 runner seconds (200.33 seconds
including import/monitoring), with Task-A and Task-B training taking 11.971
and 48.482 seconds.  Linear step scaling gives about 10.1 minutes of training
for a Rank-1+GD cell.  Full Fisher/replay/evaluation dominated R19, so budget
35--45 minutes per complete 1.14B cell, with a conservative 60-minute stop
expectation: about 5.25--6.75 GPU-hours for nine cells (9 GPU-hours reserved).
On three verified-free 40GB GPUs this is about 1.75--2.25 hours wall time,
up to 3 hours reserved.  R21 peaked at 21,335 MiB GPU and 35.96 GiB host RSS;
three concurrent cells therefore require at least 120 GiB available host RAM.

Stop only for an OOM, timeout, non-finite value, failed invariant, or changed
provenance.  Preserve every success/failure.  Never change or prune cells from
an endpoint sign.  No launch is authorized until the runner, self-check,
contract hashes, and this design receive independent review.
