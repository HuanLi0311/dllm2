# R21 SMDM-1.14B full-parameter feasibility smoke

**Frozen on 2026-09-07 before launch.  This is a mechanical, non-evidence
run and its endpoint values must not be cited in the paper.**

## Purpose

Test whether the existing R16 multitask implementation can execute the
highest-memory two-task Rank-1+GD path with every parameter of the 1.14B DLLM
trainable on one 40GB A100.  Passing only authorizes a separately designed
and pre-registered scale experiment; it is not such an experiment itself.

## Fixed smoke cell

- Node/GPU: `air-node-02`, physical GPU 0, after an ownership/free-memory
  check.
- Existing runner: `experiments/dllm_rank1_multitask.py`, unchanged.
- Model/checkpoint: `Diff_LLaMA_1028M` and the official
  `mdm-1028M-1600e18.safetensors`; all 1,142,367,744 parameters trainable.
- Stream: `d2p` facts 8--11 then `p2d` facts 12--15; seed 3407.
- Method: Rank-1 EWC plus generative distillation, with `ewc_lambda=1000`.
- Training: 100 steps/task, batch 4, learning rate `5e-5`, clip 1, native
  independent answer masks.
- Deliberately small mechanical auxiliaries: one Fisher row/fact (4 total),
  four balanced replay prompts, four diffusion steps for replay and held-out
  generation, and one held-out loss Monte Carlo repeat with eval batch 1.
- A wrapper records the exact command, exit code, wall/RSS information, and
  two-second GPU-memory samples.  It refuses an existing output and terminates
  after 90 minutes (plus a 60-second TERM grace period).

The smoke passes only if the unchanged runner self-check passes in the exact
environment, the process exits zero, writes a finite `status=ok` JSON, records
1,142,367,744 trainable parameters, and reaches both 100-step task stages plus
the intervening four-example Fisher/replay path.  OOM, timeout, non-finite
output, provenance change, or any other exception is retained as a failed
feasibility result.  There is one cell and no retry selected by endpoint sign.
