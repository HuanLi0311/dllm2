# Rank-1 Fisher consolidation in masked diffusion language models

This repository audits whether a rank-1 empirical-Fisher approximation and
Fisher-weighted consolidation transfer from visual diffusion to masked
diffusion language models (DLLMs). It separates three questions:

1. Does the source linear--Gaussian argument imply an effectively rank-1
   Fisher?
2. Does a rank-1 surrogate fitted on calibration gradients reconstruct an
   independent empirical Fisher better than a diagonal surrogate?
3. Does rank-1 EWC improve continual DLLM training beyond
   prompt-conditioned generative distillation (GD)?

Documentation: [locked R16 protocol](report/r16_protocol.md) ·
[Qwen3 scale extension](report/qwen_continual_scale_results.md) ·
[full-parameter Fisher controls](report/r23_corrected_trace_protocol.md) ·
[invalidation ledger](report/INVALIDATED_RESULTS.md)

## Findings

- The corrected linear-model gradient retains an input outer product. In an
  exact Gaussian counterexample, the second-to-first Fisher eigenvalue ratio
  tends to zero while the best rank-1 relative Frobenius residual tends to
  \(\sqrt{2/3}\). An eigengap alone is therefore insufficient.
- Under an isotropic null, same-sample scoring selects rank-1 in nearly every
  low-\(n/d\) case; independent-test scoring selects it in 0/200 repetitions.
- At maximum calibration size, the held-out geometric-mean rank-1 error is
  8.3% lower than diagonal at the 219M checkpoint but 23.7% higher at 1.14B.
  Only 6/15 and 3/15 layer--mask cells favor rank-1. The ranking is local, not
  model-wide.
- The 589,824-parameter dense attention slice favors diagonal in all three
  fixed-mask conditions: mean log error ratio \(-0.031\), 95% CI
  \([-0.040,-0.021]\).
- In the R16 two-task confirmation, GD reduces final average loss from
  \(1.485\pm0.172\) to \(0.182\pm0.011\). Rank-1+GD minus GD is
  \(+0.014\pm0.023\), with one win in three paired seeds; this supports GD,
  not a repeatable rank-1 increment.
- In the strict 33-run four-task matrix, GD reduces final average loss from
  \(2.927\pm0.231\) to \(1.114\pm0.374\) forward and from
  \(2.168\pm0.390\) to \(0.891\pm0.215\) reverse. Rank-1+GD minus GD is
  \(-0.132\pm0.603\) and \(-0.285\pm0.214\): favorable means, but only two
  of three paired seeds in either order and substantial heterogeneity.
- On the prelocked 24-run fresh-fact sensitivity extension, Rank-1+GD minus GD is
  \(+0.063\pm0.125\) forward and \(-0.067\pm0.048\) reverse, so the mean
  changes sign with order. Rank-1+GD nevertheless beats the selected
  Diagonal+GD recipe in all six fresh-fact pairs. This is a local recipe
  comparison, not evidence for a modality-wide rank-1 mechanism.
- On actual post-Task-A answer-only gradients over all 219M parameters, both
  calibration-fitted Fisher surrogates have held-out relative error essentially
  one. The held-out Fisher's own best rank-1 error is \(0.446\pm0.064\), while
  the calibration-direction oracle remains \(0.999995\pm0.000003\): the
  direction and scale, rather than low rank in the test Fisher itself, fail to
  transfer.
- In the 18-cell 219M implemented weighted-trace control, the canonical-clip
  Rank-1+GD minus Diagonal+GD final-loss difference is
  \(+0.001\pm0.052\); the inactive-clip difference is
  \(-0.005\pm0.031\). In the nine-cell full-parameter 1.14B extension it is
  \(+0.080\pm0.070\), with every seed favoring diagonal, but the matched trace
  target varies \(2.57\times10^5\)-fold across seeds. These are exploratory
  results on one two-task factual stream, not a superiority claim.
- In the last-block Qwen3 extension at 0.6B, 1.7B, and 4B, GD lowers final
  loss and forgetting versus Sequential in all nine paired runs. Rank-1+GD
  raises final loss versus GD in all nine because it impairs later-task
  acquisition. This is qualitative cross-family robustness, not a causal
  parameter-count result.

Both complete matrices pass strict audit: 33/33 main runs
and 24/24 fresh-fact runs. Their summaries are
<code>runs/r16_native_mask/{summary,fresh_summary}.json</code>. Compact final
summaries stay readable in <code>runs/</code>; de-identified compressed raw
envelopes and their hashes are in [release_evidence](release_evidence). The
separate [release_extension_evidence](release_extension_evidence) bundle holds
only the valid R18/R23/R24 continual-Fisher extensions.

## What is measured

For calibration gradients \(g_i\) and disjoint test gradients \(h_j\):

~~~text
F_cal  = mean_i (g_i g_i^T)
F_test = mean_j (h_j h_j^T)
mu     = mean_i g_i
R_mu   = c mu mu^T,  c = (mu^T F_cal mu) / ||mu||^4
D      = diag(F_cal)
~~~

Both fitted surrogates are scored by relative Frobenius error against
<code>F_test</code>. The implementation never materializes a full
parameter-space Fisher; it uses exact Gram identities.

The continual objective is:

~~~text
L = L_DLLM,current + beta * L_GD
    + sum_old (lambda / 2) * (theta - theta_old)^T Fhat_old (theta - theta_old)
~~~

The native conditional DLLM loss masks answer tokens independently, permits an
empty mask to contribute zero, and weights masked-token cross entropy by
<code>1 / t</code>. Current-task and replay minibatches use separate
deterministic random streams.

## Locked protocols

### Geometry

| Item | Setting |
|---|---|
| Checkpoints | SMDM 219,050,496 and 1,142,367,744 parameters |
| Primary text | 320 fixed-length GSM8K-text sequences |
| Controls | 131 MT-Bench and 298 reversal-relation sequences |
| Parameter slices | Three normalization vectors per checkpoint; one dense attention matrix |
| Fixed mask probabilities | 0.1, 0.3, 0.5, 0.7, 0.9 |
| Calibration/test | 219M: 32/64/128 \| 128; 1.14B: 16/32/64 \| 64 |
| Probe seeds | Five at 219M; three at 1.14B and for controls |

The text-source labels identify gradient inputs. This repository does not
report GSM8K solving accuracy or MT-Bench judge scores.

### Continual learning

| Item | Setting |
|---|---|
| Model | Full 219M SMDM checkpoint |
| Optimizer | AdamW, learning rate <code>5e-5</code>, gradient clip 1 |
| Training | 1,000 steps/task, batch size 4 |
| Evaluation | 32 independent mask replicates per test template |
| Replay | 64 generated completions per old task, exactly balanced by fact |
| Fisher | 10 examples/fact |
| Selected EWC multipliers | Rank-1 <code>1e5</code>; diagonal <code>1e3</code>; GD weight 1 |
| Seeds | 3407, 3408, 3409 |
| Main matrix | Four tasks, exact reverse order, 33 runs |
| Fresh matrix | Three unseen-fact tasks, exact reverse order, 24 runs |

The primary endpoints are final average held-out DLLM loss and past-task
forgetting. Teacher-forced accuracy, exact match, target containment, and
ROUGE-L are secondary. The benchmark is a controlled factual-association
stream, not a broad language-ability benchmark.

## Environment and inputs

Reported DLLM runs use Python 3.9.25, PyTorch 2.4.1+cu121, and NVIDIA A100
40GB GPUs. Install the pinned packages:

~~~bash
conda create -n dllm-rank1 python=3.9 -y
conda activate dllm-rank1
python -m pip install -r requirements.txt
~~~

The exploratory Qwen3 runs use Python 3.11.15, PyTorch 2.7.0+cu128, and
Transformers 4.57.6. Install those separately with
<code>requirements-qwen.txt</code>; mixing the two frozen environments is not
an exact reproduction.

The upstream SMDM source is vendored in [SMDM](SMDM), the tokenizer is in
[tokenizer](tokenizer), and the reversal rows are in the upstream data tree.
Download the two checkpoints from
[nieshen/SMDM](https://huggingface.co/nieshen/SMDM) and verify:

| Checkpoint | Bytes | SHA-256 |
|---|---:|---|
| <code>mdm-170M-100e18.safetensors</code> | 876,215,712 | <code>2d8c9b9a730715f2c772d5bc740e12951fc160e5e8511a16835f3537401ea9bb</code> |
| <code>mdm-1028M-1600e18.safetensors</code> | 4,569,486,608 | <code>ce96ce67a051613b6d7feb419c99c0b4db5bfcfaaa0833ed7f7ecbc6632841d6</code> |

Do not add optional fused kernels when reproducing the reported artifacts:
the recorded backend uses PyTorch SDPA, the project RMSNorm fallback, PyTorch
cross entropy, and <code>xformers==0.0.28.post1</code> for SwiGLU.

## Runnable checks

From this directory:

~~~bash
python experiments/dllm_rank1_transfer.py --self-check
python experiments/dllm_rank1_multitask.py --self-check
python experiments/summarize_dllm_rank1_transfer.py --self-check
python experiments/summarize_dllm_rank1_multitask.py --self-check
python experiments/simulate_fisher_null.py --self-check
python experiments/run_audited_geometry_probe.py --self-check
python experiments/make_paper_figures.py --self-check
python experiments/build_submission_manifest.py --self-check
python experiments/build_review_bundle.py --self-check
python experiments/build_review_bundle.py --verify release_evidence/release_manifest.json
python experiments/build_review_bundle.py --verify release_extension_evidence/release_manifest.json
python experiments/dllm_continual_fisher_geometry.py --self-check
python experiments/summarize_r18_continual_fisher_geometry.py --self-check
python experiments/summarize_r18_absolute_diagnostics.py --self-check
python experiments/dllm_corrected_trace_control.py --self-check
python experiments/summarize_corrected_trace_controls.py --self-check
python experiments/qwen_rank1_geometry.py --self-check
python experiments/qwen_continual_transfer.py --self-check
python experiments/summarize_qwen_continual_transfer.py self-check
~~~

Authors with the internal raw-result tree can add `--require-internal` to the
extension-bundle verification command to check source presence and hashes too.

## Representative locked continual run

The validation summary is a required, hash-bound input. A representative main
run is:

~~~bash
python experiments/dllm_rank1_multitask.py \
  --final-protocol \
  --method rank1_gd \
  --checkpoint checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors \
  --start-direction d2p --order forward \
  --group-start 8 --tasks 4 --group-count 4 \
  --fisher-per-fact 10 --steps-per-task 1000 \
  --batch-size 4 --eval-batch-size 4 --eval-mc-samples 32 \
  --replay-per-task 64 --replay-steps 32 --replay-length 52 \
  --distill-weight 1 --ewc-lambda 100000 \
  --seed 3407 --generation-seed 3407 \
  --validation-summary runs/r16_native_mask/validation_summary.json \
  --output runs/r16_native_mask/final/forward/s3407/rank1_gd.json
~~~

Every long-horizon output snapshots source, dependency, checkpoint, tokenizer,
data, and validation-summary hashes before model loading and verifies the same
snapshot immediately before writing.

Generate strict summaries and paper artifacts:

~~~bash
python experiments/summarize_dllm_rank1_multitask.py \
  --run-root runs/r16_native_mask/final \
  --output runs/r16_native_mask/summary.json \
  --protocol main

python experiments/summarize_dllm_rank1_multitask.py \
  --run-root runs/r16_native_mask/fresh \
  --output runs/r16_native_mask/fresh_summary.json \
  --protocol fresh
~~~

The summarizer rejects an incomplete matrix, unexpected files, protocol drift,
mixed hashes, replay imbalance, wrong masks or seeds, stale validation input,
and non-finite metrics. It recomputes endpoints from stage-level raw values.

## Evidence map

- [release_evidence](release_evidence): 315 de-identified compressed run
  artifacts and a SHA-256 manifest.
- [release_extension_evidence](release_extension_evidence): 34 de-identified
  R18/R23/R24 raw and summary artifacts with a separately verified manifest.
- [R18 continual-Fisher summary](release_extension_evidence/raw/runs/r18_continual_fisher_geometry/summary.json.gz)
  and [absolute diagnostics](release_extension_evidence/raw/runs/r18_continual_fisher_geometry/absolute_diagnostics.json.gz):
  full-parameter continual-task Fisher geometry.
- [R23 summary](release_extension_evidence/raw/runs/r23_corrected_trace/summary.json.gz)
  and [R24 summary](release_extension_evidence/raw/runs/r24_scale1028_corrected_trace/summary.json.gz):
  complete 219M and 1.14B implemented weighted-trace controls.
- [runs/r16_native_mask](runs/r16_native_mask): compact validation, main, and
  fresh-fact DLLM summaries.
- [runs/qwen_continual_scale](runs/qwen_continual_scale): compact Qwen3
  selections and three-scale summary.
- [runs/qwen_rank1_geometry](runs/qwen_rank1_geometry): compact Qwen3 geometry
  summaries.
- [runs/commands.md](runs/commands.md): exact geometry command ledger.
- [report/INVALIDATED_RESULTS.md](report/INVALIDATED_RESULTS.md): excluded
  pilots and exact reasons; invalid raw runs are intentionally absent.

## Scope

The broad packed-text geometry audit samples selected parameter slices. The
separate R18 diagnostic uses actual answer-only gradients over all 219M
parameters but only one learned task and checkpoint. The confirmatory continual
study covers one 219M checkpoint, controlled factual associations, and three
optimization seeds. R23 and R24 are exploratory two-task controls; even their
matched implemented weighted traces do not match directional stiffness,
parameter displacement, or the penalty encountered along a trajectory. The
Qwen3 extension changes model family, objective, and trainable fraction. The
two DLLM task orders are sensitivity analyses, not independent replications,
and the methods are not FLOP-matched. Retaining old prompts also makes GD
conditional replay, not data-free generation.

See [NOTICE.md](NOTICE.md) for upstream attribution and licenses.
