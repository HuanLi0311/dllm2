# Submission command ledger

Commands below use double-blind-safe placeholders.  Set `REPO_ROOT`,
`SMDM_ROOT`, and `PYTHON` for the local checkout and environment; GPU allocation
does not alter model arguments.  Internal envelopes retain exact machine paths
for forensic use, while the review manifest exposes only relative paths.

```bash
cd "$REPO_ROOT"
PY=${PYTHON:-python}
PROBE=agent_skills/skill-benchmark/scripts/dllm_rank1_probe.py
AUDITED_PROBE=experiments/run_audited_geometry_probe.py
```

## Runnable checks

```bash
$PY $PROBE --self-check
$PY $AUDITED_PROBE --self-check
$PY experiments/simulate_fisher_null.py --self-check
$PY experiments/make_geometry_data.py --self-check
$PY experiments/build_comparison_contract.py --self-check
$PY experiments/make_paper_figures.py --self-check
$PY experiments/build_submission_manifest.py --self-check
$PY experiments/build_review_bundle.py --self-check
$PY experiments/build_backend_replay_audit.py --self-check
```

## Primary split-sample sweep

Common arguments:

```bash
COMMON="--code-root $SMDM_ROOT \
  --data runs/data/gsm8k_tasks.jsonl --split eval \
  --mask-probabilities 0.1,0.3,0.5,0.7,0.9 --include-native-schedule \
  --sequence-length 64 --shuffle-records --loss-mode native_conditional \
  --device cuda"
```

The frozen CLI names `--include-native-schedule` and `native_schedule` are
historical.  Their implementation draws $p$ uniformly and resamples the mask
until nonempty while holding that $p$ fixed.  The paper therefore reports these
rows only as a uniform-$p$, within-$p$-nonempty diagnostic, not as the native
training distribution or its global nonempty conditioning.

The frozen probe runs model forward/backward in bfloat16, promotes logits to
float32 for cross-entropy and gradient slices to float32 for CPU transfer, then
performs Gram and reconstruction algebra in float64.

219M actual parameters, configuration alias `170`, seeds 0--4:

```bash
for SEED in 0 1 2 3 4; do
  $PY $PROBE $COMMON \
    --checkpoint checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors \
    --model-size 170 --sample-sizes 32,64,128 --test-samples 128 \
    --parameter transformer.h.0.norm_1.weight,transformer.h.8.norm_1.weight,transformer.h.17.norm_1.weight \
    --seed "$SEED" \
    --output "runs/r08_split_primary/gsm_170_s${SEED}/benchmark.json"
done
```

1.14B actual parameters, configuration alias `1028`, seeds 0--2:

```bash
for SEED in 0 1 2; do
  $PY $PROBE $COMMON \
    --checkpoint checkpoints/mdm_safetensors/mdm-1028M-1600e18.safetensors \
    --model-size 1028 --sample-sizes 16,32,64 --test-samples 64 \
    --parameter transformer.h.0.norm_1.weight,transformer.h.9.norm_1.weight,transformer.h.19.norm_1.weight \
    --seed "$SEED" \
    --output "runs/r08_split_primary/gsm_1028_s${SEED}/benchmark.json"
done
```

## Isotropic null

```bash
$PY experiments/simulate_fisher_null.py \
  --case 768:8:128 --case 768:16:128 --case 768:32:128 \
  --case 768:64:128 --case 768:128:128 \
  --case 1792:8:64 --case 1792:16:64 --case 1792:32:64 \
  --case 1792:64:64 --repetitions 200 --seed 20260825 --device cpu \
  --output runs/r08_split_primary/isotropic_null_full.json
```

## Audited three-seed comparisons

The following template used probe seeds 0--2, calibration/test size 64, and the
219M normalization slices.  Four groups were run: GSM8K all-target, GSM8K
one-target, MT-Bench (`task-id 0`), and reversal (`task-id 1`).

```bash
$PY $AUDITED_PROBE \
  --checkpoint checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors \
  --model-size 170 \
  --code-root "$SMDM_ROOT" \
  --split eval --mask-probabilities 0.1,0.3,0.5,0.7,0.9 \
  --include-native-schedule --sample-sizes 64 --test-samples 64 \
  --sequence-length 64 \
  --parameter transformer.h.0.norm_1.weight,transformer.h.8.norm_1.weight,transformer.h.17.norm_1.weight \
  --shuffle-records --seed "$SEED" --device cuda \
  --data "$DATA" $TASK_ARGS \
  --loss-mode "$LOSS_MODE" --output "$OUTPUT"
```

Outputs are under `runs/r12_audited_controls/`.  The wrapper retains source-line
and token hashes, calibration/test mask hashes, and Gram, cross-Gram, and
diagonal sufficient statistics.  `experiments/build_comparison_contract.py`
then verifies exact all-target/one-target pairing, matched sample/mask
protocols, expected task IDs and loss modes, both probe hashes, and recomputes
every reported rank-1/diagonal error from those statistics.

GSM8K examples are independently tokenized with BOS and truncated, whereas
MT-Bench/reversal are EOS-delimited streams packed into fixed chunks that can
cross document boundaries.  This control therefore measures a
dataset-plus-preprocessing pipeline change, not corpus content in isolation.

The dense control used probe split/mask seeds 0--2, parameter
`transformer.h.0.attn.proj.weight`, probabilities `0.1,0.5,0.9`, and
calibration/test size 64, writing under `runs/r10_attention_split/`.

Failure provenance is preserved.  The original r09 seed-2 controls were
mistakenly launched without CUDA and failed before model execution; they are
superseded by r12.  The r10 attention seed-0 run and r12 GSM one-target seed-1
run encountered shared-filesystem standard-library import failures; their
successful replacements are `gsm_170_attn_s0_retry` and
`gsm_one_170_s1_retry`.  Both failures occurred before argument parsing.  Their
failed invocation commands were not retained, so exact argument equality with
the retries cannot be independently audited; only the successful retry
envelopes enter evidence, and those envelopes record the full intended command.

## Backend-fingerprint replay

The exact interpreter recorded in every evidence envelope resolves attention
through PyTorch SDPA (`flash-attn` absent) and SwiGLU through
`xformers==0.0.28.post1`.  To tie that post-run package capture to the original
outputs, we replayed the fixed-$p=0.1$, layer-0, `64|64`, seed-0 row once for
each checkpoint.  The frozen probe commands are stored in
`runs/backend_replay/model_219m/benchmark.json` and
`runs/backend_replay/model_1_14b/benchmark.json`; they include the auxiliary
condition solely to preserve the original RNG order.  All 32 common selected-row
metric/metadata fields match their original r12/r08 rows exactly.  The audited
r12 original additionally retains sufficient statistics that the frozen base
probe does not emit.

`experiments/build_backend_replay_audit.py` validates those two matches, checks
that all 23 submitted envelopes use the same interpreter/software/source
environment, enforces the replay host/frozen-probe/matched-config contract,
hashes the branch-defining `diffmodel.py` and `compat.py`, and
writes the complete 197-distribution lock to
`runs/backend_replay/backend_audit.json`. It records every model-relevant
optional branch: `flash_attn`, `dropout_layer_norm`, and `xentropy_cuda_lib`
are absent; attention and RMSNorm use their PyTorch paths, the probe calls
PyTorch cross-entropy directly, and `xformers==0.0.28.post1` supplies SwiGLU.

When preparing an anonymous source archive, include the exact model/audit
closure declared by the submission manifest: all tracked `lit_gpt/` files,
`pretrain/train_mdm.py`, the upstream README, and license. The manifest
refuses a dirty source subtree and verifies every archive member against its
declared hash.

## Figures, paper, and manifest

`experiments/make_paper_figures.py` receives the eight primary r08 envelopes
through `--geometry`, the twelve audited r12 envelopes through
`--comparison-control`, the three successful r10 envelopes through
`--slice-control`, and:

```bash
--null runs/r08_split_primary/isotropic_null_full.json \
--comparison-contract runs/r12_audited_controls/comparison_contract.json \
--output-dir ../assets/iclr_2/figures
```

The paper is compiled from the adjacent submission directory:

```bash
cd ../assets/iclr_2
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The final fail-closed manifest command is recorded in
`runs/submission_manifest.json` itself.  After the paper and manifest are
frozen, `experiments/build_review_bundle.py` writes de-identified compressed
envelopes to the requested output directory.

## Historical, excluded commands

Runs `r00`--`r04`, the corrected same-sample audit `r05`--`r06`, and the
continual/MCR pilots are not submission evidence and are omitted from the
source release. See `report/INVALIDATED_RESULTS.md` for the exact exclusion
reasons.
