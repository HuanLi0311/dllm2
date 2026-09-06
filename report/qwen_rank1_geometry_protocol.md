# Qwen rank-1 Fisher geometry follow-up

Frozen before inspecting Qwen results on 2026-09-06. This follow-up does not
modify or support the submitted DLLM paper. Qwen3 is an autoregressive causal
LM, so the experiment tests only whether the held-out Fisher-surrogate ranking
extends to a different language-model objective.

## Models and objective

- Qwen3-1.7B is tested first. Qwen3-4B is run only if the 1.7B advance gate
  below passes.
- Each example is a prompt/completion pair. The loss is mean causal
  cross-entropy over completion tokens and EOS; prompt tokens are context only.
- Corpora are GSM8K training questions/solutions and the d2p and p2d reversal
  training prompts. Tokenization uses the model's own local tokenizer.
- For each corpus and seed, 32 calibration and 64 test examples are distinct.
  Seeds are 3407, 3408, and 3409.

## Parameter slices and score

The six frozen slices are the input and post-attention RMSNorm weights in the
first, middle, and last Transformer layers. For calibration gradients `g_i`
and disjoint test gradients `h_j`, fit

```text
F_cal  = mean_i (g_i g_i^T)
F_test = mean_j (h_j h_j^T)
mu     = mean_i g_i
R_mu   = c mu mu^T,  c = (mu^T F_cal mu) / ||mu||^4
D      = diag(F_cal)
s      = log(error(F_test, D) / error(F_test, R_mu))
```

Positive `s` favors rank-1. Every surrogate is fitted only on calibration
gradients and scored only on test gradients.

## Frozen 4B advance gate

Run Qwen3-4B only if all three conditions hold for Qwen3-1.7B:

1. the grand mean `s` over the 18 corpus-by-slice cells is positive;
2. at least two of three seed-level grand means are positive; and
3. at least two of three corpus means are positive.

This is an operational compute gate, not a significance test. Cell values,
seed means, and corpus means are reported whether or not it passes.
