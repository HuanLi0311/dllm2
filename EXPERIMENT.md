# Additional Qwen Fisher-geometry experiments

Status: complete on 2026-09-06. These are exploratory follow-ups. The
continual-learning scale extension appears only in the paper appendix; the
Qwen Fisher-geometry follow-up remains repository-only. Qwen3 is an
autoregressive causal LM, whereas the main paper studies masked diffusion LMs.

## Shared protocol

- Objective: mean causal cross-entropy over completion tokens and EOS; prompt
  tokens are context only.
- Surrogates: mean-gradient rank-1 and empirical-Fisher diagonal, both fitted
  on 32 calibration gradients and scored on 64 disjoint test gradients.
- Score: `s = log(error_diagonal / error_rank1)`; positive favors rank-1.
- Corpora: GSM8K answers, description-to-person (`d2p`), and
  person-to-description (`p2d`).
- Slices: input and post-attention RMSNorm weights in the first, middle, and
  last Transformer layers.
- Seeds: 3407, 3408, and 3409.
- Code: `experiments/qwen_rank1_geometry.py`.
- Frozen initial protocol: `report/qwen_rank1_geometry_protocol.md`.

## Checkpoints

All three points use official Qwen3 base checkpoints. Qwen3-0.6B was not in the
initial cache, so its official snapshot was downloaded through the configured
Hugging Face mirror rather than substituting the cached Qwen2.5-0.5B model.

| Model | Layers | Hidden size | Exact local checkpoint |
|---|---:|---:|---|
| Qwen3-0.6B | 28 | 1024 | `models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen3-1.7B | 28 | 2048 | `Qwen3-1.7B` |
| Qwen3-4B | 36 | 2560 | `models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c` |

## Held-out results

Positive `s = log(error_diagonal / error_rank1)` favors rank-1. Uncertainty is
the SEM across three seeds, not a confidence interval.

| Model | Overall `s` +/- SEM | Seed values | Positive cells |
|---|---:|---|---:|
| Qwen3-0.6B | `+0.0883 +/- 0.0205` | `[+0.0684, +0.0673, +0.1292]` | 8/18 |
| Qwen3-1.7B | `+0.0522 +/- 0.1277` | `[+0.2032, +0.1551, -0.2017]` | 6/18 |
| Qwen3-4B | `+0.0138 +/- 0.0030` | `[+0.0139, +0.0190, +0.0086]` | 5/18 |

| Model | GSM8K `s` +/- SEM | d2p `s` +/- SEM | p2d `s` +/- SEM |
|---|---:|---:|---:|
| Qwen3-0.6B | `+0.2441 +/- 0.0254` | `+0.1582 +/- 0.0284` | `-0.1374 +/- 0.0414` |
| Qwen3-1.7B | `+0.3104 +/- 0.0378` | `-0.0159 +/- 0.2424` | `-0.1378 +/- 0.1530` |
| Qwen3-4B | `+0.1450 +/- 0.0273` | `+0.0024 +/- 0.0195` | `-0.1059 +/- 0.0295` |

## Scale follow-up

The frozen protocol's original 4B gate failed at 1.7B because only 1/3 corpus
means was positive, although the grand mean and 2/3 seed means were positive.
The user explicitly overrode that gate before the 4B run. This is a declared
protocol extension, not a retrospective gate change.

Grand-score changes paired by seed were:

| Contrast | Mean paired change +/- SEM | Per-seed changes |
|---|---:|---|
| 1.7B - 0.6B | `-0.0361 +/- 0.1480` | `[+0.1348, +0.0878, -0.3309]` |
| 4B - 1.7B | `-0.0384 +/- 0.1253` | `[-0.1893, -0.1361, +0.2103]` |
| 4B - 0.6B | `-0.0745 +/- 0.0231` | `[-0.0545, -0.0483, -0.1206]` |

The matched Qwen3 grand mean decreases monotonically
`+0.0883 -> +0.0522 -> +0.0138`, and all three seed-paired 0.6B-to-4B changes
are negative. This is evidence that scale matters for this diagnostic, but not
that it has a uniform effect: only 3/18 cells decrease monotonically, 3/18
increase monotonically, and 7/18 change sign somewhere along the series.

The corpus interaction is also substantial. From 0.6B to 4B, GSM8K changes by
`-0.0991 +/- 0.0520`, d2p by `-0.1558 +/- 0.0395`, but p2d by
`+0.0315 +/- 0.0123`. Qwen3-4B has the smallest seed SEM and a near-zero grand
effect, suggesting increased geometric stability more clearly than a broad
rank-1 advantage.

Parameter count is therefore a plausible moderator, not an isolated causal
variable: training data, optimization, and the 4B architecture change with the
checkpoint. The paper's 219M SMDM score (`+0.086`) is contextual only because
it also changes model family, masked-vs-causal objective, and evaluation grid;
it must not be treated as a fourth matched scale point.

## Artifacts and audit

- Per-model summaries: `runs/qwen_rank1_geometry/qwen3_0.6b/summary.json`,
  `runs/qwen_rank1_geometry/qwen3_1.7b/summary.json`, and
  `runs/qwen_rank1_geometry/qwen3_4b/summary.json`.
- Nine formal seed files and 162 result rows passed the final finite-value,
  provenance, score-recomputation, and held-out rank-1 oracle checks.
- Runner SHA-256:
  `3282a44d22a5f18169a25f23318f1960af148f0a91423339be2fc9811300804e`.
- Protocol SHA-256:
  `f4853e60f3723f49dd04259366dbb1ca05376f2685651fccb774a8d776fa811e`.
- Inventory SHA-256 values (0.6B, 1.7B, 4B):
  `27283d16de5608b046aa51b3295222b3174f755ef7727fd01051aa45c52ba782`,
  `5e5560da0bc8233ff33528b294f22586944cb7f15642607f28db6107a39f29f0`,
  and `9f4d81d499c15f1819fa88dfc305ea9663a0c1fea355a7ee1a6aa254292b71d7`.

## Continual-learning scale extension

A 36-run formal matrix tested whether the paper's behavioral result recurs at
Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B. It used the same four-method comparison
(`Sequential`, `GD`, `Rank-1+GD`, and `Diagonal+GD`), main forward fact stream,
and three seeds. To fit a frozen GD teacher and Fisher constraints on a 40GB
A100, only the last Transformer block was trainable at all three scales. EWC
lambdas were selected using 132 validation runs on a disjoint two-task stream.

| Model | Method | Final average loss | Past-task forgetting | Answer-token accuracy |
|---|---|---:|---:|---:|
| 0.6B | Sequential | `4.2090 +/- 0.1166` | `2.2419 +/- 0.1132` | `0.7467 +/- 0.0051` |
| 0.6B | GD | `2.8328 +/- 0.0481` | `0.4226 +/- 0.0432` | `0.8516 +/- 0.0005` |
| 0.6B | Rank-1+GD | `5.0454 +/- 0.1531` | `-0.0671 +/- 0.0532` | `0.5535 +/- 0.0042` |
| 0.6B | Diagonal+GD | `3.3352 +/- 0.0567` | `-0.0788 +/- 0.0215` | `0.7126 +/- 0.0018` |
| 1.7B | Sequential | `3.8098 +/- 0.0729` | `1.8136 +/- 0.1024` | `0.7276 +/- 0.0041` |
| 1.7B | GD | `2.6940 +/- 0.0280` | `0.2791 +/- 0.0251` | `0.8404 +/- 0.0008` |
| 1.7B | Rank-1+GD | `4.0351 +/- 0.2046` | `-0.2340 +/- 0.0182` | `0.6009 +/- 0.0144` |
| 1.7B | Diagonal+GD | `2.6532 +/- 0.0241` | `0.2610 +/- 0.0180` | `0.8385 +/- 0.0005` |
| 4B | Sequential | `3.9957 +/- 0.1467` | `2.0093 +/- 0.1590` | `0.7536 +/- 0.0089` |
| 4B | GD | `2.7971 +/- 0.0288` | `0.3728 +/- 0.0576` | `0.8541 +/- 0.0010` |
| 4B | Rank-1+GD | `4.1795 +/- 0.5159` | `-0.1831 +/- 0.0994` | `0.6116 +/- 0.0407` |
| 4B | Diagonal+GD | `2.7980 +/- 0.0248` | `0.3729 +/- 0.0535` | `0.8551 +/- 0.0005` |

GD versus Sequential lowers final loss by `-1.3762 +/- 0.0885`,
`-1.1158 +/- 0.0642`, and `-1.1986 +/- 0.1754`, and lowers forgetting by
`-1.8193 +/- 0.1083`, `-1.5345 +/- 0.0789`, and `-1.6365 +/- 0.2093` at
0.6B, 1.7B, and 4B; all paired seed differences favor GD. Rank-1+GD lowers
the narrow forgetting metric but increases final loss over GD by
`+2.2127 +/- 0.1475`, `+1.3411 +/- 0.1897`, and `+1.3824 +/- 0.5241`, with
0/3 seed wins at every scale, because strong constraints prevent later-task
acquisition. Diagonal+GD is worse than GD at 0.6B, slightly better at 1.7B,
and essentially identical at 4B.

Conclusion: the qualitative behavioral result is robust across these three
Qwen3 scales. GD reliably mitigates catastrophic forgetting; the rank-1
Fisher penalty does not add stable continual-learning utility. Absolute GD
endpoints vary nonmonotonically with scale, but the reliable component does not.
This is a last-block, one-order exploratory extension rather than a causal
parameter-count result.

Full protocol, paired results, acquisition diagnostic, and audit:
`report/qwen_continual_scale_results.md`.
