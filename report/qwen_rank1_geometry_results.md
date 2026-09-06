# Qwen3-1.7B rank-1 Fisher geometry result

This report freezes the initial 1.7B gate decision. The later user-authorized
0.6B/1.7B/4B scale follow-up is recorded in `../EXPERIMENT.md`.

The frozen follow-up protocol completed for Qwen3-1.7B. Qwen3 is an
autoregressive causal LM, so this is a cross-objective geometry check and was
not added to the DLLM paper.

## Advance decision

The predeclared 4B gate failed:

| Gate item | Qwen3-1.7B result | Required |
|---|---:|---:|
| Grand mean score `s` positive | `+0.0522 +/- 0.1277` | positive |
| Positive seed means | `2/3` | at least `2/3` |
| Positive corpus means | `1/3` | at least `2/3` |

Therefore the frozen protocol did not authorize Qwen3-4B. It was subsequently
run only after the user explicitly overrode this gate.

## Held-out results

Positive `s = log(error_diagonal / error_rank1)` favors rank-1.

| Unit | Mean `s` +/- SEM | Seed values |
|---|---:|---|
| Overall | `+0.0522 +/- 0.1277` | `[+0.2032, +0.1551, -0.2017]` |
| GSM8K | `+0.3104 +/- 0.0378` | `[+0.3567, +0.2355, +0.3390]` |
| d2p | `-0.0159 +/- 0.2424` | `[+0.2443, +0.2083, -0.5003]` |
| p2d | `-0.1378 +/- 0.1530` | `[+0.0087, +0.0215, -0.4437]` |

Only 6 of 18 corpus-by-slice means favor rank-1. The last-layer input RMSNorm
favors rank-1 for all three corpora, while most early and middle slices favor
diagonal. The result is localized by layer and corpus rather than a broad
rank-1 reconstruction advantage.

## Audit

- Three exact seeds: 3407, 3408, 3409.
- Per seed and corpus: 32 calibration plus 64 disjoint test examples.
- Six fixed slices: input and post-attention RMSNorm in the first, middle, and
  last Qwen layer.
- All 54 seed-by-corpus-by-slice rows are finite and satisfy held-out best-rank-1
  oracle ordering against the fitted rank-1 surrogate.
- Model weights were hashed before the runs and rehashed successfully after
  them.
- Runner SHA-256:
  `3282a44d22a5f18169a25f23318f1960af148f0a91423339be2fc9811300804e`.
- Protocol SHA-256:
  `f4853e60f3723f49dd04259366dbb1ca05376f2685651fccb774a8d776fa811e`.
- Model-inventory SHA-256:
  `5e5560da0bc8233ff33528b294f22586944cb7f15642607f28db6107a39f29f0`.
