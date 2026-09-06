# Qwen3 continual-learning scale extension

Status: complete on 2026-09-06. This is an autoregressive, last-block adaptation
extension of the paper's controlled factual-association study. It is not a
full-parameter Qwen replication.

## Protocol

- Models: official Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B base checkpoints.
- Formal task stream: `d2p_8-11 -> p2d_12-15 -> d2p_16-19 -> p2d_20-23`.
- Methods: Sequential, GD, Rank-1+GD, and Diagonal+GD.
- Three paired seeds: 3407, 3408, and 3409.
- Final Transformer block only; 1,000 AdamW steps/task, batch 4, learning rate
  `5e-5`; answer-only causal cross-entropy.
- GD: 64 balanced retained prompts/prior task, greedy teacher generations, and
  teacher KL on generated answer tokens.
- EWC lambdas were selected before formal runs on the disjoint two-task stream
  `d2p_0-1 -> p2d_2-3`. The predeclared boundary-extension rule required 132
  validation runs across the three scales. Selected Rank-1/Diagonal lambdas
  were `1e8`/`1e9` at 0.6B, `1e7`/`1e6` at 1.7B, and `1e8`/`1e5` at 4B.

The frozen base protocol is `qwen_continual_scale_protocol.md`; the separately
frozen 1.7B addendum is `qwen_continual_scale_1.7b_protocol.md`.

## Formal results

Values are mean +/- SEM over optimization seeds. Lower loss and forgetting are
better; higher answer-token accuracy is better.

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

## Paired effects

All values are left method minus right method. Negative loss/forgetting favors
the left method.

| Model | Contrast | Final-loss change | Forgetting change | Seed wins on loss |
|---|---|---:|---:|---:|
| 0.6B | GD - Sequential | `-1.3762 +/- 0.0885` | `-1.8193 +/- 0.1083` | 3/3 |
| 1.7B | GD - Sequential | `-1.1158 +/- 0.0642` | `-1.5345 +/- 0.0789` | 3/3 |
| 4B | GD - Sequential | `-1.1986 +/- 0.1754` | `-1.6365 +/- 0.2093` | 3/3 |
| 0.6B | Rank-1+GD - GD | `+2.2127 +/- 0.1475` | `-0.4897 +/- 0.0112` | 0/3 |
| 1.7B | Rank-1+GD - GD | `+1.3411 +/- 0.1897` | `-0.5130 +/- 0.0342` | 0/3 |
| 4B | Rank-1+GD - GD | `+1.3824 +/- 0.5241` | `-0.5558 +/- 0.0446` | 0/3 |
| 0.6B | Diagonal+GD - GD | `+0.5025 +/- 0.0409` | `-0.5014 +/- 0.0460` | 0/3 |
| 1.7B | Diagonal+GD - GD | `-0.0409 +/- 0.0047` | `-0.0180 +/- 0.0076` | 3/3 |
| 4B | Diagonal+GD - GD | `+0.0009 +/- 0.0047` | `+0.0002 +/- 0.0058` | 1/3 |
| 0.6B | Rank-1+GD - Diagonal+GD | `+1.7102 +/- 0.1072` | `+0.0117 +/- 0.0527` | 0/3 |
| 1.7B | Rank-1+GD - Diagonal+GD | `+1.3819 +/- 0.1938` | `-0.4950 +/- 0.0268` | 0/3 |
| 4B | Rank-1+GD - Diagonal+GD | `+1.3814 +/- 0.5250` | `-0.5560 +/- 0.0470` | 0/3 |

GD improves final loss, forgetting, and answer-token accuracy at all three
Qwen scales; all nine paired final-loss and forgetting comparisons favor GD
over Sequential. Absolute GD endpoints vary modestly and nonmonotonically:
final losses are `2.8328`, `2.6940`, and `2.7971`, while forgetting is
`0.4226`, `0.2791`, and `0.3728` at 0.6B, 1.7B, and 4B.

The low or negative forgetting under the strongly regularized EWC runs is not
better continual learning by itself. Rank-1+GD's mean loss when each task was
first learned is `5.0958`, `4.2106`, and `4.3168` at 0.6B, 1.7B, and 4B,
versus `2.5158`, `2.4847`, and `2.5175` for GD. Its final-task loss is
`5.2761`, `5.1559`, and `4.5743`, versus `0.3241`, `0.3683`, and `0.3012` for
GD. Rank-1 therefore preserves parameters partly by failing
to acquire later tasks, which raises final loss and lowers accuracy in every
paired seed. The 0.6B diagonal setting shows the same, milder stability-
plasticity failure; the 1.7B diagonal setting is slightly better than GD, and
the 4B diagonal setting is effectively indistinguishable from GD.

## Conclusion

The paper's qualitative behavioral conclusion survives this scale extension:
prompt-conditioned GD robustly mitigates catastrophic forgetting, whereas the
rank-1 Fisher penalty does not provide a stable incremental utility benefit.
Here the evidence is stronger in direction than in the original DLLM matrix:
Rank-1+GD is worse than GD and worse than Diagonal+GD in final average loss at
all three scales and in all nine paired seeds. Absolute endpoints vary
nonmonotonically with scale, but the identity of the reliable component does not.

This conclusion is limited to one forward synthetic fact stream, three seeds,
autoregressive Qwen3, and last-block adaptation. It supports cross-scale
robustness of the qualitative conclusion, not a universal continual-learning
claim or a causal parameter-count effect.

## Audit

- 132 validation runs and 36 formal runs completed. The 1.7B addendum contains
  42 validation runs and 12 formal runs/48 stages; it passed replay-balance,
  Fisher-presence, finite-value, endpoint-recomputation, and provenance checks.
- Combined formal summary: `../runs/qwen_continual_scale/formal_summary_3scale.json`,
  SHA-256 `cf760bb8d74c3c48495d3a4076535ee356c1393cdd6c805fcb4351c36d194b3c`.
- 1.7B lambda selection: `../runs/qwen_continual_scale/selection_1.7b.json`,
  SHA-256 `08b656110764a575974c39bffa0d4f3e2f908a46921cd818bfef2df8ca556415`.
- 1.7B runner SHA-256:
  `48cccb6415751673d15f27094c7f3f12db23e886639f625b7a14f13f6dae7acf`.
- 1.7B protocol SHA-256:
  `d9a31a5eba5be58705b67af7cc093a19a466186245b8bfa7a7d8fb9bdb6eafbc`.
- Post-run model inventory verification passed for all three checkpoints.
