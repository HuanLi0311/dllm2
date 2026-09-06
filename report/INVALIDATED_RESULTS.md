# Evidence invalidation ledger

**Effective 2026-09-05. This ledger overrides every older report in this directory.**

## Excluded artifacts

| Artifact family | Status | Reason |
|---|---|---|
| `runs/r00_*` through `runs/r04_*`, `r01_aggregate_170.json`, `r02_aggregate_1028.json`, `all_geometry_aggregate.json`, and `geometry_controls.json` | Invalid | The mean-gradient rank-1 Frobenius norm used `c‖μ‖⁴` instead of `c²‖μ‖⁴`. This could make a constrained rank-1 fit appear better than the oracle rank-1 fit. The reported `27/28` result and every MCR gate derived from it are withdrawn. |
| `runs/r05_real_geometry/*` and `runs/r06_geometry_controls/*` | Audit only | The algebra is corrected and direct matrix checks pass, but the surrogate is fitted and scored on the same small gradient sample. These runs illustrate finite-sample bias; they are not primary evidence. |
| `runs/r09_split_controls/*` | Superseded | MT-Bench/reversal used `64|64`, but their original GSM8K comparator came from an r08 `64|128` row.  The original one-target run also used a different held-out record interval from r08, so the former `0.140` “paired” effect is withdrawn. |
| `runs/r11_paired_controls/*` | Superseded audit pilot | These corrected the GSM8K sample sizes to `64|64`, but did not retain the explicit record/mask hashes and sufficient statistics required by the final cross-group contract.  Audited r12 reruns replace them. |
| `runs/continual_pilot/*` and `runs/continual_formal/*` | Excluded from submission claims | The compared methods did not all use a matched effective batch and regularization exposure. The short synthetic pilot also lacks a held-out Fisher calibration protocol. It cannot establish an EWC or MCR advantage. |
| All `runs/r14_continual_transfer/*` and `runs/r15_multitask_transfer/*` | Invalid | Answer corruption forced the first answer token to be masked whenever an independent Bernoulli mask was empty, changing that token's marginal probability while retaining the `1/t` importance weight. Current/replay minibatches also shared one Python RNG, so GD consumed extra draws and changed current-task exposure. R16 restores independent Bernoulli masks (including empty masks), uses separate RNGs, and reruns every reported continual result. |
| `runs/r15_multitask_transfer/forward/*` (outside `final/`) | Invalid pilot | Evaluation mask seeds depended on training stage, so learned-time and final losses were not exactly paired. R16 uses task-identity masks. |
| `runs/r15_multitask_transfer/invalid_unbalanced_replay_20260829/*` | Invalid pilot | GD selected the first 64 training rows from data grouped by fact, producing replay counts 30/30/4/0. Because Fisher covered all four facts, this could create an artificial EWC advantage on facts omitted by replay. R16 uses 16/16/16/16 replay and audits per-fact hashes. |
| `paper/figures_insample_audit/*` | Appendix audit only | These figures visualize corrected same-sample measurements, not generalization to an independent Fisher. |

The source release omits these invalid raw files; this ledger preserves their
failure history. They must not be pooled with or cited as support for the
submission's empirical claims.

## Replacement evidence

The submission uses only:

- `runs/r08_split_primary/gsm_170_s*/benchmark.json`: five probe split/mask seeds, three layers, five fixed mask probabilities plus an auxiliary uniform-$p$, within-$p$-nonempty diagnostic, three calibration sizes, and 128 independent test examples;
- `runs/r08_split_primary/gsm_1028_s*/benchmark.json`: three probe split/mask seeds under the matched protocol with calibration sizes 16/32/64 and 64 independent test examples;
- twelve audited envelopes for probe split/mask seeds 0--2 under `runs/r12_audited_controls/` for
  matched GSM8K all-target/one-target, MT-Bench, and reversal comparisons;
- successful envelopes for probe split/mask seeds 0--2 under `runs/r10_attention_split/` for the 589,824-parameter dense attention-projection control;
- exact selected-row backend replays for both checkpoint scales under
  `runs/backend_replay/`, plus a complete installed-package and backend-source
  lock;
- `runs/r08_split_primary/isotropic_null_full.json`: 200-repetition isotropic null at the two measured parameter-slice dimensions; and
- `runs/r16_native_mask/validation/*` and `runs/r16_native_mask/confirmation/*`: three-seed lambda selection and a separate-fact confirmation using the native masking estimator;
- the exact 33-file matrix under `runs/r16_native_mask/final/` plus the 24-file core-method matrix on previously unused facts 24--29 under `runs/r16_native_mask/fresh/`; `experiments/summarize_dllm_rank1_multitask.py` has verified both complete two-order protocols and emitted their R16 summaries; and
- the analytic scaled-identity Gaussian counterexample checked by `experiments/simulate_fisher_null.py --self-check`.

Every primary geometry row is split-sample. Surrogate coefficients and
directions are estimated only from calibration gradients; all reported
reconstruction errors use a disjoint test set.  The r12 comparison contract
additionally verifies identical objective-pair record and mask hashes, matched
`64|64` dataset-pipeline sample/mask protocols, expected task IDs and loss
modes, both probe hashes, and recomputes rank-1/diagonal errors from retained
sufficient statistics.  The submission manifest fails closed on these checks,
source/checkpoint/data hashes, complete Cartesian grids, seed matching, finite
values, held-out oracle ordering, and the execution-backend replay audit.  The
three data pipelines use different tokenization/packing and are not evidence
for a corpus-content-only effect.

## Correct invariant

For a surrogate `R=c μμᵀ`,

`‖R‖²_F = c²‖μ‖⁴`,

not `c‖μ‖⁴`. The runnable probe self-check materializes both empirical Fisher
matrices and verifies the mean-rank-1 and diagonal identities plus oracle
ordering against the fitted mean direction.  The submission manifest separately
enforces oracle ordering against both fitted rank-1 directions in every evidence
row.
