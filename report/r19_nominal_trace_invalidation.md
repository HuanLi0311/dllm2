# R19 nominal-trace invalidation and stop ledger

## Scope decision

R19 is retained as an auditable failed control and is excluded from paper
evidence.  Its rank-1 multiplier used the nominal expression
`lambda_R * alpha`, which is the trace only if the stored direction has
unit Euclidean norm.  The actual R16 float32 direction is not exactly unit
length: R18 seed 3407 independently measured `||u32|| =
1.029617536421548`, so the implemented rank-1 trace was
`lambda_R * alpha * ||u32||^2`, about 6.011% above the registered target.
This is a mechanical mismatch, irrespective of the observed endpoints.
The old diagonal target also used a single long float32 `D32.sum()` rather
than the mathematical trace of the stored tensor.  That reduction error is
much smaller, but it is still incompatible with an exact implemented-trace
claim.

The corrected controls are assigned new run families R23 (219M) and R24
(1.14B).  They compute both `||u32||^2` and `trace(D32)` by fixed-chunk
float64 reductions of the stored tensors and set
`lambda_R = 1000 * trace(D32) / (alpha * ||u32||^2)`; the original float32
diagonal statistic is retained only for the R16 anchor and its discrepancy
is recorded.

## Stop and failure ledger

- The two pilot cells and every cell already completed or already running at
  discovery are preserved unchanged under `runs/r19_penalty_match/`.
- Seven queue launchers (PIDs 716416--716422) were terminated so they could
  not start further nominal-trace cells.  Their already running child
  processes were deliberately left to complete.
- A seventh child, PID 728412 (`rank1_gd`, clip `1000000`, seed 3408), began
  in the launcher-termination race and was also left to complete.  No later
  cell is to be launched from these queues.
- The pending retry watcher PID 720149 was stopped before it loaded a model
  or wrote a result, then terminated; it therefore cannot launch the old
  `diag_gd`, clip `1`, seed 3409 retry.
- The first `diag_gd`, clip `1`, seed 3409 attempt remains in the ledger: it
  failed during a concurrent pre-run Python standard-library import, before
  model loading and before any JSON result.  Its original log and exit file
  are retained.
- The stopped family ended with 14 status-`ok` JSON files.  The four cells
  never run are Diagonal+GD at seeds 3408--3409 under each of clips 1 and
  `1e6`.  Several already-running Python children became orphans when their
  queue launchers were terminated and consequently have no queue-written
  exit file; their logs and JSONs remain intact.  This incomplete family is
  not summarized as a registered 18-cell grid.

No R19 result, positive or negative, is eligible for reporting as a
trace-matched comparison.
