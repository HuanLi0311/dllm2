# R22 nominal-trace invalidation and stop ledger

R22 inherited R19's nominal rank-1 trace calculation and is excluded from
paper evidence for the same reason documented in
`report/r19_nominal_trace_invalidation.md`.  Only the already running seed
3407 subset (`gd`, `rank1_gd`, and `diag_gd`) may finish; seeds 3408--3409
must not be launched under R22.  All outputs and logs are retained.

The first `gd`, seed 3407 attempt failed during a simultaneous pre-run Python
standard-library import (`encodings.aliases`), before model loading and
before any JSON result.  The unchanged retry may finish.  The failed log has
SHA256 `8881db003ae3d614a1a1c031d9d0a8c15049a437662472e9d5761c8840b878f3`.

R24 is the independent corrected-trace replacement for the 1.14B grid.
