# R23/R24 pilot mechanical validation

Both frozen pilot subsets completed successfully before expansion.  Endpoint
signs were inspected only after the registered mechanical conditions and did
not enter the expansion decision.

## R23 seed 3407

- All six method-by-clip cells exited 0 and wrote finite status-`ok` JSONs.
- Every cell used 219,050,496 trainable parameters, 40 Fisher rows, and 64
  replay rows balanced 16 per fact.  Contract, dependency, input, and task
  artifact content hashes matched.
- All four R16 Task-A training/Fisher anchor differences were exactly zero.
  Task-A training (excluding timing), Task-A metrics, Fisher statistics,
  generated replay hash, and corrected multipliers were identical across all
  six cells.
- The stored direction norm was `1.0296175364215259`; the independent R18
  operational-direction norm was `1.029617536421548` (absolute difference
  `2.22e-14`).
- The exact stored diagonal trace was `0.002578950639335681`; the original
  float32-reduced R16 statistic was `0.002578950487077236`.  The corrected
  Rank-1 multiplier was `2824.1413001646647`, and both implemented weighted
  traces were exactly `2.578950639335681` at recorded precision.
- A separate comparison against the released R16 forward run found exact
  Task-B training and stage-2 metrics for GD and for Diagonal+GD at clip 1,
  excluding wall time.

## R24 seed 3407

- All three method cells exited 0 and wrote finite status-`ok` JSONs.
- Every cell used 1,142,367,744 trainable parameters, 40 Fisher rows, and 64
  replay rows balanced 16 per fact.  Contract, dependency, input, and task
  artifact content hashes matched.
- Task-A training (excluding timing), Task-A metrics, Fisher statistics,
  generated replay hash, and corrected multipliers were identical across all
  three cells.
- The stored direction norm was `1.077635312758915`; the exact stored
  diagonal trace was `37.14897578636429`, versus the original float32-reduced
  value `37.14897537231445`.  The corrected Rank-1 multiplier was
  `743.6057789549045`, and both implemented weighted traces were exactly
  `37148.97578636429` at recorded precision.

The frozen completion rules therefore authorize every remaining R23 and R24
cell unchanged, irrespective of its eventual endpoint direction.
