# Fresh-design capacity audit

Audited on 2026-09-07 before proposing another fresh-fact run.

The official reversal directory contains exactly 900 training and 300 test
rows per direction.  `fact_rows` partitions these in source order into 30 fact
groups, with 30 training and 10 test templates per group.  Existing frozen
studies have exposed the complete index range:

| Use | Fact groups |
|---|---|
| R16 validation | 0--3 |
| R16 confirmation | 4--7 |
| R16 four-task main matrix | 8--23 |
| R16 fresh sensitivity matrix | 24--29 |

Their union is 0--29.  Consequently, this artifact cannot supply the 16
previously unexposed fact groups required for a four-task, four-facts-per-task
replication matched to the main matrix.  Relabeling old facts or changing only
task order would not make them fresh.

The current three-task facts-24--29 matrix should therefore be described as a
**fresh-fact sensitivity extension**, not a design-matched replication.  A
genuinely matched fresh replication requires a separately constructed and
pre-frozen dataset with at least 16 new fact identities, 30 train and 10 test
templates per identity in both directions, prompt-disjoint train/test files,
and the same four-task alternation, replay balance, optimizer, masks, methods,
orders, and seeds as R16.  Dataset construction and hashes must be frozen
before any model run; until that exists, this gap remains open rather than
being filled by a weaker relabeling.
