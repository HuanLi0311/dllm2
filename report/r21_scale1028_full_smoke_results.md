# R21 SMDM-1.14B full-parameter feasibility smoke result

R21 is mechanical evidence only and is excluded from paper results.  The
single frozen cell completed on `air-node-02` GPU 0 with exit code 0.

- Existing runner self-check: passed under `PYTHONNOUSERSITE=1`.
- Runner wall time: 182.829 seconds; outer elapsed time: 200.33 seconds.
- Peak sampled GPU memory: 21,335 MiB (two-second polling).
- Maximum host RSS reported by GNU time: 37,705,040 KiB (35.96 GiB).
- The raw JSON has `status=ok`, the frozen runner/checkpoint/dependency hashes,
  two 100-step stages, a four-example full-parameter Fisher with parameter
  count 1,142,367,744, and four balanced replay examples.
- No endpoint value was inspected for a paper claim or selection decision.

Artifacts:

- Result: `../runs/r21_scale1028_full_smoke/formal/s3407_rank1_gd.json`
  (`e6f2b7d183e6406a5d4c6b9920b97c6fc2db39a634f4e166a58ce1d0c162c1cc`).
- Log: `../runs/r21_scale1028_full_smoke/logs/s3407_rank1_gd.log`
  (`e913a362a461af13f5b0fb2baf77237b7a33f5acb57045adc7c8618a700409e1`).
- GPU samples:
  `../runs/r21_scale1028_full_smoke/logs/s3407_rank1_gd_gpu_memory.csv`
  (`a61c31941cb532db204e8db2e5145d9890dd795367315f997b600ad0c1718a1b`).
- Frozen contract: `../runs/r21_scale1028_full_smoke/contract.json`.

The smoke closes only the 40GB feasibility question.  A full-scale result
requires a separately frozen runner and complete multi-seed grid.
