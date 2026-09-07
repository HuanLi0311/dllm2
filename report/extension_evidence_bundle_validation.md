# Continual-Fisher extension evidence validation

Validated 2026-09-07 after all frozen summaries passed independent
recomputation.  This bundle is separate from `release_evidence` and does not
change the primary release manifest.

## Included evidence

- R18: three full-parameter continual-task Fisher geometry results, the
  frozen summary, and the read-only absolute-diagnostics summary (5 files).
- R23: eighteen 219M corrected weighted-trace control results and the frozen
  summary (19 files).
- R24: nine 1.14B corrected weighted-trace control results and the frozen
  summary (10 files).

R19 and R22 nominal-trace outputs and R21 feasibility endpoints are excluded.
The source allowlist contains exactly 34 JSON artifacts.

## Build and strict verification

~~~bash
PYTHONNOUSERSITE=1 python experiments/build_review_bundle.py --self-check
PYTHONNOUSERSITE=1 python experiments/build_review_bundle.py \
  --submission-manifest runs/extension_evidence_manifest.json \
  --output-dir release_extension_evidence
PYTHONNOUSERSITE=1 python experiments/build_review_bundle.py \
  --verify release_extension_evidence/release_manifest.json \
  --require-internal
~~~

The self-check passed and rejected its tampered fixture.  Strict verification
reported 34 checked artifacts, 34 current internal sources, no missing or
changed source, a matching builder, and 35 bundle files including the release
manifest.

An independent expanded-bundle check then:

- decompressed all 34 payloads and parsed all 35 JSON files;
- compared the release provenance allowlist byte-for-byte after sorted JSON
  normalization with `runs/extension_evidence_manifest.json`;
- confirmed the exact family counts R18=5, R23=19, and R24=10; and
- found no occurrence of `/home/`, `air-node`, `JJ_Group`, `lih2511`, or an
  R19/R21/R22 path in the expanded bundle.

## Frozen identifiers

- Builder: `c4755066a82d15d7e1aecf51d0ea1c70d56ca36c164ee3c0e0d398753979215a`
- Source manifest: `005f7b729daf5f66adb8fa45ace046b0a67b28b54763e4e93186d1f936c89233`
- Release manifest: `b6ffb53f323c9383279039e6d7d6c7e1a4ba7d4cee8dce44b7be1102d11b1457`
- R18 frozen/absolute summaries: `73f17744a4c899033fabf222b66d4b3f433cf2dd8beb0180543a54bdcdd7ffae`,
  `fda008a3a2bb1e7f11eed95bdb548fb285e6dcc4b28f574e630b494638be73b4`
- R23 summary: `02d40caed13f0a45db86cc77c7082221f4010cd62380ac5650db5c30bd8b8ce1`
- R24 summary: `5330c146b2b5d28a2c2400b13d5b90bab193d4d5938ed0bdea5cf92413af8d7e`
