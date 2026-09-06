#!/usr/bin/env python3
"""Read-only absolute diagnostics supplement for the frozen R18 summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3407, 3408, 3409)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values) -> dict:
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean, "sem": sem, "min": min(values), "max": max(values),
        "values": values,
        "student_t_95_interval": [mean - 4.30265273 * sem, mean + 4.30265273 * sem],
    }


def summarize(run_root: Path) -> dict:
    frozen_path = run_root / "summary.json"
    contract_path = run_root / "contract.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("status") != "ok" or frozen.get("protocol") != "r18_continual_fisher_geometry_v3":
        raise ValueError("missing valid frozen R18 summary")
    if frozen.get("contract_sha256") != _sha256(contract_path):
        raise ValueError("frozen R18 summary does not match the contract")
    rows = []
    for seed in SEEDS:
        path = run_root / "formal" / f"s{seed}.json"
        if frozen["run_sha256"].get(path.name) != _sha256(path):
            raise ValueError(f"raw R18 output differs from frozen summary: {path.name}")
        run = json.loads(path.read_text())
        if run.get("status") != "ok" or run["protocol"]["seed"] != seed:
            raise ValueError(f"invalid R18 output: {path.name}")
        if run["protocol"]["prompt_overlap_count"] != 0:
            raise ValueError(f"calibration/test prompt overlap: {path.name}")
        sampling = run["fisher_sampling"]
        geometry = run["full_parameter_geometry"]
        calibration_loss = float(sampling["calibration_loss_mean"])
        test_loss = float(sampling["test_loss_mean"])
        if not calibration_loss > 0 or not test_loss > 0:
            raise ValueError(f"invalid calibration/test loss: {path.name}")
        rank1 = float(geometry["rank1_relative_frobenius_error"])
        diagonal = float(geometry["diagonal_relative_frobenius_error"])
        score = float(geometry["heldout_score_log_diag_over_rank1"])
        direction_oracle = float(geometry["heldout_direction_oracle_error"])
        best_rank1 = float(geometry["heldout_best_rank1_error"])
        if not math.isclose(score, math.log(diagonal / rank1), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"score recomputation failed: {path.name}")
        if best_rank1 > direction_oracle + 1e-8 or direction_oracle > rank1 + 1e-8:
            raise ValueError(f"oracle ordering failed: {path.name}")
        row = {
            "seed": seed,
            "calibration_loss_mean": calibration_loss,
            "test_loss_mean": test_loss,
            "test_over_calibration_loss_ratio": test_loss / calibration_loss,
            "rank1_relative_frobenius_error": rank1,
            "diagonal_relative_frobenius_error": diagonal,
            "heldout_score_log_diag_over_rank1": score,
            "heldout_direction_oracle_error": direction_oracle,
            "heldout_best_rank1_error": best_rank1,
            "rank1_direction_norm": float(geometry["rank1_direction_norm"]),
            "rank1_coefficient": float(geometry["rank1_coefficient"]),
            "diagonal_trace_float32_reported": float(geometry["diagonal_trace"]),
            "test_fisher_frobenius": float(geometry["test_fisher_frobenius"]),
        }
        if not all(math.isfinite(value) for key, value in row.items() if key != "seed"):
            raise ValueError(f"non-finite diagnostic: {path.name}")
        rows.append(row)
    metric_names = tuple(key for key in rows[0] if key != "seed")
    return {
        "schema_version": 1,
        "status": "ok",
        "role": "posthoc_read_only_absolute_diagnostics",
        "source_protocol": frozen["protocol"],
        "contract_sha256": frozen["contract_sha256"],
        "frozen_summary_sha256": _sha256(frozen_path),
        "run_sha256": frozen["run_sha256"],
        "per_seed": rows,
        "aggregate": {name: _stats([row[name] for row in rows]) for name in metric_names},
        "audit": {
            "raw_hashes_match_frozen_summary": True,
            "prompt_overlap_count": 0,
            "scores_recomputed_from_absolute_errors": True,
            "oracle_ordering_verified": True,
            "no_runner_or_protocol_change": True,
        },
    }


def _self_check() -> None:
    stats = _stats([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0 and math.isclose(stats["sem"], 1 / math.sqrt(3))
    assert math.isclose(math.log(0.8 / 0.9), -0.11778303565638351)
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/r18_continual_fisher_geometry")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    result = summarize(args.run_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
