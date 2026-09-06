#!/usr/bin/env python3
"""Fail-closed aggregation for the frozen three-seed R18 geometry grid."""

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
        "mean": mean, "sem": sem, "values": values,
        "student_t_95_interval": [mean - 4.30265273 * sem, mean + 4.30265273 * sem],
    }


def summarize(run_root: Path) -> dict:
    contract_path = run_root / "contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["status"] != "frozen_before_pilot" or contract["seeds"] != list(SEEDS):
        raise ValueError("unexpected R18 contract")
    expected = [run_root / "formal" / f"s{seed}.json" for seed in SEEDS]
    actual = sorted((run_root / "formal").glob("*.json"))
    if actual != expected:
        raise ValueError(f"R18 grid mismatch: expected {expected}, found {actual}")
    runs = [json.loads(path.read_text()) for path in expected]
    input_reference = runs[0]["inputs"]
    for seed, path, run in zip(SEEDS, expected, runs):
        if run.get("status") != "ok" or run.get("experiment") != "r18_continual_fisher_geometry":
            raise ValueError(f"invalid R18 result: {path}")
        protocol = run["protocol"]
        if protocol["seed"] != seed or protocol["prompt_overlap_count"] != 0:
            raise ValueError(f"seed/prompt contract mismatch: {path}")
        if protocol["primary_parameter_set"] != "all_trainable_parameters":
            raise ValueError(f"non-full primary parameter set: {path}")
        sampling = run["fisher_sampling"]
        if sampling["calibration_examples"] != 40 or sampling["test_examples"] != 40:
            raise ValueError(f"sample-count mismatch: {path}")
        if sampling["calibration_repeat_loss_max_abs_difference"] != 0:
            raise ValueError(f"calibration replay mismatch: {path}")
        for rel, digest in run["dependencies"].items():
            if contract["sha256"].get(rel) != digest:
                raise ValueError(f"dependency hash outside contract: {rel}")
        if run["inputs"]["checkpoint_sha256"] != contract["sha256"]["checkpoint"]:
            raise ValueError(f"checkpoint mismatch: {path}")
        for key in (
            "train_rows_sha256", "calibration_rows_sha256", "test_rows_sha256",
            "calibration_prompts_sha256", "test_prompts_sha256",
        ):
            if run["inputs"][key] != input_reference[key]:
                raise ValueError(f"input row mismatch across seeds: {key}")
        checks = run["training_state"]["r16_anchor_checks"]
        if checks["training_current_mean_abs_difference"] > 1e-9:
            raise ValueError(f"training anchor mismatch: {path}")
        if checks["training_clip_fraction_abs_difference"] != 0:
            raise ValueError(f"clip anchor mismatch: {path}")
        for item in [run["full_parameter_geometry"], *run["slice_geometry"].values()]:
            rank1 = item["rank1_relative_frobenius_error"]
            diagonal = item["diagonal_relative_frobenius_error"]
            score = item["heldout_score_log_diag_over_rank1"]
            if not all(math.isfinite(value) for value in (rank1, diagonal, score)):
                raise ValueError(f"non-finite geometry: {path}")
            if not math.isclose(score, math.log(diagonal / rank1), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"score recomputation failed: {path}")
            if item["heldout_best_rank1_error"] > item["heldout_direction_oracle_error"] + 1e-8:
                raise ValueError(f"best-rank1 oracle ordering failed: {path}")
            if item["heldout_direction_oracle_error"] > rank1 + 1e-8:
                raise ValueError(f"direction oracle ordering failed: {path}")

    probe_names = list(runs[0]["slice_geometry"])
    if any(list(run["slice_geometry"]) != probe_names for run in runs):
        raise ValueError("slice ordering differs across seeds")
    full_scores = [run["full_parameter_geometry"]["heldout_score_log_diag_over_rank1"] for run in runs]
    result = {
        "schema_version": 1,
        "status": "ok",
        "protocol": "r18_continual_fisher_geometry_v1",
        "run_count": len(runs),
        "contract_sha256": _sha256(contract_path),
        "run_sha256": {path.name: _sha256(path) for path in expected},
        "primary_full_parameter_score": _stats(full_scores),
        "primary_rank1_seed_wins": sum(value > 0 for value in full_scores),
        "full_parameter_errors": {
            "rank1": _stats([run["full_parameter_geometry"]["rank1_relative_frobenius_error"] for run in runs]),
            "diagonal": _stats([run["full_parameter_geometry"]["diagonal_relative_frobenius_error"] for run in runs]),
        },
        "slices": {
            name: {
                "score": _stats([run["slice_geometry"][name]["heldout_score_log_diag_over_rank1"] for run in runs]),
                "rank1_seed_wins": sum(run["slice_geometry"][name]["heldout_score_log_diag_over_rank1"] > 0 for run in runs),
            }
            for name in probe_names
        },
        "per_seed": [
            {
                "seed": seed,
                "full_parameter_score": run["full_parameter_geometry"]["heldout_score_log_diag_over_rank1"],
                "mean_slice_score": run["summary"]["mean_slice_score"],
                "rank1_slice_wins": run["summary"]["rank1_slice_wins"],
            }
            for seed, run in zip(SEEDS, runs)
        ],
        "audit": {
            "prompt_overlap_count": 0,
            "calibration_examples_per_seed": 40,
            "test_examples_per_seed": 40,
            "source_hashes_match_contract": True,
            "scores_recomputed_from_errors": True,
            "oracle_ordering_verified": True,
            "r16_task_a_anchors_verified": True,
        },
    }
    return result


def _self_check() -> None:
    values = [1.0, 2.0, 3.0]
    stats = _stats(values)
    assert stats["mean"] == 2.0 and math.isclose(stats["sem"], 1 / math.sqrt(3))
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
