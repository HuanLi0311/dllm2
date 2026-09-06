#!/usr/bin/env python3
"""Fail-closed aggregation for the frozen nine-cell R22 scale extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gd", "rank1_gd", "diag_gd")
SEEDS = (3407, 3408, 3409)
ENDPOINTS = ("final_average_loss", "past_task_forgetting", "final_task_loss")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values) -> dict:
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "min": min(values), "max": max(values), "values": values,
        "wins_below_zero": sum(value < 0 for value in values),
    }


def _without_timing(values: dict) -> dict:
    return {key: value for key, value in values.items() if key != "wall_time_seconds"}


def _recomputed_endpoints(run: dict) -> dict:
    task_a, task_b = run["protocol"]["task_sequence"]
    learned_a = run["stages"][0]["metrics"][task_a]["loss"]
    final = run["stages"][1]["metrics"]
    return {
        "final_average_loss": (final[task_a]["loss"] + final[task_b]["loss"]) / 2,
        "past_task_forgetting": final[task_a]["loss"] - learned_a,
        "final_task_loss": final[task_b]["loss"],
    }


def summarize(run_root: Path) -> dict:
    contract_path = run_root / "contract.json"
    contract = json.loads(contract_path.read_text())
    if contract["status"] != "frozen_before_seed3407_subset":
        raise ValueError("unexpected R22 contract status")
    if contract["grid"] != {"methods": list(METHODS), "seeds": list(SEEDS), "cell_count": 9}:
        raise ValueError("unexpected R22 grid")
    expected = {
        (method, seed): run_root / "formal" / f"{method}_s{seed}.json"
        for method in METHODS for seed in SEEDS
    }
    actual = set((run_root / "formal").glob("*.json"))
    if actual != set(expected.values()):
        raise ValueError(f"R22 grid mismatch: missing={set(expected.values()) - actual}, extra={actual - set(expected.values())}")
    runs = {cell: json.loads(path.read_text()) for cell, path in expected.items()}
    input_reference = next(iter(runs.values()))["inputs"]
    for (method, seed), run in runs.items():
        path = expected[(method, seed)]
        if run.get("status") != "ok" or run.get("experiment") != "r22_scale1028_weighted_trace_match":
            raise ValueError(f"invalid R22 result: {path}")
        protocol = run["protocol"]
        if (protocol["method"], protocol["seed"]) != (method, seed):
            raise ValueError(f"cell metadata mismatch: {path}")
        if protocol["parameter_count"] != 1_142_367_744 or not protocol["full_parameter_training"]:
            raise ValueError(f"non-full parameter cell: {path}")
        if protocol["fisher_examples"] != 40 or protocol["replay_fact_counts"] != {str(fact): 16 for fact in range(8, 12)}:
            raise ValueError(f"Fisher/replay count mismatch: {path}")
        for rel, digest in run["dependencies"].items():
            if contract["sha256"].get(rel) != digest:
                raise ValueError(f"dependency outside contract: {path}: {rel}")
        if run["inputs"]["checkpoint_sha256"] != contract["sha256"]["checkpoint"]:
            raise ValueError(f"checkpoint mismatch: {path}")
        if run["inputs"] != input_reference:
            raise ValueError(f"input artifacts differ across cells: {path}")
        if run["fisher"]["calibration_examples"] != 40 or run["fisher"]["parameter_count"] != 1_142_367_744:
            raise ValueError(f"Fisher metadata mismatch: {path}")
        if run["fisher"]["repeat_loss_max_abs_difference"] != 0:
            raise ValueError(f"Fisher mask replay mismatch: {path}")
        match = run["stiffness_match"]
        target = match["weighted_trace_target"]
        for structure in ("rank1", "diagonal"):
            if not math.isclose(match["weighted_trace_checks"][structure], target, rel_tol=1e-12):
                raise ValueError(f"weighted-trace mismatch: {path}: {structure}")
        expected_lambda = 0.0 if method == "gd" else match["matched_lambdas"]["rank1" if method == "rank1_gd" else "diagonal"]
        if not math.isclose(match["effective_lambda"], expected_lambda, rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"effective lambda mismatch: {path}")
        recomputed = _recomputed_endpoints(run)
        for endpoint, expected_value in recomputed.items():
            if not math.isclose(run["summary"][endpoint], expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"summary recomputation failed: {path}: {endpoint}")
        values = [*run["summary"].values(), *run["stages"][1]["training"].values()]
        if not all(not isinstance(value, float) or math.isfinite(value) for value in values):
            raise ValueError(f"non-finite output: {path}")

    for seed in SEEDS:
        reference = runs[("gd", seed)]
        for method in METHODS:
            run = runs[(method, seed)]
            if _without_timing(run["stages"][0]["training"]) != _without_timing(reference["stages"][0]["training"]):
                raise ValueError(f"Task-A training differs within seed {seed}: {method}")
            if run["stages"][0]["metrics"] != reference["stages"][0]["metrics"]:
                raise ValueError(f"Task-A metrics differ within seed {seed}: {method}")
            if run["fisher"] != reference["fisher"]:
                raise ValueError(f"Fisher differs within seed {seed}: {method}")
            if run["replay"]["generated_rows_sha256"] != reference["replay"]["generated_rows_sha256"]:
                raise ValueError(f"replay differs within seed {seed}: {method}")
            if run["stiffness_match"]["matched_lambdas"] != reference["stiffness_match"]["matched_lambdas"]:
                raise ValueError(f"matched lambdas differ within seed {seed}: {method}")

    aggregate = {
        method: {
            **{endpoint: _stats([runs[(method, seed)]["summary"][endpoint] for seed in SEEDS]) for endpoint in ENDPOINTS},
            "clip_fraction": _stats([runs[(method, seed)]["stages"][1]["training"]["clip_fraction"] for seed in SEEDS]),
            "preclip_gradient_norm_max": _stats([runs[(method, seed)]["stages"][1]["training"]["gradient_norm_max"] for seed in SEEDS]),
            "weighted_ewc_mean": _stats([runs[(method, seed)]["stages"][1]["training"]["ewc_loss_mean"] for seed in SEEDS]),
            "weighted_ewc_max": _stats([
                runs[(method, seed)]["stages"][1]["training"]["penalty_max"]
                * runs[(method, seed)]["stiffness_match"]["effective_lambda"] for seed in SEEDS
            ]),
            "wall_time_seconds": _stats([runs[(method, seed)]["wall_time_seconds"] for seed in SEEDS]),
        }
        for method in METHODS
    }
    contrasts = (
        ("rank1_gd_minus_gd", "rank1_gd", "gd"),
        ("diag_gd_minus_gd", "diag_gd", "gd"),
        ("rank1_gd_minus_diag_gd", "rank1_gd", "diag_gd"),
    )
    paired = {
        label: {
            endpoint: _stats([
                runs[(left, seed)]["summary"][endpoint] - runs[(right, seed)]["summary"][endpoint]
                for seed in SEEDS
            ])
            for endpoint in ENDPOINTS
        }
        for label, left, right in contrasts
    }
    return {
        "schema_version": 1,
        "status": "ok",
        "protocol": "r22_scale1028_weighted_trace_match_v1",
        "run_count": len(runs),
        "contract_sha256": _sha256(contract_path),
        "run_sha256": {path.name: _sha256(path) for path in expected.values()},
        "aggregate": aggregate,
        "paired": paired,
        "matched_lambdas": {
            str(seed): runs[("rank1_gd", seed)]["stiffness_match"]["matched_lambdas"]
            for seed in SEEDS
        },
        "audit": {
            "complete_cells": 9,
            "source_and_input_hashes_verified": True,
            "full_parameter_fisher_verified": True,
            "balanced_replay_verified": True,
            "weighted_trace_equality_verified": True,
            "within_seed_task_a_fisher_replay_identity_verified": True,
            "summary_endpoints_recomputed_from_stage_metrics": True,
            "all_outputs_finite": True,
        },
    }


def _self_check() -> None:
    stats = _stats([-1.0, 0.0, 2.0])
    assert stats["mean"] == 1 / 3 and stats["min"] == -1 and stats["max"] == 2
    toy = {
        "protocol": {"task_sequence": ["a", "b"]},
        "stages": [
            {"metrics": {"a": {"loss": 1.0}}},
            {"metrics": {"a": {"loss": 2.5}, "b": {"loss": 0.5}}},
        ],
    }
    assert _recomputed_endpoints(toy) == {
        "final_average_loss": 1.5, "past_task_forgetting": 1.5, "final_task_loss": 0.5,
    }
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/r22_scale1028_penalty_match")
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
