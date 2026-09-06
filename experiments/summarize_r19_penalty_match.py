#!/usr/bin/env python3
"""Fail-closed aggregation for the frozen 18-cell R19 control matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gd", "rank1_gd", "diag_gd")
CLIPS = (1.0, 1_000_000.0)
SEEDS = (3407, 3408, 3409)
ENDPOINTS = ("final_average_loss", "past_task_forgetting", "final_task_loss")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clip_name(value: float) -> str:
    return "1" if value == 1.0 else "1000000"


def _stats(values) -> dict:
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "values": values,
        "wins_below_zero": sum(value < 0 for value in values),
    }


def summarize(run_root: Path) -> dict:
    contract_path = run_root / "contract.json"
    contract = json.loads(contract_path.read_text())
    grid = contract["grid"]
    if grid != {"methods": list(METHODS), "b_clip": list(CLIPS), "seeds": list(SEEDS), "cell_count": 18}:
        raise ValueError("unexpected R19 contract grid")
    expected = {
        (method, clip, seed): run_root / "formal" / f"{method}_clip{_clip_name(clip)}_s{seed}.json"
        for method in METHODS for clip in CLIPS for seed in SEEDS
    }
    actual = set((run_root / "formal").glob("*.json"))
    if actual != set(expected.values()):
        raise ValueError(f"R19 grid mismatch: missing={set(expected.values()) - actual}, extra={actual - set(expected.values())}")
    runs = {cell: json.loads(path.read_text()) for cell, path in expected.items()}
    input_reference = next(iter(runs.values()))["inputs"]
    for (method, clip, seed), run in runs.items():
        path = expected[(method, clip, seed)]
        if run.get("status") != "ok" or run.get("experiment") != "r19_weighted_trace_match":
            raise ValueError(f"invalid R19 result: {path}")
        protocol = run["protocol"]
        if (protocol["method"], protocol["task_b_clip"], protocol["seed"]) != (method, clip, seed):
            raise ValueError(f"cell metadata mismatch: {path}")
        if protocol["replay_fact_counts"] != {str(fact): 16 for fact in range(8, 12)}:
            raise ValueError(f"unbalanced replay: {path}")
        for rel, digest in run["dependencies"].items():
            if contract["sha256"].get(rel) != digest:
                raise ValueError(f"dependency outside contract: {path}: {rel}")
        if run["inputs"]["checkpoint_sha256"] != contract["sha256"]["checkpoint"]:
            raise ValueError(f"checkpoint mismatch: {path}")
        if run["inputs"] != input_reference:
            raise ValueError(f"input artifacts differ across cells: {path}")
        anchors = run["task_a_anchor_checks"]
        if anchors["training_current_mean_abs_difference"] > 1e-9 or anchors["training_clip_fraction_abs_difference"] != 0:
            raise ValueError(f"R16 training anchor mismatch: {path}")
        if anchors["rank1_coefficient_relative_difference"] > 1e-6 or anchors["diagonal_trace_relative_difference"] > 1e-6:
            raise ValueError(f"R16 Fisher anchor mismatch: {path}")
        match = run["stiffness_match"]
        target = match["weighted_trace_target"]
        if not math.isclose(match["weighted_trace_checks"]["rank1"], target, rel_tol=1e-9):
            raise ValueError(f"rank-1 trace match failed: {path}")
        if not math.isclose(match["weighted_trace_checks"]["diagonal"], target, rel_tol=1e-9):
            raise ValueError(f"diagonal trace match failed: {path}")
        expected_lambda = 0.0 if method == "gd" else match["matched_lambdas"]["rank1" if method == "rank1_gd" else "diagonal"]
        if not math.isclose(match["effective_lambda"], expected_lambda, rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"effective lambda mismatch: {path}")
        values = [*run["summary"].values(), *run["stages"][1]["training"].values()]
        if not all(not isinstance(value, float) or math.isfinite(value) for value in values):
            raise ValueError(f"non-finite endpoint: {path}")

    aggregate = {}
    for clip in CLIPS:
        aggregate[str(clip)] = {
            method: {
                endpoint: _stats([runs[(method, clip, seed)]["summary"][endpoint] for seed in SEEDS])
                for endpoint in ENDPOINTS
            }
            for method in METHODS
        }
        for method in METHODS:
            aggregate[str(clip)][method]["clip_fraction"] = _stats([
                runs[(method, clip, seed)]["stages"][1]["training"]["clip_fraction"] for seed in SEEDS
            ])
            aggregate[str(clip)][method]["ewc_loss_mean"] = _stats([
                runs[(method, clip, seed)]["stages"][1]["training"]["ewc_loss_mean"] for seed in SEEDS
            ])

    paired = {}
    contrasts = (
        ("rank1_gd_minus_gd", "rank1_gd", "gd"),
        ("diag_gd_minus_gd", "diag_gd", "gd"),
        ("rank1_gd_minus_diag_gd", "rank1_gd", "diag_gd"),
    )
    for clip in CLIPS:
        paired[str(clip)] = {
            label: {
                endpoint: _stats([
                    runs[(left, clip, seed)]["summary"][endpoint]
                    - runs[(right, clip, seed)]["summary"][endpoint]
                    for seed in SEEDS
                ])
                for endpoint in ENDPOINTS
            }
            for label, left, right in contrasts
        }
    clipping_sensitivity = {
        method: {
            endpoint: _stats([
                runs[(method, 1_000_000.0, seed)]["summary"][endpoint]
                - runs[(method, 1.0, seed)]["summary"][endpoint]
                for seed in SEEDS
            ])
            for endpoint in ENDPOINTS
        }
        for method in METHODS
    }
    return {
        "schema_version": 1,
        "status": "ok",
        "protocol": "r19_weighted_trace_match_v1",
        "run_count": len(runs),
        "contract_sha256": _sha256(contract_path),
        "run_sha256": {path.name: _sha256(path) for path in expected.values()},
        "aggregate": aggregate,
        "paired": paired,
        "clipping_sensitivity_unclipped_minus_clip1": clipping_sensitivity,
        "matched_lambdas": {
            str(seed): runs[("rank1_gd", 1.0, seed)]["stiffness_match"]["matched_lambdas"]
            for seed in SEEDS
        },
        "audit": {
            "complete_cells": 18,
            "source_hashes_match_contract": True,
            "input_artifacts_identical": True,
            "r16_task_a_anchors_verified": True,
            "balanced_replay_verified": True,
            "weighted_trace_equality_verified": True,
            "all_endpoints_finite": True,
        },
    }


def _self_check() -> None:
    stats = _stats([-1.0, 0.0, 2.0])
    assert stats["mean"] == 1 / 3 and stats["wins_below_zero"] == 1
    assert _clip_name(1.0) == "1" and _clip_name(1_000_000.0) == "1000000"
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/r19_penalty_match")
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
