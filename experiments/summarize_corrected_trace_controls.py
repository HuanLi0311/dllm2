#!/usr/bin/env python3
"""Fail-closed independent aggregation for complete R23 or R24 grids."""

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
CONFIGS = {
    "r23": {
        "run_dir": "r23_corrected_trace",
        "experiment": "r23_corrected_trace_match",
        "parameter_count": 219_050_496,
        "clips": (1.0, 1_000_000.0),
        "cell_count": 18,
        "r16_anchors": True,
        "protocol": "report/r23_corrected_trace_protocol.md",
    },
    "r24": {
        "run_dir": "r24_scale1028_corrected_trace",
        "experiment": "r24_scale1028_corrected_trace_match",
        "parameter_count": 1_142_367_744,
        "clips": (1.0,),
        "cell_count": 9,
        "r16_anchors": False,
        "protocol": "report/r24_scale1028_corrected_trace_protocol.md",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clip_name(value: float) -> str:
    return "1" if value == 1.0 else "1000000"


def _expected_path(run_root: Path, config: dict, method: str, clip: float, seed: int) -> Path:
    name = f"{method}_clip{_clip_name(clip)}_s{seed}.json" if len(config["clips"]) > 1 else f"{method}_s{seed}.json"
    return run_root / "formal" / name


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


def _assert_finite(value, path: str = "run") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _validate_trace(run: dict, label: str) -> None:
    match = run["stiffness_match"]
    alpha = float(run["fisher"]["rank1_coefficient"])
    norm_sq = float(match["direction_norm_sq_float64_chunked"])
    diagonal_trace = float(match["diagonal_trace_float64_chunked"])
    reported_trace = float(match["diagonal_trace_float32_reported"])
    if not all(math.isfinite(value) and value > 0 for value in (alpha, norm_sq, diagonal_trace)):
        raise ValueError(f"non-positive corrected-trace input: {label}")
    expected_rank1_trace = alpha * norm_sq
    expected_target = 1_000.0 * diagonal_trace
    expected_rank1_lambda = expected_target / expected_rank1_trace
    if not math.isclose(match["direction_norm_float64_chunked"], math.sqrt(norm_sq), rel_tol=1e-12):
        raise ValueError(f"direction norm/square mismatch: {label}")
    if not math.isclose(match["rank1_unweighted_trace"], expected_rank1_trace, rel_tol=1e-12):
        raise ValueError(f"rank-1 unweighted trace mismatch: {label}")
    if not math.isclose(match["diagonal_unweighted_trace"], diagonal_trace, rel_tol=1e-12):
        raise ValueError(f"diagonal unweighted trace mismatch: {label}")
    if not math.isclose(match["weighted_trace_target"], expected_target, rel_tol=1e-12):
        raise ValueError(f"trace target mismatch: {label}")
    if not math.isclose(match["matched_lambdas"]["rank1"], expected_rank1_lambda, rel_tol=1e-12):
        raise ValueError(f"corrected rank-1 lambda mismatch: {label}")
    if match["matched_lambdas"]["diagonal"] != 1_000.0:
        raise ValueError(f"diagonal lambda mismatch: {label}")
    for structure in ("rank1", "diagonal"):
        if not math.isclose(match["weighted_trace_checks"][structure], expected_target, rel_tol=1e-12):
            raise ValueError(f"weighted {structure} trace mismatch: {label}")
    expected_difference = abs(diagonal_trace - reported_trace) / diagonal_trace
    if not math.isclose(match["diagonal_trace_reported_relative_difference"], expected_difference, rel_tol=1e-12, abs_tol=1e-18):
        raise ValueError(f"reported/exact diagonal trace difference mismatch: {label}")
    if match["stored_tensor_dtype"] != "torch.float32" or match["stored_tensor_device"] != "cpu":
        raise ValueError(f"unexpected stored Fisher representation: {label}")


def summarize(family: str, run_root: Path) -> dict:
    config = CONFIGS[family]
    contract_path = run_root / "contract.json"
    contract = json.loads(contract_path.read_text())
    expected_grid = {
        "methods": list(METHODS), "b_clip": list(config["clips"]),
        "seeds": list(SEEDS), "cell_count": config["cell_count"],
    }
    if contract.get("status") != "frozen_before_pilot" or contract.get("family") != family:
        raise ValueError(f"unexpected {family} contract identity")
    if contract.get("grid") != expected_grid:
        raise ValueError(f"unexpected {family} contract grid")
    summarizer_rel = "experiments/summarize_corrected_trace_controls.py"
    if contract["sha256"].get(summarizer_rel) != _sha256(Path(__file__)):
        raise ValueError("summarizer is outside the frozen contract")
    expected = {
        (method, clip, seed): _expected_path(run_root, config, method, clip, seed)
        for method in METHODS for clip in config["clips"] for seed in SEEDS
    }
    actual = set((run_root / "formal").glob("*.json"))
    if actual != set(expected.values()):
        raise ValueError(f"{family} grid mismatch: missing={set(expected.values()) - actual}, extra={actual - set(expected.values())}")
    runs = {cell: json.loads(path.read_text()) for cell, path in expected.items()}
    input_reference = contract["inputs"]
    contract_sha256 = _sha256(contract_path)

    for (method, clip, seed), run in runs.items():
        path = expected[(method, clip, seed)]
        label = path.name
        _assert_finite(run, label)
        if run.get("status") != "ok" or run.get("family") != family or run.get("experiment") != config["experiment"]:
            raise ValueError(f"invalid result identity: {label}")
        protocol = run["protocol"]
        if (protocol["method"], protocol["task_b_clip"], protocol["seed"]) != (method, clip, seed):
            raise ValueError(f"cell metadata mismatch: {label}")
        if protocol["parameter_count"] != config["parameter_count"] or not protocol["full_parameter_training"]:
            raise ValueError(f"non-full parameter cell: {label}")
        if protocol["fisher_examples"] != 40 or run["fisher"]["calibration_examples"] != 40:
            raise ValueError(f"Fisher count mismatch: {label}")
        if run["fisher"]["parameter_count"] != config["parameter_count"] or run["fisher"]["repeat_loss_max_abs_difference"] != 0:
            raise ValueError(f"Fisher implementation mismatch: {label}")
        if protocol["replay_fact_counts"] != {str(fact): 16 for fact in range(8, 12)} or run["replay"]["examples"] != 64:
            raise ValueError(f"replay mismatch: {label}")
        if protocol["matching"] != "lambda_rank1_times_alpha_times_stored_direction_norm_sq_equals_lambda_diagonal_times_trace_diagonal":
            raise ValueError(f"wrong matching formula: {label}")
        if run["stiffness_match"]["stored_tensor_numel"] != config["parameter_count"]:
            raise ValueError(f"stored Fisher size mismatch: {label}")
        chunks = math.ceil(config["parameter_count"] / run["stiffness_match"]["float64_reduction_chunk_elements"])
        if run["stiffness_match"]["float64_reduction_chunks"] != chunks:
            raise ValueError(f"stored Fisher chunk count mismatch: {label}")
        for relative, digest in run["dependencies"].items():
            if contract["sha256"].get(relative) != digest:
                raise ValueError(f"dependency outside contract: {label}: {relative}")
        if run["inputs"]["checkpoint_sha256"] != contract["sha256"]["checkpoint"]:
            raise ValueError(f"checkpoint mismatch: {label}")
        if run["inputs"] != input_reference:
            raise ValueError(f"input artifacts differ: {label}")
        if any(run["runtime"].get(key) != value for key, value in contract["environment"].items()):
            raise ValueError(f"runtime environment differs from contract: {label}")
        if run["contract"] != {"path": f"runs/{config['run_dir']}/contract.json", "sha256": contract_sha256}:
            raise ValueError(f"run contract mismatch: {label}")
        if config["r16_anchors"]:
            checks = run["task_a_anchor_checks"]
            if checks is None or run["task_a_anchor"] is None:
                raise ValueError(f"missing R16 anchor: {label}")
            if checks["training_current_mean_abs_difference"] > 1e-9 or checks["training_clip_fraction_abs_difference"] != 0:
                raise ValueError(f"R16 training anchor mismatch: {label}")
            if checks["rank1_coefficient_relative_difference"] > 1e-6 or checks["diagonal_trace_relative_difference"] > 1e-6:
                raise ValueError(f"R16 Fisher anchor mismatch: {label}")
        elif run["task_a_anchor"] is not None or run["task_a_anchor_checks"] is not None:
            raise ValueError(f"unexpected R16 anchor at scale: {label}")
        _validate_trace(run, label)
        match = run["stiffness_match"]
        expected_lambda = 0.0 if method == "gd" else match["matched_lambdas"]["rank1" if method == "rank1_gd" else "diagonal"]
        if not math.isclose(match["effective_lambda"], expected_lambda, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"effective lambda mismatch: {label}")
        for endpoint, value in _recomputed_endpoints(run).items():
            if not math.isclose(run["summary"][endpoint], value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"summary recomputation failed: {label}: {endpoint}")

    for seed in SEEDS:
        reference = runs[("gd", config["clips"][0], seed)]
        for method in METHODS:
            for clip in config["clips"]:
                run = runs[(method, clip, seed)]
                if _without_timing(run["stages"][0]["training"]) != _without_timing(reference["stages"][0]["training"]):
                    raise ValueError(f"Task-A training differs within seed {seed}: {method}/{clip}")
                if run["stages"][0]["metrics"] != reference["stages"][0]["metrics"]:
                    raise ValueError(f"Task-A metrics differ within seed {seed}: {method}/{clip}")
                if run["fisher"] != reference["fisher"]:
                    raise ValueError(f"Fisher differs within seed {seed}: {method}/{clip}")
                if run["replay"]["generated_rows_sha256"] != reference["replay"]["generated_rows_sha256"]:
                    raise ValueError(f"replay differs within seed {seed}: {method}/{clip}")
                for key in ("matched_lambdas", "direction_norm_sq_float64_chunked", "diagonal_trace_float64_chunked"):
                    if run["stiffness_match"][key] != reference["stiffness_match"][key]:
                        raise ValueError(f"corrected match differs within seed {seed}: {method}/{clip}: {key}")

    aggregate = {}
    paired = {}
    contrasts = (
        ("rank1_gd_minus_gd", "rank1_gd", "gd"),
        ("diag_gd_minus_gd", "diag_gd", "gd"),
        ("rank1_gd_minus_diag_gd", "rank1_gd", "diag_gd"),
    )
    for clip in config["clips"]:
        clip_key = str(clip)
        aggregate[clip_key] = {}
        for method in METHODS:
            cells = [runs[(method, clip, seed)] for seed in SEEDS]
            aggregate[clip_key][method] = {
                **{endpoint: _stats([cell["summary"][endpoint] for cell in cells]) for endpoint in ENDPOINTS},
                "clip_fraction": _stats([cell["stages"][1]["training"]["clip_fraction"] for cell in cells]),
                "preclip_gradient_norm_max": _stats([cell["stages"][1]["training"]["gradient_norm_max"] for cell in cells]),
                "weighted_ewc_mean": _stats([cell["stages"][1]["training"]["ewc_loss_mean"] for cell in cells]),
                "weighted_ewc_max": _stats([
                    cell["stages"][1]["training"]["penalty_max"] * cell["stiffness_match"]["effective_lambda"]
                    for cell in cells
                ]),
                "wall_time_seconds": _stats([cell["wall_time_seconds"] for cell in cells]),
            }
        paired[clip_key] = {
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
    clipping_sensitivity = None
    if len(config["clips"]) == 2:
        clipping_sensitivity = {
            method: {
                endpoint: _stats([
                    runs[(method, config["clips"][1], seed)]["summary"][endpoint]
                    - runs[(method, config["clips"][0], seed)]["summary"][endpoint]
                    for seed in SEEDS
                ])
                for endpoint in ENDPOINTS
            }
            for method in METHODS
        }
    return {
        "schema_version": 1,
        "status": "ok",
        "family": family,
        "protocol": f"{family}_stored_direction_corrected_trace_v1",
        "run_count": len(runs),
        "contract_sha256": contract_sha256,
        "run_sha256": {path.name: _sha256(path) for path in expected.values()},
        "cell_summaries": {
            path.name: {"summary": runs[cell]["summary"], "stiffness_match": runs[cell]["stiffness_match"]}
            for cell, path in expected.items()
        },
        "aggregate": aggregate,
        "paired": paired,
        "clipping_sensitivity_unclipped_minus_clip1": clipping_sensitivity,
        "matched_lambdas": {
            str(seed): runs[("rank1_gd", config["clips"][0], seed)]["stiffness_match"]["matched_lambdas"]
            for seed in SEEDS
        },
        "audit": {
            "complete_cells": config["cell_count"],
            "source_and_input_hashes_verified": True,
            "full_parameter_fisher_verified": True,
            "r16_task_a_anchors_verified": config["r16_anchors"],
            "balanced_replay_verified": True,
            "stored_direction_and_diagonal_trace_equality_verified": True,
            "within_seed_task_a_fisher_replay_identity_verified": True,
            "summary_endpoints_recomputed_from_stage_metrics": True,
            "task_b_mechanism_diagnostics_aggregated": True,
            "all_outputs_finite": True,
        },
    }


def _self_check() -> None:
    stats = _stats([-1.0, 0.0, 2.0])
    assert stats["mean"] == 1 / 3 and stats["wins_below_zero"] == 1
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
    alpha, norm_sq, diagonal_trace = 2.5, 25.0, 5.1875
    target = 1_000.0 * diagonal_trace
    reported_trace = diagonal_trace + 1e-7
    trace_run = {
        "fisher": {"rank1_coefficient": alpha},
        "stiffness_match": {
            "direction_norm_sq_float64_chunked": norm_sq,
            "direction_norm_float64_chunked": math.sqrt(norm_sq),
            "rank1_unweighted_trace": alpha * norm_sq,
            "diagonal_unweighted_trace": diagonal_trace,
            "weighted_trace_target": target,
            "matched_lambdas": {"rank1": target / (alpha * norm_sq), "diagonal": 1_000.0},
            "weighted_trace_checks": {"rank1": target, "diagonal": target},
            "diagonal_trace_float32_reported": reported_trace,
            "diagonal_trace_float64_chunked": diagonal_trace,
            "diagonal_trace_reported_relative_difference": abs(diagonal_trace - reported_trace) / diagonal_trace,
            "stored_tensor_dtype": "torch.float32", "stored_tensor_device": "cpu",
        },
    }
    _validate_trace(trace_run, "toy")
    trace_run["stiffness_match"]["matched_lambdas"]["rank1"] = target / alpha
    try:
        _validate_trace(trace_run, "nominal-trace-bug")
    except ValueError:
        pass
    else:
        raise AssertionError("summarizer accepted nominal rank-1 trace for a non-unit direction")
    try:
        _assert_finite({"nested": [float("inf")]})
    except ValueError:
        pass
    else:
        raise AssertionError("recursive finite check accepted infinity")
    assert _expected_path(Path("x"), CONFIGS["r23"], "gd", 1e6, 3409).name == "gd_clip1000000_s3409.json"
    assert _expected_path(Path("x"), CONFIGS["r24"], "gd", 1.0, 3409).name == "gd_s3409.json"
    print(json.dumps({"self_check": "ok", "nominal_trace_bug_rejected": True}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(CONFIGS), default="r23")
    parser.add_argument("--run-root", type=Path)
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
    run_root = args.run_root or ROOT / "runs" / CONFIGS[args.family]["run_dir"]
    result = summarize(args.family, run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
