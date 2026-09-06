#!/usr/bin/env python3
"""Select validation lambdas and summarize the Qwen3 continual matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MODELS = ("qwen3_0.6b", "qwen3_1.7b", "qwen3_4b")
METHODS = ("seq", "gd", "rank1_gd", "diag_gd")
EWC_METHODS = ("rank1_gd", "diag_gd")
SEEDS = (3407, 3408, 3409)
GRID = (1e2, 1e3, 1e4, 1e5, 1e6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(paths):
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            raise ValueError(f"non-ok result: {path}")
        for endpoint in ("final_average_loss", "past_task_forgetting", "final_average_answer_token_accuracy"):
            if not math.isfinite(payload["summary"][endpoint]):
                raise ValueError(f"non-finite {endpoint}: {path}")
        payloads.append(payload)
    if not payloads:
        raise ValueError("no inputs")
    if len({json.dumps(payload["data_sha256"], sort_keys=True) for payload in payloads}) != 1:
        raise ValueError("mixed data_sha256")
    for model in {payload["metadata"]["model_label"] for payload in payloads}:
        group = [payload for payload in payloads if payload["metadata"]["model_label"] == model]
        for key in ("source_sha256", "protocol_sha256"):
            if len({payload[key] for payload in group}) != 1:
                raise ValueError(f"mixed {key} within {model}")
    return payloads


def _mean_sem(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return {"mean": mean, "sem": math.sqrt(variance / len(values)), "values": values}


def select(paths, output, models, grid):
    payloads = _load(paths)
    if len(payloads) != len(models) * len(EWC_METHODS) * len(grid) * len(SEEDS):
        raise ValueError("incomplete validation selection matrix")
    cells = {}
    for payload in payloads:
        metadata = payload["metadata"]
        if metadata["run_kind"] != "validation" or metadata["task_sequence"] != ["d2p_0-1", "p2d_2-3"]:
            raise ValueError("unexpected validation protocol")
        key = (metadata["model_label"], metadata["method"], metadata["ewc_lambda"])
        if key[0] not in models or key[1] not in EWC_METHODS or key[2] not in grid:
            raise ValueError(f"unexpected validation cell {key}")
        cells.setdefault(key, {})[metadata["seed"]] = payload["summary"]["final_average_loss"]
    expected = {(model, method, value) for model in models for method in EWC_METHODS for value in grid}
    if set(cells) != expected or any(set(values) != set(SEEDS) for values in cells.values()):
        raise ValueError("incomplete or duplicated validation matrix")
    scores, selected, boundaries = {}, {}, []
    for model in models:
        selected[model] = {"seq": 0.0, "gd": 0.0}
        scores[model] = {}
        for method in EWC_METHODS:
            method_scores = {}
            for value in grid:
                values = [cells[(model, method, value)][seed] for seed in SEEDS]
                method_scores[str(value)] = _mean_sem(values)
            winner = min(grid, key=lambda value: (method_scores[str(value)]["mean"], value))
            selected[model][method] = winner
            scores[model][method] = method_scores
            if winner in (grid[0], grid[-1]):
                boundaries.append({"model": model, "method": method, "lambda": winner})
    result = {
        "schema_version": 1,
        "status": "ok" if not boundaries else "boundary_extension_required",
        "selection_rule": "lowest three-seed mean final average held-out loss",
        "grid": list(grid),
        "selected": selected,
        "scores": scores,
        "boundary_winners": boundaries,
        "audit": {
            "run_count": len(payloads),
            "runner_sha256": payloads[0]["source_sha256"],
            "protocol_sha256": payloads[0]["protocol_sha256"],
            "summarizer_sha256": _sha256(Path(__file__)),
            "model_inventory_sha256": {
                model: next(
                    payload["metadata"]["model_inventory_sha256"]
                    for payload in payloads if payload["metadata"]["model_label"] == model
                )
                for model in models
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "selected": selected, "boundary_winners": boundaries}, indent=2))


def summarize(paths, output, models):
    payloads = _load(paths)
    if len(payloads) != len(models) * len(METHODS) * len(SEEDS):
        raise ValueError("incomplete formal summary matrix")
    cells = {}
    for payload in payloads:
        metadata = payload["metadata"]
        if metadata["run_kind"] != "formal" or metadata["task_sequence"] != [
            "d2p_8-11", "p2d_12-15", "d2p_16-19", "p2d_20-23"
        ]:
            raise ValueError("unexpected formal protocol")
        key = (metadata["model_label"], metadata["method"])
        if key[0] not in models or key[1] not in METHODS:
            raise ValueError(f"unexpected formal cell {key}")
        if metadata["seed"] in cells.setdefault(key, {}):
            raise ValueError(f"duplicate formal seed {key} {metadata['seed']}")
        cells[key][metadata["seed"]] = payload
    expected = {(model, method) for model in models for method in METHODS}
    if set(cells) != expected or any(set(values) != set(SEEDS) for values in cells.values()):
        raise ValueError("incomplete formal matrix")
    for model in models:
        selections = {
            payload["metadata"]["selection_sha256"]
            for payload in payloads if payload["metadata"]["model_label"] == model
        }
        if len(selections) != 1:
            raise ValueError(f"mixed selection artifact within {model}")
    endpoints = ("final_average_loss", "past_task_forgetting", "final_average_answer_token_accuracy")
    groups = {}
    for model in models:
        groups[model] = {}
        for method in METHODS:
            groups[model][method] = {
                endpoint: _mean_sem([
                    cells[(model, method)][seed]["summary"][endpoint] for seed in SEEDS
                ])
                for endpoint in endpoints
            }
    contrasts = {}
    comparisons = (("gd", "seq"), ("rank1_gd", "gd"), ("diag_gd", "gd"), ("rank1_gd", "diag_gd"))
    for model in models:
        contrasts[model] = {}
        for left, right in comparisons:
            name = f"{left}_minus_{right}"
            contrasts[model][name] = {}
            for endpoint in endpoints:
                values = [
                    cells[(model, left)][seed]["summary"][endpoint]
                    - cells[(model, right)][seed]["summary"][endpoint]
                    for seed in SEEDS
                ]
                contrasts[model][name][endpoint] = {
                    **_mean_sem(values),
                    "negative_wins": sum(value < 0 for value in values),
                }
    result = {
        "schema_version": 1,
        "status": "ok",
        "groups": groups,
        "paired_contrasts": contrasts,
        "audit": {
            "run_count": len(payloads),
            "runner_sha256": {
                model: next(payload["source_sha256"] for payload in payloads if payload["metadata"]["model_label"] == model)
                for model in models
            },
            "protocol_sha256": {
                model: next(payload["protocol_sha256"] for payload in payloads if payload["metadata"]["model_label"] == model)
                for model in models
            },
            "summarizer_sha256": _sha256(Path(__file__)),
            "selection_sha256": {
                model: next(payload["metadata"]["selection_sha256"] for payload in payloads if payload["metadata"]["model_label"] == model)
                for model in models
            },
            "model_inventory_sha256": {
                model: next(
                    payload["metadata"]["model_inventory_sha256"]
                    for payload in payloads if payload["metadata"]["model_label"] == model
                )
                for model in models
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "groups": groups, "paired_contrasts": contrasts}, indent=2))


def _self_check():
    group = _mean_sem([1.0, 2.0, 3.0])
    assert group["mean"] == 2.0 and group["values"] == [1.0, 2.0, 3.0]
    assert math.isclose(group["sem"], 1 / math.sqrt(3))
    print(json.dumps({"self_check": "ok"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "summarize", "self-check"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--grid", nargs="+", type=float, default=GRID)
    parser.add_argument("inputs", type=Path, nargs="*")
    args = parser.parse_args()
    if args.mode == "self-check":
        _self_check()
    elif args.output is None:
        parser.error("--output is required")
    elif args.mode == "select":
        select(args.inputs, args.output, tuple(args.models), tuple(args.grid))
    else:
        summarize(args.inputs, args.output, tuple(args.models))


if __name__ == "__main__":
    main()
