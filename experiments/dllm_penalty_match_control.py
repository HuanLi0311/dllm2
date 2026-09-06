#!/usr/bin/env python3
"""Weighted-trace-matched EWC structure and clipping control for SMDM."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_mdm import flat_parameters, load_model, set_seed, trainable_parameters  # noqa: E402
import dllm_rank1_multitask as multitask  # noqa: E402
import dllm_rank1_transfer as transfer  # noqa: E402


PROTOCOL = ROOT / "report/r19_penalty_match_protocol.md"
METHODS = ("gd", "rank1_gd", "diag_gd")
CLIPS = (1.0, 1_000_000.0)
R16_ANCHORS = {
    3407: {"current_mean": 0.15928860665256275, "clip_fraction": 0.236,
           "rank1_coefficient": 0.0008613997596079841, "diagonal_trace": 0.002578950487077236},
    3408: {"current_mean": 0.14222952271252223, "clip_fraction": 0.239,
           "rank1_coefficient": 0.002652776763081589, "diagonal_trace": 0.005044732242822647},
    3409: {"current_mean": 0.1648787468732853, "clip_fraction": 0.253,
           "rank1_coefficient": 0.01223424401850455, "diagonal_trace": 0.019160481169819832},
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _dependencies() -> dict[str, str]:
    paths = (
        Path(__file__), PROTOCOL,
        ROOT / "experiments/dllm_rank1_multitask.py",
        ROOT / "experiments/dllm_rank1_transfer.py",
        ROOT / "continual_benchmark.py", ROOT / "continual_mdm.py",
        ROOT / "continual_reverse.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _summary(stages, tasks) -> dict:
    learned_a = stages[0]["metrics"][tasks[0]["name"]]["loss"]
    final = stages[1]["metrics"]
    a_final = final[tasks[0]["name"]]["loss"]
    b_final = final[tasks[1]["name"]]["loss"]
    return {
        "final_average_loss": (a_final + b_final) / 2,
        "past_task_forgetting": a_final - learned_a,
        "final_task_loss": b_final,
        "final_average_answer_token_accuracy": (
            final[tasks[0]["name"]]["answer_token_accuracy"]
            + final[tasks[1]["name"]]["answer_token_accuracy"]
        ) / 2,
        "task_a_loss_when_learned": learned_a,
        "task_a_final_loss": a_final,
    }


def run(args) -> dict:
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.seed not in R16_ANCHORS or args.method not in METHODS or args.b_clip not in CLIPS:
        raise ValueError("run is outside the frozen R19 grid")
    set_seed(args.seed)
    device = torch.device(args.device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = multitask._load_tasks(args, tokenizer)
    if [task["name"] for task in tasks] != ["d2p_8-11", "p2d_12-15"]:
        raise ValueError("unexpected R19 task sequence")
    dependencies = _dependencies()
    inputs = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer_sha256": _tree_sha256(args.tokenizer),
        "reverse_data_sha256": _tree_sha256(args.reverse_dir),
        "task_artifacts": [
            {
                "name": task["name"],
                "train": multitask._records_sha256(task["train_raw"]),
                "fisher": multitask._records_sha256(task["fisher_raw"]),
                "test": multitask._records_sha256(task["eval_raw"]),
            }
            for task in tasks
        ],
    }

    model = load_model(args, device)
    parameters = trainable_parameters(model, "all")
    args.clip = 1.0
    args.steps_per_task = 1000
    args.ewc_lambda = 0.0
    task_a_training = multitask._train_stage(
        model, tasks[0]["train"], parameters, pad_id, device, args, args.seed + 1000
    )
    task_a_metrics = multitask._measure_task(model, tokenizer, tasks[0], pad_id, device, args)
    fisher, fisher_stats = transfer._estimate_mean_and_diagonal_fisher(
        model, tasks[0]["fisher"], parameters, pad_id, device, args
    )
    anchor = R16_ANCHORS[args.seed]
    anchor_checks = {
        "training_current_mean_abs_difference": abs(task_a_training["current_mean"] - anchor["current_mean"]),
        "training_clip_fraction_abs_difference": abs(task_a_training["clip_fraction"] - anchor["clip_fraction"]),
        "rank1_coefficient_relative_difference": abs(fisher_stats["rank1_coefficient"] - anchor["rank1_coefficient"]) / anchor["rank1_coefficient"],
        "diagonal_trace_relative_difference": abs(fisher_stats["diagonal_trace"] - anchor["diagonal_trace"]) / anchor["diagonal_trace"],
    }
    if (
        anchor_checks["training_current_mean_abs_difference"] > 1e-9
        or anchor_checks["training_clip_fraction_abs_difference"] > 0
        or anchor_checks["rank1_coefficient_relative_difference"] > 1e-6
        or anchor_checks["diagonal_trace_relative_difference"] > 1e-6
    ):
        raise AssertionError(f"Task-A state does not reproduce R16: {anchor_checks}")

    prompts, replay_manifest = multitask._replay_prompts(tasks, 1, 64)
    expected_counts = {str(fact): 16 for fact in range(8, 12)}
    if len(replay_manifest) != 1 or replay_manifest[0]["fact_counts"] != expected_counts:
        raise AssertionError("R19 replay prompts are not balanced 16/16/16/16")
    replay = transfer._generate_replay(model, tokenizer, prompts, device, args)
    teacher = load_model(args, device)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    reference = flat_parameters(parameters).detach().clone()
    alpha = float(fisher["coefficient"])
    diagonal_trace = float(fisher_stats["diagonal_trace"])
    if not all(math.isfinite(value) and value > 0 for value in (alpha, diagonal_trace)):
        raise ValueError("rank-1 coefficient and diagonal trace must be finite and positive")
    weighted_trace_target = 1_000.0 * diagonal_trace
    matched_lambdas = {"rank1": weighted_trace_target / alpha, "diagonal": 1_000.0}
    weighted_trace_checks = {
        "rank1": matched_lambdas["rank1"] * alpha,
        "diagonal": matched_lambdas["diagonal"] * diagonal_trace,
    }
    if any(abs(value - weighted_trace_target) > 1e-9 * weighted_trace_target for value in weighted_trace_checks.values()):
        raise AssertionError(f"weighted-trace matching failed: {weighted_trace_checks}")

    constraint = None
    effective_lambda = 0.0
    if args.method == "rank1_gd":
        effective_lambda = matched_lambdas["rank1"]
        constraint = {
            "kind": "rank1", "reference": reference,
            "direction": fisher["direction"].to(device),
            "coefficient": float(fisher["coefficient"]),
        }
    elif args.method == "diag_gd":
        effective_lambda = matched_lambdas["diagonal"]
        constraint = {
            "kind": "diagonal", "reference": reference,
            "diagonal": fisher["diagonal"].to(device),
        }
    args.steps_per_task = 1000
    args.clip = args.b_clip
    args.ewc_lambda = effective_lambda
    task_b_training = multitask._train_stage(
        model, tasks[1]["train"], parameters, pad_id, device, args,
        args.seed + 2000, teacher=teacher, replay_rows=replay,
        constraints=[] if constraint is None else [constraint],
    )
    final_metrics = {
        task["name"]: multitask._measure_task(model, tokenizer, task, pad_id, device, args)
        for task in tasks
    }
    stages = [
        {"task": tasks[0]["name"], "training": task_a_training,
         "metrics": {tasks[0]["name"]: task_a_metrics}, "fisher": fisher_stats},
        {"task": tasks[1]["name"], "training": task_b_training,
         "metrics": final_metrics},
    ]
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "r19_weighted_trace_match",
        "role": "exploratory_predeclared_complete_grid",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "host": __import__("os").uname().nodename,
        "dependencies": dependencies,
        "inputs": inputs,
        "protocol": {
            "method": args.method, "seed": args.seed,
            "task_sequence": [task["name"] for task in tasks],
            "full_parameter_training": True,
            "objective": "r16_answer_only_independent_bernoulli_allow_empty_importance_weighted",
            "task_a_clip": 1.0, "task_b_clip": args.b_clip,
            "matching": "lambda_rank1_times_alpha_equals_lambda_diagonal_times_trace_diagonal",
            "diagonal_anchor_lambda": 1000.0,
            "replay_fact_counts": replay_manifest[0]["fact_counts"],
        },
        "task_a_anchor": anchor,
        "task_a_anchor_checks": anchor_checks,
        "fisher": fisher_stats,
        "stiffness_match": {
            "weighted_trace_target": weighted_trace_target,
            "matched_lambdas": matched_lambdas,
            "weighted_trace_checks": weighted_trace_checks,
            "effective_lambda": effective_lambda,
        },
        "replay": {
            "examples": len(replay), "manifest": replay_manifest,
            "generated_rows_sha256": multitask._records_sha256(replay),
        },
        "stages": stages,
        "summary": _summary(stages, tasks),
    }
    if _dependencies() != dependencies:
        raise RuntimeError("source/protocol provenance changed during R19 run")
    current_inputs = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer_sha256": _tree_sha256(args.tokenizer),
        "reverse_data_sha256": _tree_sha256(args.reverse_dir),
        "task_artifacts": [
            {"name": task["name"], "train": multitask._records_sha256(task["train_raw"]),
             "fisher": multitask._records_sha256(task["fisher_raw"]),
             "test": multitask._records_sha256(task["eval_raw"])}
            for task in tasks
        ],
    }
    if current_inputs != inputs:
        raise RuntimeError("input provenance changed during R19 run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check() -> None:
    direction = torch.tensor([3.0, 4.0]) / 5
    coefficient = 2.5
    diagonal = torch.tensor([0.5, 1.5])
    target = 1_000.0 * float(diagonal.sum())
    rank1_lambda = target / coefficient
    rank1_matrix = rank1_lambda * coefficient * torch.outer(direction, direction)
    diagonal_matrix = 1_000.0 * torch.diag(diagonal)
    assert math.isclose(float(torch.trace(rank1_matrix)), target, rel_tol=1e-7)
    assert math.isclose(float(torch.trace(diagonal_matrix)), target, rel_tol=1e-7)
    assert multitask._sft_losses is transfer._sft_losses
    mock = [{"name": "a", "train_raw": [
        {"fact_id": fact, "prompt": f"p-{fact}-{index}"}
        for fact in range(8, 12) for index in range(30)
    ]}]
    prompts, manifest = multitask._replay_prompts(mock, 1, 64)
    assert len(prompts) == 64
    assert manifest[0]["fact_counts"] == {str(fact): 16 for fact in range(8, 12)}
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="gd")
    parser.add_argument("--b-clip", type=float, choices=CLIPS, default=1.0)
    parser.add_argument("--seed", type=int, choices=tuple(R16_ANCHORS), default=3407)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=ROOT / "SMDM/data/reverse_experiments/june_version_7921032488")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and args.output is None:
        parser.error("--output is required")
    args.model = 170
    args.start_direction = "d2p"
    args.order = "forward"
    args.group_start = 8
    args.tasks = 2
    args.group_count = 4
    args.fisher_per_fact = 10
    args.trainable = "all"
    args.max_length = 128
    args.batch_size = 4
    args.eval_batch_size = 4
    args.eval_mc_samples = 32
    args.steps_per_task = 1000
    args.lr = 5e-5
    args.clip = 1.0
    args.mask_min = 1e-3
    args.mask_max = 1.0
    args.replay_per_task = 64
    args.replay_steps = 32
    args.replay_length = 52
    args.replay_cfg = 0.8
    args.replay_temperature = 0.0
    args.distill_weight = 1.0
    args.distill_temperature = 1.0
    args.ewc_lambda = 0.0
    args.generation_seed = args.seed
    args.generation_batch_size = 1
    args.reverse_steps = 32
    args.reverse_length = 52
    args.reverse_cfg = 0.8
    args.reverse_temperature = 0.0
    args.show_predictions = False
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()
