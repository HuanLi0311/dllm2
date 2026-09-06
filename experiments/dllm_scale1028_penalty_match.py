#!/usr/bin/env python3
"""Predeclared full-parameter SMDM-1.14B matched-Fisher scale extension."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_mdm import flat_parameters, load_model, set_seed, trainable_parameters  # noqa: E402
import dllm_penalty_match_control as r19  # noqa: E402
import dllm_rank1_multitask as multitask  # noqa: E402
import dllm_rank1_transfer as transfer  # noqa: E402


PROTOCOL = ROOT / "report/r22_scale1028_penalty_match_protocol.md"
METHODS = ("gd", "rank1_gd", "diag_gd")
SEEDS = (3407, 3408, 3409)
PARAMETER_COUNT = 1_142_367_744


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
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _dependencies() -> dict[str, str]:
    paths = (
        Path(__file__), PROTOCOL,
        ROOT / "experiments/dllm_penalty_match_control.py",
        ROOT / "experiments/dllm_rank1_multitask.py",
        ROOT / "experiments/dllm_rank1_transfer.py",
        ROOT / "continual_benchmark.py", ROOT / "continual_mdm.py",
        ROOT / "continual_reverse.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _inputs(args, tasks) -> dict:
    return {
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


def _matched_stiffness(alpha: float, diagonal_trace: float) -> dict:
    if not all(math.isfinite(value) and value > 0 for value in (alpha, diagonal_trace)):
        raise ValueError("rank-1 coefficient and diagonal trace must be finite and positive")
    target = 1_000.0 * diagonal_trace
    lambdas = {"rank1": target / alpha, "diagonal": 1_000.0}
    checks = {"rank1": lambdas["rank1"] * alpha, "diagonal": lambdas["diagonal"] * diagonal_trace}
    if any(not math.isclose(value, target, rel_tol=1e-12) for value in checks.values()):
        raise AssertionError(f"weighted-trace matching failed: {checks}")
    return {"weighted_trace_target": target, "matched_lambdas": lambdas, "weighted_trace_checks": checks}


def _assert_finite(value, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def run(args) -> dict:
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.method not in METHODS or args.seed not in SEEDS:
        raise ValueError("run is outside the frozen R22 grid")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    tasks = multitask._load_tasks(args, tokenizer)
    if [task["name"] for task in tasks] != ["d2p_8-11", "p2d_12-15"]:
        raise ValueError("unexpected R22 task sequence")
    dependencies = _dependencies()
    inputs = _inputs(args, tasks)

    model = load_model(args, device)
    parameters = trainable_parameters(model, "all")
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != PARAMETER_COUNT:
        raise ValueError(f"unexpected full parameter count: {parameter_count}")

    task_a_training = multitask._train_stage(
        model, tasks[0]["train"], parameters, pad_id, device, args, args.seed + 1000
    )
    task_a_metrics = multitask._measure_task(model, tokenizer, tasks[0], pad_id, device, args)
    fisher, fisher_stats = transfer._estimate_mean_and_diagonal_fisher(
        model, tasks[0]["fisher"], parameters, pad_id, device, args
    )
    if (
        fisher_stats["calibration_examples"] != 40
        or fisher_stats["parameter_count"] != PARAMETER_COUNT
        or fisher_stats["repeat_loss_max_abs_difference"] != 0.0
    ):
        raise AssertionError(f"unexpected R22 Fisher audit: {fisher_stats}")

    prompts, replay_manifest = multitask._replay_prompts(tasks, 1, 64)
    expected_counts = {str(fact): 16 for fact in range(8, 12)}
    if len(replay_manifest) != 1 or replay_manifest[0]["fact_counts"] != expected_counts:
        raise AssertionError("R22 replay prompts are not balanced 16/16/16/16")
    replay = transfer._generate_replay(model, tokenizer, prompts, device, args)
    if len(replay) != 64:
        raise AssertionError(f"unexpected R22 replay size: {len(replay)}")
    teacher = load_model(args, device)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    reference = flat_parameters(parameters).detach().clone()
    stiffness = _matched_stiffness(
        float(fisher["coefficient"]), float(fisher_stats["diagonal_trace"])
    )

    constraint = None
    effective_lambda = 0.0
    if args.method == "rank1_gd":
        effective_lambda = stiffness["matched_lambdas"]["rank1"]
        constraint = {
            "kind": "rank1", "reference": reference,
            "direction": fisher["direction"].to(device),
            "coefficient": float(fisher["coefficient"]),
        }
    elif args.method == "diag_gd":
        effective_lambda = stiffness["matched_lambdas"]["diagonal"]
        constraint = {
            "kind": "diagonal", "reference": reference,
            "diagonal": fisher["diagonal"].to(device),
        }
    del fisher
    args.ewc_lambda = effective_lambda
    task_b_training = multitask._train_stage(
        model, tasks[1]["train"], parameters, pad_id, device, args,
        args.seed + 2000, teacher=teacher, replay_rows=replay,
        constraints=[] if constraint is None else [constraint],
    )
    del teacher, reference, constraint
    torch.cuda.empty_cache()
    final_metrics = {
        task["name"]: multitask._measure_task(model, tokenizer, task, pad_id, device, args)
        for task in tasks
    }
    stages = [
        {"task": tasks[0]["name"], "training": task_a_training,
         "metrics": {tasks[0]["name"]: task_a_metrics}, "fisher": fisher_stats},
        {"task": tasks[1]["name"], "training": task_b_training, "metrics": final_metrics},
    ]
    stiffness["effective_lambda"] = effective_lambda
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "r22_scale1028_weighted_trace_match",
        "role": "predeclared_exploratory_scale_extension",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "host": os.uname().nodename,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "dependencies": dependencies,
        "inputs": inputs,
        "protocol": {
            "method": args.method, "seed": args.seed,
            "model": 1028, "parameter_count": parameter_count,
            "task_sequence": [task["name"] for task in tasks],
            "full_parameter_training": True,
            "objective": "r16_answer_only_independent_bernoulli_allow_empty_importance_weighted",
            "steps_per_task": 1000, "batch_size": 4, "learning_rate": 5e-5,
            "clip": 1.0, "fisher_examples": 40,
            "matching": "lambda_rank1_times_alpha_equals_lambda_diagonal_times_trace_diagonal",
            "diagonal_anchor_lambda": 1000.0,
            "replay_fact_counts": replay_manifest[0]["fact_counts"],
        },
        "training_state": {
            "fisher_sampling_stage": "immediately_after_task_a_training",
            "task_a_training": task_a_training,
            "task_a_metrics": task_a_metrics,
        },
        "fisher": fisher_stats,
        "stiffness_match": stiffness,
        "replay": {
            "examples": len(replay), "manifest": replay_manifest,
            "generated_rows_sha256": multitask._records_sha256(replay),
        },
        "stages": stages,
        "summary": r19._summary(stages, tasks),
    }
    _assert_finite(result)
    if _dependencies() != dependencies:
        raise RuntimeError("source/protocol provenance changed during R22 run")
    if _inputs(args, tasks) != inputs:
        raise RuntimeError("input provenance changed during R22 run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check() -> None:
    direction = torch.tensor([3.0, 4.0]) / 5
    diagonal = torch.tensor([0.5, 1.5])
    match = _matched_stiffness(2.5, float(diagonal.sum()))
    rank1 = match["matched_lambdas"]["rank1"] * 2.5 * torch.outer(direction, direction)
    diag = match["matched_lambdas"]["diagonal"] * torch.diag(diagonal)
    assert math.isclose(float(torch.trace(rank1)), float(torch.trace(diag)), rel_tol=1e-7)
    assert multitask._sft_losses is transfer._sft_losses
    mock = [{"name": "a", "train_raw": [
        {"fact_id": fact, "prompt": f"p-{fact}-{index}"}
        for fact in range(8, 12) for index in range(30)
    ]}]
    prompts, manifest = multitask._replay_prompts(mock, 1, 64)
    assert len(prompts) == 64
    assert manifest[0]["fact_counts"] == {str(fact): 16 for fact in range(8, 12)}
    assert r19._summary([
        {"metrics": {"a": {"loss": 1.0}}},
        {"metrics": {"a": {"loss": 2.0, "answer_token_accuracy": 0.5},
                     "b": {"loss": 0.5, "answer_token_accuracy": 0.75}}},
    ], [{"name": "a"}, {"name": "b"}])["final_average_loss"] == 1.25
    _assert_finite({"values": [0.0, 1.0]})
    try:
        _assert_finite({"values": [float("nan")]})
    except ValueError:
        pass
    else:
        raise AssertionError("recursive finite check accepted NaN")
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=False, default="gd")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=3407)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-1028M-1600e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=ROOT / "SMDM/data/reverse_experiments/june_version_7921032488")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and args.output is None:
        parser.error("--output is required")
    args.model = 1028
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
    args.final_protocol = False
    args.fresh_protocol = False
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()
