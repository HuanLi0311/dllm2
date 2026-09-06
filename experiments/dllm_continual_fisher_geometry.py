#!/usr/bin/env python3
"""Held-out Fisher geometry on the actual first R16 continual-learning task."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
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

from continual_benchmark import encode_benchmark_rows  # noqa: E402
from continual_mdm import load_model, set_seed, trainable_parameters  # noqa: E402
from continual_reverse import fact_rows  # noqa: E402
from experiments import dllm_rank1_multitask as multitask  # noqa: E402
from experiments import dllm_rank1_transfer as transfer  # noqa: E402


PROTOCOL = ROOT / "report/r18_continual_fisher_geometry_protocol.md"
PROBES = (
    "transformer.h.0.norm_1.weight",
    "transformer.h.8.norm_1.weight",
    "transformer.h.17.norm_1.weight",
    "transformer.h.0.attn.proj.weight",
)
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


def _records_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _prompt_sha256(rows: list[dict]) -> str:
    return _records_sha256([row["prompt"] for row in rows])


def _assert_disjoint(calibration: list[dict], test: list[dict]) -> None:
    overlap = set(row["prompt"] for row in calibration) & set(row["prompt"] for row in test)
    if overlap:
        raise ValueError(f"calibration/test prompt overlap: {len(overlap)} prompts")


def _parameter_moments(model) -> dict:
    count = 0
    value_sum = 0.0
    square_sum = 0.0
    for parameter in model.parameters():
        value = parameter.detach().double()
        count += value.numel()
        value_sum += float(value.sum())
        square_sum += float(value.square().sum())
    return {"parameter_count": count, "sum": value_sum, "squared_l2": square_sum}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _audited_sft_loss(model, row, pad_id, device, generator, mask_min, mask_max):
    state = generator.get_state()
    loss = transfer._sft_losses(
        model, [row], pad_id, device, generator, mask_min, mask_max
    )[0]
    audit_generator = torch.Generator(device=device)
    audit_generator.set_state(state)
    ids, _, answer = transfer._collate([row], pad_id)
    _, mask, probability = transfer._corrupt_answers(
        ids.to(device), answer.to(device), audit_generator, mask_min, mask_max
    )
    return loss, {
        "answer_tokens": int(answer.sum()),
        "masked_tokens": int(mask.sum()),
        "mask_probability": float(probability[0, 0]),
    }


def _collect_gradients(model, rows, probes, pad_id, device, seed, mask_min, mask_max):
    generator = torch.Generator(device=device).manual_seed(seed)
    collected = {name: [] for name in probes}
    losses = []
    masks = []
    model.eval()
    for index, row in enumerate(rows):
        loss, mask_audit = _audited_sft_loss(
            model, row, pad_id, device, generator, mask_min, mask_max
        )
        gradients = torch.autograd.grad(
            loss, [probes[name] for name in probes], allow_unused=True
        )
        for name, parameter, gradient in zip(probes, (probes[name] for name in probes), gradients):
            value = gradient if gradient is not None else torch.zeros_like(parameter)
            collected[name].append(value.detach().float().cpu().reshape(-1))
        losses.append(float(loss.detach().cpu()))
        masks.append(mask_audit)
        model.zero_grad(set_to_none=True)
        print(f"gradient={index + 1}/{len(rows)} seed={seed}", flush=True)
    return {name: torch.stack(values) for name, values in collected.items()}, losses, masks


def _parameter_layout(model, parameters) -> tuple[int, dict[str, tuple[int, int]]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    offset = 0
    layout = {}
    for parameter in parameters:
        name = names[id(parameter)]
        layout[name] = (offset, offset + parameter.numel())
        offset += parameter.numel()
    return offset, layout


def _full_gradient(loss, parameters, size: int) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    vector = torch.empty(size, dtype=torch.float32)
    offset = 0
    for parameter, gradient in zip(parameters, gradients):
        width = parameter.numel()
        if gradient is None:
            vector[offset : offset + width].zero_()
        else:
            vector[offset : offset + width].copy_(gradient.detach().float().cpu().reshape(-1))
        offset += width
    return vector


def _full_fisher_pass(
    model,
    rows,
    parameters,
    size,
    probe_layout,
    pad_id,
    device,
    seed,
    mean=None,
    retain_vectors=False,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    gradient_sum = torch.zeros(size, dtype=torch.float64) if mean is None else None
    diagonal_sum = torch.zeros(size, dtype=torch.float64) if mean is None else None
    projection_square_sum = 0.0
    vectors = []
    slices = {name: [] for name in PROBES}
    losses = []
    masks = []
    model.eval()
    for index, row in enumerate(rows):
        loss, mask_audit = _audited_sft_loss(
            model, row, pad_id, device, generator, 1e-3, 1.0
        )
        vector = _full_gradient(loss, parameters, size)
        value64 = vector.double()
        if mean is None:
            gradient_sum.add_(value64)
            diagonal_sum.addcmul_(value64, value64)
            for name, (left, right) in probe_layout.items():
                slices[name].append(vector[left:right].clone())
        else:
            projection_square_sum += float(torch.dot(value64, mean).square())
        if retain_vectors:
            vectors.append(vector)
            if mean is not None:
                for name, (left, right) in probe_layout.items():
                    slices[name].append(vector[left:right].clone())
        losses.append(float(loss.detach().cpu()))
        masks.append(mask_audit)
        model.zero_grad(set_to_none=True)
        del value64
        if not retain_vectors:
            del vector
        print(f"full_gradient={index + 1}/{len(rows)} seed={seed}", flush=True)
    return {
        "gradient_sum": gradient_sum,
        "diagonal_sum": diagonal_sum,
        "projection_square_sum": projection_square_sum,
        "vectors": vectors,
        "slices": slices,
        "losses": losses,
        "masks": masks,
    }


def _test_gram(vectors: list[torch.Tensor], size: int, chunk_size: int) -> torch.Tensor:
    gram = torch.zeros((len(vectors), len(vectors)), dtype=torch.float64)
    for left in range(0, size, chunk_size):
        right = min(left + chunk_size, size)
        block = torch.stack([vector[left:right] for vector in vectors]).double()
        gram.addmm_(block, block.T)
        del block
        print(f"test_gram_parameters={right}/{size}", flush=True)
    return gram


def _geometry_from_sufficient(
    mean: torch.Tensor,
    calibration_diagonal: torch.Tensor,
    calibration_projection_square_mean: float,
    test_diagonal: torch.Tensor,
    test_projection_square_mean: float,
    test_gram: torch.Tensor,
    calibration_examples: int,
    test_examples: int,
) -> dict:
    mu_norm_sq = torch.dot(mean, mean)
    if not float(mu_norm_sq) > 0.0:
        raise ValueError("zero calibration mean gradient")
    coefficient = torch.as_tensor(calibration_projection_square_mean, dtype=torch.float64) / mu_norm_sq.square()
    normalized_test_gram = test_gram / test_examples
    fisher_norm_sq = normalized_test_gram.square().sum()
    rank1_inner = coefficient * test_projection_square_mean
    rank1_norm_sq = coefficient.square() * mu_norm_sq.square()
    diagonal_inner = torch.dot(test_diagonal, calibration_diagonal)
    diagonal_norm_sq = calibration_diagonal.square().sum()
    rank1_error = torch.sqrt(
        (fisher_norm_sq - 2 * rank1_inner + rank1_norm_sq).clamp_min(0) / fisher_norm_sq
    )
    diagonal_error = torch.sqrt(
        (fisher_norm_sq - 2 * diagonal_inner + diagonal_norm_sq).clamp_min(0) / fisher_norm_sq
    )
    test_coefficient = torch.as_tensor(test_projection_square_mean, dtype=torch.float64) / mu_norm_sq.square()
    direction_oracle_error = torch.sqrt(
        (
            fisher_norm_sq
            - 2 * test_coefficient * test_projection_square_mean
            + test_coefficient.square() * mu_norm_sq.square()
        ).clamp_min(0)
        / fisher_norm_sq
    )
    top_eigenvalue = torch.linalg.eigvalsh(normalized_test_gram)[-1]
    best_rank1_error = torch.sqrt(
        (fisher_norm_sq - top_eigenvalue.square()).clamp_min(0) / fisher_norm_sq
    )
    tolerance = 1e-8
    if float(best_rank1_error) > float(direction_oracle_error) + tolerance:
        raise AssertionError("best held-out rank-1 error exceeds direction-constrained oracle")
    if float(direction_oracle_error) > float(rank1_error) + tolerance:
        raise AssertionError("held-out direction oracle exceeds calibration-fitted rank-1")
    result = {
        "calibration_examples": calibration_examples,
        "test_examples": test_examples,
        "parameter_count": mean.numel(),
        "mean_gradient_norm": float(torch.sqrt(mu_norm_sq)),
        "rank1_coefficient": float(coefficient),
        "rank1_ewc_direction_coefficient": float(coefficient * mu_norm_sq),
        "diagonal_trace": float(calibration_diagonal.sum()),
        "test_fisher_frobenius": float(torch.sqrt(fisher_norm_sq)),
        "rank1_relative_frobenius_error": float(rank1_error),
        "diagonal_relative_frobenius_error": float(diagonal_error),
        "heldout_score_log_diag_over_rank1": float(torch.log(diagonal_error / rank1_error)),
        "heldout_direction_oracle_error": float(direction_oracle_error),
        "heldout_best_rank1_error": float(best_rank1_error),
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("non-finite full-parameter geometry result")
    return result


def _geometry(calibration: torch.Tensor, test: torch.Tensor) -> dict:
    if calibration.ndim != 2 or test.ndim != 2 or calibration.shape[1] != test.shape[1]:
        raise ValueError("gradient matrices must be 2D with a shared parameter dimension")
    gc = calibration.double()
    gt = test.double()
    nc, nt = gc.shape[0], gt.shape[0]
    mu = gc.mean(dim=0)
    mu_norm_sq = torch.dot(mu, mu)
    if not float(mu_norm_sq) > 0.0:
        raise ValueError("zero calibration mean gradient")
    coefficient = (gc @ mu).square().mean() / mu_norm_sq.square()
    diagonal = gc.square().mean(dim=0)

    test_gram = gt @ gt.T / nt
    fisher_norm_sq = test_gram.square().sum()
    rank1_inner = coefficient * (gt @ mu).square().mean()
    rank1_norm_sq = coefficient.square() * mu_norm_sq.square()
    diagonal_inner = torch.dot(gt.square().mean(dim=0), diagonal)
    diagonal_norm_sq = diagonal.square().sum()
    rank1_residual_sq = (fisher_norm_sq - 2 * rank1_inner + rank1_norm_sq).clamp_min(0)
    diagonal_residual_sq = (fisher_norm_sq - 2 * diagonal_inner + diagonal_norm_sq).clamp_min(0)
    rank1_error = torch.sqrt(rank1_residual_sq / fisher_norm_sq)
    diagonal_error = torch.sqrt(diagonal_residual_sq / fisher_norm_sq)

    test_coefficient = (gt @ mu).square().mean() / mu_norm_sq.square()
    direction_oracle_sq = (
        fisher_norm_sq
        - 2 * test_coefficient * (gt @ mu).square().mean()
        + test_coefficient.square() * mu_norm_sq.square()
    ).clamp_min(0)
    eigenvalues = torch.linalg.eigvalsh(test_gram)
    best_rank1_sq = (fisher_norm_sq - eigenvalues[-1].square()).clamp_min(0)
    direction_oracle_error = torch.sqrt(direction_oracle_sq / fisher_norm_sq)
    best_rank1_error = torch.sqrt(best_rank1_sq / fisher_norm_sq)
    tolerance = 1e-9
    if float(best_rank1_error) > float(direction_oracle_error) + tolerance:
        raise AssertionError("best held-out rank-1 error exceeds direction-constrained oracle")
    if float(direction_oracle_error) > float(rank1_error) + tolerance:
        raise AssertionError("held-out direction oracle exceeds calibration-fitted rank-1")

    result = {
        "calibration_examples": nc,
        "test_examples": nt,
        "parameter_count": gc.shape[1],
        "mean_gradient_norm": float(torch.sqrt(mu_norm_sq)),
        "rank1_coefficient": float(coefficient),
        "diagonal_trace": float(diagonal.sum()),
        "test_fisher_frobenius": float(torch.sqrt(fisher_norm_sq)),
        "rank1_relative_frobenius_error": float(rank1_error),
        "diagonal_relative_frobenius_error": float(diagonal_error),
        "heldout_score_log_diag_over_rank1": float(torch.log(diagonal_error / rank1_error)),
        "heldout_direction_oracle_error": float(direction_oracle_error),
        "heldout_best_rank1_error": float(best_rank1_error),
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise ValueError("non-finite geometry result")
    return result


def _dependencies() -> dict[str, str]:
    paths = (
        Path(__file__),
        PROTOCOL,
        ROOT / "experiments/dllm_rank1_multitask.py",
        ROOT / "experiments/dllm_rank1_transfer.py",
        ROOT / "continual_benchmark.py",
        ROOT / "continual_mdm.py",
        ROOT / "continual_reverse.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def run(args) -> dict:
    started = time.monotonic()
    if args.seed not in (3407, 3408, 3409):
        raise ValueError("R18 seeds are frozen to 3407, 3408, and 3409")
    set_seed(args.seed)
    device = torch.device(args.device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, use_fast=True
    )
    pad_id = int(tokenizer.eos_token_id)
    train_raw = fact_rows(args.reverse_dir, "d2p", "train", 8, 4)
    _, calibration_raw = transfer._split_calibration(train_raw, 10)
    test_raw = fact_rows(args.reverse_dir, "d2p", "test", 8, 4)
    _assert_disjoint(calibration_raw, test_raw)
    train = encode_benchmark_rows(train_raw, tokenizer, 128)
    calibration = encode_benchmark_rows(calibration_raw, tokenizer, 128)
    test = encode_benchmark_rows(
        [{"prompt": row["prompt"], "answer": row["target"]} for row in test_raw],
        tokenizer,
        128,
    )
    if (len(train), len(calibration), len(test)) != (120, 40, 40):
        raise ValueError("unexpected frozen R18 row counts")

    dependencies = _dependencies()
    inputs = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer_sha256": _tree_sha256(args.tokenizer),
        "reverse_data_sha256": _tree_sha256(args.reverse_dir),
        "train_rows_sha256": _records_sha256(train_raw),
        "calibration_rows_sha256": _records_sha256(calibration_raw),
        "test_rows_sha256": _records_sha256(test_raw),
        "calibration_prompts_sha256": _prompt_sha256(calibration_raw),
        "test_prompts_sha256": _prompt_sha256(test_raw),
    }
    model = load_model(args, device)
    all_parameters = trainable_parameters(model, "all")
    initial_moments = _parameter_moments(model)
    training = multitask._train_stage(
        model, train, all_parameters, pad_id, device, args, args.seed + 1000
    )
    learned_moments = _parameter_moments(model)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    named = dict(model.named_parameters())
    missing = [name for name in PROBES if name not in named]
    if missing:
        raise KeyError(f"missing frozen probes: {missing}")
    probes = {name: named[name] for name in PROBES}
    probe_state_sha256 = {name: _tensor_sha256(parameter) for name, parameter in probes.items()}
    full_size, layout = _parameter_layout(model, all_parameters)
    if full_size != 219_050_496:
        raise ValueError(f"unexpected full parameter count: {full_size}")
    probe_layout = {name: layout[name] for name in PROBES}

    calibration_seed = args.seed + 2101
    test_seed = args.seed + 4101
    calibration_first = _full_fisher_pass(
        model, calibration, all_parameters, full_size, probe_layout,
        pad_id, device, calibration_seed,
    )
    calibration_mean = calibration_first["gradient_sum"].div_(len(calibration))
    calibration_diagonal = calibration_first["diagonal_sum"].div_(len(calibration))
    calibration_second = _full_fisher_pass(
        model, calibration, all_parameters, full_size, probe_layout,
        pad_id, device, calibration_seed, mean=calibration_mean,
    )
    repeat_loss_max_abs_difference = max(
        abs(first - second)
        for first, second in zip(calibration_first["losses"], calibration_second["losses"])
    )
    if repeat_loss_max_abs_difference != 0.0:
        raise AssertionError("calibration mask replay changed between Fisher passes")
    test_pass = _full_fisher_pass(
        model, test, all_parameters, full_size, probe_layout,
        pad_id, device, test_seed, mean=calibration_mean, retain_vectors=True,
    )
    test_diagonal = torch.zeros(full_size, dtype=torch.float64)
    for vector in test_pass["vectors"]:
        value64 = vector.double()
        test_diagonal.addcmul_(value64, value64)
        del value64
    test_diagonal.div_(len(test))
    test_gram = _test_gram(test_pass["vectors"], full_size, args.gram_chunk_size)
    full_geometry = _geometry_from_sufficient(
        calibration_mean,
        calibration_diagonal,
        calibration_second["projection_square_sum"] / len(calibration),
        test_diagonal,
        test_pass["projection_square_sum"] / len(test),
        test_gram,
        len(calibration),
        len(test),
    )
    anchor = R16_ANCHORS[args.seed]
    anchor_checks = {
        "training_current_mean_abs_difference": abs(training["current_mean"] - anchor["current_mean"]),
        "training_clip_fraction_abs_difference": abs(training["clip_fraction"] - anchor["clip_fraction"]),
        "rank1_coefficient_relative_difference": abs(
            full_geometry["rank1_ewc_direction_coefficient"] - anchor["rank1_coefficient"]
        ) / anchor["rank1_coefficient"],
        "diagonal_trace_relative_difference": abs(
            full_geometry["diagonal_trace"] - anchor["diagonal_trace"]
        ) / anchor["diagonal_trace"],
    }
    if (
        anchor_checks["training_current_mean_abs_difference"] > 1e-9
        or anchor_checks["training_clip_fraction_abs_difference"] > 0.0
        or anchor_checks["rank1_coefficient_relative_difference"] > 1e-4
        or anchor_checks["diagonal_trace_relative_difference"] > 1e-4
    ):
        raise AssertionError(f"post-Task-1 state does not reproduce R16 anchors: {anchor_checks}")
    slice_geometry = {
        name: _geometry(
            torch.stack(calibration_first["slices"][name]),
            torch.stack(test_pass["slices"][name]),
        )
        for name in PROBES
    }
    slice_scores = [item["heldout_score_log_diag_over_rank1"] for item in slice_geometry.values()]
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "r18_continual_fisher_geometry",
        "role": "exploratory_predeclared_complete_grid",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "host": __import__("os").uname().nodename,
        "dependencies": dependencies,
        "inputs": inputs,
        "protocol": {
            "training_state": "after_task_1_d2p_facts_8_11",
            "objective": "r16_answer_only_independent_bernoulli_allow_empty_importance_weighted",
            "model": 170,
            "full_parameter_training": True,
            "task": "d2p_8-11",
            "steps": 1000,
            "batch_size": 4,
            "learning_rate": 5e-5,
            "clip": 1.0,
            "calibration_source": "designated_task_training_examples",
            "test_source": "heldout_task_test_prompts",
            "calibration_mask_seed": calibration_seed,
            "test_mask_seed": test_seed,
            "prompt_overlap_count": 0,
            "primary_parameter_set": "all_trainable_parameters",
            "secondary_probes": list(PROBES),
            "seed": args.seed,
        },
        "training_state": {
            "r16_anchor": anchor,
            "r16_anchor_checks": anchor_checks,
            "initial_parameter_moments": initial_moments,
            "learned_parameter_moments": learned_moments,
            "probe_tensor_sha256_after_task": probe_state_sha256,
            "training": training,
        },
        "fisher_sampling": {
            "stage": "immediately_after_task_1_training",
            "calibration_examples": len(calibration),
            "test_examples": len(test),
            "calibration_loss_mean": sum(calibration_first["losses"]) / len(calibration),
            "test_loss_mean": sum(test_pass["losses"]) / len(test),
            "calibration_repeat_loss_max_abs_difference": repeat_loss_max_abs_difference,
            "calibration_empty_masks": sum(item["masked_tokens"] == 0 for item in calibration_first["masks"]),
            "test_empty_masks": sum(item["masked_tokens"] == 0 for item in test_pass["masks"]),
            "calibration_masks": calibration_first["masks"],
            "test_masks": test_pass["masks"],
        },
        "full_parameter_geometry": full_geometry,
        "slice_geometry": slice_geometry,
        "summary": {
            "full_parameter_score": full_geometry["heldout_score_log_diag_over_rank1"],
            "mean_slice_score": sum(slice_scores) / len(slice_scores),
            "rank1_slice_wins": sum(score > 0 for score in slice_scores),
            "slice_count": len(slice_scores),
            "slice_scores": slice_scores,
        },
    }
    if _dependencies() != dependencies:
        raise RuntimeError("source/protocol provenance changed during R18 run")
    current_inputs = {
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer_sha256": _tree_sha256(args.tokenizer),
        "reverse_data_sha256": _tree_sha256(args.reverse_dir),
        "train_rows_sha256": _records_sha256(train_raw),
        "calibration_rows_sha256": _records_sha256(calibration_raw),
        "test_rows_sha256": _records_sha256(test_raw),
        "calibration_prompts_sha256": _prompt_sha256(calibration_raw),
        "test_prompts_sha256": _prompt_sha256(test_raw),
    }
    if current_inputs != inputs:
        raise RuntimeError("input provenance changed during R18 run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output), "summary": result["summary"]}, indent=2))
    return result


def _self_check() -> None:
    _assert_disjoint([{"prompt": "cal"}], [{"prompt": "test"}])
    try:
        _assert_disjoint([{"prompt": "same"}], [{"prompt": "same"}])
    except ValueError:
        pass
    else:
        raise AssertionError("prompt-overlap check did not fail")

    generator = torch.Generator().manual_seed(7)
    gc = torch.randn(5, 4, generator=generator)
    gt = torch.randn(6, 4, generator=generator)
    actual = _geometry(gc, gt)
    fcal = gc.double().T @ gc.double() / len(gc)
    ftest = gt.double().T @ gt.double() / len(gt)
    mu = gc.double().mean(dim=0)
    coefficient = (mu @ fcal @ mu) / torch.dot(mu, mu).square()
    rank1 = coefficient * torch.outer(mu, mu)
    diagonal = torch.diag(torch.diag(fcal))
    rank1_error = torch.linalg.matrix_norm(ftest - rank1) / torch.linalg.matrix_norm(ftest)
    diagonal_error = torch.linalg.matrix_norm(ftest - diagonal) / torch.linalg.matrix_norm(ftest)
    assert math.isclose(actual["rank1_relative_frobenius_error"], float(rank1_error), rel_tol=1e-10)
    assert math.isclose(actual["diagonal_relative_frobenius_error"], float(diagonal_error), rel_tol=1e-10)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, ids):
            logits = self.scale * torch.ones((*ids.shape, 8))
            logits[..., 1] = logits[..., 1] + self.scale
            return logits

    row = {"ids": [1, 2, 3], "answer_start": 1, "answer_end": 3}
    model = ToyModel()
    first = torch.Generator().manual_seed(19)
    second = torch.Generator().manual_seed(19)
    audited, audit = _audited_sft_loss(model, row, 0, torch.device("cpu"), first, 1e-3, 1.0)
    direct = transfer._sft_losses(model, [row], 0, torch.device("cpu"), second, 1e-3, 1.0)[0]
    assert torch.equal(audited, direct)
    assert audit["answer_tokens"] == 2 and 0 <= audit["masked_tokens"] <= 2
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=ROOT / "SMDM/data/reverse_experiments/june_version_7921032488")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gram-chunk-size", type=int, default=1048576)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check and args.output is None:
        parser.error("--output is required")
    # R16 Task-1 training values consumed by the imported stage trainer.
    args.model = 170
    args.steps_per_task = 1000
    args.batch_size = 4
    args.lr = 5e-5
    args.clip = 1.0
    args.mask_min = 1e-3
    args.mask_max = 1.0
    args.distill_weight = 1.0
    args.distill_temperature = 1.0
    args.ewc_lambda = 0.0
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args)


if __name__ == "__main__":
    main()
