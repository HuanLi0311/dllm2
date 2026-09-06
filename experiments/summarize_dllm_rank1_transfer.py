#!/usr/bin/env python3
"""Audit the R16 two-task validation sweep."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3407, 3408, 3409)
LAMBDAS = (1e3, 1e4, 1e5, 1e6, 1e7)
METHODS = ("rank1_gd", "diag_gd")


def _lambda_name(value: float) -> str:
    return f"{value:.0e}".replace("e+0", "e").replace("e+", "e")


def _tex_lambda(value: float) -> str:
    return rf"10^{{{int(math.log10(value))}}}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value, path="root") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite value at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def _mean_sem(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)),
        "values": values,
    }


def _metrics(result: dict) -> dict:
    return {
        "final_average_loss": (result["a_loss_after_b"] + result["b_loss_after_b"]) / 2,
        "task_a_forgetting": result["a_loss_forgetting"],
        "final_task_loss": result["b_loss_after_b"],
        "task_a_answer_token_accuracy": result["a_answer_token_accuracy_after_b"],
        "final_task_answer_token_accuracy": result["b_answer_token_accuracy_after_b"],
        "gradient_norm_max": result["b_training"]["gradient_norm_max"],
        "clip_fraction": result["b_training"]["clip_fraction"],
        "ewc_loss_mean": result["b_training"]["ewc_loss_mean"],
    }


def _load_result(
    path: Path,
    method: str,
    seed: int,
    ewc_lambda: float,
    a_group_start: int = 0,
    b_group_start: int = 2,
) -> dict:
    result = json.loads(path.read_text())
    _require(result.get("status") == "ok", f"{path}: status")
    _require(result.get("experiment") == "dllm_rank1_transfer", f"{path}: experiment")
    _finite(result, str(path))
    metadata = result["metadata"]
    config = result["config"]
    expected_metadata = {
        "mask_sampling": "independent_bernoulli_allow_empty_v1",
        "minibatch_sampling": "separate_current_replay_rng_v1",
        "task_a": "d2p", "task_b": "p2d",
        "a_group_start": a_group_start, "b_group_start": b_group_start, "group_count": 2,
        "calibration_per_fact": 10, "fisher_source": "task_a_training_examples",
        "trainable": "all", "batch_size": 4, "a_steps": 1000,
        "mask_min": 1e-3, "mask_max": 1.0, "lr": 5e-5,
        "seed": seed, "replay_prompts": 64,
    }
    expected_config = {
        "method": method, "seed": seed, "generation_seed": seed,
        "b_steps": 1000, "eval_batch_size": 4, "eval_mc_samples": 32,
        "distill_weight": 1.0, "distill_temperature": 1.0,
        "ewc_lambda": ewc_lambda, "clip": 1.0,
    }
    _require(
        all(metadata.get(key) == value for key, value in expected_metadata.items()),
        f"{path}: metadata mismatch",
    )
    _require(
        all(config.get(key) == value for key, value in expected_config.items()),
        f"{path}: config mismatch",
    )
    _require(result["fisher"]["calibration_examples"] == 20, f"{path}: Fisher count")
    _require(result["ewc_lambda_effective"] == (ewc_lambda if method in METHODS else 0.0), f"{path}: lambda")
    return result


def _load_prepare(path: Path, seed: int, a_group_start: int = 0, b_group_start: int = 2) -> dict:
    result = json.loads(path.read_text())
    _require(result.get("status") == "ok", f"{path}: status")
    metadata = result["metadata"]
    _require(metadata.get("seed") == seed, f"{path}: seed")
    _require(metadata.get("a_group_start") == a_group_start, f"{path}: Task-A facts")
    _require(metadata.get("b_group_start") == b_group_start, f"{path}: Task-B facts")
    _require(metadata.get("mask_sampling") == "independent_bernoulli_allow_empty_v1", f"{path}: mask sampling")
    _require(metadata.get("minibatch_sampling") == "separate_current_replay_rng_v1", f"{path}: minibatch sampling")
    _require(result["fisher"]["calibration_examples"] == 20, f"{path}: Fisher count")
    return result


def _aggregate(rows: list[dict]) -> dict:
    metrics = _metrics(rows[0])
    return {
        metric: _mean_sem([_metrics(row)[metric] for row in rows])
        for metric in metrics
    }


def summarize_validation(run_root: Path) -> dict:
    expected = {
        run_root / f"s{seed}" / name
        for seed in SEEDS
        for name in (
            "prepare.json", "seq.json", "gd.json",
            *(f"{method}_l{_lambda_name(ewc_lambda)}.json" for method in METHODS for ewc_lambda in LAMBDAS),
        )
    }
    found = set(run_root.rglob("*.json")) if run_root.exists() else set()
    _require(found == expected, f"validation matrix mismatch: missing={expected - found}, extra={found - expected}")

    prepares = [_load_prepare(run_root / f"s{seed}/prepare.json", seed) for seed in SEEDS]
    results = {}
    for seed in SEEDS:
        results[("seq", 0.0, seed)] = _load_result(run_root / f"s{seed}/seq.json", "seq", seed, 0.0)
        results[("gd", 0.0, seed)] = _load_result(run_root / f"s{seed}/gd.json", "gd", seed, 0.0)
        for method in METHODS:
            for ewc_lambda in LAMBDAS:
                name = f"{method}_l{_lambda_name(ewc_lambda)}.json"
                results[(method, ewc_lambda, seed)] = _load_result(
                    run_root / f"s{seed}" / name, method, seed, ewc_lambda
                )

    source_hashes = {result["source_sha256"] for result in results.values()}
    checkpoint_hashes = {result["metadata"]["checkpoint_sha256"] for result in results.values()}
    _require(len(source_hashes) == len(checkpoint_hashes) == 1, "mixed source or checkpoint hashes")
    aggregate = {
        "seq": _aggregate([results[("seq", 0.0, seed)] for seed in SEEDS]),
        "gd": _aggregate([results[("gd", 0.0, seed)] for seed in SEEDS]),
    }
    selected = {}
    for method in METHODS:
        aggregate[method] = {
            _lambda_name(ewc_lambda): _aggregate([results[(method, ewc_lambda, seed)] for seed in SEEDS])
            for ewc_lambda in LAMBDAS
        }
        winner = min(
            LAMBDAS,
            key=lambda value: aggregate[method][_lambda_name(value)]["final_average_loss"]["mean"],
        )
        selected[method] = {
            "ewc_lambda": winner,
            "selection_endpoint": "mean final average held-out loss over three paired validation seeds",
            "metrics": aggregate[method][_lambda_name(winner)],
        }

    return {
        "schema_version": 1,
        "protocol": "r16_native_mask_validation_v1",
        "seeds": list(SEEDS),
        "validation_facts": "d2p 0-1 -> p2d 2-3",
        "lambda_grid": list(LAMBDAS),
        "selection_rule": "select each Fisher structure separately by lowest mean final average held-out loss",
        "selected": selected,
        "aggregate": aggregate,
        "fisher": {str(seed): prepares[index]["fisher"] for index, seed in enumerate(SEEDS)},
        "audit": {
            "source_sha256": next(iter(source_hashes)),
            "checkpoint_sha256": next(iter(checkpoint_hashes)),
            "run_count": len(results),
        },
    }


def summarize_confirmation(run_root: Path, validation: dict) -> dict:
    expected = {
        run_root / f"s{seed}" / name
        for seed in SEEDS
        for name in ("prepare.json", "seq.json", "gd.json", "rank1_gd.json", "diag_gd.json")
    }
    found = set(run_root.rglob("*.json")) if run_root.exists() else set()
    _require(found == expected, f"confirmation matrix mismatch: missing={expected - found}, extra={found - expected}")
    selected_lambdas = {
        method: validation["selected"][method]["ewc_lambda"] for method in METHODS
    }
    prepares = {
        seed: _load_prepare(run_root / f"s{seed}/prepare.json", seed, 4, 6)
        for seed in SEEDS
    }
    results = {}
    for seed in SEEDS:
        for method in ("seq", "gd"):
            results[(method, seed)] = _load_result(
                run_root / f"s{seed}/{method}.json", method, seed, 0.0, 4, 6
            )
        for method in METHODS:
            results[(method, seed)] = _load_result(
                run_root / f"s{seed}/{method}.json",
                method,
                seed,
                selected_lambdas[method],
                4,
                6,
            )
    source_hashes = {result["source_sha256"] for result in results.values()}
    checkpoint_hashes = {result["metadata"]["checkpoint_sha256"] for result in results.values()}
    _require(source_hashes == {validation["audit"]["source_sha256"]}, "validation/confirmation source mismatch")
    _require(checkpoint_hashes == {validation["audit"]["checkpoint_sha256"]}, "validation/confirmation checkpoint mismatch")
    aggregate = {
        method: _aggregate([results[(method, seed)] for seed in SEEDS])
        for method in ("seq", "gd", *METHODS)
    }
    paired = {}
    for left, right in (("rank1_gd", "gd"), ("rank1_gd", "diag_gd"), ("diag_gd", "gd")):
        paired[f"{left}_minus_{right}"] = {}
        for metric in ("final_average_loss", "task_a_forgetting", "final_task_loss"):
            differences = [
                _metrics(results[(left, seed)])[metric] - _metrics(results[(right, seed)])[metric]
                for seed in SEEDS
            ]
            paired[f"{left}_minus_{right}"][metric] = {
                **_mean_sem(differences),
                "wins": sum(value < 0 for value in differences),
            }
    return {
        "facts": "d2p 4-5 -> p2d 6-7",
        "selected_lambdas": selected_lambdas,
        "aggregate": aggregate,
        "paired": paired,
        "fisher": {str(seed): prepares[seed]["fisher"] for seed in SEEDS},
        "run_count": len(results),
    }


def _plot(summary: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"seq": "#9ca3af", "gd": "#3b82f6", "rank1_gd": "#d14d64", "diag_gd": "#e0a025"}
    labels = {"seq": "Sequential", "gd": "GD", "rank1_gd": "Rank-1 + GD", "diag_gd": "Diagonal + GD"}

    fig, axis = plt.subplots(figsize=(6.3, 3.4), constrained_layout=True)
    gd = summary["aggregate"]["gd"]["final_average_loss"]
    axis.axhline(gd["mean"], color=colors["gd"], linestyle="--", label="GD")
    axis.fill_between(LAMBDAS, gd["mean"] - gd["sem"], gd["mean"] + gd["sem"], color=colors["gd"], alpha=0.12)
    for method, marker in (("rank1_gd", "o"), ("diag_gd", "s")):
        groups = [summary["aggregate"][method][_lambda_name(value)]["final_average_loss"] for value in LAMBDAS]
        axis.errorbar(
            LAMBDAS,
            [group["mean"] for group in groups],
            yerr=[group["sem"] for group in groups],
            marker=marker,
            capsize=3,
            color=colors[method],
            label=labels[method],
        )
        for value, group in zip(LAMBDAS, groups):
            axis.scatter([value / 1.12, value, value * 1.12], group["values"], color=colors[method], s=12, alpha=0.42)
    axis.set_xscale("log")
    axis.set_xlabel(r"EWC strength $\lambda$")
    axis.set_ylabel("Final average held-out loss")
    axis.set_title("Three-seed validation sweep")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"validation_lambda_sweep.{suffix}", dpi=220)
    plt.close(fig)

    if "confirmation" not in summary:
        return
    aggregate = summary["confirmation"]["aggregate"]
    methods = ("seq", "gd", "rank1_gd", "diag_gd")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "final_average_loss", "Final average loss"),
        (axes[1], "task_a_forgetting", "Task-A forgetting"),
    ):
        for index, method in enumerate(methods):
            group = aggregate[method][metric]
            axis.scatter([index - 0.10, index, index + 0.10], group["values"], color=colors[method], s=20, alpha=0.7)
            axis.errorbar(index, group["mean"], yerr=group["sem"], fmt="_", color="black", markersize=14, capsize=3)
        axis.set_xticks(range(len(methods)), [labels[method] for method in methods], rotation=18, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[1].axhline(0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Held-out loss")
    axes[1].set_ylabel(r"$L_A^{\mathrm{after}\ B}-L_A^{\mathrm{after}\ A}$")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"two_task_confirmation.{suffix}", dpi=220)
    plt.close(fig)


def _tex_number(group: dict, signed: bool = False) -> str:
    mean = f"{group['mean']:+.3f}" if signed else f"{group['mean']:.3f}"
    return rf"${mean}_{{\pm {group['sem']:.3f}}}$"


def _write_table(summary: dict, output_dir: Path) -> None:
    if "confirmation" not in summary:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = summary["confirmation"]["aggregate"]
    labels = {"seq": "Sequential", "gd": "GD", "rank1_gd": "Rank-1 + GD", "diag_gd": "Diagonal + GD"}
    rows = []
    for method in ("seq", "gd", "rank1_gd", "diag_gd"):
        group = aggregate[method]
        rows.append(
            f"{labels[method]} & {_tex_number(group['final_average_loss'])} & "
            f"{_tex_number(group['task_a_forgetting'], True)} & {_tex_number(group['final_task_loss'])} \\\\"
        )
    rank_lambda = _tex_lambda(summary["selected"]["rank1_gd"]["ewc_lambda"])
    diag_lambda = _tex_lambda(summary["selected"]["diag_gd"]["ewc_lambda"])
    table = "\n".join((
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Two-task confirmation on separate facts (mean $\pm$ SEM over three paired seeds). Validation selected Rank-1 $\lambda={rank_lambda}$ and diagonal $\lambda={diag_lambda}$. Lower is better.}}",
        r"\label{tab:two-task}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Final average loss & Task-A forgetting & Task-B loss\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ))
    (output_dir / "two_task.tex").write_text(table)


def _self_check() -> None:
    values = _mean_sem([1.0, 2.0, 3.0])
    assert values["mean"] == 2.0 and math.isclose(values["sem"], 1 / math.sqrt(3))
    assert [_lambda_name(value) for value in LAMBDAS] == ["1e3", "1e4", "1e5", "1e6", "1e7"]
    assert _tex_lambda(1e6) == "10^{6}"
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/r16_native_mask/validation")
    parser.add_argument("--confirmation-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/r16_native_mask/validation_summary.json")
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    result = summarize_validation(args.run_root)
    if args.confirmation_root:
        result["confirmation"] = summarize_confirmation(args.confirmation_root, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure_dir:
        _plot(result, args.figure_dir)
    if args.table_dir:
        _write_table(result, args.table_dir)
    print(json.dumps({"status": "ok", "output": str(args.output), "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
