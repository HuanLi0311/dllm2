#!/usr/bin/env python3
"""Strict summaries for the locked R16 DLLM continual-learning protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3407, 3408, 3409)
VALIDATION_SUMMARY_SHA256 = "b621509724d73059362bd095d9e67a45cdc690da68d84d0b0a37b2e33dc610e4"
PROTOCOLS = {
    "main": {
        "tag": "r16_native_mask_v1",
        "methods": {
            "forward": ("seq", "gd", "rank1", "diagonal", "rank1_gd", "diag_gd", "joint"),
            "reverse": ("seq", "gd", "rank1_gd", "diag_gd"),
        },
        "forward_tasks": ("d2p_8-11", "p2d_12-15", "d2p_16-19", "p2d_20-23"),
        "group_count": 4,
    },
    "fresh": {
        "tag": "r16_fresh_facts_v1",
        "methods": {
            "forward": ("seq", "gd", "rank1_gd", "diag_gd"),
            "reverse": ("seq", "gd", "rank1_gd", "diag_gd"),
        },
        "forward_tasks": ("d2p_24-25", "p2d_26-27", "d2p_28-29"),
        "group_count": 2,
    },
}
METRICS = (
    "final_average_loss",
    "past_task_forgetting",
    "final_average_answer_token_accuracy",
    "final_average_target_containment",
    "final_average_exact_match",
    "final_average_rouge_l_f1",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(member.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with member.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _expected_paths(run_root: Path, spec: dict) -> dict[tuple[str, str, int], Path]:
    return {
        (order, method, seed): run_root / order / f"s{seed}" / f"{method}.json"
        for order, methods in spec["methods"].items()
        for method in methods
        for seed in SEEDS
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(actual: float, expected: float, message: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), message)


def _finite(value, path="root") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite value at {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{path}[{index}]")


def _validate_replay_manifest(items: list[dict], tasks, specs, label: str) -> None:
    _require(len(items) == len(tasks), f"{label}: replay manifest")
    for item, old_task, old_spec in zip(items, tasks, specs):
        group_count = old_spec["group_count"]
        old_facts = range(old_spec["group_start"], old_spec["group_start"] + group_count)
        expected_counts = {str(fact): 64 // group_count for fact in old_facts}
        _require(item["task"] == old_task, f"{label}: replay task identity")
        _require(item["count"] == 64, f"{label}: replay/task")
        _require(item["fact_counts"] == expected_counts, f"{label}: replay fact balance")
        _require(len(item["prompt_sha256"]) == 64, f"{label}: replay prompt hash")
        _require(len(item["selection_sha256"]) == 64, f"{label}: replay selection hash")
        _require(
            set(item["per_fact_prompt_sha256"]) == set(expected_counts)
            and all(len(digest) == 64 for digest in item["per_fact_prompt_sha256"].values()),
            f"{label}: replay per-fact hashes",
        )


def _mean_sem(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
        "values": values,
    }


def _validate_run(payload: dict, order: str, method: str, seed: int, spec: dict) -> None:
    label = f"{order}/{method}/s{seed}"
    _require(payload.get("status") == "ok", f"{label}: status is not ok")
    _require(payload.get("experiment") == "dllm_rank1_multitask", f"{label}: wrong experiment")
    _finite(payload, label)
    metadata = payload["metadata"]
    _require(metadata.get("protocol") == spec["tag"], f"{label}: wrong protocol")
    _require(
        metadata.get("mask_sampling") == "independent_bernoulli_allow_empty_v1",
        f"{label}: wrong mask sampling",
    )
    _require(
        metadata.get("minibatch_sampling") == "separate_current_replay_rng_v1",
        f"{label}: wrong minibatch sampling",
    )
    _require(
        metadata.get("replay_sampling") == "balanced_by_fact_in_source_order",
        f"{label}: wrong replay sampling",
    )
    _require(metadata.get("method") == method, f"{label}: method mismatch")
    _require(metadata.get("order") == order, f"{label}: order mismatch")
    _require(metadata.get("seed") == seed == metadata.get("generation_seed"), f"{label}: seed mismatch")
    _require(metadata.get("clip") == 1.0, f"{label}: clip mismatch")
    _require(metadata.get("eval_mc_samples") == 32, f"{label}: evaluation MC mismatch")
    selected_lambdas = metadata.get("selected_lambdas", {})
    _require(set(selected_lambdas) == {"rank1_gd", "diag_gd"}, f"{label}: selected lambdas")
    expected_lambda = (
        selected_lambdas["rank1_gd"] if method in ("rank1", "rank1_gd")
        else selected_lambdas["diag_gd"] if method in ("diagonal", "diag_gd")
        else 0.0
    )
    _require(metadata.get("ewc_lambda") == expected_lambda, f"{label}: EWC lambda mismatch")
    _require(len(metadata.get("validation_summary_sha256", "")) == 64, f"{label}: validation hash")
    tasks = spec["forward_tasks"] if order == "forward" else tuple(reversed(spec["forward_tasks"]))
    _require(tuple(item["name"] for item in metadata["sequence"]) == tasks, f"{label}: task mismatch")
    _require(len(metadata["task_artifacts"]) == len(tasks), f"{label}: missing task artifacts")
    for artifact, task, task_spec in zip(metadata["task_artifacts"], tasks, metadata["sequence"]):
        _require(artifact["task"] == task, f"{label}: task artifact order mismatch")
        _require(
            (artifact["train_count"], artifact["fisher_count"], artifact["eval_count"])
            == (30 * spec["group_count"], 10 * spec["group_count"], 10 * spec["group_count"]),
            f"{label}: task row counts mismatch",
        )
        facts = range(task_spec["group_start"], task_spec["group_start"] + task_spec["group_count"])
        _require(
            artifact["train_fact_counts"] == {str(fact): 30 for fact in facts},
            f"{label}: per-fact training counts mismatch",
        )
        for key in ("fisher_fact_counts", "eval_fact_counts"):
            _require(
                artifact[key] == {str(fact): 10 for fact in facts},
                f"{label}: per-fact {key} mismatch",
            )
        for key in ("train_sha256", "fisher_sha256", "eval_sha256"):
            _require(len(artifact[key]) == 64, f"{label}: invalid {key}")

    if method == "joint":
        _require(order == "forward", f"{label}: joint must be canonical only")
        _require(set(payload["final_metrics"]) == set(tasks), f"{label}: joint metrics incomplete")
        losses = [payload["final_metrics"][task]["loss"] for task in tasks]
        _close(statistics.mean(losses), payload["summary"]["final_average_loss"], f"{label}: joint mean")
        return

    stages = payload.get("stages", [])
    _require(len(stages) == len(tasks), f"{label}: wrong number of stages")
    uses_gd = method in ("gd", "rank1_gd", "diag_gd")
    uses_ewc = method in ("rank1", "diagonal", "rank1_gd", "diag_gd")
    for index, (stage, task) in enumerate(zip(stages, tasks)):
        _require(stage["stage"] == index and stage["task"] == task, f"{label}: stage mismatch")
        _require(set(stage["metrics"]) == set(tasks[: index + 1]), f"{label}: stage metrics incomplete")
        expected_replay = index * 64 if uses_gd else 0
        _require(stage["replay_examples"] == expected_replay, f"{label}: replay count mismatch")
        replay = stage["replay"]
        _require(len(replay["per_task"]) == (index if uses_gd else 0), f"{label}: replay manifest")
        if uses_gd and index:
            _validate_replay_manifest(
                replay["per_task"], tasks[:index], metadata["sequence"][:index], label
            )
            _require(len(replay["generated_rows_sha256"]) == 64, f"{label}: replay hash")
        else:
            _require(replay["generated_rows_sha256"] is None, f"{label}: unexpected replay hash")
        _require((stage["fisher"] is not None) == (uses_ewc and index + 1 < len(tasks)), f"{label}: Fisher stage")

    final = stages[-1]["metrics"]
    losses = [final[task]["loss"] for task in tasks]
    learned = [stages[index]["metrics"][task]["loss"] for index, task in enumerate(tasks)]
    forgetting = [after - before for after, before in zip(losses, learned)]
    summary = payload["summary"]
    _close(statistics.mean(losses), summary["final_average_loss"], f"{label}: final mean")
    _close(statistics.mean(forgetting[:-1]), summary["past_task_forgetting"], f"{label}: forgetting")
    _require(summary["final_task_losses"] == losses, f"{label}: final task losses")
    _require(summary["losses_when_learned"] == learned, f"{label}: learned losses")
    _require(summary["task_forgetting"] == forgetting, f"{label}: task forgetting")


def _audit_common(runs: dict) -> dict:
    first = next(iter(runs.values()))
    source = first["source_sha256"]
    dependencies = first["dependency_sha256"]
    metadata = first["metadata"]
    checkpoint_hash = metadata["checkpoint_sha256"]
    tokenizer_hash = metadata["tokenizer_sha256"]
    data_hash = metadata["reverse_data_sha256"]
    validation_hash = metadata["validation_summary_sha256"]
    task_hashes = {}
    replay_hashes = {}
    for payload in runs.values():
        current = payload["metadata"]
        _require(payload["source_sha256"] == source, "mixed multitask source hashes")
        _require(payload["dependency_sha256"] == dependencies, "mixed dependency hashes")
        _require(current["checkpoint_sha256"] == checkpoint_hash, "mixed checkpoints")
        _require(current["tokenizer_sha256"] == tokenizer_hash, "mixed tokenizers")
        _require(current["reverse_data_sha256"] == data_hash, "mixed reverse data")
        _require(current["validation_summary_sha256"] == validation_hash, "mixed validation summaries")
        for artifact in current["task_artifacts"]:
            hashes = tuple(artifact[key] for key in ("train_sha256", "fisher_sha256", "eval_sha256"))
            if artifact["task"] in task_hashes:
                _require(task_hashes[artifact["task"]] == hashes, f"row hash mismatch: {artifact['task']}")
            task_hashes[artifact["task"]] = hashes
        for stage in payload.get("stages", []):
            for artifact in stage["replay"]["per_task"]:
                hashes = (
                    artifact["prompt_sha256"],
                    artifact["selection_sha256"],
                    tuple(sorted(artifact["per_fact_prompt_sha256"].items())),
                )
                if artifact["task"] in replay_hashes:
                    _require(
                        replay_hashes[artifact["task"]] == hashes,
                        f"replay prompt hash mismatch: {artifact['task']}",
                    )
                replay_hashes[artifact["task"]] = hashes

    _require(source == _sha256(ROOT / "experiments/dllm_rank1_multitask.py"), "stale multitask source")
    for name, digest in dependencies.items():
        _require(digest == _sha256(ROOT / name), f"stale dependency: {name}")
    _require(checkpoint_hash == _sha256(Path(metadata["checkpoint"])), "stale checkpoint")
    _require(tokenizer_hash == _tree_sha256(Path(metadata["tokenizer"])), "stale tokenizer")
    _require(data_hash == _tree_sha256(Path(metadata["reverse_dir"])), "stale reverse data")
    _require(validation_hash == _sha256(Path(metadata["validation_summary"])), "stale validation summary")
    _require(validation_hash == VALIDATION_SUMMARY_SHA256, "wrong frozen validation summary")
    return {
        "source_sha256": source,
        "dependency_sha256": dependencies,
        "checkpoint_sha256": checkpoint_hash,
        "tokenizer_sha256": tokenizer_hash,
        "reverse_data_sha256": data_hash,
        "validation_summary_sha256": validation_hash,
        "selected_lambdas": metadata["selected_lambdas"],
        "task_row_sha256": task_hashes,
        "replay_prompt_sha256": replay_hashes,
    }


def _paired(runs: dict, order: str, left: str, right: str, metric) -> dict:
    values = [metric(runs[(order, left, seed)]) - metric(runs[(order, right, seed)]) for seed in SEEDS]
    return {**_mean_sem(values), "wins": sum(value < 0 for value in values)}


def summarize(run_root: Path, spec: dict) -> dict:
    expected = _expected_paths(run_root, spec)
    found = set(run_root.rglob("*.json")) if run_root.exists() else set()
    _require(found == set(expected.values()), f"run matrix mismatch: missing={set(expected.values()) - found}, extra={found - set(expected.values())}")
    runs = {}
    for key, path in expected.items():
        payload = json.loads(path.read_text())
        _validate_run(payload, *key, spec)
        runs[key] = payload
    audit = _audit_common(runs)

    aggregate = {}
    trajectories = {}
    per_task = {}
    mechanism = {}
    for order, methods in spec["methods"].items():
        aggregate[order] = {}
        trajectories[order] = {}
        per_task[order] = {}
        mechanism[order] = {}
        tasks = spec["forward_tasks"] if order == "forward" else tuple(reversed(spec["forward_tasks"]))
        for method in methods:
            rows = [runs[(order, method, seed)] for seed in SEEDS]
            aggregate[order][method] = {
                metric: _mean_sem([row["summary"][metric] for row in rows])
                for metric in METRICS
                if metric in rows[0]["summary"]
            }
            aggregate[order][method]["final_task_loss"] = _mean_sem([
                row["summary"]["final_task_losses"][-1] for row in rows
            ])
            per_task[order][method] = {}
            if method == "joint":
                for task in tasks:
                    per_task[order][method][task] = {
                        "final_loss": _mean_sem([
                            row["final_metrics"][task]["loss"] for row in rows
                        ])
                    }
                continue
            mechanism[order][method] = [
                {
                    metric: _mean_sem([
                        row["stages"][stage]["training"][metric] for row in rows
                    ])
                    for metric in (
                        "current_mean",
                        "distill_loss_weighted_mean",
                        "ewc_loss_mean",
                        "penalty_max",
                        "gradient_norm_max",
                        "clip_fraction",
                    )
                }
                for stage in range(len(tasks))
            ]
            trajectories[order][method] = [
                _mean_sem([
                    statistics.mean(row["stages"][stage]["metrics"][task]["loss"] for task in tasks[: stage + 1])
                    for row in rows
                ])
                for stage in range(len(tasks))
            ]
            for index, task in enumerate(tasks):
                per_task[order][method][task] = {
                    "final_loss": _mean_sem([
                        row["summary"]["final_task_losses"][index] for row in rows
                    ]),
                    "learned_loss": _mean_sem([
                        row["summary"]["losses_when_learned"][index] for row in rows
                    ]),
                    "forgetting": _mean_sem([
                        row["summary"]["task_forgetting"][index] for row in rows
                    ]),
                }

    paired = {}
    for order in spec["methods"]:
        paired[order] = {}
        for left, right in (("rank1_gd", "gd"), ("rank1_gd", "diag_gd"), ("diag_gd", "gd")):
            paired[order][f"{left}_minus_{right}"] = {
                "final_average_loss": _paired(
                    runs, order, left, right, lambda row: row["summary"]["final_average_loss"]
                ),
                "past_task_forgetting": _paired(
                    runs, order, left, right, lambda row: row["summary"]["past_task_forgetting"]
                ),
                "final_task_loss": _paired(
                    runs, order, left, right, lambda row: row["summary"]["final_task_losses"][-1]
                ),
            }

    return {
        "schema_version": 2,
        "protocol": spec["tag"],
        "run_count": len(runs),
        "audit": audit,
        "aggregate": aggregate,
        "trajectories": trajectories,
        "per_task": per_task,
        "mechanism": mechanism,
        "paired": paired,
    }


def _plot(summary: dict, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "seq": "Seq.", "gd": "GD", "rank1": "R1", "diagonal": "Diag.",
        "rank1_gd": "R1+GD", "diag_gd": "Diag.+GD", "joint": "Joint",
    }
    colors = {
        "seq": "#9ca3af", "gd": "#3b82f6", "rank1": "#d14d64",
        "diagonal": "#e0a025", "rank1_gd": "#d14d64",
        "diag_gd": "#e0a025", "joint": "#3a8f6b",
    }
    core = ("seq", "gd", "rank1_gd", "diag_gd")

    def endpoint(axis, order):
        methods = tuple(summary["aggregate"][order])
        for index, method in enumerate(methods):
            group = summary["aggregate"][order][method]["final_average_loss"]
            axis.scatter(np.array([-0.10, 0, 0.10]) + index, group["values"], color=colors[method], s=15, alpha=0.72)
            axis.errorbar(index, group["mean"], yerr=group["sem"], fmt="_", color="black", markersize=13, capsize=3)
        axis.set_xticks(range(len(methods)), [labels[method] for method in methods], rotation=22, ha="right")
        axis.set_title(f"{'Forward' if order == 'forward' else 'Reverse'} order")
        axis.set_ylabel(r"Final average loss $\downarrow$")
        axis.grid(axis="y", alpha=0.2)

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    endpoint(axes[0, 0], "forward")
    endpoint(axes[0, 1], "reverse")

    x = np.arange(len(core))
    width = 0.34
    for offset, order, hatch in ((-width / 2, "forward", None), (width / 2, "reverse", "//")):
        groups = [summary["aggregate"][order][method]["past_task_forgetting"] for method in core]
        axes[1, 0].bar(
            x + offset, [group["mean"] for group in groups], width,
            yerr=[group["sem"] for group in groups], color=[colors[method] for method in core],
            alpha=0.72, hatch=hatch, capsize=2, edgecolor="0.3", linewidth=0.5,
            label=order.capitalize(),
        )
    axes[1, 0].axhline(0, color="0.35", linewidth=1, linestyle="--")
    axes[1, 0].set_xticks(x, [labels[method] for method in core])
    axes[1, 0].set_ylabel(r"Past-task forgetting $\downarrow$")
    axes[1, 0].set_title("Retention across orders")
    axes[1, 0].legend(frameon=False, ncol=2)
    axes[1, 0].grid(axis="y", alpha=0.2)

    comparisons = ("rank1_gd_minus_gd", "diag_gd_minus_gd")
    comparison_labels = ("R1+GD $-$ GD", "Diag.+GD $-$ GD")
    order_colors = {"forward": "#335f7a", "reverse": "#8c5870"}
    for comparison_index, comparison in enumerate(comparisons):
        for order_index, order in enumerate(("forward", "reverse")):
            group = summary["paired"][order][comparison]["final_average_loss"]
            center = comparison_index + (-0.13 if order_index == 0 else 0.13)
            axes[1, 1].scatter(
                np.array([-0.04, 0, 0.04]) + center, group["values"],
                color=order_colors[order], s=15, alpha=0.75,
            )
            axes[1, 1].errorbar(center, group["mean"], yerr=group["sem"], fmt="_", color="black", markersize=12, capsize=3)
    axes[1, 1].axhline(0, color="0.35", linewidth=1, linestyle="--")
    axes[1, 1].set_xticks(range(len(comparisons)), comparison_labels)
    axes[1, 1].set_ylabel(r"Paired final-loss difference $\downarrow$")
    axes[1, 1].set_title("Per-seed paired effects")
    axes[1, 1].grid(axis="y", alpha=0.2)
    for order, color in order_colors.items():
        axes[1, 1].scatter([], [], color=color, label=order.capitalize())
    axes[1, 1].legend(frameon=False, ncol=2)

    stem = "multitask_results" if summary["protocol"] == "r16_native_mask_v1" else "fresh_results"
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    for axis, order in zip(axes, ("forward", "reverse")):
        for method in core:
            groups = summary["trajectories"][order][method]
            stages = np.arange(1, len(groups) + 1)
            axis.errorbar(
                stages, [group["mean"] for group in groups],
                yerr=[group["sem"] for group in groups], marker="o", markersize=3,
                capsize=2, color=colors[method], label=labels[method],
            )
        axis.set_xticks(stages)
        axis.set_xlabel("Tasks observed")
        axis.set_title(f"{'Forward' if order == 'forward' else 'Reverse'} order")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Average loss on observed tasks")
    axes[1].legend(frameon=False, ncol=2)
    trajectory_stem = "multitask_trajectories" if stem == "multitask_results" else "fresh_trajectories"
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{trajectory_stem}.{suffix}", dpi=220)
    plt.close(fig)


def _tex_value(group: dict, signed: bool = False, digits: int = 3) -> str:
    mean = f"{group['mean']:+.{digits}f}" if signed else f"{group['mean']:.{digits}f}"
    return rf"${mean}_{{\pm {group['sem']:.{digits}f}}}$"


def _tex_scientific(group: dict) -> str:
    if group["mean"] == 0.0 and group["sem"] == 0.0:
        return "$0$"
    magnitude = max(abs(group["mean"]), abs(group["sem"]))
    exponent = math.floor(math.log10(magnitude))
    scale = 10.0 ** exponent
    return (
        rf"${group['mean'] / scale:.2f}_{{\pm {group['sem'] / scale:.2f}}}"
        rf"\!\times\!10^{{{exponent}}}$"
    )


def _write_tables(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "r16" if summary["protocol"] == "r16_native_mask_v1" else "fresh"
    method_labels = {
        "seq": "Sequential", "gd": "GD", "rank1": "Rank-1",
        "diagonal": "Diagonal", "rank1_gd": "Rank-1 + GD",
        "diag_gd": "Diagonal + GD", "joint": "Joint",
    }
    order_labels = {"forward": "Forward", "reverse": "Reverse"}

    rows = []
    for order, methods in summary["aggregate"].items():
        for method, metrics in methods.items():
            forgetting = "--" if "past_task_forgetting" not in metrics else _tex_value(metrics["past_task_forgetting"], True)
            final_task = "--" if method == "joint" else _tex_value(metrics["final_task_loss"])
            rows.append(
                f"{order_labels[order]} & {method_labels[method]} & "
                f"{_tex_value(metrics['final_average_loss'])} & {forgetting} & "
                f"{final_task} \\\\"
            )
    main = "\n".join((
        r"\begin{table}[t]", r"\centering",
        r"\caption{Continual-learning endpoints (mean $\pm$ SEM over three paired seeds; lower is better). Past-task forgetting excludes the final task.}",
        rf"\label{{tab:{prefix}-main}}", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Order & Method & Final avg. loss & Past forgetting & Final-task loss\\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ))
    (output_dir / f"{prefix}_main.tex").write_text(main)

    rows = []
    for order, methods in summary["aggregate"].items():
        for method, metrics in methods.items():
            rows.append(
                f"{order_labels[order]} & {method_labels[method]} & "
                f"{_tex_value(metrics['final_average_answer_token_accuracy'])} & "
                f"{_tex_value(metrics['final_average_exact_match'])} & "
                f"{_tex_value(metrics['final_average_target_containment'])} & "
                f"{_tex_value(metrics['final_average_rouge_l_f1'])} \\\\"
            )
    secondary = "\n".join((
        r"\begin{table}[t]", r"\centering",
        r"\caption{Final behavioral endpoints (mean $\pm$ SEM; higher is better).}",
        rf"\label{{tab:{prefix}-secondary}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Order & Method & Token acc. & Exact match & Containment & ROUGE-L\\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ))
    (output_dir / f"{prefix}_secondary.tex").write_text(secondary)

    rows = []
    core = ("seq", "gd", "rank1_gd", "diag_gd")
    for order in ("forward", "reverse"):
        for method in core:
            metrics = summary["mechanism"][order][method][-1]
            rows.append(
                f"{order_labels[order]} & {method_labels[method]} & "
                f"{_tex_value(metrics['current_mean'])} & "
                f"{_tex_value(metrics['distill_loss_weighted_mean'])} & "
                f"{_tex_scientific(metrics['ewc_loss_mean'])} & "
                f"{_tex_value(metrics['clip_fraction'])} \\\\"
            )
    mechanism = "\n".join((
        r"\begin{table}[t]", r"\centering",
        r"\caption{Final-stage optimization diagnostics. Current, GD, and EWC are weighted per-step objective components; clip is the fraction of steps whose pre-clip gradient norm exceeded 1.}",
        rf"\label{{tab:{prefix}-mechanism}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Order & Method & Current & GD & EWC & Clip fraction\\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ))
    (output_dir / f"{prefix}_mechanism.tex").write_text(mechanism)

    tasks = list(summary["per_task"]["forward"]["seq"])
    rows = []
    for order in ("forward", "reverse"):
        order_tasks = tasks if order == "forward" else list(reversed(tasks))
        for method in core:
            cells = []
            for task in order_tasks:
                metrics = summary["per_task"][order][method][task]
                cells.append(
                    rf"${metrics['learned_loss']['mean']:.2f}\!\to\!{metrics['final_loss']['mean']:.2f}"
                    rf"\ ({metrics['forgetting']['mean']:+.2f})$"
                )
            rows.append(f"{order_labels[order]} & {method_labels[method]} & " + " & ".join(cells) + r" \\")
    columns = "ll" + "r" * len(tasks)
    per_task = "\n".join((
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-task learned-time $\to$ final held-out loss (forgetting), averaged over seeds. Columns follow the encountered order.}",
        rf"\label{{tab:{prefix}-per-task}}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
        rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
        "Order & Method & " + " & ".join(f"Task {index + 1}" for index in range(len(tasks))) + r"\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ))
    (output_dir / f"{prefix}_per_task.tex").write_text(per_task)

def _self_check() -> None:
    assert len(_expected_paths(Path("x"), PROTOCOLS["main"])) == 33
    assert len(_expected_paths(Path("x"), PROTOCOLS["fresh"])) == 24
    values = _mean_sem([1.0, 2.0, 3.0])
    assert values["mean"] == 2.0 and math.isclose(values["sem"], 1 / math.sqrt(3))
    facts = range(8, 12)
    good = [{
        "task": "d2p_8-11",
        "count": 64,
        "fact_counts": {str(fact): 16 for fact in facts},
        "prompt_sha256": "0" * 64,
        "selection_sha256": "1" * 64,
        "per_fact_prompt_sha256": {str(fact): "2" * 64 for fact in facts},
    }]
    _validate_replay_manifest(
        good, ["d2p_8-11"], [{"group_start": 8, "group_count": 4}], "self-check"
    )
    bad = [{**good[0], "fact_counts": {"8": 30, "9": 30, "10": 4, "11": 0}}]
    try:
        _validate_replay_manifest(
            bad, ["d2p_8-11"], [{"group_start": 8, "group_count": 4}], "self-check"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unbalanced replay must fail closed")
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/r16_native_mask/final")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/r16_native_mask/summary.json")
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="main")
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    result = summarize(args.run_root, PROTOCOLS[args.protocol])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.figure_dir:
        _plot(result, args.figure_dir)
    if args.table_dir:
        _write_tables(result, args.table_dir)
    print(json.dumps({"status": "ok", "run_count": result["run_count"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
