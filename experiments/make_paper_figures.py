#!/usr/bin/env python3
"""Generate the paper's figures directly from raw experiment envelopes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


MODEL_LABELS = {170: "219M", 1028: "1.14B"}
ROOT = Path(__file__).parents[1].resolve()
SOFT_BLUE = "#1f77b4"
SOFT_ORANGE = "#ff7f0e"
SOFT_NEUTRAL = "#f7f5f2"
PLOT_TEXT = "#173042"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def _mean_ci(values):
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    # ponytail: only n=3 and n=5 are used; add scipy if arbitrary confidence intervals become necessary.
    critical = {3: 4.303, 5: 2.776}.get(len(values), 1.96)
    return mean, critical * statistics.stdev(values) / math.sqrt(len(values))


def _wilson_ci(wins, trials, z=1.96):
    proportion = wins / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _soft_diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "soft_blue_orange", [SOFT_ORANGE, SOFT_NEUTRAL, SOFT_BLUE]
    )


def _cell_text(value):
    return f"{0.0 if abs(value) < 0.005 else value:+.2f}"


def _hierarchical_ci(np, records, seed, metric="delta"):
    seeds = sorted({record["seed"] for record in records})
    conditions = sorted({(record["parameter"], record["mask_probability"]) for record in records})
    if metric == "delta":
        value = lambda record: _diagonal_error(record) - _rank1_error(record)
    elif metric == "log_ratio":
        value = lambda record: math.log(_diagonal_error(record) / _rank1_error(record))
    else:
        raise ValueError(f"unknown bootstrap metric: {metric}")
    lookup = {(record["seed"], record["parameter"], record["mask_probability"]): value(record) for record in records}
    matrix = np.array([[lookup[(run_seed, *condition)] for condition in conditions] for run_seed in seeds])
    rng = np.random.default_rng(seed)
    draws = np.empty(10000)
    for index in range(len(draws)):
        seed_indices = rng.integers(0, len(seeds), len(seeds))
        condition_indices = rng.integers(0, len(conditions), len(conditions))
        draws[index] = matrix[seed_indices][:, condition_indices].mean()
    return float(matrix.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _layer(parameter):
    return int(re.search(r"\.h\.(\d+)\.", parameter).group(1))


def _corpus(data, task_id):
    stem = Path(data).stem
    if stem == "gsm8k_tasks":
        return "GSM8K"
    if stem == "geometry_corpora":
        return {0: "MT-Bench", 1: "Reversal"}.get(task_id, f"task-{task_id}")
    return stem


def _rank1_error(record):
    return record.get("mean_rank1_test_relative_frobenius_error", record.get("rank1_relative_frobenius_error"))


def _diagonal_error(record):
    return record.get("diagonal_test_relative_frobenius_error", record.get("diagonal_relative_frobenius_error"))


def _alignment(record):
    return record.get("calibration_mean_top1_absolute_cosine", record.get("mean_top1_absolute_cosine"))


def _load_geometry(paths):
    records, files, failures = [], [], []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        files.append({"path": _relative(path), "sha256": _sha256(path), "status": payload.get("status")})
        if payload.get("status") != "ok":
            failures.append({"path": _relative(path), "error": payload.get("error")})
            continue
        corpus = _corpus(payload["data"], payload.get("task_id"))
        for record in payload["results"]:
            records.append({**record, "corpus": corpus, "source": _relative(path)})
    if not records:
        raise ValueError("no successful geometry records")
    violations = []
    for record in records:
        oracle = record.get("test_oracle_top1_relative_frobenius_error", record.get("oracle_top1_relative_frobenius_error"))
        if oracle > _rank1_error(record) + 1e-10:
            violations.append(record)
    if violations:
        raise ValueError(f"oracle ordering violated in {len(violations)} records")
    return records, files, failures


def _groups(records, keys):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(record)
    return grouped


def _save(fig, output_dir, stem):
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "experiments/make_paper_figures.py", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")


def _model_label(model):
    return MODEL_LABELS.get(model, str(model))


def _score(record):
    """Positive values mean lower independent-test error for mean-gradient rank-1."""
    return math.log(_diagonal_error(record) / _rank1_error(record))


def _heatmaps(plt, np, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus and record["mask_probability"] is not None and record.get("loss_mode") in (None, "native_conditional")]
    models = sorted({record["model_size_m"] for record in selected})
    cells, arrays = [], []
    for model in models:
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_count = max(record["sample_count"] for record in model_records)
        parameters = sorted({record["parameter"] for record in model_records}, key=_layer)
        probabilities = sorted({record["mask_probability"] for record in model_records})
        grouped = _groups(
            [record for record in model_records if record["sample_count"] == sample_count],
            ("parameter", "mask_probability"),
        )
        array = []
        for parameter in parameters:
            row = []
            for probability in probabilities:
                values = grouped[(parameter, probability)]
                score = statistics.fmean(_score(record) for record in values)
                delta = statistics.fmean(_diagonal_error(record) - _rank1_error(record) for record in values)
                row.append(score)
                cells.append({"corpus": corpus, "model_size_m": model, "sample_count": sample_count, "layer": _layer(parameter), "mask_probability": probability, "mean_log_error_ratio": score, "mean_raw_error_difference": delta, "seeds": sorted(record["seed"] for record in values)})
            array.append(row)
        arrays.append((model, sample_count, parameters, probabilities, np.array(array)))
    vmax = max(abs(float(array.min())) for *_, array in arrays)
    vmax = max(vmax, max(abs(float(array.max())) for *_, array in arrays))
    fig, axes = plt.subplots(1, len(arrays), figsize=(8.6, 3.8), constrained_layout=True, squeeze=False)
    image = None
    cmap = _soft_diverging_cmap()
    for panel, (model, sample_count, parameters, probabilities, array) in zip(axes[0], arrays):
        image = panel.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        panel.set_xticks(range(len(probabilities)), [f"{value:.1f}" for value in probabilities])
        panel.set_yticks(range(len(parameters)), [f"layer {_layer(value)}" for value in parameters])
        panel.set_title(f"{_model_label(model)} parameters, n={sample_count}")
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                panel.text(column, row, _cell_text(value), ha="center", va="center", fontsize=11, color=PLOT_TEXT)
    axes[0, 0].set_ylabel("parameter slice")
    fig.supxlabel("mask probability")
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"score $s$")
    _save(fig, output_dir, "geometry_heatmaps")
    plt.close(fig)
    return cells


def _robustness(plt, np, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus and record["mask_probability"] is not None and record.get("loss_mode") in (None, "native_conditional")]
    models = sorted({record["model_size_m"] for record in selected})
    colors = {models[0]: "#1f77b4", models[-1]: "#ff7f0e"}
    sensitivity = []
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), constrained_layout=True)
    for model in models:
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_sizes = sorted({record["sample_count"] for record in model_records})
        means, ci_lows, ci_highs, win_rates = [], [], [], []
        for sample_count in sample_sizes:
            subset = [record for record in model_records if record["sample_count"] == sample_count]
            by_seed = _groups(subset, ("seed",))
            seed_scores = [statistics.fmean(_score(record) for record in values) for values in by_seed.values()]
            mean, half_width = _mean_ci(seed_scores)
            ci_low, ci_high = mean - half_width, mean + half_width
            _, bootstrap_low, bootstrap_high = _hierarchical_ci(np, subset, 20261825 + model + sample_count, "log_ratio")
            by_condition = _groups(subset, ("parameter", "mask_probability"))
            win_rate = statistics.fmean(statistics.fmean(_score(record) for record in values) > 0 for values in by_condition.values())
            means.append(mean)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
            win_rates.append(win_rate)
            sensitivity.append({"corpus": corpus, "model_size_m": model, "sample_count": sample_count, "seed_mean_log_error_ratios": seed_scores, "mean_log_error_ratio": mean, "seed_t_ci95": [ci_low, ci_high], "two_way_bootstrap_ci95_sensitivity": [bootstrap_low, bootstrap_high], "bootstrap_repetitions": 10000, "cell_win_rate": win_rate})
        errors = np.array([[mean - low for mean, low in zip(means, ci_lows)], [high - mean for mean, high in zip(means, ci_highs)]])
        axes[0].errorbar(sample_sizes, means, yerr=errors, marker="o", capsize=4, color=colors[model], label=_model_label(model))
        axes[1].plot(sample_sizes, win_rates, marker="o", color=colors[model], label=_model_label(model))
    axes[0].axhline(0, color="0.3", linewidth=1.2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("examples per Fisher\nestimate")
    axes[0].set_ylabel("mean paired $s$\n(95% probe-seed $t$ CI)")
    axes[0].legend(frameon=False)
    axes[1].axhline(0.5, color="0.5", linewidth=1.2, linestyle="--")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("examples per Fisher\nestimate")
    axes[1].set_ylabel("fraction of layer×mask\ncells won")

    alignment_rows = []
    markers = ["o", "s", "^"]
    for model in models:
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_count = max(record["sample_count"] for record in model_records)
        parameters = sorted({record["parameter"] for record in model_records}, key=_layer)
        grouped = _groups([record for record in model_records if record["sample_count"] == sample_count], ("parameter", "mask_probability"))
        for (parameter, probability), values in grouped.items():
            alignment = statistics.fmean(_alignment(record) for record in values)
            score = statistics.fmean(_score(record) for record in values)
            axes[2].scatter(alignment, score, color=colors[model], marker=markers[parameters.index(parameter)], s=48, alpha=0.8)
            alignment_rows.append({"corpus": corpus, "model_size_m": model, "layer": _layer(parameter), "mask_probability": probability, "mean_top1_absolute_cosine": alignment, "mean_log_error_ratio": score})
    x = np.array([row["mean_top1_absolute_cosine"] for row in alignment_rows])
    y = np.array([row["mean_log_error_ratio"] for row in alignment_rows])
    correlation = float(np.corrcoef(x, y)[0, 1])
    axes[2].axhline(0, color="0.3", linewidth=1.2)
    axes[2].set_xlabel("|cos(mean gradient,\ntop eigenvector)|")
    axes[2].set_ylabel(r"$s=\log(e_{diag}/e_{rank1})$")
    axes[2].text(0.04, 0.94, f"Pearson r={correlation:.2f}", transform=axes[2].transAxes, va="top")
    for marker, label in zip(markers, ("early", "middle", "late")):
        axes[2].scatter([], [], color="0.35", marker=marker, label=label)
    axes[2].legend(loc="lower right")
    for label, panel in zip("ABC", axes):
        panel.text(-0.16, 1.06, label, transform=panel.transAxes, fontweight="bold")
    _save(fig, output_dir, "geometry_robustness")
    plt.close(fig)
    return sensitivity, alignment_rows, correlation


def _spectrum(plt, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus]
    models = sorted({record["model_size_m"] for record in selected})
    fig, axes = plt.subplots(1, len(models), figsize=(7.2, 2.8), constrained_layout=True, squeeze=False)
    rows = []
    for panel, model in zip(axes[0], models):
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_count = max(record["sample_count"] for record in model_records)
        subset = [record for record in model_records if record["sample_count"] == sample_count]
        ranks = sorted({int(rank) for record in subset for rank in record["oracle_rank_relative_frobenius_error"]})
        means = [statistics.fmean(record["oracle_rank_relative_frobenius_error"][str(rank)] for record in subset if str(rank) in record["oracle_rank_relative_frobenius_error"]) for rank in ranks]
        mean_rank1 = statistics.fmean(record["rank1_relative_frobenius_error"] for record in subset)
        diagonal = statistics.fmean(record["diagonal_relative_frobenius_error"] for record in subset)
        panel.plot(ranks, means, marker="o", color="#009E73", label="oracle rank-k")
        panel.axhline(mean_rank1, color="#0072B2", linestyle="--", label="mean-gradient rank-1")
        panel.axhline(diagonal, color="#D55E00", linestyle=":", label="diagonal")
        panel.set_xscale("log", base=2)
        panel.set_xticks(ranks, ranks)
        panel.set_ylim(0, 1)
        panel.set_xlabel("rank k")
        panel.set_title(f"{_model_label(model)}, n={sample_count}")
        rows.append({"corpus": corpus, "model_size_m": model, "sample_count": sample_count, "ranks": ranks, "oracle_error_mean": means, "mean_gradient_rank1_error": mean_rank1, "diagonal_error": diagonal})
    axes[0, 0].set_ylabel("relative Frobenius error")
    axes[0, -1].legend()
    _save(fig, output_dir, "geometry_spectrum")
    plt.close(fig)
    return rows


def _cross_corpus(plt, np, records, output_dir):
    corpora = [name for name in ("GSM8K", "MT-Bench", "Reversal") if any(record["corpus"] == name and record["model_size_m"] == 170 for record in records)]
    if len(corpora) < 2:
        return []
    common_seeds = set.intersection(
        *[
            {record["seed"] for record in records if record["corpus"] == corpus and record["model_size_m"] == 170 and record.get("loss_mode") == "native_conditional"}
            for corpus in corpora
        ]
    )
    if not common_seeds:
        raise ValueError("cross-corpus comparison has no common seeds")
    arrays, rows = [], []
    for corpus in corpora:
        subset = [record for record in records if record["corpus"] == corpus and record["model_size_m"] == 170 and record["mask_probability"] is not None and record.get("loss_mode") == "native_conditional" and record["seed"] in common_seeds]
        sample_count = min(64, max(record["sample_count"] for record in subset))
        parameters = sorted({record["parameter"] for record in subset}, key=_layer)
        probabilities = sorted({record["mask_probability"] for record in subset})
        grouped = _groups([record for record in subset if record["sample_count"] == sample_count], ("parameter", "mask_probability"))
        array = np.array([[statistics.fmean(_score(record) for record in grouped[(parameter, probability)]) for probability in probabilities] for parameter in parameters])
        arrays.append((corpus, sample_count, parameters, probabilities, array))
        for row_index, parameter in enumerate(parameters):
            for column, probability in enumerate(probabilities):
                rows.append({"corpus": corpus, "model_size_m": 170, "sample_count": sample_count, "layer": _layer(parameter), "mask_probability": probability, "mean_log_error_ratio": float(array[row_index, column]), "seeds": sorted(common_seeds)})
    vmax = max(max(abs(float(array.min())), abs(float(array.max()))) for *_, array in arrays)
    fig, axes = plt.subplots(1, len(arrays), figsize=(10.8, 3.8), constrained_layout=True, squeeze=False)
    image = None
    cmap = _soft_diverging_cmap()
    for panel, (corpus, sample_count, parameters, probabilities, array) in zip(axes[0], arrays):
        image = panel.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        panel.set_xticks(range(len(probabilities)), [f"{value:.1f}" for value in probabilities])
        panel.set_yticks(range(len(parameters)), [f"L{_layer(value)}" for value in parameters])
        panel.set_title(f"{corpus} (n={sample_count})")
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                panel.text(column, row, _cell_text(value), ha="center", va="center", fontsize=11, color=PLOT_TEXT)
    axes[0, 0].set_ylabel("219M parameter slice")
    fig.supxlabel("mask probability")
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"score $s$")
    _save(fig, output_dir, "corpus_heatmaps")
    plt.close(fig)
    return rows


def _objective_heatmaps(plt, np, records, output_dir):
    selected = [record for record in records if record["corpus"] == "GSM8K" and record["model_size_m"] == 170 and record["mask_probability"] is not None and record.get("evaluation") == "split_sample"]
    modes = [mode for mode in ("native_conditional", "fixed_target") if any(record.get("loss_mode") == mode for record in selected)]
    if len(modes) < 2:
        return []
    common_seeds = set.intersection(*[{record["seed"] for record in selected if record.get("loss_mode") == mode} for mode in modes])
    if not common_seeds:
        raise ValueError("target-count comparison has no common seeds")
    labels = {"native_conditional": "all targets", "fixed_target": "one target"}
    arrays, rows, mode_arrays = [], [], {}
    for mode in modes:
        subset = [record for record in selected if record["loss_mode"] == mode and record["sample_count"] == 64 and record["seed"] in common_seeds]
        parameters = sorted({record["parameter"] for record in subset}, key=_layer)
        probabilities = sorted({record["mask_probability"] for record in subset})
        grouped = _groups(subset, ("parameter", "mask_probability"))
        array = np.array([[statistics.fmean(_score(record) for record in grouped[(parameter, probability)]) for probability in probabilities] for parameter in parameters])
        arrays.append((labels[mode], parameters, probabilities, array))
        mode_arrays[mode] = array
        for row_index, parameter in enumerate(parameters):
            for column, probability in enumerate(probabilities):
                rows.append({"loss_mode": mode, "sample_count": 64, "layer": _layer(parameter), "mask_probability": probability, "mean_log_error_ratio": float(array[row_index, column]), "seeds": sorted(common_seeds)})
    effect = mode_arrays["native_conditional"] - mode_arrays["fixed_target"]
    arrays.append((r"paired effect $s_{\mathrm{all}}-s_{\mathrm{one}}$", parameters, probabilities, effect))
    for row_index, parameter in enumerate(parameters):
        for column, probability in enumerate(probabilities):
            rows.append({"loss_mode": "all_minus_one_target", "sample_count": 64, "layer": _layer(parameter), "mask_probability": probability, "mean_log_error_ratio_change": float(effect[row_index, column]), "seeds": sorted(common_seeds)})
    vmax = max(max(abs(float(array.min())), abs(float(array.max()))) for *_, array in arrays)
    fig, axes = plt.subplots(1, len(arrays), figsize=(10.8, 3.8), constrained_layout=True, squeeze=False)
    image = None
    cmap = _soft_diverging_cmap()
    for panel, (label, parameters, probabilities, array) in zip(axes[0], arrays):
        image = panel.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        panel.set_xticks(range(len(probabilities)), [f"{value:.1f}" for value in probabilities])
        panel.set_yticks(range(len(parameters)), [f"L{_layer(value)}" for value in parameters])
        panel.set_title(label)
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = array[row, column]
                panel.text(column, row, _cell_text(value), ha="center", va="center", fontsize=11, color=PLOT_TEXT)
    axes[0, 0].set_ylabel("219M parameter slice")
    fig.supxlabel("context mask probability")
    fig.colorbar(image, ax=axes, shrink=0.85, label=r"score $s$")
    _save(fig, output_dir, "objective_heatmaps")
    plt.close(fig)
    return rows


def _split_controls(plt, np, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus and record.get("evaluation") == "split_sample" and record.get("loss_mode") == "native_conditional"]
    models = sorted({record["model_size_m"] for record in selected})
    methods = [
        ("test_oracle_top1_relative_frobenius_error", "oracle r1"),
        ("mean_rank1_test_relative_frobenius_error", "mean-grad r1"),
        ("diagonal_test_relative_frobenius_error", "diagonal"),
        ("calibration_top1_test_relative_frobenius_error", "calib top r1"),
        ("isotropic_test_relative_frobenius_error", "isotropic"),
        ("signed_mean_test_error_mean", "signed-mean"),
        ("random_direction_test_error_mean", "random dir."),
    ]
    colors = ["#F0E442", "#0072B2", "#D55E00", "#009E73", "#999999", "#CC79A7", "#56B4E9"]
    fig, axes = plt.subplots(1, len(models), figsize=(8.6, 4.3), constrained_layout=True, squeeze=False)
    output = []
    for panel, model in zip(axes[0], models):
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_count = max(record["sample_count"] for record in model_records)
        subset = [record for record in model_records if record["sample_count"] == sample_count]
        grouped = _groups(subset, ("parameter", "mask_condition"))
        distributions = []
        for key, label in methods:
            values = [statistics.fmean(record[key] for record in cell) for cell in grouped.values()]
            distributions.append(values)
            output.append({"corpus": corpus, "model_size_m": model, "sample_count": sample_count, "method": key, "cell_means": values, "mean": statistics.fmean(values)})
        boxes = panel.boxplot(
            distributions,
            patch_artist=True,
            widths=0.65,
            showfliers=False,
            boxprops={"linewidth": 1.4},
            whiskerprops={"linewidth": 1.4},
            capprops={"linewidth": 1.4},
            medianprops={"color": "black", "linewidth": 1.8},
        )
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        rng = np.random.default_rng(20260825 + model)
        for index, values in enumerate(distributions, start=1):
            panel.scatter(rng.normal(index, 0.045, len(values)), values, s=18, color="black", alpha=0.42)
        panel.set_xticks(range(1, len(methods) + 1), [label for _, label in methods], rotation=24, ha="right")
        panel.axhline(1.0, color="0.45", linewidth=1.2, linestyle=":")
        panel.set_yscale("log")
        panel.set_title(f"{_model_label(model)}, calibration n={sample_count}")
    axes[0, 0].set_ylabel("independent-test Fisher error")
    _save(fig, output_dir, "split_sample_controls")
    plt.close(fig)
    return output


def _cell_log_win_rate(grouped, diagonal_key, rank1_key):
    return statistics.fmean(
        statistics.fmean(math.log(record[diagonal_key] / record[rank1_key]) for record in cell) > 0
        for cell in grouped.values()
    )


def _real_in_vs_out(plt, records, corpus, output_dir):
    selected = [record for record in records if record["corpus"] == corpus and record.get("evaluation") == "split_sample" and record.get("loss_mode") == "native_conditional" and record["mask_probability"] is not None]
    if not selected:
        return []
    models = sorted({record["model_size_m"] for record in selected})
    colors = {models[0]: "#1f77b4", models[-1]: "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    rows = []
    for model in models:
        model_records = [record for record in selected if record["model_size_m"] == model]
        sample_sizes = sorted({record["sample_count"] for record in model_records})
        in_wins, test_wins, in_logs, test_logs = [], [], [], []
        in_win_lows, in_win_highs = [], []
        test_win_lows, test_win_highs = [], []
        in_log_lows, in_log_highs = [], []
        test_log_lows, test_log_highs = [], []
        for sample_count in sample_sizes:
            subset = [record for record in model_records if record["sample_count"] == sample_count]
            by_seed = _groups(subset, ("seed",))
            seed_in_wins, seed_test_wins, seed_in_logs, seed_test_logs = [], [], [], []
            for seed_records in by_seed.values():
                grouped = _groups(seed_records, ("parameter", "mask_probability"))
                seed_in_wins.append(_cell_log_win_rate(grouped, "calibration_diagonal_relative_frobenius_error", "calibration_rank1_relative_frobenius_error"))
                seed_test_wins.append(_cell_log_win_rate(grouped, "diagonal_test_relative_frobenius_error", "mean_rank1_test_relative_frobenius_error"))
                seed_in_logs.append(statistics.fmean(math.log(record["calibration_diagonal_relative_frobenius_error"] / record["calibration_rank1_relative_frobenius_error"]) for record in seed_records))
                seed_test_logs.append(statistics.fmean(math.log(record["diagonal_test_relative_frobenius_error"] / record["mean_rank1_test_relative_frobenius_error"]) for record in seed_records))
            in_win, in_win_half = _mean_ci(seed_in_wins)
            test_win, test_win_half = _mean_ci(seed_test_wins)
            in_log, in_log_half = _mean_ci(seed_in_logs)
            test_log, test_log_half = _mean_ci(seed_test_logs)
            in_wins.append(in_win)
            test_wins.append(test_win)
            in_logs.append(in_log)
            test_logs.append(test_log)
            in_win_lows.append(in_win - in_win_half)
            in_win_highs.append(in_win + in_win_half)
            test_win_lows.append(test_win - test_win_half)
            test_win_highs.append(test_win + test_win_half)
            in_log_lows.append(in_log - in_log_half)
            in_log_highs.append(in_log + in_log_half)
            test_log_lows.append(test_log - test_log_half)
            test_log_highs.append(test_log + test_log_half)
            rows.append({
                "corpus": corpus,
                "model_size_m": model,
                "sample_count": sample_count,
                "in_sample_win_rate": in_win,
                "in_sample_win_rate_ci95": [in_win - in_win_half, in_win + in_win_half],
                "test_win_rate": test_win,
                "test_win_rate_ci95": [test_win - test_win_half, test_win + test_win_half],
                "in_sample_mean_log_error_ratio": in_log,
                "in_sample_mean_log_error_ratio_ci95": [in_log - in_log_half, in_log + in_log_half],
                "test_mean_log_error_ratio": test_log,
                "test_mean_log_error_ratio_ci95": [test_log - test_log_half, test_log + test_log_half],
            })
        axes[0].fill_between(sample_sizes, [max(0.0, value) for value in in_win_lows], [min(1.0, value) for value in in_win_highs], color=colors[model], alpha=0.14, linewidth=0)
        axes[0].fill_between(sample_sizes, [max(0.0, value) for value in test_win_lows], [min(1.0, value) for value in test_win_highs], color=colors[model], alpha=0.08, linewidth=0)
        axes[0].plot(sample_sizes, in_wins, marker="o", color=colors[model], label=f"{_model_label(model)} in-sample")
        axes[0].plot(sample_sizes, test_wins, marker="o", linestyle="--", color=colors[model], label=f"{_model_label(model)} test")
        axes[1].fill_between(sample_sizes, in_log_lows, in_log_highs, color=colors[model], alpha=0.14, linewidth=0)
        axes[1].fill_between(sample_sizes, test_log_lows, test_log_highs, color=colors[model], alpha=0.08, linewidth=0)
        axes[1].plot(sample_sizes, in_logs, marker="o", color=colors[model])
        axes[1].plot(sample_sizes, test_logs, marker="o", linestyle="--", color=colors[model])
    axes[0].axhline(0.5, color="0.5", linewidth=1.2, linestyle=":")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("fraction of layer×mask\ncells won")
    axes[0].legend(loc="upper center", fontsize=10, ncol=2)
    axes[1].axhline(0, color="0.5", linewidth=1.2, linestyle=":")
    axes[1].set_ylabel(r"mean $\log(e_{diag}/e_{rank1})$")
    for panel in axes:
        panel.set_xscale("log", base=2)
        panel.set_xlabel("calibration examples")
    _save(fig, output_dir, "real_in_sample_vs_test")
    plt.close(fig)
    return rows


def _null_figures(plt, np, null_path, output_dir):
    if not null_path:
        return {}
    payload = json.loads(null_path.read_text(encoding="utf-8"))
    dimensions = sorted({case["dimension"] for case in payload["cases"]})
    colors = {dimensions[0]: "#1f77b4", dimensions[-1]: "#ff7f0e"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), constrained_layout=True)
    for panel, prefix, title in zip(axes, ("in_sample", "split", "population"), ("same-sample score", "independent-test score", "true population score")):
        for dimension in dimensions:
            cases = sorted((case for case in payload["cases"] if case["dimension"] == dimension), key=lambda case: case["calibration_sample_count"])
            x_values = np.array([case["calibration_sample_count"] / dimension for case in cases])
            y_values = np.array([case["summary"][f"{prefix}_rank1_win_rate"] for case in cases])
            interval = [
                _wilson_ci(
                    sum(row[f"{prefix}_rank1"] < row[f"{prefix}_diagonal"] for row in case["raw"]),
                    case["repetitions"],
                )
                for case in cases
            ]
            low_values = np.array([low for low, _ in interval])
            high_values = np.array([high for _, high in interval])
            panel.fill_between(x_values, low_values, high_values, color=colors[dimension], alpha=0.14, linewidth=0)
            panel.errorbar(
                x_values,
                y_values,
                yerr=[y_values - low_values, high_values - y_values],
                fmt="none",
                ecolor=colors[dimension],
                elinewidth=1.4,
                capsize=3,
                alpha=0.75,
                zorder=2,
            )
            panel.plot(
                x_values,
                y_values,
                marker="o",
                color=colors[dimension],
                label=f"d={dimension}",
            )
        panel.axhline(0.5, color="0.5", linestyle="--", linewidth=1.2)
        panel.set_xscale("log", base=2)
        panel.set_ylim(-0.03, 1.03)
        panel.set_xlabel("calibration ratio n/d")
        panel.set_title(title)
    axes[0].set_ylabel("P(rank-1 error < diagonal error)")
    axes[-1].legend(frameon=False)
    _save(fig, output_dir, "finite_sample_null")
    plt.close(fig)

    dimensions_curve = np.unique(np.geomspace(2, 4096, 300).astype(int))
    eigengap = 2 / (dimensions_curve + 2)
    rank1_error = np.sqrt(2 * (dimensions_curve - 1) / (3 * dimensions_curve))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    # ponytail: these are exact analytic curves, so no sampling interval is meaningful.
    axes[0].plot(dimensions_curve, eigengap, color=SOFT_BLUE)
    axes[1].plot(dimensions_curve, rank1_error, color=SOFT_ORANGE)
    for panel in axes:
        panel.set_xscale("log")
        panel.set_xlabel("data dimension d")
    axes[0].set_ylabel(r"$\lambda_2/\lambda_1=2/(d+2)$")
    axes[0].set_title("spectral ratio vanishes")
    axes[1].set_ylabel("best rank-1 Frobenius error")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(math.sqrt(2 / 3), color="0.45", linewidth=1.2, linestyle=":")
    axes[1].text(
        0.98,
        math.sqrt(2 / 3) + 0.015,
        r"$\sqrt{2/3}$ limit",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=12,
        color="0.35",
    )
    axes[1].set_title("rank-1 residual stays large")
    for label, panel in zip("AB", axes):
        panel.text(-0.14, 1.05, label, transform=panel.transAxes, fontweight="bold")
    _save(fig, output_dir, "gaussian_counterexample")
    plt.close(fig)
    return {
        "path": _relative(null_path),
        "sha256": _sha256(null_path),
        "cases": [{"dimension": case["dimension"], "calibration_sample_count": case["calibration_sample_count"], "test_sample_count": case["test_sample_count"], "summary": case["summary"]} for case in payload["cases"]],
        "analytic_scaled_identity_counterexample": payload["analytic_scaled_identity_counterexample"],
    }


def _score_summary(np, records, bootstrap_seed):
    by_seed = _groups(records, ("seed",))
    seed_scores = [statistics.fmean(_score(record) for record in by_seed[key]) for key in sorted(by_seed)]
    mean, half_width = _mean_ci(seed_scores)
    _, bootstrap_low, bootstrap_high = _hierarchical_ci(np, records, bootstrap_seed, "log_ratio")
    cells = _groups(records, ("parameter", "mask_condition"))
    cell_scores = [statistics.fmean(_score(record) for record in values) for values in cells.values()]
    return {
        "records": len(records),
        "seeds": sorted(key[0] for key in by_seed),
        "conditions": len(cells),
        "seed_mean_log_error_ratios": seed_scores,
        "mean_log_error_ratio": mean,
        "geometric_mean_rank1_over_diagonal_error": math.exp(-mean),
        "seed_t_ci95": [mean - half_width, mean + half_width],
        "two_way_bootstrap_ci95_sensitivity": [bootstrap_low, bootstrap_high],
        "cell_win_count": sum(score > 0 for score in cell_scores),
        "cell_count": len(cell_scores),
        "mean_rank1_error": statistics.fmean(_rank1_error(record) for record in records),
        "mean_diagonal_error": statistics.fmean(_diagonal_error(record) for record in records),
    }


def _submission_summary(np, records, comparison_records, corpus):
    native = [record for record in records if record["corpus"] == corpus and record.get("evaluation") == "split_sample" and record.get("loss_mode") == "native_conditional"]
    primary = []
    for model in sorted({record["model_size_m"] for record in native}):
        model_records = [record for record in native if record["model_size_m"] == model]
        sample_count = max(record["sample_count"] for record in model_records)
        fixed = [record for record in model_records if record["sample_count"] == sample_count and record["mask_probability"] is not None]
        schedule = [record for record in model_records if record["sample_count"] == sample_count and record["mask_probability"] is None]
        primary.append({
            "model_size_m": model,
            "parameter_count_label": _model_label(model),
            "calibration_sample_count": sample_count,
            "fixed_mask_grid": _score_summary(np, fixed, 310000 + model),
            "uniform_p_within_p_nonempty_diagnostic": _score_summary(np, schedule, 320000 + model),
        })

    corpus_names = [name for name in ("GSM8K", "MT-Bench", "Reversal") if any(record["corpus"] == name and record["model_size_m"] == 170 and record.get("loss_mode") == "native_conditional" for record in comparison_records)]
    common_corpus_seeds = set.intersection(*[{record["seed"] for record in comparison_records if record["corpus"] == name and record["model_size_m"] == 170 and record.get("loss_mode") == "native_conditional"} for name in corpus_names])
    corpora = []
    for index, name in enumerate(corpus_names):
        subset = [record for record in comparison_records if record["corpus"] == name and record["model_size_m"] == 170 and record.get("loss_mode") == "native_conditional" and record["sample_count"] == 64 and record["mask_probability"] is not None and record["seed"] in common_corpus_seeds]
        corpora.append({"corpus": name, **_score_summary(np, subset, 330000 + index)})

    modes = {}
    for index, mode in enumerate(("native_conditional", "fixed_target")):
        subset = [record for record in comparison_records if record["corpus"] == corpus and record["model_size_m"] == 170 and record.get("loss_mode") == mode and record["sample_count"] == 64 and record["mask_probability"] is not None and record["seed"] in {0, 1, 2}]
        modes[mode] = _score_summary(np, subset, 340000 + index)
    all_rows = [record for record in comparison_records if record["corpus"] == corpus and record["model_size_m"] == 170 and record.get("loss_mode") == "native_conditional" and record["sample_count"] == 64 and record["mask_probability"] is not None and record["seed"] in {0, 1, 2}]
    one_rows = [record for record in comparison_records if record["corpus"] == corpus and record["model_size_m"] == 170 and record.get("loss_mode") == "fixed_target" and record["sample_count"] == 64 and record["mask_probability"] is not None and record["seed"] in {0, 1, 2}]
    all_lookup = {(record["seed"], record["parameter"], record["mask_probability"]): _score(record) for record in all_rows}
    one_lookup = {(record["seed"], record["parameter"], record["mask_probability"]): _score(record) for record in one_rows}
    if all_lookup.keys() != one_lookup.keys():
        raise ValueError("target-count intervention is not paired")
    differences = {key: all_lookup[key] - one_lookup[key] for key in all_lookup}
    seed_effects = [statistics.fmean(value for (run_seed, _, _), value in differences.items() if run_seed == seed) for seed in sorted({key[0] for key in differences})]
    effect_mean, effect_half_width = _mean_ci(seed_effects)
    cell_effects = _groups([{"parameter": parameter, "mask_probability": probability, "effect": value} for (_, parameter, probability), value in differences.items()], ("parameter", "mask_probability"))
    cell_effect_values = [statistics.fmean(row["effect"] for row in values) for values in cell_effects.values()]
    target_effect = {
        "seed_mean_score_changes": seed_effects,
        "mean_score_change": effect_mean,
        "seed_t_ci95": [effect_mean - effect_half_width, effect_mean + effect_half_width],
        "positive_cell_count": sum(value > 0 for value in cell_effect_values),
        "cell_count": len(cell_effect_values),
    }
    return {"primary": primary, "cross_corpus_common_seeds": sorted(common_corpus_seeds), "cross_corpus": corpora, "target_count_modes": modes, "paired_target_aggregation_effect": target_effect}


def _dense_slice_control(plt, np, records, output_dir):
    if not records:
        return {}
    selected = [record for record in records if record.get("evaluation") == "split_sample" and record.get("loss_mode") == "native_conditional"]
    sample_count = max(record["sample_count"] for record in selected)
    selected = [record for record in selected if record["sample_count"] == sample_count]
    condition_order = sorted({record["mask_condition"] for record in selected}, key=lambda value: (value == "native_schedule", value))
    grouped = _groups(selected, ("mask_condition",))
    fig, panel = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    rows = []
    for index, condition in enumerate(condition_order):
        values = grouped[(condition,)]
        scores = [_score(record) for record in values]
        mean, half_width = _mean_ci(scores)
        panel.errorbar(index, mean, yerr=half_width, marker="o", color="#1f77b4", capsize=4)
        panel.scatter(np.full(len(scores), index) + np.linspace(-0.06, 0.06, len(scores)), scores, color="black", s=24, alpha=0.55)
        rows.append({"mask_condition": condition, "seed_scores": scores, "mean_log_error_ratio": mean, "seed_t_ci95": [mean - half_width, mean + half_width]})
    labels = [condition[6:] if condition.startswith("fixed_") else "uniform-p\nnonempty" for condition in condition_order]
    panel.axhline(0, color="0.45", linewidth=1.2, linestyle=":")
    panel.set_xticks(range(len(labels)), labels)
    panel.set_xlabel("mask probability / diagnostic")
    panel.set_ylabel(r"$s=\log(e_{diag}/e_{rank1})$")
    panel.set_title(f"219M attention output, d={selected[0]['slice_numel']:,}")
    _save(fig, output_dir, "dense_slice_control")
    plt.close(fig)
    fixed = [record for record in selected if record["mask_probability"] is not None]
    schedule = [record for record in selected if record["mask_probability"] is None]
    return {"sample_count": sample_count, "test_sample_count": selected[0]["test_sample_count"], "cells": rows, "fixed_mask_grid": _score_summary(np, fixed, 350001), "uniform_p_within_p_nonempty_diagnostic": _score_summary(np, schedule, 350002)}


def _continual(plt, np, summary_path, output_dir):
    if not summary_path:
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = {"seq": "sequential", "replay": "replay", "diag_replay": "diag + replay", "rank1_replay": "rank-1 + replay"}
    colors = {"seq": "#CC79A7", "replay": "#009E73", "diag_replay": "#E69F00", "rank1_replay": "#0072B2"}
    fig, panel = plt.subplots(figsize=(4.1, 3.2), constrained_layout=True)
    tradeoff = []
    for group in summary["groups"]:
        method = group["group"]["method"]
        losses = group["final_average_loss"]["values"]
        forgetting = group["average_forgetting"]["values"]
        panel.scatter(losses, forgetting, color=colors[method], alpha=0.45, s=26)
        panel.scatter(statistics.fmean(losses), statistics.fmean(forgetting), color=colors[method], marker="X", s=85, label=labels[method], edgecolor="black", linewidth=0.4)
        tradeoff.append({"method": method, "seeds": group["seeds"], "final_average_loss": losses, "average_forgetting": forgetting})
    panel.axhline(0, color="0.5", linewidth=1.2)
    panel.set_xlabel("final average masked loss (lower is better)")
    panel.set_ylabel("average forgetting (lower is better)")
    panel.legend()
    _save(fig, output_dir, "continual_tradeoff")
    plt.close(fig)

    matrices = defaultdict(list)
    for path_string in summary["input_files"]:
        payload = json.loads(Path(path_string).read_text(encoding="utf-8"))
        matrices[payload["config"]["method"]].append(np.array(payload["loss_matrix"], dtype=float))
    methods = ["seq", "replay", "diag_replay", "rank1_replay"]
    averages = [np.mean(matrices[method], axis=0) for method in methods]
    vmin = min(float(array.min()) for array in averages)
    vmax = max(float(array.max()) for array in averages)
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.5), constrained_layout=True)
    image = None
    for panel, method, array in zip(axes, methods, averages):
        image = panel.imshow(array, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
        panel.set_title(labels[method], fontsize=9)
        panel.set_xlabel("evaluated task")
        panel.set_xticks(range(array.shape[1]))
        panel.set_yticks(range(array.shape[0]))
    axes[0].set_ylabel("after learning task")
    fig.colorbar(image, ax=axes, shrink=0.8, label="masked loss")
    _save(fig, output_dir, "continual_loss_matrices")
    plt.close(fig)
    return {"tradeoff": tradeoff, "loss_matrix_means": {method: array.tolist() for method, array in zip(methods, averages)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, nargs="+")
    parser.add_argument("--comparison-control", type=Path, nargs="+")
    parser.add_argument("--comparison-contract", type=Path)
    parser.add_argument("--slice-control", type=Path, nargs="+")
    parser.add_argument("--null", type=Path)
    parser.add_argument("--continual-summary", type=Path)
    parser.add_argument("--primary-corpus", default="GSM8K")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "assets/iclr_2/figures")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        import numpy as np

        mean, ci = _mean_ci([1.0, 2.0, 3.0])
        assert mean == 2.0 and ci > 0 and _layer("transformer.h.17.norm_1.weight") == 17
        rows = [
            {"seed": seed, "parameter": f"transformer.h.{layer}.norm_1.weight", "mask_probability": 0.5, "mask_condition": "fixed_0.5", "diagonal_relative_frobenius_error": 1.0, "rank1_relative_frobenius_error": value}
            for seed, layer, value in ((0, 0, 0.7), (0, 1, 0.8), (1, 0, 0.6), (1, 1, 0.9))
        ]
        estimate, low, high = _hierarchical_ci(np, rows, 1)
        assert low <= estimate <= high
        summary = _score_summary(np, rows, 2)
        assert summary["records"] == 4 and summary["conditions"] == 2
        assert _cell_log_win_rate({("cell",): [{"d": 1.0, "r": 0.4}, {"d": 1.0, "r": 2.0}]}, "d", "r") == 1.0
        print(json.dumps({"self_check": "ok"}))
        return
    if not args.geometry or not args.comparison_control or not args.comparison_contract:
        parser.error("--geometry, --comparison-control, and --comparison-contract are required")

    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, files, failures = _load_geometry(args.geometry)
    comparison_records, comparison_files, comparison_failures = _load_geometry(args.comparison_control)
    from build_comparison_contract import validate_contract
    comparison_contract = validate_contract(args.comparison_contract)
    if args.slice_control:
        slice_records, slice_files, slice_failures = _load_geometry(args.slice_control)
    else:
        slice_records, slice_files, slice_failures = [], [], []
    heatmaps = _heatmaps(plt, np, records, args.primary_corpus, args.output_dir)
    sensitivity, alignment, correlation = _robustness(plt, np, records, args.primary_corpus, args.output_dir)
    split_mode = any(record.get("evaluation") == "split_sample" for record in records)
    spectrum = [] if split_mode else _spectrum(plt, records, args.primary_corpus, args.output_dir)
    split_controls = _split_controls(plt, np, records, args.primary_corpus, args.output_dir) if split_mode else []
    real_in_vs_out = _real_in_vs_out(plt, records, args.primary_corpus, args.output_dir) if split_mode else []
    cross_corpus = _cross_corpus(plt, np, comparison_records, args.output_dir)
    objective = _objective_heatmaps(plt, np, comparison_records, args.output_dir)
    null = _null_figures(plt, np, args.null, args.output_dir)
    continual = _continual(plt, np, args.continual_summary, args.output_dir)
    submission_summary = _submission_summary(np, records, comparison_records, args.primary_corpus) if split_mode else {}
    dense_slice_control = _dense_slice_control(plt, np, slice_records, args.output_dir)
    figure_data = {
        "schema_version": 3,
        "figure_script_sha256": _sha256(Path(__file__)),
        "geometry_inputs": files,
        "geometry_failures": failures,
        "comparison_control_inputs": comparison_files,
        "comparison_control_failures": comparison_failures,
        "comparison_contract": {
            "path": _relative(args.comparison_contract),
            "sha256": _sha256(args.comparison_contract),
            "verified": comparison_contract["verified"],
        },
        "slice_control_inputs": slice_files,
        "slice_control_failures": slice_failures,
        "primary_corpus": args.primary_corpus,
        "raw_schedule_semantics": {
            "mask_condition": "native_schedule",
            "reported_as": "uniform-p, within-p-nonempty diagnostic",
            "is_native_training_distribution": False,
        },
        "heatmap_cells": heatmaps,
        "sample_sensitivity": sensitivity,
        "alignment_cells": alignment,
        "alignment_score_pearson_r": correlation,
        "spectrum": spectrum,
        "split_sample_controls": split_controls,
        "real_in_sample_vs_test": real_in_vs_out,
        "cross_corpus_cells": cross_corpus,
        "objective_cells": objective,
        "null": null,
        "submission_summary": submission_summary,
        "dense_slice_control": dense_slice_control,
        "continual": continual,
    }
    (args.output_dir / "figure_data.json").write_text(json.dumps(figure_data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "geometry_records": len(records), "figures": sorted(path.name for path in args.output_dir.glob("*.pdf"))}, indent=2))


if __name__ == "__main__":
    main()
