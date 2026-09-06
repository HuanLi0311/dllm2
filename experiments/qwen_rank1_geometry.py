#!/usr/bin/env python3
"""Held-out rank-1 versus diagonal Fisher geometry for local Qwen3 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
DEFAULT_REVERSE = ROOT / "SMDM/data/reverse_experiments/june_version_7921032488"
DEFAULT_GSM = ROOT / "SMDM/data/gsm8k/train_no_aug.txt"
PROTOCOL = ROOT / "report/qwen_rank1_geometry_protocol.md"
SEEDS = (3407, 3408, 3409)
CORPORA = ("gsm8k", "d2p", "p2d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_sem(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return {"mean": mean, "sem": math.sqrt(variance / len(values)), "values": values}


def _model_files(model_dir: Path) -> list[Path]:
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")
    files = [model_dir / name for name in names if (model_dir / name).is_file()]
    files.extend(sorted(model_dir.glob("*.safetensors")))
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"no local safetensors checkpoint found in {model_dir}")
    return files


def inventory(model_dir: Path, output: Path) -> dict:
    model_dir = model_dir.resolve()
    payload = {
        "schema_version": 1,
        "model_dir": str(model_dir),
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in _model_files(model_dir)
        ],
    }
    if output.exists():
        expected = json.loads(output.read_text(encoding="utf-8"))
        if payload != expected:
            raise ValueError("model inventory changed")
        return payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _raw_rows(gsm_path: Path, reverse_dir: Path) -> dict[str, list[dict]]:
    gsm = []
    for line in gsm_path.read_text(encoding="utf-8").splitlines():
        if "||" not in line:
            continue
        prompt, completion = line.split("||", 1)
        gsm.append({"prompt": "Question: " + prompt + "\nAnswer:", "completion": " " + completion})
    reversal = {}
    for direction in ("d2p", "p2d"):
        reversal[direction] = [
            {"prompt": row["prompt"], "completion": row["completion"]}
            for row in _load_jsonl(reverse_dir / f"{direction}_prompts_train.jsonl")
        ]
    return {"gsm8k": gsm, **reversal}


def _encode_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer requires eos_token_id")
    encoded = []
    for source_index, row in enumerate(rows):
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        answer_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
        prefix = ([] if bos is None else [bos]) + prompt_ids
        ids = prefix + answer_ids + [eos]
        if answer_ids and len(ids) <= max_length:
            encoded.append({"source_index": source_index, "ids": ids, "answer_start": len(prefix)})
    return encoded


def _selected_parameters(model) -> tuple[list[str], list]:
    layers = int(model.config.num_hidden_layers)
    indices = (0, layers // 2, layers - 1)
    named = dict(model.named_parameters())
    names = [
        f"model.layers.{index}.{kind}.weight"
        for index in indices
        for kind in ("input_layernorm", "post_attention_layernorm")
    ]
    missing = [name for name in names if name not in named]
    if missing:
        raise ValueError(f"missing Qwen parameters: {missing}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = [named[name] for name in names]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return names, parameters


def _example_gradients(model, parameters, row, device):
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False).logits[0]
    start = row["answer_start"]
    loss = F.cross_entropy(logits[start - 1 : -1].float(), ids[0, start:])
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    return [gradient.detach().float().cpu() for gradient in gradients]


def _score(calibration, test) -> dict:
    import torch

    calibration = torch.stack(calibration).double()
    test = torch.stack(test).double()
    mean = calibration.mean(dim=0)
    mean_norm_sq = float(mean @ mean)
    eps = 1e-30
    cal_projection = calibration @ mean
    coefficient = float((cal_projection @ cal_projection) / calibration.shape[0]) / max(mean_norm_sq**2, eps)
    test_gram = test @ test.T / test.shape[0]
    fisher_norm_sq = float((test_gram * test_gram).sum())
    test_projection = test @ mean
    rank1_inner = coefficient * float((test_projection @ test_projection) / test.shape[0])
    rank1_norm_sq = coefficient**2 * mean_norm_sq**2
    rank1_error = math.sqrt(max(fisher_norm_sq - 2 * rank1_inner + rank1_norm_sq, 0.0) / max(fisher_norm_sq, eps))
    diagonal = (calibration * calibration).mean(dim=0)
    diagonal_inner = float(((test * test) * diagonal).sum() / test.shape[0])
    diagonal_norm_sq = float(diagonal @ diagonal)
    diagonal_error = math.sqrt(max(fisher_norm_sq - 2 * diagonal_inner + diagonal_norm_sq, 0.0) / max(fisher_norm_sq, eps))
    eigenvalues = torch.linalg.eigvalsh(test_gram)
    lambda1 = float(eigenvalues[-1])
    oracle_error = math.sqrt(max(fisher_norm_sq - lambda1**2, 0.0) / max(fisher_norm_sq, eps))
    return {
        "rank1_test_relative_frobenius_error": rank1_error,
        "diagonal_test_relative_frobenius_error": diagonal_error,
        "score": math.log(diagonal_error / rank1_error),
        "rank1_coefficient": coefficient,
        "rank1_trace_fraction": coefficient * mean_norm_sq / max(float((calibration * calibration).sum() / calibration.shape[0]), eps),
        "oracle_test_relative_frobenius_error": oracle_error,
    }


def _selection_hash(rows: list[dict]) -> str:
    payload = json.dumps(
        [{"source_index": row["source_index"], "ids": row["ids"], "answer_start": row["answer_start"]} for row in rows],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def run(args) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source_hash = _sha256(Path(__file__))
    inventory_payload = json.loads(args.model_inventory.read_text(encoding="utf-8"))
    if Path(inventory_payload["model_dir"]).resolve() != args.model.resolve():
        raise ValueError("model and inventory paths disagree")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen probe requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.config.use_cache = False
    model.eval()
    names, parameters = _selected_parameters(model)
    raw = _raw_rows(args.gsm, args.reverse_dir)
    results = []
    selections = {}
    for corpus_index, corpus in enumerate(CORPORA):
        rows = _encode_rows(raw[corpus], tokenizer, args.max_length)
        rng = random.Random(args.seed * 1009 + corpus_index * 100_003)
        rng.shuffle(rows)
        need = args.calibration_samples + args.test_samples
        if len(rows) < need:
            raise ValueError(f"{corpus} has only {len(rows)} usable rows, need {need}")
        calibration_rows = rows[: args.calibration_samples]
        test_rows = rows[args.calibration_samples : need]
        selections[corpus] = {
            "usable_rows": len(rows),
            "calibration_sha256": _selection_hash(calibration_rows),
            "test_sha256": _selection_hash(test_rows),
        }
        gradients = [[] for _ in parameters]
        for split, selected in (("calibration", calibration_rows), ("test", test_rows)):
            split_gradients = [[] for _ in parameters]
            for index, row in enumerate(selected, 1):
                values = _example_gradients(model, parameters, row, device)
                for parameter_index, value in enumerate(values):
                    split_gradients[parameter_index].append(value)
                if index % 8 == 0 or index == len(selected):
                    print(f"corpus={corpus} split={split} example={index}/{len(selected)}", flush=True)
            if split == "calibration":
                gradients = split_gradients
            else:
                for name, calibration, test in zip(names, gradients, split_gradients):
                    results.append({"corpus": corpus, "parameter": name, **_score(calibration, test)})
    if _sha256(Path(__file__)) != source_hash:
        raise RuntimeError("probe source changed during execution")
    return {
        "schema_version": 1,
        "status": "ok",
        "model": str(args.model.resolve()),
        "model_inventory": str(args.model_inventory.resolve()),
        "model_inventory_sha256": _sha256(args.model_inventory),
        "source_sha256": source_hash,
        "protocol_sha256": _sha256(PROTOCOL),
        "data_sha256": {
            "gsm": _sha256(args.gsm),
            "d2p": _sha256(args.reverse_dir / "d2p_prompts_train.jsonl"),
            "p2d": _sha256(args.reverse_dir / "p2d_prompts_train.jsonl"),
        },
        "config": {
            "seed": args.seed,
            "calibration_samples": args.calibration_samples,
            "test_samples": args.test_samples,
            "max_length": args.max_length,
            "loss": "mean causal cross-entropy over completion tokens and EOS",
            "parameters": names,
            "corpora": list(CORPORA),
        },
        "selection": selections,
        "software": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__},
        "results": results,
    }


def summarize(paths: list[Path]) -> dict:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(payloads) != 3 or sorted(item["config"]["seed"] for item in payloads) != list(SEEDS):
        raise ValueError("summary requires exactly seeds 3407, 3408, and 3409")
    invariant_keys = ("model_inventory_sha256", "source_sha256", "protocol_sha256", "data_sha256")
    for key in invariant_keys:
        if len({json.dumps(item[key], sort_keys=True) for item in payloads}) != 1:
            raise ValueError(f"mixed {key}")
    if len({json.dumps({k: v for k, v in item["config"].items() if k != "seed"}, sort_keys=True) for item in payloads}) != 1:
        raise ValueError("mixed probe configuration")
    by_seed = {}
    by_corpus = {corpus: [] for corpus in CORPORA}
    by_cell = {}
    for item in payloads:
        expected = {(corpus, parameter) for corpus in CORPORA for parameter in item["config"]["parameters"]}
        observed = {(row["corpus"], row["parameter"]) for row in item["results"]}
        if observed != expected or len(item["results"]) != len(expected):
            raise ValueError("incomplete or duplicated result cells")
        seed = item["config"]["seed"]
        by_seed[seed] = sum(row["score"] for row in item["results"]) / len(item["results"])
        for corpus in CORPORA:
            by_corpus[corpus].append(sum(row["score"] for row in item["results"] if row["corpus"] == corpus) / len(item["config"]["parameters"]))
        for row in item["results"]:
            by_cell.setdefault((row["corpus"], row["parameter"]), []).append(row["score"])
    seed_group = _mean_sem([by_seed[seed] for seed in SEEDS])
    corpus_groups = {corpus: _mean_sem(values) for corpus, values in by_corpus.items()}
    cell_groups = {
        f"{corpus}|{parameter}": _mean_sem(values)
        for (corpus, parameter), values in sorted(by_cell.items())
    }
    gate = {
        "grand_mean_positive": seed_group["mean"] > 0,
        "positive_seed_means": sum(value > 0 for value in seed_group["values"]),
        "positive_corpus_means": sum(group["mean"] > 0 for group in corpus_groups.values()),
    }
    gate["advance_to_4b"] = gate["grand_mean_positive"] and gate["positive_seed_means"] >= 2 and gate["positive_corpus_means"] >= 2
    return {
        "schema_version": 1,
        "status": "ok",
        "model": payloads[0]["model"],
        "run_count": len(payloads),
        "seed_score": seed_group,
        "corpus_score": corpus_groups,
        "cell_score": cell_groups,
        "gate": gate,
        "audit": {key: payloads[0][key] for key in invariant_keys},
    }


def _self_check() -> None:
    import torch

    generator = torch.Generator().manual_seed(7)
    calibration = [row for row in torch.randn(5, 4, generator=generator)]
    test = [row for row in torch.randn(7, 4, generator=generator)]
    result = _score(calibration, test)
    gc = torch.stack(calibration).double()
    gt = torch.stack(test).double()
    fcal = gc.T @ gc / len(gc)
    ftest = gt.T @ gt / len(gt)
    mean = gc.mean(0)
    coefficient = float(mean @ fcal @ mean) / float(mean @ mean) ** 2
    rank1 = coefficient * torch.outer(mean, mean)
    diagonal = torch.diag(torch.diag(fcal))
    denominator = float(torch.linalg.norm(ftest))
    assert math.isclose(result["rank1_test_relative_frobenius_error"], float(torch.linalg.norm(ftest - rank1)) / denominator, rel_tol=1e-10)
    assert math.isclose(result["diagonal_test_relative_frobenius_error"], float(torch.linalg.norm(ftest - diagonal)) / denominator, rel_tol=1e-10)
    print(json.dumps({"self_check": "ok"}))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--inventory", type=Path, help="create or verify a model inventory and exit")
    parser.add_argument("--summarize", type=Path, nargs="*", help="summarize exactly three seed outputs")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-inventory", type=Path)
    parser.add_argument("--gsm", type=Path, default=DEFAULT_GSM)
    parser.add_argument("--reverse-dir", type=Path, default=DEFAULT_REVERSE)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return
    if args.inventory:
        if not args.model:
            parser.error("--inventory requires --model")
        payload = inventory(args.model, args.inventory)
        print(json.dumps({"status": "ok", "files": len(payload["files"]), "output": str(args.inventory)}))
        return
    if args.summarize is not None:
        if not args.summarize or not args.output:
            parser.error("--summarize requires inputs and --output")
        payload = summarize(args.summarize)
    else:
        if not all((args.model, args.model_inventory, args.seed is not None, args.output)):
            parser.error("run mode requires --model, --model-inventory, --seed, and --output")
        payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output), "gate": payload.get("gate")}))


if __name__ == "__main__":
    main()
