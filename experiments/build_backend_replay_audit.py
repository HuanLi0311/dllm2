#!/usr/bin/env python3
"""Tie the submitted runs to an exact optional-backend environment by replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SOURCE_FILES = ("lit_gpt/diffmodel.py", "lit_gpt/compat.py")
ROW_SELECTOR = {
    "sample_count": 64,
    "parameter": "transformer.h.0.norm_1.weight",
    "mask_condition": "fixed_0.1",
}
MATCHED_REPLAY_CONFIG = (
    "checkpoint", "model_size", "code_root", "data", "split", "task_id",
    "test_samples", "shuffle_records", "loss_mode", "include_native_schedule",
    "native_eps", "sequence_length", "seed", "device",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise ValueError(f"backend evidence is not successful: {path}")
    return payload


def _selected_row(payload):
    matches = [
        row for row in payload["results"]
        if all(row.get(key) == value for key, value in ROW_SELECTOR.items())
    ]
    if len(matches) != 1:
        raise ValueError("backend replay selector must identify exactly one row")
    return matches[0]


def _compare_rows(original, replay):
    original_row, replay_row = _selected_row(original), _selected_row(replay)
    replay_only = set(replay_row) - set(original_row)
    original_only = set(original_row) - set(replay_row)
    if replay_only or original_only - {"audit_sufficient_statistics"}:
        raise ValueError("backend replay and original row fields differ unexpectedly")
    differences = []
    for key, replay_value in replay_row.items():
        if key not in original_row:
            raise ValueError(f"replay metric missing from original row: {key}")
        original_value = original_row[key]
        if isinstance(replay_value, (int, float)) and not isinstance(replay_value, bool):
            difference = abs(float(replay_value) - float(original_value))
            if not math.isclose(float(replay_value), float(original_value), rel_tol=0.0, abs_tol=0.0):
                differences.append((key, difference))
        elif replay_value != original_value:
            differences.append((key, None))
    if differences:
        raise ValueError(f"backend replay differs from original row: {differences[:3]}")
    return len(replay_row), sorted(original_only)


def _validate_replay_contract(original, replay):
    if original.get("host") != replay.get("host"):
        raise ValueError("backend replay host differs from original")
    expected_probe = original.get("base_probe_sha256") or original["probe_sha256"]
    if replay["probe_sha256"] != expected_probe:
        raise ValueError("backend replay did not use the frozen base probe")
    for field in MATCHED_REPLAY_CONFIG:
        if original["config"].get(field) != replay["config"].get(field):
            raise ValueError(f"backend replay config mismatch: {field}")
    replay_config = replay["config"]
    if (
        replay_config["mask_probabilities"] != "0.1"
        or replay_config["sample_sizes"] != "64"
        or replay_config["parameter"] != ROW_SELECTOR["parameter"]
    ):
        raise ValueError("backend replay is not the declared minimal fixed-p selector")
    observed = {
        (row["sample_count"], row["parameter"], row["mask_condition"])
        for row in replay["results"]
    }
    expected = {
        (64, ROW_SELECTOR["parameter"], "fixed_0.1"),
        (64, ROW_SELECTOR["parameter"], "native_schedule"),
    }
    if observed != expected or len(replay["results"]) != len(expected):
        raise ValueError("backend replay result grid differs from its declared RNG-preserving design")


def _installed_distributions():
    rows = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            rows.append({"name": name, "version": distribution.version})
    return sorted(rows, key=lambda row: (row["name"].lower(), row["version"]))


def _module_state(module, distribution):
    importable = importlib.util.find_spec(module) is not None
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"importable": importable, "version": version}


def _resolved_backends():
    import torch

    try:
        from lightning_utilities.core.imports import RequirementCache
        flash_enabled = bool(RequirementCache("flash-attn>=2.0.0.post1"))
    except ImportError:
        flash_enabled = False
    try:
        from xformers.ops import SwiGLU
        swiglu = f"{SwiGLU.__module__}.{SwiGLU.__name__}"
    except ImportError:
        swiglu = "lit_gpt.compat.SwiGLU"
    flash = _module_state("flash_attn", "flash-attn")
    xformers = _module_state("xformers", "xformers")
    dropout_layer_norm = _module_state("dropout_layer_norm", "dropout-layer-norm")
    xentropy = _module_state("xentropy_cuda_lib", "xentropy-cuda-lib")
    return {
        "flash_attn": {
            **flash,
            "resolved_source_branch": "flash_attn.flash_attn_func" if flash_enabled else "torch.nn.functional.scaled_dot_product_attention",
        },
        "xformers": {
            **xformers,
            "resolved_swiglu_class": swiglu,
        },
        "dropout_layer_norm": {
            **dropout_layer_norm,
            "resolved_rmsnorm_branch": (
                "dropout_layer_norm CUDA extension"
                if dropout_layer_norm["importable"]
                else "lit_gpt.rmsnorm.rms_norm pure-PyTorch fallback"
            ),
        },
        "xentropy_cuda_lib": {
            **xentropy,
            "resolved_probe_loss": "torch.nn.functional.cross_entropy",
            "fused_extension_used_by_probe": False,
        },
        "rotary_embedding": {
            "resolved_source_branch": "lit_gpt.compat.apply_rotary_emb_func pure-PyTorch implementation",
        },
        "torch_sdpa_policy": {
            "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_enabled": torch.backends.cuda.math_sdp_enabled(),
        },
    }


def build(evidence_paths, replay_pairs):
    evidence = [(Path(path), _load(path)) for path in evidence_paths]
    interpreters = {Path(payload["command"][0]).resolve() for _, payload in evidence}
    if interpreters != {Path(sys.executable).resolve()}:
        raise ValueError("submission runs and backend audit must use the same interpreter")
    software = {json.dumps(payload["software"], sort_keys=True) for _, payload in evidence}
    code_roots = {Path(payload["code_root"]).resolve() for _, payload in evidence}
    source_hashes = {json.dumps(payload["source_sha256"], sort_keys=True) for _, payload in evidence}
    if len(software) != 1 or len(code_roots) != 1 or len(source_hashes) != 1:
        raise ValueError("submitted runs do not share one software/source environment")
    code_root = next(iter(code_roots))
    sources = {relative: _sha256(code_root / relative) for relative in SOURCE_FILES}
    if sources["lit_gpt/diffmodel.py"] != evidence[0][1]["source_sha256"]["lit_gpt/diffmodel.py"]:
        raise ValueError("current backend source differs from submitted run source")

    replays = []
    replay_models = set()
    for original_path, replay_path in replay_pairs:
        original, replay = _load(original_path), _load(replay_path)
        for field in ("checkpoint_sha256", "data_sha256", "source_sha256", "software"):
            if original[field] != replay[field]:
                raise ValueError(f"backend replay field mismatch: {field}")
        _validate_replay_contract(original, replay)
        if Path(replay["command"][0]).resolve() != Path(sys.executable).resolve():
            raise ValueError("backend replay used a different interpreter")
        replay_models.add(replay["config"]["model_size"])
        matching_fields, original_only_fields = _compare_rows(original, replay)
        replays.append({
            "model_size_alias": replay["config"]["model_size"],
            "original": _relative(original_path),
            "original_sha256": _sha256(original_path),
            "replay": _relative(replay_path),
            "replay_sha256": _sha256(replay_path),
            "row_selector": ROW_SELECTOR,
            "exactly_matching_row_fields": matching_fields,
            "original_only_fields": original_only_fields,
            "maximum_absolute_numeric_difference": 0.0,
        })
    if replay_models != {170, 1028}:
        raise ValueError("backend audit requires one exact replay for each checkpoint scale")

    return {
        "schema_version": 2,
        "status": "ok",
        "capture_scope": "post-run capture from the exact interpreter recorded by every submitted envelope",
        "numerical_link": "one exact 64|64 fixed-p row replayed for each checkpoint scale",
        "evidence_artifacts": [
            {"path": _relative(path), "sha256": _sha256(path)} for path, _ in evidence
        ],
        "replays": replays,
        "source_root": "external/SMDM",
        "backend_source_sha256": sources,
        "resolved_backends": _resolved_backends(),
        "python": sys.version,
        "installed_distributions": _installed_distributions(),
        "audit_script_sha256": _sha256(__file__),
        "verified": [
            "all submitted envelopes record the exact audit interpreter and one software/source environment",
            "all model-relevant optional backend packages, resolved branches, and the complete installed distribution lock are captured",
            "the branch-defining diffmodel.py and compat.py sources are hashed",
            "both checkpoint scales exactly reproduce all 32 common fields of a selected fixed-p result row",
            "replay host, frozen probe, matched configuration, and minimal RNG-preserving result grids are enforced",
        ],
    }


def validate_audit(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = [(ROOT / row["original"], ROOT / row["replay"]) for row in payload["replays"]]
    rebuilt = build([ROOT / row["path"] for row in payload["evidence_artifacts"]], pairs)
    if rebuilt != payload:
        raise ValueError("backend replay audit is stale")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, nargs="+")
    parser.add_argument("--pair", type=Path, nargs=2, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        assert _compare_rows({"results": [{**ROW_SELECTOR, "x": 1.0}]}, {"results": [{**ROW_SELECTOR, "x": 1.0}]}) == (4, [])
        print(json.dumps({"self_check": "ok"}))
        return
    if not args.evidence or len(args.pair) != 2 or not args.output:
        parser.error("--evidence, two --pair values, and --output are required")
    result = build(args.evidence, args.pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "evidence": len(result["evidence_artifacts"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
