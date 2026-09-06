#!/usr/bin/env python3
"""Fail-closed provenance manifest for the submission's reported artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import tarfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


SOURCE_CLOSURE_FIXED = ("LICENSE", "README.md", "pretrain/train_mdm.py")
SOURCE_CLOSURE_TREES = ("lit_gpt",)
SOURCE_ARCHIVE_PREFIX = "SMDM/"


@lru_cache(maxsize=None)
def _sha256(path):
    return hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, text=True, capture_output=True).stdout.strip()


def _csv_values(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _validate_payload(path, payload, probe_paths):
    probes = {_sha256(probe): probe for probe in probe_paths}
    if payload.get("probe_sha256") not in probes:
        raise ValueError(f"probe hash mismatch: {path}")
    if payload.get("base_probe_sha256") and payload["base_probe_sha256"] not in probes:
        raise ValueError(f"base probe hash mismatch: {path}")
    checkpoint = Path(payload["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    data = Path(payload["data"])
    if not data.is_file() or _sha256(data) != payload["data_sha256"]:
        raise ValueError(f"data hash mismatch: {path}")
    code_root = Path(payload["code_root"])
    for relative, digest in payload["source_sha256"].items():
        if _sha256(code_root / relative) != digest:
            raise ValueError(f"source hash mismatch for {relative}: {path}")

    config = payload["config"]
    if config["test_samples"] < 2 or not config["shuffle_records"]:
        raise ValueError(f"primary evidence must use shuffled, independent test samples: {path}")
    sample_sizes = _csv_values(config["sample_sizes"], int)
    parameters = _csv_values(config["parameter"])
    conditions = [f"fixed_{value:g}" for value in _csv_values(config["mask_probabilities"], float)]
    if config["include_native_schedule"]:
        conditions.append("native_schedule")
    expected = {(sample_count, parameter, condition) for sample_count in sample_sizes for parameter in parameters for condition in conditions}
    observed = {(row["sample_count"], row["parameter"], row["mask_condition"]) for row in payload["results"]}
    if observed != expected or len(payload["results"]) != len(expected):
        raise ValueError(f"incomplete or duplicated result grid: {path}")

    for row in payload["results"]:
        numeric = [value for value in row.values() if isinstance(value, (int, float))]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite metric: {path}")
        if row["evaluation"] != "split_sample" or row["test_sample_count"] != config["test_samples"]:
            raise ValueError(f"non-independent evaluation row: {path}")
        if row["seed"] != config["seed"] or row["loss_mode"] != config["loss_mode"]:
            raise ValueError(f"row/config mismatch: {path}")
        oracle = row["test_oracle_top1_relative_frobenius_error"]
        fitted_rank1 = (
            row["mean_rank1_test_relative_frobenius_error"],
            row["calibration_top1_test_relative_frobenius_error"],
            row["random_direction_test_error_mean"],
        )
        if any(oracle > error + 1e-10 for error in fitted_rank1):
            raise ValueError(f"held-out oracle ordering violated: {path}")

    command = payload["command"]
    if "--output" not in command:
        raise ValueError(f"raw command omits output path: {path}")
    command_output = Path(command[command.index("--output") + 1]).resolve()
    if command_output != path.resolve():
        raise ValueError(f"raw command/output mismatch: {path}")


def _relative(path):
    root = Path(__file__).parents[1].resolve()
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return "external/SMDM" if resolved.name == "SMDM" else resolved.name


def _sanitize_command(command, code_root=None):
    root = str(Path(__file__).parents[1].resolve()) + "/"
    external = str(Path(code_root).resolve()) if code_root else None
    result = []
    for index, value in enumerate(command):
        value = str(value)
        if index == 0:
            result.append("python")
            continue
        value = value.replace(root, "")
        if external:
            value = value.replace(external, "external/SMDM")
        result.append(value)
    return result


def _source_closure(code_root, top_level, subtree, commit):
    pathspecs = [f"{subtree}/{path}" for path in (*SOURCE_CLOSURE_FIXED, *SOURCE_CLOSURE_TREES)]
    tracked = _git(top_level, "ls-tree", "-r", "--name-only", commit, "--", *pathspecs).splitlines()
    prefix = f"{subtree}/"
    relative_paths = sorted(path[len(prefix):] for path in tracked if path.startswith(prefix))
    for required in SOURCE_CLOSURE_FIXED:
        if required not in relative_paths:
            raise ValueError(f"required source closure file is not tracked: {required}")
    if not any(path.startswith("lit_gpt/") for path in relative_paths):
        raise ValueError("source closure has no lit_gpt files")
    closure = []
    for relative in relative_paths:
        path = code_root / relative
        if not path.is_file():
            raise ValueError(f"source closure file is absent: {relative}")
        closure.append({"path": relative, "sha256": _sha256(path)})
    return closure


def _validate_source_archive(path, closure):
    observed = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.name in observed:
                raise ValueError(f"unsupported or duplicated source archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"unreadable source archive member: {member.name}")
            observed[member.name] = hashlib.sha256(handle.read()).hexdigest()
    expected = {SOURCE_ARCHIVE_PREFIX + row["path"]: row["sha256"] for row in closure}
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(name for name in set(expected) & set(observed) if expected[name] != observed[name])
        raise ValueError(f"source archive does not match closure: missing={missing}, extra={extra}, wrong_hash={wrong}")
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "prefix": SOURCE_ARCHIVE_PREFIX,
        "regular_files": len(observed),
        "verified_against_closure": True,
    }


def build(paths, extras, comparison_contract=None, backend_audit=None, source_archive=None):
    payloads = []
    root = Path(__file__).parents[1]
    probe_paths = [
        root / "agent_skills/skill-benchmark/scripts/dllm_rank1_probe.py",
        root / "experiments/run_audited_geometry_probe.py",
    ]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise ValueError(f"non-success artifact passed as evidence: {path}")
        _validate_payload(path, payload, probe_paths)
        payloads.append((path, payload))

    critical = ("mask_probabilities", "sample_sizes", "test_samples", "sequence_length", "parameter", "shuffle_records", "loss_mode", "include_native_schedule", "native_eps")
    groups = defaultdict(list)
    for path, payload in payloads:
        signature = tuple((name, json.dumps(payload["config"].get(name), sort_keys=True)) for name in critical)
        key = (payload["checkpoint_sha256"], payload["data_sha256"], payload.get("task_id"), payload["config"]["loss_mode"], payload["probe_sha256"], signature)
        groups[key].append((path, payload))
    group_checks = []
    for key, members in groups.items():
        reference = {name: members[0][1]["config"].get(name) for name in critical}
        seeds = []
        for path, payload in members:
            observed = {name: payload["config"].get(name) for name in critical}
            if observed != reference:
                raise ValueError(f"unmatched critical configuration: {path}")
            seeds.append(payload["config"]["seed"])
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds in evidence group {key}")
        if sorted(seeds) != list(range(len(seeds))) or len(seeds) < 3:
            raise ValueError(f"evidence groups require contiguous seeds from zero and at least three seeds: {key}")
        group_checks.append({"checkpoint_sha256": key[0], "data_sha256": key[1], "task_id": key[2], "loss_mode": key[3], "probe_sha256": key[4], "critical_config": reference, "seeds": sorted(seeds)})

    code_roots = sorted({payload["code_root"] for _, payload in payloads})
    if len(code_roots) != 1 or source_archive is None:
        raise ValueError("submission requires exactly one code root and its source archive")
    repositories = []
    for root_string in code_roots:
        code_root = Path(root_string)
        top_level = Path(_git(code_root, "rev-parse", "--show-toplevel"))
        subtree = str(code_root.resolve().relative_to(top_level.resolve()))
        code_subtree_dirty = bool(_git(top_level, "status", "--porcelain", "--untracked-files=all", "--", subtree))
        if code_subtree_dirty:
            raise ValueError(f"refusing dirty experiment source subtree: {code_root}")
        commit = _git(code_root, "rev-parse", "HEAD")
        closure = _source_closure(code_root, top_level, subtree, commit)
        repositories.append({
            "path": "external/SMDM",
            "commit": commit,
            "repository_tree": _git(code_root, "rev-parse", "HEAD^{tree}"),
            "source_subtree_tree": _git(top_level, "rev-parse", f"{commit}:{subtree}"),
            "repository_dirty": bool(_git(code_root, "status", "--porcelain")),
            "code_subtree_dirty": code_subtree_dirty,
            "source_closure_definition": "LICENSE, README.md, pretrain/train_mdm.py, and every tracked file below lit_gpt/",
            "source_closure": closure,
            "source_archive": _validate_source_archive(source_archive, closure),
        })
    package_names = (
        "torch", "safetensors", "transformers", "lightning", "lightning-utilities",
        "typing-extensions", "einops", "numpy", "matplotlib", "xformers",
        "flash-attn", "dropout-layer-norm", "xentropy-cuda-lib",
    )
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    contract = None
    if comparison_contract:
        sys.path.insert(0, str(Path(__file__).parent))
        from build_comparison_contract import validate_contract
        validated = validate_contract(comparison_contract)
        geometry_paths = {Path(path).resolve() for path, _ in payloads}
        contract_paths = {(root / artifact["path"]).resolve() for artifact in validated["artifacts"]}
        if contract_paths - geometry_paths:
            raise ValueError("comparison contract contains evidence absent from the submission manifest")
        contract = {
            "path": _relative(comparison_contract),
            "sha256": _sha256(comparison_contract),
            "verified": validated["verified"],
        }
    backend = None
    if backend_audit:
        sys.path.insert(0, str(Path(__file__).parent))
        from build_backend_replay_audit import validate_audit
        validated = validate_audit(backend_audit)
        geometry_paths = {Path(path).resolve() for path, _ in payloads}
        backend_paths = {(root / artifact["path"]).resolve() for artifact in validated["evidence_artifacts"]}
        if backend_paths != geometry_paths:
            raise ValueError("backend audit must cover exactly the submitted geometry artifacts")
        backend = {
            "path": _relative(backend_audit),
            "sha256": _sha256(backend_audit),
            "resolved_backends": validated["resolved_backends"],
            "verified": validated["verified"],
        }
    return {
        "schema_version": 4,
        "status": "ok",
        "probes": [{"path": _relative(probe), "sha256": _sha256(probe)} for probe in probe_paths],
        "geometry_artifacts": [{"path": _relative(path), "sha256": _sha256(path), "command": _sanitize_command(payload["command"], payload["code_root"]), "probe_sha256": payload["probe_sha256"]} for path, payload in payloads],
        "matched_groups": group_checks,
        "comparison_contract": contract,
        "execution_backend_audit": backend,
        "repositories": repositories,
        "experiment_software": list({json.dumps(payload["software"], sort_keys=True): payload["software"] for _, payload in payloads}.values()),
        "manifest_environment": {"python": sys.version, "packages": packages},
        "extra_artifacts": [{"path": _relative(path), "sha256": _sha256(path)} for path in extras],
        "manifest_script_sha256": _sha256(Path(__file__)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, nargs="+")
    parser.add_argument("--extra", type=Path, nargs="*", default=[])
    parser.add_argument("--comparison-contract", type=Path)
    parser.add_argument("--backend-audit", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        assert len(_sha256(Path(__file__))) == 64 and _csv_values("8,4,8", int) == [8, 4, 8]
        print(json.dumps({"self_check": "ok"}))
        return
    if not args.geometry or not args.comparison_contract or not args.backend_audit or not args.source_archive or not args.output:
        parser.error("--geometry, --comparison-contract, --backend-audit, --source-archive, and --output are required")
    result = build(
        args.geometry,
        args.extra,
        args.comparison_contract,
        args.backend_audit,
        args.source_archive,
    )
    result["manifest_command"] = _sanitize_command([sys.executable, *sys.argv])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "geometry_artifacts": len(result["geometry_artifacts"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
