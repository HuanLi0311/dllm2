#!/usr/bin/env python3
"""Build a double-blind-safe bundle of hashed raw evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
FORBIDDEN = (b"/" b"home/", b"air-node", b"JJ_Group", b"lih2511")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256(path):
    return _sha256_bytes(Path(path).read_bytes())


def _has_forbidden(data):
    return any(token in data for token in FORBIDDEN)


def _archive_has_forbidden(path):
    if not path.name.endswith(".tar.gz"):
        return False
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if _has_forbidden(member.name.encode()):
                return True
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is None or _has_forbidden(handle.read()):
                    return True
    return False


def _clean_string(value, code_root=None, host=None):
    if host:
        value = value.replace(host, "anonymized-gpu-node")
    value = value.replace(str(ROOT) + "/", "")
    if code_root:
        value = value.replace(code_root, "external/SMDM")
    if value.startswith("/" + "home/"):
        return "python" if value.endswith("/python") else Path(value).name
    return value


def _clean(value, code_root=None, host=None):
    if isinstance(value, dict):
        return {key: _clean(item, code_root, host) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item, code_root, host) for item in value]
    if isinstance(value, str):
        return _clean_string(value, code_root, host)
    return value


def _write_gzip(path, payload):
    data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if _has_forbidden(data):
        raise ValueError(f"identity-bearing string remains in sanitized payload: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(data)


def build(submission_manifest, output_dir, public_artifacts):
    manifest = json.loads(submission_manifest.read_text(encoding="utf-8"))
    raw_artifacts = []
    for artifact in manifest["geometry_artifacts"]:
        source = ROOT / artifact["path"]
        payload = json.loads(source.read_text(encoding="utf-8"))
        cleaned = _clean(payload, payload.get("code_root"), payload.get("host"))
        cleaned["release_provenance"] = {
            "internal_artifact": artifact["path"],
            "internal_sha256": artifact["sha256"],
        }
        destination = output_dir / "raw" / Path(artifact["path"]).with_suffix(".json.gz")
        _write_gzip(destination, cleaned)
        raw_artifacts.append({
            "internal_artifact": artifact["path"],
            "internal_sha256": artifact["sha256"],
            "release_artifact": str(destination.resolve().relative_to(ROOT)),
            "release_sha256": _sha256(destination),
        })

    public = []
    sanitized_public = []
    for path in public_artifacts:
        relative = path.resolve().relative_to(ROOT)
        data = path.read_bytes()
        if path.suffix.lower() == ".json" and _has_forbidden(data):
            payload = json.loads(data)
            cleaned = _clean(payload, payload.get("code_root"), payload.get("host"))
            destination = output_dir / "public" / relative.with_suffix(".json.gz")
            _write_gzip(destination, cleaned)
            sanitized_public.append({
                "internal_artifact": str(relative),
                "internal_sha256": _sha256(path),
                "release_artifact": str(destination.resolve().relative_to(ROOT)),
                "release_sha256": _sha256(destination),
            })
            continue
        if _has_forbidden(data) or _archive_has_forbidden(path):
            raise ValueError(f"identity-bearing string in public artifact: {path}")
        public.append({"path": str(relative), "sha256": _sha256(path)})
    result = {
        "schema_version": 1,
        "status": "ok",
        "submission_manifest": {
            "path": str(submission_manifest.resolve().relative_to(ROOT)),
            "sha256": _sha256(submission_manifest),
        },
        "raw_artifacts": raw_artifacts,
        "sanitized_public_artifacts": sanitized_public,
        "public_artifacts": public,
        "bundle_script_sha256": _sha256(__file__),
    }
    output = output_dir / "release_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, indent=2) + "\n").encode()
    if _has_forbidden(encoded):
        raise ValueError("identity-bearing string remains in release manifest")
    output.write_bytes(encoded)
    return output, result


def _root_path(relative):
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"artifact escapes repository root: {relative}") from error
    return path


def _check_hash(path, expected, label):
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {label}: {path}: {actual} != {expected}")


def _check_released_payload(path):
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rb") as handle:
            data = handle.read()
        if _has_forbidden(data):
            raise ValueError(f"identity-bearing string in expanded artifact: {path}")
        return json.loads(data)
    data = path.read_bytes()
    if _has_forbidden(data) or _archive_has_forbidden(path):
        raise ValueError(f"identity-bearing string in public artifact: {path}")
    return None


def verify(release_manifest, require_internal=False):
    release_manifest = release_manifest.resolve()
    manifest_bytes = release_manifest.read_bytes()
    if _has_forbidden(manifest_bytes):
        raise ValueError(f"identity-bearing string in release manifest: {release_manifest}")
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "ok":
        raise ValueError("release manifest status is not ok")
    builder_script_matches = manifest.get("bundle_script_sha256") == _sha256(__file__)
    if require_internal and not builder_script_matches:
        raise ValueError("bundle builder differs from the release manifest")

    source_manifest = manifest["submission_manifest"]
    _check_hash(_root_path(source_manifest["path"]), source_manifest["sha256"], "submission manifest")
    bundle_root = release_manifest.parent
    expected_bundle_files = {release_manifest}
    checked = 0
    internal_sources_checked = 0
    missing_internal_sources = 0
    changed_internal_sources = 0

    for section in ("raw_artifacts", "sanitized_public_artifacts"):
        for artifact in manifest.get(section, []):
            internal = _root_path(artifact["internal_artifact"])
            if internal.is_file():
                internal_sources_checked += 1
                if _sha256(internal) != artifact["internal_sha256"]:
                    changed_internal_sources += 1
                    if require_internal:
                        raise ValueError(f"internal artifact changed since release: {internal}")
            else:
                missing_internal_sources += 1
                if require_internal:
                    raise ValueError(f"missing internal artifact: {internal}")
            released = _root_path(artifact["release_artifact"])
            try:
                released.relative_to(bundle_root)
            except ValueError as error:
                raise ValueError(f"released artifact escapes bundle: {released}") from error
            _check_hash(released, artifact["release_sha256"], "released artifact")
            payload = _check_released_payload(released)
            if section == "raw_artifacts" and payload.get("release_provenance") != {
                "internal_artifact": artifact["internal_artifact"],
                "internal_sha256": artifact["internal_sha256"],
            }:
                raise ValueError(f"release provenance mismatch: {released}")
            expected_bundle_files.add(released)
            checked += 1

    for artifact in manifest.get("public_artifacts", []):
        path = _root_path(artifact["path"])
        _check_hash(path, artifact["sha256"], "public artifact")
        _check_released_payload(path)
        checked += 1

    actual_bundle_files = {path.resolve() for path in bundle_root.rglob("*") if path.is_file()}
    if actual_bundle_files != expected_bundle_files:
        raise ValueError(
            f"bundle file-set mismatch: missing={expected_bundle_files - actual_bundle_files}, "
            f"extra={actual_bundle_files - expected_bundle_files}"
        )
    return {
        "status": "ok", "checked_artifacts": checked,
        "internal_sources_checked": internal_sources_checked,
        "missing_internal_sources": missing_internal_sources,
        "changed_internal_sources": changed_internal_sources,
        "builder_script_matches": builder_script_matches,
        "bundle_files": len(actual_bundle_files),
    }


def _self_check():
    host = "air-" + "node-03"
    sample = _clean(
        {"host": host, "path": "/" + "home/group/user/env/bin/python"}, host=host
    )
    encoded = json.dumps(sample).encode()
    assert not _has_forbidden(encoded) and _has_forbidden(("/" + "home/user").encode())
    with tempfile.TemporaryDirectory(prefix="bundle_self_check_", dir=ROOT / "runs") as temporary:
        temporary = Path(temporary)
        source = temporary / "source.json"
        source.write_text(json.dumps({"status": "ok", "host": host, "path": "/" + "home/user/input"}))
        submission = temporary / "submission.json"
        submission.write_text(json.dumps({
            "geometry_artifacts": [{
                "path": str(source.relative_to(ROOT)), "sha256": _sha256(source),
            }]
        }))
        release_manifest, _ = build(submission, temporary / "bundle", [])
        result = verify(release_manifest, require_internal=True)
        assert result == {
            "status": "ok", "checked_artifacts": 1,
            "internal_sources_checked": 1, "missing_internal_sources": 0,
            "changed_internal_sources": 0,
            "builder_script_matches": True,
            "bundle_files": 2,
        }
        released = next((temporary / "bundle" / "raw").rglob("*.json.gz"))
        original = released.read_bytes()
        released.write_bytes(original + b"tamper")
        try:
            verify(release_manifest, require_internal=True)
        except ValueError:
            pass
        else:
            raise AssertionError("bundle verifier accepted a modified artifact")
    print(json.dumps({"self_check": "ok", "bundle_verify": "ok", "tamper_rejected": True}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/review_bundle"))
    parser.add_argument("--public", type=Path, nargs="*", default=[])
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--require-internal", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.require_internal and not args.verify:
        parser.error("--require-internal requires --verify")
    if args.self_check:
        _self_check()
        return
    if args.verify:
        manifest = args.verify / "release_manifest.json" if args.verify.is_dir() else args.verify
        print(json.dumps(verify(manifest, require_internal=args.require_internal)))
        return
    if not args.submission_manifest:
        parser.error("--submission-manifest is required")
    output, result = build(args.submission_manifest, args.output_dir, args.public)
    print(json.dumps({"status": "ok", "raw_artifacts": len(result["raw_artifacts"]), "output": str(output)}))


if __name__ == "__main__":
    main()
