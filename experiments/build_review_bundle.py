#!/usr/bin/env python3
"""Build a double-blind-safe bundle of hashed raw evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).parents[1].resolve()
FORBIDDEN = (b"/" b"home/",)


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
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is None or _has_forbidden(handle.read()):
                    return True
    return False


def _clean_string(value, code_root=None, host=None):
    if host and value == host:
        return "anonymized-gpu-node"
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/review_bundle"))
    parser.add_argument("--public", type=Path, nargs="*", default=[])
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        host = "air-" + "node-03"
        sample = _clean({"host": host, "path": str(ROOT / "runs/x.json"), "python": "/" + "home/group/user/env/bin/python"}, host=host)
        encoded = json.dumps(sample).encode()
        assert not _has_forbidden(encoded) and _has_forbidden(("/" + "home/user").encode())
        print(json.dumps({"self_check": "ok"}))
        return
    if not args.submission_manifest:
        parser.error("--submission-manifest is required")
    output, result = build(args.submission_manifest, args.output_dir, args.public)
    print(json.dumps({"status": "ok", "raw_artifacts": len(result["raw_artifacts"]), "output": str(output)}))


if __name__ == "__main__":
    main()
