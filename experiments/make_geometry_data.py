#!/usr/bin/env python3
"""Pack checked-in instruction and reversal corpora into fixed-length DLLM rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunks(documents, tokenizer, length):
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer needs an EOS token")
    tokens = []
    for document in documents:
        tokens.extend(tokenizer.encode(document, add_special_tokens=False))
        tokens.append(eos)
    return [tokens[start : start + length] for start in range(0, len(tokens) - length + 1, length)]


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--mt-bench", type=Path)
    parser.add_argument("--reversal", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        class Tokenizer:
            eos_token_id = 9

            @staticmethod
            def encode(text, add_special_tokens=False):
                return [int(item) for item in text.split()]

        assert _chunks(["1 2", "3 4 5"], Tokenizer(), 3) == [[1, 2, 9], [3, 4, 5]]
        print(json.dumps({"self_check": "ok"}))
        return
    if not all((args.tokenizer, args.mt_bench, args.reversal, args.output)):
        parser.error("--tokenizer, --mt-bench, --reversal, and --output are required")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    mt_documents = ["\n".join(item["turns"]) for item in _jsonl(args.mt_bench)]
    reversal_documents = []
    for path in args.reversal:
        reversal_documents.extend(item["prompt"] + item["completion"] for item in _jsonl(path))
    corpora = {
        "mt_bench": _chunks(mt_documents, tokenizer, args.sequence_length),
        "reversal": _chunks(reversal_documents, tokenizer, args.sequence_length),
    }
    if any(len(rows) < 64 for rows in corpora.values()):
        raise ValueError("each corpus must produce at least 64 fixed-length rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task_id, (corpus, rows) in enumerate(corpora.items()):
            for ids in rows:
                handle.write(json.dumps({"split": "eval", "task_id": task_id, "corpus": corpus, "input_ids": ids}) + "\n")
    sources = [args.mt_bench, *args.reversal]
    manifest = {
        "schema_version": 1,
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output),
        "source_sha256": {str(path.resolve()): _sha256(path) for path in sources},
        "tokenizer": str(args.tokenizer.resolve()),
        "sequence_length": args.sequence_length,
        "records": {name: len(rows) for name, rows in corpora.items()},
        "packing": "source-order documents separated by EOS; non-overlapping fixed-length chunks; final short chunk dropped",
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
