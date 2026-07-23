"""Immutable dataset manifest helpers shared by training and benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional


PROTECTED_DATASET_TOKENS = ("repro", "frozen")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sibling_manifest_path(dataset_path: str) -> str:
    path = Path(dataset_path)
    return str(path.with_name(path.stem + "_manifest.json"))


def is_protected_dataset_path(path: str) -> bool:
    normalized = os.path.abspath(path).replace("\\", "/").lower()
    basename = os.path.basename(normalized)
    return (
        any(token in basename for token in PROTECTED_DATASET_TOKENS)
        or "/datasets/" in normalized
    )


def load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def verify_dataset_manifest(
    dataset_path: str,
    manifest_path: Optional[str] = None,
    split_path: Optional[str] = None,
    require_manifest: bool = False,
) -> Optional[Dict]:
    manifest_path = manifest_path or sibling_manifest_path(dataset_path)
    if not os.path.isfile(manifest_path):
        if require_manifest or is_protected_dataset_path(dataset_path):
            raise FileNotFoundError(
                f"immutable dataset manifest is required: {manifest_path}")
        return None
    manifest = load_manifest(manifest_path)
    expected_dataset = str(manifest.get("dataset_sha256", ""))
    actual_dataset = sha256_file(dataset_path)
    if actual_dataset != expected_dataset:
        raise RuntimeError(
            "immutable dataset hash mismatch: "
            f"expected={expected_dataset}, actual={actual_dataset}, "
            f"path={os.path.abspath(dataset_path)}")
    expected_lines = int(manifest["line_count"])
    with open(dataset_path, "r", encoding="utf-8") as stream:
        actual_lines = sum(1 for line in stream if line.strip())
    if actual_lines != expected_lines:
        raise RuntimeError(
            f"immutable dataset line count mismatch: "
            f"expected={expected_lines}, actual={actual_lines}")
    if split_path:
        expected_split = str(manifest.get("split_sha256", ""))
        actual_split = sha256_file(split_path)
        if actual_split != expected_split:
            raise RuntimeError(
                "immutable split hash mismatch: "
                f"expected={expected_split}, actual={actual_split}, "
                f"path={os.path.abspath(split_path)}")
    return manifest
