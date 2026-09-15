#!/usr/bin/env python3
"""Validate LOCI Vertex batch responses and write one canonical response set."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.extraction.extract_landmarks import (
    response_validator,
)


SCHEMA = "farfield.loci_vlm_response_manifest.v1"
REQUEST_MANIFEST_SCHEMA = "farfield.loci_vlm_request_manifest.v1"
RAW_KEYS = frozenset({"key", "processed_time", "request", "response", "status"})
RETRY_RE = re.compile(r"retry_[0-9]+\Z")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _strict_loads(text: str, where: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError(f"{where}: invalid strict JSON: {error}") from error


def _scan_jsonl(path: Path, visit: Callable[[int, Any], None]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular JSONL file: {path}")
    digest = hashlib.sha256()
    size = records = 0
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            digest.update(raw_line)
            size += len(raw_line)
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank record")
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid UTF-8: {error}") from error
            visit(line_number, _strict_loads(text, f"{path}:{line_number}"))
            records += 1
    return {"bytes": size, "records": records, "sha256": digest.hexdigest()}


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"path escapes artifact root: {path}") from error


def _manifest_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("request shard path must be a non-empty string")
    path = root / relative
    _relative(root, path)
    return path


def _mapping_keys(dataset_dir: Path, manifest: dict[str, Any]) \
        -> tuple[list[str], dict[str, Any]]:
    path = dataset_dir / "pano_id_mapping.csv"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular mapping file: {path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    keys = []
    for row in rows:
        try:
            key = Path(row["filename"]).stem
            pano_id = row["pano_id"]
        except KeyError as error:
            raise ValueError(f"invalid mapping columns in {path}") from error
        if key.split(",", 1)[0] != pano_id:
            raise ValueError(f"mapping pano_id does not match filename: {row!r}")
        keys.append(key)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError(f"mapping keys are empty or duplicated: {path}")
    digest = artifact.sha256_file(path)
    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict) \
            or upstream.get("pano_id_mapping_sha256") != digest:
        raise ValueError("active pano_id_mapping.csv differs from request manifest")
    return keys, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sha256": digest,
    }


def _drop_null_defaults(value: Any) -> Any:
    """Remove only null-valued object fields added by Vertex request echoing."""
    if isinstance(value, dict):
        return {
            key: _drop_null_defaults(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_null_defaults(item) for item in value]
    return value


def _discover_results(results_dir: Path, stems: set[str]) -> dict[str, Path]:
    if results_dir.is_symlink() or not results_dir.is_dir():
        raise ValueError(f"missing regular results directory: {results_dir}")
    candidates = sorted(results_dir.glob("*/*/predictions.jsonl"))
    if candidates != sorted(results_dir.rglob("*.jsonl")):
        raise ValueError(f"unexpected JSONL layout under {results_dir}")
    by_stem: dict[str, list[Path]] = {}
    for path in candidates:
        relative = path.relative_to(results_dir)
        if not relative.parts[1].startswith("prediction-"):
            raise ValueError(f"unexpected prediction directory: {path}")
        by_stem.setdefault(relative.parts[0], []).append(path)
    if set(by_stem) != stems:
        raise ValueError(
            f"request/result shard mismatch under {results_dir}: "
            f"requests={sorted(stems)}, results={sorted(by_stem)}")
    duplicates = {key: value for key, value in by_stem.items()
                  if len(value) != 1}
    if duplicates:
        raise ValueError(f"ambiguous result files: {duplicates}")
    return {key: value[0] for key, value in by_stem.items()}


def _read_request_file(
    path: Path,
    *,
    root: Path,
    expected: set[str],
    request_digests: dict[str, str] | None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any] | None]:
    keys: dict[str, str] = {}
    schema: dict[str, Any] | None = None

    def visit(line_number: int, value: Any) -> None:
        nonlocal schema
        if not isinstance(value, dict) or set(value) != {"key", "request"}:
            raise ValueError(f"{path}:{line_number}: invalid request record shape")
        key, request = value["key"], value["request"]
        if not isinstance(key, str) or key not in expected:
            raise ValueError(f"{path}:{line_number}: unexpected request key {key!r}")
        if key in keys or not isinstance(request, dict):
            raise ValueError(f"{path}:{line_number}: duplicate key or invalid request")
        digest = artifact.sha256_json(request)
        if request_digests is not None and request_digests.get(key) != digest:
            raise ValueError(f"{path}:{line_number}: retry request differs from original")
        keys[key] = digest
        generation = request.get("generationConfig")
        current_schema = generation.get("responseSchema") \
            if isinstance(generation, dict) else None
        if not isinstance(current_schema, dict):
            raise ValueError(f"{path}:{line_number}: missing response schema")
        if schema is None:
            schema = current_schema
        elif current_schema != schema:
            raise ValueError(f"{path}:{line_number}: response schema changed")

    metadata = _scan_jsonl(path, visit)
    metadata["path"] = _relative(root, path)
    return metadata, keys, schema


def finalize(dataset_dir: Path, artifact_root: Path) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve()
    root = artifact_root.expanduser().resolve()
    manifest_path = root / "request_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing regular request manifest: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _strict_loads(manifest_bytes.decode("utf-8"), str(manifest_path))
    if not isinstance(manifest, dict) \
            or manifest.get("schema") != REQUEST_MANIFEST_SCHEMA \
            or manifest.get("complete") is not True \
            or manifest.get("validation", {}).get("status") != "PASS":
        raise ValueError("request manifest is not a completed validated LOCI bundle")
    if manifest.get("dataset") != dataset_dir.name:
        raise ValueError("request manifest belongs to a different dataset")
    inference = manifest.get("intended_inference")
    model = inference.get("model") if isinstance(inference, dict) else None
    if not isinstance(model, str) or not model:
        raise ValueError("request manifest does not bind an inference model")

    mapping_order, mapping_metadata = _mapping_keys(dataset_dir, manifest)
    expected = set(mapping_order)
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards \
            or any(not isinstance(item, dict) or set(item) != {
                "path", "requests", "bytes", "sha256"} for item in shards):
        raise ValueError("request manifest has invalid shard records")
    if manifest.get("request_set_sha256") != artifact.sha256_json(shards):
        raise ValueError("request manifest request_set_sha256 mismatch")

    request_sources = []
    request_digests: dict[str, str] = {}
    shard_keys: dict[str, set[str]] = {}
    response_schema: dict[str, Any] | None = None
    for shard in shards:
        path = _manifest_path(root, shard["path"])
        metadata, keys, schema = _read_request_file(
            path, root=root, expected=expected, request_digests=None)
        if (metadata["bytes"] != shard["bytes"]
                or metadata["records"] != shard["requests"]
                or metadata["sha256"] != shard["sha256"]):
            raise ValueError(f"request shard differs from manifest: {path}")
        overlap = set(request_digests).intersection(keys)
        if overlap:
            raise ValueError(f"request keys occur in multiple shards: {sorted(overlap)}")
        request_digests.update(keys)
        stem = path.stem
        if stem in shard_keys:
            raise ValueError(f"duplicate request shard stem: {stem}")
        shard_keys[stem] = set(keys)
        metadata.update({"role": "original", "requests": metadata.pop("records")})
        request_sources.append(metadata)
        if response_schema is None:
            response_schema = schema
        elif schema != response_schema:
            raise ValueError("response schema differs between request shards")

    if set(request_digests) != expected:
        raise ValueError("request keys differ from active pano_id_mapping.csv")
    counts = manifest.get("counts")
    request_bytes = sum(item["bytes"] for item in request_sources)
    expected_counts = {
        "pano_id_mapping_rows": len(mapping_order),
        "request_keys": len(mapping_order),
        "unique_request_keys": len(mapping_order),
        "shards": len(shards),
        "request_jsonl_bytes": request_bytes,
    }
    if not isinstance(counts, dict) or any(
            counts.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("request manifest counts do not match request sources")
    assert response_schema is not None
    schema_digest = artifact.sha256_json(response_schema)
    if manifest.get("prompt_contract", {}).get(
            "response_schema_canonical_sha256") != schema_digest:
        raise ValueError("persisted response schema differs from request manifest")

    original_results = _discover_results(
        root / "sentences" / "results", set(shard_keys))
    sources: list[tuple[str, str | None, Path, Path, set[str]]] = [
        ("original", None, _manifest_path(root, shard["path"]),
         original_results[Path(shard["path"]).stem],
         shard_keys[Path(shard["path"]).stem])
        for shard in shards
    ]

    retries_root = root / "retries"
    if retries_root.exists():
        if retries_root.is_symlink() or not retries_root.is_dir():
            raise ValueError(f"invalid retries directory: {retries_root}")
        for retry_dir in sorted(retries_root.iterdir()):
            if retry_dir.is_symlink() or not retry_dir.is_dir() \
                    or not RETRY_RE.fullmatch(retry_dir.name):
                raise ValueError(f"unexpected retry entry: {retry_dir}")
            request_dir = retry_dir / "requests"
            if request_dir.is_symlink() or not request_dir.is_dir():
                raise ValueError(f"missing retry request directory: {request_dir}")
            retry_files = sorted(request_dir.glob("*.jsonl"))
            if not retry_files or retry_files != sorted(request_dir.rglob("*.jsonl")):
                raise ValueError(f"unexpected retry request layout: {request_dir}")
            retry_keys: dict[str, set[str]] = {}
            seen_retry_keys: set[str] = set()
            for path in retry_files:
                metadata, keys, schema = _read_request_file(
                    path, root=root, expected=expected,
                    request_digests=request_digests)
                overlap = seen_retry_keys.intersection(keys)
                if overlap:
                    raise ValueError(
                        f"duplicate keys within {retry_dir.name}: {sorted(overlap)}")
                seen_retry_keys.update(keys)
                if schema != response_schema:
                    raise ValueError(f"retry response schema changed: {path}")
                retry_keys[path.stem] = set(keys)
                metadata.update({
                    "role": "retry", "retry": retry_dir.name,
                    "requests": metadata.pop("records"),
                })
                request_sources.append(metadata)
            retry_results = _discover_results(
                retry_dir / "sentences" / "results", set(retry_keys))
            for path in retry_files:
                sources.append((
                    "retry", retry_dir.name, path,
                    retry_results[path.stem], retry_keys[path.stem]))

    validator = response_validator(response_schema)
    selected: dict[str, dict[str, Any]] = {}
    invalid_attempts: list[dict[str, Any]] = []
    raw_sources: list[dict[str, Any]] = []
    retried_keys: set[str] = set()

    for role, retry, request_path, result_path, allowed_keys in sources:
        if retry is not None:
            retried_keys.update(allowed_keys)
        seen: set[str] = set()
        source_valid = source_invalid = 0

        def visit(line_number: int, value: Any) -> None:
            nonlocal source_valid, source_invalid
            if not isinstance(value, dict) or set(value) != RAW_KEYS:
                raise ValueError(
                    f"{result_path}:{line_number}: invalid Vertex record shape")
            key = value["key"]
            if not isinstance(key, str) or key not in allowed_keys:
                raise ValueError(
                    f"{result_path}:{line_number}: unmatched response key {key!r}")
            if key in seen:
                raise ValueError(f"{result_path}:{line_number}: duplicate key {key!r}")
            seen.add(key)
            if not isinstance(value["processed_time"], str) \
                    or not value["processed_time"]:
                raise ValueError(
                    f"{result_path}:{line_number}: invalid processed_time")
            echoed = value["request"]
            if not isinstance(echoed, dict) or artifact.sha256_json(
                    _drop_null_defaults(echoed)) != request_digests[key]:
                raise ValueError(
                    f"{result_path}:{line_number}: echoed request differs from source")
            status = value["status"]
            if not isinstance(status, str):
                raise ValueError(f"{result_path}:{line_number}: status must be a string")
            failure: tuple[str, str] | None = None
            response = value["response"]
            if status:
                failure = ("provider_error", status)
            elif not isinstance(response, dict):
                failure = ("schema_invalid", "response is not an object")
            elif response.get("modelVersion") != model:
                failure = (
                    "model_mismatch",
                    f"expected {model!r}, found {response.get('modelVersion')!r}",
                )
            else:
                try:
                    validator(key, response)
                except Exception as error:  # The shared validator owns failures.
                    failure = ("schema_invalid", str(error))
            if failure is not None:
                source_invalid += 1
                summary = {
                    "detail": failure[1], "key": key, "kind": failure[0],
                    "record": line_number,
                    "source": _relative(root, result_path),
                }
                if retry is not None:
                    summary["retry"] = retry
                invalid_attempts.append(summary)
                return
            if key in selected:
                raise ValueError(
                    f"duplicate valid responses for {key!r}: "
                    f"{selected[key]['response_path']} and "
                    f"{_relative(root, result_path)}")
            source_valid += 1
            selected[key] = {
                "request_path": _relative(root, request_path),
                "response": response,
                "response_path": _relative(root, result_path),
                "response_record": line_number,
                "retry": retry,
            }

        metadata = _scan_jsonl(result_path, visit)
        if seen != allowed_keys:
            raise ValueError(
                f"request/result keys differ for {result_path}: "
                f"missing={sorted(allowed_keys - seen)}, "
                f"unexpected={sorted(seen - allowed_keys)}")
        metadata.update({
            "invalid": source_invalid,
            "path": _relative(root, result_path),
            "request_path": _relative(root, request_path),
            "role": role,
            "valid": source_valid,
        })
        if retry is not None:
            metadata["retry"] = retry
        raw_sources.append(metadata)

    missing = [key for key in mapping_order if key not in selected]
    if missing:
        raise ValueError(
            f"no valid response for {len(missing)} expected keys: {missing[:10]}")
    canonical_bytes = b"".join(
        artifact.canonical_json_bytes({
            "key": key, "response": selected[key]["response"]}) + b"\n"
        for key in mapping_order
    )
    canonical_path = root / "responses" / "canonical.jsonl"
    artifact.atomic_write_file(canonical_path, canonical_bytes)
    selected_retry_lineage = [{
        "key": key,
        "request_path": selected[key]["request_path"],
        "response_path": selected[key]["response_path"],
        "response_record": selected[key]["response_record"],
        "retry": selected[key]["retry"],
    } for key in mapping_order if selected[key]["retry"] is not None]
    raw_records = sum(item["records"] for item in raw_sources)
    result = {
        "canonical": {
            "bytes": len(canonical_bytes),
            "path": _relative(root, canonical_path),
            "records": len(mapping_order),
            "sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        },
        "complete": True,
        "counts": {
            "canonical_records": len(mapping_order),
            "expected_keys": len(mapping_order),
            "invalid_attempts": len(invalid_attempts),
            "original_response_records": sum(
                item["records"] for item in raw_sources
                if item["role"] == "original"),
            "raw_response_files": len(raw_sources),
            "raw_response_records": raw_records,
            "retried_keys": len(retried_keys),
            "retry_response_records": sum(
                item["records"] for item in raw_sources
                if item["role"] == "retry"),
            "selected_original": len(mapping_order) - len(selected_retry_lineage),
            "selected_retry": len(selected_retry_lineage),
            "valid_attempts": raw_records - len(invalid_attempts),
        },
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": manifest["dataset"],
        "invalid_attempts": invalid_attempts,
        "mapping": mapping_metadata,
        "model": model,
        "raw_response_sources": raw_sources,
        "request_manifest": {
            "bytes": len(manifest_bytes),
            "path": "request_manifest.json",
            "request_set_sha256": manifest["request_set_sha256"],
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "request_sources": request_sources,
        "response_schema_sha256": schema_digest,
        "schema": SCHEMA,
        "selected_retry_lineage": selected_retry_lineage,
        "validation": {
            "checks": [
                "active mapping matches request manifest",
                "request shard hashes, sizes, counts, and request-set hash match",
                "original and retry request/result shards pair exactly",
                "provider request echoes differ only by null default fields",
                "selected responses use the manifest-bound model and schema",
                "exactly one valid response covers every mapping key in row order",
            ],
            "status": "PASS",
        },
        "version": manifest["version"],
    }
    artifact.atomic_write_json(root / "response_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        finalize(args.dataset_dir, args.artifact_root),
        sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
