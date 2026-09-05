#!/usr/bin/env python3
"""Audit and describe a prepared LOCI panorama VLM request bundle."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image


SCHEMA = "farfield.loci_vlm_request_manifest.v1"
YAWS = (0, 90, 180, 270)
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_ULTRA_HIGH"
THINKING_LEVEL = "HIGH"

# These pin the exact request text/schema used by the production
# panov2_tuned_prompt extraction, rather than merely accepting current code.
SYSTEM_PROMPT_SHA256 = (
    "ce5cbda006a12123b76e765aff3bb3930109886bc8fbcb0866969bb8ce3d4a1c")
USER_PROMPT_SHA256 = (
    "c190ff660bff6c29706f87ac92bf0431065518853ff881b32f35e06e18d435e5")
RESPONSE_SCHEMA_SHA256 = (
    "ff2fed09280944c29af842e0de392be7b43311c9248a63d7bf3a35feaf8dd56a")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _workspace() -> Path:
    configured = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if configured:
        return Path(configured).resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    raise ValueError("cannot locate git workspace")


def _git_commit(workspace: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def _expected_keys(dataset_dir: Path) -> list[str]:
    mapping_path = dataset_dir / "pano_id_mapping.csv"
    with mapping_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    keys = [Path(row["filename"]).stem for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate panorama keys in {mapping_path}")
    for row, key in zip(rows, keys, strict=True):
        if key.split(",", 1)[0] != row["pano_id"]:
            raise ValueError(
                f"mapping pano_id does not match filename stem: {row!r}")
    return sorted(keys)


def _world_heading_authority(dataset_dir: Path) -> str:
    intrinsics_path = dataset_dir / "intrinsics.csv"
    heading_fields = (
        "computed_compass_angle_true_deg",
        "compass_angle_true_deg",
        "heading_optical_axis_true_deg",
        "heading_column0_true_deg",
        "selected_heading_source",
    )
    with intrinsics_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if any(row.get(field, "").strip() for row in rows for field in heading_fields):
        return "present_in_active_intrinsics"
    return "none"


def audit(dataset_dir: Path, pinhole_dir: Path, request_dir: Path, *,
          model: str, generator_disable_tqdm: bool = False) -> dict:
    """Fully validate source faces and all serialized requests."""
    if not isinstance(model, str) or not model.strip() or model != model.strip():
        raise ValueError("model must be a non-empty string without outer whitespace")
    dataset_dir = dataset_dir.resolve()
    pinhole_dir = pinhole_dir.resolve()
    request_dir = request_dir.resolve()
    expected_keys = _expected_keys(dataset_dir)
    actual_keys = sorted(
        path.name for path in pinhole_dir.iterdir() if path.is_dir())
    if actual_keys != expected_keys:
        raise ValueError(
            "pinhole directories differ from pano_id_mapping.csv")
    shards = sorted(request_dir.glob("*.jsonl"))
    if not shards:
        raise ValueError(f"no request shards in {request_dir}")

    pinhole_manifest_path = pinhole_dir / "manifest.json"
    pinhole_manifest = json.loads(pinhole_manifest_path.read_text())
    if pinhole_manifest.get("complete") is not True:
        raise ValueError("pinhole artifact is not complete")
    if pinhole_manifest.get("dataset") != dataset_dir.name:
        raise ValueError("pinhole artifact belongs to a different dataset")
    geometry = pinhole_manifest.get("config", {}).get("geometry", {})
    if geometry.get("resolution_x") != 2048 \
            or geometry.get("resolution_y") != 2048:
        raise ValueError("pinhole manifest does not declare 2048x2048 faces")

    seen: list[str] = []
    decoded_image_bytes = 0
    system_prompt_bytes = None
    user_prompt_bytes = None
    response_schema_bytes = None
    shard_request_counts: dict[Path, int] = {}
    for shard in shards:
        shard_request_counts[shard] = 0
        with shard.open() as source:
            for line_number, line in enumerate(source, 1):
                try:
                    item = json.loads(line)
                    key = item["key"]
                    request = item["request"]
                    content = request["contents"]
                    parts = content[0]["parts"]
                    generation = request["generationConfig"]
                except (KeyError, IndexError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid request at {shard}:{line_number}") from error
                if key in seen:
                    raise ValueError(f"duplicate request key: {key}")
                seen.append(key)
                shard_request_counts[shard] += 1
                if len(content) != 1 or content[0].get("role") != "user" \
                        or len(parts) != 5:
                    raise ValueError(f"invalid content structure for {key}")
                system_text = request["systemInstruction"]["parts"][0]["text"]
                current_system_bytes = system_text.encode("utf-8")
                if _sha256_bytes(current_system_bytes) != SYSTEM_PROMPT_SHA256:
                    raise ValueError(f"wrong system prompt for {key}")
                current_user_bytes = parts[-1]["text"].encode("utf-8")
                if _sha256_bytes(current_user_bytes) \
                        != USER_PROMPT_SHA256:
                    raise ValueError(f"wrong user prompt for {key}")
                current_schema_bytes = _canonical_json_bytes(
                    generation.get("responseSchema"))
                if generation.get("responseMimeType") != "application/json" \
                        or _sha256_bytes(current_schema_bytes) \
                        != RESPONSE_SCHEMA_SHA256:
                    raise ValueError(f"wrong response schema for {key}")
                system_prompt_bytes = current_system_bytes
                user_prompt_bytes = current_user_bytes
                response_schema_bytes = current_schema_bytes
                if generation.get("thinkingConfig") \
                        != {"thinkingLevel": THINKING_LEVEL}:
                    raise ValueError(f"wrong thinking level for {key}")
                if "mediaResolution" in generation:
                    raise ValueError(f"global mediaResolution present for {key}")

                for yaw, part in zip(YAWS, parts[:4], strict=True):
                    if part.get("media_resolution") \
                            != {"level": MEDIA_RESOLUTION}:
                        raise ValueError(
                            f"wrong per-image media resolution for {key}/{yaw}")
                    inline = part.get("inline_data", {})
                    if inline.get("mime_type") != "image/jpeg":
                        raise ValueError(f"wrong image MIME type for {key}/{yaw}")
                    try:
                        embedded = base64.b64decode(
                            inline["data"], validate=True)
                    except (KeyError, ValueError) as error:
                        raise ValueError(
                            f"invalid base64 image for {key}/{yaw}") from error
                    source_path = pinhole_dir / key / f"yaw_{yaw:03d}.jpg"
                    expected_names = {
                        f"yaw_{expected_yaw:03d}.jpg" for expected_yaw in YAWS
                    }
                    actual_names = {
                        path.name for path in source_path.parent.iterdir()
                        if path.is_file()
                    }
                    if actual_names != expected_names:
                        raise ValueError(
                            f"invalid face file set for {source_path.parent}")
                    source_bytes = source_path.read_bytes()
                    if embedded != source_bytes:
                        raise ValueError(
                            f"embedded image differs from source: {key}/{yaw}")
                    with Image.open(source_path) as image:
                        if image.size != (2048, 2048) \
                                or image.format != "JPEG":
                            raise ValueError(
                                f"invalid source image: {source_path}")
                    decoded_image_bytes += len(embedded)

    if seen != expected_keys:
        missing = sorted(set(expected_keys) - set(seen))
        unexpected = sorted(set(seen) - set(expected_keys))
        raise ValueError(
            "request keys differ from pano_id_mapping.csv: "
            f"missing={missing[:10]!r}, unexpected={unexpected[:10]!r}")

    artifact_root = request_dir.parent.parent
    shard_records = [{
        "path": shard.relative_to(artifact_root).as_posix(),
        "requests": shard_request_counts[shard],
        "bytes": shard.stat().st_size,
        "sha256": _sha256_file(shard),
    } for shard in shards]
    request_set_sha256 = _canonical_sha256(shard_records)
    workspace = _workspace()
    extractor_relative = Path(
        "experimental/overhead_matching/swag/model/"
        "semantic_landmark_extractor.py")
    extractor_path = workspace / extractor_relative
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    world_heading_authority = _world_heading_authority(dataset_dir)
    orientation_reason = (
        "No authoritative per-frame world heading exists for the active "
        "panoramas/adopted faces."
        if world_heading_authority == "none" else
        "The pinhole renderer uses fixed panorama-relative yaw offsets and "
        "does not apply the active intrinsics heading fields."
    )
    return {
        "artifact_version_root": str(artifact_root),
        "complete": True,
        "counts": {
            "jpeg_parts": len(seen) * len(YAWS),
            "jpeg_parts_per_request": len(YAWS),
            "pano_id_mapping_rows": len(expected_keys),
            "parts_per_request": len(YAWS) + 1,
            "request_jsonl_bytes": sum(
                record["bytes"] for record in shard_records),
            "request_keys": len(seen),
            "shards": len(shard_records),
            "source_face_height_px": 2048,
            "source_face_width_px": 2048,
            "source_jpeg_bytes": decoded_image_bytes,
            "source_jpeg_files": len(seen) * len(YAWS),
            "unique_request_keys": len(set(seen)),
        },
        "created": timestamp,
        "dataset": dataset_dir.name,
        "generator": {
            "bazel_target": (
                "//experimental/overhead_matching/swag/model:"
                "semantic_landmark_extractor"),
            "git_commit": _git_commit(workspace),
            "parameters": {
                "disable_tqdm": generator_disable_tqdm,
                "media_resolution": MEDIA_RESOLUTION,
                "num_workers": 8,
                "prompt_type": "osm_tags",
                "thinking_level": THINKING_LEVEL,
            },
            "source_path": extractor_relative.as_posix(),
            "source_sha256": _sha256_file(extractor_path),
            "subcommand": "create_panorama_sentences",
        },
        "intended_inference": {
            "global_media_resolution_present": False,
            "media_resolution": MEDIA_RESOLUTION,
            "media_resolution_placement": "per_image_part",
            "model": model,
            "prompt_type": "osm_tags",
            "provider_interface": "Vertex AI Gemini batch",
            "response_mime_type": "application/json",
            "submitted": False,
            "thinking_level": THINKING_LEVEL,
            "uploaded": False,
        },
        "prompt_contract": {
            "response_schema_canonical_bytes": len(response_schema_bytes),
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "response_schema_canonical_sha256": RESPONSE_SCHEMA_SHA256,
            "response_schema_canonicalization": (
                "UTF-8 JSON; sort_keys=true; separators=(comma,colon); "
                "ensure_ascii=false"),
            "system_prompt_utf8_bytes": len(system_prompt_bytes),
            "user_prompt_sha256": USER_PROMPT_SHA256,
            "user_prompt_utf8_bytes": len(user_prompt_bytes),
        },
        "request_set_sha256": request_set_sha256,
        "schema": SCHEMA,
        "scope_and_orientation": {
            "limitation": (
                "Do not interpret face yaw labels or returned yaw metadata "
                "as world bearings."),
            "orientation": "camera_as_captured",
            "reason": orientation_reason,
            "scope": "LOCI OSM-tag extraction and late fusion only",
            "world_heading_authority": world_heading_authority,
            "yaw_labels": (
                "camera_relative_despite_stock_prompt_compass_language"),
            "yaw_use": "ignored by LOCI tag extraction/late fusion",
        },
        "sentence_requests_root": str(request_dir.parent),
        "shards": shard_records,
        "upstream": {
            "pano_id_mapping_path": str(
                dataset_dir / "pano_id_mapping.csv"),
            "pano_id_mapping_sha256": _sha256_file(
                dataset_dir / "pano_id_mapping.csv"),
            "pinhole_content_digest": pinhole_manifest.get("content_digest"),
            "pinhole_dir": str(pinhole_dir),
            "pinhole_manifest_path": str(pinhole_manifest_path),
            "pinhole_manifest_sha256": _sha256_file(
                pinhole_manifest_path),
            "pinhole_version": pinhole_manifest.get("version"),
        },
        "validation": {
            "checks": [
                "mapping rows equal request count",
                ("request keys are unique and exactly match "
                 "pano_id_mapping.csv filename stems"),
                ("source pinhole directories and four-face file sets exactly "
                 "match mapping"),
                "all source faces are JPEG 2048x2048",
                "exactly five parts per request and four JPEG image parts",
                ("every embedded JPEG byte-matches "
                 "yaw_000/090/180/270 source"),
                "ULTRA_HIGH is present on every image part",
                "generationConfig has no global mediaResolution",
                "thinking level is HIGH",
                ("system prompt, user prompt, and response schema exactly "
                 "match current osm_tags extractor values"),
            ],
            "status": "PASS",
            "validated_at": timestamp,
        },
        "version": artifact_root.name,
    }


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False) as output:
        temp_path = Path(output.name)
        json.dump(value, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--pinhole_dir", type=Path, required=True)
    parser.add_argument("--request_dir", type=Path, required=True)
    parser.add_argument(
        "--model", required=True,
        help="Exact model ID intended for the external batch submission",
    )
    parser.add_argument("--output_manifest", type=Path)
    parser.add_argument("--generator_disable_tqdm", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.dataset_dir, args.pinhole_dir, args.request_dir,
        model=args.model,
        generator_disable_tqdm=args.generator_disable_tqdm)
    if args.output_manifest is not None:
        _atomic_write_json(args.output_manifest, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
