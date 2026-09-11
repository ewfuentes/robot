#!/usr/bin/env python3
"""Download one (possibly stitched) sequence from a manifest, in order.

This is the only place that writes `sequence_position` into the sidecar JSON,
and mapillary_to_vigor.py sorts on it. That matters most for stitched
trajectories: each component sequence has its own positions starting at 0, so
they would collide, and the `captured_at` fallback cannot break ties because
Mapillary timestamps repeat within a second.

Downloads are assembled in a sibling `.incomplete` directory.  Existing pairs
are skipped only after their image bytes, sidecar schema, identity, and sequence
position have been validated.  A complete manifest is written last, then the
whole directory is published with one no-clobber rename.

Callable in-process by the collection orchestrator via `download_sequence`;
the CLI is a thin wrapper over it.
"""

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experimental.overhead_matching.swag.farfield import artifact, provenance
from experimental.overhead_matching.swag.farfield.collection.api import MapillaryClient
from experimental.overhead_matching.swag.farfield.collection import seed_to_trajectory
from experimental.overhead_matching.swag.farfield.geometry import haversine_m

DOWNLOAD_SCHEMA = "farfield_mapillary_download/v1"
MANIFEST_NAME = "manifest.json"
_GENERATOR = (
    "//experimental/overhead_matching/swag/farfield/"
    "collection:extract_stitch"
)
_MANIFEST_KEYS = frozenset({
    "schema",
    "generator",
    "git_commit",
    "created",
    "source_manifest",
    "config",
    "expected",
    "files",
    "content_digest",
    "complete",
})


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _finite(value, where: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{where} must be <= {maximum}")
    return result


def _image_stem(image: dict) -> str:
    image_id = artifact.require_identifier(image["id"], "image id")
    lat = _finite(image["lat"], f"image {image_id} lat",
                  minimum=-90.0, maximum=90.0)
    lng = _finite(image["lng"], f"image {image_id} lng",
                  minimum=-180.0, maximum=180.0)
    computed = image.get("computed_compass_angle")
    heading = (computed if computed is not None
               else image.get("compass_angle", 0.0))
    heading = _finite(heading, f"image {image_id} heading")
    captured_at = image.get("captured_at", 0)
    if (isinstance(captured_at, bool)
            or not isinstance(captured_at, int)
            or captured_at < 0):
        raise ValueError(
            f"image {image_id} captured_at must be a nonnegative integer")
    timestamp = captured_at // 1000 if captured_at else 0
    return (
        f"{image_id}_lat{lat:.6f}_lng{lng:.6f}_"
        f"heading{heading:.1f}_ts{timestamp}"
    )


def _load_request(manifest_path: Path, sequence: str,
                  min_spacing_m: float) -> tuple[list[dict], dict]:
    manifest_path = Path(manifest_path)
    source_digest = artifact.sha256_file(manifest_path)
    document = seed_to_trajectory.validate_sequence_manifest(
        manifest_path, expected_sequence_id=sequence)
    selected = document["sequences"][0]
    images = selected["images"]
    normalized = []
    seen_ids = set()
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            raise ValueError(
                f"{manifest_path}: image {index} must be an object")
        stem = _image_stem(image)
        image_id = image["id"]
        if image_id in seen_ids:
            raise ValueError(
                f"{manifest_path}: duplicate image id {image_id!r}")
        seen_ids.add(image_id)
        normalized.append({**image, "_stem": stem})
    images = decimate_by_spacing(normalized, min_spacing_m)
    expected = [
        {
            "id": image["id"],
            "sequence_position": position,
            "stem": image["_stem"],
        }
        for position, image in enumerate(images)
    ]
    if len({item["stem"] for item in expected}) != len(expected):
        raise ValueError(
            f"{manifest_path}: image filenames collide after normalization")
    if artifact.sha256_file(manifest_path) != source_digest:
        raise RuntimeError(
            f"source manifest changed while it was read: {manifest_path}")
    return images, {
        "path": str(manifest_path.resolve()),
        "sha256": source_digest,
    }


def _expected_files(expected: list[dict]) -> set[str]:
    files = {MANIFEST_NAME}
    for item in expected:
        files.add(f"{item['stem']}.jpg")
        files.add(f"{item['stem']}.json")
    return files


def _validate_pair(root: Path, image: dict, expected: dict) -> dict:
    stem = expected["stem"]
    jpg_path = root / f"{stem}.jpg"
    json_path = root / f"{stem}.json"
    for path in (jpg_path, json_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular download file: {path}")
    blob = jpg_path.read_bytes()
    dimensions = jpeg_dimensions(blob)
    if not looks_like_complete_jpeg(blob) or dimensions is None:
        raise ValueError(f"invalid JPEG for image {expected['id']}: {jpg_path}")
    record = _load_json(json_path)
    if not isinstance(record, dict):
        raise ValueError(f"{json_path}: frame sidecar must be an object")
    if record.get("id") != expected["id"]:
        raise ValueError(
            f"{json_path}: image id {record.get('id')!r} does not match "
            f"{expected['id']!r}")
    if record.get("sequence_position") != expected["sequence_position"]:
        raise ValueError(
            f"{json_path}: sequence_position "
            f"{record.get('sequence_position')!r} does not match "
            f"{expected['sequence_position']}")
    if (record.get("width"), record.get("height")) != dimensions:
        raise ValueError(
            f"{json_path}: recorded dimensions do not match JPEG "
            f"{dimensions[0]}x{dimensions[1]}")
    source_url_kind = record.get("source_url_kind")
    if source_url_kind not in ("original", "thumb_2048"):
        raise ValueError(
            f"{json_path}: source_url_kind must be original or thumb_2048")
    expected_record = {
        key: value for key, value in image.items()
        if key != "_stem"
    }
    source_dimensions = (image["width"], image["height"])
    expected_record["sequence_position"] = expected["sequence_position"]
    expected_record["source_url_kind"] = source_url_kind
    if dimensions != source_dimensions:
        expected_record["api_width"], expected_record["api_height"] = (
            source_dimensions)
    expected_record["width"], expected_record["height"] = dimensions
    if record != expected_record:
        changed = sorted(
            key for key in set(record) | set(expected_record)
            if record.get(key) != expected_record.get(key)
            or (key in record) != (key in expected_record))
        raise ValueError(
            f"{json_path}: sidecar does not exactly match source manifest; "
            f"changed fields={changed}")
    return {
        "id": expected["id"],
        "sequence_position": expected["sequence_position"],
        "stem": stem,
        "width": dimensions[0],
        "height": dimensions[1],
        "jpg_sha256": artifact.sha256_file(jpg_path),
        "json_sha256": artifact.sha256_file(json_path),
    }


def _validate_complete(
        root: Path, source_manifest: dict, config: dict,
        images: list[dict], expected: list[dict]) -> dict:
    manifest_path = root / MANIFEST_NAME
    document = _load_json(manifest_path)
    if not isinstance(document, dict):
        raise ValueError(f"{manifest_path}: manifest must be an object")
    missing = sorted(_MANIFEST_KEYS - set(document))
    unknown = sorted(set(document) - _MANIFEST_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{manifest_path}: manifest missing={missing}, unknown={unknown}")
    if document["schema"] != DOWNLOAD_SCHEMA or document["complete"] is not True:
        raise ValueError(f"{manifest_path}: incomplete or unsupported download")
    if document["generator"] != _GENERATOR:
        raise ValueError(f"{manifest_path}: unexpected generator")
    for field in ("git_commit", "created"):
        if not isinstance(document[field], str) or not document[field]:
            raise ValueError(f"{manifest_path}: {field} must be non-empty")
    if document["source_manifest"] != source_manifest:
        raise ValueError(f"{manifest_path}: source manifest identity changed")
    if document["config"] != config:
        raise ValueError(f"{manifest_path}: download configuration changed")
    if document["expected"] != expected:
        raise ValueError(f"{manifest_path}: expected image set changed")
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"{root}: download contains non-regular entry {entry.name!r}")
    actual_names = {entry.name for entry in entries}
    expected_names = _expected_files(expected)
    if actual_names != expected_names:
        raise ValueError(
            f"{root}: download files missing={sorted(expected_names-actual_names)}, "
            f"extra={sorted(actual_names-expected_names)}")
    files = [
        _validate_pair(root, image, expected_item)
        for image, expected_item in zip(images, expected, strict=True)
    ]
    if document["files"] != files:
        raise ValueError(f"{manifest_path}: recorded file identities changed")
    digest = artifact.sha256_directory(root, exclude=(MANIFEST_NAME,))
    if document["content_digest"] != digest:
        raise ValueError(f"{manifest_path}: content digest mismatch")
    return document


def validate_download_directory(root: Path | str) -> dict:
    """Validate a completed stage-2 download against its exact stage-1 input."""
    root = Path(root)
    if any(part.endswith(artifact.INCOMPLETE_SUFFIX) for part in root.parts):
        raise ValueError(f"incomplete download cannot be consumed: {root}")
    manifest_path = root / MANIFEST_NAME
    document = _load_json(manifest_path)
    if not isinstance(document, dict):
        raise ValueError(f"{manifest_path}: manifest must be an object")
    missing = sorted(_MANIFEST_KEYS - set(document))
    unknown = sorted(set(document) - _MANIFEST_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{manifest_path}: manifest missing={missing}, unknown={unknown}")
    source_manifest = document["source_manifest"]
    if (not isinstance(source_manifest, dict)
            or set(source_manifest) != {"path", "sha256"}
            or not isinstance(source_manifest["path"], str)
            or not source_manifest["path"]
            or not isinstance(source_manifest["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}",
                                source_manifest["sha256"])):
        raise ValueError(f"{manifest_path}: invalid source manifest identity")
    config = document["config"]
    if (not isinstance(config, dict)
            or set(config) != {"sequence", "max_width", "min_spacing_m"}):
        raise ValueError(f"{manifest_path}: invalid download configuration")
    if not isinstance(config["sequence"], str) or not config["sequence"]:
        raise ValueError(f"{manifest_path}: invalid sequence identity")
    max_width = config["max_width"]
    if (max_width is not None
            and (isinstance(max_width, bool)
                 or not isinstance(max_width, int) or max_width <= 0)):
        raise ValueError(f"{manifest_path}: invalid max_width")
    min_spacing_m = _finite(
        config["min_spacing_m"], "min_spacing_m", minimum=0.0)
    source_path = Path(source_manifest["path"])
    images, current_source = _load_request(
        source_path, config["sequence"], min_spacing_m)
    if current_source != source_manifest:
        raise ValueError(f"{manifest_path}: source manifest identity changed")
    expected = [
        {
            "id": image["id"],
            "sequence_position": position,
            "stem": image["_stem"],
        }
        for position, image in enumerate(images)
    ]
    return _validate_complete(
        root, current_source, config, images, expected)

def decimate_by_spacing(images: list[dict], min_spacing_m: float) -> list[dict]:
    """Keep images at least min_spacing_m apart along the track.

    Applied before download so we never pay bandwidth or disk for frames we
    would discard later. Useful because several of these captures are video
    extractions at 3-30 fps where consecutive frames share a GPS fix and show
    essentially the same view.
    """
    if min_spacing_m <= 0 or not images:
        return images
    kept = [images[0]]
    last = images[0]
    for img in images[1:]:
        d = haversine_m(last["lat"], last["lng"], img["lat"], img["lng"])
        if d >= min_spacing_m:
            kept.append(img)
            last = img
    return kept


def looks_like_complete_jpeg(data: bytes) -> bool:
    """SOI/EOI marker check, so a truncated download fails loudly here.

    Without this a short read is written to disk and only surfaces much later as
    a decode error inside the converter or the VLM stage.
    """
    return len(data) > 1024 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"


def jpeg_dimensions(data: bytes):
    """Fully decode an in-memory JPEG and return its dimensions, or None."""
    try:
        from io import BytesIO

        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            dimensions = im.size
            im.verify()
        # ``verify`` checks the stream structure without decoding pixels.
        # Reopen and load so corrupt scan data cannot be reused as complete.
        with Image.open(BytesIO(data)) as im:
            im.load()
            if im.size != dimensions:
                return None
        return dimensions
    except Exception:
        return None


def download_sequence(manifest_path: Path, sequence: str, out_dir: Path,
                      workers: int, max_width: int | None,
                      min_spacing_m: float, dry_run: bool = False) -> bool:
    """Download and transactionally publish one exact manifest sequence."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if (max_width is not None
            and (isinstance(max_width, bool)
                 or not isinstance(max_width, int)
                 or max_width <= 0)):
        raise ValueError("max_width must be a positive integer or null")
    min_spacing_m = _finite(
        min_spacing_m, "min_spacing_m", minimum=0.0)
    images, source_manifest = _load_request(
        manifest_path, sequence, min_spacing_m)
    expected = [
        {
            "id": image["id"],
            "sequence_position": position,
            "stem": image["_stem"],
        }
        for position, image in enumerate(images)
    ]
    config = {
        "sequence": sequence,
        "max_width": max_width,
        "min_spacing_m": min_spacing_m,
    }
    out_dir = Path(out_dir)
    incomplete = out_dir.with_name(
        f"{out_dir.name}{artifact.INCOMPLETE_SUFFIX}")
    print(f"Sequence: {sequence}")
    print(f"Images:   {len(images)}")
    if dry_run:
        print(f"[DRY RUN] would download {len(images)} images to {out_dir} "
              f"(max_width={max_width})")
        return True

    if out_dir.exists() or out_dir.is_symlink():
        if incomplete.exists() or incomplete.is_symlink():
            raise FileExistsError(
                f"both completed and incomplete downloads exist: "
                f"{out_dir}, {incomplete}")
        _validate_complete(
            out_dir, source_manifest, config, images, expected)
        print(f"Validated completed download: {out_dir}")
        return True
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if incomplete.is_symlink():
        raise ValueError(
            f"incomplete download cannot be a symlink: {incomplete}")
    if incomplete.exists() and not incomplete.is_dir():
        raise ValueError(
            f"incomplete download is not a directory: {incomplete}")
    incomplete.mkdir(exist_ok=True)

    # A complete staging manifest can remain if the process stopped after
    # validation but before publication.  Validate it byte-for-byte and finish
    # the rename without contacting Mapillary.
    if (incomplete / MANIFEST_NAME).exists():
        _validate_complete(
            incomplete, source_manifest, config, images, expected)
        artifact.publish_directory_no_clobber(incomplete, out_dir)
        return True

    expected_without_manifest = _expected_files(expected) - {MANIFEST_NAME}
    actual = list(incomplete.iterdir())
    for entry in actual:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                f"incomplete download contains non-regular entry "
                f"{entry.name!r}")
        if entry.name not in expected_without_manifest:
            raise ValueError(
                f"incomplete download contains stale or unexpected file "
                f"{entry.name!r}")

    client = MapillaryClient()
    downloaded = failed = skipped = 0
    start = time.time()
    failures = []

    def download_one(seq_pos, image, expected_item):
        image_id = image["id"]
        stem = expected_item["stem"]
        jpg_path = incomplete / f"{stem}.jpg"
        json_path = incomplete / f"{stem}.json"
        jpg_exists = jpg_path.exists() or jpg_path.is_symlink()
        json_exists = json_path.exists() or json_path.is_symlink()
        if jpg_exists and json_exists:
            _validate_pair(incomplete, image, expected_item)
            return "skipped"
        if jpg_exists != json_exists:
            # A process can stop between the two atomic writes.  The unmatched
            # file is not a valid pair and is safe to replace inside the
            # explicitly incomplete workspace.
            (jpg_path if jpg_exists else json_path).unlink()

        url = client.get_image_url(image_id, max_width=max_width)
        if not url:
            raise RuntimeError(f"no download URL for {image_id}")
        blob = client.download_image(url)
        dimensions = jpeg_dimensions(blob)
        if not looks_like_complete_jpeg(blob) or dimensions is None:
            fallback = client.get_image_url(image_id, max_width=2048)
            if fallback and fallback != url:
                candidate = client.download_image(fallback)
                candidate_dimensions = jpeg_dimensions(candidate)
                if (looks_like_complete_jpeg(candidate)
                        and candidate_dimensions is not None):
                    blob, url, dimensions = (
                        candidate, fallback, candidate_dimensions)
        if not looks_like_complete_jpeg(blob) or dimensions is None:
            raise RuntimeError(
                f"invalid JPEG for {image_id} ({len(blob)} bytes)")
        record = {
            key: value for key, value in image.items()
            if key != "_stem"
        }
        record["sequence_position"] = seq_pos
        record["source_url_kind"] = "thumb_2048" if "2048" in url else "original"
        if dimensions != (image.get("width"), image.get("height")):
            record["api_width"] = image.get("width")
            record["api_height"] = image.get("height")
        record["width"], record["height"] = dimensions
        artifact.atomic_write_file(jpg_path, blob)
        artifact.atomic_write_json(json_path, record)
        _validate_pair(incomplete, image, expected_item)
        return "ok"

    print(f"\nDownloading {len(images)} images with {workers} workers "
          f"(max_width={max_width})...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, position, image, expected_item): image
            for position, (image, expected_item) in enumerate(
                zip(images, expected, strict=True))
        }
        for future in as_completed(futures):
            try:
                if future.result() == "skipped":
                    skipped += 1
                else:
                    downloaded += 1
            except Exception as e:
                failed += 1
                failures.append((futures[future]["id"], str(e)))
            total = downloaded + failed + skipped
            if total % 100 == 0 or total == len(images):
                el = time.time() - start
                rate = (downloaded + skipped) / el if el > 0 else 0
                print(f"  [{total}/{len(images)}] downloaded={downloaded} "
                      f"skipped={skipped} failed={failed} ({rate:.1f}/s)")

    print(f"\nDone in {time.time()-start:.1f}s. {downloaded} downloaded, "
          f"{skipped} skipped, {failed} failed.")
    if failures:
        print(f"\n{len(failures)} failure(s); first 10:")
        for img_id, err in failures[:10]:
            print(f"  {img_id}: {err}")
        print(f"Re-run to retry validated pairs in {incomplete}.")
        return False

    files = [
        _validate_pair(incomplete, image, expected_item)
        for image, expected_item in zip(images, expected, strict=True)
    ]
    document = {
        "schema": DOWNLOAD_SCHEMA,
        "generator": _GENERATOR,
        "git_commit": provenance.git_commit(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": source_manifest,
        "config": config,
        "expected": expected,
        "files": files,
        "content_digest": artifact.sha256_directory(incomplete),
        "complete": True,
    }
    artifact.atomic_write_json(incomplete / MANIFEST_NAME, document)
    _validate_complete(
        incomplete, source_manifest, config, images, expected)
    artifact.publish_directory_no_clobber(incomplete, out_dir)
    print(f"Published complete download: {out_dir}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Path to a manifest JSON (from seed_to_trajectory)")
    parser.add_argument("--sequence", required=True,
                        help="Sequence id within the manifest (e.g. a trajectory name)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for images")
    parser.add_argument("--workers", type=int, default=8, help="Download workers")
    parser.add_argument("--max_width", type=int, required=True,
                        help="Cap on stored width; fetch the 2048 thumbnail instead "
                             "of a much larger original when that still satisfies "
                             "the cap. 0 disables the cap.")
    parser.add_argument("--min_spacing_m", type=float, required=True,
                        help="Drop frames closer together than this along the "
                             "track, before downloading.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be downloaded and exit")
    args = parser.parse_args(argv)

    try:
        ok = download_sequence(args.manifest, args.sequence, args.output,
                               workers=args.workers,
                               max_width=args.max_width or None,
                               min_spacing_m=args.min_spacing_m,
                               dry_run=args.dry_run)
    except (artifact.ArtifactError, FileExistsError, OSError,
            RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
