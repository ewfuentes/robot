"""Resumable mask-shaped person anonymization for panoramic video.

This is the production companion to ``person_segmentation_preview``.  The
expensive ``scan`` stage stores immutable per-frame YOLO person-mask evidence
and can resume after interruption without rerunning committed frames.  Later
commands apply conservative temporal persistence and render a separate video;
the source video is always read-only.

The scanner deliberately binds every decision that affects inference: source
and model hashes, exact output-grid clip, preprocessing resolution, confidence
thresholds, native/half-roll passes, and the implementation hash.  A resume
with a different specification fails closed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from experimental.overhead_matching.swag.farfield.dataset_tools import anonymize_video
from experimental.overhead_matching.swag.farfield.dataset_tools import person_mask_persistence
from experimental.overhead_matching.swag.farfield.dataset_tools import person_segmentation_preview


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_FPS = 3.0
DEFAULT_SCAN_WIDTH = 1920
DEFAULT_IMGSZ = 1920
DEFAULT_CANDIDATE_CONFIDENCE = 0.05
DEFAULT_DIRECT_CONFIDENCE = 0.15
_EVIDENCE_PATTERN = re.compile(r"frame_(\d{8})\.npz")
_WORKER_DETECTOR = None
_WORKER_DIRECT_CONFIDENCE = None


def _utc_now() -> str:
    return datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_new_atomic(path: Path, payload: bytes):
    """Publish a new file atomically, refusing to replace any final file."""
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace output: {path}")
    temporary.unlink(missing_ok=True)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate(path: Path, payload: bytes):
    """Make finalization restartable without ever replacing a published file."""
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected a regular finalized file: {path}")
        if path.read_bytes() != payload:
            raise ValueError(f"existing finalized content differs: {path}")
        return
    _write_new_atomic(path, payload)


def _regular_directory(path: Path, description: str):
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{description} must be a real directory: {path}")


def _verify_regular_sha256(path: Path, expected: str, description: str):
    if (path.is_symlink() or not path.is_file()
            or anonymize_video.sha256_file(path) != expected):
        raise ValueError(f"{description} is missing or changed: {path}")


@contextlib.contextmanager
def _locked_named_stage(output_dir: Path, spec: dict, *, spec_name: str,
                        lock_name: str):
    """Claim or resume ``<output>.incomplete`` under an exclusive lock."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"completed output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(output_dir.name + ".incomplete")
    try:
        stage.mkdir()
    except FileExistsError:
        _regular_directory(stage, "resumable staging path")
    lock_path = stage / lock_name
    lock_stream = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another scan owns {stage}") from error
        spec_path = stage / spec_name
        if spec_path.exists() or spec_path.is_symlink():
            if spec_path.is_symlink() or not spec_path.is_file():
                raise ValueError(f"invalid scan specification: {spec_path}")
            existing = json.loads(spec_path.read_text())
            if existing != spec:
                raise ValueError(
                    "refusing to resume with a changed scan specification\n"
                    f"existing: {json.dumps(existing, sort_keys=True)}\n"
                    f"requested: {json.dumps(spec, sort_keys=True)}")
        else:
            _write_new_atomic(spec_path, _json_bytes(spec))
        yield stage, lock_path
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


@contextlib.contextmanager
def _locked_stage(output_dir: Path, spec: dict):
    with _locked_named_stage(
            output_dir, spec, spec_name="scan_spec.json",
            lock_name=".scan.lock") as claimed:
        yield claimed


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(
        np.ascontiguousarray(mask, dtype=np.uint8).reshape(-1),
        bitorder="little")


def _unpack_mask(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    expected_bytes = math.ceil(shape[0] * shape[1] / 8)
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1 or len(bits) != expected_bytes:
        raise ValueError(
            f"packed mask has shape {bits.shape}; expected ({expected_bytes},)")
    return np.unpackbits(
        bits, count=shape[0] * shape[1], bitorder="little").reshape(shape) != 0


def write_scan_evidence(path: Path, *, frame_index: int,
                        source_frame_index: int, direct_mask: np.ndarray,
                        weak_mask: np.ndarray, instances: list[dict],
                        vehicle_mask: np.ndarray | None = None):
    """Atomically commit one self-describing detector result."""
    if direct_mask.shape != weak_mask.shape or direct_mask.ndim != 2:
        raise ValueError("direct and weak masks must have one identical HxW shape")
    if vehicle_mask is None:
        vehicle_mask = np.zeros_like(direct_mask)
    if vehicle_mask.shape != direct_mask.shape:
        raise ValueError("vehicle context mask must match person masks")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace scan evidence: {path}")
    temporary = path.with_name(path.name + ".incomplete")
    temporary.unlink(missing_ok=True)
    metadata = [{
        "confidence": round(float(item["confidence"]), 8),
        "passes": sorted(set(item.get("passes", []))),
        "class_id": int(item.get("class_id", 0)),
        "area_pixels": int(item.get(
            "area_pixels", np.count_nonzero(item.get("mask", False)))),
    } for item in instances]
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
            frame_index=np.asarray(frame_index, dtype=np.int64),
            source_frame_index=np.asarray(source_frame_index, dtype=np.int64),
            mask_height=np.asarray(direct_mask.shape[0], dtype=np.int32),
            mask_width=np.asarray(direct_mask.shape[1], dtype=np.int32),
            direct_bits=_pack_mask(direct_mask),
            weak_bits=_pack_mask(weak_mask),
            vehicle_bits=_pack_mask(vehicle_mask),
            metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
        )
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace scan evidence: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def load_scan_evidence(path: Path, expected_shape: tuple[int, int] | None = None,
                       *, masks: bool = True) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"scan evidence must be a regular file: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            schema_version = int(payload["schema_version"])
            frame_index = int(payload["frame_index"])
            source_frame_index = int(payload["source_frame_index"])
            shape = (int(payload["mask_height"]), int(payload["mask_width"]))
            metadata = json.loads(str(payload["metadata_json"]))
            if schema_version != SCHEMA_VERSION:
                raise ValueError(f"unsupported evidence schema {schema_version}")
            if expected_shape is not None and shape != expected_shape:
                raise ValueError(
                    f"scan evidence shape is {shape}, expected {expected_shape}")
            result = {
                "frame_index": frame_index,
                "source_frame_index": source_frame_index,
                "shape": shape,
                "instances": metadata,
            }
            if masks:
                result.update({
                    "direct_mask": _unpack_mask(payload["direct_bits"], shape),
                    "weak_mask": _unpack_mask(payload["weak_bits"], shape),
                    "vehicle_mask": _unpack_mask(
                        payload["vehicle_bits"], shape),
                })
            else:
                # Validate packed lengths even when a caller only needs metadata.
                _unpack_mask(payload["direct_bits"], shape)
                _unpack_mask(payload["weak_bits"], shape)
                _unpack_mask(payload["vehicle_bits"], shape)
            return result
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid scan evidence {path}: {error}") from error


def _evidence_name(frame_index: int) -> str:
    return f"frame_{frame_index:08d}.npz"


def committed_scan_indices(frames_dir: Path, frame_count: int,
                           shape: tuple[int, int], source_start: int) -> set[int]:
    """Validate all committed evidence and return its clip-local indexes."""
    frames_dir.mkdir(exist_ok=True)
    _regular_directory(frames_dir, "scan evidence path")
    indexes = set()
    for path in frames_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unexpected scan evidence entry: {path}")
        if path.name.endswith(".npz.incomplete"):
            match = _EVIDENCE_PATTERN.fullmatch(
                path.name.removesuffix(".incomplete"))
            if match is None or int(match.group(1)) >= frame_count:
                raise ValueError(f"unexpected incomplete evidence: {path}")
            # This exact file is an unpublished write owned by this stage.
            path.unlink()
            continue
        match = _EVIDENCE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected scan evidence file: {path}")
        index = int(match.group(1))
        if index >= frame_count or index in indexes:
            raise ValueError(f"invalid or duplicate scan index {index}")
        record = load_scan_evidence(path, shape, masks=False)
        if (record["frame_index"] != index
                or record["source_frame_index"] != source_start + index):
            raise ValueError(f"frame identity mismatch in {path}")
        indexes.add(index)
    return indexes


def _cuda_available() -> bool:
    if (os.environ.get("RUNFILES_DIR")
            or os.environ.get("RUNFILES_MANIFEST_FILE")):
        import common.torch.load_torch_deps  # noqa: F401
    import torch
    return bool(torch.cuda.is_available())


def _resolve_inference_device(requested: str) -> str:
    if not isinstance(requested, str) or not requested:
        raise ValueError("inference device must be a nonempty string")
    if requested != "auto":
        return requested
    return "0" if _cuda_available() else "cpu"


def _worker_initialize(weights: str, imgsz: int, candidate_confidence: float,
                       direct_confidence: float, device: str,
                       torch_threads: int):
    global _WORKER_DETECTOR, _WORKER_DIRECT_CONFIDENCE
    try:
        if (os.environ.get("RUNFILES_DIR")
                or os.environ.get("RUNFILES_MANIFEST_FILE")):
            import common.torch.load_torch_deps  # noqa: F401
        import torch
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cv2.setNumThreads(1)
        _WORKER_DETECTOR = person_segmentation_preview.YoloPersonSegmenter(
            Path(weights), imgsz=imgsz,
            candidate_confidence=candidate_confidence, device=device,
            classes=(0, 2, 3, 5, 7))
        _WORKER_DIRECT_CONFIDENCE = direct_confidence
    except BaseException:
        traceback.print_exc()
        raise


def _worker_detect(task: tuple[int, int, np.ndarray]) -> dict:
    frame_index, source_frame_index, frame = task
    instances = _WORKER_DETECTOR.detect(frame)
    direct = np.zeros(frame.shape[:2], dtype=bool)
    weak = np.zeros_like(direct)
    vehicle = np.zeros_like(direct)
    metadata = []
    for item in instances:
        class_id = int(item.get("class_id", 0))
        if class_id != 0:
            vehicle |= item["mask"]
        elif item["confidence"] >= _WORKER_DIRECT_CONFIDENCE:
            direct |= item["mask"]
        else:
            weak |= item["mask"]
        metadata.append({
            "confidence": float(item["confidence"]),
            "passes": list(item.get("passes", [])),
            "class_id": class_id,
            "area_pixels": int(np.count_nonzero(item["mask"])),
        })
    return {
        "frame_index": frame_index,
        "source_frame_index": source_frame_index,
        "direct_bits": _pack_mask(direct),
        "weak_bits": _pack_mask(weak),
        "vehicle_bits": _pack_mask(vehicle),
        "shape": direct.shape,
        "instances": metadata,
    }


def _write_worker_result(path: Path, result: dict):
    shape = tuple(result["shape"])
    direct = _unpack_mask(result["direct_bits"], shape)
    weak = _unpack_mask(result["weak_bits"], shape)
    vehicle = _unpack_mask(result["vehicle_bits"], shape)
    instances = [{
        "confidence": item["confidence"],
        "passes": item["passes"],
        "class_id": item["class_id"],
        "area_pixels": item["area_pixels"],
    } for item in result["instances"]]
    write_scan_evidence(
        path, frame_index=result["frame_index"],
        source_frame_index=result["source_frame_index"],
        direct_mask=direct, weak_mask=weak, instances=instances,
        vehicle_mask=vehicle)


def _implementation_sha256() -> str:
    """Bind resumable scans to all local code that can affect evidence."""
    digest = hashlib.sha256(b"person_anonymize_video.scan.v2\0")
    modules = {
        "person_anonymize_video": Path(__file__).resolve(),
        "anonymize_video": Path(anonymize_video.__file__).resolve(),
        "person_segmentation_preview": Path(
            person_segmentation_preview.__file__).resolve(),
    }
    for name, path in sorted(modules.items()):
        encoded_name = name.encode()
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(anonymize_video.sha256_file(path)))
    return digest.hexdigest()


def _stage_implementation(contract: str, *, include_persistence: bool) -> dict:
    """Bind resumable policy/render output to every executing local module."""
    modules = {
        "person_anonymize_video": Path(__file__).resolve(),
        "anonymize_video": Path(anonymize_video.__file__).resolve(),
        "person_segmentation_preview": Path(
            person_segmentation_preview.__file__).resolve(),
    }
    if include_persistence:
        modules["person_mask_persistence"] = Path(
            person_mask_persistence.__file__).resolve()
    return {
        "contract": contract,
        "module_sha256": {
            name: anonymize_video.sha256_file(path)
            for name, path in sorted(modules.items())
        },
    }


def _runtime_versions() -> dict:
    ffmpeg = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True).stdout
    first_line = ffmpeg.splitlines()[0].decode("utf-8", "replace")
    return {
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "ffmpeg_first_line": first_line,
        "ffmpeg_version_output_sha256": hashlib.sha256(ffmpeg).hexdigest(),
    }


def _scan_library_versions() -> dict:
    return {
        **_runtime_versions(),
        "torch": importlib.metadata.version("torch"),
        "ultralytics": importlib.metadata.version("ultralytics"),
    }


def _verify_scan_finalization_inputs(
        spec: dict, source: Path, weights: Path):
    """Fail closed if immutable scan inputs changed while work was running."""
    _verify_regular_sha256(
        source, spec["source"]["sha256"], "scan source video")
    _verify_regular_sha256(
        weights, spec["weights"]["sha256"], "scan model weights")
    if _implementation_sha256() != spec["implementation_sha256"]:
        raise ValueError("scan implementation changed while scan was running")
    if _scan_library_versions() != spec["library_versions"]:
        raise ValueError("scan runtime changed while scan was running")


def _scan_spec(args, source: Path, weights: Path, source_info: dict,
               clip: dict, scan_height: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "person_segmentation_video_scan_spec",
        "implementation_sha256": _implementation_sha256(),
        "source": {
            "path": str(source),
            "sha256": anonymize_video.sha256_file(source),
            "capture_fps": args.capture_fps,
            "probe": source_info,
            "clip": clip,
        },
        "output_fps": args.output_fps,
        "frame_count": clip["frame_count"],
        "weights": {
            "path": str(weights),
            "sha256": anonymize_video.sha256_file(weights),
            "bytes": weights.stat().st_size,
        },
        "detector": {
            "family": "Ultralytics YOLO11x instance segmentation",
            "class": 0,
            "class_name": "person",
            "auxiliary_context_classes": {
                "2": "car",
                "3": "motorcycle",
                "5": "bus",
                "7": "truck",
                "purpose": "license-plate candidate validation only",
            },
            "imgsz": args.imgsz,
            "candidate_confidence": args.candidate_confidence,
            "direct_confidence": args.direct_confidence,
            "passes": ["native", "horizontal_roll_0.5"],
            "retina_masks": True,
            "device": args.device,
        },
        "preprocessing": {
            "resolution": [args.scan_width, scan_height],
            "decoder": "ffmpeg raw BGR",
            "scale_flags": "area",
        },
        "library_versions": _scan_library_versions(),
    }


def _frames_tree_and_ledger(frames_dir: Path, spec: dict) -> tuple[dict, bytes, dict]:
    digest = hashlib.sha256()
    total_bytes = 0
    rows = []
    direct_frames = 0
    direct_instances = 0
    candidate_frames = 0
    candidate_instances = 0
    vehicle_frames = 0
    vehicle_instances = 0
    scan_height = spec["preprocessing"]["resolution"][1]
    scan_width = spec["preprocessing"]["resolution"][0]
    source_start = spec["source"]["clip"]["start_frame"]
    fps = spec["output_fps"]
    direct_confidence = spec["detector"]["direct_confidence"]
    for index in range(spec["frame_count"]):
        path = frames_dir / _evidence_name(index)
        record = load_scan_evidence(
            path, (scan_height, scan_width), masks=False)
        file_sha = anonymize_video.sha256_file(path)
        relative = f"frames/{path.name}"
        relative_bytes = relative.encode()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(bytes.fromhex(file_sha))
        total_bytes += path.stat().st_size
        person_instances = [item for item in record["instances"]
                            if int(item.get("class_id", 0)) == 0]
        vehicle_context = [item for item in record["instances"]
                           if int(item.get("class_id", 0)) != 0]
        confidences = [float(item["confidence"])
                       for item in person_instances]
        direct_count = sum(value >= direct_confidence for value in confidences)
        if direct_count:
            direct_frames += 1
        if confidences:
            candidate_frames += 1
        direct_instances += direct_count
        candidate_instances += len(confidences)
        if vehicle_context:
            vehicle_frames += 1
        vehicle_instances += len(vehicle_context)
        rows.append({
            "frame_index": index,
            "video_t_s": round(index / fps, 6),
            "source_frame_index": source_start + index,
            "source_video_t_s": round((source_start + index) / fps, 6),
            "evidence": relative,
            "sha256": file_sha,
            "direct_count": direct_count,
            "candidate_count": len(confidences),
            "candidate_confidences": confidences,
            "vehicle_context": [{
                "class_id": int(item["class_id"]),
                "confidence": float(item["confidence"]),
                "area_pixels": int(item["area_pixels"]),
            } for item in vehicle_context],
        })
    ledger = "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    return ({
        "sha256": digest.hexdigest(),
        "regular_file_count": len(rows),
        "bytes": total_bytes,
    }, ledger, {
        "direct_instances": direct_instances,
        "direct_frames": direct_frames,
        "candidate_instances": candidate_instances,
        "candidate_frames": candidate_frames,
        "vehicle_instances": vehicle_instances,
        "vehicle_frames": vehicle_frames,
    })


def scan(args) -> int:
    source = args.source.resolve()
    weights = args.weights.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source must be a regular file: {source}")
    if not weights.is_file() or weights.is_symlink():
        raise ValueError(f"weights must be a regular file: {weights}")
    if not 0 < args.candidate_confidence < args.direct_confidence < 1:
        raise ValueError(
            "require 0 < candidate_confidence < direct_confidence < 1")
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("workers and torch_threads must both be positive")
    if (not math.isfinite(args.capture_fps) or args.capture_fps <= 0
            or not math.isfinite(args.output_fps) or args.output_fps <= 0):
        raise ValueError("capture_fps and output_fps must both be positive")
    if args.imgsz < 32:
        raise ValueError("imgsz must be at least 32")
    requested_device = args.device
    args.device = _resolve_inference_device(requested_device)
    if requested_device == "auto":
        print(f"resolved inference device: {args.device}", flush=True)
    source_info = anonymize_video.probe_video(source)
    scan_height = round(
        args.scan_width * source_info["height"] / source_info["width"])
    if (args.scan_width < 32 or scan_height < 32
            or args.scan_width % 2 or scan_height % 2):
        raise ValueError("scan dimensions must be even and at least 32 pixels")
    clip = anonymize_video.clip_metadata(
        source_info, args.output_fps, args.start_s, args.end_s)
    spec = _scan_spec(args, source, weights, source_info, clip, scan_height)
    output_dir = args.output_dir.resolve()
    expected = int(clip["frame_count"])
    started = time.monotonic()
    with _locked_stage(output_dir, spec) as (stage, lock_path):
        frames_dir = stage / "frames"
        committed = committed_scan_indices(
            frames_dir, expected, (scan_height, args.scan_width),
            int(clip["start_frame"]))
        missing = expected - len(committed)
        print(
            f"person scan: {len(committed)}/{expected} committed; "
            f"{missing} remaining at {args.scan_width}x{scan_height} with "
            f"{args.workers} worker(s) x {args.torch_threads} torch threads",
            flush=True)
        completed_this_run = 0
        last_update = time.monotonic()
        if missing:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=__import__("multiprocessing").get_context("spawn"),
                initializer=_worker_initialize,
                initargs=(
                    str(weights), args.imgsz, args.candidate_confidence,
                    args.direct_confidence, args.device, args.torch_threads),
            )
            pending: dict[concurrent.futures.Future, int] = {}

            def commit_finished(done):
                nonlocal completed_this_run, last_update
                for future in done:
                    index = pending.pop(future)
                    result = future.result()
                    if result["frame_index"] != index:
                        raise RuntimeError("worker returned the wrong frame")
                    _write_worker_result(
                        frames_dir / _evidence_name(index), result)
                    committed.add(index)
                    completed_this_run += 1
                now = time.monotonic()
                if now - last_update >= 30:
                    elapsed = now - started
                    rate = completed_this_run / max(elapsed, 1e-9)
                    remaining = expected - len(committed)
                    eta = remaining / max(rate, 1e-9)
                    print(
                        f"  committed {len(committed)}/{expected} "
                        f"({rate:.2f} new fps, ~{eta / 60:.1f} min remaining)",
                        flush=True)
                    last_update = now

            try:
                reader = anonymize_video.RawVideoReader(
                    source, source_info, args.output_fps,
                    args.scan_width, scan_height,
                    int(clip["start_frame"]),
                    int(clip["end_frame_exclusive"]), scale_flags="area")
                decoded = 0
                for index, frame in enumerate(reader):
                    decoded += 1
                    if index in committed:
                        continue
                    while len(pending) >= 2 * args.workers:
                        done, _ = concurrent.futures.wait(
                            pending, return_when=concurrent.futures.FIRST_COMPLETED)
                        commit_finished(done)
                    source_index = int(clip["start_frame"]) + index
                    future = executor.submit(
                        _worker_detect, (index, source_index, frame))
                    pending[future] = index
                if decoded != expected:
                    raise RuntimeError(
                        f"decoded {decoded} clip frames; expected {expected}")
                while pending:
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    commit_finished(done)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        committed = committed_scan_indices(
            frames_dir, expected, (scan_height, args.scan_width),
            int(clip["start_frame"]))
        if committed != set(range(expected)):
            missing_indexes = sorted(set(range(expected)) - committed)
            raise RuntimeError(
                f"scan incomplete; first missing indexes: {missing_indexes[:10]}")
        tree, ledger, counts = _frames_tree_and_ledger(frames_dir, spec)
        ledger_path = stage / "frames.jsonl"
        _write_or_validate(ledger_path, ledger)
        _verify_scan_finalization_inputs(spec, source, weights)
        manifest = {
            **spec,
            "schema_version": SCHEMA_VERSION,
            "kind": "person_segmentation_video_scan",
            "status": "scanned",
            "evidence": {
                "directory": "frames",
                "tree": tree,
                "ledger": {
                    "path": ledger_path.name,
                    "sha256": anonymize_video.sha256_file(ledger_path),
                },
                "counts": counts,
            },
            "completed_utc": _utc_now(),
            "finalization_elapsed_s": round(time.monotonic() - started, 3),
            "argv": list(sys.argv),
        }
        manifest_path = stage / "scan_manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(f"invalid completed manifest: {manifest_path}")
            existing_manifest = json.loads(manifest_path.read_text())
            for key in (
                    "schema_version", "kind", "status", "source",
                    "output_fps", "frame_count", "weights", "detector",
                    "preprocessing", "library_versions", "evidence"):
                if existing_manifest.get(key) != manifest.get(key):
                    raise ValueError(
                        f"existing completed manifest differs at {key}")
        else:
            _write_new_atomic(manifest_path, _json_bytes(manifest))
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"refusing to replace output: {output_dir}")
        os.rename(stage, output_dir)
    print(f"published person scan: {output_dir}", flush=True)
    return 0


def _load_scan(scan_dir: Path) -> tuple[dict, Path, str, list[dict]]:
    scan_dir = scan_dir.resolve()
    manifest_path = scan_dir / "scan_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing regular scan manifest: {manifest_path}")
    manifest_sha = anonymize_video.sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("kind") != "person_segmentation_video_scan"
            or manifest.get("status") != "scanned"):
        raise ValueError(f"not a completed person scan: {manifest_path}")
    ledger = scan_dir / manifest["evidence"]["ledger"]["path"]
    if ledger.is_symlink() or not ledger.is_file():
        raise ValueError("person scan ledger is missing or changed")
    ledger_bytes = anonymize_video.read_regular_file_bytes(ledger)
    if (hashlib.sha256(ledger_bytes).hexdigest()
            != manifest["evidence"]["ledger"]["sha256"]):
        raise ValueError("person scan ledger is missing or changed")
    rows = anonymize_video.parse_jsonl_bytes(ledger_bytes)
    if len(rows) != manifest["frame_count"]:
        raise ValueError("person scan ledger length changed")
    source_start = int(manifest["source"]["clip"]["start_frame"])
    frames_dir = manifest["evidence"]["directory"]
    for index, row in enumerate(rows):
        if (row.get("source_frame_index") != source_start + index
                or row.get("evidence") != f"{frames_dir}/{_evidence_name(index)}"
                or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", ""))):
            raise ValueError(f"person scan ledger mapping changed at {index}")
    return manifest, manifest_path, manifest_sha, rows


def _validate_box(box, *, context: str) -> list[float]:
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in box)):
        raise ValueError(f"invalid normalized box in {context}")
    clean = [float(value) for value in box]
    if (clean != anonymize_video.clamp_box(clean)
            or not (clean[0] < clean[2] and clean[1] < clean[3])):
        raise ValueError(f"out-of-bounds normalized box in {context}")
    return clean


def _load_plate_candidates(plate_manifest_path: Path,
                           scan_manifest: dict) -> tuple[list[list[dict]], dict]:
    """Load only raw YOLO plate candidates; faces/manual boxes never survive."""
    plate_manifest_path = plate_manifest_path.resolve()
    if plate_manifest_path.is_symlink() or not plate_manifest_path.is_file():
        raise ValueError(
            f"plate candidate manifest must be regular: {plate_manifest_path}")
    manifest_sha = anonymize_video.sha256_file(plate_manifest_path)
    manifest = json.loads(plate_manifest_path.read_text())
    expected_clip = scan_manifest["source"]["clip"]
    if manifest.get("source", {}).get("sha256") != scan_manifest["source"]["sha256"]:
        raise ValueError("plate candidates refer to a different source")
    if manifest.get("source", {}).get("clip") != expected_clip:
        raise ValueError("plate candidates use a different clip")
    if (not math.isclose(float(manifest.get("output_fps", -1)),
                         float(scan_manifest["output_fps"]), abs_tol=1e-9)
            or manifest.get("frame_count") != scan_manifest["frame_count"]):
        raise ValueError("plate candidate fps/frame count differs from scan")
    detector = manifest.get("detectors", {}).get("license_plate", {})
    model_path = Path(detector.get("model", ""))
    if (model_path.is_symlink() or not model_path.is_file()
            or anonymize_video.sha256_file(model_path)
            != detector.get("model_sha256")):
        raise ValueError("plate model provenance is missing or changed")
    raw_name = manifest.get("files", {}).get("raw_ledger", {}).get("path")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        raise ValueError("invalid raw plate-candidate ledger path")
    raw_path = plate_manifest_path.parent / raw_name
    raw_bytes = anonymize_video.read_regular_file_bytes(raw_path)
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha != manifest["files"]["raw_ledger"]["sha256"]:
        raise ValueError("raw plate-candidate ledger changed")
    rows = anonymize_video.parse_jsonl_bytes(raw_bytes)
    if len(rows) != scan_manifest["frame_count"]:
        raise ValueError("raw plate-candidate ledger length changed")
    fps = float(scan_manifest["output_fps"])
    source_start = int(expected_clip["start_frame"])
    filtered = []
    total = 0
    for index, row in enumerate(rows):
        source_index = source_start + index
        if (row.get("source_frame_index") != source_index
                or not math.isclose(float(row.get("video_t_s", -1)),
                                    index / fps, abs_tol=1e-6)
                or not math.isclose(float(row.get("source_video_t_s", -1)),
                                    source_index / fps, abs_tol=1e-6)):
            raise ValueError(f"plate candidate row mapping changed at {index}")
        candidates = []
        for detection_index, detection in enumerate(row.get("detections", [])):
            if detection.get("category") != "license_plate":
                continue
            if detection.get("source") != "yolov9_plate":
                raise ValueError(
                    f"unknown plate source at row {index}: {detection.get('source')}")
            confidence = detection.get("confidence")
            if (isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not math.isfinite(float(confidence))
                    or not 0 <= float(confidence) <= 1):
                raise ValueError(f"invalid plate confidence at row {index}")
            candidates.append({
                "category": "license_plate_candidate",
                "source": "yolov9_plate_raw_qa",
                "confidence": round(float(confidence), 6),
                "box": _validate_box(
                    detection.get("box"),
                    context=f"plate row {index} detection {detection_index}"),
            })
        filtered.append(candidates)
        total += len(candidates)
    provenance = {
        "manifest": str(plate_manifest_path),
        "manifest_sha256": manifest_sha,
        "raw_ledger": str(raw_path),
        "raw_ledger_sha256": raw_sha,
        "candidate_count": total,
        "candidate_frames": sum(bool(items) for items in filtered),
        "model": str(model_path.resolve()),
        "model_sha256": detector["model_sha256"],
        "threshold": detector.get("threshold"),
        "automatic_use": (
            "direct-frame candidates are blurred only when plausible plate "
            "geometry overlaps independent YOLO vehicle context"),
    }
    return filtered, provenance


def _box_metrics(box: list[float], width: int, height: int) -> dict:
    normalized_width = box[2] - box[0]
    normalized_height = box[3] - box[1]
    pixel_aspect = ((normalized_width * width)
                    / max(normalized_height * height, 1e-9))
    return {
        "normalized_width": normalized_width,
        "normalized_height": normalized_height,
        "normalized_area": normalized_width * normalized_height,
        "pixel_aspect": pixel_aspect,
    }


def _box_mask(box: list[float], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = anonymize_video._pixel_bounds(box, width, height)
    mask[y1:y2, x1:x2] = True
    return mask


def validate_plate_candidate(candidate: dict, vehicle_mask: np.ndarray,
                             *, max_width: float = 0.18,
                             max_height: float = 0.12,
                             max_area: float = 0.03,
                             min_aspect: float = 0.75,
                             max_aspect: float = 8.0,
                             min_vehicle_overlap: float = 0.10) -> tuple[bool, dict]:
    """Gate a plate candidate with geometry and an independent vehicle mask."""
    height, width = vehicle_mask.shape
    metrics = _box_metrics(candidate["box"], width, height)
    plausible = (
        metrics["normalized_width"] <= max_width
        and metrics["normalized_height"] <= max_height
        and metrics["normalized_area"] <= max_area
        and min_aspect <= metrics["pixel_aspect"] <= max_aspect)
    candidate_mask = _box_mask(candidate["box"], width, height)
    expanded_vehicle = cv2.dilate(
        vehicle_mask.astype(np.uint8), np.ones((41, 41), dtype=np.uint8)) != 0
    candidate_pixels = int(np.count_nonzero(candidate_mask))
    overlap = (int(np.count_nonzero(candidate_mask & expanded_vehicle))
               / max(candidate_pixels, 1))
    metrics.update({
        "geometry_plausible": plausible,
        "vehicle_overlap": overlap,
        "min_vehicle_overlap": min_vehicle_overlap,
    })
    return plausible and overlap >= min_vehicle_overlap, metrics


def _filtered_plate_policy(candidates: list[list[dict]], scan_dir: Path,
                           scan_manifest: dict, scan_rows: list[dict]) -> tuple[
                               list[list[dict]], list[list[dict]], list[dict]]:
    """Return applied plates, review candidates, and a complete QA ledger."""
    scan_width, scan_height = scan_manifest["preprocessing"]["resolution"]
    frames_dir = scan_dir / scan_manifest["evidence"]["directory"]
    base_applied: list[list[dict]] = [[] for _ in candidates]
    review: list[list[dict]] = [[] for _ in candidates]
    audit_rows = []
    for index, frame_candidates in enumerate(candidates):
        evidence_path = frames_dir / _evidence_name(index)
        if (anonymize_video.sha256_file(evidence_path)
                != scan_rows[index]["sha256"]):
            raise ValueError(f"person scan evidence changed at {index}")
        record = load_scan_evidence(
            evidence_path,
            (scan_height, scan_width), masks=True)
        decisions = []
        for candidate in frame_candidates:
            vehicle_validated, metrics = validate_plate_candidate(
                candidate, record["vehicle_mask"])
            geometry_plausible = bool(metrics["geometry_plausible"])
            accepted = vehicle_validated
            if not geometry_plausible:
                decision_name = "rejected_implausible_geometry"
            elif vehicle_validated:
                decision_name = "accepted_vehicle_supported"
            else:
                decision_name = "rejected_missing_vehicle_context"
            decision = {**candidate, "metrics": {
                key: round(float(value), 9) if isinstance(value, float) else value
                for key, value in metrics.items()},
                "decision": decision_name}
            decisions.append(decision)
            if accepted:
                base_applied[index].append({
                    "category": "license_plate",
                    "source": "yolov9_plate_vehicle_validated",
                    "confidence": candidate["confidence"],
                    "box": list(candidate["box"]),
                    "contributing_frames": [index],
                    "validation": decision["metrics"],
                })
            if not vehicle_validated:
                review[index].append(decision)
        audit_rows.append({
            "frame_index": index,
            "candidates": decisions,
        })

    # Static rectangle propagation was the source of visible over-blur in the
    # earlier pipeline.  Plates remain direct-only until a flow-based box/mask
    # policy is separately reviewed.
    return base_applied, review, audit_rows


def _conservative_resize_mask(mask: np.ndarray, width: int,
                              height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.copy()
    return cv2.resize(
        mask.astype(np.float32), (width, height),
        interpolation=cv2.INTER_AREA) > 0.0


def _display_review_mask(result) -> tuple[np.ndarray, np.ndarray]:
    raw = np.zeros_like(result.accepted_mask)
    for flag in result.review_flags:
        raw |= flag.mask
    covered = cv2.dilate(
        result.accepted_mask.astype(np.uint8),
        # At the default 960-wide flow grid, radius three corresponds to 24
        # source pixels.  The applied blur alpha reaches roughly 30 source
        # pixels beyond a direct mask, so this suppresses only suspicion that
        # is already inside the blurred footprint.
        np.ones((7, 7), dtype=np.uint8)) != 0
    display = raw & ~covered
    return raw, display


def write_policy_evidence(path: Path, *, frame_index: int,
                          source_frame_index: int,
                          temporal_mask: np.ndarray,
                          review_raw_mask: np.ndarray,
                          review_display_mask: np.ndarray,
                          metadata: dict):
    shape = temporal_mask.shape
    if (review_raw_mask.shape != shape
            or review_display_mask.shape != shape or temporal_mask.ndim != 2):
        raise ValueError("all policy masks must have the same HxW shape")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace policy evidence: {path}")
    temporary = path.with_name(path.name + ".incomplete")
    temporary.unlink(missing_ok=True)
    with temporary.open("xb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
            frame_index=np.asarray(frame_index, dtype=np.int64),
            source_frame_index=np.asarray(source_frame_index, dtype=np.int64),
            mask_height=np.asarray(shape[0], dtype=np.int32),
            mask_width=np.asarray(shape[1], dtype=np.int32),
            temporal_bits=_pack_mask(temporal_mask),
            review_raw_bits=_pack_mask(review_raw_mask),
            review_display_bits=_pack_mask(review_display_mask),
            metadata_json=np.asarray(json.dumps(
                person_segmentation_preview._json_safe(metadata),
                separators=(",", ":"))),
        )
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to replace policy evidence: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def load_policy_evidence(path: Path, expected_shape: tuple[int, int]) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"policy evidence must be regular: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            shape = (int(payload["mask_height"]), int(payload["mask_width"]))
            if shape != expected_shape:
                raise ValueError(f"policy mask shape {shape} != {expected_shape}")
            if int(payload["schema_version"]) != SCHEMA_VERSION:
                raise ValueError("unsupported policy evidence schema")
            return {
                "frame_index": int(payload["frame_index"]),
                "source_frame_index": int(payload["source_frame_index"]),
                "temporal_mask": _unpack_mask(payload["temporal_bits"], shape),
                "review_raw_mask": _unpack_mask(
                    payload["review_raw_bits"], shape),
                "review_display_mask": _unpack_mask(
                    payload["review_display_bits"], shape),
                "metadata": json.loads(str(payload["metadata_json"])),
            }
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid policy evidence {path}: {error}") from error


def committed_policy_indices(frames_dir: Path, frame_count: int,
                             shape: tuple[int, int], source_start: int) -> set[int]:
    frames_dir.mkdir(exist_ok=True)
    _regular_directory(frames_dir, "policy evidence path")
    indexes = set()
    for path in frames_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unexpected policy evidence entry: {path}")
        if path.name.endswith(".npz.incomplete"):
            match = _EVIDENCE_PATTERN.fullmatch(
                path.name.removesuffix(".incomplete"))
            if match is None or int(match.group(1)) >= frame_count:
                raise ValueError(f"unexpected incomplete policy evidence: {path}")
            path.unlink()
            continue
        match = _EVIDENCE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected policy evidence file: {path}")
        index = int(match.group(1))
        if index >= frame_count or index in indexes:
            raise ValueError(f"invalid or duplicate policy index {index}")
        record = load_policy_evidence(path, shape)
        if (record["frame_index"] != index
                or record["source_frame_index"] != source_start + index):
            raise ValueError(f"policy frame identity mismatch in {path}")
        indexes.add(index)
    return indexes


def _estimate_pair_flows(first_frame: np.ndarray, second_frame: np.ndarray,
                         config) -> tuple[np.ndarray, np.ndarray]:
    """Return first->second and second->first panorama-aware dense flow."""
    first = person_mask_persistence._as_gray_u8(first_frame)
    second = person_mask_persistence._as_gray_u8(second_frame)
    return (
        person_mask_persistence._estimate_flow(first, second, config),
        person_mask_persistence._estimate_flow(second, first, config),
    )


def _boundary_persistence(
        direct_mask: np.ndarray, weak_mask: np.ndarray,
        *, boundary_frame: np.ndarray | None = None,
        adjacent_frame: np.ndarray | None = None,
        adjacent_direct_mask: np.ndarray | None = None,
        boundary_to_adjacent_flow: np.ndarray | None = None,
        adjacent_to_boundary_flow: np.ndarray | None = None,
        config: person_mask_persistence.PersistenceConfig | None = None):
    """Keep boundary candidates visible when two-sided flow is unavailable."""
    empty = np.zeros_like(direct_mask)
    suspicion = weak_mask & ~direct_mask
    flags = []
    if np.any(suspicion):
        flags.append(person_mask_persistence.ReviewFlag(
            reasons=("boundary_unconfirmed_candidate",),
            mask=suspicion,
            metrics={
                "boundary_frame": 1.0,
                "candidate_pixels": float(np.count_nonzero(suspicion)),
            }))
    adjacent_supplied = any(value is not None for value in (
        boundary_frame, adjacent_frame, adjacent_direct_mask))
    if adjacent_supplied:
        if any(value is None for value in (
                boundary_frame, adjacent_frame, adjacent_direct_mask)):
            raise ValueError("all one-sided boundary inputs must be supplied")
        config = config or person_mask_persistence.PersistenceConfig()
        boundary_gray = person_mask_persistence._as_gray_u8(boundary_frame)
        adjacent_gray = person_mask_persistence._as_gray_u8(adjacent_frame)
        adjacent_direct = person_mask_persistence._as_mask(
            adjacent_direct_mask, direct_mask.shape, "adjacent_direct_mask")
        scene_distance = person_mask_persistence._histogram_distance(
            boundary_gray, adjacent_gray)
        if scene_distance <= config.max_scene_histogram_distance:
            flow_supplied = (boundary_to_adjacent_flow is not None
                             or adjacent_to_boundary_flow is not None)
            if flow_supplied and (boundary_to_adjacent_flow is None
                                  or adjacent_to_boundary_flow is None):
                raise ValueError(
                    "both one-sided boundary flow fields must be supplied")
            if not flow_supplied:
                boundary_to_adjacent_flow, adjacent_to_boundary_flow = (
                    _estimate_pair_flows(
                        boundary_frame, adjacent_frame, config))
            warped = person_mask_persistence._warp_source_mask_to_target(
                adjacent_direct, adjacent_gray, boundary_gray,
                adjacent_to_boundary_flow, boundary_to_adjacent_flow, config)
            supported = (
                warped.raw_mask & warped.valid_mask
                & (warped.photometric_error
                   <= config.max_component_photometric_error)
                & ~direct_mask)
            components = person_mask_persistence._components(
                supported, config.min_mask_pixels)
            if components:
                adjacent_suspicion = np.logical_or.reduce(components)
                flags.append(person_mask_persistence.ReviewFlag(
                    reasons=("boundary_one_sided_person_evidence",),
                    mask=adjacent_suspicion,
                    metrics={
                        "boundary_frame": 1.0,
                        "scene_histogram_distance": scene_distance,
                        "candidate_pixels": float(np.count_nonzero(
                            adjacent_suspicion)),
                    }))
    return person_mask_persistence.PersistenceResult(
        accepted_mask=direct_mask.copy(), temporal_fill_mask=empty,
        fills=(), review_flags=tuple(flags), metrics={"boundary_frame": 1.0})


def _commit_policy_result(path: Path, index: int, source_index: int,
                          direct_scan: np.ndarray, result):
    scan_height, scan_width = direct_scan.shape
    temporal = person_segmentation_preview._resize_mask(
        result.temporal_fill_mask, scan_width, scan_height)
    raw_review, display_review = _display_review_mask(result)
    raw_review = person_segmentation_preview._resize_mask(
        raw_review, scan_width, scan_height)
    display_review = person_segmentation_preview._resize_mask(
        display_review, scan_width, scan_height)
    write_policy_evidence(
        path, frame_index=index, source_frame_index=source_index,
        temporal_mask=temporal, review_raw_mask=raw_review,
        review_display_mask=display_review, metadata=result.metadata())


def _manual_detections(manual_regions: list[dict], source_time_s: float) -> list[dict]:
    detections = []
    for region in manual_regions:
        if (region["start_s"] <= source_time_s
                and (region["end_s"] is None
                     or source_time_s < region["end_s"])):
            detections.append({
                "category": "manual_privacy",
                "source": "manual_region",
                "manual_region_id": region["id"],
                "confidence": 1.0,
                "box": list(region["box"]),
                "reason": region["reason"],
            })
    return detections


def _policy_tree_and_ledgers(
        policy_frames: Path, scan_dir: Path, scan_manifest: dict,
        applied_plates: list[list[dict]], plate_review: list[list[dict]],
        manual_regions: list[dict]) -> tuple[dict, bytes, dict]:
    scan_width, scan_height = scan_manifest["preprocessing"]["resolution"]
    scan_frames = scan_dir / scan_manifest["evidence"]["directory"]
    source_start = int(scan_manifest["source"]["clip"]["start_frame"])
    fps = float(scan_manifest["output_fps"])
    digest = hashlib.sha256()
    total_bytes = 0
    rows = []
    temporal_frames = 0
    temporal_pixels = 0
    review_frames = 0
    review_pixels = 0
    raw_review_frames = 0
    raw_review_pixels = 0
    for index in range(scan_manifest["frame_count"]):
        path = policy_frames / _evidence_name(index)
        policy = load_policy_evidence(path, (scan_height, scan_width))
        scan = load_scan_evidence(
            scan_frames / _evidence_name(index),
            (scan_height, scan_width), masks=True)
        file_sha = anonymize_video.sha256_file(path)
        relative = f"frames/{path.name}"
        name = relative.encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(file_sha))
        total_bytes += path.stat().st_size
        temporal_count = int(np.count_nonzero(policy["temporal_mask"]))
        review_count = int(np.count_nonzero(policy["review_display_mask"]))
        raw_review_count = int(np.count_nonzero(policy["review_raw_mask"]))
        temporal_frames += bool(temporal_count)
        temporal_pixels += temporal_count
        review_frames += bool(review_count)
        review_pixels += review_count
        raw_review_frames += bool(raw_review_count)
        raw_review_pixels += raw_review_count
        source_index = source_start + index
        source_time = source_index / fps
        detections = list(applied_plates[index]) + _manual_detections(
            manual_regions, source_time)
        rows.append({
            "frame_index": index,
            "video_t_s": round(index / fps, 6),
            "source_frame_index": source_index,
            "source_video_t_s": round(source_time, 6),
            "person_masks": {
                "scan_evidence": str((
                    scan_frames / _evidence_name(index)).resolve()),
                "scan_evidence_sha256": anonymize_video.sha256_file(
                    scan_frames / _evidence_name(index)),
                "policy_evidence": relative,
                "policy_evidence_sha256": file_sha,
                "direct_pixels": int(np.count_nonzero(scan["direct_mask"])),
                "temporal_fill_pixels": temporal_count,
                "applied_pixels": int(np.count_nonzero(
                    scan["direct_mask"] | policy["temporal_mask"])),
            },
            "detections": detections,
            "review": {
                "person_suspicion_pixels": raw_review_count,
                "person_suspicion_display_pixels": review_count,
                "plate_candidates": plate_review[index],
            },
        })
    ledger = "".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    return ({
        "sha256": digest.hexdigest(),
        "regular_file_count": len(rows),
        "bytes": total_bytes,
    }, ledger, {
        "temporal_fill_frames": temporal_frames,
        "temporal_fill_pixels": temporal_pixels,
        "review_flag_frames": review_frames,
        "review_flag_pixels": review_pixels,
        "raw_review_flag_frames": raw_review_frames,
        "raw_review_flag_pixels": raw_review_pixels,
        "applied_plate_regions": sum(map(len, applied_plates)),
        "applied_plate_frames": sum(bool(items) for items in applied_plates),
        "plate_review_regions": sum(map(len, plate_review)),
        "plate_review_frames": sum(bool(items) for items in plate_review),
    })


def apply_policy(args) -> int:
    scan_dir = args.scan_dir.resolve()
    scan_manifest, scan_manifest_path, scan_manifest_sha, scan_rows = (
        _load_scan(scan_dir))
    source = Path(scan_manifest["source"]["path"])
    if (source.is_symlink() or not source.is_file()
            or anonymize_video.sha256_file(source)
            != scan_manifest["source"]["sha256"]):
        raise ValueError("person scan source is missing or changed")
    candidates, plate_provenance = _load_plate_candidates(
        args.plate_manifest, scan_manifest)
    manual_regions = anonymize_video.read_manual_regions(args.manual_regions)
    manual_provenance = None
    if args.manual_regions:
        manual_provenance = {
            "path": str(args.manual_regions.resolve()),
            "sha256": anonymize_video.sha256_file(args.manual_regions),
        }
    scan_width, scan_height = scan_manifest["preprocessing"]["resolution"]
    flow_height = round(args.flow_width * scan_height / scan_width)
    if args.flow_width < 64 or flow_height < 32:
        raise ValueError("flow resolution is unreasonably small")
    config = person_mask_persistence.PersistenceConfig()
    spec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "person_mask_policy_spec",
        "implementation": _stage_implementation(
            "person_anonymize_video.policy.v2", include_persistence=True),
        "runtime_versions": _runtime_versions(),
        "scan": {
            "path": str(scan_dir),
            "manifest": str(scan_manifest_path),
            "manifest_sha256": scan_manifest_sha,
            "evidence_tree": scan_manifest["evidence"]["tree"],
        },
        "source": {
            **{key: value for key, value in scan_manifest["source"].items()
               if key != "probe"},
            **scan_manifest["source"]["probe"],
        },
        "output_fps": scan_manifest["output_fps"],
        "frame_count": scan_manifest["frame_count"],
        "scan_resolution": [scan_width, scan_height],
        "person_policy": {
            "gap_frames": 1,
            "two_sided_evidence_required": True,
            "fill_shape": "mask only; rectangles prohibited",
            "flow_resolution": [args.flow_width, flow_height],
            "flow_preprocessing": "ffmpeg area scale",
            "direct_downsample": "INTER_AREA > 0 (conservative)",
            "config": dataclasses.asdict(config),
        },
        "plate_policy": {
            **plate_provenance,
            "geometry": {
                "max_normalized_width": 0.18,
                "max_normalized_height": 0.12,
                "max_normalized_area": 0.03,
                "pixel_aspect_range": [0.75, 8.0],
            },
            "independent_vehicle_overlap_min": 0.10,
            "vehicle_mask_dilation_scan_px": 41,
            "temporal_radius_frames": 0,
            "temporal_policy": "direct-only; static rectangle propagation prohibited",
            "geometry_implausible_candidates": "retained for QA; never blurred",
            "vehicle_unsupported_plausible_candidates": (
                "retained for QA; never blurred automatically"),
        },
        "manual_regions": manual_regions,
        "manual_regions_provenance": manual_provenance,
    }
    output_dir = args.output_dir.resolve()
    started = time.monotonic()
    with _locked_named_stage(
            output_dir, spec, spec_name="policy_spec.json",
            lock_name=".policy.lock") as (stage, lock_path):
        frames_dir = stage / "frames"
        source_start = int(scan_manifest["source"]["clip"]["start_frame"])
        expected = int(scan_manifest["frame_count"])
        committed = committed_policy_indices(
            frames_dir, expected, (scan_height, scan_width), source_start)
        print(f"person persistence: {len(committed)}/{expected} committed; "
              f"{expected - len(committed)} remaining", flush=True)
        source_info = scan_manifest["source"]["probe"]
        reader = iter(anonymize_video.RawVideoReader(
            source, source_info, scan_manifest["output_fps"],
            args.flow_width, flow_height,
            source_start,
            int(scan_manifest["source"]["clip"]["end_frame_exclusive"]),
            scale_flags="area"))
        scan_frames = scan_dir / scan_manifest["evidence"]["directory"]
        decoded = 0
        last_update = time.monotonic()

        def next_frame(index):
            nonlocal decoded
            try:
                frame = next(reader)
            except StopIteration as error:
                raise RuntimeError(
                    f"decoded only {decoded}/{expected} policy frames") from error
            decoded += 1
            scan_record = load_scan_evidence(
                scan_frames / _evidence_name(index),
                (scan_height, scan_width), masks=True)
            if (anonymize_video.sha256_file(
                    scan_frames / _evidence_name(index))
                    != scan_rows[index]["sha256"]):
                raise ValueError(f"person scan evidence changed at {index}")
            return {
                "frame": frame,
                "direct": _conservative_resize_mask(
                    scan_record["direct_mask"], args.flow_width, flow_height),
                "weak": _conservative_resize_mask(
                    scan_record["weak_mask"], args.flow_width, flow_height),
                "direct_scan": scan_record["direct_mask"],
            }

        def commit(index, frame_record, result):
            nonlocal last_update
            if index in committed:
                return
            _commit_policy_result(
                frames_dir / _evidence_name(index), index,
                source_start + index, frame_record["direct_scan"], result)
            committed.add(index)
            now = time.monotonic()
            if now - last_update >= 30:
                elapsed = now - started
                rate = len(committed) / max(elapsed, 1e-9)
                remaining = expected - len(committed)
                print(f"  persistence {len(committed)}/{expected} "
                      f"({rate:.2f} fps, ~{remaining / max(rate, 1e-9) / 60:.1f} "
                      "min remaining)", flush=True)
                last_update = now

        if expected:
            previous = next_frame(0)
            if expected == 1:
                commit(0, previous, _boundary_persistence(
                    previous["direct"], previous["weak"]))
            else:
                middle = next_frame(1)
                previous_middle_flows = None
                if 0 not in committed:
                    previous_middle_flows = _estimate_pair_flows(
                        previous["frame"], middle["frame"], config)
                    commit(0, previous, _boundary_persistence(
                        previous["direct"], previous["weak"],
                        boundary_frame=previous["frame"],
                        adjacent_frame=middle["frame"],
                        adjacent_direct_mask=middle["direct"],
                        boundary_to_adjacent_flow=previous_middle_flows[0],
                        adjacent_to_boundary_flow=previous_middle_flows[1],
                        config=config))
                for following_index in range(2, expected):
                    following = next_frame(following_index)
                    center = following_index - 1
                    middle_following_flows = None
                    if center not in committed:
                        if previous_middle_flows is None:
                            previous_middle_flows = _estimate_pair_flows(
                                previous["frame"], middle["frame"], config)
                        middle_following_flows = _estimate_pair_flows(
                            middle["frame"], following["frame"], config)
                        flows = person_mask_persistence.GapFlows(
                            previous_to_middle=previous_middle_flows[0],
                            middle_to_previous=previous_middle_flows[1],
                            next_to_middle=middle_following_flows[1],
                            middle_to_next=middle_following_flows[0],
                        )
                        result = person_mask_persistence.bridge_one_frame_gap(
                            previous["frame"], middle["frame"], following["frame"],
                            previous["direct"], middle["direct"],
                            following["direct"], middle["weak"],
                            config=config, flows=flows)
                        commit(center, middle, result)
                    previous, middle = middle, following
                    previous_middle_flows = middle_following_flows
                if expected - 1 not in committed:
                    if previous_middle_flows is None:
                        previous_middle_flows = _estimate_pair_flows(
                            previous["frame"], middle["frame"], config)
                    commit(expected - 1, middle, _boundary_persistence(
                           middle["direct"], middle["weak"],
                           boundary_frame=middle["frame"],
                           adjacent_frame=previous["frame"],
                           adjacent_direct_mask=previous["direct"],
                           boundary_to_adjacent_flow=(
                               previous_middle_flows[1]),
                           adjacent_to_boundary_flow=(
                               previous_middle_flows[0]),
                           config=config))
        try:
            next(reader)
        except StopIteration:
            pass
        else:
            raise RuntimeError("policy decoder produced excess frames")
        if decoded != expected:
            raise RuntimeError(f"decoded {decoded}/{expected} policy frames")

        committed = committed_policy_indices(
            frames_dir, expected, (scan_height, scan_width), source_start)
        if committed != set(range(expected)):
            raise RuntimeError("policy evidence is incomplete after processing")

        print("classifying plate candidates with geometry and vehicle context",
              flush=True)
        applied_plates, plate_review, plate_audit = _filtered_plate_policy(
            candidates, scan_dir, scan_manifest, scan_rows)
        plate_candidate_rows = [{
            "frame_index": index,
            "source_frame_index": source_start + index,
            "candidates": items,
        } for index, items in enumerate(candidates)]
        plate_candidates_bytes = "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in plate_candidate_rows).encode()
        plate_audit_bytes = "".join(
            json.dumps(row, separators=(",", ":")) + "\n"
            for row in plate_audit).encode()
        plate_candidates_path = stage / "plates.candidates.jsonl"
        plate_audit_path = stage / "plates.audit.jsonl"
        _write_or_validate(plate_candidates_path, plate_candidates_bytes)
        _write_or_validate(plate_audit_path, plate_audit_bytes)
        tree, ledger, counts = _policy_tree_and_ledgers(
            frames_dir, scan_dir, scan_manifest,
            applied_plates, plate_review, manual_regions)
        _verify_regular_sha256(
            source, spec["source"]["sha256"], "policy source video")
        _verify_regular_sha256(
            scan_manifest_path, spec["scan"]["manifest_sha256"],
            "policy parent scan manifest")
        _verify_regular_sha256(
            Path(plate_provenance["manifest"]),
            plate_provenance["manifest_sha256"], "plate manifest")
        _verify_regular_sha256(
            Path(plate_provenance["raw_ledger"]),
            plate_provenance["raw_ledger_sha256"], "raw plate ledger")
        _verify_regular_sha256(
            Path(plate_provenance["model"]),
            plate_provenance["model_sha256"], "plate model")
        if manual_provenance:
            _verify_regular_sha256(
                Path(manual_provenance["path"]),
                manual_provenance["sha256"], "manual regions")
        if (_stage_implementation(
                "person_anonymize_video.policy.v2",
                include_persistence=True) != spec["implementation"]
                or _runtime_versions() != spec["runtime_versions"]):
            raise ValueError(
                "policy implementation or runtime changed during processing")
        ledger_path = stage / "detections.jsonl"
        _write_or_validate(ledger_path, ledger)
        manifest = {
            **spec,
            "schema_version": SCHEMA_VERSION,
            "kind": "person_mask_anonymization_policy",
            "status": "scanned",
            "review": {
                "status": "pending",
                "instruction": (
                    "Watch the rendered review video end to end. Every "
                    "identifiable person and license plate must be blurred."),
            },
            "evidence": {
                "directory": "frames",
                "tree": tree,
                "counts": counts,
            },
            "files": {
                "applied_ledger": {
                    "path": ledger_path.name,
                    "sha256": anonymize_video.sha256_file(ledger_path),
                },
                "plate_candidates": {
                    "path": plate_candidates_path.name,
                    "sha256": anonymize_video.sha256_file(
                        plate_candidates_path),
                },
                "plate_audit": {
                    "path": plate_audit_path.name,
                    "sha256": anonymize_video.sha256_file(plate_audit_path),
                },
            },
            "completed_utc": _utc_now(),
            "finalization_elapsed_s": round(time.monotonic() - started, 3),
            "argv": list(sys.argv),
        }
        manifest_path = stage / "anonymization_manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(f"invalid completed policy manifest: {manifest_path}")
            existing = json.loads(manifest_path.read_text())
            for key in (
                    "schema_version", "kind", "status", "implementation",
                    "runtime_versions", "scan", "source",
                    "output_fps", "frame_count", "scan_resolution",
                    "person_policy", "plate_policy", "manual_regions",
                    "manual_regions_provenance", "evidence", "files"):
                if existing.get(key) != manifest.get(key):
                    raise ValueError(
                        f"existing policy manifest differs at {key}")
        else:
            _write_new_atomic(manifest_path, _json_bytes(manifest))
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"refusing to replace output: {output_dir}")
        os.rename(stage, output_dir)
    print(f"published person/plate policy: {output_dir}", flush=True)
    return 0


def _load_policy(policy_dir: Path) -> tuple[dict, Path, str, list[dict]]:
    policy_dir = policy_dir.resolve()
    manifest_path = policy_dir / "anonymization_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing regular policy manifest: {manifest_path}")
    manifest_sha = anonymize_video.sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("kind") != "person_mask_anonymization_policy"
            or manifest.get("status") != "scanned"):
        raise ValueError(f"not a completed person policy: {manifest_path}")
    for item in manifest.get("files", {}).values():
        if (not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or Path(item["path"]).name != item["path"]):
            raise ValueError("policy retained-file paths must be basenames")
    ledger_name = manifest["files"]["applied_ledger"]["path"]
    if Path(ledger_name).name != ledger_name:
        raise ValueError("policy applied ledger path must be a basename")
    ledger_path = policy_dir / ledger_name
    ledger_bytes = anonymize_video.read_regular_file_bytes(ledger_path)
    if (hashlib.sha256(ledger_bytes).hexdigest()
            != manifest["files"]["applied_ledger"]["sha256"]):
        raise ValueError("policy applied ledger changed")
    rows = anonymize_video.parse_jsonl_bytes(ledger_bytes)
    if len(rows) != manifest["frame_count"]:
        raise ValueError("policy applied ledger length changed")
    scan_manifest = Path(manifest["scan"]["manifest"])
    if (scan_manifest.is_symlink() or not scan_manifest.is_file()
            or anonymize_video.sha256_file(scan_manifest)
            != manifest["scan"]["manifest_sha256"]):
        raise ValueError("policy parent scan manifest is missing or changed")
    source_start = int(manifest["source"]["clip"]["start_frame"])
    fps = float(manifest["output_fps"])
    allowed_categories = {"license_plate", "manual_privacy"}
    for index, row in enumerate(rows):
        source_index = source_start + index
        if (row.get("source_frame_index") != source_index
                or not math.isclose(float(row.get("video_t_s", -1)),
                                    index / fps, abs_tol=1e-6)
                or not math.isclose(float(row.get("source_video_t_s", -1)),
                                    source_index / fps, abs_tol=1e-6)):
            raise ValueError(f"policy ledger mapping changed at row {index}")
        for detection in row.get("detections", []):
            if detection.get("category") not in allowed_categories:
                raise ValueError(
                    f"unexpected applied category at row {index}: "
                    f"{detection.get('category')}")
            _validate_box(detection.get("box"), context=f"policy row {index}")
    return manifest, manifest_path, manifest_sha, rows


def _merge_pixel_boxes(boxes: list[tuple[int, int, int, int]]) -> list[
        tuple[int, int, int, int]]:
    """Union intersecting component work regions."""
    remaining = list(boxes)
    merged = []
    while remaining:
        x1, y1, x2, y2 = remaining.pop()
        changed = True
        while changed:
            changed = False
            keep = []
            for a1, b1, a2, b2 in remaining:
                if a2 < x1 or x2 < a1 or b2 < y1 or y2 < b1:
                    keep.append((a1, b1, a2, b2))
                    continue
                x1, y1 = min(x1, a1), min(y1, b1)
                x2, y2 = max(x2, a2), max(y2, b2)
                changed = True
            remaining = keep
        merged.append((x1, y1, x2, y2))
    return merged


def person_blur_regions(mask: np.ndarray, source_width: int,
                        source_height: int) -> list[tuple[int, int, int, int]]:
    """Return padded full-resolution work regions for all mask components."""
    if mask.ndim != 2:
        raise ValueError("person mask must be HxW")
    if not np.any(mask):
        return []
    scan_height, scan_width = mask.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    sigma = max(10, source_width // 250)
    # Four sigmas plus the legacy dilation and feather radii makes every
    # component's alpha effectively zero at a non-image crop edge.
    full_padding = 4 * sigma + 10 + 20
    x_scale = source_width / scan_width
    y_scale = source_height / scan_height
    boxes = []
    for component in range(1, count):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        width = int(stats[component, cv2.CC_STAT_WIDTH])
        height = int(stats[component, cv2.CC_STAT_HEIGHT])
        x1 = max(0, math.floor(x * x_scale) - full_padding)
        y1 = max(0, math.floor(y * y_scale) - full_padding)
        x2 = min(source_width, math.ceil((x + width) * x_scale) + full_padding)
        y2 = min(source_height, math.ceil((y + height) * y_scale) + full_padding)
        boxes.append((x1, y1, x2, y2))
    return _merge_pixel_boxes(boxes)


def apply_person_blur(frame: np.ndarray, mask: np.ndarray):
    """Apply the approved 21/41/sigma legacy blur inside padded mask ROIs."""
    source_height, source_width = frame.shape[:2]
    scan_height, scan_width = mask.shape
    sigma = max(10, source_width // 250)
    for x1, y1, x2, y2 in person_blur_regions(
            mask, source_width, source_height):
        scan_x1 = max(0, math.floor(x1 * scan_width / source_width))
        scan_y1 = max(0, math.floor(y1 * scan_height / source_height))
        scan_x2 = min(scan_width, math.ceil(x2 * scan_width / source_width))
        scan_y2 = min(scan_height, math.ceil(y2 * scan_height / source_height))
        if scan_x2 <= scan_x1 or scan_y2 <= scan_y1:
            continue
        # Align the full-resolution ROI to the selected scan pixels so nearest
        # upsampling has deterministic boundaries.
        full_x1 = max(0, math.floor(scan_x1 * source_width / scan_width))
        full_y1 = max(0, math.floor(scan_y1 * source_height / scan_height))
        full_x2 = min(source_width, math.ceil(scan_x2 * source_width / scan_width))
        full_y2 = min(source_height, math.ceil(scan_y2 * source_height / scan_height))
        region = frame[full_y1:full_y2, full_x1:full_x2]
        region_mask = cv2.resize(
            mask[scan_y1:scan_y2, scan_x1:scan_x2].astype(np.uint8),
            (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
        expanded = cv2.dilate(
            region_mask, np.ones((21, 21), dtype=np.uint8))
        feather = cv2.GaussianBlur(
            expanded.astype(np.float32), (41, 41), 0)[..., None]
        blurred = cv2.GaussianBlur(
            region, (0, 0), sigmaX=sigma, sigmaY=sigma)
        region[:] = np.clip(
            region.astype(np.float32) * (1.0 - feather)
            + blurred.astype(np.float32) * feather,
            0, 255).astype(np.uint8)


def _draw_render_review(frame: np.ndarray, direct_mask: np.ndarray,
                        temporal_mask: np.ndarray, review_mask: np.ndarray,
                        row: dict, review_width: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    review_height = round(review_width * source_height / source_width)
    review = cv2.resize(
        frame, (review_width, review_height), interpolation=cv2.INTER_AREA)
    direct = person_segmentation_preview._resize_mask(
        direct_mask, review_width, review_height)
    temporal = person_segmentation_preview._resize_mask(
        temporal_mask, review_width, review_height)
    suspicious = person_segmentation_preview._resize_mask(
        review_mask, review_width, review_height)
    cv2.drawContours(
        review, person_segmentation_preview._mask_contours(direct), -1,
        (50, 220, 50), 2, cv2.LINE_AA)
    cv2.drawContours(
        review, person_segmentation_preview._mask_contours(temporal), -1,
        (255, 0, 255), 3, cv2.LINE_AA)
    cv2.drawContours(
        review, person_segmentation_preview._mask_contours(suspicious), -1,
        (0, 100, 255), 3, cv2.LINE_AA)
    for detection in row["detections"]:
        x1, y1, x2, y2 = anonymize_video._pixel_bounds(
            detection["box"], review_width, review_height)
        color = ((0, 215, 255) if detection["category"] == "license_plate"
                 else (255, 255, 0))
        cv2.rectangle(review, (x1, y1), (x2, y2), color, 3)
    for candidate in row.get("review", {}).get("plate_candidates", []):
        x1, y1, x2, y2 = anonymize_video._pixel_bounds(
            candidate["box"], review_width, review_height)
        cv2.rectangle(review, (x1, y1), (x2, y2), (0, 100, 255), 2)
    cv2.rectangle(review, (0, 0), (min(review_width, 1520), 54), (0, 0, 0), -1)
    text = (
        f"OUTPUT {row['video_t_s']:.3f}s FRAME {row['frame_index']}  |  "
        f"SOURCE {row['source_video_t_s']:.3f}s FRAME "
        f"{row['source_frame_index']}  |  green=direct magenta=temporal "
        "yellow=plate orange=review")
    cv2.putText(review, text, (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return review


def _chunk_ranges(frame_count: int, chunk_frames: int) -> list[tuple[int, int]]:
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    return [(start, min(frame_count, start + chunk_frames))
            for start in range(0, frame_count, chunk_frames)]


def _chunk_name(start: int, end: int) -> str:
    return f"chunk_{start:08d}_{end:08d}"


def _validate_video_contract(path: Path, *, width: int, height: int,
                             fps: float, frames: int) -> dict:
    info = anonymize_video.probe_video(path)
    if (info["width"], info["height"]) != (width, height):
        raise RuntimeError(f"video resolution changed: {path}")
    if not math.isclose(info["media_fps"], fps, abs_tol=1e-9):
        raise RuntimeError(f"video frame rate changed: {path}")
    if info["nb_frames"] != frames:
        raise RuntimeError(
            f"video has {info['nb_frames']} frames, expected {frames}: {path}")
    return info


def _validate_render_chunk(path: Path, spec: dict, start: int,
                           end: int) -> dict:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"render chunk must be a real directory: {path}")
    manifest_path = path / "chunk_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"missing chunk manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("start_frame") != start
            or manifest.get("end_frame_exclusive") != end
            or manifest.get("render_spec_sha256")
            != hashlib.sha256(_json_bytes(spec)).hexdigest()):
        raise ValueError(f"render chunk contract changed: {path}")
    for key in ("full_video", "review_video"):
        item = manifest[key]
        video = path / item["path"]
        if (video.is_symlink() or not video.is_file()
                or anonymize_video.sha256_file(video) != item["sha256"]):
            raise ValueError(f"render chunk file changed: {video}")
    expected_extracted = {
        name for index, name in (spec.get("extraction") or {}).get(
            "frames", {}).items() if start <= int(index) < end}
    extracted = manifest.get("extracted_frame_sha256", {})
    if set(extracted) != expected_extracted:
        raise ValueError(f"render chunk extraction set changed: {path}")
    for name, expected_sha in extracted.items():
        frame = path / "frames" / name
        if (frame.is_symlink() or not frame.is_file()
                or anonymize_video.sha256_file(frame) != expected_sha):
            raise ValueError(f"render chunk frame changed: {frame}")
    return manifest


def _render_chunk(stage: Path, spec: dict, policy_dir: Path,
                  policy_manifest: dict, rows: list[dict],
                  start: int, end: int):
    chunks_dir = stage / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    final = chunks_dir / _chunk_name(start, end)
    if final.exists() or final.is_symlink():
        _validate_render_chunk(final, spec, start, end)
        return
    incomplete = final.with_name(final.name + ".incomplete")
    if incomplete.exists() or incomplete.is_symlink():
        if incomplete.is_symlink() or not incomplete.is_dir():
            raise ValueError(f"unexpected incomplete render chunk: {incomplete}")
        failed = stage / "failed_chunks"
        failed.mkdir(exist_ok=True)
        destination = failed / (
            incomplete.name + "." + datetime.datetime.now(
                datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        os.rename(incomplete, destination)
    incomplete.mkdir()
    full_path = incomplete / "full.mp4"
    review_path = incomplete / "review.mp4"
    full_writer = None
    review_writer = None
    source_info = {key: policy_manifest["source"][key] for key in (
        "codec", "width", "height", "pix_fmt", "media_fps",
        "media_fps_fraction", "nb_frames", "duration_s", "size_bytes",
        "color_space", "color_transfer", "color_primaries")}
    source = Path(policy_manifest["source"]["path"])
    fps = float(policy_manifest["output_fps"])
    source_start = int(policy_manifest["source"]["clip"]["start_frame"])
    scan_dir = Path(policy_manifest["scan"]["path"])
    scan_manifest = json.loads((scan_dir / "scan_manifest.json").read_text())
    scan_frames = scan_dir / scan_manifest["evidence"]["directory"]
    policy_frames = policy_dir / policy_manifest["evidence"]["directory"]
    scan_width, scan_height = policy_manifest["scan_resolution"]
    review_width = spec["render"]["review_width"]
    review_height = round(
        review_width * source_info["height"] / source_info["width"])
    try:
        full_writer = anonymize_video.RawVideoWriter(
            full_path, source_info["width"], source_info["height"], fps,
            encoder=spec["render"]["encoder"]["backend"])
        review_writer = anonymize_video.RawVideoWriter(
            review_path, review_width, review_height,
            fps * spec["render"]["review_speedup"], review=True,
            encoder=spec["render"]["encoder"]["backend"])
        rendered = 0
        extracted_frame_sha256 = {}
        extraction_frames = (spec.get("extraction") or {}).get("frames", {})
        for offset, frame in enumerate(anonymize_video.RawVideoReader(
                source, source_info, fps,
                start_frame=source_start + start,
                end_frame=source_start + end,
                cfr_fast_seek=spec["render"]["decoder"]["enabled"])):
            index = start + offset
            scan_path = scan_frames / _evidence_name(index)
            policy_path = policy_frames / _evidence_name(index)
            expected_person = rows[index]["person_masks"]
            if (anonymize_video.sha256_file(scan_path)
                    != expected_person["scan_evidence_sha256"]
                    or anonymize_video.sha256_file(policy_path)
                    != expected_person["policy_evidence_sha256"]):
                raise ValueError(f"mask evidence changed at frame {index}")
            scan = load_scan_evidence(
                scan_path,
                (scan_height, scan_width), masks=True)
            policy = load_policy_evidence(
                policy_path,
                (scan_height, scan_width))
            applied = scan["direct_mask"] | policy["temporal_mask"]
            apply_person_blur(frame, applied)
            anonymize_video.strong_blur(frame, rows[index]["detections"])
            frame_name = extraction_frames.get(str(index))
            if frame_name is not None:
                frame_dir = incomplete / "frames"
                frame_dir.mkdir(exist_ok=True)
                destination = frame_dir / frame_name
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(
                        f"refusing to replace extracted frame: {destination}")
                if not cv2.imwrite(
                        str(destination), frame,
                        [cv2.IMWRITE_JPEG_QUALITY,
                         spec["extraction"]["jpeg_quality"]]):
                    raise RuntimeError(f"could not write {destination}")
                extracted_frame_sha256[frame_name] = (
                    anonymize_video.sha256_file(destination))
            full_writer.write(frame)
            review_writer.write(_draw_render_review(
                frame, scan["direct_mask"], policy["temporal_mask"],
                policy["review_display_mask"], rows[index], review_width))
            rendered += 1
        if rendered != end - start:
            raise RuntimeError(
                f"chunk decoded {rendered}, expected {end - start}")
        full_writer.close()
        full_writer = None
        review_writer.close()
        review_writer = None
        full_info = _validate_video_contract(
            full_path, width=source_info["width"], height=source_info["height"],
            fps=fps, frames=end - start)
        review_info = _validate_video_contract(
            review_path, width=review_width, height=review_height,
            fps=fps * spec["render"]["review_speedup"], frames=end - start)
        chunk_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "person_mask_render_chunk",
            "render_spec_sha256": hashlib.sha256(_json_bytes(spec)).hexdigest(),
            "start_frame": start,
            "end_frame_exclusive": end,
            "frame_count": end - start,
            "full_video": {
                "path": full_path.name,
                "sha256": anonymize_video.sha256_file(full_path),
                "info": full_info,
            },
            "review_video": {
                "path": review_path.name,
                "sha256": anonymize_video.sha256_file(review_path),
                "info": review_info,
            },
            "extracted_frame_sha256": extracted_frame_sha256,
            "completed_utc": _utc_now(),
        }
        _write_new_atomic(
            incomplete / "chunk_manifest.json", _json_bytes(chunk_manifest))
        os.rename(incomplete, final)
    except BaseException:
        if full_writer is not None:
            full_writer.abort()
        if review_writer is not None:
            review_writer.abort()
        raise


def _publish_extracted_frames(stage: Path, spec: dict,
                              frames_dir: Path | None) -> dict[str, str]:
    """Publish the hash-validated JPEG set after all chunks are complete."""
    extraction = spec.get("extraction")
    if extraction is None:
        if frames_dir is not None:
            raise ValueError("frames_dir supplied without an extraction plan")
        return {}
    if frames_dir is None:
        raise ValueError("extraction plan requires frames_dir")
    expected_names = set(extraction["frames"].values())
    hashes = {}
    for start, end in map(tuple, spec["render"]["chunk_ranges"]):
        chunk = stage / "chunks" / _chunk_name(start, end)
        manifest = _validate_render_chunk(chunk, spec, start, end)
        for name, digest in manifest["extracted_frame_sha256"].items():
            if name in hashes:
                raise ValueError(f"duplicate extracted frame name: {name}")
            hashes[name] = digest
    if set(hashes) != expected_names:
        raise ValueError("render chunks do not contain the complete extraction plan")

    def validate_directory(path: Path):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(
                f"extracted frames path must be a real directory: {path}")
        entries = list(path.iterdir())
        if any(item.is_symlink() or not item.is_file() for item in entries):
            raise ValueError(f"unexpected entry in extracted frames path: {path}")
        if {item.name for item in entries} != expected_names:
            raise ValueError(f"extracted frame set changed: {path}")
        for item in entries:
            if anonymize_video.sha256_file(item) != hashes[item.name]:
                raise ValueError(f"extracted frame changed: {item}")

    if frames_dir.exists() or frames_dir.is_symlink():
        validate_directory(frames_dir)
        return hashes
    frames_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = frames_dir.with_name(frames_dir.name + ".incomplete")
    if staging.exists() or staging.is_symlink():
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError(f"invalid extracted-frame staging path: {staging}")
    else:
        staging.mkdir()
    for start, end in map(tuple, spec["render"]["chunk_ranges"]):
        source_dir = stage / "chunks" / _chunk_name(start, end) / "frames"
        if not source_dir.exists():
            continue
        for source in source_dir.iterdir():
            destination = staging / source.name
            if destination.exists() or destination.is_symlink():
                if (destination.is_symlink() or not destination.is_file()
                        or anonymize_video.sha256_file(destination)
                        != hashes[source.name]):
                    raise ValueError(
                        f"changed partial extracted frame: {destination}")
                continue
            # Keep published dataset frames inode-independent from retained
            # chunk evidence so an in-place edit cannot corrupt both copies.
            shutil.copyfile(source, destination)
            if anonymize_video.sha256_file(destination) != hashes[source.name]:
                raise RuntimeError(f"copied extracted frame changed: {destination}")
    validate_directory(staging)
    if frames_dir.exists() or frames_dir.is_symlink():
        validate_directory(frames_dir)
    else:
        os.rename(staging, frames_dir)
    return hashes


def _archive_concat_artifacts(stage: Path, kind: str, paths: list[Path],
                              reason: str):
    """Quarantine only explicit stage-owned concat artifacts for diagnosis."""
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if not existing:
        return
    for path in existing:
        if not (path.is_symlink() or path.is_file()):
            raise ValueError(f"unexpected concat artifact: {path}")
    failed_root = stage / "failed_concats"
    failed_root.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ")
    destination = failed_root / f"{kind}.{stamp}"
    destination.mkdir()
    for path in existing:
        os.rename(path, destination / path.name)
    _write_new_atomic(destination / "reason.txt", (reason + "\n").encode())


def _concat_chunks(stage: Path, ranges: list[tuple[int, int]], kind: str,
                   output: Path, spec: dict) -> dict:
    if kind not in {"full", "review"}:
        raise ValueError(f"unsupported concat kind: {kind}")
    spec_sha = hashlib.sha256(_json_bytes(spec)).hexdigest()
    inputs = []
    lines = []
    for start, end in ranges:
        chunk = stage / "chunks" / _chunk_name(start, end)
        manifest = _validate_render_chunk(chunk, spec, start, end)
        item = manifest[f"{kind}_video"]
        video = chunk / item["path"]
        relative_video = video.relative_to(stage)
        if "'" in str(relative_video):
            raise ValueError("single quotes are unsupported in concat paths")
        inputs.append({
            "chunk": chunk.name,
            "start_frame": start,
            "end_frame_exclusive": end,
            "path": str(video.relative_to(stage)),
            "sha256": item["sha256"],
            "chunk_manifest_sha256": anonymize_video.sha256_file(
                chunk / "chunk_manifest.json"),
        })
        lines.append(f"file '{relative_video}'\n")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": f"person_mask_{kind}_concat",
        "render_spec_sha256": spec_sha,
        "output_name": output.name,
        "inputs": inputs,
    }
    sidecar = stage / f"concat_{kind}.json"
    incomplete = output.with_name(output.name + ".incomplete.mp4")
    reusable = False
    if ((output.exists() or output.is_symlink())
            and (sidecar.exists() or sidecar.is_symlink())):
        if (not output.is_symlink() and output.is_file()
                and not sidecar.is_symlink() and sidecar.is_file()):
            try:
                existing = json.loads(sidecar.read_text())
            except (OSError, json.JSONDecodeError):
                existing = None
            if (isinstance(existing, dict)
                    and all(existing.get(key) == value
                            for key, value in contract.items())
                    and existing.get("output_sha256")
                    == anonymize_video.sha256_file(output)):
                reusable = True
    if reusable:
        return existing
    if any(path.exists() or path.is_symlink()
           for path in (output, sidecar, incomplete)):
        _archive_concat_artifacts(
            stage, kind, [output, sidecar, incomplete],
            "orphaned or contract-mismatched concat artifact")

    list_path = stage / f"concat_{kind}.txt"
    _write_or_validate(list_path, "".join(lines).encode())
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat",
        "-safe", "0", "-i", str(list_path), "-map", "0:v:0", "-c", "copy",
        "-movflags", "+faststart", str(incomplete),
    ], check=True)
    try:
        os.link(incomplete, output)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace output: {output}") from error
    finally:
        incomplete.unlink(missing_ok=True)
    source = spec["source"]
    if kind == "full":
        width, height = source["width"], source["height"]
        fps = float(spec["output_fps"])
    else:
        width = int(spec["render"]["review_width"])
        height = round(width * source["height"] / source["width"])
        fps = float(spec["output_fps"]) * float(
            spec["render"]["review_speedup"])
    info = _validate_video_contract(
        output, width=width, height=height, fps=fps,
        frames=int(spec["frame_count"]))
    result = {
        **contract,
        "output_sha256": anonymize_video.sha256_file(output),
        "output_info": info,
        "completed_utc": _utc_now(),
    }
    _write_new_atomic(sidecar, _json_bytes(result))
    return result


def _write_render_review_html(path: Path, *, full_name: str,
                              review_name: str, ledger_name: str,
                              clip_start_s: float, output_fps: float,
                              review_speedup: float,
                              source_start_frame: int,
                              source_width: int, source_height: int,
                              frame_count: int,
                              flagged_indices: list[int]):
    document = r"""<!doctype html>
<meta charset="utf-8"><title>Anonymization review</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#181818;color:#eee}
a{color:#7ec8ff} #review{display:block;width:100%;background:#000;margin:1rem 0}
#nativePane{width:100%;height:72vh;overflow:auto;border:2px solid #777;background:#000}
#nativeStage{position:relative;width:__WIDTH__px;height:__HEIGHT__px}
#native{display:block;width:__WIDTH__px;height:__HEIGHT__px;max-width:none;background:#000}
.bad{color:#ff8c69;font-weight:700}button{margin:.3rem;padding:.5rem .8rem}
code{background:#333;padding:.15rem .3rem}textarea{width:100%;height:14rem;background:#101010;color:#eee}
</style>
<h1>Anonymization review</h1>
<p><strong>Pending human review.</strong> Watch the accelerated overview end to
end. Green outlines are direct person masks, magenta is conservative temporal
fill, yellow is an automatically accepted plate, and orange is unresolved
evidence. Look for any identifiable person or plate without an applied blur.</p>
<video id="review" controls preload="metadata"></video>
<p id="time"></p><button id="previous">Previous flagged frame</button>
<button id="next">Next flagged frame</button><span id="flagStatus"></span>
<a id="ledgerLink">Open the complete QA ledger</a>
<h2>Native-resolution inspector (required)</h2>
<p>Pause on crowded, distant, upper/lower, or ambiguous scenes, then use
<em>Sync native view</em>. The scrollable player below is exactly one source
pixel per CSS pixel (__WIDTH__×__HEIGHT__). If HEVC playback is unavailable,
<a id="fullLink">open/download the full result</a> in an HEVC-capable player.</p>
<p><button id="sync">Sync native view</button>
<button id="back">−1 frame</button><button id="forward">+1 frame</button>
<span id="nativeTime"></span></p>
<div id="nativePane"><div id="nativeStage">
<video id="native" controls preload="metadata"></video>
</div></div>
<h2>Capture a correction</h2>
<p>Pause the native player, then <strong>shift-click</strong> two opposite corners
around an unblurred face, person, or plate. Download the generated JSON and use
it as <code>--manual_regions</code> for a new policy/render revision. The default
time span is the selected 3 fps frame; expand it when the subject persists.</p>
<button id="clear">Clear corrections</button><button id="download">Download corrections JSON</button>
<p id="correctionStatus"></p><textarea id="corrections" readonly>[]</textarea>
<script>
const review=document.getElementById('review'), native=document.getElementById('native');
const fps=__FPS__, speed=__SPEED__, clipStart=__CLIP_START__, sourceStart=__SOURCE_START__;
const frameCount=__FRAME_COUNT__, fullWidth=__WIDTH__, fullHeight=__HEIGHT__;
const flagged=__FLAGGED__, corrections=[];
review.src=__REVIEW__; native.src=__FULL__;
document.getElementById('fullLink').href=__FULL__;
document.getElementById('ledgerLink').href=__LEDGER__;
document.getElementById('flagStatus').textContent=` ${flagged.length} flagged frames · `;
function clampIndex(i){return Math.max(0,Math.min(frameCount-1,i));}
function reviewIndex(){return clampIndex(Math.floor(review.currentTime*fps*speed+1e-9));}
function nativeIndex(){return clampIndex(Math.floor(native.currentTime*fps+1e-9));}
function showTime(){const i=reviewIndex(); document.getElementById('time').textContent=
  `Output ${(i/fps).toFixed(3)} s · source ${(clipStart+i/fps).toFixed(3)} s · source-grid frame ${sourceStart+i}`;}
review.addEventListener('timeupdate',showTime);
function syncNative(){native.pause();native.currentTime=reviewIndex()/fps;showNativeTime();}
document.getElementById('sync').onclick=syncNative;
function jump(direction){if(!flagged.length)return;const current=reviewIndex();
 let candidates=flagged.filter(i=>direction>0?i>current:i<current);let target=direction>0?candidates[0]:candidates.at(-1);
 if(target===undefined)target=direction>0?flagged[0]:flagged.at(-1);review.currentTime=target/(fps*speed);}
document.getElementById('previous').onclick=()=>jump(-1);document.getElementById('next').onclick=()=>jump(1);
function showNativeTime(){const i=nativeIndex();document.getElementById('nativeTime').textContent=
 ` output ${(i/fps).toFixed(3)} s · source ${(clipStart+i/fps).toFixed(3)} s · source-grid frame ${sourceStart+i}`;}
native.addEventListener('timeupdate',showNativeTime);
document.getElementById('back').onclick=()=>{native.pause();native.currentTime=clampIndex(nativeIndex()-1)/fps;};
document.getElementById('forward').onclick=()=>{native.pause();native.currentTime=clampIndex(nativeIndex()+1)/fps;};
function pointOnVideo(e){const r=native.getBoundingClientRect();return [
 Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),
 Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
let firstCorner=null;
native.addEventListener('click',e=>{if(!e.shiftKey)return;e.preventDefault();native.pause();
 const p=pointOnVideo(e), status=document.getElementById('correctionStatus');
 if(firstCorner===null){firstCorner=p;status.textContent='First corner saved; shift-click the opposite corner.';return;}
 const i=nativeIndex(), box=[Math.min(firstCorner[0],p[0]),Math.min(firstCorner[1],p[1]),
  Math.max(firstCorner[0],p[0]),Math.max(firstCorner[1],p[1])].map(x=>Number(x.toFixed(7)));
 if(box[0]>=box[2]||box[1]>=box[3]){firstCorner=null;
  status.textContent='Correction has zero area; select two distinct corners.';return;}
 const center=clipStart+i/fps, halfWindow=0.25/fps;
 const start=Math.max(clipStart,center-halfWindow);
 const end=Math.min(clipStart+frameCount/fps,center+halfWindow);
 corrections.push({id:`review_correction_${sourceStart+i}`,category:'privacy',box,
  start_s:Number(start.toFixed(9)),end_s:Number(end.toFixed(9)),
  reason:'human review correction'});
 document.getElementById('corrections').value=JSON.stringify(corrections,null,2);
 status.textContent=`Saved correction ${corrections.length} for source-grid frame ${sourceStart+i}.`;
 firstCorner=null;});
document.getElementById('clear').onclick=()=>{corrections.length=0;firstCorner=null;
 document.getElementById('corrections').value='[]';document.getElementById('correctionStatus').textContent='Cleared.';};
document.getElementById('download').onclick=()=>{const blob=new Blob(
 [JSON.stringify(corrections,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download='manual_regions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),0);};
showTime();showNativeTime();
</script>
"""
    replacements = {
        "__REVIEW__": json.dumps(review_name),
        "__FULL__": json.dumps(full_name),
        "__LEDGER__": json.dumps(ledger_name),
        "__FPS__": f"{output_fps:.12g}",
        "__SPEED__": f"{review_speedup:.12g}",
        "__CLIP_START__": f"{clip_start_s:.12g}",
        "__SOURCE_START__": str(source_start_frame),
        "__WIDTH__": str(source_width),
        "__HEIGHT__": str(source_height),
        "__FRAME_COUNT__": str(frame_count),
        "__FLAGGED__": json.dumps(flagged_indices, separators=(",", ":")),
    }
    for marker, value in replacements.items():
        document = document.replace(marker, value)
    _write_or_validate(path, document.encode())


def render(args) -> int:
    policy_dir = args.policy_dir.resolve()
    policy_manifest, policy_manifest_path, policy_manifest_sha, rows = (
        _load_policy(policy_dir))
    source = Path(policy_manifest["source"]["path"])
    if (source.is_symlink() or not source.is_file()
            or anonymize_video.sha256_file(source)
            != policy_manifest["source"]["sha256"]):
        raise ValueError("render source is missing or changed")
    source_info = {key: policy_manifest["source"][key] for key in (
        "codec", "width", "height", "pix_fmt", "media_fps",
        "media_fps_fraction", "nb_frames", "duration_s", "size_bytes",
        "color_space", "color_transfer", "color_primaries")}
    fps = float(policy_manifest["output_fps"])
    frame_count = int(policy_manifest["frame_count"])
    review_height = round(
        args.review_width * source_info["height"] / source_info["width"])
    if (args.review_width < 640 or args.review_width % 2
            or review_height % 2
            or not math.isfinite(args.review_speedup)
            or args.review_speedup <= 0
            or not 1 <= args.jpeg_quality <= 100):
        raise ValueError("review width/speedup is invalid")
    ranges = _chunk_ranges(frame_count, args.chunk_frames)
    if (args.extraction_plan is None) != (args.frames_dir is None):
        raise ValueError(
            "extraction_plan and frames_dir must be supplied together")
    extraction = None
    if args.extraction_plan is not None:
        plan_hash_before = anonymize_video.sha256_file(args.extraction_plan)
        plan = anonymize_video.load_extraction_plan(args.extraction_plan)
        plan_hash_after = anonymize_video.sha256_file(args.extraction_plan)
        if plan_hash_before != plan_hash_after:
            raise ValueError("extraction plan changed while it was loaded")
        missing = sorted(set(plan) - set(range(frame_count)))
        if missing:
            raise ValueError(f"extraction indexes outside clip: {missing[:10]}")
        extraction = {
            "plan": str(args.extraction_plan.resolve()),
            "plan_sha256": plan_hash_after,
            "frames_dir": str(args.frames_dir.resolve()),
            "jpeg_quality": args.jpeg_quality,
            "frames": {
                str(index): name for index, name in sorted(plan.items())},
        }
    decoder = {
        "mode": "ffmpeg_filter_trim_from_stream_start",
        "enabled": False,
    }
    if args.cfr_fast_seek:
        fps_integer = anonymize_video.cfr_fast_seek_fps(source_info, fps)
        _, selection_step = anonymize_video.frame_selection(
            source_info["media_fps"], fps)
        decoder = {
            "mode": "ffmpeg_input_accurate_seek_cfr",
            "enabled": True,
            "fps_integer": fps_integer,
            "media_fps_integer": round(float(source_info["media_fps"])),
            "selection_step": selection_step,
            "eligibility": (
                "operator asserts CFR; integer media fps is an exact multiple "
                "of integer output fps"),
            "seek_anchor": (
                "floor(global_start_frame/fps) whole seconds using integer math"),
            "post_seek": "frame-index trim relative to whole-second anchor",
        }
    spec = {
        "schema_version": SCHEMA_VERSION,
        "kind": "person_mask_render_spec",
        "implementation": _stage_implementation(
            "person_anonymize_video.render.v3", include_persistence=False),
        "runtime_versions": _runtime_versions(),
        "policy": {
            "path": str(policy_dir),
            "manifest": str(policy_manifest_path),
            "manifest_sha256": policy_manifest_sha,
            "applied_ledger_sha256": policy_manifest["files"][
                "applied_ledger"]["sha256"],
            "evidence_tree": policy_manifest["evidence"]["tree"],
        },
        "source": policy_manifest["source"],
        "output_fps": fps,
        "frame_count": frame_count,
        "render": {
            "output_name": args.output_name,
            "review_name": "review.mp4",
            "review_width": args.review_width,
            "review_speedup": args.review_speedup,
            "chunk_frames": args.chunk_frames,
            "chunk_ranges": [list(item) for item in ranges],
            "decoder": decoder,
            "encoder": anonymize_video.video_encoder_profile(args.encoder),
            "person_blur": {
                "mask": "direct union accepted temporal mask",
                "upsample": "nearest",
                "dilation_kernel": [21, 21],
                "feather_kernel": [41, 41],
                "gaussian_sigma": max(10, source_info["width"] // 250),
                "implementation": "component ROI, four-sigma padded",
            },
            "plate_manual_blur": (
                "expanded rectangular Gaussian-smoothed coarse mosaic"),
        },
        "extraction": extraction,
    }
    if (Path(args.output_name).name != args.output_name
            or not re.fullmatch(r"[A-Za-z0-9._-]+\.mp4", args.output_name)
            or args.output_name == "review.mp4"):
        raise ValueError("output_name must be an MP4 basename")
    output_dir = args.output_dir.resolve()
    started = time.monotonic()
    with _locked_named_stage(
            output_dir, spec, spec_name="render_spec.json",
            lock_name=".render.lock") as (stage, lock_path):
        for chunk_index, (start, end) in enumerate(ranges, 1):
            chunk = stage / "chunks" / _chunk_name(start, end)
            if chunk.exists() and not chunk.is_symlink():
                _validate_render_chunk(chunk, spec, start, end)
                print(f"render chunk {chunk_index}/{len(ranges)} already committed",
                      flush=True)
                continue
            print(f"rendering chunk {chunk_index}/{len(ranges)} "
                  f"frames [{start}, {end})", flush=True)
            _render_chunk(
                stage, spec, policy_dir, policy_manifest, rows, start, end)

        full_path = stage / args.output_name
        review_path = stage / "review.mp4"
        full_concat = _concat_chunks(
            stage, ranges, "full", full_path, spec)
        review_concat = _concat_chunks(
            stage, ranges, "review", review_path, spec)
        full_info = _validate_video_contract(
            full_path, width=source_info["width"], height=source_info["height"],
            fps=fps, frames=frame_count)
        review_info = _validate_video_contract(
            review_path, width=args.review_width, height=review_height,
            fps=fps * args.review_speedup, frames=frame_count)
        if (args.extraction_plan is not None
                and anonymize_video.sha256_file(args.extraction_plan)
                != extraction["plan_sha256"]):
            raise ValueError("extraction plan changed during render")

        _verify_regular_sha256(
            source, spec["source"]["sha256"], "render source video")
        _verify_regular_sha256(
            policy_manifest_path, spec["policy"]["manifest_sha256"],
            "render parent policy manifest")
        _verify_regular_sha256(
            Path(policy_manifest["scan"]["manifest"]),
            policy_manifest["scan"]["manifest_sha256"],
            "render parent scan manifest")
        for item in policy_manifest["files"].values():
            _verify_regular_sha256(
                policy_dir / item["path"], item["sha256"],
                "render parent policy file")
        policy_frames = policy_dir / policy_manifest["evidence"]["directory"]
        scan_parent_dir = Path(policy_manifest["scan"]["path"])
        scan_parent_manifest = json.loads(
            (scan_parent_dir / "scan_manifest.json").read_text())
        scan_frames = scan_parent_dir / scan_parent_manifest[
            "evidence"]["directory"]
        for index, row in enumerate(rows):
            masks = row["person_masks"]
            _verify_regular_sha256(
                policy_frames / _evidence_name(index),
                masks["policy_evidence_sha256"], "render policy evidence")
            _verify_regular_sha256(
                scan_frames / _evidence_name(index),
                masks["scan_evidence_sha256"], "render scan evidence")
        if (_stage_implementation(
                "person_anonymize_video.render.v3",
                include_persistence=False) != spec["implementation"]
                or _runtime_versions() != spec["runtime_versions"]):
            raise ValueError(
                "render implementation or runtime changed during processing")
        extracted_frame_sha256 = _publish_extracted_frames(
            stage, spec, args.frames_dir)

        source_ledger = policy_dir / policy_manifest["files"]["applied_ledger"]["path"]
        ledger_path = stage / "detections.jsonl"
        ledger_bytes = anonymize_video.read_regular_file_bytes(source_ledger)
        _write_or_validate(ledger_path, ledger_bytes)
        retained_files = {
            "applied_ledger": {
                "path": ledger_path.name,
                "sha256": anonymize_video.sha256_file(ledger_path),
            },
        }
        for key, item in policy_manifest["files"].items():
            if key == "applied_ledger":
                continue
            source_file = policy_dir / item["path"]
            destination = stage / Path(item["path"]).name
            payload = anonymize_video.read_regular_file_bytes(source_file)
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"policy file changed before render: {source_file}")
            _write_or_validate(destination, payload)
            retained_files[key] = {
                "path": destination.name,
                "sha256": anonymize_video.sha256_file(destination),
            }
        review_html = stage / "review.html"
        flagged_indices = [
            int(row["frame_index"]) for row in rows
            if (int(row.get("review", {}).get(
                    "person_suspicion_pixels", 0)) > 0
                or bool(row.get("review", {}).get("plate_candidates", [])))
        ]
        _write_render_review_html(
            review_html, full_name=full_path.name,
            review_name=review_path.name, ledger_name=ledger_path.name,
            clip_start_s=float(policy_manifest["source"]["clip"]["start_s"]),
            output_fps=fps, review_speedup=args.review_speedup,
            source_start_frame=int(
                policy_manifest["source"]["clip"]["start_frame"]),
            source_width=source_info["width"],
            source_height=source_info["height"], frame_count=frame_count,
            flagged_indices=flagged_indices)
        _verify_regular_sha256(
            source, spec["source"]["sha256"], "render source video")
        _verify_regular_sha256(
            policy_manifest_path, spec["policy"]["manifest_sha256"],
            "render parent policy manifest")
        if (_stage_implementation(
                "person_anonymize_video.render.v3",
                include_persistence=False) != spec["implementation"]
                or _runtime_versions() != spec["runtime_versions"]):
            raise ValueError(
                "render implementation or runtime changed before publication")
        render_completed_utc = _utc_now()
        manifest = {
            **policy_manifest,
            "kind": "person_mask_anonymization_render",
            "status": "rendered_pending_review",
            "completed_utc": render_completed_utc,
            "policy_parent": {
                "manifest": str(policy_manifest_path),
                "manifest_sha256": policy_manifest_sha,
            },
            "evidence": {
                **policy_manifest["evidence"],
                "directory": str((
                    policy_dir / policy_manifest["evidence"]["directory"]
                ).resolve()),
            },
            "files": retained_files,
            "review": {
                **policy_manifest["review"],
                "status": "pending",
                "video": review_path.name,
                "html": review_html.name,
                "speedup": args.review_speedup,
                "overview_width": args.review_width,
                "native_resolution_inspector": True,
            },
            "render": {
                "output_video": str((output_dir / full_path.name).resolve()),
                "output_video_sha256": anonymize_video.sha256_file(full_path),
                "output_video_info": full_info,
                "review_video_sha256": anonymize_video.sha256_file(review_path),
                "review_video_info": review_info,
                "review_html_sha256": anonymize_video.sha256_file(review_html),
                "concat_manifests": {
                    "full": {
                        "path": "concat_full.json",
                        "sha256": anonymize_video.sha256_file(
                            stage / "concat_full.json"),
                        "output_sha256": full_concat["output_sha256"],
                    },
                    "review": {
                        "path": "concat_review.json",
                        "sha256": anonymize_video.sha256_file(
                            stage / "concat_review.json"),
                        "output_sha256": review_concat["output_sha256"],
                    },
                },
                "chunk_count": len(ranges),
                "chunk_frames": args.chunk_frames,
                "decoder": spec["render"]["decoder"],
                "encoder": spec["render"]["encoder"],
                "render_spec_sha256": hashlib.sha256(_json_bytes(spec)).hexdigest(),
                "blur": (
                    "mask-shaped person Gaussian plus direct-frame "
                    "vehicle-validated plate mosaic"),
                "extraction_plan": (
                    str(args.extraction_plan.resolve())
                    if args.extraction_plan else None),
                "extraction_plan_sha256": (
                    extraction["plan_sha256"] if extraction else None),
                "extracted_frames_dir": (
                    str(args.frames_dir.resolve()) if args.frames_dir else None),
                "extracted_frames": len(extracted_frame_sha256),
                "extracted_frame_sha256": extracted_frame_sha256,
                "created_utc": render_completed_utc,
                "elapsed_s": round(time.monotonic() - started, 3),
                "argv": list(sys.argv),
            },
        }
        manifest_path = stage / "anonymization_manifest.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            if manifest_path.is_symlink() or not manifest_path.is_file():
                raise ValueError(f"invalid render manifest: {manifest_path}")
            existing = json.loads(manifest_path.read_text())
            for key in (
                    "kind", "status", "source", "output_fps", "frame_count",
                    "evidence", "files", "review", "policy_parent", "render"):
                if key == "render":
                    for stable in (
                            "output_video", "output_video_sha256",
                            "output_video_info", "review_video_sha256",
                            "review_video_info", "review_html_sha256",
                            "concat_manifests",
                            "chunk_count", "chunk_frames", "decoder", "encoder",
                            "render_spec_sha256", "blur", "extraction_plan",
                            "extraction_plan_sha256", "extracted_frames_dir",
                            "extracted_frames",
                            "extracted_frame_sha256"):
                        if existing[key].get(stable) != manifest[key].get(stable):
                            raise ValueError(
                                f"existing render manifest differs at render.{stable}")
                elif existing.get(key) != manifest.get(key):
                    raise ValueError(f"existing render manifest differs at {key}")
        else:
            _write_new_atomic(manifest_path, _json_bytes(manifest))
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"refusing to replace output: {output_dir}")
        os.rename(stage, output_dir)
    print(f"published anonymized video and review: {output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan_parser = subparsers.add_parser(
        "scan", help="scan a contiguous clip with restart-safe evidence")
    scan_parser.add_argument("--source", type=Path, required=True)
    scan_parser.add_argument("--weights", type=Path, required=True)
    scan_parser.add_argument("--output_dir", type=Path, required=True)
    scan_parser.add_argument("--capture_fps", type=float, required=True)
    scan_parser.add_argument("--output_fps", type=float, default=DEFAULT_OUTPUT_FPS)
    scan_parser.add_argument("--start_s", type=float, default=0.0)
    scan_parser.add_argument("--end_s", type=float)
    scan_parser.add_argument("--scan_width", type=int, default=DEFAULT_SCAN_WIDTH)
    scan_parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    scan_parser.add_argument(
        "--candidate_confidence", type=float,
        default=DEFAULT_CANDIDATE_CONFIDENCE)
    scan_parser.add_argument(
        "--direct_confidence", type=float,
        default=DEFAULT_DIRECT_CONFIDENCE)
    scan_parser.add_argument(
        "--device", default="auto",
        help="Ultralytics device; auto selects CUDA device 0 when available")
    scan_parser.add_argument("--workers", type=int, default=1)
    scan_parser.add_argument("--torch_threads", type=int, default=4)
    scan_parser.set_defaults(func=scan)

    policy_parser = subparsers.add_parser(
        "policy", help="apply one-frame person persistence and validate plates")
    policy_parser.add_argument("--scan_dir", type=Path, required=True)
    policy_parser.add_argument("--output_dir", type=Path, required=True)
    policy_parser.add_argument(
        "--plate_manifest", type=Path, required=True,
        help="prior anonymization manifest whose RAW plate detections are QA candidates")
    policy_parser.add_argument(
        "--manual_regions", type=Path,
        help="optional time-bounded normalized correction boxes for a new revision")
    policy_parser.add_argument("--flow_width", type=int, default=960)
    policy_parser.set_defaults(func=apply_policy)

    render_parser = subparsers.add_parser(
        "render", help="render full and review videos in restart-safe chunks")
    render_parser.add_argument("--policy_dir", type=Path, required=True)
    render_parser.add_argument("--output_dir", type=Path, required=True)
    render_parser.add_argument("--output_name", required=True)
    render_parser.add_argument("--chunk_frames", type=int, default=150)
    render_parser.add_argument(
        "--cfr_fast_seek", action="store_true",
        help=("use exact whole-second input seeking; requires operator-"
              "verified CFR with integer media fps that is an exact multiple "
              "of integer output fps"))
    render_parser.add_argument("--review_width", type=int, default=3840)
    render_parser.add_argument("--review_speedup", type=float, default=5.0)
    render_parser.add_argument("--extraction_plan", type=Path)
    render_parser.add_argument("--frames_dir", type=Path)
    render_parser.add_argument("--jpeg_quality", type=int, default=92)
    render_parser.add_argument(
        "--encoder", choices=anonymize_video.VIDEO_ENCODERS,
        default="software",
        help=("video encoder backend; nvenc requires FFmpeg NVENC support and "
              "a compatible NVIDIA GPU"))
    render_parser.set_defaults(func=render)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
