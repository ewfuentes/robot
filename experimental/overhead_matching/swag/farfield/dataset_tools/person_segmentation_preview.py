"""Build small, reviewable YOLO person-segmentation trials from 360 video.

This utility is deliberately separate from the full anonymization renderer.  It
scans discontinuous short sequences, preserving individual masks down to a low
confidence floor so temporal policies can be evaluated without repeatedly
running the detector.  Inputs are never modified and output directories are
published with no-clobber semantics.

The scanner requires Ultralytics at runtime.  It reproduces the supplied legacy
policy: YOLO11x-seg, COCO person class, a 1920-pixel inference size, and native
plus half-panorama-roll passes for seam handling.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from experimental.overhead_matching.swag.farfield.dataset_tools import person_mask_persistence


DEFAULT_SCAN_WIDTH = 1920
DEFAULT_PREVIEW_WIDTH = 3840
DEFAULT_CANDIDATE_CONFIDENCE = 0.05
DEFAULT_DIRECT_CONFIDENCE = 0.15
DEFAULT_IMGSZ = 1920


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> dict:
    """Bind a regular-file tree without following links."""
    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing symlink in evidence tree: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        file_digest = bytes.fromhex(sha256_file(path))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_digest)
        count += 1
        size += path.stat().st_size
    return {
        "sha256": digest.hexdigest(),
        "regular_file_count": count,
        "bytes": size,
    }


def _fraction(text: str) -> float:
    numerator, separator, denominator = text.partition("/")
    if not separator:
        return float(text)
    return float(numerator) / float(denominator)


def probe_video(path: Path) -> dict:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration,size", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if len(payload.get("streams", [])) != 1:
        raise ValueError(f"expected exactly one video stream: {path}")
    stream = payload["streams"][0]
    duration = float(stream.get("duration") or payload["format"]["duration"])
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "media_fps": _fraction(stream["avg_frame_rate"]),
        "nb_frames": (None if stream.get("nb_frames") in (None, "N/A")
                      else int(stream["nb_frames"])),
        "duration_s": duration,
        "size_bytes": int(payload["format"]["size"]),
    }


def parse_sample(text: str) -> tuple[str, float]:
    label, separator, value = text.partition("=")
    if (not separator or Path(label).name != label
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None):
        raise argparse.ArgumentTypeError(
            "samples must be LABEL=SOURCE_SECONDS with a filename-safe label")
    try:
        source_time_s = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid sample time: {value}") from error
    if not math.isfinite(source_time_s) or source_time_s < 0:
        raise argparse.ArgumentTypeError("sample time must be finite and nonnegative")
    return label, source_time_s


def mask_iou(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(left & right))
    if not intersection:
        return 0.0, 0.0
    left_area = int(np.count_nonzero(left))
    right_area = int(np.count_nonzero(right))
    union = left_area + right_area - intersection
    return intersection / union, intersection / min(left_area, right_area)


def deduplicate_instances(instances: list[dict]) -> list[dict]:
    """Merge duplicate native/rolled detections while retaining weak masks."""
    kept: list[dict] = []
    for candidate in sorted(instances, key=lambda item: item["confidence"],
                            reverse=True):
        duplicate = None
        for current in kept:
            if current.get("class_id", 0) != candidate.get("class_id", 0):
                continue
            # Segmentation NMS can retain several near-identical masks for one
            # heavily distorted close person.  Joining masks at this degree of
            # overlap is safe for redaction and gives persistence one instance
            # instead of several competing pseudo-tracks.
            iou, iom = mask_iou(current["mask"], candidate["mask"])
            if iou >= 0.50 or iom >= 0.85:
                duplicate = current
                break
        if duplicate is None:
            kept.append({
                "confidence": float(candidate["confidence"]),
                "mask": candidate["mask"].copy(),
                "passes": list(candidate["passes"]),
                "class_id": int(candidate.get("class_id", 0)),
            })
        else:
            duplicate["mask"] |= candidate["mask"]
            duplicate["confidence"] = max(
                duplicate["confidence"], float(candidate["confidence"]))
            duplicate["passes"].extend(candidate["passes"])
    return kept


class YoloPersonSegmenter:
    def __init__(self, weights: Path, *, imgsz: int, candidate_confidence: float,
                 device: str, classes: tuple[int, ...] = (0,)):
        try:
            # CUDA-enabled PyTorch wheels place shared libraries in sibling
            # runfiles under Bazel.  The repository loader preloads those
            # libraries before torch/Ultralytics import; ordinary venvs use
            # PyTorch's native package layout and do not need the workaround.
            if (os.environ.get("RUNFILES_DIR")
                    or os.environ.get("RUNFILES_MANIFEST_FILE")):
                import common.torch.load_torch_deps  # noqa: F401
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "scanning requires an environment with the ultralytics package") \
                from error
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.candidate_confidence = candidate_confidence
        self.device = device
        self.classes = tuple(classes)

    def detect(self, frame: np.ndarray) -> list[dict]:
        height, width = frame.shape[:2]
        instances = []
        for pass_name, roll in (("native", 0), ("horizontal_roll_0.5", width // 2)):
            input_frame = np.roll(frame, roll, axis=1) if roll else frame
            result = self.model.predict(
                input_frame, imgsz=self.imgsz,
                conf=self.candidate_confidence, classes=list(self.classes),
                retina_masks=True, verbose=False, device=self.device)[0]
            if result.masks is None or result.boxes is None:
                continue
            masks = result.masks.data.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            if masks.shape[1:] != (height, width):
                masks = np.stack([
                    cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                    for mask in masks
                ])
            for mask, confidence, class_id in zip(
                    masks, confidences, classes, strict=True):
                binary = mask > 0.5
                if roll:
                    binary = np.roll(binary, -roll, axis=1)
                if np.any(binary):
                    instances.append({
                        "confidence": float(confidence),
                        "mask": binary,
                        "passes": [pass_name],
                        "class_id": int(class_id),
                    })
        return deduplicate_instances(instances)


def _safe_stage(output_dir: Path) -> Path:
    stage = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output_dir}")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(f"staging output already exists: {stage}")
    stage.mkdir(parents=True)
    return stage


def _publish_stage(stage: Path, output_dir: Path):
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output_dir}")
    os.rename(stage, output_dir)


def _open_video(source: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source}")
    return capture


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
        raise RuntimeError(f"could not seek to frame {frame_index}")
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {frame_index}")
    position = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if math.isfinite(position) and abs(position - (frame_index + 1)) > 0.51:
        raise RuntimeError(
            f"decoder seek mismatch for frame {frame_index}: now at {position}")
    return frame


def scan_samples(args) -> int:
    source = args.source.resolve()
    weights = args.weights.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source must be a regular file: {source}")
    if not weights.is_file() or weights.is_symlink():
        raise ValueError(f"weights must be a regular file: {weights}")
    if not 0 < args.candidate_confidence < args.direct_confidence < 1:
        raise ValueError(
            "require 0 < candidate_confidence < direct_confidence < 1")
    if args.radius_frames < 1:
        raise ValueError("radius_frames must be at least one")

    video = probe_video(source)
    fps = video["media_fps"]
    if fps <= 0:
        raise ValueError("video frame rate must be positive")
    scan_height = round(args.scan_width * video["height"] / video["width"])
    preview_height = round(
        args.preview_width * video["height"] / video["width"])
    if min(scan_height, preview_height) < 2:
        raise ValueError("invalid output dimensions")

    labels = [label for label, _ in args.sample]
    if len(labels) != len(set(labels)):
        raise ValueError("sample labels must be unique")
    specs = []
    for label, center_s in args.sample:
        center_index = round(center_s * fps)
        indices = list(range(
            center_index - args.radius_frames,
            center_index + args.radius_frames + 1))
        if indices[0] < 0:
            raise ValueError(f"sample {label} begins before the video")
        if video["nb_frames"] is not None and indices[-1] >= video["nb_frames"]:
            raise ValueError(f"sample {label} ends after the video")
        if indices[-1] / fps >= video["duration_s"]:
            raise ValueError(f"sample {label} ends after the video")
        if args.clip_start_s is not None and indices[0] / fps < args.clip_start_s:
            raise ValueError(f"sample {label} begins before clip_start_s")
        if args.clip_end_s is not None and indices[-1] / fps >= args.clip_end_s:
            raise ValueError(f"sample {label} reaches exclusive clip_end_s")
        specs.append({
            "label": label,
            "requested_center_s": center_s,
            "center_frame_index": center_index,
            "frame_indices": indices,
        })

    stage = _safe_stage(args.output_dir)
    capture = None
    started = time.monotonic()
    try:
        detector = YoloPersonSegmenter(
            weights, imgsz=args.imgsz,
            candidate_confidence=args.candidate_confidence,
            device=args.device)
        capture = _open_video(source)
        frame_records = []
        total = len(specs) * (2 * args.radius_frames + 1)
        completed = 0
        for spec in specs:
            sequence_dir = stage / spec["label"]
            sequence_dir.mkdir()
            sequence_records = []
            for frame_index in spec["frame_indices"]:
                frame_started = time.monotonic()
                original = _read_frame(capture, frame_index)
                scan_frame = cv2.resize(
                    original, (args.scan_width, scan_height),
                    interpolation=cv2.INTER_AREA)
                preview_frame = cv2.resize(
                    original, (args.preview_width, preview_height),
                    interpolation=cv2.INTER_AREA)
                instances = detector.detect(scan_frame)
                stem = f"frame_{frame_index:08d}"
                frame_name = stem + ".jpg"
                mask_name = stem + ".npz"
                if not cv2.imwrite(
                        str(sequence_dir / frame_name), preview_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                    raise RuntimeError(f"could not write {frame_name}")
                masks = (np.stack([item["mask"] for item in instances])
                         if instances else
                         np.empty((0, scan_height, args.scan_width), dtype=bool))
                np.savez_compressed(
                    sequence_dir / mask_name,
                    masks=masks.astype(np.uint8),
                    confidences=np.asarray(
                        [item["confidence"] for item in instances],
                        dtype=np.float32))
                record = {
                    "frame_index": frame_index,
                    "source_time_s": round(frame_index / fps, 6),
                    "frame_file": frame_name,
                    "mask_file": mask_name,
                    "instances": [{
                        "mask_index": index,
                        "confidence": round(item["confidence"], 8),
                        "passes": sorted(set(item["passes"])),
                        "area_fraction": round(float(item["mask"].mean()), 9),
                    } for index, item in enumerate(instances)],
                    "direct_count": sum(
                        item["confidence"] >= args.direct_confidence
                        for item in instances),
                    "candidate_count": len(instances),
                }
                sequence_records.append(record)
                frame_records.append(record)
                completed += 1
                elapsed = time.monotonic() - started
                print(
                    f"[{completed:3d}/{total}] {spec['label']} "
                    f"source {frame_index / fps:.3f}s: "
                    f"{record['direct_count']} direct / {len(instances)} candidates "
                    f"({time.monotonic() - frame_started:.1f}s, "
                    f"mean {elapsed / completed:.1f}s/frame)", flush=True)
            with (sequence_dir / "sequence.json").open("x") as stream:
                json.dump({**spec, "frames": sequence_records}, stream, indent=2)
                stream.write("\n")

        manifest = {
            "schema_version": 1,
            "kind": "person_segmentation_sample_scan",
            "source": {
                "path": str(source),
                "sha256": sha256_file(source),
                **video,
            },
            "weights": {
                "path": str(weights),
                "sha256": sha256_file(weights),
                "bytes": weights.stat().st_size,
            },
            "detector": {
                "family": "Ultralytics YOLO11x instance segmentation",
                "class": 0,
                "class_name": "person",
                "imgsz": args.imgsz,
                "candidate_confidence": args.candidate_confidence,
                "direct_confidence": args.direct_confidence,
                "passes": ["native", "horizontal_roll_0.5"],
                "retina_masks": True,
                "device": args.device,
            },
            "scan_resolution": [args.scan_width, scan_height],
            "preview_resolution": [args.preview_width, preview_height],
            "samples": specs,
            "frame_count": len(frame_records),
            "elapsed_s": round(time.monotonic() - started, 3),
            "library_versions": {
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        }
        try:
            import torch
            import ultralytics
            manifest["library_versions"].update({
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
            })
        except ImportError:
            pass
        with (stage / "scan_manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        _publish_stage(stage, args.output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if capture is not None:
            capture.release()

    print(f"published sample scan: {args.output_dir}", flush=True)
    return 0


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST) != 0


def _load_sequence(scan_dir: Path, spec: dict, manifest: dict,
                   flow_width: int) -> list[dict]:
    sequence_dir = scan_dir / spec["label"]
    sequence_path = sequence_dir / "sequence.json"
    sequence = json.loads(sequence_path.read_text())
    if sequence.get("frame_indices") != spec["frame_indices"]:
        raise ValueError(f"sequence manifest mismatch: {sequence_path}")
    scan_width, scan_height = manifest["scan_resolution"]
    flow_height = round(flow_width * scan_height / scan_width)
    direct_confidence = manifest["detector"]["direct_confidence"]
    frames = []
    for record in sequence["frames"]:
        frame_file = sequence_dir / record["frame_file"]
        mask_file = sequence_dir / record["mask_file"]
        if (frame_file.is_symlink() or not frame_file.is_file()
                or mask_file.is_symlink() or not mask_file.is_file()):
            raise ValueError(f"missing regular scan evidence for {record}")
        preview = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        if preview is None:
            raise ValueError(f"could not read preview frame: {frame_file}")
        with np.load(mask_file, allow_pickle=False) as payload:
            masks = payload["masks"].astype(bool)
            confidences = payload["confidences"].astype(np.float32)
        if masks.ndim != 3 or masks.shape[1:] != (scan_height, scan_width):
            raise ValueError(f"mask shape changed in {mask_file}: {masks.shape}")
        if len(masks) != len(confidences):
            raise ValueError(f"mask/confidence count mismatch: {mask_file}")
        direct = np.zeros((flow_height, flow_width), dtype=bool)
        weak = np.zeros_like(direct)
        for mask, confidence in zip(masks, confidences, strict=True):
            resized = _resize_mask(mask, flow_width, flow_height)
            if confidence >= direct_confidence:
                direct |= resized
            else:
                weak |= resized
        flow_frame = cv2.resize(
            preview, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
        frames.append({
            "record": record,
            "preview": preview,
            "flow_frame": flow_frame,
            "direct_mask": direct,
            "weak_mask": weak,
            "confidences": [float(value) for value in confidences],
        })
    return frames


def _mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _tint_mask(frame: np.ndarray, mask: np.ndarray,
               color: tuple[int, int, int], alpha: float):
    if not np.any(mask):
        return
    pixels = frame[mask].astype(np.float32)
    frame[mask] = np.clip(
        (1.0 - alpha) * pixels + alpha * np.asarray(color),
        0, 255).astype(np.uint8)


def _draw_mask(frame: np.ndarray, mask: np.ndarray,
               color: tuple[int, int, int], *, alpha: float = 0.35,
               thickness: int = 3):
    _tint_mask(frame, mask, color, alpha)
    cv2.drawContours(
        frame, _mask_contours(mask), -1, color, thickness, cv2.LINE_AA)


def _title(frame: np.ndarray, title: str, detail: str = ""):
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 58), (0, 0, 0), -1)
    cv2.putText(frame, title, (16, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(frame, detail, (16, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (210, 210, 210), 1, cv2.LINE_AA)


def _blur_masked(frame: np.ndarray, mask: np.ndarray,
                 source_width: int) -> np.ndarray:
    """Preview the supplied legacy dilation, feathering, and Gaussian blur."""
    width = frame.shape[1]
    scale = width / source_width
    dilation_size = max(3, int(round(21 * scale)) | 1)
    feather_size = max(3, int(round(41 * scale)) | 1)
    expanded = cv2.dilate(
        mask.astype(np.uint8), np.ones(
            (dilation_size, dilation_size), dtype=np.uint8))
    feather = cv2.GaussianBlur(
        expanded.astype(np.float32), (feather_size, feather_size), 0)
    sigma = max(2.5, max(10, source_width // 250) * scale)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = feather[..., None]
    return np.clip(
        frame.astype(np.float32) * (1.0 - alpha)
        + blurred.astype(np.float32) * alpha,
        0, 255).astype(np.uint8)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class _ReviewVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.path = path
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{width}x{height}", "-r", f"{fps:.12g}",
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        if self.process.stdin is None:
            raise RuntimeError("review encoder stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(
                f"review encoder failed with exit status {return_code}")


def _comparison_frame(frame: dict, result,
                      panel_width: int, source_width: int) -> np.ndarray:
    preview = frame["preview"]
    panel_height = round(panel_width * preview.shape[0] / preview.shape[1])
    original = cv2.resize(
        preview, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    direct_mask = _resize_mask(
        frame["direct_mask"], panel_width, panel_height)
    weak_mask = _resize_mask(frame["weak_mask"], panel_width, panel_height)
    temporal_mask = _resize_mask(
        result.temporal_fill_mask, panel_width, panel_height)
    accepted_mask = _resize_mask(
        result.accepted_mask, panel_width, panel_height)
    flagged_mask = np.zeros_like(frame["direct_mask"])
    for flag in result.review_flags:
        flagged_mask |= flag.mask
    # Optical-flow disagreement often leaves a one-pixel fringe just outside
    # an otherwise complete direct mask.  Keep it in the metrics ledger, but
    # do not draw a misleading full-person orange contour for that fringe.
    covered = cv2.dilate(
        result.accepted_mask.astype(np.uint8),
        np.ones((15, 15), dtype=np.uint8)) != 0
    flagged_mask &= ~covered
    flagged_mask = cv2.morphologyEx(
        flagged_mask.astype(np.uint8), cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8)) != 0
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            flagged_mask.astype(np.uint8), connectivity=8))
    flagged_mask.fill(False)
    for component in range(1, component_count):
        if component_stats[component, cv2.CC_STAT_AREA] >= 9:
            flagged_mask |= component_labels == component
    flagged_mask = _resize_mask(flagged_mask, panel_width, panel_height)

    source_time = frame["record"]["source_time_s"]
    frame_index = frame["record"]["frame_index"]
    identity = f"SOURCE {source_time:.3f}s  FRAME {frame_index}"

    original_panel = original.copy()
    _title(original_panel, "1  ORIGINAL (local review only)", identity)

    direct_panel = original.copy()
    _draw_mask(direct_panel, direct_mask, (50, 220, 50), alpha=0.32)
    _draw_mask(direct_panel, weak_mask, (0, 215, 255),
               alpha=0.12, thickness=2)
    _title(
        direct_panel, "2  RAW YOLO11x-SEG PERSON MASKS",
        f"green=accepted >=0.15  yellow=weak 0.05-0.15  {identity}")

    temporal_panel = original.copy()
    _draw_mask(temporal_panel, direct_mask, (50, 220, 50), alpha=0.24)
    _draw_mask(temporal_panel, temporal_mask, (255, 0, 255), alpha=0.46)
    _draw_mask(temporal_panel, flagged_mask, (0, 100, 255),
               alpha=0.12, thickness=3)
    modes = ",".join(fill.mode for fill in result.fills) or "none"
    _title(
        temporal_panel, "3  TEMPORAL EVIDENCE",
        f"magenta=accepted fill ({modes})  orange=review-only suspicion  {identity}")

    blurred_panel = _blur_masked(original, accepted_mask, source_width)
    cv2.drawContours(
        blurred_panel, _mask_contours(direct_mask), -1,
        (50, 220, 50), 2, cv2.LINE_AA)
    cv2.drawContours(
        blurred_panel, _mask_contours(temporal_mask), -1,
        (255, 0, 255), 2, cv2.LINE_AA)
    _title(
        blurred_panel, "4  RESULT USED FOR SAMPLE",
        f"mask-shaped blur only; review flags are NOT blurred  {identity}")
    return np.vstack((
        np.hstack((original_panel, direct_panel)),
        np.hstack((temporal_panel, blurred_panel)),
    ))


def _matching_direct_component(direct_mask: np.ndarray,
                               proposal: np.ndarray) -> np.ndarray | None:
    count, labels = cv2.connectedComponents(
        direct_mask.astype(np.uint8), connectivity=8)
    best = None
    best_overlap = 0
    for label in range(1, count):
        component = labels == label
        overlap = int(np.count_nonzero(component & proposal))
        if overlap > best_overlap:
            best = component
            best_overlap = overlap
    return best


def _controlled_dropout(frame: dict, previous: dict, following: dict,
                        config: person_mask_persistence.PersistenceConfig):
    """Find one real direct instance that the temporal policy can recover."""
    empty = np.zeros_like(frame["direct_mask"])
    flows = person_mask_persistence.estimate_gap_flows(
        previous["flow_frame"], frame["flow_frame"],
        following["flow_frame"], config)
    all_missing = person_mask_persistence.bridge_one_frame_gap(
        previous["flow_frame"], frame["flow_frame"],
        following["flow_frame"], previous["direct_mask"], empty,
        following["direct_mask"], None, config=config, flows=flows)
    for fill in sorted(
            all_missing.fills,
            key=lambda item: np.count_nonzero(item.mask), reverse=True):
        target = _matching_direct_component(frame["direct_mask"], fill.mask)
        if target is None:
            continue
        modified_direct = frame["direct_mask"] & ~target
        recovered = person_mask_persistence.bridge_one_frame_gap(
            previous["flow_frame"], frame["flow_frame"],
            following["flow_frame"], previous["direct_mask"],
            modified_direct, following["direct_mask"], frame["weak_mask"],
            config=config, flows=flows)
        if np.any(recovered.temporal_fill_mask & target):
            return target, modified_direct, recovered
    return None


def _square_crop_bounds(mask: np.ndarray, minimum_size: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot crop an empty target mask")
    height, width = mask.shape
    target_width = int(xs.max() - xs.min() + 1)
    target_height = int(ys.max() - ys.min() + 1)
    size = min(
        min(height, width),
        max(minimum_size, 3 * max(target_width, target_height)))
    center_x = int(round((int(xs.min()) + int(xs.max())) / 2))
    center_y = int(round((int(ys.min()) + int(ys.max())) / 2))
    x1 = max(0, min(width - size, center_x - size // 2))
    y1 = max(0, min(height - size, center_y - size // 2))
    return x1, y1, x1 + size, y1 + size


def _dropout_comparison_frame(frame: dict, target_mask: np.ndarray,
                              modified_direct: np.ndarray, result,
                              panel_width: int, source_width: int) -> tuple[
                                  np.ndarray, np.ndarray]:
    preview = frame["preview"]
    panel_height = round(panel_width * preview.shape[0] / preview.shape[1])
    original = cv2.resize(
        preview, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    actual = _resize_mask(frame["direct_mask"], panel_width, panel_height)
    modified = _resize_mask(modified_direct, panel_width, panel_height)
    target = _resize_mask(target_mask, panel_width, panel_height)
    temporal = _resize_mask(
        result.temporal_fill_mask, panel_width, panel_height)
    accepted = _resize_mask(result.accepted_mask, panel_width, panel_height)
    identity = (f"SOURCE {frame['record']['source_time_s']:.3f}s  "
                f"FRAME {frame['record']['frame_index']}")

    panels = []
    first = original.copy()
    _title(first, "1  ORIGINAL", f"CONTROLLED ABLATION - not an observed miss  {identity}")
    panels.append(first)

    second = original.copy()
    _draw_mask(second, actual, (50, 220, 50), alpha=0.30)
    cv2.drawContours(second, _mask_contours(target), -1,
                     (255, 255, 0), 5, cv2.LINE_AA)
    _title(second, "2  ACTUAL RAW YOLO",
           f"cyan=instance deliberately removed in next panel  {identity}")
    panels.append(second)

    third = original.copy()
    _draw_mask(third, modified, (50, 220, 50), alpha=0.24)
    _draw_mask(third, temporal, (255, 0, 255), alpha=0.50)
    _title(third, "3  FORCED ONE-INSTANCE MISS",
           f"magenta=two-sided optical-flow recovery  {identity}")
    panels.append(third)

    fourth = _blur_masked(original, accepted, source_width)
    cv2.drawContours(fourth, _mask_contours(temporal), -1,
                     (255, 0, 255), 3, cv2.LINE_AA)
    _title(fourth, "4  RECOVERED RESULT",
           f"only mask-shaped accepted evidence is blurred  {identity}")
    panels.append(fourth)

    comparison = np.vstack((
        np.hstack((panels[0], panels[1])),
        np.hstack((panels[2], panels[3])),
    ))
    x1, y1, x2, y2 = _square_crop_bounds(target, minimum_size=360)
    zoom_panels = []
    zoom_titles = ("ORIGINAL", "ACTUAL YOLO", "FORCED MISS + RECOVERY", "BLURRED")
    for panel, title in zip(panels, zoom_titles, strict=True):
        crop = cv2.resize(
            panel[y1:y2, x1:x2], (640, 640), interpolation=cv2.INTER_CUBIC)
        _title(crop, title, "controlled ablation")
        zoom_panels.append(crop)
    return comparison, np.hstack(zoom_panels)


def _write_review_html(path: Path, manifest: dict):
    sections = []
    for sequence in manifest["sequences"]:
        label = html.escape(sequence["label"])
        sections.append(f"""
<section>
  <h2>{label}</h2>
  <p>{sequence['frame_count']} frames around source {sequence['requested_center_s']:.3f}s;
     temporal fills: {sequence['accepted_fill_count']};
     review-only flags: {sequence['review_flag_count']}.</p>
  <video controls loop preload="metadata" src="{label}/comparison.mp4"></video>
  <p><a href="{label}/contact.jpg">Open the all-frame contact sheet</a> ·
     <a href="{label}/persistence.json">Open persistence metrics</a></p>
</section>""")
    ablations = []
    for dropout in manifest.get("controlled_dropouts", []):
        label = html.escape(dropout["label"])
        ablations.append(f"""
<section>
  <h2>Controlled dropout: {label}</h2>
  <p>This is a test, not an observed detector miss. One actual middle-frame
     person mask was deliberately removed; the magenta mask is what the
     two-sided persistence policy recovered.</p>
  <img loading="lazy" src="controlled_dropouts/{label}_zoom.jpg">
  <p><a href="controlled_dropouts/{label}.jpg">Open the full panorama comparison</a></p>
</section>""")
    ablation_section = ("<h1>Controlled one-instance dropout checks</h1>"
                        + "".join(ablations)) if ablations else ""
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Franconia person-mask persistence samples</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#181818;color:#eee}}
a{{color:#7ec8ff}} video,img{{display:block;width:100%;height:auto;background:#000}} section{{margin:3rem 0}}
code{{background:#333;padding:.15rem .3rem}}
</style>
<h1>Person-segmentation + conservative persistence samples</h1>
<p>This is local, unapproved review evidence—not a privacy-cleared dataset. Each
frame is a 2×2 comparison: original, raw YOLO masks, temporal evidence, and the
blurred result. Green is a direct detection; magenta is a two-sided temporal
fill; orange is suspicious evidence that failed an automatic gate and was
<strong>not</strong> blurred.</p>
<p>The policy only bridges a single missing frame when both neighboring masks
agree after cylindrical optical flow. It never propagates a rectangle.</p>
<p><a href="overview.jpg">Open the six-scene overview</a> ·
<a href="preview_manifest.json">Open provenance and counts</a></p>
{''.join(sections)}
{ablation_section}
"""
    path.write_text(document)


def render_samples(args) -> int:
    scan_dir = args.scan_dir.resolve()
    scan_manifest_path = scan_dir / "scan_manifest.json"
    if (scan_manifest_path.is_symlink()
            or not scan_manifest_path.is_file()):
        raise ValueError(f"missing scan manifest: {scan_manifest_path}")
    scan_manifest = json.loads(scan_manifest_path.read_text())
    if scan_manifest.get("kind") != "person_segmentation_sample_scan":
        raise ValueError("not a person-segmentation sample scan")
    if args.flow_width < 64 or args.panel_width < 320:
        raise ValueError("flow_width/panel_width are unreasonably small")
    if args.controlled_dropouts < 0:
        raise ValueError("controlled_dropouts must be nonnegative")

    stage = _safe_stage(args.output_dir)
    started = time.monotonic()
    overview_frames = []
    sequence_summaries = []
    controlled_dropouts = []
    try:
        config = person_mask_persistence.PersistenceConfig()
        for spec in scan_manifest["samples"]:
            print(f"rendering persistence sample {spec['label']}", flush=True)
            frames = _load_sequence(scan_dir, spec, scan_manifest, args.flow_width)
            empty = np.zeros_like(frames[0]["direct_mask"])
            results = []
            records = []
            for index, frame in enumerate(frames):
                if index == 0 or index == len(frames) - 1:
                    result = person_mask_persistence.PersistenceResult(
                        accepted_mask=frame["direct_mask"].copy(),
                        temporal_fill_mask=empty.copy(),
                        fills=(), review_flags=(),
                        metrics={"boundary_frame": 1.0})
                else:
                    result = person_mask_persistence.bridge_one_frame_gap(
                        frames[index - 1]["flow_frame"],
                        frame["flow_frame"],
                        frames[index + 1]["flow_frame"],
                        frames[index - 1]["direct_mask"],
                        frame["direct_mask"],
                        frames[index + 1]["direct_mask"],
                        frame["weak_mask"],
                        config=config)
                results.append(result)
                records.append({
                    **frame["record"],
                    "candidate_confidences": frame["confidences"],
                    "persistence": _json_safe(result.metadata()),
                })

            sequence_dir = stage / spec["label"]
            frame_dir = sequence_dir / "frames"
            frame_dir.mkdir(parents=True)
            video_path = sequence_dir / "comparison.mp4"
            panel_height = round(
                args.panel_width
                * scan_manifest["preview_resolution"][1]
                / scan_manifest["preview_resolution"][0])
            writer = _ReviewVideoWriter(
                video_path, 2 * args.panel_width, 2 * panel_height,
                scan_manifest["source"]["media_fps"])
            comparison_frames = []
            try:
                for frame, result in zip(frames, results, strict=True):
                    comparison = _comparison_frame(
                        frame, result, args.panel_width,
                        scan_manifest["source"]["width"])
                    writer.write(comparison)
                    still_path = frame_dir / (
                        f"source_{frame['record']['source_time_s']:011.3f}s.jpg")
                    if not cv2.imwrite(
                            str(still_path), comparison,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                        raise RuntimeError(f"could not write {still_path}")
                    thumbnail = cv2.resize(
                        comparison, (args.contact_width, round(
                            args.contact_width * comparison.shape[0]
                            / comparison.shape[1])),
                        interpolation=cv2.INTER_AREA)
                    comparison_frames.append(thumbnail)
            finally:
                writer.close()
            contact = np.vstack(comparison_frames)
            if not cv2.imwrite(
                    str(sequence_dir / "contact.jpg"), contact,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                raise RuntimeError("could not write sequence contact sheet")
            center = comparison_frames[len(comparison_frames) // 2]
            overview_frames.append(center)
            with (sequence_dir / "persistence.json").open("x") as stream:
                json.dump({"label": spec["label"], "frames": records},
                          stream, indent=2)
                stream.write("\n")
            ledger_path = sequence_dir / "persistence.json"
            contact_path = sequence_dir / "contact.jpg"
            summary = {
                "label": spec["label"],
                "requested_center_s": spec["requested_center_s"],
                "frame_count": len(frames),
                "accepted_fill_count": sum(len(result.fills) for result in results),
                "accepted_fill_frames": sum(bool(result.fills) for result in results),
                "review_flag_count": sum(
                    len(result.review_flags) for result in results),
                "review_flag_frames": sum(
                    bool(result.review_flags) for result in results),
                "comparison_video": f"{spec['label']}/comparison.mp4",
                "comparison_video_sha256": sha256_file(video_path),
                "contact_sheet": f"{spec['label']}/contact.jpg",
                "contact_sheet_sha256": sha256_file(contact_path),
                "ledger": f"{spec['label']}/persistence.json",
                "ledger_sha256": sha256_file(ledger_path),
            }
            sequence_summaries.append(summary)

            if len(controlled_dropouts) < args.controlled_dropouts:
                center_index = len(frames) // 2
                dropout = _controlled_dropout(
                    frames[center_index], frames[center_index - 1],
                    frames[center_index + 1], config)
                if dropout is not None:
                    target, modified_direct, recovered = dropout
                    controlled_dir = stage / "controlled_dropouts"
                    controlled_dir.mkdir(exist_ok=True)
                    comparison, zoom = _dropout_comparison_frame(
                        frames[center_index], target, modified_direct, recovered,
                        args.panel_width, scan_manifest["source"]["width"])
                    comparison_path = controlled_dir / f"{spec['label']}.jpg"
                    zoom_path = controlled_dir / f"{spec['label']}_zoom.jpg"
                    if (not cv2.imwrite(
                            str(comparison_path), comparison,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                            or not cv2.imwrite(
                                str(zoom_path), zoom,
                                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])):
                        raise RuntimeError("could not write controlled dropout")
                    controlled_dropouts.append({
                        "label": spec["label"],
                        "source_time_s": frames[center_index]["record"][
                            "source_time_s"],
                        "frame_index": frames[center_index]["record"][
                            "frame_index"],
                        "kind": "controlled_ablation_not_observed_miss",
                        "removed_direct_pixels": int(np.count_nonzero(target)),
                        "recovered_temporal_pixels": int(np.count_nonzero(
                            recovered.temporal_fill_mask & target)),
                        "result": _json_safe(recovered.metadata()),
                        "comparison": f"controlled_dropouts/{spec['label']}.jpg",
                        "comparison_sha256": sha256_file(comparison_path),
                        "zoom": f"controlled_dropouts/{spec['label']}_zoom.jpg",
                        "zoom_sha256": sha256_file(zoom_path),
                    })

        overview = np.vstack(overview_frames)
        if not cv2.imwrite(
                str(stage / "overview.jpg"), overview,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
            raise RuntimeError("could not write overview")
        manifest = {
            "schema_version": 1,
            "kind": "person_segmentation_persistence_preview",
            "status": "pending_human_review",
            "scan": {
                "path": str(scan_dir),
                "manifest_sha256": sha256_file(scan_manifest_path),
                "tree": sha256_tree(scan_dir),
                "source_sha256": scan_manifest["source"]["sha256"],
                "weights_sha256": scan_manifest["weights"]["sha256"],
            },
            "policy": {
                "direct_confidence": scan_manifest["detector"][
                    "direct_confidence"],
                "candidate_confidence": scan_manifest["detector"][
                    "candidate_confidence"],
                "gap_frames": 1,
                "two_sided_evidence_required": True,
                "flow_width": args.flow_width,
                "fill_shape": "mask only; rectangles prohibited",
                "config": _json_safe(vars(config)),
            },
            "panel_width": args.panel_width,
            "sequences": sequence_summaries,
            "controlled_dropouts": controlled_dropouts,
            "accepted_fill_count": sum(
                item["accepted_fill_count"] for item in sequence_summaries),
            "review_flag_count": sum(
                item["review_flag_count"] for item in sequence_summaries),
            "elapsed_s": round(time.monotonic() - started, 3),
            "overview": {
                "path": "overview.jpg",
                "sha256": sha256_file(stage / "overview.jpg"),
            },
        }
        _write_review_html(stage / "index.html", manifest)
        manifest["review_html"] = {
            "path": "index.html",
            "sha256": sha256_file(stage / "index.html"),
        }
        with (stage / "preview_manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
        _publish_stage(stage, args.output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"published review gallery: {args.output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser(
        "scan", help="run YOLO on short, discontinuous sample sequences")
    scan.add_argument("--source", type=Path, required=True)
    scan.add_argument("--weights", type=Path, required=True)
    scan.add_argument("--output_dir", type=Path, required=True)
    scan.add_argument(
        "--sample", action="append", type=parse_sample, required=True,
        help="repeatable LABEL=SOURCE_SECONDS sample center")
    scan.add_argument("--radius_frames", type=int, default=3)
    scan.add_argument("--clip_start_s", type=float)
    scan.add_argument("--clip_end_s", type=float)
    scan.add_argument("--scan_width", type=int, default=DEFAULT_SCAN_WIDTH)
    scan.add_argument("--preview_width", type=int, default=DEFAULT_PREVIEW_WIDTH)
    scan.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    scan.add_argument(
        "--candidate_confidence", type=float,
        default=DEFAULT_CANDIDATE_CONFIDENCE)
    scan.add_argument(
        "--direct_confidence", type=float,
        default=DEFAULT_DIRECT_CONFIDENCE)
    scan.add_argument("--device", default="cpu")
    scan.add_argument("--jpeg_quality", type=int, default=95)
    scan.set_defaults(func=scan_samples)

    render = subparsers.add_parser(
        "render", help="apply persistence and build a side-by-side review gallery")
    render.add_argument("--scan_dir", type=Path, required=True)
    render.add_argument("--output_dir", type=Path, required=True)
    render.add_argument("--flow_width", type=int, default=960)
    render.add_argument("--panel_width", type=int, default=1920)
    render.add_argument("--contact_width", type=int, default=1920)
    render.add_argument(
        "--controlled_dropouts", type=int, default=2,
        help="number of clearly labeled single-instance ablation checks")
    render.add_argument("--jpeg_quality", type=int, default=92)
    render.set_defaults(func=render_samples)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
