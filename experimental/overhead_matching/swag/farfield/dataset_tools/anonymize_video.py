"""Detect plates, irreversibly blur privacy regions, and build review evidence.

This is a pre-ingest privacy stage for self-collected farfield recordings.  It
never edits the input.  ``scan`` decodes a low-resolution 3 fps view and writes
an auditable per-frame detection ledger.  ``render`` applies that ledger to a
fresh full-resolution decode, publishes a separate blurred 3 fps video, saves
any requested dataset frames, and creates an accelerated review video.  The
review video deliberately draws the applied regions: a human should watch for
an identifiable face or plate *without* a surrounding outline.

The detector ledger is an intermediate aid, not a claim that automated privacy
redaction is perfect.  Human approval is therefore explicit and starts in the
``pending`` state.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_FPS = 3.0
DEFAULT_SCAN_WIDTH = 1920
DEFAULT_REVIEW_WIDTH = 3840
DEFAULT_REVIEW_SPEEDUP = 5.0
VIDEO_ENCODERS = ("software", "nvenc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_regular_file_bytes(path: Path) -> bytes:
    """Read one stable regular-file snapshot without following a symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"expected a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _fraction(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    return float(numerator) / float(denominator)


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
        "nb_frames,duration,color_space,color_transfer,color_primaries",
        "-show_entries", "format=duration,size", "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if len(payload.get("streams", [])) != 1:
        raise ValueError(f"expected exactly one video stream in {path}")
    stream = payload["streams"][0]
    fps_text = stream.get("avg_frame_rate") or stream["r_frame_rate"]
    duration = float(stream.get("duration") or payload["format"]["duration"])
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "media_fps": _fraction(fps_text),
        "media_fps_fraction": fps_text,
        "nb_frames": (int(stream["nb_frames"])
                      if stream.get("nb_frames") not in (None, "N/A")
                      else None),
        "duration_s": duration,
        "size_bytes": int(payload["format"]["size"]),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
    }


def frame_selection(media_fps: float, output_fps: float) -> tuple[str, int | None]:
    """Return an ffmpeg selection expression and exact integer frame step."""
    ratio = media_fps / output_fps
    rounded = round(ratio)
    if rounded >= 1 and math.isclose(ratio, rounded, rel_tol=0, abs_tol=1e-6):
        if rounded == 1:
            return "", 1
        return f"select=not(mod(n\\,{rounded}))", rounded
    return f"fps={output_fps:.12g}:start_time=0:round=near", None


def expected_output_frames(video: dict, output_fps: float) -> int:
    _, step = frame_selection(video["media_fps"], output_fps)
    if step is not None and video["nb_frames"] is not None:
        return (video["nb_frames"] - 1) // step + 1
    return max(1, int(round(video["duration_s"] * output_fps)))


def clip_frame_window(video: dict, output_fps: float,
                      start_s: float = 0.0,
                      end_s: float | None = None) -> tuple[int, int]:
    """Return the end-exclusive frame window on the output-fps source grid."""
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("clip start must be a finite nonnegative time")
    if end_s is not None and (not math.isfinite(end_s) or end_s <= start_s):
        raise ValueError("clip end must be finite and later than clip start")
    total = expected_output_frames(video, output_fps)
    # A clip is [start, end): include the first grid frame at or after start,
    # and exclude the first grid frame at or after end.
    start_frame = math.ceil(start_s * output_fps - 1e-9)
    end_frame = (total if end_s is None else
                 math.ceil(end_s * output_fps - 1e-9))
    start_frame = min(total, max(0, start_frame))
    end_frame = min(total, max(0, end_frame))
    if end_frame <= start_frame:
        raise ValueError(
            f"clip resolves to an empty frame window [{start_frame}, "
            f"{end_frame})")
    return start_frame, end_frame


def clip_metadata(video: dict, output_fps: float,
                  start_s: float = 0.0, end_s: float | None = None) -> dict:
    start_frame, end_frame = clip_frame_window(
        video, output_fps, start_s, end_s)
    return {
        "start_s": start_frame / output_fps,
        "end_s": end_frame / output_fps,
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "frame_count": end_frame - start_frame,
    }


def cfr_fast_seek_fps(source_info: dict, output_fps: float) -> int:
    """Validate the strict CFR fast-seek contract and return integer fps."""
    _, step = frame_selection(source_info["media_fps"], output_fps)
    output_integer = round(output_fps)
    media_integer = round(float(source_info["media_fps"]))
    if (step is None
            or not math.isclose(
                output_fps, output_integer, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(
                float(source_info["media_fps"]), media_integer,
                rel_tol=0, abs_tol=1e-9)
            or media_integer != output_integer * step):
        raise ValueError(
            "CFR fast seek requires integer media fps that is an exact "
            "multiple of integer output fps")
    return output_integer


class RawVideoReader:
    """Single-pass ffmpeg decoder yielding fixed-size BGR arrays."""

    def __init__(self, source: Path, source_info: dict, output_fps: float,
                 width: int | None = None, height: int | None = None,
                 start_frame: int = 0, end_frame: int | None = None,
                 scale_flags: str = "lanczos", *,
                 cfr_fast_seek: bool = False):
        self.source = source
        self.width = width or source_info["width"]
        self.height = height or source_info["height"]
        select, step = frame_selection(source_info["media_fps"], output_fps)
        seek_seconds = 0
        trim_start = start_frame
        trim_end = end_frame
        if cfr_fast_seek:
            fps_integer = cfr_fast_seek_fps(source_info, output_fps)
            if (start_frame < 0
                    or (end_frame is not None and end_frame < start_frame)):
                raise ValueError("invalid CFR fast-seek frame interval")
            # A whole-second anchor preserves the phase of exact integer frame
            # selection because media_fps = output_fps * selection_step.
            # Trim the sub-second remainder on the selected output grid.
            seek_seconds = start_frame // fps_integer
            anchor_frame = seek_seconds * fps_integer
            trim_start = start_frame - anchor_frame
            trim_end = (None if end_frame is None
                        else end_frame - anchor_frame)
        filters = []
        if select:
            filters.append(select)
        if trim_start or trim_end is not None:
            trim = f"trim=start_frame={trim_start}"
            if trim_end is not None:
                trim += f":end_frame={trim_end}"
            filters.extend([trim, "setpts=PTS-STARTPTS"])
        if (self.width != source_info["width"]
                or self.height != source_info["height"]):
            if not re.fullmatch(r"[A-Za-z0-9_+]+", scale_flags):
                raise ValueError(f"invalid ffmpeg scale flags: {scale_flags}")
            filters.append(
                f"scale={self.width}:{self.height}:flags={scale_flags}")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if cfr_fast_seek and seek_seconds:
            command += ["-ss", str(seek_seconds), "-accurate_seek"]
        command += ["-i", str(source), "-map", "0:v:0"]
        if filters:
            command += ["-vf", ",".join(filters)]
        # A select filter changes frame count, not the input stream's nominal
        # rate.  Passthrough is essential here: ffmpeg otherwise duplicates
        # the selected frames back to 30 fps on a rawvideo output.
        command += ["-an", "-sn", "-dn", "-fps_mode", "passthrough",
                    "-pix_fmt", "bgr24",
                    "-f", "rawvideo", "pipe:1"]
        self.command = command
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.frame_bytes = self.width * self.height * 3

    def __iter__(self):
        if self.process.stdout is None:
            raise RuntimeError("decoder stdout is unavailable")
        while True:
            buffer = bytearray(self.frame_bytes)
            view = memoryview(buffer)
            offset = 0
            while offset < self.frame_bytes:
                count = self.process.stdout.readinto(view[offset:])
                if not count:
                    break
                offset += count
            if offset == 0:
                break
            if offset != self.frame_bytes:
                self._finish()
                raise RuntimeError(
                    f"partial decoded frame from {self.source}: "
                    f"{offset}/{self.frame_bytes} bytes")
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(
                self.height, self.width, 3)
        self._finish()

    def _finish(self):
        if self.process.stdout:
            self.process.stdout.close()
        stderr = ""
        if self.process.stderr:
            stderr = self.process.stderr.read().decode("utf-8", "replace")
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(
                f"ffmpeg decode failed ({return_code}) for {self.source}: "
                f"{stderr.strip()}")


def video_encoder_profile(backend: str) -> dict:
    """Return the exact full/review ffmpeg settings for one encoder backend."""
    if backend == "software":
        return {
            "backend": backend,
            "full": {
                "codec": "libx265",
                "ffmpeg_args": [
                    "-c:v", "libx265", "-preset", "ultrafast", "-crf", "18",
                    "-x265-params",
                    "pools=16:frame-threads=2:log-level=error",
                ],
            },
            "review": {
                "codec": "libx264",
                "ffmpeg_args": [
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                ],
            },
        }
    if backend == "nvenc":
        return {
            "backend": backend,
            "full": {
                "codec": "hevc_nvenc",
                "ffmpeg_args": [
                    "-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq",
                    "-rc", "vbr", "-cq", "18", "-b:v", "0",
                ],
            },
            "review": {
                "codec": "h264_nvenc",
                "ffmpeg_args": [
                    "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                    "-rc", "vbr", "-cq", "23", "-b:v", "0",
                ],
            },
        }
    raise ValueError(
        f"unsupported video encoder {backend!r}; choose from {VIDEO_ENCODERS}")


class RawVideoWriter:
    """Raw-BGR ffmpeg encoder with no-clobber incomplete-file publication."""

    def __init__(self, output: Path, width: int, height: int, fps: float,
                 *, review: bool = False, encoder: str = "software"):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to replace output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.incomplete = output.with_name(output.name + ".incomplete.mp4")
        if self.incomplete.exists() or self.incomplete.is_symlink():
            raise FileExistsError(
                f"incomplete output already exists: {self.incomplete}")
        self.encoder_profile = video_encoder_profile(encoder)
        stream_kind = "review" if review else "full"
        codec = self.encoder_profile[stream_kind]["ffmpeg_args"]
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
            "-framerate", f"{fps:.12g}", "-i", "pipe:0", "-an", *codec,
            "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-colorspace", "bt709",
            "-movflags", "+faststart", str(self.incomplete),
        ]
        self.command = command
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def write(self, frame: np.ndarray):
        if self.process.stdin is None:
            raise RuntimeError("encoder stdin is unavailable")
        view = memoryview(np.ascontiguousarray(frame)).cast("B")
        while view:
            written = self.process.stdin.write(view)
            if not written:
                raise BrokenPipeError(f"encoder closed early for {self.output}")
            view = view[written:]

    def close(self, *, publish: bool = True):
        if self.process.stdin:
            self.process.stdin.close()
        stderr = ""
        if self.process.stderr:
            stderr = self.process.stderr.read().decode("utf-8", "replace")
            self.process.stderr.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(
                f"ffmpeg encode failed ({return_code}) for {self.output}: "
                f"{stderr.strip()}")
        if publish:
            self.publish()

    def publish(self):
        """Atomically publish without replacing a concurrently-created file."""
        try:
            os.link(self.incomplete, self.output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace output: {self.output}") from exc
        self.incomplete.unlink()

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.stderr and not self.process.stderr.closed:
            self.process.stderr.close()
        self.incomplete.unlink(missing_ok=True)


def remove_flat_staging_directory(path: Path | None):
    """Remove only a flat staging directory created by this invocation."""
    if path is None or not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"unexpected frame staging path: {path}")
    for child in path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise RuntimeError(f"unexpected entry in frame staging: {child}")
        child.unlink()
    path.rmdir()


def publish_file_no_clobber(staging: Path, output: Path):
    """Publish a same-filesystem staged file without an overwrite race."""
    try:
        os.link(staging, output)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to replace output: {output}") from exc
    staging.unlink()


def clamp_box(box: list[float]) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(1.0, x2)
    y1, y2 = max(0.0, y1), min(1.0, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def expand_box(box: list[float], x_fraction: float,
               top_fraction: float, bottom_fraction: float) -> list[float] | None:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return clamp_box([
        x1 - width * x_fraction,
        y1 - height * top_fraction,
        x2 + width * x_fraction,
        y2 + height * bottom_fraction,
    ])


def circular_box(box_px: list[float], start_x: float, full_width: int,
                 full_height: int) -> list[list[float]]:
    """Map a crop-local pixel box onto a horizontally circular full frame."""
    x1, y1, x2, y2 = box_px
    width = max(0.0, x2 - x1)
    if width <= 0 or y2 <= y1:
        return []
    start = (start_x + x1) % full_width
    y1n = max(0.0, y1 / full_height)
    y2n = min(1.0, y2 / full_height)
    end = start + width
    if end <= full_width:
        return [[start / full_width, y1n, end / full_width, y2n]]
    return [
        [start / full_width, y1n, 1.0, y2n],
        [0.0, y1n, (end - full_width) / full_width, y2n],
    ]


def box_iom(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denominator = min(area_a, area_b)
    return 0.0 if denominator <= 0 else intersection / denominator


def merge_detections(detections: list[dict], threshold: float = 0.45) -> list[dict]:
    """Union substantially overlapping same-category regions, conservatively."""
    ordered = sorted(
        detections, key=lambda item: float(item.get("confidence", 1.0)),
        reverse=True)
    merged: list[dict] = []
    for detection in ordered:
        current = dict(detection)
        current["box"] = list(detection["box"])
        for existing in merged:
            if existing["category"] != current["category"]:
                continue
            if box_iom(existing["box"], current["box"]) < threshold:
                continue
            a, b = existing["box"], current["box"]
            existing["box"] = [
                min(a[0], b[0]), min(a[1], b[1]),
                max(a[2], b[2]), max(a[3], b[3]),
            ]
            existing["confidence"] = max(
                float(existing.get("confidence", 1.0)),
                float(current.get("confidence", 1.0)))
            sources = set(existing.get("contributing_frames", []))
            sources.update(current.get("contributing_frames", []))
            existing["contributing_frames"] = sorted(sources)
            break
        else:
            merged.append(current)
    return merged


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (new_width, new_height),
                         interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_width) / 2, (size - new_height) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1])
    return tensor[None].astype(np.float32) / 255.0, ratio, dw, dh


class PlateDetector:
    def __init__(self, model: Path, score_threshold: float):
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(16, os.cpu_count() or 1))
        self.session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or shape[0] != 1 or shape[2] != shape[3]:
            raise ValueError(f"unexpected plate model input shape: {shape}")
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        self.size = int(shape[2])
        self.score_threshold = score_threshold

    def _tile_detections(self, tile: np.ndarray) -> list[tuple[list[float], float]]:
        tensor, ratio, dw, dh = letterbox(tile, self.size)
        predictions = self.session.run(
            [self.output_name], {self.input_name: tensor})[0]
        output = []
        for prediction in predictions:
            if len(prediction) < 7 or float(prediction[6]) < self.score_threshold:
                continue
            x1, y1, x2, y2 = map(float, prediction[1:5])
            box = [
                (x1 - dw) / ratio,
                (y1 - dh) / ratio,
                (x2 - dw) / ratio,
                (y2 - dh) / ratio,
            ]
            output.append((box, float(prediction[6])))
        return output

    def detect(self, frame: np.ndarray) -> list[dict]:
        height, width = frame.shape[:2]
        tile_width = round(width * 0.30)
        tile_height = round(height * 0.60)
        x_centers = np.linspace(0, width, 4, endpoint=False)
        y_starts = (0, height - tile_height)
        detections = []
        for center in x_centers:
            start_x = round(center - tile_width / 2)
            indices = np.arange(start_x, start_x + tile_width) % width
            for start_y in y_starts:
                tile = frame[
                    start_y:start_y + tile_height, indices.astype(int)]
                for box, confidence in self._tile_detections(tile):
                    box[1] += start_y
                    box[3] += start_y
                    # Give the blur margin around the physical plate, not just
                    # the model's tight glyph-bearing rectangle.
                    plate_width = box[2] - box[0]
                    plate_height = box[3] - box[1]
                    box = [
                        box[0] - 0.20 * plate_width,
                        box[1] - 0.35 * plate_height,
                        box[2] + 0.20 * plate_width,
                        box[3] + 0.35 * plate_height,
                    ]
                    for mapped in circular_box(
                            box, start_x, width, height):
                        clean = clamp_box(mapped)
                        if clean:
                            detections.append({
                                "category": "license_plate",
                                "source": "yolov9_plate",
                                "confidence": round(confidence, 6),
                                "box": clean,
                            })
        return merge_detections(detections)


def read_manual_regions(path: Path | None) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("manual regions JSON must contain a list")
    regions = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"manual region {index} is not an object")
        box = item.get("box")
        if (not isinstance(box, list) or len(box) != 4
                or any(not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) for value in box)):
            raise ValueError(f"manual region {index} has an invalid box")
        clean = clamp_box([float(value) for value in box])
        if clean is None or clean != [float(value) for value in box]:
            raise ValueError(
                f"manual region {index} must be ordered and within [0, 1]")
        start = float(item.get("start_s", 0.0))
        end_value = item.get("end_s")
        end = None if end_value is None else float(end_value)
        if (not math.isfinite(start) or start < 0
                or (end is not None
                    and (not math.isfinite(end) or end <= start))):
            raise ValueError(f"manual region {index} has an invalid time range")
        regions.append({
            "id": item.get("id", f"manual_{index}"),
            "category": item.get("category", "face"),
            "box": clean,
            "start_s": start,
            "end_s": end,
            "reason": item.get("reason", "human-specified privacy region"),
        })
    return regions


def densify(raw_frames: list[dict], temporal_radius: int,
            manual_regions: list[dict], fps: float) -> list[dict]:
    if temporal_radius < 0:
        raise ValueError("temporal radius must be nonnegative")
    detections = [[] for _ in raw_frames]
    for source_index, frame in enumerate(raw_frames):
        for detection in frame["detections"]:
            for target_index in range(
                    max(0, source_index - temporal_radius),
                    min(len(raw_frames), source_index + temporal_radius + 1)):
                propagated = dict(detection)
                propagated["box"] = list(detection["box"])
                propagated["contributing_frames"] = [source_index]
                detections[target_index].append(propagated)
    for frame_index in range(len(raw_frames)):
        source_time = raw_frames[frame_index].get(
            "source_video_t_s", frame_index / fps)
        for region in manual_regions:
            if (region["start_s"] <= source_time
                    and (region["end_s"] is None
                         or source_time < region["end_s"])):
                detections[frame_index].append({
                    "category": region["category"],
                    "source": "manual_region",
                    "manual_region_id": region["id"],
                    "confidence": 1.0,
                    "box": list(region["box"]),
                    "contributing_frames": [frame_index],
                })
    output = []
    for frame_index, frame_detections in enumerate(detections):
        row = {
            key: value for key, value in raw_frames[frame_index].items()
            if key != "detections"
        }
        row["detections"] = merge_detections(frame_detections)
        output.append(row)
    return output


def write_jsonl(path: Path, rows: list[dict]):
    temporary = path.with_name(path.name + ".incomplete")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"refusing to replace ledger: {path}")
    # Exclusive creation matters even though the final destination was
    # checked above: two long scans can both pass that check before either
    # reaches publication.  Only one may become the ledger producer.
    with temporary.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def parse_jsonl_bytes(payload: bytes) -> list[dict]:
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()
            if line.strip()]
    for index, row in enumerate(rows):
        if row.get("frame_index") != index:
            raise ValueError(
                f"non-contiguous ledger at row {index}: {row.get('frame_index')}")
    return rows


def read_jsonl(path: Path) -> list[dict]:
    return parse_jsonl_bytes(read_regular_file_bytes(path))


def detection_counts(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    frames: dict[str, int] = {}
    for row in rows:
        present = set()
        for detection in row["detections"]:
            category = detection["category"]
            counts[category] = counts.get(category, 0) + 1
            present.add(category)
        for category in present:
            frames[category] = frames.get(category, 0) + 1
    return {category: {
        "regions": count,
        "frames": frames.get(category, 0),
    } for category, count in sorted(counts.items())}


def scan(args) -> int:
    source = args.source.resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source must be a regular file: {source}")
    output_dir = args.output_dir
    if args.temporal_radius_frames < 0:
        raise ValueError("temporal radius must be nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "detections.raw.jsonl"
    ledger_path = output_dir / "detections.jsonl"
    manifest_path = output_dir / "anonymization_manifest.json"
    for path in (raw_path, ledger_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace existing output: {path}")

    source_info = probe_video(source)
    scan_width = args.scan_width
    scan_height = round(scan_width * source_info["height"] / source_info["width"])
    if scan_width % 2 or scan_height % 2:
        raise ValueError("scan dimensions must be even")
    clip = clip_metadata(
        source_info, args.output_fps, args.start_s, args.end_s)
    expected = clip["frame_count"]
    manual = read_manual_regions(args.manual_regions)

    print(f"scanning {source}: {expected} frames at {args.output_fps:g} fps, "
          f"source [{clip['start_s']:.3f}, {clip['end_s']:.3f}) s, "
          f"{scan_width}x{scan_height}", flush=True)
    started = time.monotonic()
    plate = PlateDetector(args.plate_model, args.plate_threshold)
    raw_frames = []
    last_update = started
    for index, frame in enumerate(RawVideoReader(
            source, source_info, args.output_fps, scan_width, scan_height,
            clip["start_frame"], clip["end_frame_exclusive"])):
        found = plate.detect(frame)
        source_frame_index = clip["start_frame"] + index
        raw_frames.append({
            "frame_index": index,
            "video_t_s": round(index / args.output_fps, 6),
            "source_frame_index": source_frame_index,
            "source_video_t_s": round(
                source_frame_index / args.output_fps, 6),
            "detections": merge_detections(found),
        })
        now = time.monotonic()
        if now - last_update >= 30:
            elapsed = now - started
            rate = (index + 1) / elapsed
            remaining = max(0, expected - index - 1) / max(rate, 1e-9)
            print(f"  {index + 1}/{expected} ({rate:.2f} fps, "
                  f"~{remaining / 60:.1f} min remaining)", flush=True)
            last_update = now
    if len(raw_frames) != expected:
        raise RuntimeError(
            f"decoded {len(raw_frames)} frames; expected {expected} from ffprobe")

    applied = densify(
        raw_frames, args.temporal_radius_frames, manual, args.output_fps)
    write_jsonl(raw_path, raw_frames)
    write_jsonl(ledger_path, applied)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "scanned",
        "review": {
            "status": "pending",
            "instruction": (
                "Watch review.mp4 end to end. Every identifiable face or "
                "license plate must be blurred and enclosed by an outline."
            ),
        },
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "capture_fps": args.capture_fps,
            "clip": clip,
            **source_info,
        },
        "output_fps": args.output_fps,
        "scan_resolution": [scan_width, scan_height],
        "frame_count": len(applied),
        "detectors": {
            "license_plate": {
                "model": str(args.plate_model.resolve()),
                "model_sha256": sha256_file(args.plate_model),
                "threshold": args.plate_threshold,
                "tiling": "4x2 overlapping circular equirectangular tiles",
                "mask": "expanded rectangle",
            },
        },
        "temporal_radius_frames": args.temporal_radius_frames,
        "manual_regions": manual,
        "raw_detection_counts": detection_counts(raw_frames),
        "applied_detection_counts": detection_counts(applied),
        "files": {
            "raw_ledger": {
                "path": raw_path.name,
                "sha256": sha256_file(raw_path),
            },
            "applied_ledger": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
            },
        },
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    elapsed = time.monotonic() - started
    print(f"wrote {ledger_path} in {elapsed / 60:.1f} min: "
          f"{manifest['applied_detection_counts']}", flush=True)
    return 0


def apply_policy(args) -> int:
    """Reapply temporal/manual policy to a retained raw detection ledger."""
    if args.temporal_radius_frames < 0:
        raise ValueError("temporal radius must be nonnegative")
    input_dir = args.scan_dir.resolve()
    input_manifest_path = input_dir / "anonymization_manifest.json"
    manifest_bytes = read_regular_file_bytes(input_manifest_path)
    parent_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported scan manifest schema")
    if manifest.get("status") not in ("scanned", "rendered_pending_review"):
        raise ValueError("policy parent is not a completed scan")
    parent_fps = manifest.get("output_fps")
    if (isinstance(parent_fps, bool)
            or not isinstance(parent_fps, (int, float))
            or not math.isfinite(float(parent_fps)) or parent_fps <= 0):
        raise ValueError("policy parent has an invalid output fps")
    parent_frame_count = manifest.get("frame_count")
    if (isinstance(parent_frame_count, bool)
            or not isinstance(parent_frame_count, int)
            or parent_frame_count < 0):
        raise ValueError("policy parent has an invalid frame count")
    raw_name = manifest["files"]["raw_ledger"]["path"]
    if Path(raw_name).name != raw_name:
        raise ValueError("raw ledger path in manifest must be a basename")
    input_raw_path = input_dir / raw_name
    raw_bytes = read_regular_file_bytes(input_raw_path)
    if (hashlib.sha256(raw_bytes).hexdigest()
            != manifest["files"]["raw_ledger"]["sha256"]):
        raise ValueError("raw detection ledger changed after scan")
    raw_frames = parse_jsonl_bytes(raw_bytes)
    if len(raw_frames) != parent_frame_count:
        raise ValueError("raw detection ledger length differs from scan manifest")
    manual = read_manual_regions(args.manual_regions)
    applied = densify(
        raw_frames, args.temporal_radius_frames, manual,
        float(parent_fps))

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # mkdir is the atomic no-clobber claim for the complete revision path.
    # The manifest is written last, so a crash cannot leave an apparently
    # complete policy revision.
    output_dir.mkdir()
    raw_path = output_dir / "detections.raw.jsonl"
    ledger_path = output_dir / "detections.jsonl"
    manifest_path = output_dir / "anonymization_manifest.json"
    manifest_staging = output_dir / "anonymization_manifest.json.incomplete"
    try:
        with raw_path.open("xb") as stream:
            stream.write(raw_bytes)
        write_jsonl(ledger_path, applied)
        current_manifest = read_regular_file_bytes(input_manifest_path)
        if hashlib.sha256(current_manifest).hexdigest() != parent_sha256:
            raise ValueError("scan manifest changed during policy application")
        manifest.pop("render", None)
        manifest.update({
            "status": "scanned",
            "review": {
                "status": "pending",
                "instruction": (
                    "Watch review.mp4 end to end and inspect ambiguous scenes "
                    "at native resolution. Every identifiable face or license "
                    "plate must be blurred and enclosed by an outline."),
            },
            "temporal_radius_frames": args.temporal_radius_frames,
            "manual_regions": manual,
            "raw_detection_counts": detection_counts(raw_frames),
            "applied_detection_counts": detection_counts(applied),
            "files": {
                "raw_ledger": {
                    "path": raw_path.name,
                    "sha256": sha256_file(raw_path),
                },
                "applied_ledger": {
                    "path": ledger_path.name,
                    "sha256": sha256_file(ledger_path),
                },
            },
            "policy_parent": {
                "manifest": str(input_manifest_path),
                "manifest_sha256": parent_sha256,
            },
            "created_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "argv": list(sys.argv),
        })
        with manifest_staging.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        os.replace(manifest_staging, manifest_path)
    except BaseException:
        for path in (
                manifest_staging, manifest_path,
                ledger_path.with_name(ledger_path.name + ".incomplete"),
                ledger_path, raw_path):
            path.unlink(missing_ok=True)
        # Only remove our exclusively claimed directory, and only if no
        # unexpected concurrent writer added content to it.
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise
    print(f"wrote policy revision {output_dir}: "
          f"{manifest['applied_detection_counts']}", flush=True)
    return 0


def _pixel_bounds(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, math.floor(box[0] * width)))
    y1 = max(0, min(height - 1, math.floor(box[1] * height)))
    x2 = max(x1 + 1, min(width, math.ceil(box[2] * width)))
    y2 = max(y1 + 1, min(height, math.ceil(box[3] * height)))
    return x1, y1, x2, y2


def strong_blur(frame: np.ndarray, detections: list[dict]):
    """Apply an irreversible Gaussian-smoothed coarse mosaic in-place.

    Smooth the deliberately tiny mosaic before expanding it.  Applying a
    large Gaussian kernel after expansion has the same privacy intent but is
    needlessly expensive for fail-safe regions spanning much of an 8K frame.
    """
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = _pixel_bounds(detection["box"], width, height)
        region = frame[y1:y2, x1:x2]
        if not region.size:
            continue
        region_height, region_width = region.shape[:2]
        block = max(8, min(region_width, region_height) // 10)
        small_width = max(1, math.ceil(region_width / block))
        small_height = max(1, math.ceil(region_height / block))
        mosaic = cv2.resize(
            region, (small_width, small_height), interpolation=cv2.INTER_AREA)
        sigma = max(1.0, min(small_width, small_height) / 6.0)
        mosaic = cv2.GaussianBlur(
            mosaic, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mosaic = cv2.resize(
            mosaic, (region_width, region_height), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = mosaic


def load_extraction_plan(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    required = {"frame_index", "frame_file"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"extraction plan must contain {sorted(required)}: {path}")
    plan = {}
    for row in rows:
        index = int(row["frame_index"])
        name = row["frame_file"]
        if index < 0 or Path(name).name != name or not name.lower().endswith(".jpg"):
            raise ValueError(f"invalid extraction row: {row}")
        if index in plan:
            raise ValueError(f"duplicate extraction frame index {index}")
        plan[index] = name
    return plan


def _draw_review(frame: np.ndarray, detections: list[dict], frame_index: int,
                 fps: float, source_width: int, source_time_s: float,
                 source_frame_index: int) -> np.ndarray:
    height = round(source_width * frame.shape[0] / frame.shape[1])
    review = cv2.resize(frame, (source_width, height), interpolation=cv2.INTER_AREA)
    colors = {
        "face": (0, 0, 255),
        "license_plate": (0, 215, 255),
    }
    for detection in detections:
        color = ((0, 255, 0) if detection.get("source") == "manual_region"
                 else colors.get(detection["category"], (255, 0, 255)))
        x1, y1, x2, y2 = _pixel_bounds(
            detection["box"], review.shape[1], review.shape[0])
        cv2.rectangle(review, (x1, y1), (x2, y2), color, 3)
        label = detection["category"]
        if detection.get("source") == "manual_region":
            label += ":manual"
        cv2.putText(review, label, (x1, max(22, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    output_time = frame_index / fps
    text = (f"SOURCE {source_time_s:8.3f}s F{source_frame_index:06d}  "
            f"OUTPUT {output_time:8.3f}s F{frame_index:06d}  "
            f"REGIONS {len(detections)}")
    cv2.rectangle(review, (0, 0), (min(review.shape[1], 1120), 42),
                  (0, 0, 0), -1)
    cv2.putText(review, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return review


def _write_review_html(path: Path, review_name: str, full_video_name: str,
                       ledger_name: str, speedup: float,
                       source_start_s: float, output_fps: float,
                       full_width: int, full_height: int):
    html = """<!doctype html>
<meta charset="utf-8">
<title>Anonymization review</title>
<style>
body { max-width: 1400px; margin: 2rem auto; font: 16px sans-serif; background:#111; color:#eee; }
video { width:100%; background:#000; }
button, input { font:inherit; margin:.35rem; }
code { color:#9ee; }
#nativePane { width:100%; height:72vh; overflow:auto; border:2px solid #777; background:#000; }
#nativeStage { position:relative; }
#full { position:absolute; left:0; top:0; max-width:none; }
#overlay { position:absolute; left:0; top:0; pointer-events:none; }
#nativeStatus { color:#f7d774; }
</style>
<h1>Anonymization review</h1>
<h2>1. Full-panorama overview</h2>
<p>Watch the whole accelerated overview. Red outlines are detected faces, yellow
outlines are detected plates, and green outlines are manual regions. Report
any identifiable face or plate that is not blurred and outlined, using the
<code>SOURCE</code> time burned into the upper-left corner.</p>
<video id="v" controls></video>
<p><button onclick="jump(-10)">-10 source s</button>
<button onclick="jump(10)">+10 source s</button>
Review playback: <input id="rate" type="number" min="0.25" max="4" step="0.25" value="1">
<span id="clock"></span> <span id="pointer"></span></p>
<h2>2. Native-resolution inspector (required)</h2>
<p>The overview is downscaled. Pause on crowded, distant, upper/lower, or
otherwise ambiguous scenes and click <em>Sync native view</em>. The scrollable
inspector displays the blurred output at one source pixel per CSS pixel and
draws the exact ledger boxes in the browser. If this browser cannot decode the
HEVC output, <a id="fullLink">open or download it</a> in an HEVC-capable player;
do not approve from the overview alone.</p>
<p><button onclick="syncNative()">Sync native view</button>
<button onclick="toggleNative()">Play/pause native</button>
<button onclick="nativeJump(-1)">-1 s</button>
<button onclick="nativeJump(1)">+1 s</button>
Native playback: <input id="fullRate" type="number" min="0.25" max="4" step="0.25" value="1">
<span id="nativeClock"></span> <span id="nativePointer"></span></p>
<div id="nativeStatus">Loading the detection ledger…</div>
<div id="nativePane"><div id="nativeStage">
  <video id="full" controls></video><svg id="overlay"></svg>
</div></div>
<p>For a correction region, pause and shift-click two opposite corners. Copy
the generated entry into the recording's manual-regions JSON and adjust the
time interval as needed. Rerun into new anonymization, video, and frame-staging
paths; existing evidence is intentionally no-clobber.</p>
<pre id="region"></pre>
<script>
const reviewUrl=__REVIEW_URL__, fullUrl=__FULL_URL__, ledgerUrl=__LEDGER_URL__;
const speedup=__SPEEDUP__, sourceStart=__SOURCE_START__, outputFps=__OUTPUT_FPS__;
const fullWidth=__FULL_WIDTH__, fullHeight=__FULL_HEIGHT__;
const v=document.getElementById('v');
const full=document.getElementById('full'), stage=document.getElementById('nativeStage');
const overlay=document.getElementById('overlay');
v.src=reviewUrl; full.src=fullUrl; document.getElementById('fullLink').href=fullUrl;
stage.style.width=fullWidth+'px'; stage.style.height=fullHeight+'px';
full.style.width=fullWidth+'px'; full.style.height=fullHeight+'px';
overlay.setAttribute('width',fullWidth); overlay.setAttribute('height',fullHeight);
overlay.setAttribute('viewBox',`0 0 ${fullWidth} ${fullHeight}`);
document.getElementById('rate').onchange=e=>v.playbackRate=Number(e.target.value);
document.getElementById('fullRate').onchange=e=>full.playbackRate=Number(e.target.value);
function jump(sourceSeconds) { v.currentTime += sourceSeconds/speedup; }
function nativeJump(seconds) { full.currentTime=Math.max(0,full.currentTime+seconds); }
function syncNative() { full.pause(); full.currentTime=v.currentTime*speedup; drawNative(); }
function toggleNative() { if(full.paused) full.play(); else full.pause(); }
function panNative(p) { const pane=document.getElementById('nativePane');
  pane.scrollTo({left:p[0]*fullWidth-pane.clientWidth/2,
    top:p[1]*fullHeight-pane.clientHeight/2,behavior:'smooth'}); }
v.ontimeupdate=()=>document.getElementById('clock').textContent=
  `source time ${(sourceStart+v.currentTime*speedup).toFixed(3)} s`;
function pointOn(element,e) { const r=element.getBoundingClientRect(); return [
  Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),
  Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]; }
v.onmousemove=e=>{ const p=pointOn(v,e); document.getElementById('pointer').textContent=
  `pointer (${p[0].toFixed(4)}, ${p[1].toFixed(4)})`; };
stage.onmousemove=e=>{ const p=pointOn(stage,e); document.getElementById('nativePointer').textContent=
  `pointer (${p[0].toFixed(5)}, ${p[1].toFixed(5)})`; };
let firstCorner=null;
function correctionCorner(p,t) {
  if(firstCorner===null) { firstCorner=p; document.getElementById('region').textContent=
    'First corner saved; shift-click the opposite corner.'; return; }
  const box=[Math.min(firstCorner[0],p[0]),
    Math.min(firstCorner[1],p[1]),Math.max(firstCorner[0],p[0]),
    Math.max(firstCorner[1],p[1])].map(x=>Number(x.toFixed(6)));
  document.getElementById('region').textContent=JSON.stringify({
    id:'review_correction',category:'face',box:box,
    start_s:Number(Math.max(0,t-1).toFixed(3)),end_s:Number((t+1).toFixed(3)),
    reason:'human review correction'},null,2); firstCorner=null;
}
v.onclick=e=>{ const p=pointOn(v,e); if(e.shiftKey) { e.preventDefault();
  correctionCorner(p,sourceStart+v.currentTime*speedup); }
  else { syncNative(); panNative(p); } };
stage.onclick=e=>{ if(e.shiftKey) { e.preventDefault(); correctionCorner(
  pointOn(stage,e),sourceStart+full.currentTime); } };
let ledger=[];
fetch(ledgerUrl).then(r=>{if(!r.ok) throw Error(r.status); return r.text();})
  .then(text=>{ledger=text.trim().split(/\n/).filter(Boolean).map(JSON.parse);
    document.getElementById('nativeStatus').textContent=
      `Loaded ${ledger.length} frame records; native overlay is active.`; drawNative();})
  .catch(error=>document.getElementById('nativeStatus').textContent=
    `Could not load overlay ledger: ${error}. Do not approve until fixed.`);
function drawNative() {
  document.getElementById('nativeClock').textContent=
    `source time ${(sourceStart+full.currentTime).toFixed(3)} s`;
  overlay.replaceChildren(); if(!ledger.length) return;
  const index=Math.max(0,Math.min(ledger.length-1,Math.round(full.currentTime*outputFps)));
  for(const detection of ledger[index].detections) {
    const b=detection.box, rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('x',b[0]*fullWidth); rect.setAttribute('y',b[1]*fullHeight);
    rect.setAttribute('width',(b[2]-b[0])*fullWidth);
    rect.setAttribute('height',(b[3]-b[1])*fullHeight);
    const color=detection.source==='manual_region'?'#00ff00':
      (detection.category==='face'?'#ff3030':'#ffd700');
    rect.setAttribute('fill','none'); rect.setAttribute('stroke',color);
    rect.setAttribute('stroke-width','8'); overlay.appendChild(rect);
  }
}
full.ontimeupdate=drawNative;
</script>
"""
    replacements = {
        "__REVIEW_URL__": json.dumps(review_name),
        "__FULL_URL__": json.dumps(full_video_name),
        "__LEDGER_URL__": json.dumps(ledger_name),
        "__SPEEDUP__": repr(float(speedup)),
        "__SOURCE_START__": repr(float(source_start_s)),
        "__OUTPUT_FPS__": repr(float(output_fps)),
        "__FULL_WIDTH__": str(int(full_width)),
        "__FULL_HEIGHT__": str(int(full_height)),
    }
    for marker, value in replacements.items():
        html = html.replace(marker, value)
    path.write_text(html)


def render(args) -> int:
    source = args.source.resolve()
    source_info = probe_video(source)
    manifest_path = args.output_dir / "anonymization_manifest.json"
    ledger_path = args.output_dir / "detections.jsonl"
    manifest = json.loads(manifest_path.read_text())
    ledger = read_jsonl(ledger_path)
    if not math.isclose(
            manifest["output_fps"], args.output_fps, abs_tol=1e-9):
        raise ValueError("render output fps differs from detection scan")
    clip = manifest["source"].get("clip") or clip_metadata(
        source_info, args.output_fps)
    expected = int(clip["frame_count"])
    if len(ledger) != expected:
        raise ValueError(
            f"ledger has {len(ledger)} frames but source resolves to {expected}")
    if manifest["source"]["sha256"] != sha256_file(source):
        raise ValueError("source video changed after detection scan")
    if manifest["files"]["applied_ledger"]["sha256"] != sha256_file(ledger_path):
        raise ValueError("detection ledger changed after scan")

    extraction_plan_sha256 = None
    if args.extraction_plan:
        plan_hash_before = sha256_file(args.extraction_plan)
        plan = load_extraction_plan(args.extraction_plan)
        extraction_plan_sha256 = sha256_file(args.extraction_plan)
        if plan_hash_before != extraction_plan_sha256:
            raise ValueError("extraction plan changed while it was loaded")
    else:
        plan = {}
    if plan and args.frames_dir is None:
        raise ValueError("--frames_dir is required with --extraction_plan")
    missing_indices = sorted(set(plan) - set(range(expected)))
    if missing_indices:
        raise ValueError(f"extraction indices outside video: {missing_indices[:5]}")
    frames_staging = None
    if args.frames_dir:
        if args.frames_dir.exists():
            if (args.frames_dir.is_symlink() or not args.frames_dir.is_dir()
                    or any(args.frames_dir.iterdir())):
                raise FileExistsError(
                    f"frames path is not an empty directory: {args.frames_dir}")
        else:
            args.frames_dir.parent.mkdir(parents=True, exist_ok=True)
        frames_staging = args.frames_dir.with_name(
            args.frames_dir.name + ".incomplete")
        if frames_staging.exists() or frames_staging.is_symlink():
            raise FileExistsError(
                f"frame staging directory already exists: {frames_staging}")
        frames_staging.mkdir()

    encoder = getattr(args, "encoder", "software")
    encoder_profile = video_encoder_profile(encoder)
    output = args.output_video
    review_path = args.output_dir / "review.mp4"
    review_html_path = args.output_dir / "review.html"
    review_html_staging = args.output_dir / "review.html.incomplete"
    manifest_staging = args.output_dir / "anonymization_manifest.json.incomplete"
    for path in (review_html_path, review_html_staging, manifest_staging):
        if path.exists() or path.is_symlink():
            remove_flat_staging_directory(frames_staging)
            raise FileExistsError(f"refusing to replace render output: {path}")
    review_height = round(
        args.review_width * source_info["height"] / source_info["width"])
    started = time.monotonic()
    last_update = started
    extracted = 0
    rendered = 0
    extracted_frame_sha256 = {}
    output_writer = None
    review_writer = None
    published_files = []
    published_frames = False
    try:
        output_writer = RawVideoWriter(
            output, source_info["width"], source_info["height"],
            args.output_fps, encoder=encoder)
        review_writer = RawVideoWriter(
            review_path, args.review_width, review_height,
            args.output_fps * args.review_speedup, review=True,
            encoder=encoder)
        for index, frame in enumerate(RawVideoReader(
                source, source_info, args.output_fps,
                start_frame=int(clip["start_frame"]),
                end_frame=int(clip["end_frame_exclusive"]))):
            row = ledger[index]
            detections = row["detections"]
            strong_blur(frame, detections)
            if index in plan:
                destination = frames_staging / plan[index]
                if destination.exists() or destination.is_symlink():
                    raise FileExistsError(
                        f"refusing to replace extracted frame: {destination}")
                ok = cv2.imwrite(
                    str(destination), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    raise RuntimeError(f"failed to write {destination}")
                extracted_frame_sha256[destination.name] = sha256_file(
                    destination)
                extracted += 1
            output_writer.write(frame)
            review_writer.write(_draw_review(
                frame, detections, index, args.output_fps, args.review_width,
                float(row.get("source_video_t_s", index / args.output_fps)),
                int(row.get("source_frame_index", index))))
            now = time.monotonic()
            if now - last_update >= 30:
                elapsed = now - started
                rate = (index + 1) / elapsed
                remaining = max(0, expected - index - 1) / max(rate, 1e-9)
                print(f"  rendered {index + 1}/{expected} ({rate:.2f} fps, "
                      f"~{remaining / 60:.1f} min remaining)", flush=True)
                last_update = now
            rendered = index + 1
        if rendered != expected:
            raise RuntimeError(f"rendered {rendered}, expected {expected}")
        if (args.extraction_plan
                and sha256_file(args.extraction_plan)
                != extraction_plan_sha256):
            raise ValueError("extraction plan changed during render")
        if extracted != len(plan):
            raise RuntimeError(
                f"extracted {extracted}/{len(plan)} requested frames")

        # Finish both encoders before publishing either output. This keeps a
        # late review-encoder error from orphaning an apparently valid video.
        output_writer.close(publish=False)
        review_writer.close(publish=False)
        output_info = probe_video(output_writer.incomplete)
        review_info = probe_video(review_writer.incomplete)
        if (output_info["width"], output_info["height"]) != (
                source_info["width"], source_info["height"]):
            raise RuntimeError("rendered video resolution changed")
        if not math.isclose(
                output_info["media_fps"], args.output_fps, abs_tol=1e-9):
            raise RuntimeError("rendered video is not at the requested frame rate")
        if output_info["nb_frames"] != expected:
            raise RuntimeError(
                f"rendered video has {output_info['nb_frames']} frames, "
                f"expected {expected}")
        if review_info["nb_frames"] != expected:
            raise RuntimeError(
                f"review video has {review_info['nb_frames']} frames, "
                f"expected {expected}")
        if (review_info["width"], review_info["height"]) != (
                args.review_width, review_height):
            raise RuntimeError("review video resolution changed")
        if not math.isclose(
                review_info["media_fps"],
                args.output_fps * args.review_speedup, abs_tol=1e-9):
            raise RuntimeError("review video frame rate changed")

        output_video_sha256 = sha256_file(output_writer.incomplete)
        review_video_sha256 = sha256_file(review_writer.incomplete)
        full_video_name = os.path.relpath(output, args.output_dir)
        _write_review_html(
            review_html_staging, review_path.name, full_video_name,
            ledger_path.name, args.review_speedup, float(clip["start_s"]),
            args.output_fps, source_info["width"], source_info["height"])
        render_completed_utc = datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds")
        manifest.update({
            "completed_utc": render_completed_utc,
            "status": "rendered_pending_review",
            "review": {
                **manifest["review"],
                "status": "pending",
                "video": review_path.name,
                "html": review_html_path.name,
                "speedup": args.review_speedup,
                "overview_width": args.review_width,
                "native_resolution_inspector": True,
            },
            "render": {
                "output_video": str(output.resolve()),
                "output_video_sha256": output_video_sha256,
                "output_video_info": output_info,
                "review_video_sha256": review_video_sha256,
                "review_html_sha256": sha256_file(review_html_staging),
                "encoder": encoder_profile,
                "blur": "expanded rectangular Gaussian-smoothed coarse mosaic",
                "extraction_plan": (str(args.extraction_plan.resolve())
                                    if args.extraction_plan else None),
                "extraction_plan_sha256": extraction_plan_sha256,
                "extracted_frames": extracted,
                "extracted_frame_sha256": extracted_frame_sha256,
                "created_utc": render_completed_utc,
                "argv": list(sys.argv),
            },
        })
        manifest_staging.write_text(json.dumps(manifest, indent=2) + "\n")

        # All expensive work and validation succeeded. Publish the new files,
        # then replace the scan manifest last as the transaction's commit point.
        output_writer.publish()
        published_files.append(output)
        review_writer.publish()
        published_files.append(review_path)
        if frames_staging is not None:
            if args.frames_dir.exists():
                args.frames_dir.rmdir()
            os.rename(frames_staging, args.frames_dir)
            frames_staging = None
            published_frames = True
        publish_file_no_clobber(review_html_staging, review_html_path)
        published_files.append(review_html_path)
        os.replace(manifest_staging, manifest_path)
    except BaseException:
        if output_writer is not None:
            output_writer.abort()
        if review_writer is not None:
            review_writer.abort()
        review_html_staging.unlink(missing_ok=True)
        manifest_staging.unlink(missing_ok=True)
        if published_frames:
            remove_flat_staging_directory(args.frames_dir)
        else:
            remove_flat_staging_directory(frames_staging)
        for path in reversed(published_files):
            path.unlink(missing_ok=True)
        raise

    print(f"published {output} and {review_path}; human review is pending", flush=True)
    return 0


def mark_review(args) -> int:
    manifest_path = args.output_dir / "anonymization_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "rendered_pending_review":
        raise ValueError("only a rendered_pending_review manifest can be reviewed")
    evidence = (
        (Path(manifest["render"]["output_video"]),
         manifest["render"]["output_video_sha256"], "blurred output video"),
        (args.output_dir / manifest["review"]["video"],
         manifest["render"]["review_video_sha256"], "review video"),
        (args.output_dir / manifest["review"]["html"],
         manifest["render"]["review_html_sha256"], "review HTML"),
        (args.output_dir / manifest["files"]["applied_ledger"]["path"],
         manifest["files"]["applied_ledger"]["sha256"], "applied ledger"),
    )
    for path, expected_sha256, label in evidence:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is not a regular file: {path}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"{label} changed after render: {path}")
    decision_path = args.output_dir / "review_decision.json"
    if decision_path.exists() or decision_path.is_symlink():
        raise FileExistsError(
            f"refusing to replace review decision: {decision_path}")
    decision = {
        "schema_version": 1,
        "anonymization_manifest": manifest_path.name,
        "anonymization_manifest_sha256": sha256_file(manifest_path),
        "status": args.decision,
        "reviewer": args.reviewer,
        "note": args.note,
        "reviewed_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = decision_path.with_name(decision_path.name + ".incomplete")
    with temporary.open("x") as stream:
        stream.write(json.dumps(decision, indent=2) + "\n")
    os.replace(temporary, decision_path)
    print(f"marked {args.output_dir}: {args.decision}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan", help="detect license-plate candidates")
    scan_parser.add_argument("--source", type=Path, required=True)
    scan_parser.add_argument("--output_dir", type=Path, required=True)
    scan_parser.add_argument("--plate_model", type=Path, required=True)
    scan_parser.add_argument("--manual_regions", type=Path)
    scan_parser.add_argument("--capture_fps", type=float, required=True,
                             help="camera capture rate before any prior export")
    scan_parser.add_argument("--output_fps", type=float, default=DEFAULT_OUTPUT_FPS)
    scan_parser.add_argument(
        "--start_s", type=float, default=0.0,
        help="source-video start time; included on the output frame grid")
    scan_parser.add_argument(
        "--end_s", type=float,
        help="source-video end time, exclusive; defaults to end of input")
    scan_parser.add_argument("--scan_width", type=int, default=DEFAULT_SCAN_WIDTH)
    scan_parser.add_argument("--plate_threshold", type=float, default=0.20)
    scan_parser.add_argument("--temporal_radius_frames", type=int, default=1)
    scan_parser.set_defaults(func=scan)

    policy_parser = subparsers.add_parser(
        "apply-policy",
        help="reapply temporal/manual policy without rerunning detectors")
    policy_parser.add_argument("--scan_dir", type=Path, required=True)
    policy_parser.add_argument("--output_dir", type=Path, required=True)
    policy_parser.add_argument("--manual_regions", type=Path)
    policy_parser.add_argument("--temporal_radius_frames", type=int, default=1)
    policy_parser.set_defaults(func=apply_policy)

    render_parser = subparsers.add_parser(
        "render", help="apply a completed ledger to full-resolution video")
    render_parser.add_argument("--source", type=Path, required=True)
    render_parser.add_argument("--output_dir", type=Path, required=True)
    render_parser.add_argument("--output_video", type=Path, required=True)
    render_parser.add_argument("--output_fps", type=float, default=DEFAULT_OUTPUT_FPS)
    render_parser.add_argument("--review_width", type=int, default=DEFAULT_REVIEW_WIDTH)
    render_parser.add_argument("--review_speedup", type=float,
                               default=DEFAULT_REVIEW_SPEEDUP)
    render_parser.add_argument("--extraction_plan", type=Path)
    render_parser.add_argument("--frames_dir", type=Path)
    render_parser.add_argument("--jpeg_quality", type=int, default=95)
    render_parser.add_argument(
        "--encoder", choices=VIDEO_ENCODERS, default="software",
        help=("video encoder backend; nvenc requires FFmpeg NVENC support and "
              "a compatible NVIDIA GPU"))
    render_parser.set_defaults(func=render)

    review_parser = subparsers.add_parser(
        "mark-review", help="record a human review decision")
    review_parser.add_argument("--output_dir", type=Path, required=True)
    review_parser.add_argument("--decision", choices=("approved", "needs_corrections"),
                               required=True)
    review_parser.add_argument("--reviewer", required=True)
    review_parser.add_argument("--note", default="")
    review_parser.set_defaults(func=mark_review)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "output_fps", 1) <= 0:
        raise ValueError("--output_fps must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
