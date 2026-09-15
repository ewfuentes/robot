"""Plan GPS-tagged frame extraction for self-collected 360 recordings.

The input is a collection JSON describing reusable GPS sources and one or more
recordings.  ``plan`` resolves each recording's video clock onto its sensor
track, samples a distance grid, and writes ``frames_gps.csv`` before any costly
full-resolution render begins.  ``anonymize_video render`` can consume that
CSV directly and save the requested blurred frames during its one full decode.
``finalize`` then binds the anonymized output and privacy ledger into the extra
metadata consumed by ``ingest_selfcollect``.

All video-to-sensor anchors are expressed in seconds.  Capture frame rate is
kept separately from media frame rate so a prior 30 -> 3 fps export cannot
silently change the meaning of a sync observation made on the original.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.dataset_tools import anonymize_video
from experimental.overhead_matching.swag.farfield.dataset_tools import fit_decoder


EARTH_RADIUS_M = 6371008.8
FRAME_COLUMNS = [
    "frame_id", "idx", "frame_index", "video_t_s", "source_video_t_s",
    "sensor_elapsed_s", "dist_m", "route_distance_m", "latitude",
    "longitude", "latitude_raw", "longitude_raw", "altitude_m", "speed_mps",
    "course_deg", "gps_quality", "fix_gap_s", "fix_dt_s",
    "frame_time_error_s", "source_capture_fps", "frame_file",
]
FLOAT_FORMATS = {
    "video_t_s": 6,
    "source_video_t_s": 6,
    "sensor_elapsed_s": 6,
    "dist_m": 2,
    "route_distance_m": 2,
    "latitude": 9,
    "longitude": 9,
    "latitude_raw": 9,
    "longitude_raw": 9,
    "altitude_m": 2,
    "speed_mps": 3,
    "course_deg": 3,
    "fix_gap_s": 3,
    "fix_dt_s": 3,
    "frame_time_error_s": 6,
    "source_capture_fps": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_utc(value: datetime.datetime) -> str:
    return value.astimezone(datetime.timezone.utc).isoformat().replace(
        "+00:00", "Z")


class Track:
    """A smoothed one-second track with honest fix-distance diagnostics."""

    def __init__(self, rows: list[dict], *, sigma_s: float,
                 velocity_limit_mps: float):
        if len(rows) < 2:
            raise ValueError("GPS source has fewer than two positioned fixes")
        rows = sorted(rows, key=lambda row: row["elapsed_s"])
        time_values = np.array([row["elapsed_s"] for row in rows], dtype=float)
        latitude = np.array([row["latitude"] for row in rows], dtype=float)
        longitude = np.array([row["longitude"] for row in rows], dtype=float)
        altitude = np.array([row.get("altitude_m", math.nan) for row in rows])
        speed = np.array([row.get("speed_mps", math.nan) for row in rows])
        if np.any(np.diff(time_values) <= 0):
            unique = np.concatenate([[True], np.diff(time_values) > 0])
            time_values, latitude, longitude, altitude, speed = (
                values[unique] for values in
                (time_values, latitude, longitude, altitude, speed))

        self.latitude_origin = float(np.mean(latitude))
        keep = self._kinematic_gate(
            time_values, latitude, longitude, velocity_limit_mps)
        self.dropped_fixes = int(np.count_nonzero(~keep))
        time_values, latitude, longitude, altitude, speed = (
            values[keep] for values in
            (time_values, latitude, longitude, altitude, speed))
        if len(time_values) < 2:
            raise ValueError("kinematic GPS gate removed all but one fix")
        self.fix_time = time_values

        self.grid = np.arange(
            math.floor(time_values[0]), math.ceil(time_values[-1]) + 1,
            1.0, dtype=float)
        self.raw_latitude = np.interp(self.grid, time_values, latitude)
        self.raw_longitude = np.interp(self.grid, time_values, longitude)
        self.latitude = self._smooth(self.raw_latitude, sigma_s)
        self.longitude = self._smooth(self.raw_longitude, sigma_s)
        self.altitude = self._interp_optional(
            self.grid, time_values, altitude)
        self.speed = self._interp_optional(self.grid, time_values, speed)

        east, north = self._east_north(self.latitude, self.longitude)
        steps = np.hypot(np.diff(east), np.diff(north))
        self.distance = np.concatenate([[0.0], np.cumsum(steps)])

        insertion = np.searchsorted(time_values, self.grid)
        previous = np.clip(insertion - 1, 0, len(time_values) - 1)
        following = np.clip(insertion, 0, len(time_values) - 1)
        self.fix_gap = time_values[following] - time_values[previous]
        self.fix_distance_time = np.minimum(
            np.abs(self.grid - time_values[previous]),
            np.abs(time_values[following] - self.grid))

    def _east_north(self, latitude: np.ndarray,
                    longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        east = (np.radians(longitude)
                * math.cos(math.radians(self.latitude_origin))
                * EARTH_RADIUS_M)
        north = np.radians(latitude) * EARTH_RADIUS_M
        return east, north

    def _kinematic_gate(self, time_values: np.ndarray, latitude: np.ndarray,
                        longitude: np.ndarray, limit: float) -> np.ndarray:
        east, north = self._east_north(latitude, longitude)
        keep = np.zeros(len(time_values), dtype=bool)
        keep[0] = True
        last = 0
        for index in range(1, len(time_values)):
            elapsed = time_values[index] - time_values[last]
            distance = math.hypot(
                east[index] - east[last], north[index] - north[last])
            if elapsed > 0 and distance / elapsed <= limit:
                keep[index] = True
                last = index
        return keep

    @staticmethod
    def _smooth(values: np.ndarray, sigma_s: float) -> np.ndarray:
        if sigma_s <= 0:
            return values.copy()
        half_width = int(math.ceil(3 * sigma_s))
        offsets = np.arange(-half_width, half_width + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma_s) ** 2)
        kernel /= kernel.sum()
        return np.convolve(
            np.pad(values, half_width, mode="edge"), kernel, mode="valid")

    @staticmethod
    def _interp_optional(grid: np.ndarray, time_values: np.ndarray,
                         values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        if not np.any(valid):
            return np.full(len(grid), math.nan)
        return np.interp(grid, time_values[valid], values[valid])

    def at(self, elapsed_s: float, values: np.ndarray) -> float:
        return float(np.interp(elapsed_s, self.grid, values))

    def distance_at(self, elapsed_s: float) -> float:
        return self.at(elapsed_s, self.distance)

    def time_at_distance(self, distance_m: float) -> float:
        return float(np.interp(distance_m, self.distance, self.grid))

    def course_at_distance(self, distance_m: float, radius_m: float) -> float:
        first_time = self.time_at_distance(
            max(self.distance[0], distance_m - radius_m))
        second_time = self.time_at_distance(
            min(self.distance[-1], distance_m + radius_m))
        lat1 = math.radians(self.at(first_time, self.latitude))
        lon1 = math.radians(self.at(first_time, self.longitude))
        lat2 = math.radians(self.at(second_time, self.latitude))
        lon2 = math.radians(self.at(second_time, self.longitude))
        delta_lon = lon2 - lon1
        return math.degrees(math.atan2(
            math.sin(delta_lon) * math.cos(lat2),
            math.cos(lat1) * math.sin(lat2)
            - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))) % 360.0


def load_fit(path: Path) -> tuple[list[dict], dict]:
    messages = fit_decoder.decode(path)
    by_name: dict[str, list[dict]] = {}
    for name, fields in messages:
        by_name.setdefault(name, []).append(fields)
    sessions = by_name.get("session", [])
    if not sessions:
        raise ValueError(f"FIT source has no session message: {path}")
    start = fit_decoder.timestamp(sessions[0].get("start_time"))
    if start is None:
        raise ValueError(f"FIT session has no start time: {path}")
    rows = []
    for record in by_name.get("record", []):
        when = fit_decoder.timestamp(record.get("timestamp"))
        lat = record.get("position_lat")
        lon = record.get("position_long")
        if when is None or lat is None or lon is None:
            continue
        altitude = record.get("enhanced_altitude")
        if altitude is None:
            altitude = record.get("altitude")
        speed = record.get("enhanced_speed")
        if speed is None:
            speed = record.get("speed")
        rows.append({
            "elapsed_s": (when - start).total_seconds(),
            "latitude": lat * fit_decoder.SEMICIRCLE_DEG,
            "longitude": lon * fit_decoder.SEMICIRCLE_DEG,
            "altitude_m": (altitude / fit_decoder.ALTITUDE_SCALE
                           - fit_decoder.ALTITUDE_OFFSET
                           if altitude is not None else math.nan),
            "speed_mps": (speed / fit_decoder.SPEED_SCALE
                          if speed is not None else math.nan),
        })
    session = sessions[0]
    elapsed = session.get("total_elapsed_time")
    timer = session.get("total_timer_time")
    metadata = {
        "type": "garmin_fit",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "log_start_utc": iso_utc(start),
        "positioned_records": len(rows),
        "session_total_elapsed_s": (
            elapsed / fit_decoder.TIME_SCALE if elapsed is not None else None),
        "session_total_timer_s": (
            timer / fit_decoder.TIME_SCALE if timer is not None else None),
        "timer_matches_elapsed": elapsed == timer,
    }
    return rows, metadata


def load_sensor_logger(path: Path) -> tuple[list[dict], dict]:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name for name in archive.namelist()
            if name == "Location.csv" or name.endswith("/Location.csv")]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one Location.csv in {path}, found {candidates}")
        with archive.open(candidates[0]) as raw:
            lines = (line.decode("utf-8-sig") for line in raw)
            source_rows = list(csv.DictReader(lines))
    rows = []
    epoch_starts = []
    for row in source_rows:
        elapsed = float(row["seconds_elapsed"])
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        time_ns = int(row["time"])
        epoch_starts.append(time_ns / 1e9 - elapsed)
        altitude_text = (row.get("altitudeAboveMeanSeaLevel")
                         or row.get("altitude") or "")
        speed_text = row.get("speed", "")
        rows.append({
            "elapsed_s": elapsed,
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": (float(altitude_text)
                           if altitude_text else math.nan),
            "speed_mps": float(speed_text) if speed_text else math.nan,
        })
    start_epoch = float(np.median(epoch_starts))
    start = datetime.datetime.fromtimestamp(
        start_epoch, tz=datetime.timezone.utc)
    return rows, {
        "type": "sensor_logger_zip",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "location_csv": candidates[0],
        "log_start_utc": iso_utc(start),
        "positioned_records": len(rows),
        "clock_origin_spread_s": round(max(epoch_starts) - min(epoch_starts), 6),
    }


def load_timestamped_csv(path: Path) -> tuple[list[dict], dict]:
    """Load Unix timestamp, latitude, and longitude from a plain CSV."""
    with path.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    required = {"timestamp", "lat", "lon"}
    if not source_rows or not required.issubset(source_rows[0]):
        raise ValueError(
            f"{path} must contain timestamp, lat, and lon columns")
    start_epoch = min(float(row["timestamp"]) for row in source_rows)
    rows = []
    for row in source_rows:
        rows.append({
            "elapsed_s": float(row["timestamp"]) - start_epoch,
            "latitude": float(row["lat"]),
            "longitude": float(row["lon"]),
            "altitude_m": (float(row["altitude_m"])
                           if row.get("altitude_m") else math.nan),
            "speed_mps": (float(row["speed_mps"])
                          if row.get("speed_mps") else math.nan),
        })
    start = datetime.datetime.fromtimestamp(
        start_epoch, tz=datetime.timezone.utc)
    return rows, {
        "type": "timestamped_csv",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "log_start_utc": iso_utc(start),
        "positioned_records": len(rows),
    }


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def data_root_relative(path: Path) -> str:
    """Return a portable data-root path when a known lane is in the path."""
    parts = path.resolve().parts
    for lane in ("raw_material", "datasets", "models"):
        indices = [index for index, part in enumerate(parts) if part == lane]
        if indices:
            return Path(*parts[indices[-1]:]).as_posix()
    return str(path.resolve())


def load_collection(config_path: Path) -> tuple[dict, dict[str, tuple[Track, dict]]]:
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("recordings"), list):
        raise ValueError("collection config must contain a recordings list")
    sources = {}
    for source_id, source in payload.get("gps_sources", {}).items():
        path = resolve_path(config_path, source["path"]).resolve()
        if source["type"] == "fit":
            rows, metadata = load_fit(path)
        elif source["type"] == "sensor_logger_zip":
            rows, metadata = load_sensor_logger(path)
        elif source["type"] == "timestamped_csv":
            rows, metadata = load_timestamped_csv(path)
        else:
            raise ValueError(f"unknown GPS source type: {source['type']}")
        track = Track(
            rows,
            sigma_s=float(source.get("sigma_position_s", 1.0)),
            velocity_limit_mps=float(source["velocity_limit_mps"]),
        )
        metadata["dropped_kinematic_fixes"] = track.dropped_fixes
        sources[source_id] = (track, metadata)
    return payload, sources


def select_recordings(payload: dict, dataset_ids: list[str]) -> list[dict]:
    recordings = payload["recordings"]
    ids = [recording["dataset_id"] for recording in recordings]
    if len(ids) != len(set(ids)):
        raise ValueError("recording dataset_id values must be unique")
    if not dataset_ids:
        return recordings
    missing = sorted(set(dataset_ids) - set(ids))
    if missing:
        raise ValueError(f"unknown dataset IDs: {missing}")
    wanted = set(dataset_ids)
    return [recording for recording in recordings
            if recording["dataset_id"] in wanted]


def recording_clip(recording: dict, source_info: dict) -> dict:
    """Resolve a recording's requested source window onto its output grid."""
    return anonymize_video.clip_metadata(
        source_info, float(recording.get("output_fps", 3.0)),
        float(recording.get("clip_start_s", 0.0)),
        (float(recording["clip_end_s"])
         if recording.get("clip_end_s") is not None else None))


def sample_recording(track: Track, recording: dict, source_info: dict) -> list[dict]:
    fps = float(recording.get("output_fps", 3.0))
    clip = recording_clip(recording, source_info)
    frame_count = clip["frame_count"]
    source_sensor_start = float(
        recording["sync"]["sensor_elapsed_at_video_start_s"])
    output_sensor_start = source_sensor_start + clip["start_s"]
    trim_head = float(recording.get("trim_head_s", 0.0))
    trim_tail = float(recording.get("trim_tail_s", 0.0))
    first_video_time = trim_head
    last_video_time = (frame_count - 1) / fps - trim_tail
    first_elapsed = output_sensor_start + first_video_time
    last_elapsed = output_sensor_start + last_video_time
    if first_elapsed < track.grid[0] or last_elapsed > track.grid[-1]:
        raise ValueError(
            f"{recording['dataset_id']} video window [{first_elapsed:.3f}, "
            f"{last_elapsed:.3f}] is outside GPS [{track.grid[0]:.3f}, "
            f"{track.grid[-1]:.3f}]")
    spacing = float(recording["sampling"]["distance_m"])
    if spacing <= 0:
        raise ValueError("sampling distance must be positive")
    course_radius = float(recording["sampling"]["course_radius_m"])
    max_gap = float(recording["gps_quality"]["max_gap_s"])
    fix_near = float(recording["gps_quality"].get("fix_near_s", 1.5))

    first_distance = track.distance_at(first_elapsed)
    last_distance = track.distance_at(last_elapsed)
    target_distance = math.ceil(first_distance / spacing) * spacing
    rows = []
    seen_frames = {}
    while target_distance <= last_distance + 1e-6:
        ideal_elapsed = track.time_at_distance(target_distance)
        ideal_video_time = ideal_elapsed - output_sensor_start
        frame_index = int(round(ideal_video_time * fps))
        frame_index = max(0, min(frame_count - 1, frame_index))
        if frame_index in seen_frames:
            raise ValueError(
                f"{recording['dataset_id']}: {spacing:g} m is too fine; "
                f"route distances {seen_frames[frame_index]:.1f} and "
                f"{target_distance:.1f} both map to frame {frame_index}")
        seen_frames[frame_index] = target_distance
        video_time = frame_index / fps
        source_video_time = clip["start_s"] + video_time
        elapsed = source_sensor_start + source_video_time
        fix_gap = track.at(elapsed, track.fix_gap)
        fix_dt = track.at(elapsed, track.fix_distance_time)
        if fix_gap > max_gap:
            quality = "unusable"
        elif fix_dt <= fix_near:
            quality = "fix"
        else:
            quality = "interp"
        frame_id = f"{recording['dataset_id']}_p{frame_index:06d}"
        rows.append({
            "frame_id": frame_id,
            "frame_index": frame_index,
            "video_t_s": video_time,
            "source_video_t_s": source_video_time,
            "sensor_elapsed_s": elapsed,
            "route_distance_m": target_distance,
            "latitude": track.at(elapsed, track.latitude),
            "longitude": track.at(elapsed, track.longitude),
            "latitude_raw": track.at(elapsed, track.raw_latitude),
            "longitude_raw": track.at(elapsed, track.raw_longitude),
            "altitude_m": track.at(elapsed, track.altitude),
            "speed_mps": track.at(elapsed, track.speed),
            "course_deg": track.course_at_distance(target_distance, course_radius),
            "gps_quality": quality,
            "fix_gap_s": fix_gap,
            "fix_dt_s": fix_dt,
            "frame_time_error_s": video_time - ideal_video_time,
            "source_capture_fps": float(recording["capture_fps"]),
            "frame_file": f"{frame_id}_t{video_time:010.3f}s.jpg",
        })
        target_distance += spacing
    if not rows:
        raise ValueError(f"{recording['dataset_id']} produced no frame samples")
    distance_origin = track.distance_at(rows[0]["sensor_elapsed_s"])
    for index, row in enumerate(rows):
        row["idx"] = index
        row["dist_m"] = track.distance_at(
            row["sensor_elapsed_s"]) - distance_origin
    return rows


def write_frame_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        for row in rows:
            output = {}
            for column in FRAME_COLUMNS:
                value = row[column]
                if row["gps_quality"] == "unusable" and column in {
                        "latitude", "longitude", "latitude_raw", "longitude_raw",
                        "altitude_m", "speed_mps", "course_deg", "dist_m",
                        "route_distance_m"}:
                    output[column] = ""
                elif column in FLOAT_FORMATS and math.isfinite(float(value)):
                    output[column] = f"{float(value):.{FLOAT_FORMATS[column]}f}"
                elif column in FLOAT_FORMATS:
                    output[column] = ""
                else:
                    output[column] = value
            writer.writerow(output)


def recording_paths(config_path: Path, recording: dict) -> dict[str, Path]:
    output_dir = resolve_path(config_path, recording["output_dir"]).resolve()
    return {
        "output_dir": output_dir,
        "frames": output_dir / "frames",
        "csv": output_dir / "frames_gps.csv",
        "plan_manifest": output_dir / "plan_manifest.json",
        "extra_base": output_dir / "extra_metadata.base.json",
        "extra_final": output_dir / "extra_metadata.json",
        "source_video": resolve_path(config_path, recording["video"]).resolve(),
        "anonymization_dir": resolve_path(
            config_path, recording["anonymization_dir"]).resolve(),
        "anonymized_video": resolve_path(
            config_path, recording["anonymized_video"]).resolve(),
    }


def verify_extracted_frame_hashes(frame_files: list[Path],
                                  expected_names: set[str],
                                  expected_sha256: object,
                                  dataset_id: str):
    actual_names = {path.name for path in frame_files}
    if actual_names != expected_names:
        raise ValueError(f"{dataset_id} extracted frame names do not match plan")
    if any(path.is_symlink() or not path.is_file() for path in frame_files):
        raise ValueError(f"{dataset_id} extracted frames must be regular files")
    if (not isinstance(expected_sha256, dict)
            or set(expected_sha256) != expected_names):
        raise ValueError(f"{dataset_id} render manifest does not bind every frame")
    for path in frame_files:
        if sha256_file(path) != expected_sha256[path.name]:
            raise ValueError(f"{dataset_id} extracted frame changed: {path.name}")


def plan(args) -> int:
    config_path = args.config.resolve()
    payload, gps_sources = load_collection(config_path)
    for recording in select_recordings(payload, args.dataset):
        dataset_id = recording["dataset_id"]
        if recording["gps_source"] not in gps_sources:
            raise ValueError(f"{dataset_id} names an unknown GPS source")
        track, gps_metadata = gps_sources[recording["gps_source"]]
        paths = recording_paths(config_path, recording)
        if paths["output_dir"].exists() or paths["output_dir"].is_symlink():
            raise FileExistsError(
                f"refusing to replace plan output: {paths['output_dir']}")
        source_info = anonymize_video.probe_video(paths["source_video"])
        clip = recording_clip(recording, source_info)
        rows = sample_recording(track, recording, source_info)
        paths["output_dir"].mkdir(parents=True)
        write_frame_csv(paths["csv"], rows)
        quality_counts = {}
        for row in rows:
            quality = row["gps_quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        sync = recording["sync"]
        log_start = datetime.datetime.fromisoformat(
            gps_metadata["log_start_utc"].replace("Z", "+00:00"))
        source_start_elapsed = float(
            sync["sensor_elapsed_at_video_start_s"])
        output_start_elapsed = source_start_elapsed + clip["start_s"]
        source_start_utc = log_start + datetime.timedelta(
            seconds=source_start_elapsed)
        output_start_utc = log_start + datetime.timedelta(
            seconds=output_start_elapsed)
        plan_manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_video": {
                "path": str(paths["source_video"]),
                "sha256": sha256_file(paths["source_video"]),
                "capture_fps": float(recording["capture_fps"]),
                "clip": clip,
                **source_info,
            },
            "gps_source": gps_metadata,
            "sync": {
                **sync,
                "equation": (
                    "sensor_elapsed_s = sensor_elapsed_at_video_start_s "
                    "+ source_video_t_s"),
                "sensor_elapsed_at_clipped_video_start_s": (
                    output_start_elapsed),
                "source_video_start_utc": iso_utc(source_start_utc),
                "video_start_utc": iso_utc(output_start_utc),
            },
            "sampling": recording["sampling"],
            "trim_head_s": float(recording.get("trim_head_s", 0.0)),
            "trim_tail_s": float(recording.get("trim_tail_s", 0.0)),
            "frame_count": len(rows),
            "quality_counts": quality_counts,
            "trajectory_m": round(rows[-1]["dist_m"], 2),
            "frames_gps_sha256": sha256_file(paths["csv"]),
            "created_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        paths["plan_manifest"].write_text(
            json.dumps(plan_manifest, indent=2) + "\n")
        extra = {
            "collection_preprocessing": {
                "status": "planned_pending_anonymized_render",
                "plan_manifest": str(paths["plan_manifest"]),
                "plan_manifest_sha256": sha256_file(paths["plan_manifest"]),
                "source_capture_fps": float(recording["capture_fps"]),
                "source_media_fps": source_info["media_fps"],
                "output_fps": float(recording.get("output_fps", 3.0)),
                "source_clip": clip,
                "sync": plan_manifest["sync"],
                "privacy_review_status": "pending",
            },
        }
        paths["extra_base"].write_text(json.dumps(extra, indent=2) + "\n")
        print(f"planned {dataset_id}: {len(rows)} frames, "
              f"{rows[-1]['dist_m'] / 1000:.2f} km, {quality_counts}")
    return 0


def finalize(args) -> int:
    config_path = args.config.resolve()
    payload = json.loads(config_path.read_text())
    for recording in select_recordings(payload, args.dataset):
        dataset_id = recording["dataset_id"]
        paths = recording_paths(config_path, recording)
        if paths["extra_final"].exists() or paths["extra_final"].is_symlink():
            raise FileExistsError(
                f"refusing to replace finalized metadata: {paths['extra_final']}")
        manifest_path = paths["anonymization_dir"] / "anonymization_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        plan_manifest = json.loads(paths["plan_manifest"].read_text())
        extra = json.loads(paths["extra_base"].read_text())
        if sha256_file(paths["plan_manifest"]) != extra[
                "collection_preprocessing"]["plan_manifest_sha256"]:
            raise ValueError(f"{dataset_id} plan manifest changed after planning")
        if sha256_file(paths["csv"]) != plan_manifest["frames_gps_sha256"]:
            raise ValueError(f"{dataset_id} extraction plan changed after planning")
        if sha256_file(config_path) != plan_manifest["config_sha256"]:
            raise ValueError(f"{dataset_id} collection config changed after planning")
        if manifest.get("status") != "rendered_pending_review":
            raise ValueError(
                f"{dataset_id} anonymization is not a completed render")
        if manifest["render"]["output_video"] != str(paths["anonymized_video"]):
            raise ValueError(f"{dataset_id} anonymized video path mismatch")
        if manifest["render"]["output_video_sha256"] != sha256_file(
                paths["anonymized_video"]):
            raise ValueError(f"{dataset_id} anonymized video changed")
        if manifest["render"].get("extraction_plan_sha256") != (
                plan_manifest["frames_gps_sha256"]):
            raise ValueError(
                f"{dataset_id} render used a different extraction plan")
        expected_source_sha256 = plan_manifest["source_video"]["sha256"]
        if manifest["source"]["sha256"] != expected_source_sha256:
            raise ValueError(
                f"{dataset_id} anonymization used a different source video")
        if not math.isclose(
                float(manifest["source"]["capture_fps"]),
                float(plan_manifest["source_video"]["capture_fps"]),
                abs_tol=1e-9):
            raise ValueError(f"{dataset_id} capture frame rate mismatch")
        if not math.isclose(
                float(manifest["output_fps"]),
                float(extra["collection_preprocessing"]["output_fps"]),
                abs_tol=1e-9):
            raise ValueError(f"{dataset_id} output frame rate mismatch")
        expected_clip = plan_manifest["source_video"]["clip"]
        if manifest["source"].get("clip") != expected_clip:
            raise ValueError(
                f"{dataset_id} anonymization used a different source clip")
        frame_files = sorted(paths["frames"].glob("*.jpg"))
        with paths["csv"].open() as stream:
            frame_rows = list(csv.DictReader(stream))
        if len(frame_files) != len(frame_rows):
            raise ValueError(
                f"{dataset_id} has {len(frame_files)} images for "
                f"{len(frame_rows)} planned rows")
        expected_names = {row["frame_file"] for row in frame_rows}
        verify_extracted_frame_hashes(
            frame_files, expected_names,
            manifest["render"].get("extracted_frame_sha256"), dataset_id)

        retained_evidence = (
            (paths["anonymization_dir"] / manifest["review"]["video"],
             manifest["render"].get("review_video_sha256"), "review video"),
            (paths["anonymization_dir"] / manifest["review"]["html"],
             manifest["render"].get("review_html_sha256"), "review HTML"),
            (paths["anonymization_dir"]
             / manifest["files"]["applied_ledger"]["path"],
             manifest["files"]["applied_ledger"]["sha256"],
             "anonymization ledger"),
        )
        for path, expected_sha256, label in retained_evidence:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{dataset_id} {label} is not a regular file")
            if not expected_sha256 or sha256_file(path) != expected_sha256:
                raise ValueError(f"{dataset_id} {label} changed after render")

        decision_path = paths["anonymization_dir"] / "review_decision.json"
        decision = None
        if decision_path.is_file() and not decision_path.is_symlink():
            decision = json.loads(decision_path.read_text())
            if decision.get("anonymization_manifest_sha256") != sha256_file(
                    manifest_path):
                raise ValueError(
                    f"{dataset_id} review decision does not bind this manifest")
            if decision.get("status") == "needs_corrections":
                raise ValueError(
                    f"{dataset_id} review requires corrections; do not ingest")
            if decision.get("status") != "approved":
                raise ValueError(
                    f"{dataset_id} has an unknown review decision status")
        if decision is None:
            raise ValueError(
                f"{dataset_id} has no approved human privacy review; "
                "review and mark the render before finalizing")
        review_status = "approved"
        video_info = manifest["render"]["output_video_info"]
        extra["video"] = {
            "source_video": data_root_relative(paths["anonymized_video"]),
            "source_video_sha256": manifest["render"]["output_video_sha256"],
            "video_resolution": (
                f"{video_info['width']}x{video_info['height']}"),
            "video_codec": video_info["codec"],
            "video_fps": video_info["media_fps"],
            "video_originally_fps": float(recording["capture_fps"]),
            "input_media_fps": manifest["source"]["media_fps"],
            "source_clip": expected_clip,
            "video_duration_s": video_info["duration_s"],
            "video_frames": video_info["nb_frames"],
            "sampling": recording["sampling"],
            "trim_head_s": float(recording.get("trim_head_s", 0.0)),
            "trim_tail_s": float(recording.get("trim_tail_s", 0.0)),
            "sync": extra["collection_preprocessing"]["sync"],
            "anonymized": True,
            "privacy_review_status": review_status,
            "retained": True,
        }
        extra["collection_preprocessing"].update({
            "status": "rendered_privacy_approved",
            "anonymized_video": str(paths["anonymized_video"]),
            "anonymized_video_sha256": manifest["render"]["output_video_sha256"],
            "anonymization_manifest": str(manifest_path),
            "anonymization_manifest_sha256": sha256_file(manifest_path),
            "anonymization_ledger_sha256": manifest[
                "files"]["applied_ledger"]["sha256"],
            "privacy_review_status": review_status,
            "privacy_review": {
                "status": review_status,
                "review_video": str(paths["anonymization_dir"] / "review.mp4"),
                "review_html": str(paths["anonymization_dir"] / "review.html"),
                "decision_record": str(decision_path),
                "decision_record_sha256": (
                    sha256_file(decision_path) if decision else None),
            },
            "extracted_frames": len(frame_files),
        })
        paths["extra_final"].write_text(json.dumps(extra, indent=2) + "\n")
        print(f"finalized {dataset_id}: {len(frame_files)} blurred frames; "
              f"privacy review {review_status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, function in (("plan", plan), ("finalize", finalize)):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--dataset", action="append", default=[],
                           help="dataset ID; repeat, or omit for all")
        child.set_defaults(func=function)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
