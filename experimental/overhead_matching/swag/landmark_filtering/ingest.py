"""Ingest stage: raw Gemini predictions -> Frames + Observations.

Responsibilities:
- Build the frame table from panorama filenames (`f0005,<lat>,<lon>,.jpg`),
  joined against frames_gps.csv for time/odometry.
- Parse the Gemini predictions JSONL (one line per panorama).
- Convert each landmark's bounding boxes to camera-frame bearings, merging
  boxes that continue across pinhole-face seams into single observations.

Nothing is filtered here; every parsed landmark box group becomes an
Observation with final_disposition="kept".
"""

import csv
import json
from pathlib import Path

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    bearing_geometry as bg,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

VALID_FACE_YAWS = ("0", "90", "180", "270")


class IngestResult:
    def __init__(self, frames, observations, anchor_lat, anchor_lon, stats):
        self.frames: list[schema.Frame] = frames
        self.observations: list[schema.Observation] = observations
        self.anchor_lat: float = anchor_lat
        self.anchor_lon: float = anchor_lon
        self.stats: schema.SummaryStats = stats


def load_frames(dataset_base: Path) -> list[schema.Frame]:
    """Frame table from panorama filenames, joined with frames_gps.csv."""
    gps_rows = {}
    gps_csv = dataset_base / "frames_gps.csv"
    if gps_csv.exists():
        with open(gps_csv) as f:
            for row in csv.DictReader(f):
                gps_rows[int(row["idx"])] = row

    frames = []
    pano_paths = sorted((dataset_base / "panorama").glob("*.jpg"))
    for path in pano_paths:
        pano_id, lat_str, lon_str, _ = path.stem.split(",")
        gps = gps_rows.get(int(pano_id[1:]))
        frames.append(schema.Frame(
            frame_idx=len(frames),
            pano_id=pano_id,
            pano_stem=path.stem,
            lat=float(lat_str),
            lon=float(lon_str),
            x_m=0.0,
            y_m=0.0,
            dist_along_m=float(gps["dist_m"]) if gps else None,
            time_s=float(gps["video_t_s"]) if gps else None,
        ))
    frames.sort(key=lambda fr: fr.pano_id)
    for idx, frame in enumerate(frames):
        frame.frame_idx = idx
    return frames


def fill_enu(frames: list[schema.Frame]) -> tuple[float, float]:
    """Compute the anchor (mean lat/lon) and each frame's ENU position."""
    anchor_lat = sum(fr.lat for fr in frames) / len(frames)
    anchor_lon = sum(fr.lon for fr in frames) / len(frames)
    for frame in frames:
        frame.x_m, frame.y_m = bg.enu_from_latlon(
            frame.lat, frame.lon, anchor_lat, anchor_lon)
    return anchor_lat, anchor_lon


def load_predictions(landmark_base: Path) -> tuple[dict, int]:
    """Parse Gemini predictions.jsonl files.

    Returns ({pano_stem: {"location_type": str, "landmarks": [...]}},
    n_parse_failures).
    """
    predictions = {}
    n_failures = 0
    jsonl_paths = sorted(
        landmark_base.glob("sentences/results/*/prediction-*/predictions.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(
            f"No predictions.jsonl under {landmark_base}/sentences/results/")
    for jsonl_path in jsonl_paths:
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = record["key"]
                    text = (record["response"]["candidates"][0]["content"]
                            ["parts"][0]["text"])
                    text = text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(text)
                    predictions[key] = {
                        "location_type": parsed.get("location_type", ""),
                        "landmarks": parsed.get("landmarks", []),
                    }
                except (KeyError, IndexError, json.JSONDecodeError,
                        TypeError):
                    n_failures += 1
    return predictions, n_failures


def _box_edges_unwrapped(boxes: list[schema.BBox], fov_deg: float):
    """(left, right) bearing edges per box, unwrapped to a continuous range.

    Angles are lifted out of [0, 360) so that a group of boxes spanning the
    0/360 wrap forms one continuous interval.
    """
    edges = []
    reference = None
    for box in boxes:
        left = bg.bearing_camera_deg(box.face_yaw_deg, box.xmin, fov_deg)
        right = bg.bearing_camera_deg(box.face_yaw_deg, box.xmax, fov_deg)
        if right < left:
            right += 360.0
        if reference is None:
            reference = left
        else:
            while left < reference - 180.0:
                left += 360.0
                right += 360.0
            while left > reference + 180.0:
                left -= 360.0
                right -= 360.0
        edges.append((left, right))
    return edges


def _is_seam_pair(box_a: schema.BBox, box_b: schema.BBox,
                  config: IngestConfig) -> bool:
    """True if box_a's right edge continues into box_b's left edge on the
    adjacent face (or vice versa).

    FIXED 2026-08-19 (was a KNOWN ISSUE from 2026-08-05): the adjoining face is
    `A - 90`, not `A + 90`. In the verified render convention the panorama is laid
    out `180 | 90 | 0 | 270` left to right and camera azimuth increases
    image-right, so faces appear in azimuth order 180 -> 90 -> 0 -> 270 and face
    A's image-right edge physically adjoins face `(A + 270) mod 360`'s image-left
    edge. The old `+ 90` check followed `bearing_geometry`'s pre-2026-08-19 copy
    of the camera frame; it left real seam continuations unmerged (the landmark
    became two observations) and the pairs it did merge were ~180 deg apart
    physically. See docs/conventions.md.
    """
    for first, second in ((box_a, box_b), (box_b, box_a)):
        if (second.face_yaw_deg - first.face_yaw_deg) % 360 != 270:
            continue
        if first.xmax < bg.BBOX_NORM_MAX - config.seam_gap_norm:
            continue
        if second.xmin > config.seam_gap_norm:
            continue
        overlap = (min(first.ymax, second.ymax)
                   - max(first.ymin, second.ymin))
        union = (max(first.ymax, second.ymax)
                 - min(first.ymin, second.ymin))
        if union > 0 and overlap / union >= config.seam_min_y_iou:
            return True
    return False


def group_seam_boxes(boxes: list[schema.BBox],
                     config: IngestConfig) -> list[list[schema.BBox]]:
    """Union-find grouping of one landmark's boxes across face seams."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _is_seam_pair(boxes[i], boxes[j], config):
                parent[find(i)] = find(j)

    groups: dict[int, list[schema.BBox]] = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)
    return list(groups.values())


def _observation_from_group(pano_id: str, frame_idx: int, landmark_idx: int,
                            box_group_idx: int, landmark: dict,
                            group: list[schema.BBox],
                            config: IngestConfig) -> schema.Observation:
    edges = _box_edges_unwrapped(group, config.fov_deg)
    left = min(e[0] for e in edges)
    right = max(e[1] for e in edges)
    bearing = ((left + right) / 2.0) % 360.0
    width = right - left
    elevation = sum(
        bg.elevation_deg((box.ymin + box.ymax) / 2.0, config.fov_deg)
        for box in group) / len(group)

    primary_tag = landmark.get("primary_tag") or {}
    additional_tags = [
        [tag.get("key", ""), tag.get("value", "")]
        for tag in landmark.get("additional_tags") or []
    ]
    return schema.Observation(
        obs_id=f"{pano_id}__lm{landmark_idx}__box{box_group_idx}",
        pano_id=pano_id,
        frame_idx=frame_idx,
        landmark_idx=landmark_idx,
        embedding_id=f"{pano_id}__landmark_{landmark_idx}",
        primary_tag_key=primary_tag.get("key", ""),
        primary_tag_value=primary_tag.get("value", ""),
        additional_tags=additional_tags,
        confidence=landmark.get("confidence", ""),
        description=landmark.get("description", ""),
        boxes=group,
        seam_merged=len(group) > 1,
        bearing_camera_deg=bearing,
        bearing_global_deg=bearing,
        elevation_deg=elevation,
        angular_width_deg=width,
        decisions=[],
    )


def run_ingest(dataset_base: Path, landmark_base: Path,
               config: IngestConfig) -> IngestResult:
    frames = load_frames(dataset_base)
    if not frames:
        raise FileNotFoundError(f"No panoramas under {dataset_base}/panorama")
    anchor_lat, anchor_lon = fill_enu(frames)
    predictions, n_parse_failures = load_predictions(landmark_base)

    stats = schema.SummaryStats(
        n_frames=len(frames), n_parse_failures=n_parse_failures)
    observations: list[schema.Observation] = []
    for frame in frames:
        prediction = predictions.get(frame.pano_stem)
        if prediction is None:
            continue
        n_frame_obs = 0
        for landmark_idx, landmark in enumerate(prediction["landmarks"]):
            stats.n_raw_landmark_entries += 1
            boxes = []
            for raw_box in landmark.get("bounding_boxes") or []:
                if str(raw_box.get("yaw_angle")) not in VALID_FACE_YAWS:
                    stats.n_boxes_invalid_yaw += 1
                    continue
                boxes.append(schema.BBox(
                    face_yaw_deg=int(raw_box["yaw_angle"]),
                    xmin=int(raw_box["xmin"]),
                    ymin=int(raw_box["ymin"]),
                    xmax=int(raw_box["xmax"]),
                    ymax=int(raw_box["ymax"]),
                ))
            if not boxes:
                stats.n_landmarks_without_valid_boxes += 1
                continue
            groups = group_seam_boxes(boxes, config)
            for box_group_idx, group in enumerate(groups):
                observations.append(_observation_from_group(
                    frame.pano_id, frame.frame_idx, landmark_idx,
                    box_group_idx, landmark, group, config))
                n_frame_obs += 1
        frame.n_observations = n_frame_obs

    stats.n_observations = len(observations)
    stats.n_kept = len(observations)
    for frame in frames:
        key = str(frame.n_observations)
        stats.obs_per_frame_histogram[key] = (
            stats.obs_per_frame_histogram.get(key, 0) + 1)
    return IngestResult(frames, observations, anchor_lat, anchor_lon, stats)
