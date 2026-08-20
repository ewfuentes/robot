"""The dataset contract: frames, metadata validation, and detection ingest.

A farfield dataset directory is a frozen problem definition:

    panorama/f####,<lat>,<lon>,.jpg   equirectangular frames (GPS in the name)
    frames_gps.csv                    idx, latitude, longitude, dist_m, video_t_s
    intrinsics.csv / extraction_log.csv / pano_id_mapping.csv
    pipeline_metadata.json            conventions, video pointer, mount offset

This module is the only reader of that contract. It loads the frame table,
fills the local ENU frame, validates the metadata conventions (loudly — see
below), and converts VLM detections (the `frame_landmarks` artifact) into
camera-frame Observations with seam-continuation boxes merged.

Two validations here exist because their absence produced real 180-degree
incidents (docs/conventions.md):

- `require_camera_frame_panoramas` refuses a dataset whose panoramas are
  north-aligned or of unrecorded orientation. Every downstream stage assumes
  camera azimuth is a fixed function of the pano column; on a north-aligned
  dataset that silently becomes an absolute azimuth and heading gets counted
  twice, with every number still well-formed.
- `mount_offset_record` refuses a `mount_offset` block that does not state its
  frame (`geometry.MOUNT_OFFSET_FRAME`) and whether it is already applied to
  the intrinsics' heading_deg column. The same key has meant both on this
  project; consuming it without the qualifiers is how pohang shipped 180
  degrees out.

Nothing in this module ever writes to a dataset directory.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from experimental.overhead_matching.swag.farfield import geometry as geo

VALID_FACE_YAWS = ("0", "90", "180", "270")


class ContractViolation(Exception):
    """The dataset does not satisfy the contract this pipeline assumes."""


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    frame_idx: int          # positional, 0..N-1 after sorting by pano_id
    pano_id: str            # e.g. "f0005" — NOT necessarily equal to frame_idx
    pano_stem: str          # full filename stem, the cross-artifact join key
    lat: float
    lon: float
    x_m: float = 0.0
    y_m: float = 0.0
    dist_along_m: float | None = None
    time_s: float | None = None
    n_observations: int = 0


def load_frames(dataset_base: Path) -> list[Frame]:
    """Frame table from panorama filenames, joined with frames_gps.csv.

    `frame_idx` is positional. When a panorama is missing from the middle of a
    dataset, `int(pano_id[1:])` and `frame_idx` diverge for everything after
    the gap — so nothing may parse a pano id to get a frame index. Use
    `frame_index_by_pano_id` for that join.
    """
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
        frames.append(Frame(
            frame_idx=len(frames),
            pano_id=pano_id,
            pano_stem=path.stem,
            lat=float(lat_str),
            lon=float(lon_str),
            dist_along_m=float(gps["dist_m"]) if gps else None,
            time_s=float(gps["video_t_s"]) if gps else None,
        ))
    frames.sort(key=lambda fr: fr.pano_id)
    for idx, frame in enumerate(frames):
        frame.frame_idx = idx
    return frames


def frame_index_by_pano_id(frames: list[Frame]) -> dict:
    """The one sanctioned pano_id -> frame_idx join.

    Callers that need a frame index for an id parsed out of an obs_id or a
    filename go through this map; parsing digits out of the id instead breaks
    silently the first time a panorama is missing.
    """
    return {fr.pano_id: fr.frame_idx for fr in frames}


def fill_enu(frames: list[Frame]) -> tuple[float, float]:
    """Compute the anchor (mean lat/lon) and each frame's ENU position.

    The anchor is data-dependent: trimming or extending a dataset moves it,
    which correctly invalidates anything keyed on ENU coordinates (catalog
    caches key on the anchor for exactly this reason).
    """
    anchor_lat = sum(fr.lat for fr in frames) / len(frames)
    anchor_lon = sum(fr.lon for fr in frames) / len(frames)
    for frame in frames:
        frame.x_m, frame.y_m = geo.enu_from_latlon(
            frame.lat, frame.lon, anchor_lat, anchor_lon)
    return anchor_lat, anchor_lon


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------

def load_metadata(dataset_base: Path) -> dict:
    path = Path(dataset_base) / "pipeline_metadata.json"
    if not path.exists():
        raise ContractViolation(
            f"{path} not found — every dataset carries pipeline_metadata.json "
            f"(conventions, video pointer, mount offset).")
    return json.loads(path.read_text())


def require_camera_frame_panoramas(metadata: dict, dataset_base: Path) -> None:
    """Refuse a dataset whose panoramas are not stored in the camera frame.

    The whole tracking pipeline assumes `north_aligned: false` (camera azimuth
    is a fixed function of the pano column). This must be recorded, not
    assumed: on a north-aligned dataset every bearing silently becomes
    absolute and downstream adds heading to it again.
    """
    north_aligned = metadata.get("north_aligned")
    if north_aligned is None:
        raise ContractViolation(
            f"{dataset_base}: pipeline_metadata.json does not record "
            f"'north_aligned'. The pipeline requires camera-frame panoramas "
            f"and refuses to guess; record the orientation the images are "
            f"actually stored in.")
    if north_aligned:
        raise ContractViolation(
            f"{dataset_base}: panoramas are north-aligned (north_aligned: "
            f"true). This pipeline computes camera-frame azimuths from pano "
            f"columns; feeding it north-aligned frames double-counts heading "
            f"with every number still well-formed. Ingest the raw "
            f"camera-frame frames instead.")


@dataclass
class MountOffset:
    """A validated mount-offset record. See geometry.MOUNT_OFFSET_CONVENTION."""
    offset_deg: float
    frame: str
    applied_to_heading_deg: bool
    status: str = ""
    accuracy_validated: bool = False
    source: dict = field(default_factory=dict)


def mount_offset_record(metadata: dict,
                        dataset_base: Path) -> MountOffset | None:
    """The dataset's mount-offset block, validated, or None when absent.

    Absent is legitimate (an uncalibrated dataset); present-but-unqualified is
    not: a block without `frame` and `applied_to_heading_deg` cannot be
    consumed safely, because the same key has meant both "camera-frame,
    unapplied" and "already baked into heading_deg" on this project, and a
    frame slip is exactly 180 degrees with no symptom.
    """
    block = metadata.get("mount_offset")
    if block is None:
        return None
    problems = []
    if block.get("frame") != geo.MOUNT_OFFSET_FRAME:
        problems.append(
            f"frame={block.get('frame')!r} (must be "
            f"{geo.MOUNT_OFFSET_FRAME!r}; a column-0 offset is exactly 180 "
            f"deg out — see geometry.MOUNT_OFFSET_CONVENTION)")
    if not isinstance(block.get("applied_to_heading_deg"), bool):
        problems.append(
            "applied_to_heading_deg missing (must state whether heading_deg "
            "in intrinsics.csv already includes this offset)")
    if not isinstance(block.get("mount_offset_deg"), (int, float)):
        problems.append("mount_offset_deg missing or non-numeric")
    if problems:
        raise ContractViolation(
            f"{dataset_base}: mount_offset block is unusable:\n" +
            "\n".join(f"  - {p}" for p in problems) +
            "\nFix the metadata (see the data-migration step in REORG.md); "
            "do not consume an unqualified offset.")
    return MountOffset(
        offset_deg=float(block["mount_offset_deg"]),
        frame=block["frame"],
        applied_to_heading_deg=block["applied_to_heading_deg"],
        status=block.get("status", ""),
        accuracy_validated=bool(block.get("accuracy_validated", False)),
        source={k: v for k, v in block.items()
                if k not in ("mount_offset_deg", "frame",
                             "applied_to_heading_deg", "status",
                             "accuracy_validated")},
    )


# ---------------------------------------------------------------------------
# VLM detections -> Observations
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    face_yaw_deg: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class Observation:
    obs_id: str             # f"{pano_id}__lm{landmark_idx}__box{group_idx}"
    pano_id: str
    frame_idx: int
    landmark_idx: int
    embedding_id: str
    primary_tag_key: str
    primary_tag_value: str
    additional_tags: list
    confidence: str
    description: str
    boxes: list
    seam_merged: bool
    bearing_camera_deg: float
    elevation_deg: float
    angular_width_deg: float


@dataclass
class IngestParams:
    """Detection-ingest parameters. No defaults on purpose (REORG.md rule 2):
    values come from the run's recorded config."""
    fov_deg: float
    seam_gap_norm: float    # bbox units (0-1000): margin for a seam candidate
    seam_min_y_iou: float   # vertical IoU to accept a seam continuation


@dataclass
class IngestStats:
    n_frames: int = 0
    n_parse_failures: int = 0
    n_raw_landmark_entries: int = 0
    n_boxes_invalid_yaw: int = 0
    n_landmarks_without_valid_boxes: int = 0
    n_observations: int = 0


class IngestResult:
    def __init__(self, frames, observations, anchor_lat, anchor_lon, stats):
        self.frames: list[Frame] = frames
        self.observations: list[Observation] = observations
        self.anchor_lat: float = anchor_lat
        self.anchor_lon: float = anchor_lon
        self.stats: IngestStats = stats
        self.frame_index_by_pano_id: dict = frame_index_by_pano_id(frames)


def load_predictions(frame_landmarks_dir: Path) -> tuple[dict, int]:
    """Parse the extraction artifact's predictions JSONL files.

    Returns ({pano_stem: {"location_type": str, "landmarks": [...]}},
    n_parse_failures).
    """
    predictions = {}
    n_failures = 0
    jsonl_paths = sorted(Path(frame_landmarks_dir).glob(
        "sentences/results/*/prediction-*/predictions.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(
            f"No predictions.jsonl under "
            f"{frame_landmarks_dir}/sentences/results/ — run the extraction "
            f"stage first.")
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


def _box_edges_unwrapped(boxes: list, fov_deg: float):
    """(left, right) bearing edges per box, unwrapped to a continuous range.

    Angles are lifted out of [0, 360) so that a group of boxes spanning the
    0/360 wrap forms one continuous interval.
    """
    edges = []
    reference = None
    for box in boxes:
        left = geo.bearing_camera_deg(box.face_yaw_deg, box.xmin, fov_deg)
        right = geo.bearing_camera_deg(box.face_yaw_deg, box.xmax, fov_deg)
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


def _is_seam_pair(box_a: BBox, box_b: BBox, params: IngestParams) -> bool:
    """True if box_a's right edge continues into box_b's left edge on the
    adjacent face (or vice versa).

    The adjoining face is `A - 90`: the panorama is laid out 180 | 90 | 0 |
    270 left to right and camera azimuth increases image-right, so face A's
    image-right edge physically adjoins face (A + 270) mod 360's image-left
    edge.
    """
    for first, second in ((box_a, box_b), (box_b, box_a)):
        if (second.face_yaw_deg - first.face_yaw_deg) % 360 != 270:
            continue
        if first.xmax < geo.BBOX_NORM_MAX - params.seam_gap_norm:
            continue
        if second.xmin > params.seam_gap_norm:
            continue
        overlap = (min(first.ymax, second.ymax)
                   - max(first.ymin, second.ymin))
        union = (max(first.ymax, second.ymax)
                 - min(first.ymin, second.ymin))
        if union > 0 and overlap / union >= params.seam_min_y_iou:
            return True
    return False


def group_seam_boxes(boxes: list, params: IngestParams) -> list:
    """Union-find grouping of one landmark's boxes across face seams."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _is_seam_pair(boxes[i], boxes[j], params):
                parent[find(i)] = find(j)

    groups: dict = {}
    for i, box in enumerate(boxes):
        groups.setdefault(find(i), []).append(box)
    return list(groups.values())


def _observation_from_group(pano_id: str, frame_idx: int, landmark_idx: int,
                            box_group_idx: int, landmark: dict,
                            group: list, params: IngestParams) -> Observation:
    edges = _box_edges_unwrapped(group, params.fov_deg)
    left = min(e[0] for e in edges)
    right = max(e[1] for e in edges)
    bearing = ((left + right) / 2.0) % 360.0
    width = right - left
    # Elevation at each box's own center (off-axis correct), averaged over
    # the seam group.
    elevation = sum(
        geo.direction_from_face_px(
            box.face_yaw_deg, (box.xmin + box.xmax) / 2.0,
            (box.ymin + box.ymax) / 2.0, params.fov_deg)[1]
        for box in group) / len(group)

    primary_tag = landmark.get("primary_tag") or {}
    additional_tags = [
        [tag.get("key", ""), tag.get("value", "")]
        for tag in landmark.get("additional_tags") or []
    ]
    return Observation(
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
        elevation_deg=elevation,
        angular_width_deg=width,
    )


def run_ingest(dataset_base: Path, frame_landmarks_dir: Path,
               params: IngestParams) -> IngestResult:
    """Frames + camera-frame Observations for one dataset.

    Validates the metadata conventions before touching a pixel; a dataset
    that fails the contract raises rather than producing well-formed wrong
    numbers.
    """
    dataset_base = Path(dataset_base)
    metadata = load_metadata(dataset_base)
    require_camera_frame_panoramas(metadata, dataset_base)

    frames = load_frames(dataset_base)
    if not frames:
        raise FileNotFoundError(f"No panoramas under {dataset_base}/panorama")
    anchor_lat, anchor_lon = fill_enu(frames)
    predictions, n_parse_failures = load_predictions(frame_landmarks_dir)

    stats = IngestStats(n_frames=len(frames),
                        n_parse_failures=n_parse_failures)
    observations: list[Observation] = []
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
                boxes.append(BBox(
                    face_yaw_deg=int(raw_box["yaw_angle"]),
                    xmin=int(raw_box["xmin"]),
                    ymin=int(raw_box["ymin"]),
                    xmax=int(raw_box["xmax"]),
                    ymax=int(raw_box["ymax"]),
                ))
            if not boxes:
                stats.n_landmarks_without_valid_boxes += 1
                continue
            groups = group_seam_boxes(boxes, params)
            for box_group_idx, group in enumerate(groups):
                observations.append(_observation_from_group(
                    frame.pano_id, frame.frame_idx, landmark_idx,
                    box_group_idx, landmark, group, params))
                n_frame_obs += 1
        frame.n_observations = n_frame_obs

    stats.n_observations = len(observations)
    return IngestResult(frames, observations, anchor_lat, anchor_lon, stats)
