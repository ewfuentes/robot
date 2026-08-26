"""The dataset contract: frames, metadata validation, and detection ingest.

A farfield dataset directory is a frozen problem definition:

    panorama/f####,<lat>,<lon>,.jpg   equirectangular frames (GPS in the name)
    frames_gps.csv                    idx, latitude, longitude, dist_m, video_t_s
    intrinsics.csv / extraction_log.csv / pano_id_mapping.csv
    pipeline_metadata.json            conventions and source-video pointer

This module is the only reader of that contract. It loads the frame table,
fills the local ENU frame, validates the metadata conventions (loudly — see
below), and converts a completed, dataset-bound `frame_landmarks` artifact
into camera-frame Observations with seam-continuation boxes merged.

Two validations here exist because their absence produced real 180-degree
incidents (docs/conventions.md):

- `require_camera_frame_panoramas` refuses a dataset whose panoramas are
  north-aligned or of unrecorded orientation. Every downstream stage assumes
  camera azimuth is a fixed function of the pano column; on a north-aligned
  dataset that silently becomes an absolute azimuth and heading gets counted
  twice, with every number still well-formed.
Nothing in this module ever writes to a dataset directory.
"""

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    artifact,
    geometry as geo,
    paths as paths_lib,
)

VALID_FACE_YAWS = ("0", "90", "180", "270")
PANO_ID_RE = re.compile(r"f[0-9]+\Z")
PREDICTIONS_NAME = "predictions.jsonl"


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
    dataset_base = Path(dataset_base)
    gps_rows = {}
    gps_csv = dataset_base / "frames_gps.csv"
    if not gps_csv.is_file():
        raise ContractViolation(f"{gps_csv} does not exist")
    with open(gps_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"idx", "latitude", "longitude", "dist_m", "video_t_s"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ContractViolation(
                f"{gps_csv} must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                idx = int(row["idx"])
                values = [float(row[name]) for name in
                          ("latitude", "longitude", "dist_m", "video_t_s")]
            except (TypeError, ValueError) as exc:
                raise ContractViolation(
                    f"{gps_csv}:{row_number}: invalid numeric field") from exc
            if idx < 0 or not all(math.isfinite(value) for value in values):
                raise ContractViolation(
                    f"{gps_csv}:{row_number}: indices must be nonnegative and "
                    "numeric fields finite")
            if idx in gps_rows:
                raise ContractViolation(f"{gps_csv}: duplicate idx {idx}")
            gps_rows[idx] = row

    frames = []
    panorama_dir = dataset_base / "panorama"
    if not panorama_dir.is_dir():
        raise ContractViolation(f"{panorama_dir} does not exist")
    pano_paths = sorted(panorama_dir.glob("*.jpg"))
    seen_ids = set()
    seen_stems = set()
    used_gps = set()
    for path in pano_paths:
        parts = path.stem.split(",")
        if len(parts) != 4 or parts[-1] != "" or not PANO_ID_RE.fullmatch(parts[0]):
            raise ContractViolation(
                f"{path}: panorama filename must be "
                "f####,<latitude>,<longitude>,.jpg")
        pano_id, lat_str, lon_str, _ = parts
        if pano_id in seen_ids or path.stem in seen_stems:
            raise ContractViolation(f"{path}: duplicate panorama identity")
        seen_ids.add(pano_id)
        seen_stems.add(path.stem)
        gps_idx = int(pano_id[1:])
        gps = gps_rows.get(gps_idx)
        if gps is None:
            raise ContractViolation(
                f"{path}: no frames_gps.csv row for idx {gps_idx}")
        used_gps.add(gps_idx)
        try:
            lat, lon = float(lat_str), float(lon_str)
        except ValueError as exc:
            raise ContractViolation(f"{path}: invalid latitude/longitude") from exc
        if not (math.isfinite(lat) and math.isfinite(lon)
                and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ContractViolation(f"{path}: invalid latitude/longitude")
        frames.append(Frame(
            frame_idx=len(frames),
            pano_id=pano_id,
            pano_stem=path.stem,
            lat=lat,
            lon=lon,
            dist_along_m=float(gps["dist_m"]),
            time_s=float(gps["video_t_s"]),
        ))
    extra_gps = sorted(set(gps_rows) - used_gps)
    if extra_gps:
        raise ContractViolation(
            f"{gps_csv}: rows without panoramas: {extra_gps[:10]}")
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
    result = {fr.pano_id: fr.frame_idx for fr in frames}
    if len(result) != len(frames):
        raise ContractViolation("duplicate pano_id in frame table")
    return result


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
            f"(camera-storage conventions and source-video pointer).")
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractViolation(f"{path} is invalid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ContractViolation(f"{path}: root must be a JSON object")
    dataset_name = metadata.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ContractViolation(f"{path}: dataset_name must be a non-empty string")
    return metadata


def require_camera_frame_panoramas(metadata: dict, dataset_base: Path) -> None:
    """Refuse a dataset whose panoramas are not stored in the camera frame.

    The whole tracking pipeline assumes `north_aligned: false` (camera azimuth
    is a fixed function of the pano column). This must be recorded, not
    assumed: on a north-aligned dataset every bearing silently becomes
    absolute and downstream adds heading to it again.
    """
    is_equirectangular = metadata.get("is_equirectangular")
    if type(is_equirectangular) is not bool:
        raise ContractViolation(
            f"{dataset_base}: pipeline_metadata.json 'is_equirectangular' "
            "must be an actual boolean")
    if not is_equirectangular:
        raise ContractViolation(
            f"{dataset_base}: perspective imagery is not supported; "
            "is_equirectangular must be true")

    north_aligned = metadata.get("north_aligned")
    if type(north_aligned) is not bool:
        raise ContractViolation(
            f"{dataset_base}: pipeline_metadata.json 'north_aligned' must be "
            "an actual boolean. The pipeline requires camera-frame panoramas "
            "and refuses to guess their orientation.")
    if north_aligned:
        raise ContractViolation(
            f"{dataset_base}: panoramas are north-aligned (north_aligned: "
            f"true). This pipeline computes camera-frame azimuths from pano "
            f"columns; feeding it north-aligned frames double-counts heading "
            f"with every number still well-formed. Ingest the raw "
            f"camera-frame frames instead.")

    convention = metadata.get("azimuth_convention")
    if not isinstance(convention, dict):
        raise ContractViolation(
            f"{dataset_base}: pipeline_metadata.json 'azimuth_convention' "
            "must be an object")
    images_rotated = convention.get("images_rotated")
    if type(images_rotated) is not bool:
        raise ContractViolation(
            f"{dataset_base}: azimuth_convention.images_rotated must be an "
            "actual boolean")
    if images_rotated:
        raise ContractViolation(
            f"{dataset_base}: azimuth_convention.images_rotated must be false; "
            "prediction requires unrotated camera-frame panoramas")
    if convention.get("camera_frame") != geo.CAMERA_FRAME:
        raise ContractViolation(
            f"{dataset_base}: azimuth_convention.camera_frame must be the "
            f"canonical camera frame {geo.CAMERA_FRAME!r}")


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


@dataclass(frozen=True)
class ObservationKey:
    """Dataset- and artifact-scoped identity for one local detection."""

    dataset: str
    frame_landmarks_version: str
    frame_landmarks_content_digest: str
    local_obs_id: str

    @property
    def global_id(self) -> str:
        identity = {
            "dataset": self.dataset,
            "kind": paths_lib.FRAME_LANDMARKS,
            "version": self.frame_landmarks_version,
            "content_digest": self.frame_landmarks_content_digest,
            "local_obs_id": self.local_obs_id,
        }
        return "obs-" + artifact.sha256_json(identity)


@dataclass(frozen=True)
class Observation:
    key: ObservationKey
    obs_id: str             # globally unique, derived from key
    local_obs_id: str       # f"{pano_id}__lm{landmark_idx}__box{group_idx}"
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
    bearing_camera_cw_deg: float
    elevation_deg: float
    angular_width_deg: float


@dataclass
class IngestParams:
    """Detection-ingest parameters supplied by the recorded build config."""
    fov_deg: float
    seam_gap_norm: float    # bbox units (0-1000): margin for a seam candidate
    seam_min_y_iou: float   # vertical IoU to accept a seam continuation


@dataclass
class IngestStats:
    n_frames: int = 0
    n_raw_landmark_entries: int = 0
    n_observations: int = 0
    n_boxes_invalid_geometry: int = 0
    n_landmarks_without_valid_boxes: int = 0

    @property
    def lossy(self) -> bool:
        """Whether ingest silently discarded predicted geometry."""
        return bool(self.n_boxes_invalid_geometry
                    or self.n_landmarks_without_valid_boxes)

    def summary(self) -> str:
        """One line naming what was read and what was dropped.

        `run_ingest` drops a malformed bounding box rather than raising, so
        this is the only place a reader learns that detections went missing.
        A run that quietly ingested 90% of its geometry and one that ingested
        all of it otherwise look identical, all the way to a localization
        result. The `WARNING` prefix is deliberate: log sweeps grep for it.
        """
        line = (f"ingest: {self.n_frames} frames, "
                f"{self.n_raw_landmark_entries} predicted landmarks -> "
                f"{self.n_observations} observations")
        if not self.lossy:
            return line + "; no predicted geometry discarded"
        return (f"WARNING {line}; DISCARDED "
                f"{self.n_boxes_invalid_geometry} malformed bounding box(es) "
                f"and dropped {self.n_landmarks_without_valid_boxes} landmark"
                f"(s) left with none. Detections are missing from everything "
                f"downstream of this ingest.")


class IngestResult:
    def __init__(self, frames, observations, anchor_lat, anchor_lon, stats,
                 dataset_name, frame_landmarks_ref):
        self.frames: list[Frame] = frames
        self.observations: list[Observation] = observations
        self.anchor_lat: float = anchor_lat
        self.anchor_lon: float = anchor_lon
        self.stats: IngestStats = stats
        self.dataset_name: str = dataset_name
        self.frame_landmarks_ref: artifact.ArtifactRef = frame_landmarks_ref
        self.frame_index_by_pano_id: dict = frame_index_by_pano_id(frames)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractViolation(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: dict, expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractViolation(
            f"{where}: expected keys {sorted(expected)}, found {sorted(actual)}")


def load_predictions(frame_landmarks_dir: Path,
                     expected_pano_stems: set[str]) -> dict:
    """Read the one canonical, fully-covered extraction result.

    Transport responses and attempts are deliberately not accepted here. The
    extraction stage must first validate them and publish one canonical
    ``predictions.jsonl`` record for every panorama.
    """
    predictions = {}
    jsonl_path = Path(frame_landmarks_dir) / PREDICTIONS_NAME
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"{jsonl_path} does not exist; publish the completed extraction "
            "artifact first")
    try:
        with jsonl_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: blank records are invalid")
                try:
                    record = json.loads(
                        line, object_pairs_hook=_reject_duplicate_json_keys,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ContractViolation(
                                f"invalid non-finite JSON value {value!r}")))
                except ContractViolation:
                    raise
                except json.JSONDecodeError as exc:
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: record must be an object")
                _require_exact_keys(
                    record, {"key", "prediction"},
                    f"{jsonl_path}:{line_number}")
                key = record["key"]
                prediction = record["prediction"]
                if not isinstance(key, str) or not key:
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: key must be non-empty")
                if key in predictions:
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: duplicate key {key!r}")
                if not isinstance(prediction, dict):
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: prediction must be an object")
                _require_exact_keys(
                    prediction, {"location_type", "landmarks"},
                    f"{jsonl_path}:{line_number}.prediction")
                if not isinstance(prediction["location_type"], str):
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: location_type must be a string")
                if not isinstance(prediction["landmarks"], list):
                    raise ContractViolation(
                        f"{jsonl_path}:{line_number}: landmarks must be a list")
                predictions[key] = prediction
    except OSError as exc:
        raise ContractViolation(f"cannot read {jsonl_path}: {exc}") from exc
    actual = set(predictions)
    if actual != expected_pano_stems:
        missing = sorted(expected_pano_stems - actual)
        unknown = sorted(actual - expected_pano_stems)
        raise ContractViolation(
            "frame_landmarks coverage is not exact: "
            f"missing={missing[:10]}, unknown={unknown[:10]}")
    return predictions


def _box_edges_unwrapped(boxes: list, fov_deg: float):
    """(left, right) bearing edges per box, unwrapped to a continuous range.

    Angles are lifted out of [0, 360) so that a group of boxes spanning the
    0/360 wrap forms one continuous interval.
    """
    edges = []
    reference = None
    for box in boxes:
        left = geo.bearing_camera_cw_deg(box.face_yaw_deg, box.xmin, fov_deg)
        right = geo.bearing_camera_cw_deg(box.face_yaw_deg, box.xmax, fov_deg)
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


def _finite_number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractViolation(f"{where} must be finite")
    return result


def _validated_landmark(landmark, where: str) \
        -> tuple[dict, list[BBox], int]:
    if not isinstance(landmark, dict):
        raise ContractViolation(f"{where} must be an object")
    _require_exact_keys(
        landmark,
        {"description", "confidence", "primary_tag", "additional_tags",
         "bounding_boxes"},
        where)
    for key in ("description", "confidence"):
        if not isinstance(landmark[key], str):
            raise ContractViolation(f"{where}.{key} must be a string")
    primary_tag = landmark["primary_tag"]
    if not isinstance(primary_tag, dict):
        raise ContractViolation(f"{where}.primary_tag must be an object")
    _require_exact_keys(primary_tag, {"key", "value"},
                        f"{where}.primary_tag")
    if not all(isinstance(primary_tag[key], str) for key in ("key", "value")):
        raise ContractViolation(
            f"{where}.primary_tag key and value must be strings")
    additional_tags = landmark["additional_tags"]
    if not isinstance(additional_tags, list):
        raise ContractViolation(f"{where}.additional_tags must be a list")
    for index, tag in enumerate(additional_tags):
        tag_where = f"{where}.additional_tags[{index}]"
        if not isinstance(tag, dict):
            raise ContractViolation(f"{tag_where} must be an object")
        _require_exact_keys(tag, {"key", "value"}, tag_where)
        if not all(isinstance(tag[key], str) for key in ("key", "value")):
            raise ContractViolation(f"{tag_where} values must be strings")

    raw_boxes = landmark["bounding_boxes"]
    if not isinstance(raw_boxes, list):
        raise ContractViolation(f"{where}.bounding_boxes must be a list")
    boxes = []
    n_invalid = 0
    for index, raw_box in enumerate(raw_boxes):
        box_where = f"{where}.bounding_boxes[{index}]"
        try:
            if not isinstance(raw_box, dict):
                raise ContractViolation(f"{box_where} must be an object")
            _require_exact_keys(
                raw_box, {"yaw_angle", "xmin", "ymin", "xmax", "ymax"},
                box_where)
            yaw = raw_box["yaw_angle"]
            if (isinstance(yaw, bool) or not isinstance(yaw, int)
                    or str(yaw) not in VALID_FACE_YAWS):
                raise ContractViolation(
                    f"{box_where}.yaw_angle must be one of "
                    f"{[int(value) for value in VALID_FACE_YAWS]}")
            coordinates = {
                name: _finite_number(raw_box[name], f"{box_where}.{name}")
                for name in ("xmin", "ymin", "xmax", "ymax")
            }
            if not all(0.0 <= value <= geo.BBOX_NORM_MAX
                       for value in coordinates.values()):
                raise ContractViolation(
                    f"{box_where}: coordinates must be in "
                    f"[0, {geo.BBOX_NORM_MAX}]")
            if (coordinates["xmin"] >= coordinates["xmax"]
                    or coordinates["ymin"] >= coordinates["ymax"]):
                raise ContractViolation(
                    f"{box_where}: bbox must have positive width and height")
        except ContractViolation:
            n_invalid += 1
            continue
        boxes.append(BBox(face_yaw_deg=yaw, **coordinates))
    return landmark, boxes, n_invalid


def _observation_from_group(pano_id: str, frame_idx: int, landmark_idx: int,
                            box_group_idx: int, landmark: dict,
                            group: list, params: IngestParams,
                            dataset_name: str,
                            frame_landmarks_ref: artifact.ArtifactRef) \
        -> Observation:
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

    primary_tag = landmark["primary_tag"]
    additional_tags = [
        [tag["key"], tag["value"]]
        for tag in landmark["additional_tags"]
    ]
    local_obs_id = f"{pano_id}__lm{landmark_idx}__box{box_group_idx}"
    key = ObservationKey(
        dataset=dataset_name,
        frame_landmarks_version=frame_landmarks_ref.version,
        frame_landmarks_content_digest=frame_landmarks_ref.content_digest,
        local_obs_id=local_obs_id)
    embedding_identity = {
        "dataset": dataset_name,
        "kind": paths_lib.FRAME_LANDMARKS,
        "version": frame_landmarks_ref.version,
        "content_digest": frame_landmarks_ref.content_digest,
        "local_embedding_id": f"{pano_id}__landmark_{landmark_idx}",
    }
    return Observation(
        key=key,
        obs_id=key.global_id,
        local_obs_id=local_obs_id,
        pano_id=pano_id,
        frame_idx=frame_idx,
        landmark_idx=landmark_idx,
        embedding_id="emb-" + artifact.sha256_json(embedding_identity),
        primary_tag_key=primary_tag["key"],
        primary_tag_value=primary_tag["value"],
        additional_tags=additional_tags,
        confidence=landmark["confidence"],
        description=landmark["description"],
        boxes=group,
        seam_merged=len(group) > 1,
        bearing_camera_cw_deg=bearing,
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
    dataset_name = metadata["dataset_name"]
    try:
        frame_landmarks_ref = artifact.open_artifact(
            frame_landmarks_dir,
            expected_kind=paths_lib.FRAME_LANDMARKS,
            expected_dataset=dataset_name)
    except artifact.ArtifactError as exc:
        raise ContractViolation(
            f"invalid frame_landmarks artifact {frame_landmarks_dir}: {exc}") \
            from exc

    frames = load_frames(dataset_base)
    if not frames:
        raise FileNotFoundError(f"No panoramas under {dataset_base}/panorama")
    anchor_lat, anchor_lon = fill_enu(frames)
    predictions = load_predictions(
        frame_landmarks_dir, {frame.pano_stem for frame in frames})

    stats = IngestStats(n_frames=len(frames))
    observations: list[Observation] = []
    for frame in frames:
        prediction = predictions[frame.pano_stem]
        n_frame_obs = 0
        for landmark_idx, landmark in enumerate(prediction["landmarks"]):
            stats.n_raw_landmark_entries += 1
            landmark, boxes, n_invalid = _validated_landmark(
                landmark,
                f"prediction[{frame.pano_stem!r}].landmarks[{landmark_idx}]")
            stats.n_boxes_invalid_geometry += n_invalid
            if not boxes:
                stats.n_landmarks_without_valid_boxes += 1
                continue
            groups = group_seam_boxes(boxes, params)
            for box_group_idx, group in enumerate(groups):
                observations.append(_observation_from_group(
                    frame.pano_id, frame.frame_idx, landmark_idx,
                    box_group_idx, landmark, group, params, dataset_name,
                    frame_landmarks_ref))
                n_frame_obs += 1
        frame.n_observations = n_frame_obs

    stats.n_observations = len(observations)
    # Reported here rather than left to each caller: there are four call
    # sites, a caller that forgets is indistinguishable from a clean ingest,
    # and the counters exist precisely because this function drops geometry
    # instead of raising. stderr so it cannot disturb parsed stdout.
    print(stats.summary(), file=sys.stderr, flush=True)
    return IngestResult(
        frames, observations, anchor_lat, anchor_lon, stats,
        dataset_name, frame_landmarks_ref)
