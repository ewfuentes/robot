"""Run-artifact schema for the landmark filtering pipeline.

One self-describing JSON file per filter run. The viewer renders exclusively
from this artifact: dispositions/reasons/details are opaque strings and dicts
grouped generically, so new filters require no schema or viewer changes.

Configs (echoed into the artifact) use MSGSPEC_STRUCT_OPTS (tagged + frozen);
the records here are plain mutable Structs because observations are annotated
incrementally as pipeline stages run.
"""

from pathlib import Path

import msgspec

from common.python.serialization import msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    FilterPipelineConfig,
)

SCHEMA_VERSION = "1.0"

KEPT = "kept"
FILTERED = "filtered"


class BBox(msgspec.Struct):
    """One Gemini bounding box, coordinates normalized 0-1000 on its face."""
    face_yaw_deg: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class FilterDecision(msgspec.Struct):
    filter_name: str
    disposition: str  # KEPT | FILTERED
    reason: str  # machine-readable, "" when kept
    details: dict[str, float] = {}


class Observation(msgspec.Struct):
    obs_id: str  # f"{pano_id}__lm{landmark_idx}__box{box_group_idx}"
    pano_id: str
    frame_idx: int
    landmark_idx: int
    embedding_id: str | None
    primary_tag_key: str
    primary_tag_value: str
    additional_tags: list[list[str]]
    confidence: str
    description: str
    boxes: list[BBox]
    seam_merged: bool
    bearing_camera_deg: float
    bearing_global_deg: float
    elevation_deg: float
    angular_width_deg: float
    decisions: list[FilterDecision] = []
    final_disposition: str = KEPT
    final_reason: str = ""
    track_id: int | None = None


class Frame(msgspec.Struct):
    frame_idx: int
    pano_id: str
    pano_stem: str  # full "f0005,42.35...,-71.09...," form, locates images
    lat: float
    lon: float
    x_m: float  # local tangent plane east, relative to artifact anchor
    y_m: float  # local tangent plane north
    dist_along_m: float | None = None
    time_s: float | None = None
    n_observations: int = 0


class TriangulationResult(msgspec.Struct):
    solved: bool
    observability: str  # "near" | "far" | "degenerate"
    degenerate_reason: str = ""
    x_m: float | None = None
    y_m: float | None = None
    lat: float | None = None
    lon: float | None = None
    mean_range_m: float | None = None
    residual_rms_deg: float | None = None
    n_inliers: int = 0
    n_outliers: int = 0
    cov_enu: list[list[float]] | None = None
    sigma_major_m: float | None = None
    sigma_minor_m: float | None = None
    parallax_deg: float = 0.0


class Track(msgspec.Struct):
    track_id: int
    obs_ids: list[str]
    first_frame_idx: int
    last_frame_idx: int
    representative_obs_id: str
    mean_pairwise_similarity: float | None = None
    disposition: str = KEPT
    reason: str = ""
    triangulation: TriangulationResult | None = None


class SimilarityPairExample(msgspec.Struct):
    obs_id_a: str
    obs_id_b: str
    score: float
    same_track: bool


class SemanticDiagnostics(msgspec.Struct):
    backend: str
    # Histograms are {bin_left_edge_str: count} with fixed 0.05-wide bins.
    intra_track_similarity_histogram: dict[str, int] = {}
    inter_track_similarity_histogram: dict[str, int] = {}
    top_cross_track_pairs: list[SimilarityPairExample] = []
    bottom_intra_track_pairs: list[SimilarityPairExample] = []
    missing_embedding_values: list[str] = []


class BackendAgreement(msgspec.Struct):
    backend_a: str
    backend_b: str
    correlation: float
    n_pairs: int
    n_large_disagreements: int
    example_disagreements: list[SimilarityPairExample] = []


class SummaryStats(msgspec.Struct):
    n_frames: int = 0
    n_raw_landmark_entries: int = 0
    n_parse_failures: int = 0
    n_boxes_invalid_yaw: int = 0
    n_landmarks_without_valid_boxes: int = 0
    n_observations: int = 0
    n_kept: int = 0
    n_filtered: int = 0
    filtered_by_reason: dict[str, int] = {}
    filtered_by_filter: dict[str, int] = {}
    n_tracks: int = 0
    tracks_by_observability: dict[str, int] = {}
    n_singleton_obs: int = 0
    obs_per_frame_histogram: dict[str, int] = {}


class RunArtifact(msgspec.Struct):
    schema_version: str
    created_at: str  # ISO 8601
    git_hash: str
    dataset_base: str
    pinhole_base: str
    landmark_base: str
    config: FilterPipelineConfig
    stages_run: list[str]
    anchor_lat: float
    anchor_lon: float
    yaw_offset_deg: float = 0.0
    # Slow rotation drift of the stabilized panorama frame; the offset at
    # frame i is yaw_offset_deg + yaw_drift_deg_per_frame * i.
    yaw_drift_deg_per_frame: float = 0.0
    yaw_offset_method: str = "fixed"
    yaw_offset_details: dict[str, float] = {}
    frames: list[Frame] = []
    observations: list[Observation] = []
    tracks: list[Track] = []
    semantic_diagnostics: list[SemanticDiagnostics] = []
    backend_agreements: list[BackendAgreement] = []
    stats: SummaryStats = msgspec.field(default_factory=SummaryStats)


def save_artifact(artifact: RunArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(msgspec.json.format(
            msgspec.json.encode(artifact, enc_hook=msgspec_enc_hook), indent=1))


def load_artifact(path: Path) -> RunArtifact:
    with open(path, "rb") as f:
        return msgspec.json.decode(
            f.read(), type=RunArtifact, dec_hook=msgspec_dec_hook)
