"""Config structs for the landmark filtering pipeline.

Loaded from YAML via msgspec (repo convention: MSGSPEC_STRUCT_OPTS makes
structs tagged + frozen). The full config is echoed into the run artifact so
every artifact is self-describing.
"""

from pathlib import Path

import msgspec

from common.python.serialization import (
    MSGSPEC_STRUCT_OPTS,
    msgspec_dec_hook,
    msgspec_enc_hook,
)


class IngestConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    fov_deg: float = 90.0
    # Bbox coordinates are normalized 0-1000, so seam margins are in those units.
    seam_gap_norm: int = 25
    seam_min_y_iou: float = 0.3


class ConfidenceGateConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    enabled: bool = True
    # Keep observations with confidence at or above this level.
    min_confidence: str = "medium"


class AngularWidthGateConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    enabled: bool = True
    max_angular_width_deg: float = 45.0


class TagBlocklistConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    enabled: bool = True
    blocked_primary_keys: list[str] = msgspec.field(
        default_factory=lambda: ["highway", "barrier", "crossing"])
    # Exact (key, value) pairs to block, e.g. [["natural", "tree"]].
    blocked_primary_tags: list[list[str]] = msgspec.field(default_factory=list)


class ElevationGateConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    enabled: bool = True
    # Observations whose bbox-center elevation (up positive) is below this are
    # near-field ground clutter.
    min_center_elevation_deg: float = -25.0


class EdgeTruncationConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    enabled: bool = True
    # A box touching a vertical face edge (within this margin, 0-1000 units)
    # without a seam partner has an unreliable bearing.
    edge_margin_norm: int = 25


class IntraFrameDedupConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    enabled: bool = True
    max_bearing_sep_deg: float = 5.0


class HeuristicConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    confidence_gate: ConfidenceGateConfig = msgspec.field(
        default_factory=ConfidenceGateConfig)
    angular_width_gate: AngularWidthGateConfig = msgspec.field(
        default_factory=AngularWidthGateConfig)
    tag_blocklist: TagBlocklistConfig = msgspec.field(
        default_factory=TagBlocklistConfig)
    elevation_gate: ElevationGateConfig = msgspec.field(
        default_factory=ElevationGateConfig)
    edge_truncation: EdgeTruncationConfig = msgspec.field(
        default_factory=EdgeTruncationConfig)
    intra_frame_dedup: IntraFrameDedupConfig = msgspec.field(
        default_factory=IntraFrameDedupConfig)


class AssociationConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    bearing_sigma_deg: float = 3.0
    gate_chi2: float = 9.0
    # Extra gate allowance per meter of camera baseline between matched frames;
    # absorbs the bearing swing of near-field objects (~5 m between frames).
    motion_gate_gain_deg_per_m: float = 2.0
    max_frame_gap: int = 3
    # One of: "description_cosine" | "correspondence_model" | "primary_tag_equality"
    semantic_backend: str = "description_cosine"
    min_similarity: float = 0.75
    similarity_cost_weight: float = 1.0
    min_track_length: int = 3


class TriangulationConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    bearing_sigma_deg: float = 3.0
    # Huber transition point, in units of sigma.
    huber_delta: float = 1.5
    min_parallax_deg: float = 2.0
    # Tracks whose fitted range exceeds this are "far" (bearing-only useful).
    far_range_m: float = 200.0
    max_sigma_major_m: float = 500.0


class YawOffsetConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    # One of: "fixed" | "triangulation_sweep"
    method: str = "fixed"
    fixed_offset_deg: float = 0.0
    # Slow rotation drift of the stabilized frame, used with method="fixed"
    # (e.g. from an external calibration; walk_along_river measures ~-0.23).
    fixed_drift_deg_per_frame: float = 0.0
    sweep_step_deg: float = 1.0
    sweep_top_n_tracks: int = 20
    # +1 if camera-frame bearings increase clockwise-from-above (compass-like),
    # -1 if counterclockwise. A mirror flip is an isometry, so triangulation
    # self-consistency cannot detect it; verify on the map view instead.
    bearing_sign: int = 1


class SemanticSimilarityConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    # Backends to compute diagnostics for (association uses
    # AssociationConfig.semantic_backend).
    diagnostic_backends: list[str] = msgspec.field(
        default_factory=lambda: ["description_cosine", "correspondence_model"])
    # Cap on observations sampled for pairwise diagnostics.
    max_diagnostic_obs: int = 400
    correspondence_model_path: Path = Path(
        "/data/overhead_matching/training_outputs/landmark_correspondence/"
        "simple_v1_v5/best_model.pt")
    text_embeddings_path: Path = Path(
        "/data/overhead_matching/datasets/landmark_correspondence/"
        "eval_text_embeddings_panov2_tuned_v5_all.pkl")
    num_example_pairs: int = 50


class FilterPipelineConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    ingest: IngestConfig = msgspec.field(default_factory=IngestConfig)
    heuristic: HeuristicConfig = msgspec.field(default_factory=HeuristicConfig)
    association: AssociationConfig = msgspec.field(
        default_factory=AssociationConfig)
    triangulation: TriangulationConfig = msgspec.field(
        default_factory=TriangulationConfig)
    yaw_offset: YawOffsetConfig = msgspec.field(default_factory=YawOffsetConfig)
    semantic_similarity: SemanticSimilarityConfig = msgspec.field(
        default_factory=SemanticSimilarityConfig)


def load_config(path: Path) -> FilterPipelineConfig:
    with open(path, "rb") as f:
        return msgspec.yaml.decode(
            f.read(), type=FilterPipelineConfig, dec_hook=msgspec_dec_hook)


def config_to_yaml(config: FilterPipelineConfig) -> str:
    return msgspec.yaml.encode(config, enc_hook=msgspec_enc_hook).decode()
