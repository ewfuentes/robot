"""Typed messages and run-artifact records for bearing-only localization.

Mirrors landmark_filtering/artifact_schema.py conventions: configs and
interface messages are tagged + frozen (MSGSPEC_STRUCT_OPTS); per-keyframe
log records are plain mutable Structs.

The CompatibilityTable follows the matcher-seam contract of
docs/localization-design-doc.md §6. TrackletMeasurement is the sparse
information-epoch event of §5.3: one fused body-frame bearing per tracklet
per epoch, anchored at a keyframe. Bearings are body-frame (relative to
vehicle heading, degrees clockwise positive) per the §4 contract — nothing
upstream is assumed north-aligned.
"""

import msgspec

from common.python.serialization import MSGSPEC_STRUCT_OPTS

SCHEMA_VERSION = "0.1"


class LandmarkEntry(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Catalog stub: one known-position landmark."""
    landmark_id: str
    lat_deg: float
    lon_deg: float
    type_key: str


class CompatibilityEntry(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    landmark_id: str
    log_lr: float


class CompatibilityTable(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Matcher-seam output for one tracklet (design doc §6).

    Landmarks absent from `entries` score `default_log_lr`. The filter clips
    every log_lr to [clip_lo, clip_hi]; with an uncalibrated matcher the
    clips and the null hypothesis carry the safety burden.
    """
    tracklet_id: str
    matcher_version: str
    entries: list[CompatibilityEntry]
    default_log_lr: float
    clip_lo: float
    clip_hi: float
    status: str  # "fast" | "refined"


class TrackletMeasurement(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """One information epoch of one tracklet: a fused body-frame bearing."""
    tracklet_id: str
    anchor_keyframe_idx: int
    bearing_body_deg: float  # relative to vehicle heading, CW positive
    kappa: float  # von Mises concentration of the fused bearing


class OdometryDelta(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """World-frame translation from keyframe_idx-1 to keyframe_idx (§5.2).

    GPS-derived deltas are world-frame, so position propagation never routes
    through the heading estimate. Course over ground is the weak heading
    signal; None when unavailable (e.g. too slow).
    """
    keyframe_idx: int
    dx_m: float  # east
    dy_m: float  # north
    sigma_m: float  # per-axis translation noise std
    speed_mps: float
    course_deg: float | None = None
    course_sigma_deg: float | None = None


class GaussianInit(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Local prior: isotropic Gaussian position, uniform heading."""
    mean_east_m: float
    mean_north_m: float
    sigma_m: float


class UniformBoxInit(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Global prior: uniform position over a box, uniform heading."""
    east_min_m: float
    east_max_m: float
    north_min_m: float
    north_max_m: float


class FilterConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    n_particles: int
    seed: int
    init: GaussianInit | UniformBoxInit
    # Per-tracklet-class null-hypothesis probability (single class in v1).
    # 0.2 rather than a token few percent: with an uncalibrated matcher (§6)
    # the null is what stops the filter chasing accidentally-aligned clutter
    # into a confident wrong fix, and VLM false positives are common.
    pi0: float = 0.2
    ess_resample_frac: float = 0.5
    # Heading random-walk noise per keyframe, on top of delta-course
    # rotation (true heading is piecewise-constant between turns, so this
    # only covers unmodeled yaw drift).
    heading_random_walk_deg: float = 1.0
    # Kernel bandwidth for regularized resampling, as a multiple of the
    # Silverman-style rule sigma * n^(-1/6). Required for consistency, not a
    # tuning nicety: at 0 the resampled cloud collapses to duplicated atoms
    # and the filter reports a spread it no longer represents.
    resample_regularization: float = 1.0
    # Absolute jitter floors added on top of the kernel bandwidth, for
    # beliefs too collapsed for a proportional bandwidth to recover.
    position_roughening_m: float = 0.0
    heading_roughening_deg: float = 0.0
    # Bin size for the highest-density (MAP) position estimate.
    map_cell_size_m: float = 50.0
    checkpoint_every: int = 10


class AssociationPosterior(msgspec.Struct):
    """Per-measurement association responsibilities, averaged over the
    posterior particle weights (design doc §5.4; per-mode reporting comes
    with the mode tracker in a later milestone)."""
    tracklet_id: str
    anchor_keyframe_idx: int
    null_share: float
    responsibilities: dict[str, float] = {}


class HealthRecord(msgspec.Struct):
    """Tier-0 health scalars, one per keyframe (design doc §5.6/§7.1).

    All belief statistics describe the WEIGHTED posterior at this keyframe,
    before any resampling or roughening. `resampled` reports whether a
    resample was triggered by this keyframe's ESS, i.e. it acts on the state
    carried into the next keyframe.
    """
    keyframe_idx: int
    ess: float
    resampled: bool
    mean_east_m: float
    mean_north_m: float
    mean_heading_deg: float
    map_east_m: float
    map_north_m: float
    map_heading_deg: float
    position_std_m: float
    heading_std_deg: float
    n_measurements: int
    associations: list[AssociationPosterior] = []


class TruthPose(msgspec.Struct):
    """Ground-truth pose (synthetic runs only)."""
    keyframe_idx: int
    east_m: float
    north_m: float
    heading_deg: float


class RunManifest(msgspec.Struct):
    schema_version: str
    scenario_name: str
    anchor_lat_deg: float
    anchor_lon_deg: float
    n_keyframes: int
    filter_config: FilterConfig
    landmarks: list[LandmarkEntry]
    matcher_version: str
    particle_history_sha256: str = ""
