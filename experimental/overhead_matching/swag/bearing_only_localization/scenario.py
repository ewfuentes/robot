"""Synthetic scenario generation for the bearing-only localization harness.

Produces the filter's full input log from a made-up trajectory and a handful
of hardcoded landmarks: sparse tracklet measurements (one fused body-frame
bearing per tracklet per information epoch, anchors staggered across
tracklets — design doc §5.3), identity-stub CompatibilityTables, world-frame
odometry, course over ground, and ground truth.

Perfect observability is the default: every landmark is always visible, no
dropout, no clutter, and the generator's noise model is exactly the filter's.
That default measures "does Bayes invert this generator", which is necessary
but not sufficient — the model-mismatch knobs (crab bias, bearing bias,
outliers, catalog position error, dropout, clutter) are what make the T-F
suite meaningful, and none of them is visible to the filter.
"""

import dataclasses
import math

import msgspec
import numpy as np

from common.python.serialization import MSGSPEC_STRUCT_OPTS
from experimental.overhead_matching.swag.bearing_only_localization import (
    catalog as catalog_mod,
    geodesy,
    structs,
)

MATCHER_VERSION = "identity_stub_v1"

_DEFAULT_ANCHOR_LAT_DEG = 42.335
_DEFAULT_ANCHOR_LON_DEG = -70.99

# (landmark_id, east_m, north_m, type_key) relative to the default anchor.
# Made-up but plausibly-ranged Boston-Harbor-ish far-field landmarks.
_DEFAULT_LANDMARK_SPECS = [
    ("graves_light", 2200.0, 1800.0, "lighthouse"),
    ("boston_light", 2600.0, -2100.0, "lighthouse"),
    ("deer_island_tank", -1500.0, 1200.0, "storage_tank"),
]


class ScenarioConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    name: str
    anchor_lat_deg: float
    anchor_lon_deg: float
    landmarks: list[structs.LandmarkEntry]
    waypoints_east_m: list[float]
    waypoints_north_m: list[float]
    speed_mps: float = 5.0
    keyframe_period_s: float = 2.0
    epoch_length_keyframes: int = 5
    bearing_sigma_deg: float = 1.0
    # GPS-derived deltas have ~constant per-fix-pair noise (correlated
    # absolute errors difference out to ~1 m), unlike wheel odometry's
    # sqrt-distance scaling. This is also the filter's only position
    # diversity between resamples — starving it strands the cloud.
    odom_sigma_m: float = 1.0
    # Differenced-GPS course noise after light smoothing (~2 keyframe
    # baselines): sigma_rel*sqrt(2)/(2*step) ~ 1.4 m / 50 m ~ 1.5 deg
    # (design doc §5.2 noise ∝ 1/(v*window)). Heading noise floors position
    # accuracy at ~range*sigma_heading, so this knob dominates convergence.
    course_sigma_deg: float = 1.5
    # --- model-mismatch knobs: the filter is NOT told about any of these ---
    # Constant COG-vs-heading offset (leeway/crab + mount misalignment).
    course_bias_deg: float = 0.0
    # Systematic bearing offset (residual yaw miscalibration upstream).
    bearing_bias_deg: float = 0.0
    # Fraction of measurements replaced by a uniformly-random bearing.
    outlier_frac: float = 0.0
    # Catalog position error, applied to the positions the FILTER sees.
    catalog_position_sigma_m: float = 0.0
    dropout_prob: float = 0.0
    clutter_only: bool = False
    generation_seed: int = 1234
    identity_clip: float = 4.0
    identity_default_log_lr: float = -2.0


@dataclasses.dataclass
class ScenarioData:
    config: ScenarioConfig
    frame: geodesy.RegionFrame
    # What the filter is given: positions carry catalog_position_sigma_m.
    catalog: catalog_mod.LandmarkCatalog
    # Where the landmarks really are (bearings are generated from these).
    true_east_m: np.ndarray
    true_north_m: np.ndarray
    truth: list  # list[structs.TruthPose], keyframes 0..T
    odometry: list  # list[structs.OdometryDelta], keyframes 1..T
    measurements: list  # list[structs.TrackletMeasurement]
    tables: dict  # tracklet_id -> structs.CompatibilityTable

    @property
    def n_keyframes(self) -> int:
        return len(self.truth)

    @property
    def landmark_ids(self) -> list:
        return self.catalog.landmark_ids


def _build_truth(config: ScenarioConfig) -> list:
    east = np.asarray(config.waypoints_east_m, dtype=np.float64)
    north = np.asarray(config.waypoints_north_m, dtype=np.float64)
    assert len(east) >= 2 and len(east) == len(north)
    seg_d_east = np.diff(east)
    seg_d_north = np.diff(north)
    seg_len = np.hypot(seg_d_east, seg_d_north)
    assert np.all(seg_len > 0.0), "degenerate waypoint segment"
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum_len[-1])
    step = config.speed_mps * config.keyframe_period_s
    n_keyframes = int(math.floor(total / step + 1e-9)) + 1

    truth = []
    for kf in range(n_keyframes):
        s = min(kf * step, total)
        seg = min(int(np.searchsorted(cum_len[1:], s, side="right")),
                  len(seg_len) - 1)
        frac = (s - cum_len[seg]) / seg_len[seg]
        pos_e = east[seg] + frac * seg_d_east[seg]
        pos_n = north[seg] + frac * seg_d_north[seg]
        course_deg = math.degrees(
            math.atan2(seg_d_east[seg], seg_d_north[seg])) % 360.0
        # True heading = course over ground + crab/leeway + mount offset.
        # Bearings are measured relative to heading; odometry reports course.
        truth.append(structs.TruthPose(
            keyframe_idx=kf, east_m=float(pos_e), north_m=float(pos_n),
            heading_deg=(course_deg + config.course_bias_deg) % 360.0))
    return truth


def generate(config: ScenarioConfig) -> ScenarioData:
    assert config.bearing_sigma_deg > 0.0, (
        "bearing_sigma_deg must be positive (kappa = 1/sigma^2)")
    rng = np.random.default_rng(config.generation_seed)
    frame = geodesy.RegionFrame(config.anchor_lat_deg, config.anchor_lon_deg)
    truth = _build_truth(config)
    n_keyframes = len(truth)

    landmark_ids = [lm.landmark_id for lm in config.landmarks]
    landmark_east_m, landmark_north_m = frame.enu_from_latlon(
        np.array([lm.lat_deg for lm in config.landmarks]),
        np.array([lm.lon_deg for lm in config.landmarks]))

    # Odometry: world-frame deltas + sqrt-distance noise; course over ground.
    odometry = []
    for kf in range(1, n_keyframes):
        d_east = truth[kf].east_m - truth[kf - 1].east_m
        d_north = truth[kf].north_m - truth[kf - 1].north_m
        step_m = math.hypot(d_east, d_north)
        sigma_m = config.odom_sigma_m
        course_true_deg = math.degrees(math.atan2(d_east, d_north)) % 360.0
        odometry.append(structs.OdometryDelta(
            keyframe_idx=kf,
            dx_m=d_east + rng.normal(0.0, sigma_m),
            dy_m=d_north + rng.normal(0.0, sigma_m),
            sigma_m=sigma_m,
            speed_mps=step_m / config.keyframe_period_s,
            course_deg=(course_true_deg
                        + rng.normal(0.0, config.course_sigma_deg)) % 360.0,
            course_sigma_deg=config.course_sigma_deg))

    # Tracklet measurements: one fused body-frame bearing per landmark per
    # epoch, anchors staggered round-robin across tracklets.
    kappa = 1.0 / math.radians(config.bearing_sigma_deg) ** 2
    epoch = max(1, config.epoch_length_keyframes)
    measurements = []
    for i, lm_id in enumerate(landmark_ids):
        tracklet_id = f"trk_{lm_id}"
        offset = (i * epoch) // max(1, len(landmark_ids))
        for kf in range(offset, n_keyframes, epoch):
            if rng.random() < config.dropout_prob:
                continue
            pose = truth[kf]
            bearing_world_rad = math.atan2(
                float(landmark_east_m[i]) - pose.east_m,
                float(landmark_north_m[i]) - pose.north_m)
            body_rad = float(geodesy.wrap_rad(
                bearing_world_rad - math.radians(pose.heading_deg)))
            if config.clutter_only or rng.random() < config.outlier_frac:
                observed_rad = float(rng.uniform(-math.pi, math.pi))
            else:
                observed_rad = float(rng.vonmises(body_rad, kappa)
                                     + math.radians(config.bearing_bias_deg))
            measurements.append(structs.TrackletMeasurement(
                tracklet_id=tracklet_id,
                anchor_keyframe_idx=kf,
                bearing_body_deg=math.degrees(
                    float(geodesy.wrap_rad(observed_rad))),
                kappa=kappa))
    measurements.sort(key=lambda m: (m.anchor_keyframe_idx, m.tracklet_id))

    # Identity-stub CompatibilityTables (§6 v1 posture): the true landmark
    # scores +clip, everything else the default. Clutter tracklets prefer
    # nothing.
    tables = {}
    for lm_id in landmark_ids:
        tracklet_id = f"trk_{lm_id}"
        entries = [] if config.clutter_only else [
            structs.CompatibilityEntry(lm_id, config.identity_clip)]
        tables[tracklet_id] = structs.CompatibilityTable(
            tracklet_id=tracklet_id,
            matcher_version=MATCHER_VERSION,
            entries=entries,
            default_log_lr=config.identity_default_log_lr,
            clip_lo=-config.identity_clip,
            clip_hi=config.identity_clip,
            status="fast")

    true_east_m = np.asarray(landmark_east_m, dtype=np.float64)
    true_north_m = np.asarray(landmark_north_m, dtype=np.float64)
    catalog = catalog_mod.LandmarkCatalog(landmark_ids, true_east_m,
                                          true_north_m)
    if config.catalog_position_sigma_m > 0.0:
        catalog = catalog.perturbed(config.catalog_position_sigma_m, rng)

    return ScenarioData(
        config=config, frame=frame, catalog=catalog,
        true_east_m=true_east_m, true_north_m=true_north_m,
        truth=truth, odometry=odometry, measurements=measurements,
        tables=tables)


def _landmarks_from_specs(specs) -> list:
    frame = geodesy.RegionFrame(_DEFAULT_ANCHOR_LAT_DEG,
                                _DEFAULT_ANCHOR_LON_DEG)
    out = []
    for lm_id, east_m, north_m, type_key in specs:
        lat, lon = frame.latlon_from_enu(east_m, north_m)
        out.append(structs.LandmarkEntry(
            landmark_id=lm_id, lat_deg=float(lat), lon_deg=float(lon),
            type_key=type_key))
    return out


def _make_config(name: str, waypoints_east_m: list, waypoints_north_m: list,
                 overrides: dict, specs=None) -> ScenarioConfig:
    base = dict(
        name=name,
        anchor_lat_deg=_DEFAULT_ANCHOR_LAT_DEG,
        anchor_lon_deg=_DEFAULT_ANCHOR_LON_DEG,
        landmarks=_landmarks_from_specs(specs or _DEFAULT_LANDMARK_SPECS),
        waypoints_east_m=waypoints_east_m,
        waypoints_north_m=waypoints_north_m)
    base.update(overrides)
    return ScenarioConfig(**base)


def straight_leg(**overrides) -> ScenarioConfig:
    return _make_config("straight_leg",
                        [-1000.0, 1000.0], [-500.0, -500.0], overrides)


def l_turn(**overrides) -> ScenarioConfig:
    return _make_config("l_turn",
                        [-1000.0, 500.0, 500.0], [-800.0, -800.0, 500.0],
                        overrides)


def harbor_loop(**overrides) -> ScenarioConfig:
    return _make_config(
        "harbor_loop",
        [-800.0, 800.0, 800.0, -800.0, -800.0],
        [-800.0, -800.0, 600.0, 600.0, -800.0], overrides)


def symmetric_pair(**overrides) -> ScenarioConfig:
    """Two identical lighthouses mirrored about the track (T-F3).

    The trajectory runs due north up the east=0 axis, so a pose and its
    mirror image produce identical bearing sets: the posterior is genuinely
    bimodal and no amount of evidence from these two landmarks can break the
    tie. Premature collapse to one mode is the failure this exists to catch.
    """
    return _make_config(
        "symmetric_pair", [0.0, 0.0], [-1000.0, 1000.0], overrides,
        specs=[("twin_light_west", -2000.0, 0.0, "lighthouse"),
               ("twin_light_east", 2000.0, 0.0, "lighthouse")])


SCENARIO_BUILDERS = {
    "straight_leg": straight_leg,
    "l_turn": l_turn,
    "harbor_loop": harbor_loop,
    "symmetric_pair": symmetric_pair,
}


def get_scenario_config(name: str, **overrides) -> ScenarioConfig:
    return SCENARIO_BUILDERS[name](**overrides)
