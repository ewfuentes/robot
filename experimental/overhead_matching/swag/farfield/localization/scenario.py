"""Synthetic scenario generation for the bearing-only localization harness.

Produces the filter's full input log from a made-up trajectory and a handful
of hardcoded landmarks: sparse tracklet measurements (one fused forward-frame
bearing per tracklet per information epoch, anchors staggered across
tracklets — design doc §5.3), identity-stub CompatibilityTables, body-frame
odometry increments (§5.2), and ground truth.

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
from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    filter_catalog as catalog_mod,
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
    # Catalog visibility radius the run is built with. REQUIRED, no default:
    # it decides which positions the proposal thinks a landmark could be
    # seen from, so a silently-defaulted value changes the hypotheses (see
    # filter_catalog.LandmarkCatalog).
    max_visible_range_m: float
    speed_mps: float = 5.0
    keyframe_period_s: float = 2.0
    # Waypoint corners are rounded with constant-radius arc fillets so the
    # synthetic trajectories satisfy the same constant-turn-rate assumption
    # real (smoothly turning) platforms do — an instantaneous corner would
    # put a whole turn into one keyframe, where the §5.2 midpoint chord is
    # undefined behaviour for any producer. 0 disables (sharp corners).
    corner_radius_m: float = 100.0
    epoch_length_keyframes: int = 5
    bearing_sigma_deg: float = 1.0
    # Per-increment translation noise: sigma_m = odom_sigma_m
    # + odom_sigma_per_m * step. Noise POLICY is a producer decision (§5.2);
    # both parts are declared honestly on the emitted increments.
    odom_sigma_m: float = 1.0
    odom_sigma_per_m: float = 0.0
    # Per-increment yaw noise, declared honestly. Unlike the archived
    # absolute-course model (whose differenced errors telescoped), these
    # errors are independent per step, so heading random-walks between
    # bearing observations — heading noise floors position accuracy at
    # ~range*sigma_heading, so this knob dominates convergence.
    dyaw_sigma_deg: float = 1.5
    # --- model-mismatch knobs: the filter is NOT told about any of these ---
    # Constant COG-vs-heading offset: leeway/crab ONLY. The camera mount
    # camera-to-forward rotation is removed upstream of the filter and is NOT
    # part of the synthetic forward-axis state model.
    # Increments are emitted course-aligned (forward = step, left = 0), the
    # way the gps_to_odometry derivation must (§5.2): real crab is thereby
    # misassigned, and the filter drifts ~step*sin(crab) cross-track.
    course_bias_deg: float = 0.0
    # Constant yaw-rate bias folded into every dyaw (deg per hour).
    gyro_bias_deg_per_hr: float = 0.0
    # Multiplicative error on the translation increments.
    odom_scale_error: float = 0.0
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
    frame: geo.RegionFrame
    # What the filter is given, including the configured uniform map sigma.
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


def _fillet_pieces(east, north, radius_m: float) -> list:
    """Waypoint polyline -> ("line", p0, p1) / ("arc", center, r, a0, sweep)
    pieces with corners rounded by tangent arc fillets.

    `a0` is the angle (standard math convention, in the east/north plane)
    from the arc center to the entry point; `sweep` is signed CCW. A fillet
    is clamped so it consumes at most 45% of each adjoining segment, which
    keeps consecutive fillets from overlapping.
    """
    points = np.stack([east, north], axis=1)
    pieces = []
    cursor = points[0]
    for i in range(1, len(points) - 1):
        v_in = points[i] - points[i - 1]
        v_out = points[i + 1] - points[i]
        len_in, len_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        u_in, u_out = v_in / len_in, v_out / len_out
        turn = math.atan2(float(u_in[0] * u_out[1] - u_in[1] * u_out[0]),
                          float(np.clip(u_in @ u_out, -1.0, 1.0)))
        if radius_m <= 0.0 or abs(turn) < 1e-9:
            pieces.append(("line", cursor, points[i]))
            cursor = points[i]
            continue
        tangent = min(radius_m * math.tan(abs(turn) / 2.0),
                      0.45 * min(len_in, len_out))
        r_eff = tangent / math.tan(abs(turn) / 2.0)
        entry = points[i] - u_in * tangent
        left_of_travel = np.array([-u_in[1], u_in[0]])
        center = entry + left_of_travel * r_eff * math.copysign(1.0, turn)
        pieces.append(("line", cursor, entry))
        a0 = math.atan2(entry[1] - center[1], entry[0] - center[0])
        pieces.append(("arc", center, r_eff, a0, turn))
        cursor = points[i] + u_out * tangent
    pieces.append(("line", cursor, points[-1]))
    return [p for p in pieces
            if not (p[0] == "line" and np.linalg.norm(p[2] - p[1]) < 1e-12)]


def _piece_length(piece) -> float:
    if piece[0] == "line":
        return float(np.linalg.norm(piece[2] - piece[1]))
    _, _, r, _, sweep = piece
    return r * abs(sweep)


def _piece_pose(piece, s: float):
    """(east, north, course_rad) at arc length s into the piece."""
    if piece[0] == "line":
        _, p0, p1 = piece
        u = (p1 - p0) / np.linalg.norm(p1 - p0)
        pos = p0 + u * s
        return float(pos[0]), float(pos[1]), math.atan2(u[0], u[1])
    _, center, r, a0, sweep = piece
    a = a0 + math.copysign(s / r, sweep)
    pos = center + r * np.array([math.cos(a), math.sin(a)])
    tangent = math.copysign(1.0, sweep) * np.array([-math.sin(a),
                                                    math.cos(a)])
    return float(pos[0]), float(pos[1]), math.atan2(tangent[0], tangent[1])


def _build_truth(config: ScenarioConfig) -> list:
    east = np.asarray(config.waypoints_east_m, dtype=np.float64)
    north = np.asarray(config.waypoints_north_m, dtype=np.float64)
    assert len(east) >= 2 and len(east) == len(north)
    assert np.all(np.hypot(np.diff(east), np.diff(north)) > 0.0), (
        "degenerate waypoint segment")
    pieces = _fillet_pieces(east, north, config.corner_radius_m)
    lengths = [_piece_length(p) for p in pieces]
    cum_len = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cum_len[-1])
    step = config.speed_mps * config.keyframe_period_s
    n_keyframes = int(math.floor(total / step + 1e-9)) + 1

    truth = []
    for kf in range(n_keyframes):
        s = min(kf * step, total)
        index = min(int(np.searchsorted(cum_len[1:], s, side="right")),
                    len(pieces) - 1)
        pos_e, pos_n, course_rad = _piece_pose(pieces[index],
                                               s - cum_len[index])
        # Truth records trajectory course only. The unobserved crab/leeway
        # bias enters the forward-frame bearing generator below; it must not
        # masquerade as measured course in serialized truth.
        truth.append(structs.TruthPose(
            keyframe_idx=kf, east_m=pos_e, north_m=pos_n,
            course_world_cw_deg=math.degrees(course_rad) % 360.0))
    return truth


def generate(config: ScenarioConfig) -> ScenarioData:
    assert config.bearing_sigma_deg > 0.0, (
        "bearing_sigma_deg must be positive (kappa = 1/sigma^2)")
    rng = np.random.default_rng(config.generation_seed)
    frame = geo.RegionFrame(config.anchor_lat_deg, config.anchor_lon_deg)
    truth = _build_truth(config)
    n_keyframes = len(truth)

    landmark_ids = [lm.landmark_id for lm in config.landmarks]
    landmark_east_m, landmark_north_m = frame.enu_from_latlon(
        np.array([lm.lat_deg for lm in config.landmarks]),
        np.array([lm.lon_deg for lm in config.landmarks]))

    # Odometry matches gps_to_odometry's forward/left/CW-yaw contract:
    # differenced chord course, then rotate first and move forward in the
    # updated frame. Real leeway remains motion-model uncertainty. Declared
    # sigmas match injected noise; mismatch knobs do not appear in what the
    # filter sees.
    odometry = []
    prev_course_rad = None
    gyro_bias_rad = (math.radians(config.gyro_bias_deg_per_hr)
                     * config.keyframe_period_s / 3600.0)
    sigma_yaw_rad = math.radians(config.dyaw_sigma_deg)
    for kf in range(1, n_keyframes):
        d_east = truth[kf].east_m - truth[kf - 1].east_m
        d_north = truth[kf].north_m - truth[kf - 1].north_m
        step_m = math.hypot(d_east, d_north)
        course_rad = math.atan2(d_east, d_north)
        dyaw_true_rad = 0.0
        if prev_course_rad is not None:
            dyaw_true_rad = float(geo.wrap_rad(
                course_rad - prev_course_rad))
        prev_course_rad = course_rad
        scale = 1.0 + config.odom_scale_error
        sigma_m = config.odom_sigma_m + config.odom_sigma_per_m * step_m
        odometry.append(structs.OdometryDelta(
            keyframe_idx=kf,
            forward_m=(step_m * scale + rng.normal(0.0, sigma_m)),
            left_m=rng.normal(0.0, sigma_m),
            delta_yaw_cw_rad=(dyaw_true_rad + gyro_bias_rad
                      + rng.normal(0.0, sigma_yaw_rad)),
            sigma_m=sigma_m,
            sigma_yaw_rad=sigma_yaw_rad))

    # Tracklet measurements: one fused forward-frame bearing per landmark per
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
            forward_world_cw_deg = (
                pose.course_world_cw_deg + config.course_bias_deg) % 360.0
            forward_rad = float(geo.wrap_rad(
                bearing_world_rad - math.radians(forward_world_cw_deg)))
            if config.clutter_only or rng.random() < config.outlier_frac:
                observed_rad = float(rng.uniform(-math.pi, math.pi))
            else:
                observed_rad = float(rng.vonmises(forward_rad, kappa)
                                     + math.radians(config.bearing_bias_deg))
            measurements.append(structs.TrackletMeasurement(
                tracklet_id=tracklet_id,
                anchor_keyframe_idx=kf,
                bearing_forward_cw_deg=math.degrees(
                    float(geo.wrap_rad(observed_rad))) % 360.0,
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
    configured_sigmas = {
        float(landmark.position_sigma_m) for landmark in config.landmarks}
    if len(configured_sigmas) != 1:
        raise ValueError(
            "synthetic landmarks must use one uniform position_sigma_m")
    configured_position_sigma_m = configured_sigmas.pop()
    catalog = catalog_mod.LandmarkCatalog(
        landmark_ids, true_east_m, true_north_m,
        max_visible_range_m=config.max_visible_range_m,
        position_sigma_m=configured_position_sigma_m)
    if config.catalog_position_sigma_m > 0.0:
        catalog = catalog.perturbed(config.catalog_position_sigma_m, rng)

    return ScenarioData(
        config=config, frame=frame, catalog=catalog,
        true_east_m=true_east_m, true_north_m=true_north_m,
        truth=truth, odometry=odometry, measurements=measurements,
        tables=tables)


def apply_kidnap(data: ScenarioData, at_keyframe: int, east_m: float,
                 north_m: float) -> ScenarioData:
    """Teleport the vehicle mid-run, leaving odometry unaware (T-F6).

    Truth and post-kidnap bearing ideals move; odometry does not. Existing
    bearing residuals are transferred to the moved ideal, preserving noise,
    bias, and outliers. Pre-kidnap measurements remain byte-identical.
    """
    truth = [structs.TruthPose(
        keyframe_idx=pose.keyframe_idx,
        east_m=pose.east_m + (east_m if pose.keyframe_idx >= at_keyframe
                              else 0.0),
        north_m=pose.north_m + (north_m if pose.keyframe_idx >= at_keyframe
                                else 0.0),
        course_world_cw_deg=pose.course_world_cw_deg) for pose in data.truth]
    old_truth_by_kf = {item.keyframe_idx: item for item in data.truth}
    new_truth_by_kf = {item.keyframe_idx: item for item in truth}

    measurements = []
    for measurement in data.measurements:
        if measurement.anchor_keyframe_idx < at_keyframe:
            measurements.append(measurement)
            continue
        old_pose = old_truth_by_kf[measurement.anchor_keyframe_idx]
        new_pose = new_truth_by_kf[measurement.anchor_keyframe_idx]
        index = data.catalog.index_of(
            measurement.tracklet_id.removeprefix("trk_"))
        landmark_east = float(data.true_east_m[index])
        landmark_north = float(data.true_north_m[index])

        def ideal_bearing(pose):
            world = math.atan2(landmark_east - pose.east_m,
                               landmark_north - pose.north_m)
            forward_world = math.radians(
                (pose.course_world_cw_deg
                 + data.config.course_bias_deg) % 360.0)
            return float(geo.wrap_rad(world - forward_world))

        old_ideal = ideal_bearing(old_pose)
        residual = float(geo.wrap_rad(
            math.radians(measurement.bearing_forward_cw_deg) - old_ideal))
        moved_bearing = float(geo.wrap_rad(
            ideal_bearing(new_pose) + residual))
        measurements.append(structs.TrackletMeasurement(
            tracklet_id=measurement.tracklet_id,
            anchor_keyframe_idx=measurement.anchor_keyframe_idx,
            bearing_forward_cw_deg=math.degrees(moved_bearing) % 360.0,
            kappa=measurement.kappa))
    return dataclasses.replace(data, truth=truth, measurements=measurements)


def _landmarks_from_specs(specs) -> list:
    frame = geo.RegionFrame(_DEFAULT_ANCHOR_LAT_DEG,
                                _DEFAULT_ANCHOR_LON_DEG)
    out = []
    for lm_id, east_m, north_m, type_key in specs:
        lat, lon = frame.latlon_from_enu(east_m, north_m)
        out.append(structs.LandmarkEntry(
            landmark_id=lm_id, lat_deg=float(lat), lon_deg=float(lon),
            type_key=type_key, position_sigma_m=1.0))
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
