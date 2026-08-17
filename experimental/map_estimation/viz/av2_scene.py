"""Logs an Argoverse 2 log into rerun: the HD map, the vehicle, its path, the lidar, the cameras.

The entity tree *is* the transform hierarchy -- a child inherits its parent's ``Transform3D`` --
so the layout below is the whole coordinate story:

    world                    city frame, right-handed z-up
    world/map/...            the log's vector HD map, static, in city coordinates
    world/path               the whole drive as one polyline, static, in city coordinates
    world/lidar              the returns, one Points3D per sweep, in city coordinates
    world/ego                city_SE3_egovehicle, one per pose timestamp
    world/ego/wireframe      the vehicle outline, logged once and carried by the ego transform
    world/ego/cameras/<name> ego_SE3_cam + Pinhole, both static; one EncodedImage per frame

The wireframe line is the part worth internalizing: it is logged **once**, as static data, and
it moves because its parent moves. Nothing re-logs geometry per frame. The cameras go one level
further and compose two transforms, a static ``ego_SE3_cam`` under a per-timestamp ego pose, so
an image logged with no pose information at all still lands in the right place in the city.

The map, the path, and the lidar are **siblings** of the ego rather than children, and that is
load bearing: all three are expressed in city coordinates, so inheriting the ego transform would
drag the road along with the car instead of letting the car drive through it.

The lidar is the one entity that had a genuine choice of frame, since AV2 hands the sweeps over
already egomotion-compensated into the *ego* frame -- see :func:`log_lidar` for why they are
transformed into city coordinates anyway, and :func:`log_cameras` for why the imagery, which had
the same choice, went the other way.

Model output belongs under :data:`PREDICTION_CITY` or :data:`PREDICTION_EGO`, siblings of the
ground-truth paths under the same transforms, so the two can be toggled and compared without
either code path knowing about the other.
"""

import dataclasses
import logging

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from experimental.map_estimation.viz import av2_source

WORLD = "world"
MAP = f"{WORLD}/map"
LANE_BOUNDARIES = f"{MAP}/lane_boundaries"
CENTERLINES = f"{MAP}/centerlines"
CROSSWALKS = f"{MAP}/crosswalks"
DRIVABLE_AREAS = f"{MAP}/drivable_areas"
PATH = f"{WORLD}/path"
LIDAR = f"{WORLD}/lidar"
EGO = f"{WORLD}/ego"
WIREFRAME = f"{EGO}/wireframe"
CAMERAS = f"{EGO}/cameras"

PREDICTION_CITY = f"{WORLD}/prediction"
"""Where to log model output expressed in city coordinates."""
PREDICTION_EGO = f"{EGO}/prediction"
"""Where to log model output expressed in the egovehicle frame."""

TIMELINE_ELAPSED = "elapsed"
"""Seconds since the log's first pose. The timeline you actually scrub.

NOT named ``log_time``: that one is reserved. Rerun stamps every ``rr.log`` call with built-in
``log_time`` (wall clock) and ``log_tick`` (call counter) timelines, so reusing the name makes
the SDK complain that the timeline changed type and then interleaves our values with its own.
"""
TIMELINE_TIMESTAMP = "timestamp_ns"
"""Raw AV2 nanosecond timestamps, for cross-referencing a frame against files on disk.

A **sequence** rather than a duration or a timestamp, because that is what these numbers are:
integers naming files, whose epoch AV2 never documents. Reading them as time since the Unix
epoch would date the tbv log to January 1980.

The cost is that scrubbing it is nearly useless -- one step is one nanosecond -- which is why
:func:`default_blueprint` opens on :data:`TIMELINE_ELAPSED` instead of letting the viewer
choose. Switch to this one to read a frame's filename off the timeline, not to play the log.
"""

_PATH_COLOR = (80, 170, 255)
_BODY_COLOR = (235, 235, 235)
_WHEEL_COLOR = (140, 140, 150)
_NOSE_COLOR = (255, 120, 60)

# Lane boundaries are bucketed by the color of the paint on them. LaneMarkType spells the color
# into its own name -- DASH_SOLID_YELLOW, DOUBLE_SOLID_WHITE, SOLID_BLUE -- so a substring scan
# is the classification, and NONE/UNKNOWN fall through to "unmarked", which genuinely means
# there is no paint there and the lane's extent is implied.
_PAINT_COLORS = {
    "yellow": (230, 200, 60),
    "white": (225, 225, 230),
    "blue": (80, 140, 230),
    "unmarked": (90, 90, 110),
}
_CENTERLINE_COLOR = (120, 190, 150)
_CROSSWALK_COLOR = (200, 170, 60)
_DRIVABLE_COLOR = (60, 80, 100)

_LANE_RADIUS_M = 0.08
_CENTERLINE_RADIUS_M = 0.06
_MAP_RADIUS_M = 0.10

# Viridis, as five stops interpolated per channel. Spelled out here rather than imported:
# rerun ships no colormap helper, matplotlib is only a transitive dependency of av2, and
# av2.rendering.color.create_range_map does not do what its name says (it colors by z, rounds to
# integers, and indexes negatively for anything below ground).
_VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
                    dtype=np.float64)
_VIRIDIS_STOPS = np.linspace(0.0, 1.0, len(_VIRIDIS))

# Intensity is uint8 but nothing like uniform over it: measured across a sweep, the median is 30
# and the 99th percentile 108, so scaling by the full 255 leaves the whole cloud in the dark end
# of the ramp. Clipping here puts road surface around blue-teal and retroreflective paint and
# signs up in the yellow, which is the contrast worth having next to the drawn lane boundaries.
_INTENSITY_CLIP = 110.0

# Screen-space, not meters: rerun encodes a ui-point radius as a negative value, and a point
# cloud wants constant apparent size so distant returns thin out rather than disappear.
_LIDAR_RADIUS_UI = 1.0

# How much lidar history the default view shows behind the cursor. One second is ten sweeps,
# roughly 900k points and ~7 m of road at city speed -- enough for the ground to read as a
# surface, little enough that the current sweep is still distinguishable inside it.
_LIDAR_DECAY_S = 1.0

# Axis lengths for the two Transform3D entities that can draw arrows. Both are deliberately
# short: at the scale of the vehicle, an ego frame plus seven camera frames turn the car into a
# pincushion and hide the thing the axes are there to help you look at.
#
# The camera one has no effect on the default layout, which keeps cameras out of the 3D view
# altogether. It is set anyway so that adding one back -- dragging it into a 3D view in the GUI,
# or widening the contents filter -- gets a tick mark rather than whatever the viewer picks.
_EGO_AXIS_M = 0.5
_CAMERA_AXIS_M = 0.25

# What a camera's 2D view draws, beyond its own image. Reaching out to the lidar and the map is
# the entire mechanism behind the overlay: a 2D view anchored at a pinhole projects any 3D entity
# in its contents down through that pinhole, including entities from a completely different
# branch of the tree. Nothing is re-logged and nothing is projected by hand -- `world/lidar` is
# the same city-frame cloud the 3D view draws, seen through `city_SE3_ego @ ego_SE3_cam`.
#
# Which makes this the real extrinsics-and-pose check the frustums never were: returns that do
# not land on the objects they came from, or lane paint that floats off the road, is a
# calibration or pose error you can see without measuring anything.
#
# The path is deliberately excluded -- it runs through the car, so it would smear a bright line
# across the bottom of every frame.
_CAMERA_2D_CONTENTS = ("$origin/**", LIDAR, f"{MAP}/**")

# Reading order for the 2D camera grid: left to right, front to back. Sorting the names instead
# would file the rear cameras between the front and the side ones, which is exactly wrong for a
# layout whose whole job is to look like where the cameras point. Anything absent is skipped and
# anything unrecognized is appended, so this never decides *whether* a camera is drawn.
_CAMERA_GRID_ORDER = (
    "ring_front_left", "ring_front_center", "ring_front_right",
    "ring_side_left", "ring_side_right",
    "ring_rear_left", "ring_rear_right",
    "stereo_front_left", "stereo_front_right",
)

# Columns in that grid. Three makes the first row the forward-facing trio; the seven ring
# cameras then leave one view alone on the last row, which is the price of not being able to
# leave a hole in the middle.
_CAMERA_GRID_COLUMNS = 3

# Where the 3D view's camera starts, in **egovehicle coordinates** -- which is what the view
# renders in, its origin being EGO. So (0, 0, 0) is the vehicle's own origin: the center of the
# rear axle at ground level.
#
# Without this the viewer picks an eye by auto-framing the scene bounding box, and that box is
# dominated by the whole-log map and path -- hundreds of meters of it -- so the opening shot
# centers on the map's centroid with the car an invisible speck somewhere in it. Worse, the
# centroid moves in ego coordinates as the vehicle drives, so the framing is not even stable.
#
# Behind and above, looking slightly down at the axle: the chase view you would pick by hand.
_EYE_POSITION_M = (-14.0, 0.0, 7.0)
_EYE_TARGET_M = (0.0, 0.0, 0.0)

# Nominal Ford Fusion Hybrid dimensions, in meters. AV2 ships no vehicle model, so these are
# stated here rather than read from anywhere -- they are for orientation, not measurement.
#
# The egovehicle frame's origin is the center of the rear axle at ground level, x forward,
# y left, z up. The calibration rig agrees: every sensor sits at x 1.09..1.64, and the lidar at
# (1.35, 0.0, 1.64) is a roof rack a little ahead of the vehicle's middle.
_WHEELBASE_M = 2.85
_REAR_OVERHANG_M = 1.07
_FRONT_OVERHANG_M = 0.95
_HALF_WIDTH_M = 0.925
_WHEEL_RADIUS_M = 0.35
_SILL_M = 0.35
_BELTLINE_M = 1.00
_ROOF_M = 1.47

_NOSE_M = _WHEELBASE_M + _FRONT_OVERHANG_M
_TAIL_M = -_REAR_OVERHANG_M


@dataclasses.dataclass
class SceneSummary:
    """What actually got logged, for the CLI to report."""

    log_id: str
    poses: int = 0
    path_length_m: float = 0.0
    duration_s: float = 0.0
    lane_segments: int = 0
    crosswalks: int = 0
    drivable_areas: int = 0
    lidar_sweeps: int = 0
    lidar_points: int = 0
    cameras: int = 0
    camera_frames: int = 0


def _rect(corners: list[tuple[float, float, float]]) -> np.ndarray:
    """Close a list of corners into a loop."""
    return np.array(corners + [corners[0]], dtype=np.float64)


def _wheel(center_x: float, y: float, *, sides: int = 8) -> np.ndarray:
    """A wheel as a polygon in the vehicle's xz plane, at the axle it belongs to."""
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    xs = center_x + _WHEEL_RADIUS_M * np.cos(angles)
    zs = _WHEEL_RADIUS_M + _WHEEL_RADIUS_M * np.sin(angles)
    loop = np.stack([xs, np.full_like(xs, y), zs], axis=-1)
    return np.vstack([loop, loop[:1]])


def vehicle_wireframe() -> list[np.ndarray]:
    """A crude car outline in the egovehicle frame, as a list of polylines.

    Deliberately not a plain box. A box has no heading, and heading is the one thing you need to
    read off a vehicle when you are checking whether a pose stream is right -- hence the tapered
    cabin and the nose chevron. The wheels are there because they sit on the axles, and the rear
    axle is the frame's origin, which is otherwise invisible.
    """
    half_w, sill, belt, roof = _HALF_WIDTH_M, _SILL_M, _BELTLINE_M, _ROOF_M

    # Body: a sill loop and a beltline loop, joined at the corners.
    body_low = _rect([(_TAIL_M, -half_w, sill), (_NOSE_M, -half_w, sill),
                      (_NOSE_M, half_w, sill), (_TAIL_M, half_w, sill)])
    body_high = _rect([(_TAIL_M, -half_w, belt), (_NOSE_M, -half_w, belt),
                       (_NOSE_M, half_w, belt), (_TAIL_M, half_w, belt)])
    posts = [
        np.array([(x, y, sill), (x, y, belt)], dtype=np.float64)
        for x in (_TAIL_M, _NOSE_M)
        for y in (-half_w, half_w)
    ]

    # Cabin: inset from the body on all sides, which is what makes the front readable.
    cabin_back, cabin_front, cabin_half_w = 0.35, 2.55, half_w - 0.12
    roof_back, roof_front = 0.75, 2.15
    cabin_base = _rect([(cabin_back, -cabin_half_w, belt), (cabin_front, -cabin_half_w, belt),
                        (cabin_front, cabin_half_w, belt), (cabin_back, cabin_half_w, belt)])
    roof_loop = _rect([(roof_back, -cabin_half_w + 0.1, roof),
                       (roof_front, -cabin_half_w + 0.1, roof),
                       (roof_front, cabin_half_w - 0.1, roof),
                       (roof_back, cabin_half_w - 0.1, roof)])
    pillars = [
        np.array([(bx, by, belt), (rx, ry, roof)], dtype=np.float64)
        for (bx, by), (rx, ry) in (
            ((cabin_back, -cabin_half_w), (roof_back, -cabin_half_w + 0.1)),
            ((cabin_front, -cabin_half_w), (roof_front, -cabin_half_w + 0.1)),
            ((cabin_front, cabin_half_w), (roof_front, cabin_half_w - 0.1)),
            ((cabin_back, cabin_half_w), (roof_back, cabin_half_w - 0.1)),
        )
    ]

    return [body_low, body_high, *posts, cabin_base, roof_loop, *pillars]


def wheel_outlines() -> list[np.ndarray]:
    """The four wheels, on the axles. The rear pair straddles the frame origin."""
    return [
        _wheel(center_x, y)
        for center_x in (0.0, _WHEELBASE_M)
        for y in (-_HALF_WIDTH_M, _HALF_WIDTH_M)
    ]


def nose_chevron() -> np.ndarray:
    """A forward-pointing V at the front of the vehicle, so heading is unambiguous."""
    tip = _NOSE_M + 0.45
    return np.array(
        [(_NOSE_M, -_HALF_WIDTH_M * 0.6, _BELTLINE_M),
         (tip, 0.0, _BELTLINE_M),
         (_NOSE_M, _HALF_WIDTH_M * 0.6, _BELTLINE_M)],
        dtype=np.float64,
    )


def log_vehicle() -> None:
    """Log the vehicle outline once, as static data under the ego transform.

    ``static=True`` means it has no timestamp and is therefore valid at every point on the
    timeline. It follows the vehicle because ``world/ego`` moves, not because anything re-logs
    it -- the single most useful habit to pick up from rerun's data model.
    """
    rr.log(f"{WIREFRAME}/body", rr.LineStrips3D(vehicle_wireframe(), colors=_BODY_COLOR,
                                                radii=0.02), static=True)
    rr.log(f"{WIREFRAME}/wheels", rr.LineStrips3D(wheel_outlines(), colors=_WHEEL_COLOR,
                                                  radii=0.03), static=True)
    rr.log(f"{WIREFRAME}/nose", rr.LineStrips3D([nose_chevron()], colors=_NOSE_COLOR,
                                                radii=0.04), static=True)


def log_coordinate_frames() -> None:
    """Declare the city frame's handedness.

    Without ``ViewCoordinates`` the viewer has no idea which way is up and starts the camera in
    an arbitrary orientation; AV2's city frame is right-handed with +z up.
    """
    rr.log(WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def paint_bucket(mark_type) -> str:
    """Which color bucket a ``LaneMarkType`` belongs to.

    ``SOLID_BLUE`` gets its own bucket rather than falling in with the unmarked boundaries: it is
    real paint, and "unmarked" is the one label that asserts there is none.
    """
    name = str(getattr(mark_type, "value", mark_type)).upper()
    for paint in ("yellow", "white", "blue"):
        if paint.upper() in name:
            return paint
    return "unmarked"


def log_map(source: av2_source.LogSource) -> tuple[int, int, int]:
    """Log the log's vector HD map, static, in city coordinates.

    Everything here is ``static=True``: the map does not change over a log, so it carries no
    timestamps and is valid wherever you scrub to.

    One batched ``LineStrips3D`` per bucket rather than one entity per lane segment. A hundred
    entities would be slower to draw and would bury the entity tree under ids nobody can read;
    N strips in one archetype is one entity with N instances.

    Returns:
        counts of (lane segments, crosswalks, drivable areas) logged.
    """
    static_map = source.static_map()

    # Left and right boundaries are classified independently -- a lane commonly has a yellow
    # centerline on one side and a white edge line on the other.
    by_paint: dict[str, list[np.ndarray]] = {paint: [] for paint in _PAINT_COLORS}
    for segment in static_map.vector_lane_segments.values():
        for boundary, mark_type in (
            (segment.left_lane_boundary, segment.left_mark_type),
            (segment.right_lane_boundary, segment.right_mark_type),
        ):
            by_paint[paint_bucket(mark_type)].append(boundary.xyz)

    for paint, strips in by_paint.items():
        if not strips:
            continue
        rr.log(f"{LANE_BOUNDARIES}/{paint}",
               rr.LineStrips3D(strips, colors=_PAINT_COLORS[paint], radii=_LANE_RADIUS_M),
               static=True)

    # Centerlines are DERIVED, not annotated: AV2 stores a lane as a ladder of two boundaries and
    # has no centerline field. get_lane_segment_centerline resamples both boundaries to
    # NUM_CENTERLINE_INTERP_PTS (10) and averages them pairwise, so what is drawn here is a
    # 10-point approximation of the middle -- fine to look at, not something to regress against.
    centerlines = [
        static_map.get_lane_segment_centerline(segment_id)
        for segment_id in static_map.vector_lane_segments
    ]
    if centerlines:
        rr.log(CENTERLINES,
               rr.LineStrips3D(centerlines, colors=_CENTERLINE_COLOR,
                               radii=_CENTERLINE_RADIUS_M),
               static=True)

    # Both of these arrive already closed: PedestrianCrossing.polygon repeats its first vertex,
    # and DrivableArea.from_dict appends the first point to the boundary as it parses.
    crossings = [crossing.polygon for crossing in static_map.vector_pedestrian_crossings.values()]
    if crossings:
        rr.log(CROSSWALKS,
               rr.LineStrips3D(crossings, colors=_CROSSWALK_COLOR, radii=_MAP_RADIUS_M),
               static=True)

    areas = [area.xyz for area in static_map.vector_drivable_areas.values()]
    if areas:
        rr.log(DRIVABLE_AREAS,
               rr.LineStrips3D(areas, colors=_DRIVABLE_COLOR, radii=_MAP_RADIUS_M),
               static=True)

    return len(static_map.vector_lane_segments), len(crossings), len(areas)


def log_ego_path(source: av2_source.LogSource) -> SceneSummary:
    """Log the ego pose over time, plus the whole drive as one static polyline.

    The path is static rather than grown frame by frame: it is context for wherever you scrub
    to, and one polyline of N vertices is far cheaper than N incrementally longer ones.
    """
    summary = SceneSummary(log_id=source.log_id)
    poses = source.city_SE3_ego()
    if not poses:
        raise av2_source.MissingStreamError(f"{source.log_id} has an empty pose stream")

    timestamps = sorted(poses)
    t0_ns = timestamps[0]

    # How long the pose's axis arrows are drawn is a property of the *visualization*, not of any
    # one pose, so it is logged once and statically rather than restated 8801 times alongside
    # the transforms it decorates.
    rr.log(EGO, rr.TransformAxes3D(_EGO_AXIS_M), static=True)

    translations = []
    for timestamp_ns in timestamps:
        pose = poses[timestamp_ns]
        rr.set_time(TIMELINE_ELAPSED, duration=(timestamp_ns - t0_ns) / 1e9)
        rr.set_time(TIMELINE_TIMESTAMP, sequence=timestamp_ns)
        rr.log(EGO, rr.Transform3D(translation=pose.translation, mat3x3=pose.rotation))
        translations.append(pose.translation)

    track = np.asarray(translations)
    rr.log(PATH, rr.LineStrips3D([track], colors=_PATH_COLOR, radii=0.25), static=True)

    summary.poses = len(timestamps)
    summary.path_length_m = float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())
    summary.duration_s = (timestamps[-1] - t0_ns) / 1e9
    return summary


def intensity_colors(intensity: np.ndarray) -> np.ndarray:
    """Map lidar intensity onto viridis, clipped at :data:`_INTENSITY_CLIP`.

    Args:
        intensity: (N,) uint8 returns.

    Returns:
        (N,3) uint8 RGB.
    """
    fraction = np.clip(np.asarray(intensity, dtype=np.float64) / _INTENSITY_CLIP, 0.0, 1.0)
    channels = [np.interp(fraction, _VIRIDIS_STOPS, _VIRIDIS[:, c]) for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def log_lidar(source: av2_source.LogSource) -> tuple[int, int]:
    """Log every lidar sweep, in city coordinates, on its own timestamps.

    **The points are transformed into the city frame before being logged**, which is a deliberate
    departure from the usual rerun idiom of logging at the sensor and letting the transform
    hierarchy compose. AV2 hands the sweep over already in the ego frame, so ``world/ego/lidar``
    would have been the free option. Two reasons not to take it:

    * A decay window -- ``VisibleTimeRanges``, which is what :func:`default_blueprint` sets and
      what makes a trail behind the vehicle possible at all -- only draws a *correct* trail in a
      frame that does not move. The viewer resolves an entity's transform chain once, at the
      cursor time, while the visible-time-range query is per-visualizer; so ten sweeps under a
      moving ``world/ego`` would all be drawn at the *current* pose, piling onto the car instead
      of laying down road behind it.
    * The smear is the diagnostic. Accumulated returns doubling a wall or blurring the ground is
      the most direct read on pose quality this viewer can offer, and it only shows up in a
      fixed frame.

    Precision is not a concern: rerun stores positions as float32, whose ulp at a city-frame
    coordinate of ~6700 m is 0.8 mm. The returns themselves are float16 on disk, some 80x
    coarser than that.

    Returns:
        counts of (sweeps, points) logged.
    """
    poses = source.city_SE3_ego()
    if not poses:
        raise av2_source.MissingStreamError(f"{source.log_id} has an empty pose stream")
    # The first *pose*, matching log_ego_path -- not the first sweep, which trails it by ~0.1 s
    # and would put the two streams on offset elapsed timelines.
    t0_ns = min(poses)

    sweeps = points = unposed = 0
    for sweep in source.lidar_sweeps():
        pose = poses.get(sweep.timestamp_ns)
        if pose is None:
            # Every sweep timestamp is a pose timestamp in the logs on hand (157/157), the pose
            # stream being ~17x denser. Skip rather than KeyError if that ever fails to hold:
            # one unplaceable sweep should not cost the other 156.
            unposed += 1
            continue

        rr.set_time(TIMELINE_ELAPSED, duration=(sweep.timestamp_ns - t0_ns) / 1e9)
        rr.set_time(TIMELINE_TIMESTAMP, sequence=sweep.timestamp_ns)
        rr.log(LIDAR, rr.Points3D(
            pose.transform_point_cloud(sweep.xyz).astype(np.float32),
            colors=intensity_colors(sweep.intensity),
            radii=rr.Radius.ui_points(_LIDAR_RADIUS_UI),
        ))
        sweeps += 1
        points += len(sweep.xyz)

    if unposed:
        logging.warning("%s: skipped %d sweep(s) with no pose at their timestamp",
                        source.log_id, unposed)
    return sweeps, points


def log_cameras(source: av2_source.LogSource) -> tuple[int, int]:
    """Log every camera on disk: its calibration once, then one frame per timestamp.

    **These stay in the sensor frame**, which is the opposite of what :func:`log_lidar` does with
    data that arrived in the same egovehicle frame. The reason the lidar had to leave is that a
    decay window only trails correctly in a frame that does not move; an image has no trail to
    leave, so there is nothing to trade away, and staying puts the whole placement in the
    transform tree where the viewer can show it: pick a camera in the Selection panel and the
    composition ``city_SE3_ego @ ego_SE3_cam`` is right there.

    Nothing here reads a pose. The images are placed by their *parent*, and a camera timestamp
    that had no pose would simply inherit the nearest earlier one instead of being dropped --
    which is moot in practice, since all 8048 camera timestamps across the tbv log's seven
    cameras are exact pose timestamps.

    The jpegs are logged **by path and never decoded**: ``EncodedImage`` stores the compressed
    bytes and the viewer decompresses only the frames it draws. That is what keeps 611 MB of
    imagery to a flat 178 MB of logger memory, and it is why cameras cost less to log than the
    lidar does despite being four times the bytes on disk.

    Returns:
        counts of (cameras, frames) logged.
    """
    poses = source.city_SE3_ego()
    if not poses:
        raise av2_source.MissingStreamError(f"{source.log_id} has an empty pose stream")
    # The first pose, matching log_ego_path and log_lidar, so every stream shares one origin on
    # the elapsed timeline.
    t0_ns = min(poses)

    cameras = frames = 0
    for item in source.cameras():
        camera = source.camera_model(item)
        entity = f"{CAMERAS}/{item.token}"

        # Both static: the rig does not move relative to the vehicle over a log. Transform3D is
        # the parent->entity transform and Pinhole is the projection at the entity, and rerun is
        # happy to take them in one call on the same path.
        rr.log(entity,
               rr.Transform3D(translation=camera.ego_SE3_cam.translation,
                              mat3x3=camera.ego_SE3_cam.rotation),
               rr.TransformAxes3D(_CAMERA_AXIS_M),
               # Logged even though the default view draws no frustum from it. Pinhole is what
               # makes this entity a *camera* rather than a place images are filed: it gives the
               # 2D view its projection, and it is what any later 3D-into-image overlay would
               # need. Which of its visualizers run is a view decision, made in
               # default_blueprint, and not this function's business.
               rr.Pinhole(image_from_camera=camera.intrinsics.K,
                          resolution=[camera.width_px, camera.height_px],
                          # AV2's camera frame is x right, y down, z forward -- readable
                          # straight off ego_SE3_cam, whose rotation columns are ego -y, -z, +x.
                          # That is also rerun's default, but an unset camera_xyz logs no
                          # component at all, so the orientation would quietly follow whatever
                          # the viewer defaults to next.
                          camera_xyz=rr.ViewCoordinates.RDF),
               static=True)

        for timestamp_ns, frame_path in source.camera_frames(item):
            rr.set_time(TIMELINE_ELAPSED, duration=(timestamp_ns - t0_ns) / 1e9)
            rr.set_time(TIMELINE_TIMESTAMP, sequence=timestamp_ns)
            rr.log(entity, rr.EncodedImage(path=frame_path))
            frames += 1
        cameras += 1

    return cameras, frames


def default_blueprint(source: av2_source.LogSource) -> rrb.Blueprint:
    """A 3D view anchored to the vehicle, beside a grid of the log's camera images.

    A view's ``origin`` is the frame it renders in, and that is the whole of "follow the
    vehicle" -- rerun applies the inverse ego transform to everything else, so the car holds
    still and the city sweeps past it. There is no separate follow toggle.

    The eye is placed explicitly, in those same egovehicle coordinates -- see
    :data:`_EYE_POSITION_M`. Left to itself the viewer frames the scene bounding box, which the
    whole-log map dwarfs.

    ``contents`` has to be widened at the same time, and forgetting is the trap. It defaults to
    ``$origin/**``, which under ``world/ego`` resolves to the vehicle and its cameras alone: PATH
    is a *sibling* of EGO, not a descendant. That placement is deliberate -- the path is in city
    coordinates and must not inherit the ego transform, or it would drag along with the car
    instead of staying fixed to the map -- so the view has to reach outside its own origin to
    show it.

    ``/**`` minus the cameras, so the streams that land under ``world/`` later (annotations)
    appear without anyone editing this, while the camera frustums stay out of the way.

    The lidar gets a **decay window**: instead of the newest sweep alone, the view shows every
    sweep from one second back up to the cursor, which is what turns a ring of returns into a
    stretch of road. It is scoped to the lidar entity rather than passed as the view's own
    ``time_ranges``, because a view-level range would also range-query ``world/ego`` -- whose
    transform draws axes, and would draw ten of them. It is also just a starting point:
    **Visible time range** in the Selection panel edits it.

    **The cameras are not in the 3D view at all**, and each 2D view instead reaches back out to
    the lidar and the map, which the pinhole projects into the image -- see
    :data:`_CAMERA_2D_CONTENTS`. The world goes into the cameras rather than the cameras into
    the world, which is both less cluttered and a far better check on the calibration.

    The camera grid is built from what is **on disk**, so it needs the source. That costs
    nothing in ordering: deciding which cameras exist is a directory check on an already-built
    :class:`~av2_source.LogSource`, not a read of the imagery, so this still runs before the
    sink is chosen and before anything is logged. A log with no cameras -- every ``sensor/val``
    log on hand, and the whole ``lidar`` dataset -- gets the bare 3D view rather than an empty
    pane.

    Args:
        source: the log about to be logged, consulted only for which cameras are present.
    """
    decay = rrb.VisibleTimeRanges([
        rrb.VisibleTimeRange(
            TIMELINE_ELAPSED,
            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-_LIDAR_DECAY_S),
            end=rrb.TimeRangeBoundary.cursor_relative(),
        )
    ])
    # Open on `elapsed`, and say so rather than hoping. The viewer otherwise picks
    # `timestamp_ns`, which is a *sequence* timeline of raw AV2 nanoseconds -- 18-digit tick
    # labels, and playback that steps one nanosecond at a time, so pressing play advances the
    # scene by 30 ns per second and the recording looks frozen. `elapsed` is a duration
    # timeline, so it scrubs and plays in real seconds.
    time_panel = rrb.TimePanel(timeline=TIMELINE_ELAPSED)

    present = [item.token for item in source.cameras()]
    ordered = sorted(present, key=lambda name: (_CAMERA_GRID_ORDER.index(name)
                                                if name in _CAMERA_GRID_ORDER
                                                else len(_CAMERA_GRID_ORDER)))

    # Cameras are excluded rather than merely shrunk: a frustum is a wireframe pyramid with the
    # picture hanging off the end of it, and seven of them bury the scene they sit in. The
    # exclusion has to name both the subtree and the entity itself -- `- .../**` does not cover
    # the entity the images and the Pinhole are actually logged at.
    #
    # Dropping them here costs nothing, because everything the frustum was for now happens the
    # other way round: instead of putting cameras in the world, the 2D views put the world in
    # the cameras. See _CAMERA_2D_CONTENTS.
    scene = rrb.Spatial3DView(
        origin=EGO, name="follow vehicle", overrides={LIDAR: decay},
        contents=["/**", f"- {CAMERAS}/**"] + [f"- {CAMERAS}/{name}" for name in ordered],
        # Orbital rather than FirstPerson so that dragging pivots around the vehicle, which is
        # what you want when the question is "where is the car relative to the map".
        eye_controls=rrb.EyeControls3D(kind="Orbital",
                                       position=_EYE_POSITION_M,
                                       look_target=_EYE_TARGET_M,
                                       eye_up=(0.0, 0.0, 1.0)),
    )
    if not present:
        return rrb.Blueprint(scene, time_panel)

    grid = rrb.Grid(
        contents=[rrb.Spatial2DView(origin=f"{CAMERAS}/{name}", name=name,
                                    contents=list(_CAMERA_2D_CONTENTS))
                  for name in ordered],
        grid_columns=_CAMERA_GRID_COLUMNS,
    )
    # The 3D view is the one you scrub in and the one that needs room; the grid is for glancing
    # at. Two thirds to one third rather than an even split.
    return rrb.Blueprint(rrb.Horizontal(scene, grid, column_shares=[2, 1]), time_panel)


def log_scene(source: av2_source.LogSource) -> SceneSummary:
    """Log the coordinate frame, the map, the vehicle, its path, the lidar, and the cameras."""
    log_coordinate_frames()
    log_vehicle()
    summary = log_ego_path(source)

    # Each optional stream gets its own guard. They are downloaded independently -- two of the
    # three sensor/val logs on hand have no sensors/ directory at all -- so a log missing one is
    # ordinary, and the layers that did arrive are still worth looking at. An empty *pose* stream
    # stays fatal, since nothing can be placed without it, and log_ego_path raises for that
    # above.
    try:
        summary.lane_segments, summary.crosswalks, summary.drivable_areas = log_map(source)
    except av2_source.MissingStreamError as error:
        logging.warning("drawing %s without a map: %s", source.log_id, error)

    try:
        summary.lidar_sweeps, summary.lidar_points = log_lidar(source)
    except av2_source.MissingStreamError as error:
        logging.warning("drawing %s without lidar: %s", source.log_id, error)

    # No warning when a log simply has no cameras -- unlike the map and the lidar, which every
    # readable log ships, imagery is genuinely optional and the lidar dataset has none by
    # definition. log_cameras returns (0, 0) for that; the error it can still raise is a log
    # that has cameras but no calibration to place them with.
    try:
        summary.cameras, summary.camera_frames = log_cameras(source)
    except av2_source.MissingStreamError as error:
        logging.warning("drawing %s without cameras: %s", source.log_id, error)

    return summary
