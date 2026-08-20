"""Absolute mount-offset check from the sun's position in the panorama.

Every other estimate of `mount_offset_deg` is *relative*.
`mount_offset_sweep` finds the offset that makes rays to unknown objects
intersect, which fixes the angle only up to whatever the poses and the heading
model get wrong together. `bow_calibration` measures the bow, which is not the
direction of travel. The operator's prior is a look at one frame.

The sun is different: at a known time and place its azimuth is known to a
hundredth of a degree from ephemeris alone. Find it in the panorama and the
offset falls out of one subtraction,

    bearing_world = (course + bearing_camera - offset) mod 360
    => offset = course + az_camera_sun - az_world_sun

with **no map, no tracks, and no tracklet geometry** involved. That makes it the
only check here that can catch an error the sweep and the poses share -- most
importantly a 180 deg convention slip, which a relative method reproduces
happily.

Why this is not circular: the offset is a *yaw* rotation, so it cannot change
the sun's **elevation**. The ephemeris elevation is therefore free to gate the
search -- it says which bright blob is the sun rather than sun-glitter on the
water, cloud, or a lamp -- while the ephemeris **azimuth** is held back as the
quantity being compared. Blobs are accepted on elevation and scored on azimuth.

What makes it fail, and how that shows up:

- **The vehicle's own sunlit structure.** This is the one that bites hardest.
  A sunlit white sail is as bright as the sun and at the same elevation, so
  neither brightness nor the elevation gate rejects it. It gave charles_river a
  blob at camera azimuth 44-56 deg across courses of 112, 138, 142 and 205 deg
  -- and a blob that holds still in the *camera* frame while the vehicle turns
  cannot be the sun. Two defences, both cheap: a **rig mask** built from the
  temporal median over the sampled frames (structure bolted to the vehicle is
  the only thing bright in every frame, since the sun sweeps the camera frame as
  the vehicle turns), and a **fixed-object model test** that fits `az_camera =
  const` alongside the sun model and reports which explains the frames better.
- **Overcast.** A blown-out sky puts the brightest blob on zenith cloud. The
  elevation gate rejects most of those, and what survives scatters; the reported
  circular concentration R collapses. mount_washington's three legs are the
  worked example -- measured elevations 38-90 deg against a true 31-44, R=0.65.
- **Pitch and roll.** A pitching boat tilts the sun's apparent azimuth by a few
  degrees. It averages out over frames but sets the floor on precision; treat a
  spread of a few degrees as normal and do not read the mean past ~2 deg.
- **Sun behind the mast.** Occluded frames simply find no blob in the band.

Read `R` first. Above ~0.95 the frames agree and the mean is a real measurement;
below ~0.8 the tool found weather, not the sun, and says nothing either way.

    bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:sun_offset_check -- \\
        --run_dir <runs>/r001_v4 --n_frames 40
"""

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    heading as heading_mod,
    pano_geometry as pg,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

# Half-width of the elevation band searched around the ephemeris elevation.
#
# This is the most important number in the module and it wants to be TIGHT. The
# first version used 12 deg, budgeting for boat pitch; measurement says the sun's
# elevation is recovered to 0.5 deg on charles_river, and the slack was spent
# admitting impostors instead. Two live at the same elevation as the sun and are
# just as bright: the vehicle's sunlit structure, and the antipodal ghost that a
# dual-fisheye stitch throws roughly 180 deg opposite a bright source.
#
# Tightening the band on charles_river walks straight down that list --
#
#   tolerance   median |elev err|   offset    R_sun   verdict
#      12.0            5.4          306.7     0.165   FIXED-OBJECT
#       6.0            2.7          282.7     0.360   FIXED-OBJECT
#       4.0            1.4          275.3     0.661   SCATTERED
#       2.5            0.5          272.4     0.965   AGREEING
#
# -- and R rising monotonically as a *yaw-invariant* gate tightens is the
# signature of closing in on the real sun rather than of overfitting: the gate
# cannot see the quantity being estimated. Widen this only for a platform with
# genuine pitch, and expect the ghost back when you do.
ELEVATION_TOLERANCE_DEG = 3.0

# A blob must be this bright relative to the frame's own dynamic range. The sun
# in a clear sky saturates; a threshold on the frame's max rather than an
# absolute 255 keeps a slightly under-exposed frame usable.
BRIGHT_FRACTION = 0.97

# A compact source subtends a fraction of a degree; allowing this much covers
# bloom and a coarse threshold. Anything wider is not a sun -- an overcast band
# thresholds into one enormous run -- and is rejected rather than averaged, which
# is what stops a white sky from reporting a confident azimuth.
MAX_BLOB_WIDTH_DEG = 25.0

# One shared definition. Restating it is what produced the pohang 180 deg
# slip; see docs/conventions.md.
CONVENTION = pg.MOUNT_OFFSET_CONVENTION

# Circular concentration above which the per-frame estimates are calling the
# same angle rather than scattering. Below the lower bound the check abstains.
R_TRUSTWORTHY = 0.95
R_USELESS = 0.80


def solar_position(when: datetime, lat_deg: float, lon_deg: float):
    """(azimuth_deg_cw_from_north, elevation_deg) of the sun. Low-precision
    almanac formulae -- about 0.01 deg, far inside what a panorama can resolve.
    """
    # Days from the J2000.0 epoch. `when` must be timezone-aware UTC.
    delta = when - datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
    n = delta.total_seconds() / 86400.0

    mean_longitude = math.radians((280.460 + 0.9856474 * n) % 360.0)
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_longitude = mean_longitude + math.radians(
        1.915 * math.sin(mean_anomaly) + 0.020 * math.sin(2 * mean_anomaly))
    obliquity = math.radians(23.439 - 0.0000004 * n)

    right_ascension = math.atan2(
        math.cos(obliquity) * math.sin(ecliptic_longitude),
        math.cos(ecliptic_longitude))
    declination = math.asin(
        math.sin(obliquity) * math.sin(ecliptic_longitude))

    greenwich_sidereal = (18.697374558 + 24.06570982441908 * n) % 24.0
    local_sidereal_deg = (greenwich_sidereal * 15.0 + lon_deg) % 360.0
    hour_angle = math.radians(
        (local_sidereal_deg - math.degrees(right_ascension)) % 360.0)

    lat = math.radians(lat_deg)
    east = -math.cos(declination) * math.sin(hour_angle)
    north = (math.sin(declination) * math.cos(lat)
             - math.cos(declination) * math.sin(lat) * math.cos(hour_angle))
    up = (math.sin(declination) * math.sin(lat)
          + math.cos(declination) * math.cos(lat) * math.cos(hour_angle))
    return math.degrees(math.atan2(east, north)) % 360.0, math.degrees(
        math.asin(max(-1.0, min(1.0, up))))


def brightest_blob_in_band(pano: np.ndarray, elevation_deg: float,
                           tolerance_deg: float = ELEVATION_TOLERANCE_DEG,
                           bright_fraction: float = BRIGHT_FRACTION,
                           max_width_deg: float = MAX_BLOB_WIDTH_DEG,
                           mask: np.ndarray | None = None):
    """(az_camera_deg, elevation_deg, n_pixels) of the brightest compact blob
    whose elevation is consistent with `elevation_deg`, or None.

    Only rows inside the elevation band are searched, which is what keeps the
    water's sun-glitter -- as bright as the sun and always below the horizon --
    from being mistaken for it.

    The blob is the **connected run of bright columns** containing the brightest
    column, not a fixed window around the brightest pixel. The sun saturates, so
    its peak is a plateau and `argmax` returns the plateau's left edge; a window
    centred there is centred on the edge of the blob and reads about half a blob
    radius low. A run has no such handedness.
    """
    height, width = pano.shape[:2]
    grey = pano.astype(np.float32).mean(axis=2) if pano.ndim == 3 else \
        pano.astype(np.float32)

    hi = pg.pano_px_from_direction(0.0, elevation_deg + tolerance_deg,
                                   width, height)[1]
    lo = pg.pano_px_from_direction(0.0, elevation_deg - tolerance_deg,
                                   width, height)[1]
    row_lo, row_hi = int(math.floor(min(hi, lo))), int(math.ceil(max(hi, lo)))
    row_lo, row_hi = max(0, row_lo), min(height, row_hi + 1)
    if row_hi - row_lo < 2:
        return None

    band = grey[row_lo:row_hi]
    bright = band >= band.max() * bright_fraction
    if mask is not None:
        # Masked pixels are the vehicle's own structure. Zeroing them after the
        # threshold, not before, keeps the threshold defined by the frame's true
        # dynamic range rather than by whatever is left over.
        bright = bright & ~mask[row_lo:row_hi]
    if not bright.any():
        return None

    # Seed on the brightest *column* rather than the brightest pixel: a column
    # sum is not decided by one saturated pixel among equals.
    column_bright = bright.any(axis=0)
    seed = int(np.argmax(np.where(column_bright, band.sum(axis=0), -np.inf)))

    # Walk out from the seed while columns stay bright, wrapping at the seam.
    left = seed
    while column_bright[(left - 1) % width] and (seed - left) % width < width - 1:
        left = (left - 1) % width
    right = seed
    while column_bright[(right + 1) % width] and (right - seed) % width < width - 1:
        right = (right + 1) % width
    run_width = (right - left) % width + 1
    if run_width * 360.0 / width > max_width_deg:
        return None

    columns = [(left + i) % width for i in range(run_width)]
    sub = bright[:, columns]
    rows_idx, cols_idx = np.nonzero(sub)
    if rows_idx.size == 0:
        return None
    weights = band[rows_idx, np.asarray(columns)[cols_idx]].astype(np.float64)

    # Circular mean over columns so a blob straddling the seam does not average
    # to the opposite side of the panorama.
    angles = 2 * math.pi * np.asarray(columns)[cols_idx] / width
    mean_col = (math.atan2(float((weights * np.sin(angles)).sum()),
                           float((weights * np.cos(angles)).sum()))
                / (2 * math.pi) * width) % width
    mean_row = float((weights * (rows_idx + row_lo)).sum() / weights.sum())
    az, el = pg.direction_from_pano_px(mean_col, mean_row, width, height)
    return az, el, int(weights.size)


def rig_mask(greys, bright_fraction: float = BRIGHT_FRACTION):
    """Pixels that are bright in the *median* frame: the vehicle's own structure.

    The sun sweeps the camera frame whenever the vehicle turns, so it is bright
    in a few frames and dark in the rest -- a median over frames drops it. A mast,
    boom, or sunlit sail is bolted to the camera and is bright in every frame, so
    it survives. Masking what survives is therefore a rig mask, and it needs no
    knowledge of the rig.

    Needs real course variation to work: on a dead-straight run the sun does not
    move in the camera frame either, and the median cannot tell them apart. The
    caller checks the course spread and says so.
    """
    if len(greys) < 3:
        return np.zeros(greys[0].shape, dtype=bool)
    median = np.median(np.stack(greys), axis=0)
    return median >= median.max() * bright_fraction


def fixed_object_concentration(rows):
    """(R_sun, R_fixed) -- how well each model explains the blobs found.

    `R_sun` is the agreement of the recovered offsets, which is the sun model.
    `R_fixed` is the agreement of the raw camera azimuths, which is the model
    "some bright thing is bolted to the camera". Whichever is larger names what
    was actually being tracked, and the comparison is only meaningful when the
    course varied -- otherwise the two models are the same model.
    """
    _, r_sun = circular_stats([r["offset_deg"] for r in rows])
    _, r_fixed = circular_stats([r["sun_az_camera_deg"] for r in rows])
    return r_sun, r_fixed


def circular_stats(angles_deg):
    """(mean_deg, R) -- R is 1 for perfect agreement, 0 for uniform scatter."""
    if not angles_deg:
        return None, 0.0
    radians = [math.radians(a) for a in angles_deg]
    east = sum(math.sin(r) for r in radians) / len(radians)
    north = sum(math.cos(r) for r in radians) / len(radians)
    return math.degrees(math.atan2(east, north)) % 360.0, math.hypot(east, north)


def write_metadata(paths, record, *, supersede_validated=False):
    """Publish a sun result into the dataset's `pipeline_metadata.json`.

    The sun check writes and the sweep corroborates, not the other way round.
    The sweep is relative -- it can only find the offset that makes rays agree
    with each other, so it reproduces any error the poses and the heading model
    share, including a 180 deg convention slip. The sun is absolute. Checked
    against boston_harbor_leg1's independently surveyed 214.0 deg it returns
    215.0, so `accuracy_validated` is earned rather than assumed -- but only for
    an AGREEING verdict, since a scattered one measured weather.

    Refuses to replace an existing `accuracy_validated` value, for the same
    reason `mount_offset_sweep` does: artifacts already built under it need it to
    stay put, and a 1 deg improvement is not worth desynchronising them.
    """
    meta_path = paths.metadata_path
    meta = json.loads(meta_path.read_text())
    previous = meta.get("mount_offset") or {}

    if previous.get("accuracy_validated") and not supersede_validated:
        old = previous.get("mount_offset_deg")
        print(f"\nREFUSING to overwrite {meta_path}")
        print(f"  existing:   {old} deg, accuracy_validated=true "
              f"({previous.get('source', 'unrecorded')})")
        print(f"  this check: {record['offset_deg']} deg (sun, "
              f"R={record['concentration_R']})")
        if old is not None:
            delta = (record["offset_deg"] - old + 180.0) % 360.0 - 180.0
            print(f"  they differ by {abs(delta):.1f} deg -- corroboration, "
                  f"not a reason to rewrite. Pass --supersede_validated to "
                  f"force.")
        return False

    block = {
        "mount_offset_deg": round(record["offset_deg"], 1),
        "status": "sun_verified" if record["usable"] else record["verdict"].lower(),
        "source": (f"sun azimuth vs ephemeris over "
                   f"{record['n_frames_used']} keyframes "
                   f"(sun_offset_check.py)"),
        "method": ("offset = course + az_camera_sun - az_ephemeris_sun; no map, "
                   "no tracks, no operator judgement"),
        "concentration_R": record["concentration_R"],
        "median_abs_elevation_error_deg": record[
            "median_abs_elevation_error_deg"],
        "accuracy_validated": bool(record["usable"]),
        "accuracy_note": ("The method returns 215.0 deg on boston_harbor_leg1, "
                          "whose 214.0 deg was established independently "
                          "against a surveyed building over 72 keyframes."),
        "convention": CONVENTION,
    }
    if not record["usable"]:
        block["rejected_offset_deg"] = block.pop("mount_offset_deg")
    if previous:
        block["superseded"] = previous
    meta["mount_offset"] = block
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote mount_offset block to {meta_path}")
    if previous:
        print(f"  previous value ({previous.get('mount_offset_deg')} deg, "
              f"{previous.get('status')}) kept under mount_offset.superseded")
    print("NOTE: pipeline_metadata.json is listed in the dataset's "
          "checksums.sha256 - regenerate it.")
    return True


def log_start_utc(metadata: dict):
    """Absolute clock for the recording, or None with the reason printed."""
    raw = metadata.get("log_start_utc")
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
        timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="Only used to infer the dataset")
    parser.add_argument("--n_frames", type=int, default=40)
    parser.add_argument("--min_speed_mps", type=float, default=0.8,
                        help="Course is meaningless when nearly stopped, and "
                             "the offset is measured against course")
    parser.add_argument("--elevation_tolerance_deg", type=float,
                        default=ELEVATION_TOLERANCE_DEG)
    parser.add_argument("--work_width", type=int, default=1440,
                        help="Panoramas are searched at this width; 1440 is "
                             "0.25 deg/px, finer than the blob centroid")
    parser.add_argument("--mask_rig", action="store_true", default=True,
                        help="Mask structure that is bright in the temporal "
                             "median, i.e. bolted to the camera (default on)")
    parser.add_argument("--no_mask_rig", dest="mask_rig",
                        action="store_false")
    parser.add_argument("--scan_tolerance", action="store_true",
                        help="Report the whole tolerance-vs-R table instead of "
                             "one verdict. That table is the evidence: R must "
                             "rise as the band tightens if the blob is the sun")
    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument("--write_metadata", action="store_true",
                        help="record the result in the dataset's "
                             "pipeline_metadata.json")
    parser.add_argument("--supersede_validated", action="store_true",
                        help="allow --write_metadata to replace an offset "
                             "already marked accuracy_validated")
    args = parser.parse_args()
    paths = farfield_paths.resolve(parser, args, infer_from=args.run_dir,
                                   require=("dataset_base", "panorama_dir"))

    metadata = json.loads(paths.metadata_path.read_text())
    start = log_start_utc(metadata)
    if start is None:
        raise SystemExit(
            f"{paths.metadata_path} has no `log_start_utc`, so the sun's "
            f"position cannot be computed. Without an absolute clock this "
            f"check is impossible; the sweep remains the only estimate.")
    print(f"log starts {start.isoformat()}")

    result = ingest.run_ingest(paths.dataset_base, paths.frame_landmarks,
                              IngestConfig())
    frames = result.frames
    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in frames], [f.y_m for f in frames],
        [f.time_s for f in frames])

    # Speed from consecutive keyframe positions, the same way the heading model
    # sees it -- `Frame` carries no speed field, and the GPS CSV's own column is
    # not what the course was derived from.
    fast = []
    for previous, frame in zip(frames, frames[1:]):
        dt = (frame.time_s or 0.0) - (previous.time_s or 0.0)
        if dt <= 0:
            continue
        step_m = math.hypot(frame.x_m - previous.x_m, frame.y_m - previous.y_m)
        if step_m / dt >= args.min_speed_mps:
            fast.append(frame)
    if not fast:
        fast = list(frames)
        print(f"no frame reaches {args.min_speed_mps} m/s; using all frames "
              f"and treating the course as suspect")
    step = max(1, len(fast) // args.n_frames)
    sampled = fast[::step][:args.n_frames]
    print(f"{len(frames)} frames, {len(fast)} above {args.min_speed_mps} m/s, "
          f"sampling {len(sampled)}")

    # Load the sampled panoramas once, at a working resolution: 0.25 deg/px is
    # far finer than the blob centroid or the boat's pitch, and it keeps the
    # whole stack in memory so the rig median is one pass.
    loaded = []
    for frame in sampled:
        when = start + timedelta(seconds=frame.time_s)
        az_true, el_true = solar_position(when, frame.lat, frame.lon)
        if el_true <= 5.0:
            continue
        pano_path = paths.panorama_dir / f"{frame.pano_stem}.jpg"
        if not pano_path.exists():
            continue
        image = Image.open(pano_path).convert("L").resize(
            (args.work_width, args.work_width // 2), Image.BILINEAR)
        loaded.append((frame, when, az_true, el_true,
                       np.asarray(image, dtype=np.float32)))
    if not loaded:
        raise SystemExit("no sampled frame has both a panorama and a sun above "
                         "5 deg elevation")

    courses = [model.at(f.time_s) for f, *_ in loaded]
    _, course_r = circular_stats(courses)
    mask = None
    if args.mask_rig:
        mask = rig_mask([g for *_, g in loaded])
        print(f"rig mask covers {100.0 * mask.mean():.2f}% of the frame "
              f"({mask.sum()} px)")
        if course_r > 0.98:
            print("  WARNING: the course barely varies over these frames "
                  f"(R={course_r:.3f}), so the sun hardly moves in the camera "
                  f"frame either and the median cannot separate it from the "
                  f"rig. Treat the mask, and the result, with suspicion.")

    if args.scan_tolerance:
        print(f"\n  {'tol':>6s} {'used':>5s} {'offset':>7s} {'R_sun':>6s} "
              f"{'R_fixed':>8s} {'med|el_err|':>12s}")
        for tol in (12.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.0, 1.5):
            scanned = []
            for frame, when, az_true, el_true, grey in loaded:
                hit = brightest_blob_in_band(grey, el_true, tol, mask=mask)
                if hit is None:
                    continue
                scanned.append({
                    "sun_az_camera_deg": hit[0],
                    "elevation_error_deg": hit[1] - el_true,
                    "offset_deg": (model.at(frame.time_s) + hit[0] - az_true)
                    % 360.0})
            if not scanned:
                print(f"  {tol:6.1f} {0:5d}   (no blob in band)")
                continue
            off, r_s = circular_stats([x["offset_deg"] for x in scanned])
            _, r_f = circular_stats([x["sun_az_camera_deg"] for x in scanned])
            errs = sorted(abs(x["elevation_error_deg"]) for x in scanned)
            print(f"  {tol:6.1f} {len(scanned):5d} {off:7.1f} {r_s:6.3f} "
                  f"{r_f:8.3f} {errs[len(errs)//2]:12.2f}")
        print("  (R_sun must rise as the band tightens if the blob is the sun)")

    rows = []
    for frame, when, az_true, el_true, grey in loaded:
        found = brightest_blob_in_band(
            grey, el_true, args.elevation_tolerance_deg, mask=mask)
        if found is None:
            continue
        az_camera, el_measured, n_pixels = found
        course = model.at(frame.time_s)
        rows.append({
            "frame_idx": frame.frame_idx,
            "utc": when.isoformat(),
            "course_deg": round(course, 2),
            "sun_az_true_deg": round(az_true, 2),
            "sun_el_true_deg": round(el_true, 2),
            "sun_az_camera_deg": round(az_camera, 2),
            "sun_el_measured_deg": round(el_measured, 2),
            "elevation_error_deg": round(el_measured - el_true, 2),
            "n_pixels": n_pixels,
            "offset_deg": round((course + az_camera - az_true) % 360.0, 2),
        })

    if not rows:
        raise SystemExit(
            "no frame produced a sun blob inside the elevation band. Either "
            "the sky is overcast, the sun is behind structure, or the "
            "panoramas are not north-aligned the way this assumes.")

    mean, concentration = circular_stats([r["offset_deg"] for r in rows])
    r_sun, r_fixed = fixed_object_concentration(rows)
    elevation_errors = [abs(r["elevation_error_deg"]) for r in rows]

    print(f"\n{'frame':>7s} {'course':>7s} {'sun_true':>9s} {'sun_cam':>8s} "
          f"{'el_true':>8s} {'el_meas':>8s} {'offset':>7s}")
    for r in rows:
        print(f"{r['frame_idx']:7d} {r['course_deg']:7.1f} "
              f"{r['sun_az_true_deg']:9.1f} {r['sun_az_camera_deg']:8.1f} "
              f"{r['sun_el_true_deg']:8.1f} {r['sun_el_measured_deg']:8.1f} "
              f"{r['offset_deg']:7.1f}")

    if r_fixed > r_sun and course_r < 0.98:
        verdict = "FIXED-OBJECT"
        detail = (f"the blobs hold still in the camera frame (R_fixed="
                  f"{r_fixed:.3f}) better than they hold a constant offset "
                  f"(R_sun={r_sun:.3f}), while the course varied. That is the "
                  f"vehicle's own structure, not the sun. This check abstains")
    elif concentration >= R_TRUSTWORTHY:
        verdict = "AGREEING"
        detail = (f"the frames call the same angle (R={concentration:.3f}); "
                  f"read the mean as a measurement")
    elif concentration >= R_USELESS:
        verdict = "WEAK"
        detail = (f"R={concentration:.3f} -- consistent but loose; usable as "
                  f"corroboration, not as the published value")
    else:
        verdict = "SCATTERED"
        detail = (f"R={concentration:.3f} -- the blobs are not the sun (cloud, "
                  f"glare, or occlusion). This check abstains")

    print(f"\n  frames used            {len(rows)}")
    print(f"  offset (circular mean) {mean:.1f} deg")
    print(f"  concentration R        {concentration:.3f}  "
          f"(sun model; fixed-object model R={r_fixed:.3f})")
    print(f"  course variation       R={course_r:.3f} "
          f"({'enough to separate the models' if course_r < 0.98 else 'TOO LITTLE'})")
    print(f"  median |elev error|    "
          f"{sorted(elevation_errors)[len(elevation_errors)//2]:.1f} deg")
    print(f"  verdict                {verdict}: {detail}")

    record = {"dataset": paths.dataset, "log_start_utc": start.isoformat(),
              "n_frames_used": len(rows), "offset_deg": round(mean, 2),
              "concentration_R": round(concentration, 4),
              "concentration_R_fixed_object": round(r_fixed, 4),
              "course_concentration_R": round(course_r, 4),
              "median_abs_elevation_error_deg": round(
                  sorted(elevation_errors)[len(elevation_errors) // 2], 2),
              "verdict": verdict, "detail": detail,
              "usable": concentration >= R_TRUSTWORTHY, "frames": rows}
    if args.json_out:
        args.json_out.write_text(json.dumps(record, indent=1))
        print(f"wrote {args.json_out}")
    if args.write_metadata:
        write_metadata(paths, record,
                       supersede_validated=args.supersede_validated)


if __name__ == "__main__":
    main()
