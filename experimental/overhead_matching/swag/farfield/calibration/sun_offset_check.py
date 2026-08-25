"""Solar ephemeris and image helpers for alignment diagnostics.

The helpers extract diagnostic evidence only. They do not write run-directory
sidecars, publish a mount calibration, or grant a candidate authority for
localization; `build_alignment_diagnostics` owns the typed artifact contract.
"""

import math
from datetime import datetime, timezone

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo

# Half-width of the elevation band searched around the ephemeris elevation.
#
# This is the most important number in the module and it wants to be TIGHT.
# The first version used 12 deg, budgeting for boat pitch; measurement says
# the sun's elevation is recovered to about half a degree, and the slack was
# spent admitting impostors instead. Two live at the same elevation as the
# sun and are just as bright: the vehicle's sunlit structure, and the
# antipodal ghost that a dual-fisheye stitch throws roughly 180 deg opposite
# a bright source. R rising monotonically as this *yaw-invariant* gate
# tightens is the signature of closing in on the real sun rather than of
# overfitting: the gate cannot see the quantity being estimated. Widen this
# only for a platform with genuine pitch, and expect the ghost back when you do.
ELEVATION_TOLERANCE_DEG = 3.0

# A blob must be this bright relative to the frame's own dynamic range. The
# sun in a clear sky saturates; a threshold on the frame's max rather than an
# absolute 255 keeps a slightly under-exposed frame usable.
BRIGHT_FRACTION = 0.97

# A compact source subtends a fraction of a degree; allowing this much covers
# bloom and a coarse threshold. Anything wider is not a sun -- an overcast
# band thresholds into one enormous run -- and is rejected rather than
# averaged, which is what stops a white sky from reporting a confident
# azimuth.
MAX_BLOB_WIDTH_DEG = 25.0

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
    """Return the brightest compact blob in the requested elevation band.

    The result is ``(az_camera_deg, elevation_deg, n_pixels,
    centre_x_px, centre_y_px)`` in the supplied image. Pixel coordinates are
    retained so a diagnostic publisher can show the exact evidence to a
    human reviewer instead of exposing only a derived angle.

    Only rows inside the elevation band are searched, which is what keeps the
    water's sun-glitter -- as bright as the sun and always below the horizon
    -- from being mistaken for it.

    The blob is the **connected run of bright columns** containing the
    brightest column, not a fixed window around the brightest pixel. The sun
    saturates, so its peak is a plateau and `argmax` returns the plateau's
    left edge; a window centred there is centred on the edge of the blob and
    reads about half a blob radius low. A run has no such handedness.
    """
    height, width = pano.shape[:2]
    grey = pano.astype(np.float32).mean(axis=2) if pano.ndim == 3 else \
        pano.astype(np.float32)

    hi = geo.pano_px_from_direction(0.0, elevation_deg + tolerance_deg,
                                    width, height)[1]
    lo = geo.pano_px_from_direction(0.0, elevation_deg - tolerance_deg,
                                    width, height)[1]
    row_lo, row_hi = int(math.floor(min(hi, lo))), int(math.ceil(max(hi, lo)))
    row_lo, row_hi = max(0, row_lo), min(height, row_hi + 1)
    if row_hi - row_lo < 2:
        return None

    band = grey[row_lo:row_hi]
    bright = band >= band.max() * bright_fraction
    if mask is not None:
        # Masked pixels are the vehicle's own structure. Zeroing them after
        # the threshold, not before, keeps the threshold defined by the
        # frame's true dynamic range rather than by whatever is left over.
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

    # Circular mean over columns so a blob straddling the seam does not
    # average to the opposite side of the panorama.
    angles = 2 * math.pi * np.asarray(columns)[cols_idx] / width
    mean_col = (math.atan2(float((weights * np.sin(angles)).sum()),
                           float((weights * np.cos(angles)).sum()))
                / (2 * math.pi) * width) % width
    mean_row = float((weights * (rows_idx + row_lo)).sum() / weights.sum())
    az, el = geo.direction_from_pano_px(mean_col, mean_row, width, height)
    return az, el, int(weights.size), float(mean_col), float(mean_row)


def rig_mask(greys, bright_fraction: float = BRIGHT_FRACTION):
    """Pixels that are bright in the *median* frame: the vehicle's own
    structure.

    The sun sweeps the camera frame whenever the vehicle turns, so it is
    bright in a few frames and dark in the rest -- a median over frames drops
    it. A mast, boom, or sunlit sail is bolted to the camera and is bright in
    every frame, so it survives. Masking what survives is therefore a rig
    mask, and it needs no knowledge of the rig.

    Needs real course variation to work: on a dead-straight run the sun does
    not move in the camera frame either, and the median cannot tell them
    apart. The caller checks the course spread and says so.
    """
    if len(greys) < 3:
        return np.zeros(greys[0].shape, dtype=bool)
    median = np.median(np.stack(greys), axis=0)
    return median >= median.max() * bright_fraction


def circular_stats(angles_deg):
    """(mean_deg, R) -- R is 1 for perfect agreement, 0 for uniform scatter.

    The mean matches geometry.circular_mean_deg; this local twin exists only
    because the resultant length R is needed alongside it.
    """
    if len(angles_deg) == 0:
        return None, 0.0
    radians = np.radians(np.asarray(angles_deg, dtype=np.float64))
    east = float(np.mean(np.sin(radians)))
    north = float(np.mean(np.cos(radians)))
    mean_deg = math.degrees(math.atan2(east, north)) % 360.0
    # A tiny negative angle mod 360 can round to exactly 360.0 (same guard as
    # geometry.circular_mean_deg).
    return (0.0 if mean_deg >= 360.0 else mean_deg), math.hypot(east, north)


def log_start_utc(metadata: dict):
    """Return the recording's absolute UTC start, or None when absent."""
    raw = metadata.get("log_start_utc")
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
        timezone.utc)
