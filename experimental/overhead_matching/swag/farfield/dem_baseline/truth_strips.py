"""Shared pieces of the truth-pose QA views: heading, strips, overlays.

One implementation of the photo/render azimuth convention, used by both the
still page (``export_truth_compare``) and the video (``export_truth_video``),
because a frame convention that exists twice eventually disagrees with itself
(docs/conventions.md).

Conventions here:

* Equirectangular photo: rows span +90..-90 deg elevation top to bottom, the
  centre column is camera azimuth 0, azimuth is CW-positive to the right.
* Strips produced by ``pano_strip`` and ``depth_render.render_cylinder`` share
  one frame: column 0 is grid north, azimuth increases CW.
* Map azimuth of the pano centre column = course - bearing_camera_cw_deg,
  where ``bearing_camera_cw_deg`` (the dataset's approved nominal forward) is
  the camera azimuth at which the direction of travel appears.
* Course comes from the same GPS course model the pipeline uses
  (``calibration.heading``), but evaluated on the surface's projected grid, so
  it is a grid azimuth like the renderer's -- no grid-convergence correction is
  needed anywhere in this module.
"""

import csv
import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.farfield.calibration import heading

# Pipeline defaults (gps_course.* in the build config); overridable by flags.
DEFAULT_MIN_DISPLACEMENT_M = 3.0
DEFAULT_SMOOTH_WINDOW_S = 10.0

HORIZON_RGB = (255, 64, 255)
COURSE_RGB = (80, 255, 120)


def load_frames(dataset_dir: Path) -> list[dict]:
    with open(dataset_dir / "frames_gps.csv", newline="") as f:
        return list(csv.DictReader(f))


def frame_id(frame: dict) -> str:
    return frame["frame_file"].split(",")[0]


class GridCourse:
    """Course over ground per frame, in a projected CRS's grid frame."""

    def __init__(self, frames: list[dict], crs: str, *,
                 min_displacement_m: float = DEFAULT_MIN_DISPLACEMENT_M,
                 smooth_window_s: float = DEFAULT_SMOOTH_WINDOW_S):
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", crs,
                                                  always_xy=True)
        lon = [float(f["longitude"]) for f in frames]
        lat = [float(f["latitude"]) for f in frames]
        x, y = transformer.transform(lon, lat)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.times_s = np.asarray([float(f["sensor_elapsed_s"])
                                   for f in frames], dtype=np.float64)
        self.model = heading.gps_course_model_from_positions(
            self.x, self.y, self.times_s,
            min_displacement_m=min_displacement_m,
            smooth_window_s=smooth_window_s)

    def course_deg(self, i: int) -> float | None:
        """Grid course CW from grid north at frame ``i``; None if the model
        abstained for the whole track (never moved far enough)."""
        if self.model is None:
            return None
        return float(self.model.course_world_cw_deg_at(self.times_s[i])) \
            % 360.0

    def xy(self, i: int) -> tuple[float, float]:
        return float(self.x[i]), float(self.y[i])


def open_pano(path: Path, min_width: int) -> Image.Image:
    """Open an equirectangular photo, letting JPEG decode at reduced scale.

    ``draft`` only takes power-of-two DCT scalings, so the result is at least
    ``min_width`` wide and the elevation/azimuth mapping is unchanged.
    """
    pano = Image.open(path)
    pano.draft("RGB", (min_width, max(min_width // 2, 1)))
    return pano


def pano_strip(pano: Image.Image, center_yaw_deg: float, *,
               elev_min_deg: float, elev_max_deg: float,
               n_az: int, n_rows: int) -> np.ndarray:
    """Elevation band of a photo, rolled so column 0 is grid north (CW)."""
    array = np.asarray(pano.convert("RGB"))
    height, width = array.shape[:2]
    r0 = int(round((90.0 - elev_max_deg) / 180.0 * height))
    r1 = int(round((90.0 - elev_min_deg) / 180.0 * height))
    band = array[max(r0, 0):min(r1, height)]
    col_of_north = int(round(((0.0 - center_yaw_deg) / 360.0 + 0.5) * width)) \
        % width
    band = np.roll(band, -col_of_north, axis=1)
    return np.asarray(Image.fromarray(band).resize((n_az, n_rows),
                                                   Image.BILINEAR))


def horizon_rows(depth_m: np.ndarray) -> np.ndarray:
    """Topmost row holding terrain, per column; ``n_rows`` for all-sky columns.

    The render's horizon is the skyline the surface predicts, so drawing it
    over the photo is the alignment check itself.
    """
    finite = np.isfinite(depth_m)
    hit = finite.any(axis=0)
    return np.where(hit, np.argmax(finite, axis=0), depth_m.shape[0])


def draw_horizon(image: Image.Image, rows: np.ndarray,
                 color: tuple[int, int, int] = HORIZON_RGB) -> Image.Image:
    """Overlay a horizon polyline (one point per column) on a strip."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    n_rows = out.height
    for col, row in enumerate(rows):
        if 0 <= row < n_rows:
            draw.point((col, int(row)), fill=color)
    return out


def add_compass_ticks(image: Image.Image, *,
                      course_deg: float | None = None) -> Image.Image:
    """Vertical N/E/S/W lines (and optionally the course) over a north strip."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for frac, label in ((0.0, "N"), (0.25, "E"), (0.5, "S"), (0.75, "W")):
        x = int(frac * out.width)
        draw.line([(x, 0), (x, out.height)], fill=(255, 255, 255), width=1)
        draw.text((x + 3, 2), label, fill=(255, 255, 255))
    if course_deg is not None:
        x = int((course_deg % 360.0) / 360.0 * out.width)
        draw.line([(x, 0), (x, out.height)], fill=COURSE_RGB, width=1)
        draw.text((x + 3, out.height - 12), "course", fill=COURSE_RGB)
    return out


def photo_sky_mask(band: np.ndarray) -> np.ndarray:
    """Per-pixel "looks like sky" mask.

    Deliberately crude (bright, not deeply saturated, not warmer than it is
    blue): it only has to follow the skyline well enough for a per-column
    profile and a boundary row.
    """
    f = np.asarray(band, dtype=np.float32) / 255.0
    value = f.max(axis=2)
    chroma = value - f.min(axis=2)
    saturation = np.where(value > 0, chroma / np.maximum(value, 1e-6), 0.0)
    return (value > 0.45) & (f[..., 2] >= f[..., 0] - 0.02) & (saturation < 0.6)


def photo_sky_fraction(band: np.ndarray) -> np.ndarray:
    """Per-column fraction of pixels that look like sky."""
    return photo_sky_mask(band).mean(axis=0)


def photo_skyline_rows(band: np.ndarray) -> np.ndarray:
    """Per-column row where the photo stops being sky, top down.

    ``n_rows`` where the column never stops (all sky) and 0 where it starts
    non-sky (occluder or terrain filling the band).
    """
    sky = photo_sky_mask(band)
    ground = ~sky
    hit = ground.any(axis=0)
    return np.where(hit, np.argmax(ground, axis=0), sky.shape[0])


def skyline_offset_deg(band: np.ndarray, depth_m: np.ndarray, *,
                       elev_min_deg: float, elev_max_deg: float) \
        -> tuple[float, int]:
    """Median vertical offset (deg) between the photo's skyline and the
    render's horizon, positive when the PHOTO's skyline sits higher.

    Columns where either side is all-sky or all-ground carry no boundary, so
    they are excluded; the count of usable columns is returned with the median.
    A nonzero offset is a vertical misregistration between query and render --
    camera pitch, an equirectangular mapping that is not a full 180 degrees,
    or geometry missing from the surface (canopy) -- and it does not move
    features horizontally, so it does not bias the azimuth estimate.
    """
    n_rows = depth_m.shape[0]
    photo_rows = photo_skyline_rows(band)
    render_rows = horizon_rows(depth_m)
    usable = ((photo_rows > 0) & (photo_rows < n_rows)
              & (render_rows > 0) & (render_rows < n_rows))
    if not usable.any():
        return float("nan"), 0
    deg_per_row = (elev_max_deg - elev_min_deg) / max(n_rows - 1, 1)
    offset = (render_rows[usable] - photo_rows[usable]) * deg_per_row
    return float(np.median(offset)), int(usable.sum())


def render_sky_fraction(depth_m: np.ndarray) -> np.ndarray:
    return (~np.isfinite(depth_m)).mean(axis=0)


@dataclasses.dataclass(frozen=True)
class ShiftEstimate:
    """One frame's photo-vs-render azimuth correction, with its sharpness.

    ``peak`` says how alike the two profiles look at the best shift. It does
    NOT say how well the data pin that shift down, and the two come apart
    constantly: a smooth sky profile over featureless terrain correlates at
    0.9+ while sliding tens of degrees costs almost nothing. ``fwhm_deg``
    measures that breadth and ``prominence`` measures multimodality (a second
    ridge that explains the frame nearly as well), so callers gate on those.
    """

    delta_deg: float  # add to centre yaw to align the photo with the render
    peak: float  # normalized circular correlation at the chosen shift
    prominence: float  # peak minus the best rival outside the peak's own width
    fwhm_deg: float  # full width at half maximum above the curve's baseline
    sigma_deg: float  # fwhm converted to a Gaussian sigma, for weighting


def highpass_circular(profile: np.ndarray, window_deg: float) -> np.ndarray:
    """Profile minus its circular moving average over ``window_deg``.

    Raw sky profiles are dominated by one slow lobe -- sky high on the open
    side, low where the land is -- and that lobe correlates broadly at every
    shift, so the correlation curve is ~110 deg wide on every real frame,
    sharp ones included. The azimuth information lives in the fine structure
    on top of it, so both profiles are high-passed before correlating.
    """
    n = profile.size
    bins = max(int(round(window_deg * n / 360.0)), 1)
    if bins <= 1 or bins >= n:
        return profile - profile.mean()
    kernel = np.zeros(n)
    kernel[:bins] = 1.0 / bins
    smooth = np.fft.irfft(np.fft.rfft(profile) * np.conj(np.fft.rfft(kernel)),
                          n)
    return profile - smooth


def estimate_shift(photo_profile: np.ndarray, render_profile: np.ndarray, *,
                   min_exclusion_deg: float = 5.0,
                   highpass_deg: float = 45.0) -> ShiftEstimate:
    """Azimuth correction that best lines the photo profile up with the render.

    ``delta_deg`` is what should be ADDED to ``center_yaw_deg``. Both profiles
    must be equal-length and span 360 degrees. ``highpass_deg`` of 0 disables
    the detrend (see ``highpass_circular`` for why it is on by default).
    """
    photo = np.asarray(photo_profile, dtype=np.float64)
    render = np.asarray(render_profile, dtype=np.float64)
    if photo.shape != render.shape or photo.ndim != 1:
        raise ValueError("profiles must be matching 1-D arrays")
    n = photo.size
    if highpass_deg > 0.0:
        photo = highpass_circular(photo, highpass_deg)
        render = highpass_circular(render, highpass_deg)
    p = photo - photo.mean()
    r = render - render.mean()
    denom = np.linalg.norm(p) * np.linalg.norm(r)
    if denom <= 0:
        return ShiftEstimate(0.0, 0.0, 0.0, 360.0, float("inf"))
    corr = np.fft.irfft(np.fft.rfft(p) * np.conj(np.fft.rfft(r)), n) / denom
    k = int(np.argmax(corr))
    # corr[k] pairs photo column c+k with render column c: the photo feature
    # sits k columns clockwise of where the surface puts it, so the centre yaw
    # used was k columns too large.
    shift = k if k <= n // 2 else k - n
    delta_deg = -shift * 360.0 / n

    peak = float(corr[k])
    baseline = float(np.median(corr))
    half_level = baseline + 0.5 * (peak - baseline)
    above = np.roll(corr, -k) > half_level
    if above.all():
        fwhm_bins = n
    else:
        right = int(np.argmin(above))  # first bin at or below the half level
        left = int(np.argmin(above[::-1]))
        fwhm_bins = right + left
    fwhm_deg = fwhm_bins * 360.0 / n

    # Rivals are searched outside the peak's OWN width, so a broad unimodal
    # peak is penalized by fwhm_deg rather than counted as ambiguous here.
    exclusion_bins = max(int(round(min_exclusion_deg * n / 360.0)),
                         fwhm_bins // 2, 1)
    rival = np.roll(corr, -k)[exclusion_bins:n - exclusion_bins]
    prominence = peak - float(rival.max()) if rival.size else 0.0
    sigma_deg = fwhm_deg / 2.3548  # FWHM of a Gaussian
    return ShiftEstimate(delta_deg=delta_deg, peak=peak,
                         prominence=prominence, fwhm_deg=fwhm_deg,
                         sigma_deg=sigma_deg)


def weighted_circular_mean_deg(values: list[float],
                               sigmas_deg: list[float]) -> float | None:
    """Inverse-variance weighted mean direction; None when nothing is usable."""
    if not values:
        return None
    angles = np.radians(np.asarray(values, dtype=np.float64))
    sigma = np.asarray(sigmas_deg, dtype=np.float64)
    weights = np.where(np.isfinite(sigma) & (sigma > 0), 1.0 / sigma ** 2, 0.0)
    if not weights.any():
        return None
    mean = np.arctan2((weights * np.sin(angles)).sum(),
                      (weights * np.cos(angles)).sum())
    return float((np.degrees(mean) + 180.0) % 360.0 - 180.0)


def circular_median_deg(values: list[float]) -> float | None:
    """Median direction of a set of angles, robust to wrap-around."""
    if not values:
        return None
    angles = np.radians(np.asarray(values, dtype=np.float64))
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    centered = np.degrees((angles - mean + np.pi) % (2 * np.pi) - np.pi)
    return float((np.degrees(mean) + np.median(centered) + 180.0) % 360.0
                 - 180.0)
