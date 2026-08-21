#!/usr/bin/env python3
"""Convert a Mapillary image sequence into a farfield/VIGOR-format dataset.

Reads Mapillary image + JSON metadata pairs and writes a dataset directory with
pano_id_mapping.csv, frames_gps.csv, intrinsics.csv, extraction_log.csv and
pipeline_metadata.json (+ manifest.json via farfield.provenance).

Images are stored UNROTATED for both projections -- equirectangular panoramas are
not north-aligned. Rotating would bake a heading estimate into the pixels, where
an error in it is unfixable without going back to the originals and cannot be
recalibrated the way a recorded angle can. Orientation is carried per frame in
intrinsics.csv, with the column->azimuth formula in pipeline_metadata.json.

The metadata's `azimuth_convention` block carries the mount-offset frame
warning (geometry.MOUNT_OFFSET_CONVENTION) verbatim, plus a machine-readable
`frame_if_derived_from_formula` tag: `heading_deg` here is the bearing of
COLUMN 0 (Mapillary's convention), so an offset derived from it is in the
column-0 frame and is exactly 180 degrees out if consumed as a pano_geometry
mount offset. Three docs claimed both dataset writers recorded this; only the
self-collect writer did, and the 20 datasets this writer shipped without it
carried exactly the metadata shape behind the pohang 180-degree incident.
"""

from collections import Counter

import argparse
import csv
import json
import math
import multiprocessing
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2

from experimental.overhead_matching.swag.farfield import geometry, provenance
# The camera_type spellings live with the API models; see models.py for the
# verification note (both spellings are plain 2:1 equirectangular).
from experimental.overhead_matching.swag.farfield.collection.models import (
    EQUIRECT_CAMERA_TYPES,
)

try:
    from tqdm import tqdm
except ImportError:  # progress bars are a convenience, not a dependency
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if desc:
            print(f"  {desc}...", flush=True)
        return iterable if iterable is not None else []

NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)


# ── Metadata loading ──────────────────────────────────────────────────────────


def load_sequence_metadata(sequence_dir: Path) -> list[dict]:
    """Load the frame-sidecar JSONs from a Mapillary staging directory.

    Only JSONs with a sibling .jpg are frame sidecars; anything else in the
    directory (the staging manifest.json, notes) is ignored. Returns the list
    sorted by sequence_position (authoritative spatial order from the
    Mapillary API), falling back to captured_at for legacy data.
    """
    metadata = []
    for json_path in sorted(sequence_dir.glob("*.json")):
        if not json_path.with_suffix(".jpg").exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        meta["json_path"] = str(json_path)
        meta["image_path"] = str(json_path.with_suffix(".jpg"))
        metadata.append(meta)

    has_seq_pos = any("sequence_position" in m for m in metadata)
    if has_seq_pos:
        metadata.sort(key=lambda m: m.get("sequence_position", float("inf")))
    else:
        print("  Warning: no sequence_position in metadata, falling back to captured_at sort")
        print("  (Re-download with extract_stitch.py for correct ordering)")
        metadata.sort(key=lambda m: m["captured_at"])

    # Warn about duplicate timestamps (1-second resolution can scramble order)
    ts_counts = Counter(m["captured_at"] for m in metadata)
    dupes = {ts: c for ts, c in ts_counts.items() if c > 1}
    if dupes:
        total_tied = sum(c for c in dupes.values())
        print(f"  Warning: {len(dupes)} duplicate timestamps ({total_tied}/{len(metadata)} images share a timestamp)")
        if not has_seq_pos:
            print("  This WILL cause ordering errors. Re-download with "
                  "extract_stitch.py, which writes sequence_position.")

    return metadata


# ── GPS utilities (all through farfield.geometry — the one owner) ─────────────


def gps_heading(m1, m2):
    """Compass heading from m1 to m2 (metadata dicts with lat/lng)."""
    east_m, north_m = geometry.enu_from_latlon(
        m2["lat"], m2["lng"], m1["lat"], m1["lng"])
    if abs(east_m) < 1e-4 and abs(north_m) < 1e-4:
        return None
    return geometry.compass_bearing_deg(east_m, north_m)


def compute_bbox_and_stats(metadata: list[dict]) -> dict:
    """Compute bounding box and area statistics from metadata GPS points."""
    lats = [m["lat"] for m in metadata]
    lngs = [m["lng"] for m in metadata]

    south, north = min(lats), max(lats)
    west, east = min(lngs), max(lngs)

    mid_lat = (south + north) / 2
    width_km = geometry.haversine_m(mid_lat, west, mid_lat, east) / 1000
    height_km = geometry.haversine_m(south, west, north, west) / 1000
    area_km2 = width_km * height_km

    total_dist_m = 0
    for i in range(1, len(metadata)):
        total_dist_m += geometry.haversine_m(
            metadata[i - 1]["lat"], metadata[i - 1]["lng"],
            metadata[i]["lat"], metadata[i]["lng"],
        )

    return {
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "width_km": width_km,
        "height_km": height_km,
        "area_km2": area_km2,
        "trajectory_km": total_dist_m / 1000,
        "num_images": len(metadata),
    }


# ── Heading source selection / validation ────────────────────────────────────

HEADING_FIELDS = {"computed": "computed_compass_angle", "compass": "compass_angle"}


def _circ_err(a: float, b: float) -> float:
    return abs(float(geometry.circular_diff_deg(a, b)))


def score_heading_source(metadata: list[dict], field: str) -> dict:
    """Score a heading field against the GPS-derived travel bearing.

    For a forward-facing rig the equirect-center bearing tracks the travel
    bearing, so a large median disagreement means the field is unusable for
    north alignment (e.g. Mapillary SfM returning 0.0 where reconstruction
    failed, or a reconstruction with a global rotation error). Frames with
    no GPS motion are skipped.
    """
    errs = []
    for i in range(len(metadata) - 1):
        gh = gps_heading(metadata[i], metadata[i + 1])
        if gh is None:
            continue
        errs.append(_circ_err(metadata[i][field], gh))
    n = len(metadata)
    zero = sum(1 for m in metadata if m[field] == 0.0)
    if not errs:
        return {"field": field, "n_pairs": 0, "median_err_deg": None,
                "mean_err_deg": None, "frac_exactly_zero": round(zero / n, 4)}
    return {
        "field": field,
        "n_pairs": len(errs),
        "median_err_deg": round(statistics.median(errs), 2),
        "mean_err_deg": round(statistics.mean(errs), 2),
        "frac_exactly_zero": round(zero / n, 4),
    }


# ── Camera-to-travel offset diagnostic ───────────────────────────────────────

# The frame tag stamped on the offset diagnostic: it is derived from
# heading_deg, whose reference is column 0 for equirect frames and the optical
# axis for perspective ones — NEITHER is the pano_geometry camera frame
# (azimuth 0 = CENTRE column), so consuming it as a mount offset is the
# exact 180-degree error geometry.MOUNT_OFFSET_CONVENTION warns about.
OFFSET_FRAME_EQUIRECT = "column_0_NOT_usable_as_mount_offset"
OFFSET_FRAME_PERSPECTIVE = "optical_axis_NOT_usable_as_mount_offset"


def compute_heading_travel_offset(metadata: list[dict], heading_field: str) -> dict:
    """Median offset between the selected heading and the GPS travel bearing.

    A DIAGNOSTIC of mount consistency, nothing more: a small spread means a
    fixed mount, a large one means a panning/hand-held camera. The value is in
    the heading_reference frame (column 0 / optical axis) and must never be
    consumed as a mount_offset — see OFFSET_FRAME_* above.
    """
    offsets = []
    for i in range(len(metadata) - 1):
        gh = gps_heading(metadata[i], metadata[i + 1])
        if gh is None:
            continue
        offsets.append(float(geometry.circular_diff_deg(
            metadata[i][heading_field], gh)))

    if not offsets:
        return {"offset_deg": 0.0, "std_deg": 0.0, "n_samples": 0}

    med = statistics.median(offsets)
    std = statistics.stdev(offsets) if len(offsets) > 1 else 0.0
    return {
        "offset_deg": round(med, 1),
        "std_deg": round(std, 1),
        "n_samples": len(offsets),
    }


# ── Projection / camera model ────────────────────────────────────────────────

# Limited-FOV models, stored as-is with intrinsics alongside.
PERSPECTIVE_CAMERA_TYPES = ("perspective", "brown", "fisheye", "radial", "simple_radial")


def classify_projection(metadata: list[dict]) -> bool:
    """True if the capture is equirectangular, False if limited-FOV.

    Exits on a mixed or unknown set: a trajectory that silently blends 360 and
    perspective frames would have a different pixel-to-azimuth mapping per
    frame, and every downstream bearing would be wrong for some of them.
    """
    kinds = {}
    for m in metadata:
        kinds.setdefault(m.get("camera_type", ""), []).append(m["id"])

    unknown = [k for k in kinds if k not in EQUIRECT_CAMERA_TYPES + PERSPECTIVE_CAMERA_TYPES]
    if unknown:
        print(f"ERROR: unrecognized camera_type(s) {unknown}; "
              f"example image ids {[kinds[k][0] for k in unknown]}. Add them to "
              f"EQUIRECT_CAMERA_TYPES or PERSPECTIVE_CAMERA_TYPES once their "
              f"projection is confirmed.")
        sys.exit(1)

    equirect = [k for k in kinds if k in EQUIRECT_CAMERA_TYPES]
    perspective = [k for k in kinds if k in PERSPECTIVE_CAMERA_TYPES]
    if equirect and perspective:
        print(f"ERROR: trajectory mixes equirectangular {equirect} and perspective "
              f"{perspective} frames. Example ids: "
              f"{[kinds[k][0] for k in equirect + perspective]}. Split it into "
              f"one dataset per projection.")
        sys.exit(1)
    return bool(equirect)


def fov_from_camera_parameters(meta: dict):
    """(hfov_deg, vfov_deg) from Mapillary's [focal_normalized, k1, k2].

    focal is normalized by max(width, height), so the pinhole half-angle on each
    axis uses that axis' extent over the same normalizer. Distortion (k1, k2) is
    not applied here — it is recorded in intrinsics.csv for consumers that want
    to undistort.
    """
    params = meta.get("camera_parameters")
    if not params or not params[0]:
        return None, None
    focal = params[0]
    w, h = meta["width"], meta["height"]
    norm = max(w, h)
    hfov = 2 * math.degrees(math.atan((w / norm) / (2 * focal)))
    vfov = 2 * math.degrees(math.atan((h / norm) / (2 * focal)))
    return hfov, vfov


# A perspective capture's horizontal field of view is physically bounded: below
# ~25 deg is a telephoto no phone or action cam carries, and above ~160 deg is a
# circular fisheye that would not be tagged `perspective`. Mapillary's focal
# comes from SfM or EXIF and is sometimes wildly wrong for a run of frames.
PLAUSIBLE_HFOV_DEG = (25.0, 160.0)

# Fewest plausible frames that can support a median. The substituted share does
# not matter much -- these are fixed single-camera captures, so the true FOV is
# near-constant and a median is a good estimate of it -- but the size of the set
# the median comes from does.
MIN_PLAUSIBLE_BASIS = 30


def repair_implausible_focals(metadata, verbose=True):
    """Replace unphysical focal lengths with the trajectory's median.

    Returns {pano_id: replacement_focal} for the frames that need one.

    The estimator has to be trajectory-wide, not per-sequence. On tokyo_bay the
    bad value (focal 5.7999, a 9.85 deg FOV) is the *majority* of its own
    sequence -- 70 of 106 frames -- so a per-sequence median substitutes the
    garbage back in. Across the trajectory it is 4.3% of frames and the other 15
    sequences agree to within 0.48-0.67, which is what makes the median sound.

    Substituting rather than dropping is deliberate: the frames themselves are
    fine (visual inspection confirms they are as wide as their neighbours), it is
    only the recorded lens that is wrong, and dropping 70 contiguous frames would
    put a hole in a stitched track. Every substituted frame is labelled
    `focal_source=substituted_implausible` in intrinsics.csv so a consumer
    needing exact intrinsics can exclude them instead.
    """
    lo, hi = PLAUSIBLE_HFOV_DEG
    plausible, suspect = [], []
    for meta in metadata:
        hfov, _ = fov_from_camera_parameters(meta)
        if hfov is None:
            continue
        (plausible if lo <= hfov <= hi else suspect).append(meta)
    if not suspect:
        return {}
    if len(plausible) < MIN_PLAUSIBLE_BASIS:
        raise SystemExit(
            f"ERROR: only {len(plausible)} frame(s) report a field of view "
            f"inside {lo}-{hi} deg, which is too thin a basis for a median "
            f"({MIN_PLAUSIBLE_BASIS} required). Inspect camera_parameters for "
            "this trajectory by hand.")

    focals = sorted(m["camera_parameters"][0] for m in plausible)
    median = focals[len(focals) // 2]
    if verbose:
        bad_fovs = sorted(fov_from_camera_parameters(m)[0] for m in suspect)
        print(f"  WARNING: {len(suspect)} of {len(plausible) + len(suspect)} "
              f"frames report an implausible FOV "
              f"({bad_fovs[0]:.2f}-{bad_fovs[-1]:.2f}°, outside {lo}-{hi}°); "
              f"substituting the trajectory median focal {median:.4f} "
              f"(≈{2 * math.degrees(math.atan(1 / (2 * median))):.1f}° on a "
              f"landscape frame) and labelling them in intrinsics.csv")
    return {m["pano_id"]: median for m in suspect}


# ── Image processing ─────────────────────────────────────────────────────────


def process_single_image(args):
    """Process a single Mapillary image (for parallel execution).

    Returns (idx, output_filename, error_msg). (This used to be handed a
    rig_offset and an is_equirect it never read; both are gone — frames are
    stored unrotated whatever the projection, so per-image processing is the
    same resize+re-encode either way.)
    """
    (idx, meta, output_dir, jpeg_quality, target_width, pano_id) = args

    image_path = meta["image_path"]
    lat = meta["lat"]
    lng = meta["lng"]

    # pano_id is a zero-padded fN index, not the Mapillary id: the filtering
    # pipeline joins frames with int(pano_id[1:]) against frames_gps.csv's idx
    # and orders frames by sorting the id as a string. A bare numeric Mapillary
    # id satisfies neither.
    output_filename = f"{pano_id},{lat:.6f},{lng:.6f},.jpg"
    output_path = output_dir / output_filename

    image = cv2.imread(image_path)
    if image is None:
        return idx, None, f"Failed to read: {image_path}"

    height, width = image.shape[:2]

    # Images are stored UNROTATED, in the camera frame as captured, for both
    # projections. Rotating to north-align would bake a heading estimate into
    # the pixels: any error in it becomes unfixable without re-deriving from the
    # originals, and it cannot be recalibrated downstream the way a recorded
    # angle can. The per-frame reference azimuth goes to intrinsics.csv instead,
    # and pipeline_metadata.json records the column->azimuth formula.
    # (This matches boston_harbor, which is also not north-aligned and has its
    # yaw offset fitted per leg.)

    if target_width and width > target_width:
        scale = target_width / width
        new_h = int(height * scale)
        image = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return idx, output_filename, None


# ── Visualization ─────────────────────────────────────────────────────────────


def create_visualization(
    original, adjusted, lat, lng, computed_compass_angle, compass_angle_val,
    all_metadata, current_idx, frame_idx, total_frames,
    trajectory_km, output_path, heading_used=None,
):
    """Create 4-panel visualization figure adapted for Mapillary data."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - BUILD carries the dep
        raise SystemExit(
            "--visualize needs matplotlib; it is a declared dependency of the "
            "bazel target, so an ImportError here means the target was run "
            f"outside bazel without it installed ({exc})")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The green "corrected" arrow shows the validated heading source's value.
    # Frames are stored UNROTATED (2026-08-12 decision) — nothing is rolled;
    # this arrow is metadata, not a pixel operation.
    if heading_used is None:
        heading_used = computed_compass_angle
    corrected_heading = heading_used % 360

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Panel 1: Original panorama
    ax1 = axes[0, 0]
    display_original = cv2.resize(original, (960, 480))
    display_original = cv2.cvtColor(display_original, cv2.COLOR_BGR2RGB)
    ax1.imshow(display_original)
    ax1.set_title(
        f"Original Panorama (Frame {frame_idx}/{total_frames}, "
        f"computed: {computed_compass_angle:.1f}°, corrected: {corrected_heading:.1f}°)",
        fontsize=11,
    )
    ax1.axis("off")

    # Panel 2: stored frame (unrotated)
    ax2 = axes[0, 1]
    display_adjusted = cv2.resize(adjusted, (960, 480))
    display_adjusted = cv2.cvtColor(display_adjusted, cv2.COLOR_BGR2RGB)
    ax2.imshow(display_adjusted)
    ax2.set_title("Stored Frame (unrotated, as captured)", fontsize=11)
    ax2.axis("off")

    # Panel 3: GPS trajectory map
    ax3 = axes[1, 0]
    ref_lat = all_metadata[0]["lat"]
    ref_lon = all_metadata[0]["lng"]

    all_x = []
    all_y = []
    for m in all_metadata:
        x, y = geometry.enu_from_latlon(m["lat"], m["lng"], ref_lat, ref_lon)
        all_x.append(x)
        all_y.append(y)

    ax3.plot(all_x, all_y, "b-", alpha=0.3, linewidth=0.5)

    # Past points
    past_x = all_x[: current_idx + 1]
    past_y = all_y[: current_idx + 1]
    ax3.scatter(past_x, past_y, c="green", s=10, alpha=0.5, label="Processed")
    ax3.scatter([all_x[current_idx]], [all_y[current_idx]], c="red", s=100, marker="o",
                zorder=5, label="Current")

    # Selected-heading arrow on trajectory
    arrow_len = max(20, (max(all_x) - min(all_x) + max(all_y) - min(all_y)) / 40)
    corr_angle = math.radians(90 - corrected_heading)
    dx = arrow_len * math.cos(corr_angle)
    dy = arrow_len * math.sin(corr_angle)
    ax3.arrow(all_x[current_idx], all_y[current_idx], dx, dy,
              head_width=arrow_len * 0.25, head_length=arrow_len * 0.15,
              fc="green", ec="darkgreen", zorder=6)

    # Cumulative distance to this frame
    cum_dist = 0
    for i in range(1, current_idx + 1):
        cum_dist += geometry.haversine_m(
            all_metadata[i - 1]["lat"], all_metadata[i - 1]["lng"],
            all_metadata[i]["lat"], all_metadata[i]["lng"],
        )

    ax3.set_xlabel("East (m)")
    ax3.set_ylabel("North (m)")
    ax3.set_title(
        f"GPS Trajectory — Frame {frame_idx}/{total_frames}, "
        f"{cum_dist/1000:.1f}/{trajectory_km:.1f} km",
    )
    ax3.legend(loc="upper right")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Compass rose — show all three headings
    ax4 = axes[1, 1]
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-1.5, 1.5)
    ax4.set_aspect("equal")

    circle = plt.Circle((0, 0), 1, fill=False, color="black", linewidth=2)
    ax4.add_patch(circle)

    for label, (ddx, ddy) in {"N": (0, 1.2), "E": (1.2, 0), "S": (0, -1.2), "W": (-1.2, 0)}.items():
        ax4.text(ddx, ddy, label, ha="center", va="center", fontsize=14, fontweight="bold")

    # computed_compass_angle (blue)
    rig_angle = math.radians(90 - computed_compass_angle)
    rig_x = 0.9 * math.cos(rig_angle)
    rig_y = 0.9 * math.sin(rig_angle)
    ax4.arrow(0, 0, rig_x, rig_y, head_width=0.12, head_length=0.08,
              fc="blue", ec="blue", linewidth=2, alpha=0.7)

    # compass_angle (red)
    travel_angle = math.radians(90 - compass_angle_val)
    travel_x = 0.9 * math.cos(travel_angle)
    travel_y = 0.9 * math.sin(travel_angle)
    ax4.arrow(0, 0, travel_x, travel_y, head_width=0.12, head_length=0.08,
              fc="red", ec="red", linewidth=2)

    # selected heading (green)
    corr_rose_angle = math.radians(90 - corrected_heading)
    corr_rose_x = 0.9 * math.cos(corr_rose_angle)
    corr_rose_y = 0.9 * math.sin(corr_rose_angle)
    ax4.arrow(0, 0, corr_rose_x, corr_rose_y, head_width=0.12, head_length=0.08,
              fc="green", ec="darkgreen", linewidth=2)

    ax4.set_title(
        f"compass: {compass_angle_val:.0f}° (red)  "
        f"computed: {computed_compass_angle:.0f}° (blue)  "
        f"selected: {corrected_heading:.0f}° (green)",
        fontsize=10,
    )
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()


# ── pipeline_metadata.json ────────────────────────────────────────────────────


def build_azimuth_convention(is_equirect: bool) -> dict:
    """The machine-readable azimuth contract stored with every dataset.

    Images are stored unrotated, so azimuth depends on the per-frame heading in
    intrinsics.csv. The equirect column-0 semantics were measured, not assumed:
    verified against solar azimuth and the vessel's wake bearing on
    folkestone_dover (docs/conventions.md holds the register of every frame
    convention).

    `mount_offset_frame` is geometry.MOUNT_OFFSET_CONVENTION verbatim, and
    `frame_if_derived_from_formula` machine-tags the trap: an offset computed
    from heading_deg / this block's formula is in the column-0 (equirect) or
    optical-axis (perspective) frame and MUST NOT be consumed as a
    pano_geometry mount offset — that error is exactly 180 degrees and was the
    pohang incident.
    """
    if is_equirect:
        return {
            "images_rotated": False,
            "frame": "camera (as captured)",
            "bearing_increases": "left_to_right",
            "heading_deg_is_bearing_of": "column_0",
            "formula": "azimuth_deg = (heading_deg + (col / width) * 360) mod 360",
            "heading_per_frame": "intrinsics.csv:heading_deg",
            "verified_by": "solar azimuth + wake bearing on folkestone_dover",
            "mount_offset_frame": geometry.MOUNT_OFFSET_CONVENTION,
            "frame_if_derived_from_formula": OFFSET_FRAME_EQUIRECT,
        }
    return {
        "images_rotated": False,
        "frame": "camera (as captured)",
        "heading_deg_is_bearing_of": "optical_axis",
        "formula": ("azimuth_deg = (heading_deg + degrees(atan((2*col/width - 1) "
                    "* tan(radians(hfov_deg)/2)))) mod 360"),
        "heading_per_frame": "intrinsics.csv:heading_deg",
        "distortion": "k1,k2 recorded in intrinsics.csv, NOT applied",
        "mount_offset_frame": geometry.MOUNT_OFFSET_CONVENTION,
        "frame_if_derived_from_formula": OFFSET_FRAME_PERSPECTIVE,
    }


def build_pipeline_metadata(*, dataset_name: str, metadata: list[dict],
                            is_equirect: bool, stats: dict, scores: dict,
                            heading_source: str, offset_info: dict,
                            substituted_count: int, image_dir_name: str,
                            num_written: int, resize, min_spacing: float,
                            jpeg_quality: int, max_heading_error_deg: float,
                            max_heading_source_disagreement_deg: float,
                            max_perspective_offset_std_deg: float,
                            skip_heading_validation: bool) -> dict:
    """Assemble pipeline_metadata.json. Pure — no I/O, unit-testable.

    Everything a consumer needs to interpret the images at all lives here:
    whether column-to-azimuth holds (is_equirectangular), that nothing is
    north-aligned, where positions came from, and the azimuth/mount-offset
    conventions (see build_azimuth_convention).
    """
    heading_field = HEADING_FIELDS[heading_source]
    sel = scores[heading_source]
    captured_at_ms = metadata[0]["captured_at"]
    capture_date = datetime.fromtimestamp(
        captured_at_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    component_sequences = sorted({m.get("sequence_id", "") for m in metadata} - {""})
    geometry_sources = Counter(m.get("geometry_source", "") for m in metadata)
    sample_hfov, sample_vfov = fov_from_camera_parameters(metadata[0])
    disagreements = [_circ_err(m["computed_compass_angle"], m["compass_angle"])
                     for m in metadata]
    med_disagreement = (round(statistics.median(disagreements), 2)
                        if disagreements else None)

    return {
        "dataset_name": dataset_name,
        "source": "mapillary",
        "sequence_id": metadata[0]["sequence_id"],
        "component_sequence_ids": component_sequences,
        "stitched_from_n_sequences": len(component_sequences),
        "projection": "equirectangular" if is_equirect else "perspective",
        "is_equirectangular": is_equirect,
        "north_aligned": False,
        "azimuth_convention": build_azimuth_convention(is_equirect),
        "camera_type": metadata[0].get("camera_type", ""),
        "camera_parameters": metadata[0].get("camera_parameters"),
        "hfov_deg": round(sample_hfov, 3) if sample_hfov else None,
        "vfov_deg": round(sample_vfov, 3) if sample_vfov else None,
        "intrinsics_csv": "intrinsics.csv" if not is_equirect else None,
        # How many frames' lens the API got wrong. Non-zero means those rows
        # carry an estimate, not a measurement -- see focal_source in
        # intrinsics.csv for which ones.
        "focals_substituted": substituted_count,
        "image_dir": image_dir_name,
        "geometry_source_counts": dict(geometry_sources),
        "resize_max_width": resize,
        "min_spacing_m": min_spacing,
        "capture_date": capture_date,
        "captured_at_ms": captured_at_ms,
        "num_images": num_written,
        "resolution": f"{metadata[0]['width']}x{metadata[0]['height']}",
        "heading_source": heading_field,
        "heading_source_scores": scores,
        "heading_reliable": bool(
            sel["median_err_deg"] is not None
            and sel["median_err_deg"] <= max_heading_error_deg
        ) if is_equirect else None,
        "heading_sources_median_disagreement_deg": med_disagreement,
        "heading_sources_disagree": (
            None if is_equirect else bool(
                med_disagreement is not None
                and med_disagreement > max_heading_source_disagreement_deg)),
        "camera_pans_relative_to_travel": (
            None if is_equirect else bool(
                offset_info["n_samples"] > 0
                and offset_info["std_deg"] > max_perspective_offset_std_deg)),
        "max_heading_error_deg": max_heading_error_deg,
        "heading_validation_overridden": bool(skip_heading_validation),
        # Diagnostic ONLY. In the heading_reference frame (column 0 / optical
        # axis), NOT the pano_geometry camera frame; its spread measures mount
        # fixedness. There is deliberately no mount_offset block here — that is
        # written by the explicit calibration publish tool, with its frame and
        # applied fields (dataset.mount_offset_record refuses anything less).
        "heading_vs_travel_offset_diagnostic": {
            **offset_info,
            "frame": (OFFSET_FRAME_EQUIRECT if is_equirect
                      else OFFSET_FRAME_PERSPECTIVE),
            "note": "median(heading_deg - GPS travel bearing); NOT a "
                    "mount_offset (see azimuth_convention.mount_offset_frame)",
        },
        "jpeg_quality": jpeg_quality,
        "bbox": {
            "south": stats["south"],
            "north": stats["north"],
            "west": stats["west"],
            "east": stats["east"],
        },
        "extent_km": {
            "width": round(stats["width_km"], 3),
            "height": round(stats["height_km"], 3),
            "area_km2": round(stats["area_km2"], 3),
        },
        "trajectory_km": round(stats["trajectory_km"], 3),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Mapillary panorama sequence to VIGOR-format dataset"
    )
    parser.add_argument("--sequence_dir", required=True, help="Mapillary sequence directory")
    parser.add_argument("--vigor_dir", required=True, help="Output VIGOR directory")
    parser.add_argument("--dataset_name", required=True, help="Dataset/city name")
    parser.add_argument("--heading_source", choices=("auto", "computed", "compass"),
                        required=True,
                        help="Heading recorded per frame: Mapillary's SfM "
                             "'computed_compass_angle', the magnetometer "
                             "'compass_angle', or 'auto' (validate both against "
                             "the GPS travel bearing and pick the better one). "
                             "(The old default was 'auto')")
    parser.add_argument("--max_heading_error_deg", type=float, required=True,
                        help="Record heading_reliable=false if the selected "
                             "heading source's median disagreement with the GPS "
                             "travel bearing exceeds this (the old default was 10)")
    parser.add_argument("--max_perspective_offset_std_deg", type=float, required=True,
                        help="Report a perspective capture as hand-held rather than "
                             "fixed-mount above this camera-to-travel offset spread. "
                             "Informational only — a panning camera is still usable "
                             "(the old default was 45)")
    parser.add_argument("--max_heading_source_disagreement_deg", type=float,
                        required=True,
                        help="Warn when a perspective capture's two heading sources "
                             "(SfM vs magnetometer) disagree by more than this median "
                             "amount, meaning at least one is wrong (the old default "
                             "was 25)")
    parser.add_argument("--skip_heading_validation", action="store_true",
                        help="Proceed even if the selected heading source fails "
                             "the GPS-bearing validation")
    parser.add_argument("--jpeg_quality", type=int, required=True,
                        help="JPEG output quality (the old default was 95)")
    parser.add_argument("--resize", type=int, required=True,
                        help="Resize to this max width, keeping aspect ratio; "
                             "0 stores original resolution (the old default "
                             "was no resize)")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Number of parallel workers")
    parser.add_argument("--trim_start", type=float, default=0,
                        help="Trim first N%% of trajectory (e.g. 50 = keep last "
                             "half); 0 (the default) trims nothing")
    parser.add_argument("--trim_end", type=float, default=0,
                        help="Trim last N%% of trajectory; 0 (the default) trims "
                             "nothing")
    parser.add_argument("--min_spacing", type=float, required=True,
                        help="Min spacing in meters between consecutive images; "
                             "0 keeps all (the old default was 0; extract_stitch "
                             "usually decimated already)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate 4-panel visualization images and mp4 video")
    args = parser.parse_args(argv)

    sequence_dir = Path(args.sequence_dir)
    vigor_dir = Path(args.vigor_dir)
    resize = args.resize or None

    if not sequence_dir.exists():
        print(f"ERROR: Sequence directory not found: {sequence_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"Mapillary → VIGOR Conversion: {args.dataset_name}")
    print("=" * 60)

    # ── Step 1: Load metadata ──
    print("\n[1/6] Loading Mapillary metadata...")
    metadata = load_sequence_metadata(sequence_dir)
    print(f"  Found {len(metadata)} images")
    if not metadata:
        print("ERROR: No images found")
        sys.exit(1)

    # Apply trim
    orig_count = len(metadata)
    if args.trim_start > 0 or args.trim_end > 0:
        n = len(metadata)
        i_start = int(n * args.trim_start / 100)
        i_end = int(n * (1 - args.trim_end / 100))
        metadata = metadata[i_start:i_end]
        print(f"  Trimmed: kept indices {i_start}-{i_end} ({len(metadata)}/{orig_count} images)")

    # Apply min spacing
    if args.min_spacing > 0:
        filtered = [metadata[0]]
        for m in metadata[1:]:
            d = geometry.haversine_m(filtered[-1]["lat"], filtered[-1]["lng"],
                                     m["lat"], m["lng"])
            if d >= args.min_spacing:
                filtered.append(m)
        print(f"  Min spacing {args.min_spacing}m: kept {len(filtered)}/{len(metadata)} images")
        metadata = filtered

    if not metadata:
        print("ERROR: No images after filtering")
        sys.exit(1)

    sample = metadata[0]
    print(f"  Resolution: {sample['width']}x{sample['height']}")
    print(f"  Camera type: {sample['camera_type']}")
    is_equirect = classify_projection(metadata)
    print(f"  Projection: {'equirectangular (360)' if is_equirect else 'perspective (limited FOV)'}")
    if not is_equirect:
        hfov, vfov = fov_from_camera_parameters(sample)
        if hfov is None:
            print("ERROR: perspective images need camera_parameters to recover FOV; "
                  "none present. Re-download metadata with the current api.py "
                  "(FULL_SEARCH_FIELDS includes camera_parameters).")
            sys.exit(1)
        print(f"  FOV: {hfov:.1f}° horizontal x {vfov:.1f}° vertical "
              f"(focal_norm={sample['camera_parameters'][0]:.4f})")
        print("  Frames are stored as captured — per-frame heading + intrinsics "
              "go to intrinsics.csv.")

    # ── Step 2: Compute GPS stats ──
    print("\n[2/6] Computing GPS statistics...")
    stats = compute_bbox_and_stats(metadata)
    print(f"  Bounding box: {stats['south']:.6f},{stats['west']:.6f} → "
          f"{stats['north']:.6f},{stats['east']:.6f}")
    print(f"  Extent: {stats['width_km']:.2f} × {stats['height_km']:.2f} km "
          f"({stats['area_km2']:.2f} km²)")
    print(f"  Trajectory length: {stats['trajectory_km']:.2f} km")

    # ── Step 3a: Heading source selection + validation ──
    print("\n[3/6] Heading source selection...")
    scores = {name: score_heading_source(metadata, field)
              for name, field in HEADING_FIELDS.items()}
    for name, s in scores.items():
        print(f"  {HEADING_FIELDS[name]:24} median err vs GPS bearing: "
              f"{s['median_err_deg']}°  (mean {s['mean_err_deg']}°, "
              f"exactly-0.0 frames: {100 * s['frac_exactly_zero']:.1f}%)")

    if args.heading_source != "auto":
        heading_source = args.heading_source
        print(f"  Requested: {HEADING_FIELDS[heading_source]}")
    elif is_equirect:
        # For a 360 capture the centre-of-image bearing does track the travel
        # bearing (it absorbs even a sideways rig mount), so agreement with the
        # GPS bearing is a valid way to pick the better field.
        # Exclude degenerate fields first. A source that is exactly 0.0 on
        # nearly every frame carries no heading at all, but can still win on
        # median error by luck -- kurashiki's compass_angle is 0.0 on 100% of
        # frames and was being selected over a real SfM bearing.
        usable = {n: s for n, s in scores.items()
                  if s["median_err_deg"] is not None and s["frac_exactly_zero"] < 0.5}
        if not usable:
            usable = {n: s for n, s in scores.items() if s["median_err_deg"] is not None}
            print("  WARNING: every heading source is mostly exactly 0.0; "
                  "the recorded reference azimuth will not be trustworthy")
        heading_source = min(usable, key=lambda n: usable[n]["median_err_deg"])
        dropped = [HEADING_FIELDS[n] for n in scores if n not in usable]
        if dropped:
            print(f"  Excluded as degenerate (mostly exactly 0.0): {', '.join(dropped)}")
        print(f"  Auto-selected: {HEADING_FIELDS[heading_source]}")
    else:
        # For a limited-FOV capture the camera need not point along travel, so
        # ranking by agreement with the GPS bearing would systematically prefer
        # whichever field happens to look forward-facing — the wrong criterion.
        # Prefer Mapillary's SfM-derived bearing, which is more reliable than a
        # device magnetometer (especially around steel hulls), unless SfM mostly
        # failed and returned exact zeros.
        zero_frac = scores["computed"]["frac_exactly_zero"]
        if zero_frac > 0.2:
            heading_source = "compass"
            print(f"  Auto-selected: compass_angle "
                  f"(computed_compass_angle is exactly 0.0 on "
                  f"{100*zero_frac:.0f}% of frames — SfM did not solve)")
        else:
            heading_source = "computed"
            print("  Auto-selected: computed_compass_angle "
                  "(SfM-derived; preferred over the magnetometer for "
                  "perspective captures)")
        # Cross-check the two independent sources against each other. This is
        # the only meaningful consistency test available when the camera's
        # pointing direction is unconstrained by the direction of travel.
        diffs = [_circ_err(m["computed_compass_angle"], m["compass_angle"])
                 for m in metadata]
        med_diff = statistics.median(diffs) if diffs else None
        print(f"  computed vs compass: median disagreement {med_diff:.1f}°")
        if med_diff is not None and med_diff > args.max_heading_source_disagreement_deg:
            print(f"  WARNING: the two heading sources disagree by {med_diff:.1f}° "
                  f"(> {args.max_heading_source_disagreement_deg}°), so at least one is "
                  f"wrong and bearings from the selected one may be off by that much. "
                  f"Recorded in pipeline_metadata.json as heading_sources_disagree.")

    sel = scores[heading_source]
    heading_gate_applies = is_equirect
    if not heading_gate_applies:
        # For a limited-FOV capture the camera need not point along travel: the
        # ferry photos here are shot sideways at the skyline, so a large median
        # disagreement with the GPS bearing is the expected mount geometry, not
        # a broken heading. Nothing is rolled by this value either — it is
        # recorded per frame in intrinsics.csv. What matters instead is that the
        # offset is *consistent* (a fixed mount), which the offset spread below
        # measures, so gate on that rather than on absolute error.
        print(f"  Perspective capture: skipping the GPS-bearing gate "
              f"(median err {sel['median_err_deg']}° is mount geometry, not error). "
              f"Consistency is checked via the offset spread below.")
    elif sel["median_err_deg"] is None or sel["median_err_deg"] > args.max_heading_error_deg:
        # Not fatal: images are stored unrotated, so a doubtful heading makes the
        # recorded reference azimuth unreliable but corrupts nothing. Downstream
        # can refit it, exactly as boston_harbor fits a per-leg yaw offset. The
        # verdict is recorded as heading_reliable in pipeline_metadata.json.
        print(f"  WARNING: '{HEADING_FIELDS[heading_source]}' disagrees with the GPS "
              f"travel bearing (median {sel['median_err_deg']}° > "
              f"{args.max_heading_error_deg}°). For a forward-facing 360 rig these "
              f"should track each other, so treat the recorded azimuth as needing "
              f"calibration (heading_reliable=false).")

    heading_field = HEADING_FIELDS[heading_source]
    for m in metadata:
        m["heading_used"] = m[heading_field]

    # ── Step 3b: camera-to-travel offset diagnostic ──
    print("\n[3/6] Camera-to-travel offset diagnostic...")
    offset_info = compute_heading_travel_offset(metadata, heading_field)
    print(f"  median offset {offset_info['offset_deg']:+.1f}° "
          f"(std: {offset_info['std_deg']:.1f}°, "
          f"n={offset_info['n_samples']} pairs) — diagnostic only, in the "
          f"heading_reference frame; NOT a mount offset")

    if not heading_gate_applies:
        # Report the mount, do not gate on it. A large spread means the camera
        # was hand-held and panned rather than fixed to the vessel — which is
        # perfectly usable, because the per-frame heading describes where the
        # camera actually pointed regardless of where the vessel was going. The
        # thing that would make bearings unreliable is a wrong heading, and that
        # is cross-checked between the two heading sources above.
        spread = offset_info["std_deg"]
        if offset_info["n_samples"] == 0:
            print("  No GPS motion pairs, so the camera-to-travel offset is "
                  "undefined (stationary capture).")
        elif spread > args.max_perspective_offset_std_deg:
            print(f"  Camera pans relative to travel (offset spread {spread:.1f}°): "
                  f"hand-held, not a fixed mount. Per-frame heading still applies; "
                  f"do not assume heading == direction of travel for this dataset.")
        else:
            print(f"  Mount looks fixed (offset spread {spread:.1f}°)")

    # ── Step 4: Process images ──
    # Always "frames": images are stored as captured, so
    # "north_aligned_panoramas" would be a false name.
    image_dir_name = "frames"
    frames_dir = vigor_dir / image_dir_name
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Zero-padded so a plain string sort is temporal order, which is what
    # detection ingest relies on when it sorts frames by pano_id.
    pad = max(4, len(str(len(metadata) - 1)))
    for i, meta in enumerate(metadata):
        meta["pano_id"] = f"f{i:0{pad}d}"

    print(f"\n[4/6] Processing {len(metadata)} images with {args.num_workers} workers...")
    print(f"  Output dir: {image_dir_name}/")
    if resize:
        print(f"  Resizing to max width: {resize}")

    work_items = [
        (i, meta, frames_dir, args.jpeg_quality, resize, meta["pano_id"])
        for i, meta in enumerate(metadata)
    ]

    results = {}  # idx -> output_filename
    errors = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_image, item): item[0] for item in work_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Writing frames"):
            idx, filename, error = future.result()
            if error:
                errors.append(error)
            else:
                results[idx] = filename

    if errors:
        print(f"  Warnings: {len(errors)} images had issues:")
        for err in errors[:5]:
            print(f"    {err}")

    print(f"  Successfully processed {len(results)}/{len(metadata)} images")

    # ── Step 5: Write output files ──
    print("\n[5/6] Writing output files...")

    # pano_id_mapping.csv
    mapping_path = vigor_dir / "pano_id_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pano_id", "lat", "lon", "filename"])
        writer.writeheader()
        for i, meta in enumerate(metadata):
            if i not in results:
                continue
            writer.writerow({
                "pano_id": meta["pano_id"],
                "lat": meta["lat"],
                "lon": meta["lng"],
                "filename": results[i],
            })
    print(f"  Wrote {mapping_path} ({len(results)} entries)")

    # frames_gps.csv — the per-frame table the farfield pipeline reads
    # (idx, dist_m, video_t_s). Nothing else writes this file; the schema
    # mirrors the self-collected legs. There is no video here, so video_t_s is
    # seconds since the first capture, which is what the frame-indexing code
    # needs it to mean.
    gps_path = vigor_dir / "frames_gps.csv"
    kept = [(i, m) for i, m in enumerate(metadata) if i in results]
    t0_ms = kept[0][1]["captured_at"] if kept else 0
    cumulative_m = 0.0
    with open(gps_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "video_t_s", "sensor_elapsed_s", "dist_m", "latitude",
            "longitude", "altitude_m", "speed_mps", "frame_file",
        ])
        writer.writeheader()
        for out_idx, (i, meta) in enumerate(kept):
            if out_idx > 0:
                prev = kept[out_idx - 1][1]
                step = geometry.haversine_m(prev["lat"], prev["lng"],
                                            meta["lat"], meta["lng"])
                cumulative_m += step
                dt = (meta["captured_at"] - prev["captured_at"]) / 1000.0
                # -1 marks "undefined", the convention the self-collected legs
                # use when the platform is stationary or the interval is
                # degenerate.
                speed = round(step / dt, 3) if dt > 0 else -1.0
            else:
                speed = -1.0
            t_s = round((meta["captured_at"] - t0_ms) / 1000.0, 3)
            writer.writerow({
                "idx": out_idx,
                "video_t_s": t_s,
                "sensor_elapsed_s": t_s,
                "dist_m": round(cumulative_m, 1),
                "latitude": f"{meta['lat']:.7f}",
                "longitude": f"{meta['lng']:.7f}",
                "altitude_m": "",
                "speed_mps": speed,
                "frame_file": results[i],
            })
    print(f"  Wrote {gps_path} ({len(kept)} rows, {cumulative_m/1000:.2f} km)")

    # intrinsics.csv — written for BOTH projections, because images are stored
    # unrotated and this is the only per-frame record of where the camera was
    # pointing. For equirectangular frames it carries the bearing of column 0;
    # for perspective frames the bearing of the optical axis. The exact
    # column->azimuth formula for each is in pipeline_metadata.json.
    intrinsics_path = vigor_dir / "intrinsics.csv"
    # Unphysical focals are repaired here rather than at download time, so
    # the raw sidecars keep exactly what the API said.
    substituted = ({} if is_equirect
                   else repair_implausible_focals([m for _, m in kept]))
    with open(intrinsics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "pano_id", "projection", "width", "height", "focal_norm",
            "k1", "k2", "hfov_deg", "vfov_deg", "heading_deg",
            "heading_reference", "heading_source", "focal_source",
        ])
        writer.writeheader()
        for out_idx, (i, meta) in enumerate(kept):
            params = list(meta.get("camera_parameters") or [None, None, None])
            replacement = substituted.get(meta["pano_id"])
            if replacement is not None:
                params[0] = replacement
                meta = dict(meta, camera_parameters=params)
            if is_equirect:
                # An equirectangular frame spans the full sphere; the lens
                # model is meaningless and Mapillary's camera_parameters for
                # these is a placeholder (e.g. [0.85, 0, 0]).
                params = [None, None, None]
                hfov, vfov = 360.0, 180.0
            else:
                hfov, vfov = fov_from_camera_parameters(meta)
            stored_w = min(meta["width"], resize) if resize else meta["width"]
            scale = stored_w / meta["width"]
            writer.writerow({
                "idx": out_idx,
                "pano_id": meta["pano_id"],
                "projection": "equirectangular" if is_equirect else "perspective",
                # What heading_deg is the bearing OF -- the single most
                # confusable thing here, so it is stated per row.
                "heading_reference": "column_0" if is_equirect else "optical_axis",
                # Dimensions as stored on disk, so a consumer computing
                # focal in pixels does not have to know the resize factor.
                "width": int(round(meta["width"] * scale)),
                "height": int(round(meta["height"] * scale)),
                "focal_norm": params[0],
                "k1": params[1] if len(params) > 1 else None,
                "k2": params[2] if len(params) > 2 else None,
                "hfov_deg": round(hfov, 4) if hfov else None,
                "vfov_deg": round(vfov, 4) if vfov else None,
                "heading_deg": round(meta["heading_used"], 3),
                "heading_source": heading_field,
                "focal_source": (
                    "n/a" if is_equirect
                    else "substituted_implausible" if replacement is not None
                    else "api"),
            })
    print(f"  Wrote {intrinsics_path} ({len(kept)} rows"
          + (f", {len(substituted)} focal(s) substituted" if substituted else "")
          + ")")

    # extraction_log.csv
    log_path = vigor_dir / "extraction_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame_idx", "pano_id", "mapillary_id", "sequence_id",
            "sequence_position", "camera_type", "geometry_source", "lat", "lng",
            "computed_compass_angle", "compass_angle", "heading_used",
            "captured_at", "original_path", "output_filename",
        ])
        writer.writeheader()
        for i, meta in enumerate(metadata):
            if i not in results:
                continue
            writer.writerow({
                "frame_idx": i,
                "pano_id": meta["pano_id"],
                "mapillary_id": meta["id"],
                "sequence_id": meta.get("sequence_id", ""),
                "sequence_position": meta.get("sequence_position", ""),
                "camera_type": meta.get("camera_type", ""),
                "geometry_source": meta.get("geometry_source", ""),
                "lat": meta["lat"],
                "lng": meta["lng"],
                "computed_compass_angle": meta["computed_compass_angle"],
                "compass_angle": meta["compass_angle"],
                "heading_used": meta["heading_used"],
                "captured_at": meta["captured_at"],
                "original_path": meta["image_path"],
                "output_filename": results[i],
            })
    print(f"  Wrote {log_path}")

    # panorama symlink. Relative, so the dataset can be moved or archived —
    # absolute link targets silently break every image read when relocated.
    panorama_link = vigor_dir / "panorama"
    if panorama_link.exists() or panorama_link.is_symlink():
        panorama_link.unlink()
    panorama_link.symlink_to(image_dir_name)
    print(f"  Created relative symlink: panorama → {image_dir_name}")

    # pipeline_metadata.json
    meta_path = vigor_dir / "pipeline_metadata.json"
    pipeline_meta = build_pipeline_metadata(
        dataset_name=args.dataset_name,
        metadata=metadata,
        is_equirect=is_equirect,
        stats=stats,
        scores=scores,
        heading_source=heading_source,
        offset_info=offset_info,
        substituted_count=len(substituted),
        image_dir_name=image_dir_name,
        num_written=len(results),
        resize=resize,
        min_spacing=args.min_spacing,
        jpeg_quality=args.jpeg_quality,
        max_heading_error_deg=args.max_heading_error_deg,
        max_heading_source_disagreement_deg=args.max_heading_source_disagreement_deg,
        max_perspective_offset_std_deg=args.max_perspective_offset_std_deg,
        skip_heading_validation=args.skip_heading_validation,
    )
    with open(meta_path, "w") as f:
        json.dump(pipeline_meta, f, indent=2)
    print(f"  Wrote {meta_path}")

    provenance.write(
        vigor_dir,
        generator="//experimental/overhead_matching/swag/farfield/"
                  "collection:mapillary_to_vigor",
        inputs={"sequence_dir": sequence_dir.resolve()},
        config={
            "dataset_name": args.dataset_name,
            "heading_source": args.heading_source,
            "heading_source_selected": heading_field,
            "max_heading_error_deg": args.max_heading_error_deg,
            "max_perspective_offset_std_deg": args.max_perspective_offset_std_deg,
            "max_heading_source_disagreement_deg":
                args.max_heading_source_disagreement_deg,
            "jpeg_quality": args.jpeg_quality,
            "resize": resize,
            "trim_start": args.trim_start,
            "trim_end": args.trim_end,
            "min_spacing": args.min_spacing,
        },
    )

    # ── Step 6: Visualizations (optional) ──
    if args.visualize:
        vis_dir = vigor_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[6/6] Creating {len(metadata)} visualizations (all frames)...")
        for idx in tqdm(range(len(metadata)), desc="Visualizing"):
            meta = metadata[idx]
            if idx not in results:
                continue

            original = cv2.imread(meta["image_path"])
            adjusted = cv2.imread(str(frames_dir / results[idx]))
            if original is None or adjusted is None:
                continue

            # Sequential numbering for ffmpeg: vis_%04d.jpg
            vis_path = vis_dir / f"vis_{idx:04d}.jpg"
            create_visualization(
                original, adjusted,
                meta["lat"], meta["lng"],
                meta["computed_compass_angle"],
                meta["compass_angle"],
                metadata, idx, idx + 1, len(metadata),
                stats["trajectory_km"],
                vis_path,
                heading_used=meta["heading_used"],
            )
        print(f"  Saved to {vis_dir}")

        # Compile into video
        video_name = f"{args.dataset_name.lower()}_alignment.mp4"
        video_path = vigor_dir / video_name
        ffmpeg_cmd = [
            "ffmpeg", "-framerate", "10",
            "-i", str(vis_dir / "vis_%04d.jpg"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(video_path), "-y",
        ]
        print(f"  Encoding video: {video_path}")
        subprocess.run(ffmpeg_cmd, capture_output=True)
        print(f"  Video saved: {video_path}")
    else:
        print("\n[6/6] Skipping visualizations (use --visualize to enable)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)
    print(f"  Output: {vigor_dir}")
    print(f"  Frames: {frames_dir} ({len(results)} images, unrotated)")
    print(f"  Bounding box (for satellite download):")
    print(f"    --bbox {stats['south']:.6f} {stats['west']:.6f} "
          f"{stats['north']:.6f} {stats['east']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
