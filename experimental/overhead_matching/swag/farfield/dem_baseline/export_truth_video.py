"""Truth-pose alignment video: every keyframe's photo over the surface's
cylindrical depth render at the same (ground-truth) pose, heading-aligned.

One video frame per dataset keyframe (``frames_gps.csv`` row = the keyframe
index the pipeline uses). Both strips are in one frame -- column 0 is grid
north, azimuth CW -- so a real misalignment shows up as a constant horizontal
offset between the photo's skyline and the magenta render horizon drawn over
it, and course error shows up as offset that wanders.

The render is a full 360-degree cylinder and does not depend on heading, so
the heading model only rolls the photo. That makes the horizontal offset
between the two panels a direct measurement of camera-to-course offset error:
``--measure_shift`` estimates it per frame by circular correlation of the
photo's sky profile against the render's, and the summary reports the median.
Nothing is applied to the video unless ``--apply_measured_offset`` is passed.

    bazel run //experimental/overhead_matching/swag/farfield/dem_baseline:export_truth_video -- \
        --dataset_dir /data/farfield_matching/datasets/mount_washington_20260815_leg2 \
        --surface /data/farfield_matching/artifacts/dem_surfaces/mount_washington/v2/surface \
        --out_dir /data/farfield_matching/runs/dem_baseline_dev/truth_video/leg2_v2 \
        --observer_height_m 1.7 --elev_min_deg -25 --elev_max_deg 10
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

import numpy as np
from PIL import Image, ImageDraw

from experimental.overhead_matching.swag.farfield.dem_baseline import (
    depth_render,
    terrain,
    truth_strips,
    viz,
)
from experimental.overhead_matching.swag.farfield.viewers import page

GENERATOR = "dem_baseline.export_truth_video"
CAPTION_H = 30
GAP_H = 4
BG_RGB = (24, 24, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True,
                        help="HeightField base path (no suffix)")
    parser.add_argument("--background", type=Path, action="append",
                        default=[],
                        help="Coarse HeightField for the far field and for "
                             "holes in --surface (base path, no suffix); "
                             "repeatable, fine to coarse")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1,
                        help="keyframe stride (1 = every keyframe)")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--elev_min_deg", type=float, default=-20.0)
    parser.add_argument("--elev_max_deg", type=float, default=20.0)
    parser.add_argument("--n_az", type=int, default=1440)
    parser.add_argument("--n_rows", type=int, default=200)
    parser.add_argument("--observer_height_m", type=float, default=3.0)
    parser.add_argument("--max_range_m", type=float, default=30000.0)
    parser.add_argument("--yaw_offset_deg", type=float, default=0.0,
                        help="added to the nominal-forward centre yaw")
    parser.add_argument("--measure_shift", action="store_true", default=True)
    parser.add_argument("--no_measure_shift", dest="measure_shift",
                        action="store_false")
    parser.add_argument("--apply_measured_offset", action="store_true",
                        help="calibrate on a subsample, then render with the "
                             "measured median offset applied (a fitted view, "
                             "not an independent check)")
    parser.add_argument("--calibration_frames", type=int, default=40)
    parser.add_argument("--min_peak", type=float, default=0.35,
                        help="ignore shift estimates whose (detrended) profile "
                             "correlation is weaker than this")
    parser.add_argument("--min_prominence", type=float, default=0.05,
                        help="ignore shifts with a rival peak nearly as good "
                             "(ambiguous azimuth)")
    parser.add_argument("--max_fwhm_deg", type=float, default=60.0,
                        help="ignore shifts whose correlation peak is broader "
                             "than this (weak azimuth constraint)")
    parser.add_argument("--min_usable_fraction", type=float, default=0.5,
                        help="abstain from applying a measured offset unless "
                             "this fraction of calibration frames is usable")
    parser.add_argument("--max_calibration_mad_deg", type=float, default=15.0,
                        help="abstain when the usable calibration frames "
                             "disagree by more than this (per-frame gates can "
                             "pass while the ensemble is incoherent)")
    parser.add_argument("--keep_frames", action="store_true")
    parser.add_argument("--course_min_displacement_m", type=float,
                        default=truth_strips.DEFAULT_MIN_DISPLACEMENT_M)
    parser.add_argument("--course_smooth_window_s", type=float,
                        default=truth_strips.DEFAULT_SMOOTH_WINDOW_S)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class Renderer:
    """Photo/render strip pair for one keyframe."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        hf = terrain.HeightField.load(args.surface)
        self.crs = hf.crs
        self.tt = depth_render.TerrainTensor.chain_from_height_fields(
            [hf] + [terrain.HeightField.load(p) for p in args.background],
            device=args.device)
        self.config = depth_render.RenderConfig(
            max_range_m=args.max_range_m,
            observer_height_m=args.observer_height_m)
        self.frames = truth_strips.load_frames(args.dataset_dir)
        self.course = truth_strips.GridCourse(
            self.frames, self.crs,
            min_displacement_m=args.course_min_displacement_m,
            smooth_window_s=args.course_smooth_window_s)
        nominal = json.loads(
            (args.dataset_dir / "nominal_forward.json").read_text())
        self.bearing_camera_cw = float(nominal["bearing_camera_cw_deg"])

    def strips(self, i: int, extra_yaw_deg: float) -> dict:
        args = self.args
        frame = self.frames[i]
        course = self.course.course_deg(i)
        if course is None:
            raise RuntimeError(
                "the GPS course model abstained for this whole track")
        center_yaw = (course - self.bearing_camera_cw + extra_yaw_deg) % 360.0
        x, y = self.course.xy(i)
        cyl = depth_render.render_cylinder(
            self.tt, self.config, x, y, n_az=args.n_az,
            elev_min_deg=args.elev_min_deg, elev_max_deg=args.elev_max_deg,
            n_rows=args.n_rows)
        depth = cyl.depth_m.cpu().numpy()
        pano = truth_strips.open_pano(
            args.dataset_dir / "panorama" / frame["frame_file"], args.n_az * 2)
        photo = truth_strips.pano_strip(
            pano, center_yaw, elev_min_deg=args.elev_min_deg,
            elev_max_deg=args.elev_max_deg, n_az=args.n_az,
            n_rows=args.n_rows)
        return dict(frame=frame, course=course, center_yaw=center_yaw,
                    depth=depth, photo=photo, coverage=cyl.coverage,
                    observer_z_m=cyl.observer_z_m)


def measure(strips: dict, args: argparse.Namespace) \
        -> tuple[truth_strips.ShiftEstimate, bool]:
    """This frame's shift estimate, and whether it clears the gates."""
    estimate = truth_strips.estimate_shift(
        truth_strips.photo_sky_fraction(strips["photo"]),
        truth_strips.render_sky_fraction(strips["depth"]))
    usable = (estimate.peak >= args.min_peak
              and estimate.prominence >= args.min_prominence
              and estimate.fwhm_deg <= args.max_fwhm_deg)
    return estimate, usable


def compose(strips: dict, args: argparse.Namespace, caption: str) -> Image.Image:
    depth_img = viz.depth_image(strips["depth"], max_px=args.n_az)
    horizon = truth_strips.horizon_rows(strips["depth"])
    photo = truth_strips.draw_horizon(
        Image.fromarray(strips["photo"]), horizon)
    photo = truth_strips.add_compass_ticks(photo,
                                          course_deg=strips["course"])
    canvas = Image.new("RGB", (args.n_az,
                              args.n_rows * 2 + GAP_H + CAPTION_H), BG_RGB)
    canvas.paste(photo, (0, 0))
    canvas.paste(truth_strips.add_compass_ticks(depth_img.convert("RGB")),
                 (0, args.n_rows + GAP_H))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, args.n_rows * 2 + GAP_H + 6), caption, fill=(230, 230, 230))
    return canvas


def encode(frames_dir: Path, video_path: Path, fps: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
         "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264",
         "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(video_path)],
        check=True)


def main() -> None:
    args = parse_args()
    renderer = Renderer(args)
    indices = list(range(0, len(renderer.frames), max(args.stride, 1)))

    applied_offset = args.yaw_offset_deg
    calibration = None
    if args.apply_measured_offset:
        sample = [int(i) for i in np.unique(np.linspace(
            0, len(indices) - 1,
            min(args.calibration_frames, len(indices))).astype(int))]
        deltas = []
        for k in sample:
            estimate, usable = measure(
                renderer.strips(indices[k], args.yaw_offset_deg), args)
            if usable:
                deltas.append(estimate.delta_deg)
        median = truth_strips.circular_median_deg(deltas)
        mad = (float(np.median(np.abs(
            (np.asarray(deltas) - median + 180.0) % 360.0 - 180.0)))
            if median is not None else None)
        enough = len(deltas) >= args.min_usable_fraction * len(sample)
        coherent = mad is not None and mad <= args.max_calibration_mad_deg
        calibration = dict(n_sampled=len(sample), n_used=len(deltas),
                           median_shift_deg=median, mad_deg=mad,
                           applied=bool(median is not None and enough
                                        and coherent))
        if not calibration["applied"]:
            print(f"calibration abstains ({len(deltas)}/{len(sample)} frames "
                  "cleared the sharpness gates"
                  + ("" if mad is None else f", MAD {mad:.1f} deg")
                  + "); rendering with the nominal heading")
        else:
            applied_offset += median
            print(f"applying measured offset {median:+.2f} deg "
                  f"({len(deltas)}/{len(sample)} frames)")

    frames_dir = args.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for n, i in enumerate(indices):
        strips = renderer.strips(i, applied_offset)
        estimate, usable = (measure(strips, args) if args.measure_shift
                            else (None, False))
        vertical_deg, vertical_cols = truth_strips.skyline_offset_deg(
            strips["photo"], strips["depth"],
            elev_min_deg=args.elev_min_deg, elev_max_deg=args.elev_max_deg)
        fid = truth_strips.frame_id(strips["frame"])
        caption = (
            f"{args.dataset_dir.name}  {fid}  ({n + 1}/{len(indices)})  "
            f"{float(strips['frame']['latitude']):.5f},"
            f"{float(strips['frame']['longitude']):.5f}  "
            f"course {strips['course']:6.1f}  yaw {strips['center_yaw']:6.1f}"
            f" (off {applied_offset:+.1f})  "
            f"z {strips['observer_z_m']:.0f} m  cov {strips['coverage']:.2f}")
        if estimate is not None:
            caption += (
                f"  meas {estimate.delta_deg:+.1f}"
                f"{'' if usable else ' (rejected)'}  r {estimate.peak:.2f}"
                f" prom {estimate.prominence:.2f}"
                f" fwhm {estimate.fwhm_deg:.0f}")
        compose(strips, args, caption).save(frames_dir / f"{n:05d}.png")
        rows.append(dict(sequence=n, keyframe_idx=i, frame_id=fid,
                         latitude=float(strips["frame"]["latitude"]),
                         longitude=float(strips["frame"]["longitude"]),
                         course_grid_cw_deg=strips["course"],
                         center_yaw_deg=strips["center_yaw"],
                         coverage=strips["coverage"],
                         measured_shift_deg=(None if estimate is None
                                             else estimate.delta_deg),
                         usable=usable,
                         correlation_peak=(None if estimate is None
                                           else estimate.peak),
                         prominence=(None if estimate is None
                                     else estimate.prominence),
                         fwhm_deg=(None if estimate is None
                                   else estimate.fwhm_deg),
                         sigma_deg=(None if estimate is None
                                    else estimate.sigma_deg),
                         skyline_offset_deg=(None if not np.isfinite(vertical_deg)
                                             else vertical_deg),
                         skyline_offset_cols=vertical_cols))
        if (n + 1) % 25 == 0 or n + 1 == len(indices):
            rate = (time.time() - started) / (n + 1)
            print(f"{n + 1}/{len(indices)} frames  {rate:.2f} s/frame  "
                  f"eta {rate * (len(indices) - n - 1) / 60:.1f} min",
                  flush=True)

    video_path = args.out_dir / "alignment.mp4"
    encode(frames_dir, video_path, args.fps)
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    vertical = [r["skyline_offset_deg"] for r in rows
                if r["skyline_offset_deg"] is not None]
    kept = [r for r in rows if r["usable"]]
    usable = [r["measured_shift_deg"] for r in kept]
    summary = dict(
        generator=GENERATOR,
        dataset=args.dataset_dir.name,
        surface=str(args.surface),
        backgrounds=[str(p) for p in args.background],
        n_keyframes=len(renderer.frames),
        n_rendered=len(rows),
        stride=args.stride,
        fps=args.fps,
        bearing_camera_cw_deg=renderer.bearing_camera_cw,
        applied_yaw_offset_deg=applied_offset,
        calibration=calibration,
        elevation_band_deg=[args.elev_min_deg, args.elev_max_deg],
        observer_height_m=args.observer_height_m,
        max_range_m=args.max_range_m,
        course_model=dict(
            min_displacement_m=args.course_min_displacement_m,
            smooth_window_s=args.course_smooth_window_s,
            frame="projected grid azimuth (surface CRS)"),
        skyline_offset=dict(
            n_frames=len(vertical),
            median_deg=float(np.median(vertical)) if vertical else None,
            p10_deg=float(np.percentile(vertical, 10)) if vertical else None,
            p90_deg=float(np.percentile(vertical, 90)) if vertical else None,
            note="positive = the photo's skyline sits ABOVE the render's "
                 "horizon; vertical only, so it does not bias the azimuth "
                 "estimate"),
        measured_shift=dict(
            n_usable=len(usable),
            gates=dict(min_peak=args.min_peak,
                       min_prominence=args.min_prominence,
                       max_fwhm_deg=args.max_fwhm_deg),
            median_deg=truth_strips.circular_median_deg(usable),
            weighted_mean_deg=truth_strips.weighted_circular_mean_deg(
                usable, [r["sigma_deg"] for r in kept]),
            mad_deg=(float(np.median(np.abs(
                np.asarray(usable) - np.median(usable)))) if usable else None),
            median_fwhm_deg=(float(np.median([r["fwhm_deg"] for r in kept]))
                             if kept else None),
            p10_deg=float(np.percentile(usable, 10)) if usable else None,
            p90_deg=float(np.percentile(usable, 90)) if usable else None),
        frames=rows)
    (args.out_dir / "alignment.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True))

    measured = summary["measured_shift"]
    body = (
        f"<video src='{video_path.name}' controls loop "
        "style='width:100%'></video>"
        f"<p>{len(rows)} of {len(renderer.frames)} keyframes at "
        f"{args.fps:g} fps. Photo (top, magenta = render horizon) over the "
        "cylindrical depth render at the truth pose (bottom); column 0 is "
        "grid north, azimuth CW; elevation band "
        f"[{args.elev_min_deg:.0f}&deg;, {args.elev_max_deg:.0f}&deg;].</p>"
        f"<p>Heading: GPS course (min displacement "
        f"{args.course_min_displacement_m:g} m, smoothing "
        f"{args.course_smooth_window_s:g} s) minus approved nominal forward "
        f"{renderer.bearing_camera_cw:.1f}&deg;, plus an applied offset of "
        f"{applied_offset:+.2f}&deg;.</p>"
        f"<p>Measured photo-vs-render azimuth correction: median "
        + ("n/a" if measured["median_deg"] is None
           else f"{measured['median_deg']:+.2f}&deg;")
        + (", inverse-variance weighted mean "
           + ("n/a" if measured["weighted_mean_deg"] is None
              else f"{measured['weighted_mean_deg']:+.2f}&deg;"))
        + f", over the {measured['n_usable']} of {len(rows)} frames that pass "
        f"all three gates (correlation &ge; {args.min_peak:g}, peak "
        f"prominence &ge; {args.min_prominence:g}, peak width &le; "
        f"{args.max_fwhm_deg:g}&deg;) "
        f"(MAD "
        + ("n/a" if measured["mad_deg"] is None
           else f"{measured['mad_deg']:.1f}&deg;")
        + ", p10/p90 "
        + ("n/a" if measured["p10_deg"] is None
           else f"{measured['p10_deg']:+.1f}&deg;/{measured['p90_deg']:+.1f}"
                "&deg;")
        + ", median peak width "
        + ("n/a" if measured["median_fwhm_deg"] is None
           else f"{measured['median_fwhm_deg']:.0f}&deg;")
        + "). Positive means the surface puts the skyline clockwise of where "
        "the photo shows it, i.e. the centre yaw used was too small. "
        "Correlation alone is not azimuth sharpness &mdash; a smooth sky "
        "profile correlates well at many shifts &mdash; which is why the "
        "width and prominence gates exist.</p>"
        + (("<p>Calibration: "
            + ("applied" if calibration["applied"] else "abstained")
            + f", {calibration['n_used']}/{calibration['n_sampled']} sampled "
            "frames usable. A fitted view is not an independent check of the "
            "surface.</p>") if calibration is not None else "")
        + "<p class='muted'>This measurement is review evidence only: it "
        "mixes camera mount offset with GPS course error, platform crab and "
        "surface error, and cannot modify an approved nominal-forward "
        "record.</p>"
        f"<p class='muted'>surface: {page.esc(str(args.surface))}"
        + "".join(f"<br>background: {page.esc(str(p))}"
                  for p in args.background)
        + "</p>")
    (args.out_dir / "viewer.html").write_text(page.page(
        f"alignment video: {args.dataset_dir.name}", body,
        generator=GENERATOR))
    print(f"wrote {video_path}")


if __name__ == "__main__":
    main()
