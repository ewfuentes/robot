"""Draw the estimated direction of travel onto keyframes, so a mount offset can
be checked by eye rather than argued about from residuals.

`mount_offset_sweep` says which offset best explains the bearings; it cannot say
whether that offset points at the bow. This does, because these datasets'
panoramas are stored in the **camera** frame
(`north_aligned: false`), so camera azimuth is a fixed function of the column:
`pano_geometry` maps them with azimuth 0 at the centre column, which is the
convention `track_merge` stamps into every `bearing_camera_deg`. Given that, the
mount-offset convention

    mount_offset_deg = azimuth, IN THE CAMERA FRAME, of the direction of travel

turns each candidate offset into one vertical line on every keyframe. Whether
the line sits on the bow, the road ahead, or the wake is a thing a person can
see in one glance -- which is what makes it worth rendering when the sweeps
disagree (boston_harbor's three legs land 142 deg apart, and each leg was a
separate camera position) or when nothing supports a number at all
(charles_river's curve is FLAT; mount_washington leg1's rests on one tracklet).

Every candidate gets its reverse drawn faintly too, because a 180 deg flip is
both the commonest mount error and invisible to a residual-minimising sweep: a
tracklet triangulates just as well from either end of the ray.

The figure alongside is the heading estimate itself: course over the leg, speed
(course is meaningless under the speed gate, so the plot says where that bites),
and how far the course swings inside one of m6's 5-keyframe fusion epochs --
the quantity that decides whether fusing bearings across an epoch is sound. On
the harbour legs that spread is ~3 deg; on the auto road it is 17-38 deg.

Run:
  bazel run //...object_tracking:heading_check_viewer -- \\
      --dataset boston_harbor_leg1 --run_dir <runs>/r004_v4_landmarks \\
      --out_dir /tmp/heading_leg1
"""

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    heading as heading_mod,
    pano_geometry as pg,
    viz_common as vc,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

# Distinct hues; the first candidate is the one the zoom crop centres on.
CANDIDATE_COLORS = ((60, 230, 90), (90, 170, 255), (255, 170, 40),
                    (230, 90, 200), (200, 200, 90))
REVERSE_COLOR = (120, 120, 120)
ZOOM_HALFWIDTH_DEG = 45.0
EPOCH_KEYFRAMES = 5  # m6's default bearing-fusion window


def candidate_offsets(paths, run_dir: Path, extra: list) -> list:
    """[(label, offset_deg)], first entry is the primary. Everything on offer:
    the dataset's recorded value, this run's sweep, and anything asked for."""
    out = []
    for item in extra:
        label, _, value = item.partition("=")
        out.append((label or f"{float(value):.0f} deg", float(value)))
    if run_dir is not None:
        sweep_path = run_dir / "mount_offset_sweep.json"
        if sweep_path.exists():
            sweep = json.loads(sweep_path.read_text())
            state = "usable" if sweep.get("usable") else "REFUSED"
            out.append((f"sweep {sweep['mount_offset_deg']:.0f} deg "
                        f"({sweep['verdict']}, {state})",
                        float(sweep["mount_offset_deg"])))
    block = json.loads(paths.metadata_path.read_text()).get("mount_offset") or {}
    if block.get("mount_offset_deg") is not None:
        out.append((f"metadata {block['mount_offset_deg']:.0f} deg "
                    f"({block.get('status', '?')})",
                    float(block["mount_offset_deg"])))
    if not out:
        raise SystemExit("no candidate offsets: pass --offset NAME=DEG, or "
                         "point --run_dir at a run with a sweep")
    return out


def column_for(offset_deg: float, width: int) -> float:
    """Pano column showing camera azimuth `offset_deg`.

    Straight through `pano_geometry`, which is the single source of truth for
    this mapping and puts azimuth 0 at the **centre** column
    (`x = ((az/360 + 0.5) mod 1) * W`) -- the same convention
    `track_merge` uses to produce the `bearing_camera_deg` the sweep consumes.
    Reimplementing it as `offset/360 * W` (azimuth 0 at column 0, which is what
    the dataset metadata's world-bearing formula keys off) puts every marker
    exactly 180 deg out, and on a boat it lands convincingly on the wake.
    """
    x, _ = pg.pano_px_from_direction(offset_deg, 0.0, width, 2)
    return x


def draw_marker(draw, x: float, height: int, color, label: str, font,
                dashed: bool = False, label_y: int = 4):
    x = int(round(x))
    if dashed:
        for y in range(0, height, 24):
            draw.line([(x, y), (x, min(y + 12, height))], fill=color, width=2)
    else:
        draw.line([(x, 0), (x, height)], fill=color, width=3)
    draw.rectangle([x + 4, label_y, x + 8 + 7 * len(label), label_y + 18],
                   fill=(0, 0, 0))
    vc.draw_caption(draw, label, font, xy=(x + 6, label_y + 2))


def render_keyframe(pano: np.ndarray, candidates: list, width: int, font):
    """Full pano, downscaled, with one marker per candidate (+ faint reverse)."""
    image = Image.fromarray(pano)
    scale = width / image.width
    image = image.resize((width, max(1, int(image.height * scale))),
                         Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    for i, (label, offset) in enumerate(candidates):
        color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
        draw_marker(draw, column_for(offset, width), image.height, color,
                    label, font, label_y=4 + 22 * i)
        draw_marker(draw, column_for(offset + 180.0, width), image.height,
                    REVERSE_COLOR, "reverse", font, dashed=True,
                    label_y=4 + 22 * i)
    return image


def render_zoom(pano: np.ndarray, offset_deg: float, label: str, font,
                out_width: int = 900):
    """A +/-45 deg crop centred where the primary candidate says we are going."""
    pano_h, pano_w = pano.shape[:2]
    window_w = int(2 * ZOOM_HALFWIDTH_DEG / 360.0 * pano_w)
    centre_x = column_for(offset_deg, pano_w)  # same convention as the bearings
    crop, _ = pg.extract_window(pano, centre_x - window_w / 2.0,
                                pano_h / 2.0 - window_w / 4.0,
                                window_w, max(1, window_w // 2))
    image = Image.fromarray(crop)
    image = image.resize((out_width, max(1, int(image.height * out_width
                                                / image.width))),
                         Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    middle = image.width // 2
    draw.line([(middle, 0), (middle, image.height)], fill=CANDIDATE_COLORS[0],
              width=3)
    draw.line([(middle - 18, image.height // 2), (middle + 18, image.height // 2)],
              fill=CANDIDATE_COLORS[0], width=2)
    vc.draw_caption(draw, f"{label} — is this straight ahead?", font, xy=(6, 6))
    return image


def course_series(frames, model):
    """Per-keyframe course, speed and the course spread over one fusion epoch."""
    course = np.array([model.at(f.time_s) for f in frames])
    speed, spread = [], []
    for i, frame in enumerate(frames):
        if i == 0:
            speed.append(0.0)
        else:
            previous = frames[i - 1]
            dt = max(frame.time_s - previous.time_s, 1e-6)
            speed.append(math.hypot(frame.x_m - previous.x_m,
                                    frame.y_m - previous.y_m) / dt)
        window = course[i:i + EPOCH_KEYFRAMES]
        deltas = np.abs(((window - window[0] + 180.0) % 360.0) - 180.0)
        spread.append(float(deltas.max()) if len(window) > 1 else 0.0)
    return course, np.array(speed), np.array(spread)


def write_figure(frames, course, speed, spread, sampled, path: Path):
    indices = [f.frame_idx for f in frames]
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(indices, course, lw=1.0, color="#3070c0")
    axes[0].set_ylabel("course (deg)")
    axes[0].set_title("heading estimate: GPS course over the leg")
    axes[1].plot(indices, speed, lw=1.0, color="#209060")
    axes[1].axhline(1.0, color="#c04040", ls=":", lw=1.0,
                    label="~speed gate (2 m step / 2 s)")
    axes[1].set_ylabel("speed (m/s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(indices, spread, lw=1.0, color="#c07020")
    axes[2].axhline(5.0, color="#606060", ls=":", lw=1.0)
    axes[2].set_ylabel(f"course swing within\n{EPOCH_KEYFRAMES} keyframes (deg)")
    axes[2].set_xlabel("keyframe")
    for axis in axes:
        for frame in sampled:
            axis.axvline(frame.frame_idx, color="#999999", lw=0.5, alpha=0.5)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    farfield_paths.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="read this run's mount_offset_sweep.json")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n_frames", type=int, default=12,
                        help="evenly spaced keyframes to render")
    parser.add_argument("--frames", type=int, nargs="*", default=None,
                        help="explicit keyframe indices instead")
    parser.add_argument("--offset", action="append", default=[],
                        metavar="LABEL=DEG",
                        help="extra candidate offset (repeatable); the first "
                             "candidate is what the zoom crop centres on")
    parser.add_argument("--width", type=int, default=1800,
                        help="rendered pano width in px")
    args = parser.parse_args()
    paths = farfield_paths.resolve(
        parser, args,
        infer_from=args.run_dir if args.run_dir else None,
        require=("dataset_base", "frame_landmarks"))

    result = ingest.run_ingest(paths.dataset_base, paths.frame_landmarks,
                              IngestConfig())
    frames = sorted(result.frames, key=lambda f: f.frame_idx)
    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in frames], [f.y_m for f in frames],
        [f.time_s for f in frames])
    course, speed, spread = course_series(frames, model)

    if args.frames:
        wanted = set(args.frames)
        sampled = [f for f in frames if f.frame_idx in wanted]
    else:
        step = max(1, len(frames) // max(args.n_frames, 1))
        sampled = frames[::step][:args.n_frames]

    candidates = candidate_offsets(paths, args.run_dir, args.offset)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = vc.load_font(15)
    write_figure(frames, course, speed, spread, sampled,
                 args.out_dir / "heading.png")

    by_index = {f.frame_idx: i for i, f in enumerate(frames)}
    rows = []
    for frame in sampled:
        pano = np.asarray(Image.open(
            paths.dataset_base / "panorama" / f"{frame.pano_stem}.jpg"))
        full = render_keyframe(pano, candidates, args.width, font)
        zoom = render_zoom(pano, candidates[0][1], candidates[0][0], font)
        full_rel = f"f{frame.frame_idx:04d}_pano.jpg"
        zoom_rel = f"f{frame.frame_idx:04d}_zoom.jpg"
        full.save(args.out_dir / full_rel, quality=86)
        zoom.save(args.out_dir / zoom_rel, quality=90)
        i = by_index[frame.frame_idx]
        rows.append((frame, full_rel, zoom_rel, course[i], speed[i], spread[i]))
        print(f"rendered f{frame.frame_idx:04d} course={course[i]:.1f} deg "
              f"speed={speed[i]:.1f} m/s")

    legend = " ".join(
        f"<span style='color:rgb{CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]}'>"
        f"&#9608; {html.escape(label)}</span>"
        for i, (label, _) in enumerate(candidates))
    parts = [
        "<html><head><title>heading check: "
        f"{html.escape(paths.dataset)}</title>",
        "<style>body{font-family:sans-serif;background:#161616;color:#ddd;"
        "margin:16px}img{display:block;max-width:100%;margin:4px 0}"
        "h2{margin-top:34px}.meta{color:#aaa;font-size:13px}</style>",
        "</head><body>",
        f"<h1>Direction of travel on the keyframes — "
        f"{html.escape(paths.dataset)}</h1>",
        "<p class='meta'>Panoramas are stored in the camera frame, so each "
        "candidate mount offset is one fixed column: the line is where that "
        "offset <em>claims</em> the vehicle is heading. Grey dashed = the same "
        "candidate flipped 180&deg;, the error a residual sweep cannot see. "
        "The zoom is &plusmn;45&deg; around the first candidate.</p>",
        f"<p>{legend} &nbsp; <span style='color:rgb{REVERSE_COLOR}'>&#9608; "
        "reverse (candidate + 180&deg;)</span></p>",
        "<img src='heading.png'>",
    ]
    for frame, full_rel, zoom_rel, crs, spd, spr in rows:
        parts.append(
            f"<h2>f{frame.frame_idx:04d}</h2>"
            f"<p class='meta'>t={frame.time_s:.1f}s &nbsp; course={crs:.1f}&deg; "
            f"&nbsp; speed={spd:.1f} m/s &nbsp; course swing over the next "
            f"{EPOCH_KEYFRAMES} keyframes={spr:.1f}&deg;</p>"
            f"<img src='{zoom_rel}' loading='lazy'>"
            f"<img src='{full_rel}' loading='lazy'>")
    parts.append("</body></html>")
    (args.out_dir / "index.html").write_text("\n".join(parts))
    print(f"\nwrote {args.out_dir}/index.html ({len(rows)} keyframes, "
          f"{len(candidates)} candidate offsets)")


if __name__ == "__main__":
    main()
