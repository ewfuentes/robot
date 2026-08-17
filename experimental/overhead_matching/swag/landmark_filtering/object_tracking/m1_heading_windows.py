"""M1 visualization: heading model + heading-compensated tracking windows.

Validates the two M1 building blocks before any SAM tracking:
- VideoFrameProvider: intermediate 3 fps frames pulled from the leg video,
  checked against the extracted keyframe JPEGs (same content expected).
- HeadingModel: GPS-course heading over the leg, plotted, and used to place
  tracking windows. For each test case a strip is rendered across two
  keyframe intervals with three rows:
    row 0: fixed window (no compensation)
    row 1: compensated, az_window = az_anchor - dHeading
    row 2: compensated with flipped sign (az_anchor + dHeading)
  The correct sign is whichever row keeps the landmark horizontally pinned;
  the flipped row should drift twice as fast as row 0.

Run:
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m1_heading_windows
"""

import argparse
import html
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    heading as heading_mod,
    pano_geometry as pg,
    video_frames,
    viz_common as vc,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)

DEFAULT_DATASET = Path("/data/farfield_matching/datasets/boston_harbor_leg1")
DEFAULT_LANDMARKS = Path(
    "/data/farfield_matching/artifacts/frame_landmarks/boston_harbor_leg1/v1")
DEFAULT_VIDEO = Path(
    "/data/farfield_matching/raw_material/boston_harbor_20260712/videos/long_wharf_to_hull_wharf.mp4")
DEFAULT_OUTPUT = Path(
    "/data/farfield_matching/artifacts/object_tracks/boston_harbor_leg1/v1/m1_heading")

# (name, anchor_obs_id, n_keyframe_intervals, video-frame stride).
# The departure case spans the boat's ~50 deg rotation away from the dock:
# nearly pure yaw with little translation, so it isolates the heading-
# compensation sign from parallax.
TEST_CASES = [
    ("f0000_custom_house_tower", "f0000__lm0__box0", 2, 2),
    ("f0000_cht_departure", "f0000__lm0__box0", 12, 8),
    ("f0122_crane_group", "f0122__lm2__box0", 2, 2),
    ("f0149_fort_independence", "f0149__lm0__box0", 2, 2),
]

WIN_W, WIN_H = 1024, 576
THUMB_W, THUMB_H = 384, 216


def plot_heading(model, frames, anchor_frames, out_path):
    times = np.array([f.time_s for f in frames])
    xs = np.array([f.x_m for f in frames])
    ys = np.array([f.y_m for f in frames])
    speed = np.hypot(np.gradient(xs, times), np.gradient(ys, times))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    dense_t = np.linspace(times[0], times[-1], 2000)
    ax1.plot(dense_t, model.at(dense_t), color="tab:blue")
    ax1.set_ylabel("heading, unwrapped deg CW from N")
    ax1.grid(True, alpha=0.3)
    ax2.plot(times, speed, color="tab:orange")
    ax2.set_ylabel("speed [m/s]")
    ax2.set_xlabel("video time [s]")
    ax2.grid(True, alpha=0.3)
    for name, frame in anchor_frames:
        for ax in (ax1, ax2):
            ax.axvline(frame.time_s, color="tab:red", alpha=0.5, ls="--")
        ax1.annotate(name, (frame.time_s, ax1.get_ylim()[1]), rotation=90,
                     fontsize=8, va="top")
    fig.suptitle("GPS-course heading model (leg1)")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_strip(provider, model, frames_by_idx, obs_by_frame, anchor,
                 pano_w, pano_h, font, stride=2, n_intervals=2):
    """Three-row compensation-comparison strip for one anchor observation."""
    x_min, y_min, x_max, y_max = pg.pano_bbox_for_observation(
        anchor.boxes, pano_w, pano_h)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    az_anchor, _ = pg.direction_from_pano_px(cx, cy, pano_w, pano_h)
    k = anchor.frame_idx
    t0 = frames_by_idx[k].time_s
    t1 = frames_by_idx[min(k + n_intervals, max(frames_by_idx))].time_s
    keyframe_video_idx = {
        provider.index_at_time(frames_by_idx[j].time_s): j
        for j in range(k, k + n_intervals + 1) if j in frames_by_idx
    }

    columns = list(provider.frames_between(t0, t1, stride=stride))
    rows = [("fixed", 0.0), ("comp az-dh", -1.0), ("flipped az+dh", +1.0)]
    strip = Image.new("RGB",
                      (THUMB_W * len(columns), THUMB_H * len(rows) + 20),
                      (24, 24, 24))
    draw_strip = ImageDraw.Draw(strip)
    for col, (vidx, t, frame_rgb) in enumerate(columns):
        dh = model.delta(t, t0)
        for row, (row_name, sign) in enumerate(rows):
            az_w = az_anchor + sign * dh
            wx, _ = pg.pano_px_from_direction(az_w, 0.0, pano_w, pano_h)
            wx0 = wx - WIN_W / 2.0
            wy0 = cy - WIN_H / 2.0
            crop, y_start = pg.extract_window(frame_rgb, wx0, wy0, WIN_W, WIN_H)
            img = Image.fromarray(crop).resize((THUMB_W, THUMB_H),
                                               Image.BILINEAR)
            draw = ImageDraw.Draw(img)
            kf = keyframe_video_idx.get(vidx)
            if kf is not None:
                for obs in obs_by_frame.get(kf, []):
                    vc.draw_obs_box(draw, obs, pano_w, pano_h, wx0, y_start,
                                    THUMB_W, THUMB_H, THUMB_W / WIN_W, font,
                                    highlight=(obs.obs_id == anchor.obs_id),
                                    with_label=False)
            if row == 0:
                vc.draw_caption(draw, f"+{t - t0:.1f}s dh={dh:+.1f}", font)
            if col == 0:
                vc.draw_caption(draw, row_name, font, xy=(5, THUMB_H - 20))
            strip.paste(img, (col * THUMB_W, row * THUMB_H))
    vc.draw_caption(
        draw_strip,
        f"{anchor.obs_id}: az={az_anchor:.1f} deg, keyframe boxes drawn on "
        "keyframe columns; correct sign pins the landmark",
        font, xy=(5, THUMB_H * len(rows) + 2))
    return strip


def render_alignment_check(provider, dataset_base, frame, out_path, font):
    """Side-by-side: extracted keyframe JPEG vs video frame at its time."""
    matches = sorted((dataset_base / "frames").glob(
        f"f{int(frame.pano_id[1:]):04d}_*.jpg"))
    extracted = np.asarray(Image.open(matches[0]))
    decoded = provider.frame(provider.index_at_time(frame.time_s))
    diff = float(np.mean(np.abs(extracted.astype(np.int16)
                                - decoded.astype(np.int16))))
    h = 300
    scale = h / extracted.shape[0]
    w = int(extracted.shape[1] * scale)
    side = Image.new("RGB", (w * 2 + 4, h + 20), (24, 24, 24))
    side.paste(Image.fromarray(extracted).resize((w, h)), (0, 0))
    side.paste(Image.fromarray(decoded).resize((w, h)), (w + 4, 0))
    draw = ImageDraw.Draw(side)
    vc.draw_caption(
        draw, f"{frame.pano_id}: extracted keyframe | video decode "
        f"t={frame.time_s:.2f}s  mean|diff|={diff:.2f}", font, xy=(5, h + 2))
    side.save(out_path, quality=88)
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmark_base", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result = ingest.run_ingest(args.dataset_base, args.landmark_base,
                               IngestConfig())
    frames_by_idx = {f.frame_idx: f for f in result.frames}
    obs_by_frame = {}
    obs_by_id = {}
    for obs in result.observations:
        obs_by_frame.setdefault(obs.frame_idx, []).append(obs)
        obs_by_id[obs.obs_id] = obs

    model = heading_mod.heading_model_from_positions(
        [f.x_m for f in result.frames], [f.y_m for f in result.frames],
        [f.time_s for f in result.frames])

    anchors = [(name, obs_by_id[oid], n_int, stride)
               for name, oid, n_int, stride in TEST_CASES]
    plot_heading(model, result.frames,
                 [(n, frames_by_idx[a.frame_idx]) for n, a, _, _ in anchors],
                 out / "heading.png")
    print("wrote heading.png")

    provider = video_frames.VideoFrameProvider(args.video)
    print(f"video: {provider.n_frames} frames @ {provider.fps:.3f} fps")
    probe = Image.open(
        args.dataset_base / "panorama" / f"{result.frames[0].pano_stem}.jpg")
    pano_w, pano_h = probe.size
    font = vc.load_font(14)

    html_parts = [
        "<html><head><title>M1: heading + windows</title>",
        "<style>body{font-family:sans-serif;background:#181818;color:#ddd}"
        "img{display:block;margin:4px 0;max-width:100%}</style></head><body>",
        "<h1>M1: heading model, video access, compensated windows</h1>",
        "<h2>Heading model</h2><img src='heading.png'>",
        "<h2>Keyframe / video alignment</h2>",
    ]
    for name, anchor, _, _ in anchors:
        frame = frames_by_idx[anchor.frame_idx]
        rel = f"align_{frame.pano_id}.jpg"
        diff = render_alignment_check(provider, args.dataset_base, frame,
                                      out / rel, font)
        print(f"alignment {frame.pano_id}: mean|diff|={diff:.2f}")
        html_parts.append(f"<img src='{rel}' loading='lazy'>")

    html_parts.append("<h2>Window compensation strips</h2>"
                      "<p>rows: fixed | az-dh | az+dh; boxes on keyframe "
                      "columns; anchor box highlighted</p>")
    for name, anchor, n_intervals, stride in anchors:
        strip = render_strip(provider, model, frames_by_idx, obs_by_frame,
                             anchor, pano_w, pano_h, font,
                             stride=stride, n_intervals=n_intervals)
        rel = f"strip_{name}.jpg"
        strip.save(out / rel, quality=90)
        print(f"wrote {rel}")
        html_parts.append(f"<h3>{html.escape(name)}</h3>"
                          f"<img src='{rel}' loading='lazy'>")
    html_parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(html_parts))
    print(f"wrote {out}/index.html")


if __name__ == "__main__":
    main()
