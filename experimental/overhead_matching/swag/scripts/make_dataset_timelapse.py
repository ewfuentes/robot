"""Trajectory plot and GPS-overlay timelapse for a collected dataset.

Produces the two artifacts the self-collected datasets carry, from any dataset
that has `panorama/` and `frames_gps.csv`:

  trajectory.png      the track, with frames coloured by time and start/end marked
  gps_timelapse.mp4   each frame with an inset map and a dot at its own position

The timelapse is the cheapest way to catch a dataset whose frames and positions
have come apart -- a stitched trajectory that jumps, an ordering that is not
temporal, a frame sequence that runs backwards. Those are all obvious in fifteen
seconds of video and invisible in a summary table.

    bazel run //experimental/overhead_matching/swag/scripts:make_dataset_timelapse -- \
        --dataset_path /data/farfield_matching/mapillary_datasets/folkestone_dover
"""

import argparse
import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image


def load_frames(dataset_path: Path):
    """(frame paths, lats, lons, times, dists) in dataset order."""
    rows = list(csv.DictReader(open(dataset_path / "frames_gps.csv")))
    pano_dir = dataset_path / "panorama"
    paths, lats, lons, times, dists = [], [], [], [], []
    missing = 0
    for row in rows:
        path = pano_dir / row["frame_file"]
        if not path.exists():
            missing += 1
            continue
        paths.append(path)
        lats.append(float(row["latitude"]))
        lons.append(float(row["longitude"]))
        times.append(float(row["video_t_s"]))
        dists.append(float(row["dist_m"]))
    if missing:
        print(f"  WARNING: {missing} row(s) in frames_gps.csv have no image")
    return paths, np.array(lats), np.array(lons), np.array(times), np.array(dists)


def stage_plot(dataset_path: Path, lats, lons, times, dists, out: Path):
    # Size the canvas from the track's own shape. A Channel crossing is 30 km
    # wide and 2 km tall, so a fixed square figure spends most of its area on
    # blank margin once the aspect ratio is locked to true scale.
    mid = math.radians(float(lats.mean()))
    span_x = max((lons.max() - lons.min()) * math.cos(mid), 1e-6)
    span_y = max(lats.max() - lats.min(), 1e-6)
    ratio = min(max(span_x / span_y, 0.4), 4.0)
    height = 7.0
    fig, ax = plt.subplots(figsize=(min(height * ratio, 18.0) + 2.0, height),
                           constrained_layout=True)
    ax.plot(lons, lats, "-", color="0.5", lw=1, label="track")
    scatter = ax.scatter(lons, lats, c=times, cmap="viridis", s=14, zorder=3,
                         label=f"{len(lats)} frames")
    ax.scatter([lons[0]], [lats[0]], c="lime", s=130, marker="o", edgecolor="k",
               zorder=4, label="start")
    ax.scatter([lons[-1]], [lats[-1]], c="red", s=130, marker="s", edgecolor="k",
               zorder=4, label="end")
    ax.set_aspect(1 / math.cos(math.radians(float(lats.mean()))))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{dataset_path.name} — {dists[-1] / 1000:.1f} km, "
                 f"{len(lats)} frames, {times[-1] / 60:.0f} min")
    # Outside the axes: on a thin track any in-axes placement covers the data.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0)
    bar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    bar.set_label("time since first frame (s)")
    ax.grid(alpha=0.3)
    ax.margins(0.06)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def stage_video(paths, lats, lons, out: Path, width: int, fps: int,
                max_frames: int | None):
    step = 1
    if max_frames and len(paths) > max_frames:
        step = math.ceil(len(paths) / max_frames)
        print(f"  {len(paths)} frames > --max_frames {max_frames}: "
              f"taking every {step}th")
    indices = list(range(0, len(paths), step))

    height = width // 2
    work = Path(tempfile.mkdtemp(prefix="timelapse_"))
    try:
        written = 0
        for position, i in enumerate(indices):
            image = Image.open(paths[i]).convert("RGB")
            # Perspective frames are 4:3, panoramas 2:1; fit either into the
            # canvas without distorting it.
            frame = Image.new("RGB", (width, height), (12, 12, 12))
            scaled = image.copy()
            scaled.thumbnail((width, height), Image.LANCZOS)
            offset_x = (width - scaled.width) // 2
            offset_y = (height - scaled.height) // 2
            frame.paste(scaled, (offset_x, offset_y))
            # Anchor and size the inset against the photo, not the canvas. A 4:3
            # frame letterboxes to ~60% of a 2:1 canvas, so canvas-relative
            # numbers put half the map on the black bar and make it cover 40% of
            # the image.
            photo_right = offset_x + scaled.width
            photo_top = offset_y
            inset = max(110, scaled.width // 5)

            fig = plt.figure(figsize=(inset / 100, inset / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.plot(lons, lats, "-", color="white", lw=1.2, alpha=0.9)
            ax.scatter([lons[i]], [lats[i]], c="red", s=45, zorder=5,
                       edgecolor="white")
            pad_x = max((lons.max() - lons.min()) * 0.05, 1e-4)
            pad_y = max((lats.max() - lats.min()) * 0.05, 1e-4)
            ax.set_xlim(lons.min() - pad_x, lons.max() + pad_x)
            ax.set_ylim(lats.min() - pad_y, lats.max() + pad_y)
            ax.set_aspect(1 / math.cos(math.radians(float(lats.mean()))))
            fig.patch.set_alpha(0.0)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            overlay = Image.frombuffer("RGBA", canvas.get_width_height(),
                                       canvas.buffer_rgba(), "raw", "RGBA", 0, 1).copy()
            plt.close(fig)

            pad = 8
            box = Image.new("RGBA", (overlay.width + 2 * pad, overlay.height + 2 * pad),
                            (0, 0, 0, 120))
            box.alpha_composite(overlay, (pad, pad))
            composited = frame.convert("RGBA")
            composited.alpha_composite(
                box, (max(0, photo_right - box.width - 10), photo_top + 10))
            composited.convert("RGB").save(work / f"c{position:05d}.jpg", quality=85)
            written += 1

        if not written:
            print("  no frames composited; skipping video")
            return
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
             "-pattern_type", "glob", "-i", str(work / "c*.jpg"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)],
            check=True)
        print(f"  wrote {out} ({written} frames @ {fps} fps "
              f"≈ {written / fps:.0f}s)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1280,
                        help="Output video width (default: 1280)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max_frames", type=int, default=1500,
                        help="Subsample above this many frames so a dense "
                             "trajectory does not become a 20-minute video "
                             "(default: 1500, 0 disables)")
    parser.add_argument("--skip_video", action="store_true",
                        help="Only write trajectory.png")
    args = parser.parse_args()

    dataset = args.dataset_path
    for required in ("frames_gps.csv", "panorama"):
        if not (dataset / required).exists():
            raise SystemExit(f"{dataset} has no {required}")

    paths, lats, lons, times, dists = load_frames(dataset)
    if len(paths) < 2:
        raise SystemExit(f"{dataset}: need at least 2 frames, got {len(paths)}")
    print(f"{dataset.name}: {len(paths)} frames, {dists[-1] / 1000:.2f} km")

    stage_plot(dataset, lats, lons, times, dists, dataset / "trajectory.png")
    if not args.skip_video:
        stage_video(paths, lats, lons, dataset / "gps_timelapse.mp4",
                    args.width, args.fps, args.max_frames or None)


if __name__ == "__main__":
    main()
