"""Trajectory plot and GPS-overlay timelapse for collected datasets.

Produces the review artifacts the self-collected datasets carry, from any
dataset that has `panorama/` and `frames_gps.csv`:

  trajectory.png      the track, frames coloured by time, start/end marked
  gps_timelapse.mp4   each frame with an inset map and a dot at its position

When the dataset also has an approved ``nominal_forward.json``, the same
transaction includes:

  north_aligned_timelapse.mp4  low-resolution panorama rolled to true north

The north-aligned video is review evidence only.  Stored panoramas remain in
their original camera frame.

The timelapse is the cheapest way to catch a dataset whose frames and
positions have come apart — a stitched trajectory that jumps, an ordering that
is not temporal, a frame sequence that runs backwards. Those are all obvious
in fifteen seconds of video and invisible in a summary table.

Outputs are published together under the dataset's
`_manifests/timelapse/` directory (the derived lane beside the frozen
definition; excluded from checksums.sha256). The directory is built as a
sibling `.incomplete` artifact and becomes visible with one no-clobber rename,
so a reviewer can never mistake one new view plus one stale or missing view for
a completed review set.

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:make_dataset_timelapse -- \\
        --dataset_path /path/to/dataset [more...]
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from experimental.overhead_matching.swag.farfield import (  # noqa: E402
    artifact,
    geometry,
    nominal_forward,
    paths as paths_lib,
    publication,
    provenance,
)
from experimental.overhead_matching.swag.farfield.calibration import (  # noqa: E402
    heading,
)

TRAJECTORY_NAME = "trajectory.png"
TIMELAPSE_NAME = "gps_timelapse.mp4"
NORTH_ALIGNED_NAME = "north_aligned_timelapse.mp4"
REVIEW_DIRECTORY_NAME = "timelapse"
REVIEW_KIND = "dataset_timelapse"
REVIEW_VERSION = "v1"
COURSE_MIN_DISPLACEMENT_M = 3.0
COURSE_SMOOTH_WINDOW_S = 10.0


def view_output_dir(dataset_path: Path) -> Path:
    """Return the transactional directory containing the review views."""
    dataset_path = Path(dataset_path)
    return dataset_path / "_manifests" / REVIEW_DIRECTORY_NAME


def load_frames(dataset_path: Path):
    """(frame paths, lats, lons, times, dists) in dataset order."""
    with (dataset_path / "frames_gps.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    pano_dir = dataset_path / "panorama"
    paths, lats, lons, times, dists = [], [], [], [], []
    frame_names = []
    for row in rows:
        frame_file = row.get("frame_file")
        if (not frame_file or Path(frame_file).name != frame_file
                or frame_file in frame_names):
            raise ValueError(
                f"invalid or duplicate frames_gps.csv frame_file {frame_file!r}")
        frame_names.append(frame_file)
        path = pano_dir / frame_file
        target = path.resolve()
        if not target.is_file() or target.is_symlink():
            raise ValueError(
                f"frames_gps.csv references no regular image: {path}")
        paths.append(path)
        lats.append(float(row["latitude"]))
        lons.append(float(row["longitude"]))
        times.append(float(row["video_t_s"]))
        dists.append(float(row["dist_m"]))
    panorama_names = sorted(path.name for path in pano_dir.glob("*.jpg"))
    if sorted(frame_names) != panorama_names:
        missing_rows = sorted(set(panorama_names) - set(frame_names))
        missing_images = sorted(set(frame_names) - set(panorama_names))
        raise ValueError(
            "frames_gps.csv and panorama JPEGs do not have exact coverage: "
            f"unlisted_images={missing_rows}, missing_images={missing_images}")
    return (paths, np.array(lats), np.array(lons), np.array(times),
            np.array(dists))


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
    ax.scatter([lons[0]], [lats[0]], c="lime", s=130, marker="o",
               edgecolor="k", zorder=4, label="start")
    ax.scatter([lons[-1]], [lats[-1]], c="red", s=130, marker="s",
               edgecolor="k", zorder=4, label="end")
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
    out.parent.mkdir(parents=True, exist_ok=True)
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
            # Anchor and size the inset against the photo, not the canvas. A
            # 4:3 frame letterboxes to ~60% of a 2:1 canvas, so
            # canvas-relative numbers put half the map on the black bar and
            # make it cover 40% of the image.
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
            overlay = Image.frombuffer(
                "RGBA", canvas.get_width_height(), canvas.buffer_rgba(),
                "raw", "RGBA", 0, 1).copy()
            plt.close(fig)

            pad = 8
            box = Image.new(
                "RGBA", (overlay.width + 2 * pad, overlay.height + 2 * pad),
                (0, 0, 0, 120))
            box.alpha_composite(overlay, (pad, pad))
            composited = frame.convert("RGBA")
            composited.alpha_composite(
                box, (max(0, photo_right - box.width - 10), photo_top + 10))
            composited.convert("RGB").save(work / f"c{position:05d}.jpg",
                                           quality=85)
            written += 1

        if not written:
            print("  no frames composited; skipping video")
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
             "-pattern_type", "glob", "-i", str(work / "c*.jpg"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             str(out)],
            check=True)
        print(f"  wrote {out} ({written} frames @ {fps} fps "
              f"≈ {written / fps:.0f}s)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _approved_nominal_forward(dataset: Path):
    path = dataset / "nominal_forward.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"nominal-forward calibration is not a regular file: {path}")
    return nominal_forward.load(path, expected_dataset=dataset.name)


def _course_degrees(dataset: Path, lats, lons, times) -> np.ndarray:
    with (dataset / "frames_gps.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    recorded = [(row.get("gps_course_deg") or "").strip() for row in rows]
    if recorded and all(recorded):
        return np.asarray([float(value) % 360.0 for value in recorded])

    frame = geometry.RegionFrame(float(np.mean(lats)), float(np.mean(lons)))
    east, north = frame.enu_from_latlon(lats, lons)
    model = heading.gps_course_model_from_positions(
        east, north, times,
        min_displacement_m=COURSE_MIN_DISPLACEMENT_M,
        smooth_window_s=COURSE_SMOOTH_WINDOW_S)
    if model is None:
        raise ValueError(f"{dataset}: cannot derive a north-alignment course")
    return np.asarray(model.course_world_cw_deg_at(times)) % 360.0


def north_aligned_panorama(image: Image.Image,
                           center_yaw_cw_deg: float) -> Image.Image:
    """Roll an equirectangular image so output column zero is true north."""
    array = np.asarray(image.convert("RGB"))
    col_of_north = int(round(
        ((-center_yaw_cw_deg / 360.0) + 0.5) * array.shape[1])) \
        % array.shape[1]
    return Image.fromarray(np.roll(array, -col_of_north, axis=1))


def stage_north_aligned_video(paths, times, courses, calibration, out: Path,
                              width: int, fps: int,
                              max_frames: int | None):
    step = 1
    if max_frames and len(paths) > max_frames:
        step = math.ceil(len(paths) / max_frames)
    indices = list(range(0, len(paths), step))
    height = width // 2
    work = Path(tempfile.mkdtemp(prefix="north_aligned_"))
    try:
        for position, i in enumerate(indices):
            with Image.open(paths[i]) as source:
                source.draft("RGB", (width, height))
                pano = source.convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR)
            center_yaw = (
                courses[i] - calibration.bearing_camera_cw_deg) % 360.0
            frame = north_aligned_panorama(pano, center_yaw)
            draw = ImageDraw.Draw(frame, "RGBA")
            for fraction, label in (
                    (0.0, "N"), (0.25, "E"), (0.5, "S"), (0.75, "W")):
                x = min(int(round(fraction * width)), width - 1)
                draw.line((x, 0, x, height), fill=(255, 255, 255, 150),
                          width=1)
                draw.text((x + 4, 4), label, fill=(255, 255, 255, 255))
            course_x = min(int(round((courses[i] % 360.0) / 360.0 * width)),
                           width - 1)
            draw.line((course_x, 0, course_x, height),
                      fill=(80, 255, 120, 220), width=2)
            caption = (
                f"REVIEW ONLY  {paths[i].stem.split(',')[0]}  "
                f"t={times[i]:.1f}s  course={courses[i]:.1f} deg  "
                f"nominal-forward={calibration.bearing_camera_cw_deg:.1f} deg")
            box_height = 24
            draw.rectangle((0, height - box_height, width, height),
                           fill=(0, 0, 0, 150))
            draw.text((6, height - box_height + 5), caption,
                      fill=(255, 255, 255, 255))
            frame.save(work / f"c{position:05d}.jpg", quality=85)

        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
             "-pattern_type", "glob", "-i", str(work / "c*.jpg"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
             str(out)],
            check=True)
        print(f"  wrote {out} ({len(indices)} frames @ {fps} fps "
              f"≈ {len(indices) / fps:.0f}s)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _review_outputs(skip_video: bool,
                    include_north_aligned: bool = False) -> tuple[str, ...]:
    names = [TRAJECTORY_NAME]
    if not skip_video:
        names.append(TIMELAPSE_NAME)
        if include_north_aligned:
            names.append(NORTH_ALIGNED_NAME)
    return tuple(sorted(names))


def _input_digests(dataset: Path) -> tuple[dict[str, str], dict[str, str]]:
    source_digests = paths_lib.dataset_source_digests(dataset)
    inputs = {
        "pipeline_metadata": source_digests[
            paths_lib.DATASET_PIPELINE_METADATA_SHA256],
        "frames_gps": source_digests[
            paths_lib.DATASET_FRAMES_GPS_SHA256],
        "panorama_directory": source_digests[
            paths_lib.DATASET_PANORAMA_SHA256],
    }
    nominal_path = dataset / "nominal_forward.json"
    if nominal_path.exists() or nominal_path.is_symlink():
        if nominal_path.is_symlink() or not nominal_path.is_file():
            raise ValueError(
                "nominal-forward calibration is not a regular file: "
                f"{nominal_path}")
        digest = artifact.sha256_file(nominal_path)
        source_digests["nominal_forward_sha256"] = digest
        inputs["nominal_forward"] = digest
    return source_digests, inputs


def _validate_review_outputs(root: Path, outputs: tuple[str, ...]) -> None:
    trajectory = root / TRAJECTORY_NAME
    if trajectory.is_symlink() or not trajectory.is_file():
        raise ValueError(f"trajectory output is not a regular file: {trajectory}")
    try:
        with Image.open(trajectory) as image:
            image.load()
            if image.format != "PNG" or image.width < 1 or image.height < 1:
                raise ValueError(
                    f"trajectory output is not a valid PNG: {trajectory}")
    except OSError as exc:
        raise ValueError(
            f"trajectory output is not a decodable PNG: {trajectory}") from exc

    for name in (TIMELAPSE_NAME, NORTH_ALIGNED_NAME):
        if name not in outputs:
            continue
        video = root / name
        if video.is_symlink() or not video.is_file():
            raise ValueError(f"timelapse output is not a regular file: {video}")
        with video.open("rb") as stream:
            header = stream.read(12)
        if len(header) < 12 or header[4:8] != b"ftyp":
            raise ValueError(f"timelapse output is not an MP4 file: {video}")


def reject_legacy_views(dataset: Path) -> None:
    legacy = [
        dataset / name
        for name in (TRAJECTORY_NAME, TIMELAPSE_NAME, NORTH_ALIGNED_NAME)
    ] + [
        dataset / "_manifests" / name
        for name in (TRAJECTORY_NAME, TIMELAPSE_NAME, NORTH_ALIGNED_NAME)
    ]
    present = [path for path in legacy if path.exists() or path.is_symlink()]
    if present:
        raise FileExistsError(
            "legacy timelapse outputs would conflict with the transactional "
            f"review artifact; move or remove them explicitly: {present}")


def validate_completed(
        dataset: Path, *, expected_config: dict | None = None,
        expected_outputs: tuple[str, ...] | None = None,
) -> artifact.ArtifactRef:
    """Validate a completed review artifact against current dataset bytes."""
    dataset = Path(dataset)
    out_dir = view_output_dir(dataset)
    reference = artifact.open_artifact(
        out_dir,
        expected_kind=REVIEW_KIND,
        expected_dataset=dataset.name,
        expected_version=REVIEW_VERSION)
    manifest = artifact.load_manifest(out_dir)
    outputs = manifest.declared_outputs
    if outputs not in (
            _review_outputs(False), _review_outputs(True),
            _review_outputs(False, True)):
        raise ValueError(
            f"completed timelapse has invalid outputs: {out_dir}")
    if expected_outputs is not None and outputs != expected_outputs:
        raise ValueError(
            f"completed timelapse has different outputs: {out_dir}")
    if expected_config is not None:
        if dict(manifest.config) != expected_config:
            raise ValueError(
                f"completed timelapse has a different identity: {out_dir}")
    elif manifest.config.get("input_digests") != _input_digests(dataset)[1]:
        raise ValueError(
            f"completed timelapse does not bind current dataset bytes: "
            f"{out_dir}")
    _validate_review_outputs(out_dir, outputs)
    return reference


def render(dataset: Path, width: int, fps: int, max_frames: int | None,
           skip_video: bool) -> artifact.ArtifactRef:
    dataset = Path(dataset)
    for required in ("pipeline_metadata.json", "frames_gps.csv", "panorama"):
        path = dataset / required
        missing = (not path.is_dir() if required == "panorama"
                   else path.is_symlink() or not path.is_file())
        if missing:
            raise ValueError(f"{dataset} has no regular {required}")
    reject_legacy_views(dataset)
    source_digests, input_digests = _input_digests(dataset)
    paths, lats, lons, times, dists = load_frames(dataset)
    calibration = _approved_nominal_forward(dataset)
    if len(paths) < 2:
        raise ValueError(f"{dataset}: need at least 2 frames, got "
                         f"{len(paths)}")
    print(f"{dataset.name}: {len(paths)} frames, {dists[-1] / 1000:.2f} km")
    out_dir = view_output_dir(dataset)
    include_north_aligned = calibration is not None and not skip_video
    outputs = _review_outputs(skip_video, include_north_aligned)
    config = {
        "input_digests": input_digests,
        "render": {
            "width": width,
            "fps": fps,
            "max_frames": max_frames,
            "skip_video": skip_video,
            "n_frames": len(paths),
            "north_aligned": include_north_aligned,
        },
    }
    if out_dir.exists() or out_dir.is_symlink():
        reference = validate_completed(
            dataset, expected_config=config, expected_outputs=outputs)
        print(f"  reusing complete timelapse review artifact {out_dir}")
        return reference

    with publication.published_artifact(
            out_dir,
            kind=REVIEW_KIND,
            dataset=dataset.name,
            version=REVIEW_VERSION,
            generator="farfield.dataset_tools.make_dataset_timelapse",
            git_commit=provenance.git_commit(),
            config=config,
            declared_outputs=outputs) as builder:
        stage_plot(
            dataset, lats, lons, times, dists,
            builder.output_path(TRAJECTORY_NAME))
        if not skip_video:
            stage_video(
                paths, lats, lons,
                builder.output_path(TIMELAPSE_NAME),
                width, fps, max_frames)
            if calibration is not None:
                stage_north_aligned_video(
                    paths, times, _course_degrees(dataset, lats, lons, times),
                    calibration, builder.output_path(NORTH_ALIGNED_NAME),
                    width, fps, max_frames)
        _validate_review_outputs(builder.path, outputs)
        if _input_digests(dataset)[0] != source_digests:
            raise ValueError(
                "dataset source bytes changed during timelapse rendering")
    assert builder.artifact_ref is not None
    return builder.artifact_ref


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", nargs="+", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1280,
                        help="Output video width (default: 1280)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max_frames", type=int, default=1500,
                        help="Subsample above this many frames so a dense "
                             "trajectory does not become a 20-minute video "
                             "(default: 1500, 0 disables)")
    parser.add_argument("--skip_video", action="store_true",
                        help="Only write trajectory.png")
    args = parser.parse_args(argv)

    for dataset in args.dataset_path:
        render(dataset, args.width, args.fps, args.max_frames or None,
               args.skip_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
