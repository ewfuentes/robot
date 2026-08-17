"""M3: build mask-anchored tracks over keyframe ranges and render a track
board per range for inspection. The richer per-track viewer (videos,
evidence tables, timeline) is m3_track_viewer.

Run (all three test ranges by default):
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m3_build_tracks
"""

import argparse
import html
from pathlib import Path

from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    range_runner as rr,
    track_builder as tb,
    viz_common as vc,
)

DEFAULT_DATASET = Path("/data/farfield_matching/datasets/boston_harbor_leg1")
DEFAULT_LANDMARKS = Path(
    "/data/farfield_matching/artifacts/frame_landmarks/boston_harbor_leg1/v1")
DEFAULT_VIDEO = Path(
    "/data/farfield_matching/raw_material/boston_harbor_20260712/videos/long_wharf_to_hull_wharf.mp4")
DEFAULT_OUTPUT = Path(
    "/data/farfield_matching/artifacts/object_tracks/boston_harbor_leg1/v1/m3_tracks")
DEFAULT_CHECKPOINT = Path(
    "/data/farfield_matching/models/sam2/sam2.1_hiera_large.pt")

DEFAULT_RANGES = [("f0000_departure", 0, 30), ("f0122_port", 114, 144),
                  ("f0149_fort", 141, 171)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmark_base", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--range", nargs=3, action="append", default=None,
                        metavar=("NAME", "K_START", "K_END"),
                        help="override default keyframe ranges")
    args = parser.parse_args()

    ranges = ([(n, int(a), int(b)) for n, a, b in args.range]
              if args.range else DEFAULT_RANGES)
    ctx = rr.load_context(args.dataset_base, args.landmark_base, args.video,
                          args.checkpoint)
    font = vc.load_font(13)
    builder_cfg = tb.TrackBuilderConfig()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    html_parts = [
        "<html><head><title>M3: track boards</title>",
        "<style>body{font-family:sans-serif;background:#181818;color:#ddd}"
        "img{display:block;margin:4px 0;max-width:100%}"
        "td,th{padding:2px 8px;text-align:left}</style></head><body>",
        "<h1>M3 track boards</h1>",
        "<p>border: green=birth/reanchor, blue=mask continue, "
        "orange=unsupported, red=closed at this keyframe</p>"]
    for range_name, k_start, k_end in ranges:
        print(f"range {range_name}: f{k_start:04d}..f{k_end:04d}")
        renderer = rr.BoardRenderer(font)
        builder, artifact = rr.run_range(
            range_name, k_start, k_end, builder_cfg, ctx["backend"],
            ctx["provider"], ctx["model"], ctx["result"], ctx["obs_by_frame"],
            ctx["det_pano_boxes"], ctx["pano_w"], ctx["pano_h"],
            args.dataset_base, renderer=renderer)
        board = renderer.compose(builder, k_start, k_end, font)
        board_rel = f"board_{range_name}.jpg"
        board.save(out / board_rel, quality=88)
        rr.write_artifact(artifact, out, range_name)
        html_parts.append(f"<h2>{html.escape(range_name)}</h2>")
        html_parts.append(f"<img src='{board_rel}' loading='lazy'>")
        html_parts.append("<table><tr><th>track</th><th>label</th>"
                          "<th>born</th><th>status</th><th>supported kf</th>"
                          "<th>span</th></tr>")
        for t in artifact["tracks"]:
            span = (f"f{t['birth_keyframe']:04d}..f{t['end_keyframe']:04d}"
                    if t["end_keyframe"] is not None else "-")
            html_parts.append(
                f"<tr><td>T{t['track_id']}</td>"
                f"<td>{html.escape(t['modal_label'])}</td>"
                f"<td>f{t['birth_keyframe']:04d}</td>"
                f"<td>{t['status']} {t['close_reason']}</td>"
                f"<td>{t['n_supported_keyframes']}</td><td>{span}</td></tr>")
        html_parts.append("</table>")
    html_parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(html_parts))
    print(f"wrote {out}/index.html")


if __name__ == "__main__":
    main()
