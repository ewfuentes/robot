"""M3: build mask-anchored tracks over keyframe ranges and render a track
board per range for inspection. The richer per-track viewer (videos,
evidence tables, timeline) is m3_track_viewer.

Run (all three test ranges by default):
  bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:m3_build_tracks
"""

import argparse
import html
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    range_runner as rr,
    track_builder as tb,
    viz_common as vc,
)

STAGE_DIR = "m3_tracks"

# Short dev ranges chosen on boston_harbor_leg1 for iteration speed. This
# board tool is their only home: the production stage (m3_track_viewer)
# defaults to the full leg, and run_pipeline always passes --range full.
# Each fixture exercises a known-hard case (departure = pure rotation,
# port = association ambiguity in a crane cluster, fort = a huge object
# outgrowing its window). They are keyframe indices, so they mean nothing in
# particular on another leg; without --range they are clamped to whatever
# that dataset actually has.
LEG1_DEV_RANGES = [("f0000_departure", 0, 30), ("f0122_port", 114, 144),
                   ("f0149_fort", 141, 171)]


def clamp_ranges(ranges, last_keyframe):
    """Drop or shorten dev ranges that run past the end of a dataset."""
    kept = []
    for name, k_start, k_end in ranges:
        if k_start > last_keyframe:
            print(f"skipping range {name}: starts past f{last_keyframe:04d}")
            continue
        if k_end > last_keyframe:
            print(f"clamping range {name} to f{last_keyframe:04d}")
            k_end = last_keyframe
        kept.append((name, k_start, k_end))
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    farfield_paths.add_arguments(parser, video=True, checkpoint=True)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help=f"default: <object_tracks artifact>/{STAGE_DIR}")
    parser.add_argument("--range", nargs=3, action="append", default=None,
                        metavar=("NAME", "K_START", "K_END"),
                        help="override default keyframe ranges")
    args = parser.parse_args()

    paths = farfield_paths.resolve(
        parser, args,
        require=("dataset_base", "frame_landmarks", "sam2_checkpoint"))
    # No source video is a mode, not an error (see m3_track_viewer).
    try:
        video = paths.video
    except farfield_paths.MissingInput:
        video = None
        print(f"{paths.dataset}: no source video in metadata; tracking "
              f"across keyframes only")
    ctx = rr.load_context(paths.dataset_base, paths.frame_landmarks,
                          video, paths.sam2_checkpoint)
    if args.range:
        ranges = [(n, int(a), int(b)) for n, a, b in args.range]
    else:
        print("no --range given: running the boston_harbor_leg1 dev fixtures "
              "(clamped to this dataset); pass --range for real coverage")
        ranges = clamp_ranges(LEG1_DEV_RANGES,
                              max(f.frame_idx for f in ctx["result"].frames))
        if not ranges:
            parser.error("no default range fits this dataset; pass --range "
                         "NAME K_START K_END")
    font = vc.load_font(13)
    builder_cfg = tb.TrackBuilderConfig()
    out = args.output_dir or paths.tracks_stage(STAGE_DIR)
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
            paths.dataset_base, renderer=renderer)
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
