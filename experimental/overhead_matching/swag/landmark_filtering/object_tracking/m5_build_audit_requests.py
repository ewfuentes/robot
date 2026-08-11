"""Build per-track semantic-audit VLM requests from a tracking run.

For every track with enough supports, assembles the dossier + chips defined
in semantic_audit.py and writes:

  <run_dir>/semantic_audit/requests.jsonl   Vertex requests (inline images)
  <run_dir>/semantic_audit/audit_meta.json  key -> track join info
  <run_dir>/semantic_audit/chips/           chip JPEGs (also inlined)
  <run_dir>/semantic_audit/preview/index.html  eyeball prompts before spending

Submit either with --submit (runs the requests online through Vertex via
vertex_batch_manager's run-online path; needs GOOGLE_CLOUD_PROJECT,
GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI) or later by hand:

  bazel run //experimental/overhead_matching/swag/scripts:vertex_batch_manager -- \\
      run-online --input <run_dir>/semantic_audit/requests.jsonl \\
      --output <run_dir>/semantic_audit/results.jsonl --model <model>

Run:
  bazel run //...object_tracking:m5_build_audit_requests -- \\
      --run_dir <runs>/r002_full_leg1 [--submit]
"""

import argparse
import base64
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.landmark_filtering import ingest
from experimental.overhead_matching.swag.landmark_filtering.object_tracking import (
    semantic_audit as sa,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    IngestConfig,
)
from experimental.overhead_matching.swag.scripts import vertex_batch_manager as vbm

DEFAULT_DATASET = Path("/data/farfield_matching/boston_harbor_dataset/processed/leg1")
DEFAULT_LANDMARKS = Path(
    "/data/farfield_matching/boston_harbor_dataset/panorama_landmarks/boston_harbor_leg1")


def render_all_chips(dossiers, frames_by_idx, dataset_base, chips_dir,
                     cfg) -> dict:
    """Render every selected chip, decoding each pano once.
    Returns {(track_id, t, is_context): chip_path}."""
    chips_dir.mkdir(parents=True, exist_ok=True)
    by_keyframe = defaultdict(list)
    for d in dossiers:
        for e in d["chip_entries"]:
            by_keyframe[e["keyframe"]].append((d, e))

    probe = Image.open(
        dataset_base / "panorama"
        / f"{frames_by_idx[min(frames_by_idx)].pano_stem}.jpg")
    pano_w, pano_h = probe.size

    paths = {}
    for keyframe in sorted(by_keyframe):
        frame = frames_by_idx.get(keyframe)
        if frame is None:
            continue
        pano = np.asarray(Image.open(
            dataset_base / "panorama" / f"{frame.pano_stem}.jpg"))
        for d, e in by_keyframe[keyframe]:
            det_box, mask_box = sa.chip_boxes_for_entry(
                e, e["obs"], pano_w, pano_h)
            out = chips_dir / (f"T{d['track_id']}_t{e['t']:04d}"
                               f"{'_ctx' if e['is_context'] else ''}.jpg")
            sa.render_chip(pano, det_box, mask_box, out, cfg.chip_height_px)
            paths[(d["track_id"], e["t"], e["is_context"])] = out
    return paths


def write_preview(out_dir, dossiers, chip_paths, texts):
    preview = out_dir / "preview"
    preview.mkdir(exist_ok=True)
    parts = [
        "<html><head><title>semantic audit request preview</title><style>",
        "body{font-family:sans-serif;background:#161616;color:#ddd;margin:16px}",
        "pre{background:#1d1d1d;padding:12px;border-radius:6px;",
        "white-space:pre-wrap;font-size:12.5px;max-width:1100px}",
        ".chip{display:inline-block;background:#222;margin:3px;padding:4px;",
        "border-radius:4px;vertical-align:top;max-width:340px;font-size:12px}",
        ".chip img{height:200px;display:block;border-radius:3px}",
        "h2{margin-top:40px;border-top:1px solid #333;padding-top:16px}",
        "</style></head><body>",
        f"<h1>semantic audit requests: {len(dossiers)} tracks</h1>",
        "<p>Exact prompt text + images per request. System prompt shown "
        "once.</p>",
        "<h2>system prompt (all requests)</h2>",
        f"<pre>{html.escape(sa.SYSTEM_PROMPT)}</pre>"]
    for d in dossiers:
        parts.append(f"<h2>T{d['track_id']}</h2>")
        parts.append(f"<pre>{html.escape(texts[d['track_id']])}</pre>")
        for i, e in enumerate(d["chip_entries"], 1):
            path = chip_paths.get((d["track_id"], e["t"], e["is_context"]))
            if path is None:
                continue
            rel = f"../chips/{path.name}"
            caption = html.escape(sa.chip_caption(e, i))
            parts.append(f"<div class='chip'><img src='{rel}' "
                         f"loading='lazy'>{caption}</div>")
    parts.append("</body></html>")
    (preview / "index.html").write_text("\n".join(parts))
    return preview / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset_base", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmark_base", type=Path, default=DEFAULT_LANDMARKS)
    parser.add_argument("--min_supports", type=int, default=3)
    parser.add_argument("--max_tracks", type=int, default=None,
                        help="Cap request count (debugging)")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--submit", action="store_true",
                        help="Run the requests through Vertex now (online)")
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    cfg = sa.AuditConfig(min_supports=args.min_supports)
    artifact = json.loads(next(args.run_dir.glob("tracks_*.json")).read_text())
    result = ingest.run_ingest(args.dataset_base, args.landmark_base,
                               IngestConfig())
    obs_by_id = {o.obs_id: o for o in result.observations}
    frames_by_idx = {f.frame_idx: f for f in result.frames}

    dossiers = []
    for track in artifact["tracks"]:
        if not track["records"]:
            continue
        d = sa.build_dossier(track, obs_by_id, cfg)
        if d["n_supports"] >= cfg.min_supports:
            dossiers.append(d)
    dossiers.sort(key=lambda d: -d["n_supports"])
    if args.max_tracks:
        dossiers = dossiers[:args.max_tracks]
    print(f"{len(dossiers)} eligible tracks "
          f"(>= {cfg.min_supports} supports) of "
          f"{len(artifact['tracks'])} total")

    out_dir = args.run_dir / "semantic_audit"
    out_dir.mkdir(exist_ok=True)
    chip_paths = render_all_chips(dossiers, frames_by_idx, args.dataset_base,
                                  out_dir / "chips", cfg)
    print(f"rendered {len(chip_paths)} chips")

    requests_path = out_dir / "requests.jsonl"
    meta = {}
    texts = {}
    with open(requests_path, "w") as f:
        for d in dossiers:
            text = sa.render_dossier_text(d)
            texts[d["track_id"]] = text
            chips = []
            for i, e in enumerate(d["chip_entries"], 1):
                path = chip_paths.get((d["track_id"], e["t"], e["is_context"]))
                if path is None:
                    continue
                b64 = base64.b64encode(path.read_bytes()).decode()
                chips.append((sa.chip_caption(e, i), b64))
            key = f"T{d['track_id']}"
            f.write(json.dumps(sa.build_request(key, text, chips, cfg)) + "\n")
            meta[key] = {
                "track_id": d["track_id"],
                "birth_keyframe": d["birth_keyframe"],
                "n_supports": d["n_supports"],
                "support_obs_by_t": {
                    e["t"]: e["obs"].obs_id for e in d["supports"]},
                "chips": [str(chip_paths[(d["track_id"], e["t"],
                                          e["is_context"])])
                          for e in d["chip_entries"]
                          if (d["track_id"], e["t"], e["is_context"])
                          in chip_paths],
            }
    (out_dir / "audit_meta.json").write_text(json.dumps(meta, indent=1))
    size_mb = requests_path.stat().st_size / 1e6
    print(f"wrote {requests_path} ({size_mb:.1f} MB)")

    preview_path = write_preview(out_dir, dossiers, chip_paths, texts)
    print(f"preview: {preview_path}")

    if args.submit:
        vbm.cmd_run_online(argparse.Namespace(
            input=str(requests_path),
            output=str(out_dir / "results.jsonl"),
            model=args.model, parallel=args.parallel))
    else:
        print("\nto submit:")
        print("  bazel run //experimental/overhead_matching/swag/scripts:"
              "vertex_batch_manager -- run-online \\")
        print(f"      --input {requests_path} \\")
        print(f"      --output {out_dir / 'results.jsonl'} \\")
        print(f"      --model {args.model} --parallel {args.parallel}")


if __name__ == "__main__":
    main()
