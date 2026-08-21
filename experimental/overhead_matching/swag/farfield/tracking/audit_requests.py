"""Build per-track semantic-audit VLM requests from a tracking run.

For every track with enough supports, assembles the dossier + chips defined
in semantic_audit.py and writes:

  <run_dir>/semantic_audit/requests.jsonl   Vertex requests (inline images)
  <run_dir>/semantic_audit/audit_meta.json  key -> track join info
  <run_dir>/semantic_audit/settings.json    the provenance record readers use
  <run_dir>/semantic_audit/manifest.json    standard farfield manifest
  <run_dir>/semantic_audit/chips/           chip JPEGs (also inlined)
  <run_dir>/semantic_audit/preview/index.html  eyeball prompts before spending

Every tracks_*.json in the run is consumed (a run may be split across
ranges), and each track's supports are classified under ITS artifact's
RECORDED TrackBuilderConfig -- never a fresh default.

Submit either with --submit (through vertex_batch_manager; needs
GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI) or
later by hand:

  bazel run //experimental/overhead_matching/swag/farfield/extraction:vertex_batch_manager -- \\
      run-online --input <run_dir>/semantic_audit/requests.jsonl \\
      --output <run_dir>/semantic_audit/results.jsonl --model <model>

Run:
  bazel run //experimental/overhead_matching/swag/farfield/tracking:audit_requests -- \\
      --run_dir <runs>/r003 [required flags...] [--submit]
"""

import argparse
import base64
import dataclasses
import hashlib
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    dataset,
    paths as paths_lib,
    provenance,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    vertex_batch_manager as vbm,
)
from experimental.overhead_matching.swag.farfield.tracking import (
    keyframe_viewer as kv,
    semantic_audit as sa,
)
from experimental.overhead_matching.swag.farfield.viewers import (
    page as page_lib,
)

GENERATOR = ("//experimental/overhead_matching/swag/farfield/tracking"
             ":audit_requests")

# Pages go through the one shared farfield.viewers.page helper (one CSS, a
# provenance footer on every page); only the classes specific to this viewer
# live here.
EXTRA_STYLE = """
pre{white-space:pre-wrap;font-size:12.5px;max-width:1100px}
.chip{display:inline-block;background:#222;margin:3px;padding:4px;
border-radius:4px;vertical-align:top;max-width:340px;font-size:12px}
.chip img{height:200px;display:block;border-radius:3px}
h2{margin-top:40px;border-top:1px solid #333;padding-top:16px}
"""


def render_all_chips(dossiers, frames_by_idx, dataset_base, chips_dir,
                     chip_height_px, fov_deg) -> dict:
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
                e, e["obs"], pano_w, pano_h, fov_deg)
            out = chips_dir / (f"T{d['track_id']}_t{e['t']:04d}"
                               f"{'_ctx' if e['is_context'] else ''}.jpg")
            sa.render_chip(pano, det_box, mask_box, out, chip_height_px)
            paths[(d["track_id"], e["t"], e["is_context"])] = out
    return paths


def write_preview(out_dir, dossiers, chip_paths, texts):
    preview = out_dir / "preview"
    preview.mkdir(exist_ok=True)
    parts = [
        "<p>Exact prompt text + images per request. System prompt shown "
        "once.</p>",
        "<h2>system prompt (all requests)</h2>",
        f"<pre>{html.escape(sa.SYSTEM_PROMPT)}</pre>"]
    for d in dossiers:
        # The id anchors the review page's "request preview" links.
        parts.append(f"<h2 id='T{d['track_id']}'>T{d['track_id']}</h2>")
        parts.append(f"<pre>{html.escape(texts[d['track_id']])}</pre>")
        for i, e in enumerate(d["chip_entries"], 1):
            path = chip_paths.get((d["track_id"], e["t"], e["is_context"]))
            if path is None:
                continue
            rel = f"../chips/{path.name}"
            caption = html.escape(sa.chip_caption(e, i))
            parts.append(f"<div class='chip'><img src='{rel}' "
                         f"loading='lazy'>{caption}</div>")
    (preview / "index.html").write_text(page_lib.page(
        f"semantic audit requests: {len(dossiers)} tracks", "\n".join(parts),
        generator=GENERATOR, extra_style=EXTRA_STYLE))
    return preview / "index.html"


def write_settings(out_dir, args, paths, artifacts, ingest_params,
                   audit_config: dict, n_tracks_total: int, n_eligible: int,
                   n_requests: int) -> None:
    """The provenance record for this audit (the old stage's P0 hole:
    audit_meta.json recorded nothing about HOW the requests were built --
    no model, no thresholds, no prompt identity).

    Follows the matching stage's settings.json pattern: a sibling
    settings.json readers consume (audit_review classifies supports under
    the values recorded here, never a fresh default), plus the standard
    provenance manifest. The prompt is hashed rather than named because
    SYSTEM_PROMPT is a module constant that can be edited in place.
    """
    settings = {
        "generator": GENERATOR,
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "dataset": paths.dataset,
        "run": args.run_dir.name,
        "model": args.model,
        "transport": "online" if args.online else "batch",
        "submitted_by_stage": bool(args.submit),
        "thinking_level": args.thinking_level,
        "min_supports": args.min_supports,
        "system_prompt_sha256": hashlib.sha256(
            sa.SYSTEM_PROMPT.encode()).hexdigest(),
        # AuditConfig minus `classifier`: the classifier is not one value --
        # each range's supports are classified under that range's RECORDED
        # TrackBuilderConfig, echoed below.
        "audit_config": audit_config,
        "classifier_by_range": {rn: artifacts[rn]["config"]
                                for rn in artifacts},
        "ingest": dataclasses.asdict(ingest_params),
        "tracks_files": sorted(
            p.name for p in args.run_dir.glob("tracks_*.json")),
        "n_tracks_total": n_tracks_total,
        "n_eligible": n_eligible,
        "n_requests": n_requests,
    }
    (out_dir / "settings.json").write_text(json.dumps(settings, indent=1)
                                           + "\n")
    print(f"wrote {out_dir}/settings.json")
    provenance.write(
        out_dir,
        generator=GENERATOR,
        inputs={"run_dir": args.run_dir,
                "dataset_base": paths.dataset_base,
                "frame_landmarks": paths.frame_landmarks},
        config=settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths_lib.add_arguments(parser)
    parser.add_argument("--run_dir", type=Path, required=True)
    # Result-shaping values are required (REORG.md rule 2: no stale defaults
    # on assumption-carrying args). Previous defaults are quoted for
    # reference, not authority.
    # TODO(REORG.md PR 12): these move into the run's recorded config
    # (run_config.json); CLI-required until then.
    parser.add_argument("--min_supports", type=int, required=True,
                        help="Supports required to audit a track; tracks "
                             "below this are dropped from the pipeline "
                             "entirely - un-audited tracks are not carried "
                             "forward (previously 2, i.e. 3 detections "
                             "counting the birth)")
    parser.add_argument("--thinking_level", required=True,
                        help="Gemini thinkingLevel for every request "
                             "(previously HIGH)")
    parser.add_argument("--max_support_chips", type=int, required=True,
                        help="Support chips rendered per request "
                             "(previously 6)")
    parser.add_argument("--max_context_chips", type=int, required=True,
                        help="Context chips rendered per request "
                             "(previously 2)")
    parser.add_argument("--max_description_samples", type=int, required=True,
                        help="Verbatim detection descriptions quoted per "
                             "dossier (previously 10)")
    parser.add_argument("--chip_height_px", type=int, required=True,
                        help="Rendered chip height in pixels (previously "
                             "320)")
    parser.add_argument("--max_tracks", type=int, default=None,
                        help="Cap request count (debugging)")
    # TODO(run_config): the ingest values move into the run's recorded config
    # (run_config.json, REORG.md PR 12); required on the CLI until then. No
    # defaults on purpose -- these shape which detections exist at all.
    parser.add_argument("--fov_deg", type=float, required=True,
                        help="Pinhole-face FOV the extraction rendered with, "
                             "degrees (previously 90.0 on all datasets)")
    parser.add_argument("--seam_gap_norm", type=float, required=True,
                        help="Seam-merge margin in bbox units 0-1000 "
                             "(previously 25)")
    parser.add_argument("--seam_min_y_iou", type=float, required=True,
                        help="Vertical IoU to accept a seam continuation "
                             "(previously 0.3)")
    vbm.add_execution_arguments(parser)
    parser.add_argument("--submit", action="store_true",
                        help="Run the requests through Vertex now")
    args = parser.parse_args()
    paths = paths_lib.resolve(
        parser, args, infer_from=args.run_dir,
        require=("dataset_base", "frame_landmarks"))

    # EVERY tracks_*.json in the run (the old stage took next(glob) -- the
    # first range only), each classified under its own recorded config.
    artifacts = kv.load_track_artifacts(args.run_dir)
    _, range_by_track = sa.merge_tracks(artifacts)
    cfg_by_range = {
        rn: sa.AuditConfig(
            min_supports=args.min_supports,
            max_support_chips=args.max_support_chips,
            max_context_chips=args.max_context_chips,
            max_description_samples=args.max_description_samples,
            chip_height_px=args.chip_height_px,
            thinking_level=args.thinking_level,
            classifier=kv.recorded_config(artifact))
        for rn, artifact in artifacts.items()}

    ingest_params = dataset.IngestParams(
        fov_deg=args.fov_deg, seam_gap_norm=args.seam_gap_norm,
        seam_min_y_iou=args.seam_min_y_iou)
    result = dataset.run_ingest(paths.dataset_base, paths.frame_landmarks,
                                ingest_params)
    obs_by_id = {o.obs_id: o for o in result.observations}
    frames_by_idx = {f.frame_idx: f for f in result.frames}

    dossiers = []
    n_tracks_total = 0
    for range_name, artifact in artifacts.items():
        cfg = cfg_by_range[range_name]
        for track in artifact["tracks"]:
            n_tracks_total += 1
            if not track["records"]:
                continue
            d = sa.build_dossier(track, obs_by_id, cfg)
            if d["n_supports"] >= cfg.min_supports:
                dossiers.append(d)
    dossiers.sort(key=lambda d: -d["n_supports"])
    n_eligible = len(dossiers)
    if args.max_tracks:
        dossiers = dossiers[:args.max_tracks]
    print(f"{n_eligible} eligible tracks "
          f"(>= {args.min_supports} supports) of {n_tracks_total} total")

    out_dir = args.run_dir / "semantic_audit"
    out_dir.mkdir(exist_ok=True)
    chip_paths = render_all_chips(dossiers, frames_by_idx, paths.dataset_base,
                                  out_dir / "chips", args.chip_height_px,
                                  args.fov_deg)
    print(f"rendered {len(chip_paths)} chips")

    requests_path = out_dir / "requests.jsonl"
    meta = {}
    texts = {}
    with open(requests_path, "w") as f:
        for d in dossiers:
            range_name = range_by_track[d["track_id"]]
            cfg = cfg_by_range[range_name]
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
                "range": range_name,
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

    audit_config = {k: v for k, v in dataclasses.asdict(
        next(iter(cfg_by_range.values()))).items() if k != "classifier"}
    write_settings(out_dir, args, paths, artifacts, ingest_params,
                   audit_config, n_tracks_total, n_eligible, len(dossiers))

    preview_path = write_preview(out_dir, dossiers, chip_paths, texts)
    print(f"preview: {preview_path}")

    if args.submit:
        vbm.run_requests(args, requests_path, out_dir / "results.jsonl",
                         tag=f"{paths.dataset}_audit_{args.run_dir.name}")
    else:
        transport = "run-online" if args.online else "run-batch"
        print(f"\nto submit ({transport}):")
        print("  bazel run //experimental/overhead_matching/swag/farfield/"
              f"extraction:vertex_batch_manager -- {transport} \\")
        print(f"      --input {requests_path} \\")
        print(f"      --output {out_dir / 'results.jsonl'} \\")
        print(f"      --model {args.model}")


if __name__ == "__main__":
    main()
