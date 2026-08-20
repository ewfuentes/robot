"""Run a dataset end to end: frames -> tracks -> matched landmarks.

One command for the whole sequence the runbook lays out by hand. It exists
because the sequence is cheap: measured on boston_harbor_leg1, the entire LLM
bill is ~$26 at on-demand list price and about half that through the Batch API,
so there is no reason to stop at every stage waiting for a human when the
viewers can be read afterwards. Every stage still writes its viewer; nothing is
hidden, it is just not blocking.

    bazel run //experimental/overhead_matching/swag/landmark_filtering/object_tracking:run_pipeline -- \
        --dataset boston_harbor_leg2 --run_name r001_full

What it does NOT do is press on through a failure that would make everything
after it meaningless. Two conditions stop the run, because continuing past
either produces confident nonsense rather than an error:

- **an incomplete extraction** -- frames with no VLM response are read
  downstream as frames containing no objects, so tracks crossing them starve;
- **a mount-offset curve that is FLAT or MULTIMODAL** -- matching aims a bearing
  into the map, and an offset picked from noise aims it somewhere else entirely.

Everything else is advisory: it prints, records, and carries on.

Stages, in order (`--from` / `--to` / `--only` select a slice, and a stage whose
output already exists is skipped unless `--force`):

    extract   pinhole_images + frame_landmarks artifacts (Gemini, batch)
    boxes     m0 detection-box viewer  (geometry spot-check)
    tracks    m3 tracking + per-track viewer  (GPU, the long pole)
    keyframes per-keyframe detection viewer  (the pages track pages link into)
    audit     m5 semantic audit requests + execution
    review    m5 audit results viewer
    merge     m6 duplicate merge -> merged/measurements.json
    offset    mount-offset sweep from the merged bearings
    match     m9 tracklet-to-map matching
    matchview m10 match review viewer
    index     run_index landing pages

Model stages default to the Batch API (half price); `--online` swaps the whole
run back to on-demand calls.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.scripts import llm_cost

OT = "//experimental/overhead_matching/swag/landmark_filtering/object_tracking"
SCRIPTS = "//experimental/overhead_matching/swag/scripts"

# bazel run must be invoked from the source workspace, not the runfiles tree.
WORKSPACE = os.environ.get("BUILD_WORKSPACE_DIRECTORY")

STAGES = ("extract", "boxes", "tracks", "keyframes", "audit", "review", "merge",
          "offset", "match", "matchview", "index")

# Prices come from llm_cost.MODEL_RATES, keyed by the model that produced the
# responses -- a run that extracts on Flash and audits on another model bills at
# two different rates, and reporting both at one rate is simply a wrong number.
# Used only to report what a run cost, never to decide anything.


def run(cmd, description, dry_run=False, check=True):
    """Run one stage as a subprocess, streaming its output."""
    print(f"\n{'=' * 72}\n{description}\n{'=' * 72}")
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        print("  [DRY RUN] skipped")
        return 0
    started = time.time()
    result = subprocess.run([str(c) for c in cmd], cwd=WORKSPACE)
    elapsed = time.time() - started
    print(f"\n  {description}: exit {result.returncode} in {elapsed:.0f}s",
          flush=True)
    if check and result.returncode != 0:
        sys.exit(f"\n{description} failed; stopping.")
    return result.returncode


def token_cost(paths_to_results, model: str | None = None) -> dict:
    """Tally tokens and list-price cost from stored `usageMetadata`.

    Every response the pipeline stores carries its own usage block, so what a
    run cost is a measurement rather than an estimate. Reported at on-demand
    list price for `model`; halve it for the batch stages.
    """
    prompt = output = thinking = calls = 0
    for path in paths_to_results:
        if not Path(path).exists():
            continue
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                response = record.get("response")
                usage = (response or {}).get("usageMetadata") if isinstance(
                    response, dict) else None
                if not usage:
                    continue
                calls += 1
                prompt += usage.get("promptTokenCount", 0) or 0
                output += usage.get("candidatesTokenCount", 0) or 0
                thinking += usage.get("thoughtsTokenCount", 0) or 0
    out_tokens = output + thinking
    input_usd, output_usd, rate_label = llm_cost.rates_for(model)
    return {
        "calls": calls, "prompt_tokens": prompt, "output_tokens": out_tokens,
        "total_tokens": prompt + out_tokens,
        "usd_on_demand": round(prompt * input_usd["small"]
                               + out_tokens * output_usd["small"], 2),
        "rate_label": rate_label,
    }


def merge_costs(*costs) -> dict:
    """Sum per-model tallies into one, keeping each stage's rate visible."""
    merged = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
              "total_tokens": 0, "usd_on_demand": 0.0, "rates": []}
    for cost in costs:
        for key in ("calls", "prompt_tokens", "output_tokens", "total_tokens"):
            merged[key] += cost[key]
        merged["usd_on_demand"] += cost["usd_on_demand"]
        if cost["calls"]:
            merged["rates"].append(cost["rate_label"])
    merged["usd_on_demand"] = round(merged["usd_on_demand"], 2)
    return merged


# What each stage actually reads, mirroring the `require=` each stage's own main()
# declares. Keep in sync with those: this is the gate that runs first.
STAGE_REQUIRES = {
    "extract":   ("dataset_base", "panorama_dir"),
    "boxes":     ("dataset_base", "frame_landmarks"),
    "tracks":    ("dataset_base", "frame_landmarks", "video", "sam2_checkpoint"),
    "keyframes": ("dataset_base", "frame_landmarks"),
    "audit":     ("dataset_base", "frame_landmarks"),
    "review":    ("dataset_base",),
    "merge":     ("dataset_base",),
    "offset":    ("dataset_base",),
    "match":     ("dataset_base", "feather"),
    "matchview": ("dataset_base", "feather"),
    "index":     ("dataset_base",),
}


def stage_outputs(paths, run_dir: Path) -> dict:
    """The file whose existence means a stage has already run."""
    return {
        "extract": paths.frame_landmarks / "manifest.json",
        "boxes": paths.tracks_stage("m0_boxes") / "index.html",
        "tracks": run_dir / "run_meta.json",
        "keyframes": run_dir / "keyframes" / "index.html",
        "audit": run_dir / "semantic_audit" / "results.jsonl",
        "review": run_dir / "semantic_audit" / "review" / "index.html",
        "merge": run_dir / "merged" / "measurements.json",
        "offset": run_dir / "mount_offset_sweep.json",
        "match": run_dir / "matching" / "matches.json",
        "matchview": run_dir / "matching" / "review" / "index.html",
        "index": run_dir / "index.html",
    }


def check_extraction(paths, dry_run: bool):
    """Stop if any panorama lacks a usable VLM response.

    Not advisory: `ingest` skips a frame with no prediction silently, so an
    incomplete extraction reads downstream as a stretch of the leg containing no
    objects, and tracks crossing it starve. The extraction tool has the same
    gate; this repeats it because the orchestrator may be resuming into a
    pipeline whose extraction predates that gate.
    """
    if dry_run:
        return
    manifest = paths.frame_landmarks / "manifest.json"
    if not manifest.exists():
        return
    coverage = json.loads(manifest.read_text()).get("config", {})
    if coverage.get("complete") is False:
        n = coverage.get("n_no_usable_response", "?")
        sys.exit(
            f"\nSTOPPING: {paths.frame_landmarks} is an incomplete extraction "
            f"({n} panoramas with no usable response). Downstream would read "
            f"those frames as containing no objects. Repair with:\n"
            f"  bazel run {SCRIPTS}:extract_gemini_landmarks_from_panoramas -- "
            f"--dataset {paths.dataset} --retry_failed")


def check_offset(run_dir: Path, dry_run: bool):
    """Stop before matching if the offset curve does not support its minimum."""
    if dry_run:
        return
    sweep = run_dir / "mount_offset_sweep.json"
    if not sweep.exists():
        return
    record = json.loads(sweep.read_text())
    if not record.get("usable"):
        sys.exit(
            f"\nSTOPPING before matching: the mount-offset curve is "
            f"{record.get('verdict')} - {record.get('detail')}\n"
            f"Matching aims a bearing into the map, so an offset taken from "
            f"noise points it somewhere else entirely. Inspect "
            f"{sweep} and fix the bearings or poses first, or pass "
            f"--skip offset --skip match to go no further.")


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    farfield_paths.add_arguments(parser, video=True, checkpoint=True,
                                 feather=True)
    parser.add_argument("--run_name", default=None,
                        help="m3 run name (default: r001_<dataset>)")
    parser.add_argument("--notes", default="",
                        help="why this run exists; lands in run_meta.json")
    parser.add_argument("--from", dest="from_stage", choices=STAGES,
                        default=STAGES[0])
    parser.add_argument("--to", dest="to_stage", choices=STAGES,
                        default=STAGES[-1])
    parser.add_argument("--only", choices=STAGES, default=None,
                        help="run exactly one stage")
    parser.add_argument("--skip", action="append", default=[], choices=STAGES,
                        help="skip this stage (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="re-run stages whose output already exists")
    parser.add_argument("--online", action="store_true",
                        help="on-demand model calls instead of the Batch API: "
                             "returns in minutes rather than up to a day, at "
                             "twice the price")
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="model for the audit and matching stages")
    parser.add_argument("--extract_model", default="gemini-3.1-pro-preview",
                        help="model for landmark extraction")
    parser.add_argument("--pinhole_resolution", type=int, default=2048)
    parser.add_argument("--range", nargs=3, action="append", default=None,
                        metavar=("NAME", "K_START", "K_END"),
                        help="tracking range(s); default is the whole leg")
    parser.add_argument("--cost_limit", type=float, default=50.0,
                        help="refuse any single model step estimated above this "
                             "many USD until approved (default: 50)")
    parser.add_argument("--approve_cost", action="store_true",
                        help="approve steps that exceed --cost_limit")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the stage commands without running them")
    args = parser.parse_args()

    if args.only:
        selected = [args.only]
    else:
        lo, hi = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
        if lo > hi:
            parser.error(f"--from {args.from_stage} comes after --to "
                         f"{args.to_stage}")
        selected = [s for s in STAGES[lo:hi + 1] if s not in args.skip]

    # Require only what the selected stages read. Demanding the union of every
    # stage's inputs refuses work the machine can actually do: no tracking stage
    # touches the landmark catalog -- only `match` does -- so an alpine dataset
    # with a deliberately absent v1_trimmed.feather could not be tracked at all.
    needed = set()
    for stage in selected:
        needed |= set(STAGE_REQUIRES[stage])
    # A dataset with no source video can still be tracked: m3 falls back to
    # propagating across the keyframes themselves (KeyframeProvider), so the
    # video is best-effort rather than required.
    want_video = "video" in needed
    needed.discard("video")
    paths = farfield_paths.resolve(parser, args, require=tuple(sorted(needed)))
    keyframe_only = False
    if want_video:
        try:
            paths.video
        except farfield_paths.MissingInput:
            keyframe_only = True
    run_name = args.run_name or f"r001_{paths.dataset}"
    run_dir = paths.tracks_runs_root / run_name

    n_frames = len(list(paths.panorama_dir.glob("*.jpg")))
    ranges = args.range or [["full", "0", str(max(n_frames - 1, 0))]]
    transport = ["--online"] if args.online else []
    # Stages that resolve from --dataset need the version too; the --run_dir
    # stages get it from the run's recorded inputs instead.
    version = (["--frame_landmarks_version", args.frame_landmarks_version]
               if args.frame_landmarks_version else [])
    # Every model stage carries the ceiling, so no stage can spend past it.
    guard = ["--cost_limit", args.cost_limit]
    if args.approve_cost:
        guard.append("--approve_cost")

    print(f"dataset:   {paths.dataset} ({n_frames} frames)")
    print(f"run:       {run_dir}")
    print(f"stages:    {' -> '.join(selected)}")
    print(f"transport: {'on-demand (--online)' if args.online else 'Batch API'}")
    if keyframe_only:
        print("substrate: no source video - tracking propagates across "
              "keyframes only")

    outputs = stage_outputs(paths, run_dir)
    commands = {
        "extract": ["bazel", "run",
                    f"{SCRIPTS}:extract_gemini_landmarks_from_panoramas", "--",
                    "--dataset", paths.dataset,
                    "--prompt_type", "osm_tags_farfield",
                    "--pinhole_resolution", args.pinhole_resolution,
                    "--media_resolution", "MEDIA_RESOLUTION_ULTRA_HIGH",
                    "--model", args.extract_model, "--force"] + guard + version,
        "boxes": ["bazel", "run", f"{OT}:m0_render_boxes", "--",
                  "--dataset", paths.dataset] + version,
        "tracks": ["bazel", "run", f"{OT}:m3_track_viewer", "--",
                   "--dataset", paths.dataset, "--run_name", run_name,
                   "--notes", args.notes or f"run_pipeline {run_name}"]
                  + version + [a for r in ranges for a in ("--range", *r)],
        # Ground level of the viewer hierarchy: track pages and the audit
        # review link into keyframes/f####.html, so a run without this stage
        # ships dead links.
        "keyframes": ["bazel", "run", f"{OT}:keyframe_viewer", "--",
                      "--run_dir", run_dir],
        "audit": ["bazel", "run", f"{OT}:m5_build_audit_requests", "--",
                  "--run_dir", run_dir, "--submit", "--model", args.model]
                 + transport + guard,
        "review": ["bazel", "run", f"{OT}:m5_audit_results_viewer", "--",
                   "--run_dir", run_dir],
        "merge": ["bazel", "run", f"{OT}:m6_merge_tracks", "--",
                  "--run_dir", run_dir],
        # Corroborator; the sun check runs first as a pre-step above.
        "offset": ["bazel", "run", f"{OT}:mount_offset_sweep", "--",
                   "--run_dir", run_dir],
        "match": ["bazel", "run", f"{OT}:m9_match_landmarks", "--",
                  "--run_dir", run_dir, "--submit", "--model", args.model]
                 + transport + guard,
        "matchview": ["bazel", "run", f"{OT}:m10_match_viewer", "--",
                      "--run_dir", run_dir],
        "index": ["bazel", "run", f"{OT}:run_index", "--",
                  "--run_dir", run_dir],
    }

    started = time.time()
    ran, skipped = [], []
    for stage in selected:
        target = outputs[stage]
        if target.exists() and not args.force:
            print(f"\n-- {stage}: already done ({target}); --force to redo")
            skipped.append(stage)
            continue
        # Gates sit before the stage that would consume the bad input.
        if stage == "boxes":
            check_extraction(paths, args.dry_run)
        if stage == "offset":
            # The absolute estimate first, the relative one after. The sweep
            # only makes rays agree with each other, so it reproduces any error
            # the poses and the heading model share -- a 180 deg convention slip
            # fits it perfectly -- while the sun is checkable against
            # ephemeris. Not `check`ed: it abstains on overcast (all three
            # mount_washington legs), and abstaining is a correct outcome that
            # must not stop the pipeline. It refuses to overwrite an
            # already-validated offset on its own.
            run(["bazel", "run", f"{OT}:sun_offset_check", "--",
                 "--run_dir", run_dir, "--write_metadata"],
                "offset (sun, absolute)", dry_run=args.dry_run, check=False)
        if stage == "match":
            check_offset(run_dir, args.dry_run)
        run(commands[stage], f"{stage}", dry_run=args.dry_run)
        ran.append(stage)

    print(f"\n{'=' * 72}")
    print(f"pipeline finished in {(time.time() - started) / 60:.1f} min")
    print(f"  ran:     {', '.join(ran) or 'nothing'}")
    print(f"  skipped: {', '.join(skipped) or 'nothing'}")

    if not args.dry_run:
        cost = merge_costs(
            token_cost(list(paths.frame_landmarks.rglob("predictions.jsonl")),
                       args.extract_model),
            token_cost([run_dir / "semantic_audit" / "results.jsonl",
                        run_dir / "matching" / "results.jsonl"], args.model))
        actual = (cost["usd_on_demand"] if args.online
                  else cost["usd_on_demand"] / 2)
        print(f"\n  model calls: {cost['calls']}, "
              f"{cost['total_tokens']:,} tokens")
        print(f"  cost: ${cost['usd_on_demand']:.2f} at on-demand list price"
              + ("" if args.online else
                 f"; ~${actual:.2f} if the batch stages billed at half"))
        for label in dict.fromkeys(cost["rates"]):
            print(f"    priced at {label}")
        print("\n  viewers to read when convenient:")
        for label, page in (
                ("m0 boxes", paths.tracks_stage("m0_boxes") / "index.html"),
                ("m3 tracks", run_dir / "index.html"),
                ("keyframes", run_dir / "keyframes" / "index.html"),
                ("audit review",
                 run_dir / "semantic_audit" / "review" / "index.html"),
                ("merge", run_dir / "merged" / "index.html"),
                ("matching review",
                 run_dir / "matching" / "review" / "index.html")):
            if page.exists():
                print(f"    {label:<16} {page}")
        print(f"\n  serve with: cd {paths.object_tracks} && "
              f"python3 -m http.server 8935")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
