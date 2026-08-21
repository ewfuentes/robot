"""Run a dataset end to end: frames -> tracks -> matches -> localization.

One command for the whole sequence, in two steps:

  # 1. Create a run: validates the config (every result-shaping value
  #    explicit — see configs/harbor_example.yaml) and records it.
  bazel run //experimental/overhead_matching/swag/farfield:pipeline -- new-run \\
      --dataset boston_harbor_leg2 --run_name r001 \\
      --config $PWD/experimental/overhead_matching/swag/farfield/configs/harbor_example.yaml

  # 2. Execute stages. Parameters come from the run's recorded config;
  #    the only knobs here select WHICH stages run.
  bazel run //...farfield:pipeline -- run --run_dir <object_tracks>/runs/r001
  bazel run //...farfield:pipeline -- run --run_dir ... --from offset
  bazel run //...farfield:pipeline -- status --run_dir ...

The sequence is cheap (a full harbor leg bills ~$26 of LLM at list price),
so nothing blocks on a human: every stage writes its viewer, and the index
chain refreshes after every stage so the data root stays fully browsable.

Two conditions stop the run, because continuing past either produces
confident nonsense rather than an error:

- an INCOMPLETE EXTRACTION — frames with no VLM response are read downstream
  as frames containing no objects, so tracks crossing them starve. Checked
  before ANY detection-consuming stage (the old orchestrator only checked it
  on one early stage, so --from bypassed the gate).
- NO USABLE MOUNT OFFSET at export time — the export bakes the offset into
  every bearing; with no validated dataset record and no usable sidecar,
  exporting would aim every bearing somewhere else entirely.

A stage whose completion marker exists is skipped (--force to redo). The
tracking stage's marker is tracks_complete.json covering every declared
range — written per range AFTER it finishes, so a mid-tracking crash resumes
instead of silently skipping the GPU stage (the old marker was written
before the work started).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    paths as paths_lib,
    run_config,
)
from experimental.overhead_matching.swag.farfield.extraction import llm_cost
from experimental.overhead_matching.swag.farfield.viewers import indexes

FF = "//experimental/overhead_matching/swag/farfield"

# bazel run must be invoked from the source workspace, not the runfiles tree.
WORKSPACE = os.environ.get("BUILD_WORKSPACE_DIRECTORY")

STAGES = ("extract", "track", "keyframes", "audit", "review", "offset",
          "match", "matchview", "export", "localize", "plots", "viewer")

# Stages that read the VLM detections and therefore sit behind the
# extraction-completeness gate.
DETECTION_CONSUMERS = ("track", "keyframes", "audit", "review", "offset",
                       "match", "matchview", "export")

# Every result-shaping value a run must record before any stage runs
# (run_config.create validates the whole list at once).
REQUIRED_CONFIG = (
    "experiment.name",
    "artifacts.frame_landmarks_version",
    "artifacts.pinhole_images_version",
    "artifacts.object_tracks_version",
    "catalog.name",
    "extraction.model",
    "extraction.prompt_type",
    "extraction.pinhole_resolution",
    "extraction.media_resolution",
    "extraction.thinking_level",
    "ingest.fov_deg",
    "ingest.seam_gap_norm",
    "ingest.seam_min_y_iou",
    "tracking.sam2_checkpoint",
    "audit.model",
    "audit.min_supports",
    "audit.thinking_level",
    "audit.max_support_chips",
    "audit.max_context_chips",
    "audit.max_description_samples",
    "audit.chip_height_px",
    "matching.model",
    "matching.query_batch",
    "matching.chunk_size",
    "matching.thinking_level",
    "matching.confidence_floor",
    "matching.instance_max_rows",
    "calibration.sun.n_frames",
    "calibration.sun.min_speed_mps",
    "calibration.sun.elevation_tolerance_deg",
    "calibration.sun.work_width",
    "calibration.sweep.coarse_step",
    "calibration.sweep.fine_step",
    "calibration.sweep.fine_halfwidth",
    "calibration.sweep.min_observations",
    "calibration.sweep.min_arc_deg",
    "calibration.sweep.max_condition",
    "calibration.sweep.min_tracklets",
    "calibration.sweep.min_support_frac",
    "fusion.epoch_keyframes",
    "fusion.bearing_sigma_deg",
    "export.default_log_lr",
    "export.clip",
    "export.min_step_m",
    "export.sigma_pair_m",
    "export.max_visible_range_m",
    "localization.n_particles",
    "localization.margin_m",
    "localization.position_roughening_m",
    "localization.heading_roughening_deg",
    "cost.limit_usd",
)


def run(cmd, description, dry_run=False, check=True):
    """Run one stage as a subprocess, streaming its output."""
    print(f"\n{'=' * 72}\n{description}\n{'=' * 72}")
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        print("  [DRY RUN] skipped")
        return 0
    if WORKSPACE is None:
        sys.exit("BUILD_WORKSPACE_DIRECTORY is unset: run the pipeline via "
                 "`bazel run`, not directly from the runfiles tree.")
    started = time.time()
    result = subprocess.run([str(c) for c in cmd], cwd=WORKSPACE)
    elapsed = time.time() - started
    print(f"\n  {description}: exit {result.returncode} in {elapsed:.0f}s",
          flush=True)
    if check and result.returncode != 0:
        sys.exit(f"\n{description} failed; stopping.")
    return result.returncode


def token_cost(paths_to_results, model: str) -> dict:
    """Tally tokens and list-price cost from stored `usageMetadata`.

    Every response the pipeline stores carries its own usage block, so what
    a run cost is a measurement rather than an estimate.
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
        "calls": calls, "prompt_tokens": prompt,
        "output_tokens": out_tokens, "total_tokens": prompt + out_tokens,
        "usd_on_demand": round(prompt * input_usd["small"]
                               + out_tokens * output_usd["small"], 2),
        "rate_label": rate_label,
    }


def stage_done(stage: str, paths, run_dir: Path, loc_run: Path) -> bool:
    """Whether a stage's completion marker exists."""
    if stage == "extract":
        manifest = paths.frame_landmarks / "manifest.json"
        if not manifest.exists():
            return False
        return json.loads(manifest.read_text()).get(
            "config", {}).get("complete", False) is not False
    if stage == "track":
        # Import here: pulls torch transitively, and only this marker needs it.
        from experimental.overhead_matching.swag.farfield.tracking import (
            run_tracking,
        )
        return (run_dir / "tracks_complete.json").exists() and \
            not run_tracking.unfinished_ranges(run_dir)
    markers = {
        "keyframes": run_dir / "keyframes" / "index.html",
        "audit": run_dir / "semantic_audit" / "results.jsonl",
        "review": run_dir / "semantic_audit" / "review" / "index.html",
        "offset": run_dir / "mount_offset_sweep.json",
        "match": run_dir / "matching" / "matches.json",
        "matchview": run_dir / "matching" / "review" / "index.html",
        "export": run_dir / "localization_export" / "export_meta.json",
        "localize": loc_run / "manifest.json",
        "plots": loc_run / "plots" / "map.png",
        "viewer": loc_run / "viewer.html",
    }
    return markers[stage].exists()


def check_extraction(paths, dry_run: bool):
    """Stop if any panorama lacks a usable VLM response (see module doc)."""
    if dry_run:
        return
    manifest = paths.frame_landmarks / "manifest.json"
    if not manifest.exists():
        sys.exit(f"\nSTOPPING: no extraction manifest at {manifest}; run the "
                 f"extract stage first.")
    coverage = json.loads(manifest.read_text()).get("config", {})
    if coverage.get("complete") is False:
        n = coverage.get("n_no_usable_response", "?")
        sys.exit(
            f"\nSTOPPING: {paths.frame_landmarks} is an incomplete "
            f"extraction ({n} panoramas with no usable response). Downstream "
            f"would read those frames as containing no objects. Repair with "
            f"the extract stage's --retry_failed (never --force: that "
            f"re-bills the whole extraction).")


def check_offset(run_dir: Path, dry_run: bool):
    """Stop before the export if no usable offset exists anywhere."""
    if dry_run:
        return
    for name in ("sun_offset_check.json", "mount_offset_sweep.json"):
        path = run_dir / name
        if path.exists() and json.loads(path.read_text()).get("usable"):
            return
    # build_export itself falls back to a validated dataset record; let it
    # decide — but warn that both run-local estimates abstained.
    print("\n  NOTE: neither offset sidecar is usable; the export will "
          "accept only a validated dataset record or --mount_offset_deg.")


def cmd_new_run(args):
    paths = paths_lib.from_args(args)
    try:
        import yaml
        config = yaml.safe_load(Path(args.config).read_text())
    except FileNotFoundError:
        sys.exit(f"config file {args.config} not found")
    run_dir = paths.tracks_runs_root / args.run_name
    path = run_config.create(
        run_dir, config, required=REQUIRED_CONFIG,
        generator="farfield.pipeline new-run",
        inputs={"dataset_base": paths.dataset_base,
                "config_file": str(Path(args.config).resolve())},
        notes=args.notes)
    print(f"run created: {run_dir}")
    print(f"config recorded: {path}")
    print(f"next: bazel run {FF}:pipeline -- run --run_dir {run_dir}")


def resolve_run(parser, args):
    """(paths, run_dir, config doc, loc_run dir) for an existing run."""
    run_dir = Path(args.run_dir)
    doc = run_config.load(run_dir)  # pointed error if not a run
    paths = paths_lib.resolve(parser, args, infer_from=run_dir)
    cfg = doc["config"]
    # Version/catalog resolution comes from the recorded config.
    paths.versions.setdefault(
        paths_lib.FRAME_LANDMARKS, cfg["artifacts"]["frame_landmarks_version"])
    paths.versions.setdefault(
        paths_lib.PINHOLE_IMAGES, cfg["artifacts"]["pinhole_images_version"])
    if paths.catalog is None:
        paths.catalog = cfg["catalog"]["name"]
    experiment = paths.experiment_dir(cfg["experiment"]["name"])
    loc_run = experiment / f"{paths.dataset}_{run_dir.name}"
    return paths, run_dir, doc, loc_run


def build_commands(paths, run_dir: Path, cfg: dict, loc_run: Path,
                   args) -> dict:
    v = lambda key: run_config.value({"config": cfg}, key)  # noqa: E731
    ingest = ["--fov_deg", v("ingest.fov_deg"),
              "--seam_gap_norm", v("ingest.seam_gap_norm"),
              "--seam_min_y_iou", v("ingest.seam_min_y_iou")]
    guard = ["--cost_limit", v("cost.limit_usd")]
    if args.approve_cost:
        guard.append("--approve_cost")
    transport = ["--online"] if args.online else []
    fusion = ["--epoch_keyframes", v("fusion.epoch_keyframes"),
              "--bearing_sigma_deg", v("fusion.bearing_sigma_deg")]
    export_dir = run_dir / "localization_export"
    tables = (run_dir / "matching" / "compatibility.json"
              if not args.uninformative_tables else "uninformative")

    return {
        "extract": ["bazel", "run", f"{FF}/extraction:extract_landmarks",
                    "--", "--dataset", paths.dataset,
                    "--prompt_type", v("extraction.prompt_type"),
                    "--pinhole_resolution", v("extraction.pinhole_resolution"),
                    "--media_resolution", v("extraction.media_resolution"),
                    "--model", v("extraction.model"),
                    "--thinking_level", v("extraction.thinking_level"),
                    "--frame_landmarks_version",
                    v("artifacts.frame_landmarks_version"),
                    "--pinhole_version",
                    v("artifacts.pinhole_images_version"),
                    ] + guard + transport,
        "track": ["bazel", "run", f"{FF}/tracking:run_tracking", "--",
                  "--dataset", paths.dataset,
                  "--frame_landmarks_version",
                  v("artifacts.frame_landmarks_version"),
                  "--run_name", run_dir.name,
                  "--runs_root", run_dir.parent,
                  "--checkpoint", paths.sam2_checkpoint
                  if "sam2_checkpoint" in paths.overrides
                  else paths.models_root / v("tracking.sam2_checkpoint"),
                  "--notes", args.notes or f"pipeline {run_dir.name}",
                  "--skip_existing_ranges"] + ingest
                 + [a for r in (args.range or []) for a in ("--range", *r)],
        "keyframes": ["bazel", "run", f"{FF}/tracking:keyframe_viewer", "--",
                      "--run_dir", run_dir] + ingest,
        "audit": ["bazel", "run", f"{FF}/tracking:audit_requests", "--",
                  "--run_dir", run_dir, "--submit",
                  "--model", v("audit.model"),
                  "--min_supports", v("audit.min_supports"),
                  "--thinking_level", v("audit.thinking_level"),
                  "--max_support_chips", v("audit.max_support_chips"),
                  "--max_context_chips", v("audit.max_context_chips"),
                  "--max_description_samples",
                  v("audit.max_description_samples"),
                  "--chip_height_px", v("audit.chip_height_px"),
                  ] + ingest + guard + transport,
        "review": ["bazel", "run", f"{FF}/tracking:audit_review", "--",
                   "--run_dir", run_dir] + ingest,
        "offset": ["bazel", "run", f"{FF}/calibration:mount_offset_sweep",
                   "--", "--run_dir", run_dir,
                   "--coarse_step", v("calibration.sweep.coarse_step"),
                   "--fine_step", v("calibration.sweep.fine_step"),
                   "--fine_halfwidth", v("calibration.sweep.fine_halfwidth"),
                   "--min_observations",
                   v("calibration.sweep.min_observations"),
                   "--min_arc_deg", v("calibration.sweep.min_arc_deg"),
                   "--max_condition", v("calibration.sweep.max_condition"),
                   "--min_tracklets", v("calibration.sweep.min_tracklets"),
                   "--min_support_frac",
                   v("calibration.sweep.min_support_frac")] + fusion,
        "match": ["bazel", "run", f"{FF}/matching:match_landmarks", "--",
                  "--run_dir", run_dir, "--submit",
                  "--catalog", v("catalog.name"),
                  "--model", v("matching.model"),
                  "--query_batch", v("matching.query_batch"),
                  "--chunk_size", v("matching.chunk_size"),
                  "--thinking_level", v("matching.thinking_level"),
                  "--confidence_floor", v("matching.confidence_floor"),
                  "--instance_max_rows", v("matching.instance_max_rows"),
                  ] + guard + transport,
        "matchview": ["bazel", "run", f"{FF}/matching:match_viewer", "--",
                      "--run_dir", run_dir] + fusion,
        "export": ["bazel", "run", f"{FF}/localization:build_export", "--",
                   "--run_dir", run_dir,
                   "--output_dir", export_dir,
                   "--tables", tables,
                   "--catalog", v("catalog.name"),
                   "--default_log_lr", v("export.default_log_lr"),
                   "--clip", v("export.clip"),
                   "--min_step_m", v("export.min_step_m"),
                   "--sigma_pair_m", v("export.sigma_pair_m"),
                   "--max_visible_range_m", v("export.max_visible_range_m"),
                   ] + fusion,
        "localize": ["bazel", "run", f"{FF}/localization:run_export", "--",
                     "--export_dir", export_dir,
                     "--output_dir", loc_run,
                     "--init", "uniform",
                     "--n_particles", v("localization.n_particles"),
                     "--margin_m", v("localization.margin_m"),
                     "--max_visible_range_m", v("export.max_visible_range_m"),
                     "--position_roughening_m",
                     v("localization.position_roughening_m"),
                     "--heading_roughening_deg",
                     v("localization.heading_roughening_deg")],
        "plots": ["bazel", "run", f"{FF}/localization:plot_run", "--",
                  "--run_dir", loc_run, "--animate"],
        "viewer": ["bazel", "run", f"{FF}/localization:viewer", "--",
                   "--run_dir", loc_run],
    }


def cmd_run(args, parser):
    sys.stdout.reconfigure(line_buffering=True)
    paths, run_dir, doc, loc_run = resolve_run(parser, args)
    cfg = doc["config"]

    if args.only:
        selected = [args.only]
    else:
        lo, hi = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
        if lo > hi:
            parser.error(f"--from {args.from_stage} comes after --to "
                         f"{args.to_stage}")
        selected = [s for s in STAGES[lo:hi + 1] if s not in args.skip]

    sun_cmd = ["bazel", "run", f"{FF}/calibration:sun_offset_check", "--",
               "--run_dir", run_dir,
               "--n_frames", cfg["calibration"]["sun"]["n_frames"],
               "--min_speed_mps", cfg["calibration"]["sun"]["min_speed_mps"],
               "--elevation_tolerance_deg",
               cfg["calibration"]["sun"]["elevation_tolerance_deg"],
               "--work_width", cfg["calibration"]["sun"]["work_width"]]
    commands = build_commands(paths, run_dir, cfg, loc_run, args)

    experiment = loc_run.parent
    if any(s in selected for s in ("localize", "plots", "viewer")):
        experiment.mkdir(parents=True, exist_ok=True)
        if not (experiment / "experiment.md").exists():
            print(f"NOTE: {experiment}/experiment.md does not exist yet — "
                  f"every experiment directory carries one (what is being "
                  f"explored, status, conclusions).")

    print(f"dataset:    {paths.dataset}")
    print(f"run:        {run_dir}")
    print(f"experiment: {experiment}")
    print(f"stages:     {' -> '.join(selected)}")
    print(f"transport:  "
          f"{'on-demand (--online)' if args.online else 'Batch API'}")

    started = time.time()
    ran, skipped = [], []
    gated_extraction = False
    for stage in selected:
        if stage in DETECTION_CONSUMERS and not gated_extraction:
            check_extraction(paths, args.dry_run)
            gated_extraction = True
        if stage == "offset":
            # Absolute estimate first, relative after. Not `check`ed: the
            # sun check abstains on overcast, and abstaining is a correct
            # outcome that must not stop the pipeline.
            run(sun_cmd, "offset (sun, absolute)", dry_run=args.dry_run,
                check=False)
        if stage == "export":
            check_offset(run_dir, args.dry_run)
        if stage_done(stage, paths, run_dir, loc_run) and not args.force:
            print(f"\n-- {stage}: already done; --force to redo")
            skipped.append(stage)
            continue
        run(commands[stage], stage, dry_run=args.dry_run)
        ran.append(stage)
        if not args.dry_run:
            result = indexes.refresh(paths.root)
            print(f"  [indexes refreshed: {len(result['written'])} pages]")

    print(f"\n{'=' * 72}")
    print(f"pipeline finished in {(time.time() - started) / 60:.1f} min")
    print(f"  ran:     {', '.join(ran) or 'nothing'}")
    print(f"  skipped: {', '.join(skipped) or 'nothing'}")
    if not args.dry_run:
        extract_cost = token_cost(
            list(paths.frame_landmarks.rglob("predictions.jsonl")),
            cfg["extraction"]["model"])
        llm_cost_totals = token_cost(
            [run_dir / "semantic_audit" / "results.jsonl",
             run_dir / "matching" / "results.jsonl"],
            cfg["matching"]["model"])
        total = extract_cost["usd_on_demand"] + \
            llm_cost_totals["usd_on_demand"]
        print(f"  model spend to date (measured, on-demand list price): "
              f"${total:.2f}")
        print(f"  serve everything: cd {paths.root} && "
              f"python3 -m http.server 8935")
    print(f"{'=' * 72}")


def cmd_status(args, parser):
    paths, run_dir, doc, loc_run = resolve_run(parser, args)
    print(f"run {run_dir}\nexperiment output: {loc_run}")
    for stage in STAGES:
        try:
            done = stage_done(stage, paths, run_dir, loc_run)
        except Exception as exc:  # a marker probe must never crash status
            print(f"  {stage:<10} ?  ({exc})")
            continue
        print(f"  {stage:<10} {'done' if done else '—'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new-run", help="create + record a run config")
    paths_lib.add_arguments(p_new, dataset_required=True)
    p_new.add_argument("--run_name", required=True)
    p_new.add_argument("--config", required=True,
                       help="YAML with every result-shaping value (see "
                            "farfield/configs/)")
    p_new.add_argument("--notes", default="")

    p_run = sub.add_parser("run", help="execute stages of an existing run")
    paths_lib.add_arguments(p_run, checkpoint=True, feather=True, video=True)
    p_run.add_argument("--run_dir", type=Path, required=True)
    p_run.add_argument("--from", dest="from_stage", choices=STAGES,
                       default=STAGES[0])
    p_run.add_argument("--to", dest="to_stage", choices=STAGES,
                       default=STAGES[-1])
    p_run.add_argument("--only", choices=STAGES, default=None)
    p_run.add_argument("--skip", action="append", default=[], choices=STAGES)
    p_run.add_argument("--force", action="store_true",
                       help="re-run stages whose marker exists")
    p_run.add_argument("--online", action="store_true",
                       help="on-demand model calls instead of the Batch API")
    p_run.add_argument("--approve_cost", action="store_true")
    p_run.add_argument("--uninformative_tables", action="store_true",
                       help="export with flat tables (association-ambiguity "
                            "floor) instead of the matching stage's")
    p_run.add_argument("--range", nargs=3, action="append", default=None,
                       metavar=("NAME", "K_START", "K_END"))
    p_run.add_argument("--notes", default="")
    p_run.add_argument("--dry_run", action="store_true")

    p_status = sub.add_parser("status", help="stage markers for a run")
    paths_lib.add_arguments(p_status)
    p_status.add_argument("--run_dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "new-run":
        cmd_new_run(args)
    elif args.command == "run":
        cmd_run(args, parser)
    else:
        cmd_status(args, parser)


if __name__ == "__main__":
    main()
