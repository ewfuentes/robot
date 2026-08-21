"""VLM landmark extraction: panoramas -> pinhole faces -> Gemini -> frame_landmarks.

The pipeline's stage 1. Produces the `frame_landmarks` artifact every
downstream stage reads (`sentences/results/**/predictions.jsonl`, parsed by
`farfield/dataset.py`) and the `pinhole_images` artifact it is derived from.

Three stages:

  1. PINHOLE   - Render panoramas to pinhole faces (the pinhole_images
                 artifact). A full member of the pipeline: it renders into the
                 artifact lane and writes that artifact's manifest, so a leg
                 that has never been rendered needs no separate preparation.
  2. REQUESTS  - Build the Gemini batch request JSONL files (prompts.py).
  3. EXECUTE   - Run the requests through vertex_batch_manager.run_requests
                 (upload, submit, poll, download and the cost ceiling all live
                 there -- this stage never shells out to run them), then verify
                 that every panorama got a usable response. An incomplete
                 artifact stops here: downstream ingest reads a frame with no
                 response as a frame containing no objects, silently.

Every result-shaping argument is required -- model, prompt type, pinhole and
media resolution, thinking level, both artifact versions. There are no
defaults on purpose (REORG.md rule 2): each of these has already meant two
different values in two different places on this project, and a default is
how a run lands on a value nobody chose.

Example:

    bazel run //experimental/overhead_matching/swag/farfield/extraction:extract_landmarks -- \\
        --dataset boston_harbor_leg2 \\
        --frame_landmarks_version v5 \\
        --pinhole_version v1 \\
        --prompt_type osm_tags_farfield_v2 \\
        --pinhole_resolution 2048 \\
        --media_resolution MEDIA_RESOLUTION_ULTRA_HIGH \\
        --thinking_level HIGH \\
        --model <model-id> \\
        --gcs_prefix gs://<bucket>/<staging>/
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path

from experimental.overhead_matching.swag.farfield import (
    paths as paths_lib,
    provenance,
)
from experimental.overhead_matching.swag.farfield.extraction import (
    llm_cost,
    panorama_to_pinhole,
    prompts,
    vertex_batch_manager as vbm,
)

GENERATOR = ("//experimental/overhead_matching/swag/farfield/extraction"
             ":extract_landmarks")

# Faces are rendered at 90-degree yaw intervals with a 90-degree FOV. This is
# a convention, not a knob: geometry.direction_from_face_px is the verified
# inverse of exactly this render, and dataset ingest validates face yaws
# against it.
FACE_FOV_RAD = math.pi / 2.0


class Stage(IntEnum):
    PINHOLE = 1
    REQUESTS = 2
    EXECUTE = 3


STAGE_NAMES = {
    Stage.PINHOLE: "Convert panoramas to pinhole images",
    Stage.REQUESTS: "Create batch API request files",
    Stage.EXECUTE: "Execute requests via vertex_batch_manager",
}


@dataclass
class Config:
    """Resolved inputs, outputs and settings for one extraction run."""

    dataset: str
    root: Path
    version: str            # frame_landmarks artifact version
    pinhole_version: str    # pinhole_images artifact version
    dataset_base: Path
    panorama_dir: Path
    pinhole_dir: Path
    artifact_dir: Path
    prompt_type: str
    pinhole_resolution: int
    media_resolution: str
    thinking_level: str
    num_workers: int
    allow_incomplete: bool
    force: bool
    start_stage: int
    end_stage: int
    # The parsed CLI namespace, carrying vertex_batch_manager's execution
    # flags (--model, --online, --gcs_prefix, --parallel, --poll_interval,
    # --cost_limit, --approve_cost). Passed to run_requests as-is.
    execution: argparse.Namespace

    @property
    def requests_dir(self) -> Path:
        return self.artifact_dir / "sentence_requests"

    @property
    def sentences_dir(self) -> Path:
        return self.artifact_dir / "sentences"

    @property
    def main_predictions(self) -> Path:
        """The one resumable results file the execute stage appends to.

        Lives under `sentences/results/*/prediction-*/` because that is the
        glob every consumer reads (farfield.dataset.load_predictions). Retry
        results land in a sibling directory sorting after this one, so they
        supersede failed keys without mutating anything already written.
        """
        return (self.sentences_dir / "results" / "000_main" /
                "prediction-main" / "predictions.jsonl")


# ---------------------------------------------------------------------------
# Stage 1: pinhole render
# ---------------------------------------------------------------------------

def panorama_stems(panorama_dir: Path) -> list:
    """Stems of the panoramas to render, sorted.

    Stems carry identity through the whole pipeline -- they name the pinhole
    subdirectories and key the requests and predictions -- so they are
    compared literally rather than counted.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(f.stem for f in Path(panorama_dir).iterdir()
                  if f.is_file() and f.suffix.lower() in exts)


def check_pinhole_reuse(config: Config) -> bool:
    """Decide whether an existing pinhole render can be reused.

    Verifies the *contract* -- one directory per pano stem, all four faces
    present, rendered at the requested resolution -- instead of re-deriving
    the pixels. Resolution is read back from a face JPEG header, which catches
    a resolution change even on renders that carry no manifest.
    """
    pinhole_dir = config.pinhole_dir
    if not pinhole_dir.exists():
        return False

    want = panorama_stems(config.panorama_dir)
    if not want:
        sys.exit(f"  ERROR: no panoramas found in {config.panorama_dir}")

    have = {d.name for d in pinhole_dir.iterdir() if d.is_dir()}
    if not have:
        return False

    print(f"  Found existing pinhole render at {pinhole_dir} "
          f"({len(have)} panoramas)")

    missing = [s for s in want if s not in have]
    if missing:
        print(f"  Incomplete: {len(missing)} of {len(want)} stems absent "
              f"(e.g. {missing[0]})")
        return False

    extra = len(have) - len(want)
    if extra > 0:
        # A superset still covers this run, but it means the directory was
        # rendered from a different panorama set, which is worth saying out
        # loud (the request builder excludes the stale stems).
        print(f"  Note: {extra} rendered stem(s) are not in "
              f"{config.panorama_dir}")

    incomplete = [s for s in want
                  if not all((pinhole_dir / s / f"{face}.jpg").exists()
                             for face in prompts.PINHOLE_FACES)]
    if incomplete:
        print(f"  Incomplete: {len(incomplete)} stem(s) missing faces "
              f"(e.g. {incomplete[0]})")
        return False

    probe = pinhole_dir / want[0] / f"{prompts.PINHOLE_FACES[0]}.jpg"
    try:
        from PIL import Image
        with Image.open(probe) as img:
            width = img.width
    except OSError as exc:  # an unreadable face is not reusable
        print(f"  Could not read {probe}: {exc}")
        return False

    if width != config.pinhole_resolution:
        print(f"  Resolution mismatch: rendered at {width}px, requested "
              f"{config.pinhole_resolution}px")
        return False

    print(f"  Verified {len(want)} stems x {len(prompts.PINHOLE_FACES)} faces "
          f"at {width}px - reusing")
    return True


def observed_pinhole_geometry(config: Config) -> dict:
    """Read the rendered face geometry back off disk.

    Recorded from the output rather than from the flags, so the manifest
    states what the faces *are* rather than what was requested.
    """
    stems = panorama_stems(config.panorama_dir)
    geometry = {"n_panoramas": len(stems),
                "faces": list(prompts.PINHOLE_FACES)}
    if not stems:
        return geometry
    probe = config.pinhole_dir / stems[0] / f"{prompts.PINHOLE_FACES[0]}.jpg"
    try:
        from PIL import Image
        with Image.open(probe) as img:
            geometry["res_x"], geometry["res_y"] = img.width, img.height
    except OSError:
        geometry["res_x"] = config.pinhole_resolution
        geometry["res_y"] = None
        geometry["note"] = f"could not read {probe} to confirm resolution"
    return geometry


def write_pinhole_manifest(config: Config, *, adopted: bool = False) -> Path:
    """Record how the pinhole faces were rendered, next to the faces."""
    notes = ("Dir names are panorama stems and must match panorama/ exactly: "
             "they key the requests and predictions downstream.")
    if adopted:
        notes += (" Adopted: this render predates this invocation and was "
                  "verified against the contract (stems x faces x "
                  "resolution) rather than re-rendered, so argv/git here "
                  "describe the verifying run, not the original render.")
    path = provenance.write(
        config.pinhole_dir,
        generator=f"{GENERATOR} (stage 1) -> extraction/panorama_to_pinhole",
        inputs={"panorama_dir": paths_lib.relative_to_root(
            config.panorama_dir, config.root)},
        config={
            **observed_pinhole_geometry(config),
            "requested_res_x": config.pinhole_resolution,
            "fov_deg": math.degrees(FACE_FOV_RAD),
            "layout": "one dir per pano stem, 4 face JPEGs each",
            "convention": ("faces at 90-deg yaw intervals, 90-deg FOV; "
                           "geometry.direction_from_face_px is the render's "
                           "verified inverse (see farfield/geometry.py)"),
        },
        extra={"kind": paths_lib.PINHOLE_IMAGES, "dataset": config.dataset,
               "version": config.pinhole_version},
        notes=notes,
    )
    print(f"  Wrote {path}")
    return path


def stage_pinhole(config: Config):
    """Stage 1: render panoramas to pinhole faces."""
    if check_pinhole_reuse(config):
        # A reused render is still an artifact; give it a manifest if its
        # producer predates the contract. An existing manifest is left alone.
        if not (config.pinhole_dir / provenance.MANIFEST_NAME).exists():
            write_pinhole_manifest(config, adopted=True)
        return

    if config.pinhole_dir.exists() and not config.force:
        response = input(
            f"  Pinhole dir {config.pinhole_dir} exists but verification "
            f"failed. Re-render? [y/N]: ")
        if response.lower() != "y":
            print("  Aborting")
            sys.exit(1)

    config.pinhole_dir.mkdir(parents=True, exist_ok=True)
    panorama_to_pinhole.process_panoramas(
        config.panorama_dir,
        config.pinhole_dir,
        FACE_FOV_RAD,
        FACE_FOV_RAD,
        config.pinhole_resolution,
        config.pinhole_resolution,  # fov_x == fov_y, so res_y == res_x
        num_workers=config.num_workers,
    )
    write_pinhole_manifest(config)


# ---------------------------------------------------------------------------
# Stage 2: requests
# ---------------------------------------------------------------------------

def stage_requests(config: Config):
    """Stage 2: build the Gemini batch request JSONL files."""
    written = prompts.write_requests(
        config.pinhole_dir,
        config.panorama_dir,
        config.requests_dir,
        prompt_type=config.prompt_type,
        media_resolution=config.media_resolution,
        thinking_level=config.thinking_level,
        num_workers=config.num_workers,
    )
    print(f"  {len(written)} request file(s) in {config.requests_dir}")


# ---------------------------------------------------------------------------
# Stage 3: execute
# ---------------------------------------------------------------------------

def _total_estimate(request_files: list, model: str) -> llm_cost.Estimate:
    """One estimate over several request files, priced at the actual model."""
    _, _, rate_label = llm_cost.rates_for(model)
    total = llm_cost.Estimate(model=model, rate_label=rate_label)
    for path in request_files:
        part = llm_cost.estimate_jsonl(path, model=model)
        for field in ("n_requests", "prompt_tokens", "output_tokens",
                      "n_images", "text_chars", "n_large_prompts",
                      "usd_on_demand"):
            setattr(total, field, getattr(total, field) + getattr(part, field))
    total.usd_batch = total.usd_on_demand * llm_cost.BATCH_MULTIPLIER
    return total


def stage_execute(config: Config):
    """Stage 3: run every request file through vertex_batch_manager.

    `run_requests` owns the transport (batch by default, `--online` to swap),
    the staging bucket contract, resumability (keys that already have a
    usable response in the output are skipped) and the cost ceiling priced at
    the run's `--model`. This stage only sequences the files into the one
    results file every consumer globs.
    """
    execution = config.execution
    files = sorted(config.requests_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"no request files under {config.requests_dir}; run stage "
                 f"{int(Stage.REQUESTS)} (requests) first")

    out = config.main_predictions
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    if len(files) == 1:
        # run_requests carries the cost gate itself, priced at --model.
        vbm.run_requests(execution, files[0], out,
                         tag=f"{config.dataset}_{config.version}_{stamp}")
        return

    # Several request files are still ONE step to the operator, so the
    # ceiling is enforced once over the total, up front ...
    total = _total_estimate(files, execution.model)
    try:
        llm_cost.enforce_limit(
            total, limit_usd=execution.cost_limit,
            label=(f"{config.dataset} landmark extraction "
                   f"({len(files)} request files)"),
            online=execution.online,
            approved=execution.approve_cost)
    except llm_cost.CostLimitExceeded as exc:
        sys.exit(f"\n{exc}")
    # ... and the per-file gates inside run_requests are then marked approved,
    # so one human answer covers the step instead of one per file. Each file's
    # estimate is a subset of the total the human just saw.
    approved = argparse.Namespace(**{**vars(execution), "approve_cost": True})
    for idx, path in enumerate(files):
        vbm.run_requests(
            approved, path, out,
            tag=f"{config.dataset}_{config.version}_{stamp}_p{idx:02d}")


# ---------------------------------------------------------------------------
# Completeness gate
# ---------------------------------------------------------------------------

def validate_predictions(sentences_dir: Path, panorama_dir: Path) -> dict:
    """Classify every stored response: usable, empty, or failed.

    The batch API reports success at the *job* level while individual requests
    can still come back with no candidates at all -- leg2 lost 23 of 236
    frames to transient `TPU device returned error` and the old pipeline
    printed "Pipeline complete!" over it. Nothing downstream notices either:
    ingest skips a frame with no prediction with a bare `continue`, so
    tracking would simply see 10% of the leg as containing no objects and
    starve tracks crossing it.

    `empty` is not a failure. A frame of open water genuinely has no
    landmarks, and on leg2 19 frames are legitimately in that state.
    """
    paths = sorted(Path(sentences_dir).glob(
        "results/*/prediction-*/predictions.jsonl"))
    report = {"files": len(paths), "ok": [], "empty": [], "failed": [],
              "unparseable": []}
    # Later files win, matching dataset.load_predictions' sorted-glob dict
    # build, so a retry directory supersedes the attempt it repairs.
    by_key = {}
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    report["unparseable"].append(str(path))
                    continue
                key = record.get("key")
                if key:
                    by_key[key] = record

    for key, record in by_key.items():
        response = record.get("response") or {}
        if not isinstance(response, dict) or "candidates" not in response:
            report["failed"].append(key)
            continue
        try:
            text = (response["candidates"][0]["content"]["parts"][0]["text"]
                    ).strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            landmarks = json.loads(text).get("landmarks") or []
        except (KeyError, IndexError, json.JSONDecodeError, TypeError):
            report["failed"].append(key)
            continue
        report["empty" if not landmarks else "ok"].append(key)

    stems = panorama_stems(panorama_dir)
    report["n_panoramas"] = len(stems)
    report["missing"] = [s for s in stems if s not in by_key]
    return report


def _repair_hint(config: Config) -> str:
    return (f"bazel run {GENERATOR} -- \\\n"
            f"        --dataset {config.dataset} "
            f"--frame_landmarks_version {config.version} "
            f"--pinhole_version {config.pinhole_version} \\\n"
            f"        --prompt_type {config.prompt_type} "
            f"--pinhole_resolution {config.pinhole_resolution} \\\n"
            f"        --media_resolution {config.media_resolution} "
            f"--thinking_level {config.thinking_level} \\\n"
            f"        --model {config.execution.model} --retry_failed")


def print_validation(report: dict, *, repair_hint: str) -> bool:
    """Report response coverage. Returns True when the artifact is complete."""
    n_ok, n_empty = len(report["ok"]), len(report["empty"])
    failed, missing = report["failed"], report["missing"]
    total = report.get("n_panoramas", 0)
    print(f"  {n_ok} with landmarks, {n_empty} legitimately empty, "
          f"{len(failed)} failed, {len(missing)} absent (of {total} panoramas)")
    if not failed and not missing:
        print("  Complete: every panorama has a usable response.")
        return True
    print(f"\n  INCOMPLETE ARTIFACT: {len(failed) + len(missing)} of {total} "
          f"panoramas ({100 * (len(failed) + len(missing)) / max(total, 1):.1f}%)"
          f" have no usable response.")
    for key in (failed + missing)[:5]:
        print(f"    {key}")
    if len(failed) + len(missing) > 5:
        print(f"    ... and {len(failed) + len(missing) - 5} more")
    print("\n  Downstream will NOT complain: ingest (farfield/dataset.py) "
          "skips a frame with no prediction silently, so tracking would read "
          "these as frames containing no objects.")
    print(f"\n  Repair with:\n    {repair_hint}")
    return False


def coverage_summary(report: dict) -> dict:
    """Response coverage, recorded so an accepted gap stays visible.

    A consumer reading the manifest can otherwise not tell a leg with
    genuinely few landmarks from one that lost frames to API errors -- both
    look like frames with no observations.
    """
    gaps = report["failed"] + report["missing"]
    summary = {
        "n_panoramas": report.get("n_panoramas", 0),
        "n_with_landmarks": len(report["ok"]),
        "n_empty_responses": len(report["empty"]),
        "n_no_usable_response": len(gaps),
    }
    if gaps:
        summary["complete"] = False
        summary["missing_keys"] = sorted(gaps)
        summary["warning"] = (
            "Panoramas listed in missing_keys have NO usable response. ingest "
            "skips them silently, so downstream reads them as frames "
            "containing no objects. Repair with --retry_failed.")
    else:
        summary["complete"] = True
    return summary


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

def retry_request_records(requests_dir: Path, wanted: set) -> list:
    """The stored request records whose keys are in `wanted`."""
    subset = []
    for path in sorted(Path(requests_dir).glob("*.jsonl")):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("key") in wanted:
                    subset.append(record)
    return subset


def retry_failed(config: Config):
    """Re-run only the requests that came back without a usable response.

    Written as an *additional* predictions file rather than by editing the
    original: consumers build a dict over a sorted glob, so a directory
    sorting after the first attempt supersedes it key by key. The failed
    attempt stays on disk as the record of what happened, and no published
    file is mutated. After a repair the manifest is refreshed so its recorded
    coverage matches the artifact again.
    """
    report = validate_predictions(config.sentences_dir, config.panorama_dir)
    broken = report["failed"] + report["missing"]
    if not broken:
        print("Nothing to retry: every panorama has a usable response.")
        return
    print(f"retrying {len(broken)} request(s)")

    wanted = set(broken)
    subset = retry_request_records(config.requests_dir, wanted)
    if len(subset) != len(wanted):
        found = {r.get("key") for r in subset}
        print(f"  WARNING: {len(wanted - found)} failed key(s) have no "
              f"request on disk; {config.requests_dir} may have been deleted")
    if not subset:
        sys.exit("no requests available to retry")

    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    retry_dir = (config.sentences_dir / "results" /
                 f"zz_retry_{stamp}" / f"prediction-retry-{stamp}")
    retry_dir.mkdir(parents=True, exist_ok=True)
    requests_path = retry_dir / "requests.jsonl"
    with open(requests_path, "w") as handle:
        for record in subset:
            handle.write(json.dumps(record) + "\n")
    print(f"  wrote {requests_path} ({len(subset)} requests)")

    # Online rather than batch: a handful of requests does not justify a
    # batch job's queue latency, and run-online is itself resumable on error.
    # The cost gate inside run_requests still applies, priced at --model.
    online = argparse.Namespace(**{**vars(config.execution), "online": True})
    vbm.run_requests(online, requests_path, retry_dir / "predictions.jsonl",
                     tag=f"{config.dataset}_{config.version}_retry_{stamp}")

    print("\nafter retry:")
    report = validate_predictions(config.sentences_dir, config.panorama_dir)
    complete = print_validation(report, repair_hint=_repair_hint(config))
    if complete or config.allow_incomplete:
        write_frame_landmarks_manifest(config, report)


# ---------------------------------------------------------------------------
# frame_landmarks manifest
# ---------------------------------------------------------------------------

def request_fingerprint(requests_dir: Path) -> dict:
    """Hash what was actually sent to the model.

    `prompt_type` names a key in `prompts.SYSTEM_PROMPTS`, and that key's
    *text* can be edited without the name changing -- so the name alone does
    not pin the extraction. The request JSONL holds the prompt as it went
    out, alongside every image, so hashing those files pins the whole
    request: change the prompt, the resolution, or the pano set, and the
    digest moves.
    """
    files = sorted(Path(requests_dir).glob("*.jsonl"))
    if not files:
        return {"request_sha256": None,
                "note": "request files absent at manifest time"}
    digest = hashlib.sha256()
    n_requests = 0
    for path in files:
        with open(path, "rb") as handle:
            for line in handle:
                digest.update(line)
                n_requests += 1
    return {"request_sha256": digest.hexdigest(),
            "request_files": [f.name for f in files],
            "n_requests": n_requests}


def prompt_fingerprint(requests_dir: Path) -> dict:
    """Hash the system prompt TEXT alone, as it was actually sent.

    `request_sha256` pins the whole request, images included, so two datasets
    extracted with an identical prompt still get different digests and the
    manifest cannot answer "did these two runs use the same prompt?". That
    question cost a hand comparison across 4.4 GB of stored request JSONL
    when the boston_harbor_leg1 v1 -> v4 regression was traced to a silent
    rewrite of `osm_tags_farfield`. This digest answers it directly:
    `grep prompt_sha256 artifacts/frame_landmarks/*/*/manifest.json`.

    Read from the request file rather than from the prompt registry so it
    records what went out, not what the tree says now.
    """
    files = sorted(Path(requests_dir).glob("*.jsonl"))
    if not files:
        return {"prompt_sha256": None}
    with open(files[0]) as handle:
        first = handle.readline()
    if not first.strip():
        return {"prompt_sha256": None}
    try:
        request = json.loads(first).get("request", {})
        instruction = (request.get("systemInstruction")
                       or request.get("system_instruction") or {})
        text = "".join(part.get("text", "")
                       for part in instruction.get("parts", []))
    except (json.JSONDecodeError, AttributeError):
        return {"prompt_sha256": None}
    if not text:
        return {"prompt_sha256": None}
    return {"prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_chars": len(text)}


def landmark_counts(sentences_dir: Path) -> dict:
    """How many predictions came back, for a size check against the panos."""
    files = sorted(Path(sentences_dir).rglob("predictions.jsonl"))
    if not files:
        return {"note": "predictions.jsonl absent at manifest time"}
    lines = 0
    for path in files:
        with open(path, "rb") as handle:
            lines += sum(1 for _ in handle)
    return {"n_prediction_files": len(files), "n_prediction_lines": lines}


def write_frame_landmarks_manifest(config: Config, report: dict) -> Path:
    """Record the extraction that produced the `frame_landmarks` artifact."""
    root = config.root
    path = provenance.write(
        config.artifact_dir,
        generator=GENERATOR,
        inputs={
            "dataset_base": paths_lib.relative_to_root(config.dataset_base,
                                                       root),
            "panorama_dir": paths_lib.relative_to_root(config.panorama_dir,
                                                       root),
            "pinhole_images": paths_lib.relative_to_root(config.pinhole_dir,
                                                         root),
        },
        config={
            "prompt_type": config.prompt_type,
            # The name is a lookup key whose text can change under it; the
            # digest is what actually pins this extraction.
            **prompt_fingerprint(config.requests_dir),
            **request_fingerprint(config.requests_dir),
            "model": config.execution.model,
            "pinhole_resolution": config.pinhole_resolution,
            "media_resolution": config.media_resolution,
            "thinking_level": config.thinking_level,
            "execution": ("online (on-demand)" if config.execution.online
                          else "batch"),
            "gcs_prefix": config.execution.gcs_prefix,
            **landmark_counts(config.sentences_dir),
            **coverage_summary(report),
        },
        extra={"kind": paths_lib.FRAME_LANDMARKS, "dataset": config.dataset,
               "version": config.version},
        notes=("Consumers glob sentences/results/**/predictions.jsonl "
               "(farfield/dataset.py). DO NOT DELETE sentence_requests/: no "
               "stage reads it back, but each request line carries the "
               "system prompt verbatim, which makes it the only record of "
               "the prompt text this artifact was built with - `prompt_type` "
               "names a registry key whose text changes over time, and "
               "git_commit pins the tree rather than the working copy that "
               "ran. It is also what request_sha256 is computed over, so the "
               "digest cannot be recomputed once it is gone."),
    )
    print(f"  Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

STAGE_FUNCS = {
    Stage.PINHOLE: stage_pinhole,
    Stage.REQUESTS: stage_requests,
    Stage.EXECUTE: stage_execute,
}


def run_pipeline(config: Config):
    print("=" * 70)
    print("Farfield landmark extraction")
    print("=" * 70)
    print(f"  Dataset:        {config.dataset}")
    print(f"  Panorama dir:   {config.panorama_dir}")
    print(f"  Pinhole dir:    {config.pinhole_dir} "
          f"(pinhole_images {config.pinhole_version})")
    print(f"  Artifact dir:   {config.artifact_dir} "
          f"(frame_landmarks {config.version})")
    print(f"  Prompt type:    {config.prompt_type} "
          f"(sha256 {prompts.prompt_sha256(config.prompt_type)[:12]}...)")
    print(f"  Pinhole res:    {config.pinhole_resolution}")
    print(f"  Media res:      {config.media_resolution}")
    print(f"  Thinking level: {config.thinking_level}")
    print(f"  Model:          {config.execution.model}")
    print(f"  Transport:      "
          f"{'online (on-demand)' if config.execution.online else 'batch'}")
    print(f"  Stages:         {config.start_stage} -> {config.end_stage}")
    print()

    stages = [s for s in Stage
              if config.start_stage <= s <= config.end_stage]

    if Stage.EXECUTE in stages:
        # The old orchestrator exported this into every vertex subprocess; the
        # imported client needs it in our own environment instead. Explicit
        # settings are respected.
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        vbm.check_environment()

    pipeline_start = time.time()
    for stage in stages:
        print(f"\n{'=' * 70}")
        print(f"STAGE {stage.value}: {STAGE_NAMES[stage]}")
        print(f"{'=' * 70}")
        stage_start = time.time()
        STAGE_FUNCS[stage](config)
        print(f"\n  Stage {stage.value} completed in "
              f"{time.time() - stage_start:.1f}s")

    # The manifest certifies the artifact, so it is written only once the
    # stage that produces predictions has run -- and only past the
    # completeness gate (or with the gap explicitly accepted and recorded).
    if config.end_stage >= Stage.EXECUTE:
        print(f"\n{'=' * 70}")
        print("VALIDATE: response coverage")
        print(f"{'=' * 70}")
        report = validate_predictions(config.sentences_dir,
                                      config.panorama_dir)
        complete = print_validation(report, repair_hint=_repair_hint(config))
        if not complete and not config.allow_incomplete:
            print("\n  Stopping without a manifest. Pass --allow_incomplete "
                  "to write one anyway; the manifest records the gap.")
            sys.exit(1)
        write_frame_landmarks_manifest(config, report)

    print(f"\n{'=' * 70}")
    print(f"Extraction finished in {time.time() - pipeline_start:.1f}s")
    if config.end_stage >= Stage.EXECUTE:
        print(f"frame_landmarks artifact: {config.artifact_dir}")
    print(f"{'=' * 70}")


def main():
    # Stages run for tens of minutes and print progress as they go. Redirected
    # to a file, Python block-buffers stdout, so a live run looks frozen for
    # 8 KB at a time -- which is indistinguishable from a hang exactly when
    # you most want to know it is still working.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    paths_lib.add_arguments(parser, pinhole=True)
    vbm.add_execution_arguments(parser)

    parser.add_argument(
        "--prompt_type", required=True, choices=list(prompts.PROMPT_TYPES),
        help="Which farfield prompt to extract with. Required: v1 and v2 "
             "behave measurably differently (see the registry comments in "
             "prompts.py), so which one a run used must be chosen, and is "
             "recorded in the manifest as text digest, not just name.")
    parser.add_argument(
        "--pinhole_resolution", type=int, required=True,
        help="Pinhole face resolution in pixels. Required: previously 1024 "
             "here and 2048 in the collector -- which is why there is no "
             "default.")
    parser.add_argument(
        "--media_resolution", required=True,
        choices=list(prompts.MEDIA_RESOLUTIONS),
        help="Media resolution for Gemini image processing. Required: it "
             "changes what the model can read off a face, and it is priced "
             "differently.")
    parser.add_argument(
        "--thinking_level", required=True,
        choices=list(prompts.THINKING_LEVELS),
        help="Gemini thinking level. Required: it shapes both the result and "
             "the bill (thinking tokens run ~4.5x visible output).")
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Workers for pinhole rendering and image encoding (default: 8)")
    parser.add_argument(
        "--start_stage", type=int, default=int(Stage.PINHOLE),
        help=f"Resume from this stage (1-{int(Stage.EXECUTE)}, default: 1)")
    parser.add_argument(
        "--end_stage", type=int, default=int(Stage.EXECUTE),
        help=f"Stop after this stage (1-{int(Stage.EXECUTE)}, default: "
             f"{int(Stage.EXECUTE)})")
    parser.add_argument(
        "--allow_incomplete", action="store_true",
        help="Write a manifest even when some panoramas have no usable "
             "response (the gap is recorded in the manifest)")
    parser.add_argument(
        "--retry_failed", action="store_true",
        help="Re-run only the requests with no usable response and stop. "
             "Results are written as an additional predictions file that "
             "supersedes the failed attempt; nothing is overwritten.")
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Report response coverage for an existing extraction and stop")
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the re-render confirmation when an existing pinhole "
             "render fails verification. Never implied by any other flag.")

    args = parser.parse_args()

    # The artifact versions name what this run writes; a default here is how
    # one version's data is silently read against another's (REORG.md rule 2).
    if not args.frame_landmarks_version:
        parser.error(
            "--frame_landmarks_version is required: it names the output "
            "artifact version, and there is no default on purpose.")
    if not args.pinhole_version:
        parser.error(
            "--pinhole_version is required: it names the pinhole_images "
            "artifact this extraction renders into / reads from, and there "
            "is no default on purpose.")

    last = int(Stage.EXECUTE)
    if not (1 <= args.start_stage <= last):
        parser.error(f"--start_stage must be between 1 and {last}")
    if not (1 <= args.end_stage <= last):
        parser.error(f"--end_stage must be between 1 and {last}")
    if args.start_stage > args.end_stage:
        parser.error("--start_stage must be <= --end_stage")

    paths = paths_lib.resolve(parser, args,
                              require=("dataset_base", "panorama_dir"))

    config = Config(
        dataset=paths.dataset,
        root=paths.root,
        version=args.frame_landmarks_version,
        pinhole_version=args.pinhole_version,
        dataset_base=paths.dataset_base,
        panorama_dir=paths.panorama_dir,
        pinhole_dir=paths.pinhole_images,
        artifact_dir=paths.frame_landmarks,
        prompt_type=args.prompt_type,
        pinhole_resolution=args.pinhole_resolution,
        media_resolution=args.media_resolution,
        thinking_level=args.thinking_level,
        num_workers=args.num_workers,
        allow_incomplete=args.allow_incomplete,
        force=args.force,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        execution=args,
    )

    if args.validate_only:
        report = validate_predictions(config.sentences_dir,
                                      config.panorama_dir)
        print_validation(report, repair_hint=_repair_hint(config))
        return
    if args.retry_failed:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
        vbm.check_environment()
        retry_failed(config)
        return

    run_pipeline(config)


if __name__ == "__main__":
    main()
