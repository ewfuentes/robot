"""End-to-end orchestration for Gemini landmark extraction from panoramas.

Runs the full 7-stage pipeline:
  1. PINHOLE   - Render panoramas to pinhole faces (the pinhole_images artifact)
  2. REQUESTS  - Create batch API request JSONL files
  3. UPLOAD    - Upload requests to GCS
  4. SUBMIT    - Submit Vertex AI batch jobs
  5. WAIT      - Poll until all batch jobs complete
  6. DOWNLOAD  - Download results from GCS, then verify every panorama got a
                 usable response (an incomplete artifact stops here)
  7. EMBEDDINGS - Create embeddings from batch results (OPT-IN, see below)

Stages 1-6 are the pipeline; **stage 7 is opt-in** via `--with_embeddings`.
Nothing in the tracking pipeline reads `embeddings.pkl` -- `ingest.py` reads
`sentences/results/**/predictions.jsonl` -- so it is only needed for the older
cosine matcher in `landmark_filtering/semantic_similarity.py`.

Produces the `frame_landmarks` artifact every downstream stage reads, and the
`pinhole_images` artifact it is derived from. Stage 1 is a full member of the
pipeline: it renders into the artifact lane and writes that artifact's
manifest, so a leg that has never been rendered needs no separate preparation
step.

Farfield mode -- resolves the panorama input and both artifact outputs from the
dataset name (see swag/data/farfield_paths.py):

    bazel run //experimental/overhead_matching/swag/scripts:extract_gemini_landmarks_from_panoramas -- \
        --dataset boston_harbor_leg2 \
        --prompt_type osm_tags_farfield \
        --pinhole_resolution 2048 \
        --media_resolution MEDIA_RESOLUTION_ULTRA_HIGH \
        --model gemini-3.1-pro-preview

Legacy mode -- explicit directories, for the older VIGOR-style sets that live
outside the farfield lanes and whose layout predates the artifact contract:

    bazel run //experimental/overhead_matching/swag/scripts:extract_gemini_landmarks_from_panoramas -- \
        --name nightdrive \
        --panorama_dir /data/overhead_matching/datasets/VIGOR/nightdrive/panorama/ \
        --output_base /data/overhead_matching/datasets/semantic_landmark_embeddings/nightdrive_osm_tags/

Legacy pinholes now default to `<output_base>/<name>/pinhole_images` rather than
a global directory keyed only by `--name`, so each set stays self-contained. To
reuse an existing render from the old shared location, point at it:
`--pinhole_dir /data/overhead_matching/datasets/pinhole_images/<name>`;
otherwise stage 1 re-renders, which is correct but not free.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path

from experimental.overhead_matching.swag.data import farfield_paths
from experimental.overhead_matching.swag.scripts import llm_cost


class Stage(IntEnum):
    PINHOLE = 1
    REQUESTS = 2
    UPLOAD = 3
    SUBMIT = 4
    WAIT = 5
    DOWNLOAD = 6
    EMBEDDINGS = 7


STAGE_NAMES = {
    Stage.PINHOLE: "Convert panoramas to pinhole images",
    Stage.REQUESTS: "Create batch API request files",
    Stage.UPLOAD: "Upload requests to GCS",
    Stage.SUBMIT: "Submit Vertex AI batch jobs",
    Stage.WAIT: "Wait for batch jobs to complete",
    Stage.DOWNLOAD: "Download results from GCS",
    Stage.EMBEDDINGS: "Create embeddings from results",
}

# Faces are rendered at 90-degree yaw intervals with a 90-degree FOV, which is
# what panorama_to_pinhole emits and what pano_geometry assumes when it maps a
# detection's box back to a panorama azimuth.
PINHOLE_FACES = ("yaw_000", "yaw_090", "yaw_180", "yaw_270")

# When launched via `bazel run`, nested bazel calls must run from the workspace root.
WORKSPACE_DIR = os.environ.get("BUILD_WORKSPACE_DIRECTORY")

# Bazel targets used by the pipeline
BAZEL_TARGETS = [
    "//experimental/overhead_matching/swag/scripts:panorama_to_pinhole",
    "//experimental/overhead_matching/swag/model:semantic_landmark_extractor",
    "//experimental/overhead_matching/swag/scripts:vertex_batch_manager",
    "//experimental/overhead_matching/swag/scripts:create_embeddings_with_gemini",
]


@dataclass
class PipelineConfig:
    """Resolved inputs and outputs for one extraction run.

    The three directories are resolved by the caller rather than derived here,
    which is what lets farfield mode write into `artifacts/<kind>/<dataset>/v<N>`
    while legacy mode keeps the flat `<output_base>/<name>` layout the older
    VIGOR sets are stored in.
    """

    name: str
    panorama_dir: Path
    pinhole_dir: Path
    artifact_dir: Path
    prompt_type: str
    model: str
    gcs_bucket: str
    num_pinhole_workers: int
    poll_interval: int
    dry_run: bool
    force: bool
    start_stage: int
    end_stage: int
    pinhole_resolution: int = 1024
    media_resolution: str = "MEDIA_RESOLUTION_HIGH"
    thinking_level: str = "HIGH"
    embedding_model: str = "gemini-embedding-001"
    allow_incomplete: bool = False
    cost_limit: float = 50.0
    approve_cost: bool = False
    # Present in farfield mode only; drives manifest writing. Legacy sets get no
    # manifests because their layout predates the artifact contract and nothing
    # reads one there.
    paths: farfield_paths.FarfieldPaths | None = None
    gcs_prefix: str = ""

    @property
    def gcs_prefix_record(self) -> Path:
        """Where the staging prefix is remembered, for a later resume."""
        return self.artifact_dir / "gcs_prefix.txt"

    @property
    def gcs_requests_uri(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_prefix}/requests/"

    @property
    def gcs_results_uri(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_prefix}/results/"

    @property
    def sentence_requests_dir(self) -> Path:
        return self.artifact_dir / "sentence_requests"

    @property
    def sentence_requests_jsonl_dir(self) -> Path:
        return self.sentence_requests_dir / "panorama_sentence_requests"

    @property
    def sentences_dir(self) -> Path:
        return self.artifact_dir / "sentences"

    @property
    def embeddings_dir(self) -> Path:
        return self.artifact_dir / "embeddings"

    @property
    def embeddings_file(self) -> Path:
        return self.embeddings_dir / "embeddings.pkl"

    @property
    def job_names_file(self) -> Path:
        return self.artifact_dir / "submitted_job_names.txt"

    @property
    def pinhole_manifest(self) -> Path:
        return self.pinhole_dir / "manifest.json"


def run_command(cmd, desc, dry_run=False, env=None, check=True):
    """Run a subprocess command with logging."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n  $ {cmd_str}")

    if dry_run:
        print("  [DRY RUN] Skipped")
        return None

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    result = subprocess.run(
        [str(c) for c in cmd],
        env=merged_env,
        capture_output=False,
        cwd=WORKSPACE_DIR,
    )

    if check and result.returncode != 0:
        print(f"\nERROR: {desc} failed with return code {result.returncode}")
        sys.exit(1)

    return result


def run_command_capture(cmd, desc, dry_run=False, env=None, check=True):
    """Run a subprocess command and capture output."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n  $ {cmd_str}")

    if dry_run:
        print("  [DRY RUN] Skipped")
        return None

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    result = subprocess.run(
        [str(c) for c in cmd],
        env=merged_env,
        capture_output=True,
        text=True,
        cwd=WORKSPACE_DIR,
    )

    if check and result.returncode != 0:
        print(f"\nERROR: {desc} failed with return code {result.returncode}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
        sys.exit(1)

    return result


def panorama_stems(panorama_dir: Path) -> list[str]:
    """Stems of the panoramas to render, sorted.

    Stems carry identity through the whole pipeline -- they name the pinhole
    subdirectories and key `embeddings.pkl` -- so they are compared literally
    rather than counted.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(f.stem for f in panorama_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in exts)


def check_pinhole_reuse(config: PipelineConfig) -> bool:
    """Decide whether an existing pinhole render can be reused.

    Verifies the *contract* -- one directory per pano stem, all four faces
    present, rendered at the requested resolution -- instead of re-deriving the
    pixels. The previous version re-rendered the entire panorama set into a temp
    directory in order to hash-compare a single panorama (an acknowledged
    workaround for `panorama_to_pinhole` having no filter flag), so it paid a
    full render to decide whether it could skip a full render. Resolution is read
    back from a face JPEG header, which catches a resolution change even on the
    pre-contract renders that carry no manifest.
    """
    pinhole_dir = config.pinhole_dir
    if not pinhole_dir.exists():
        return False

    want = panorama_stems(config.panorama_dir)
    if not want:
        print(f"  ERROR: no panoramas found in {config.panorama_dir}")
        sys.exit(1)

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
        # rendered from a different panorama set, which is worth saying out loud.
        print(f"  Note: {extra} rendered stem(s) are not in "
              f"{config.panorama_dir}")

    incomplete = [s for s in want
                  if not all((pinhole_dir / s / f"{face}.jpg").exists()
                             for face in PINHOLE_FACES)]
    if incomplete:
        print(f"  Incomplete: {len(incomplete)} stem(s) missing faces "
              f"(e.g. {incomplete[0]})")
        return False

    probe = pinhole_dir / want[0] / f"{PINHOLE_FACES[0]}.jpg"
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

    print(f"  Verified {len(want)} stems x {len(PINHOLE_FACES)} faces at "
          f"{width}px - reusing")
    return True


def prebuild_targets(dry_run: bool):
    """Pre-build all bazel targets to catch errors early."""
    print("\nPre-building bazel targets...")
    if dry_run:
        for target in BAZEL_TARGETS:
            print(f"  [DRY RUN] Would build {target}")
        return

    run_command(
        ["bazel", "build"] + BAZEL_TARGETS,
        "pre-build bazel targets",
    )
    print("  All targets built successfully")


def validate_environment():
    """Check required environment variables."""
    required = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print("ERROR: Missing required environment variables:")
        for v in missing:
            print(f"  {v}")
        print("\nSet them with:")
        print("  export GOOGLE_CLOUD_PROJECT=your-project-id")
        print("  export GOOGLE_CLOUD_LOCATION=us-central1")
        sys.exit(1)


def stage_pinhole(config: PipelineConfig):
    """Stage 1: render panoramas to pinhole faces.

    A full stage rather than a prerequisite: it renders into the
    `pinhole_images` artifact lane and writes that artifact's manifest, so a leg
    that has never been rendered needs no manual preparation. An explicit
    `--pinhole_dir` now redirects where faces are written and read instead of
    silently skipping the render, which is what made a fresh dataset look like a
    configuration error.
    """
    if check_pinhole_reuse(config):
        return

    if config.pinhole_dir.exists() and not config.force and not config.dry_run:
        response = input(
            f"  Pinhole dir {config.pinhole_dir} exists but verification "
            f"failed. Re-render? [y/N]: "
        )
        if response.lower() != "y":
            print("  Aborting")
            sys.exit(1)

    config.pinhole_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "bazel", "run",
            "//experimental/overhead_matching/swag/scripts:panorama_to_pinhole",
            "--",
            str(config.panorama_dir),
            str(config.pinhole_dir),
            "--num_workers", str(config.num_pinhole_workers),
            "--res_x", str(config.pinhole_resolution),
        ],
        "panorama to pinhole conversion",
        dry_run=config.dry_run,
    )
    write_pinhole_manifest(config)


def observed_pinhole_geometry(config: PipelineConfig) -> dict:
    """Read the rendered face geometry back off disk.

    Recorded from the output rather than from the flags, so the manifest states
    what the faces *are* rather than what was requested. `panorama_to_pinhole`
    derives `res_y` from the FOV ratio and takes FOV defaults this orchestrator
    does not pass, so asserting them from here would be a guess that silently
    goes stale the first time a flag is added.
    """
    stems = panorama_stems(config.panorama_dir)
    geometry = {"n_panoramas": len(stems), "faces": list(PINHOLE_FACES)}
    if not stems:
        return geometry
    probe = config.pinhole_dir / stems[0] / f"{PINHOLE_FACES[0]}.jpg"
    try:
        from PIL import Image
        with Image.open(probe) as img:
            geometry["res_x"], geometry["res_y"] = img.width, img.height
    except OSError:
        geometry["res_x"] = config.pinhole_resolution
        geometry["res_y"] = None
        geometry["note"] = f"could not read {probe} to confirm resolution"
    return geometry


def write_pinhole_manifest(config: PipelineConfig):
    """Record how the pinhole faces were rendered, next to the faces."""
    if config.dry_run or config.paths is None:
        return
    geometry = observed_pinhole_geometry(config)
    path = config.paths.write_manifest(
        farfield_paths.PINHOLE_IMAGES,
        generator=("//experimental/overhead_matching/swag/scripts:"
                   "extract_gemini_landmarks_from_panoramas (stage 1) -> "
                   "//experimental/overhead_matching/swag/scripts:"
                   "panorama_to_pinhole"),
        config={
            **geometry,
            "requested_res_x": config.pinhole_resolution,
            "fov_deg": "panorama_to_pinhole default (90x90); not overridden "
                       "by this orchestrator",
            "layout": "one dir per pano stem, 4 face JPEGs each",
            "convention": ("face layout left-to-right in the pano is "
                           "180|90|0|270; see pano_geometry.py"),
        },
        inputs=[farfield_paths.relative_to_root(config.panorama_dir,
                                                config.paths.root)],
        notes=("Dir names are panorama stems and must match panorama/ exactly: "
               "they key embeddings.pkl and the detection ids downstream."),
    )
    print(f"  Wrote {path}")


def stage_requests(config: PipelineConfig):
    """Stage 2: Create batch API request JSONL files."""
    run_command(
        [
            "bazel", "run",
            "//experimental/overhead_matching/swag/model:semantic_landmark_extractor",
            "--",
            "create_panorama_sentences",
            "--pinhole_dir", str(config.pinhole_dir),
            "--panorama_dir", str(config.panorama_dir),
            "--output_base", str(config.sentence_requests_dir),
            "--prompt_type", config.prompt_type,
            "--num_workers", "8",
            "--media_resolution", config.media_resolution,
            "--thinking_level", config.thinking_level,
        ],
        "create panorama sentence requests",
        dry_run=config.dry_run,
    )


def new_gcs_prefix(name: str, paths) -> str:
    """A staging prefix no other invocation can collide with.

    Stage 6 downloads *everything* under the results prefix, so any two runs
    sharing a prefix silently merge: the older run's predictions land in the
    newer artifact and, because the retry logic keys on panorama stem, the
    result looks complete and healthy. Version alone is not enough -- re-running
    the same version (a --force redo, or recovering a partial run) collides too
    -- so the timestamp goes to the second.
    """
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    if paths is not None:
        version = paths.version(farfield_paths.FRAME_LANDMARKS)
        return f"{name}_{version}_{stamp}"
    return f"{name}_{stamp}"


def stage_upload(config: PipelineConfig):
    """Stage 3: Upload request JSONL files to GCS."""
    # Recorded here, at the first stage that touches GCS, so --start_stage 5 or 6
    # in a later invocation can find the staging this run actually used.
    if not config.dry_run:
        config.gcs_prefix_record.parent.mkdir(parents=True, exist_ok=True)
        config.gcs_prefix_record.write_text(config.gcs_prefix + "\n")
        print(f"  recorded staging prefix in {config.gcs_prefix_record}")

    jsonl_dir = config.sentence_requests_jsonl_dir

    if not config.dry_run:
        if not jsonl_dir.exists():
            print(f"  ERROR: Request directory not found: {jsonl_dir}")
            sys.exit(1)

        jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"  ERROR: No JSONL files found in {jsonl_dir}")
            sys.exit(1)

        print(f"  Found {len(jsonl_files)} JSONL file(s) to upload")

    run_command(
        [
            "gcloud", "storage", "cp",
            str(jsonl_dir / "*.jsonl"),
            config.gcs_requests_uri,
        ],
        "upload requests to GCS",
        dry_run=config.dry_run,
    )


def stage_submit(config: PipelineConfig):
    """Stage 4: Submit Vertex AI batch jobs.

    Captures job names from submit output and saves them to a file
    so stage_wait can track only jobs from this pipeline run.

    This is the last moment before money is spent -- the requests exist, the
    upload is free -- so the cost ceiling is enforced here.
    """
    vertex_env = {"GOOGLE_GENAI_USE_VERTEXAI": "True"}

    if not config.dry_run:
        _, _, rate_label = llm_cost.rates_for(config.model)
        estimate = llm_cost.Estimate(model=config.model, rate_label=rate_label)
        for path in sorted(config.sentence_requests_jsonl_dir.glob("*.jsonl")):
            part = llm_cost.estimate_jsonl(path, model=config.model)
            for field in ("n_requests", "prompt_tokens", "output_tokens",
                          "n_images", "text_chars", "n_large_prompts",
                          "usd_on_demand"):
                setattr(estimate, field,
                        getattr(estimate, field) + getattr(part, field))
        estimate.usd_batch = estimate.usd_on_demand * llm_cost.BATCH_MULTIPLIER
        try:
            llm_cost.enforce_limit(
                estimate, limit_usd=config.cost_limit,
                label=f"{config.name} landmark extraction",
                online=False, approved=config.approve_cost)
        except llm_cost.CostLimitExceeded as exc:
            sys.exit(f"\n{exc}")

    result = run_command_capture(
        [
            "bazel", "run",
            "//experimental/overhead_matching/swag/scripts:vertex_batch_manager",
            "--",
            "submit-all",
            "--input_prefix", config.gcs_requests_uri,
            "--output_prefix", config.gcs_results_uri,
            "--model", config.model,
            "--force",
        ],
        "submit batch jobs",
        dry_run=config.dry_run,
        env=vertex_env,
    )

    if config.dry_run:
        return

    # Print captured output so user can see it
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    # Parse job names from output lines like:
    #   ✓ Job created: projects/.../batchPredictionJobs/12345
    job_names = []
    for line in (result.stdout or "").splitlines():
        if "Job created:" in line:
            job_name = line.split("Job created:")[-1].strip()
            job_names.append(job_name)

    if not job_names:
        print("  WARNING: No job names parsed from submit output")
        return

    # Save for resumability (stage_wait can load these)
    config.job_names_file.parent.mkdir(parents=True, exist_ok=True)
    config.job_names_file.write_text("\n".join(job_names) + "\n")
    print(f"  Saved {len(job_names)} job name(s) to {config.job_names_file}")


def stage_wait(config: PipelineConfig):
    """Stage 5: Poll until all submitted batch jobs complete.

    Loads job names saved by stage_submit and polls only those jobs,
    so unrelated active jobs in the project don't block the pipeline.
    """
    vertex_env = {"GOOGLE_GENAI_USE_VERTEXAI": "True"}

    if config.dry_run:
        print("  [DRY RUN] Would poll vertex_batch_manager status for each submitted job")
        return

    # Load job names from file (written by stage_submit)
    if not config.job_names_file.exists():
        print(f"  ERROR: Job names file not found: {config.job_names_file}")
        print(f"  This file is created by stage 4 (submit). Either run from stage 4,")
        print(f"  or create the file manually with one job name per line.")
        sys.exit(1)

    job_names = [
        line.strip()
        for line in config.job_names_file.read_text().splitlines()
        if line.strip()
    ]

    if not job_names:
        print("  No job names found in file. Nothing to wait for.")
        return

    print(f"  Tracking {len(job_names)} job(s):")
    for name in job_names:
        print(f"    {name}")

    # Extract job IDs (last path component) for matching against list output.
    # Full names look like: projects/.../batchPredictionJobs/12345
    # The list command truncates long names but always preserves the trailing ID.
    job_ids = {name.rsplit("/", 1)[-1] for name in job_names}

    start_time = time.time()
    poll_count = 0

    while True:
        poll_count += 1
        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"\n  Poll #{poll_count} (elapsed: {elapsed_str})")

        # Single `list --active` call, then check if any of our job IDs appear
        result = run_command_capture(
            [
                "bazel", "run",
                "//experimental/overhead_matching/swag/scripts:vertex_batch_manager",
                "--",
                "list", "--active",
            ],
            "check active jobs",
            env=vertex_env,
            check=False,
        )

        output = result.stdout or ""

        # Find which of our jobs are still active
        still_active = [jid for jid in job_ids if jid in output]

        if not still_active:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(f"  All {len(job_names)} job(s) complete! (total wait: {elapsed_str})")
            return

        print(f"  {len(still_active)}/{len(job_ids)} job(s) still active: "
              f"{', '.join(still_active)}")
        print(f"  Waiting {config.poll_interval}s before next poll...")
        time.sleep(config.poll_interval)


def stage_download(config: PipelineConfig):
    """Stage 6: download results from GCS, then verify coverage.

    Validation lives here rather than in a stage of its own so that
    --start_stage/--end_stage numbering stays as documented: the check belongs
    to the stage that produces predictions.
    """
    config.sentences_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "gcloud", "storage", "cp", "-r",
            config.gcs_results_uri,
            str(config.sentences_dir) + "/",
        ],
        "download results from GCS",
        dry_run=config.dry_run,
    )
    stage_validate(config)


def validate_predictions(config: PipelineConfig) -> dict:
    """Classify every downloaded response: usable, empty, or failed.

    The batch API reports success at the *job* level while individual requests
    can still come back with no candidates at all -- leg2 lost 23 of 236 frames
    to transient `TPU device returned error` and the pipeline printed "Pipeline
    complete!" over it. Nothing downstream notices either: `ingest` skips a
    frame with no prediction with a bare `continue`, so tracking would simply
    see 10% of the leg as containing no objects and starve tracks crossing it.

    `empty` is not a failure. A frame of open water genuinely has no landmarks,
    and on leg2 19 frames are legitimately in that state.
    """
    paths = sorted(config.sentences_dir.glob(
        "results/*/prediction-*/predictions.jsonl"))
    report = {"files": len(paths), "ok": [], "empty": [], "failed": [],
              "unparseable": []}
    if not paths:
        return report
    # Later files win, matching ingest.load_predictions' sorted-glob dict build,
    # so a retry directory supersedes the attempt it repairs.
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

    report["n_panoramas"] = len(panorama_stems(config.panorama_dir))
    report["missing"] = [s for s in panorama_stems(config.panorama_dir)
                         if s not in by_key]
    return report


def print_validation(config: PipelineConfig, report: dict) -> bool:
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
    print("\n  Downstream will NOT complain: ingest skips a frame with no "
          "prediction silently, so tracking would read these as frames "
          "containing no objects.")
    print(f"\n  Repair with:\n    bazel run //experimental/overhead_matching/"
          f"swag/scripts:extract_gemini_landmarks_from_panoramas -- \\\n"
          f"        --dataset {config.paths.dataset if config.paths else ''} "
          f"--retry_failed")
    return False


def stage_validate(config: PipelineConfig):
    """Gate before embeddings: refuse to certify an incomplete extraction."""
    if config.dry_run:
        print("  [DRY RUN] Would validate downloaded responses")
        return
    report = validate_predictions(config)
    complete = print_validation(config, report)
    if not complete and not config.allow_incomplete:
        print("\n  Stopping. Pass --allow_incomplete to build embeddings and "
              "write a manifest anyway; the manifest records the gap.")
        sys.exit(1)


def retry_failed(config: PipelineConfig):
    """Re-run only the requests that came back without a usable response.

    Written as an *additional* predictions file rather than by editing the
    original: `ingest.load_predictions` builds a dict over a sorted glob, so a
    directory sorting after the first attempt supersedes it key by key. The
    failed attempt stays on disk as the record of what happened, and no
    published file is mutated.
    """
    report = validate_predictions(config)
    broken = report["failed"] + report["missing"]
    if not broken:
        print("Nothing to retry: every panorama has a usable response.")
        return
    print(f"retrying {len(broken)} request(s)")

    wanted = set(broken)
    subset = []
    for path in sorted(config.sentence_requests_jsonl_dir.glob("*.jsonl")):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("key") in wanted:
                    subset.append(record)
    if len(subset) != len(wanted):
        found = {r.get("key") for r in subset}
        print(f"  WARNING: {len(wanted - found)} failed key(s) have no request "
              f"on disk; sentence_requests/ may have been deleted")
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

    if config.dry_run:
        print("  [DRY RUN] Would run these online")
        return

    # Online rather than batch: a handful of requests does not justify a batch
    # job's queue latency, and run-online is itself resumable on error.
    run_command(
        [
            "bazel", "run",
            "//experimental/overhead_matching/swag/scripts:vertex_batch_manager",
            "--", "run-online",
            "--input", str(requests_path),
            "--output", str(retry_dir / "predictions.jsonl"),
            "--model", config.model,
            "--parallel", "4",
        ],
        "retry failed requests",
        env={"GOOGLE_GENAI_USE_VERTEXAI": "True"},
    )
    print("\nafter retry:")
    print_validation(config, validate_predictions(config))


def stage_embeddings(config: PipelineConfig):
    """Stage 7: Create embeddings from batch results."""
    vertex_env = {"GOOGLE_GENAI_USE_VERTEXAI": "True"}

    config.embeddings_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            "bazel", "run",
            "//experimental/overhead_matching/swag/scripts:create_embeddings_with_gemini",
            "--",
            "--mode", "panorama",
            "--input_dir", str(config.sentences_dir),
            "--output_file", str(config.embeddings_file),
            "--model", config.embedding_model,
            "--force",
        ],
        "create embeddings",
        dry_run=config.dry_run,
        env=vertex_env,
    )


STAGE_FUNCS = {
    Stage.PINHOLE: stage_pinhole,
    Stage.REQUESTS: stage_requests,
    Stage.UPLOAD: stage_upload,
    Stage.SUBMIT: stage_submit,
    Stage.WAIT: stage_wait,
    Stage.DOWNLOAD: stage_download,
    Stage.EMBEDDINGS: stage_embeddings,
}


def request_fingerprint(config: PipelineConfig) -> dict:
    """Hash what was actually sent to the model.

    `prompt: osm_tags_farfield` names a key in `SYSTEM_PROMPTS`, and that key's
    *text* can be edited without the name changing -- so the name alone does not
    pin the extraction. The request JSONL holds the prompt as it went out,
    alongside every image reference, so hashing those files pins the whole
    request: change the prompt, the resolution, or the pano set, and the digest
    moves.
    """
    files = sorted(config.sentence_requests_jsonl_dir.glob("*.jsonl"))
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


def prompt_fingerprint(config: PipelineConfig) -> dict:
    """Hash the system prompt TEXT alone, as it was actually sent.

    `request_sha256` pins the whole request, images included, so two datasets
    extracted with an identical prompt still get different digests and the
    manifest cannot answer "did these two runs use the same prompt?". That
    question cost a hand comparison across 4.4 GB of stored request JSONL when
    the boston_harbor_leg1 v1 -> v4 regression was traced to a silent rewrite
    of `osm_tags_farfield`. This digest answers it directly:
    `grep prompt_sha256 artifacts/frame_landmarks/*/*/manifest.json`.

    Read from the request file rather than from SYSTEM_PROMPTS so it records
    what went out, not what the tree says now.
    """
    files = sorted(config.sentence_requests_jsonl_dir.glob("*.jsonl"))
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


def landmark_counts(config: PipelineConfig) -> dict:
    """How many predictions came back, for a size check against the panos."""
    files = sorted(config.sentences_dir.rglob("predictions.jsonl"))
    if not files:
        return {"note": "predictions.jsonl absent at manifest time"}
    lines = 0
    for path in files:
        with open(path, "rb") as handle:
            lines += sum(1 for _ in handle)
    return {"n_prediction_files": len(files), "n_prediction_lines": lines}


def coverage_summary(config: PipelineConfig) -> dict:
    """Response coverage, recorded so an accepted gap stays visible.

    A consumer reading this manifest can otherwise not tell a leg with genuinely
    few landmarks from one that lost frames to API errors -- both look like
    frames with no observations.
    """
    report = validate_predictions(config)
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


def write_frame_landmarks_manifest(config: PipelineConfig):
    """Record the extraction that produced the `frame_landmarks` artifact.

    Written only when the run actually reached the embeddings stage, so a
    partial run (`--end_stage 2`) never leaves a manifest claiming a complete
    artifact.
    """
    if config.dry_run or config.paths is None:
        return
    root = config.paths.root
    path = config.paths.write_manifest(
        farfield_paths.FRAME_LANDMARKS,
        generator=("//experimental/overhead_matching/swag/scripts:"
                   "extract_gemini_landmarks_from_panoramas"),
        config={
            "prompt": config.prompt_type,
            # The name is a lookup key whose text can change under it; the
            # digest is what actually pins this extraction.
            **prompt_fingerprint(config),
            **request_fingerprint(config),
            "model": config.model,
            "embedding_model": (config.embedding_model
                                if config.end_stage >= Stage.EMBEDDINGS
                                else "not run (stage 7 skipped)"),
            "pinhole_resolution": config.pinhole_resolution,
            "media_resolution": config.media_resolution,
            "thinking_level": config.thinking_level,
            "gcs_prefix": config.gcs_prefix,
            **landmark_counts(config),
            **coverage_summary(config),
        },
        inputs=[
            farfield_paths.relative_to_root(config.paths.dataset_base, root),
            farfield_paths.relative_to_root(config.pinhole_dir, root),
        ],
        notes=("Consumers glob sentences/results/**/predictions.jsonl "
               "(ingest.py) and read embeddings/embeddings.pkl keyed by FULL "
               "panorama stem (semantic_similarity.py). DO NOT DELETE "
               "sentence_requests/: no stage reads it back, but each request "
               "line carries the system prompt verbatim, which makes it the "
               "only record of the prompt text this artifact was built with - "
               "`prompt` names a SYSTEM_PROMPTS key whose text changes over "
               "time, and git_commit pins the tree rather than the working "
               "copy that ran. It is also what request_sha256 is computed "
               "over, so the digest cannot be recomputed once it is gone."),
    )
    print(f"  Wrote {path}")


def run_pipeline(config: PipelineConfig):
    """Run the pipeline from start_stage to end_stage."""
    print("=" * 70)
    print("Gemini Landmark Extraction Pipeline")
    print("=" * 70)
    print(f"  Name:           {config.name}")
    print(f"  Panorama dir:   {config.panorama_dir}")
    print(f"  Pinhole dir:    {config.pinhole_dir}")
    print(f"  Artifact dir:   {config.artifact_dir}")
    print(f"  Pinhole res:    {config.pinhole_resolution}")
    print(f"  Prompt type:    {config.prompt_type}")
    print(f"  Media res:      {config.media_resolution}")
    print(f"  Thinking level: {config.thinking_level}")
    print(f"  Model:          {config.model}")
    if config.end_stage >= Stage.EMBEDDINGS:
        print(f"  Embedding model: {config.embedding_model}")
    print(f"  GCS prefix:     {config.gcs_prefix}")
    print(f"  Stages:         {config.start_stage} -> {config.end_stage}")
    if config.paths is None:
        print("  Mode:           legacy (explicit dirs, no manifests)")
    if config.dry_run:
        print(f"  *** DRY RUN MODE ***")
    print()

    # Validate environment for stages that need GCP access
    if config.start_stage <= Stage.SUBMIT and config.end_stage >= Stage.SUBMIT:
        validate_environment()

    # Pre-build all targets
    stages_to_run = [
        s for s in Stage if config.start_stage <= s <= config.end_stage
    ]
    prebuild_targets(config.dry_run)

    # Run stages
    pipeline_start = time.time()
    for stage in stages_to_run:
        print(f"\n{'=' * 70}")
        print(f"STAGE {stage.value}: {STAGE_NAMES[stage]}")
        print(f"{'=' * 70}")

        stage_start = time.time()
        STAGE_FUNCS[stage](config)
        stage_elapsed = time.time() - stage_start

        print(f"\n  Stage {stage.value} completed in {stage_elapsed:.1f}s")


    # The manifest certifies the artifact, so it is written once the stage that
    # produces predictions has run -- not once embeddings have, which are
    # optional and which nothing in the tracking pipeline reads.
    if config.end_stage >= Stage.DOWNLOAD:
        write_frame_landmarks_manifest(config)

    pipeline_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete! Total time: {pipeline_elapsed:.1f}s")
    if not config.dry_run and config.end_stage >= Stage.DOWNLOAD:
        print(f"frame_landmarks artifact: {config.artifact_dir}")
    if not config.dry_run and config.end_stage >= Stage.EMBEDDINGS:
        print(f"Embeddings saved to: {config.embeddings_file}")
    print(f"{'=' * 70}")


def main():
    # Stages run for tens of minutes and print progress as they go. Redirected
    # to a file, Python block-buffers stdout, so a live run looks frozen for
    # 8 KB at a time -- which is indistinguishable from a hang exactly when you
    # most want to know it is still working.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="End-to-end Gemini landmark extraction from panoramas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    farfield_paths.add_arguments(parser, pinhole=True)
    parser.add_argument(
        "--name", default=None,
        help="Run name for the GCS prefix (default: the dataset name). "
             "Required in legacy mode.",
    )
    parser.add_argument(
        "--panorama_dir", type=Path, default=None,
        help="Input directory of panorama JPGs (default: resolved from "
             "--dataset)",
    )
    parser.add_argument(
        "--output_base", type=Path, default=None,
        help="LEGACY: flat output root, producing <output_base>/<name>/. Use "
             "--dataset instead so output lands in the frame_landmarks "
             "artifact lane.",
    )
    parser.add_argument(
        "--start_stage", type=int, default=1,
        help="Resume from this stage (1-7, default: 1)",
    )
    parser.add_argument(
        "--end_stage", type=int, default=None,
        help="Stop after this stage (1-7). Default: 6 (download), because "
             "nothing in the tracking pipeline reads embeddings.pkl - "
             "ingest.py reads predictions.jsonl. Use --with_embeddings for 7.",
    )
    parser.add_argument(
        "--with_embeddings", action="store_true",
        help="Also run stage 7 (embeddings.pkl). Only the older cosine matcher "
             "in landmark_filtering/semantic_similarity.py consumes it; the "
             "m0-m11 tracking stages do not.",
    )
    parser.add_argument(
        "--prompt_type", default="osm_tags",
        choices=["osm_tags", "panorama", "osm_tags_farfield",
                 "osm_tags_farfield_v2"],
        help="Prompt type (default: osm_tags)",
    )
    parser.add_argument(
        "--model", default="gemini-3-flash-preview",
        help="Gemini model for batch inference (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--gcs_bucket", default="crossview",
        help="GCS bucket (default: crossview)",
    )
    parser.add_argument(
        "--gcs_prefix", default=None,
        help="Pin the staging prefix under the bucket. Default: a fresh "
             "<name>_<version>_<YYMMDD_HHMMSS>, recorded in the artifact as "
             "gcs_prefix.txt so --start_stage 5/6 resumes against the same "
             "staging. Pass explicitly to adopt another run's staging.",
    )
    parser.add_argument(
        "--num_pinhole_workers", type=int, default=8,
        help="Workers for panorama-to-pinhole (default: 8)",
    )
    parser.add_argument(
        "--poll_interval", type=int, default=120,
        help="Seconds between batch job polling (default: 120)",
    )
    parser.add_argument(
        "--pinhole_resolution", type=int, default=1024,
        help="Pinhole image resolution in pixels (default: 1024)",
    )
    parser.add_argument(
        "--media_resolution", default="MEDIA_RESOLUTION_HIGH",
        choices=["MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM",
                 "MEDIA_RESOLUTION_HIGH", "MEDIA_RESOLUTION_ULTRA_HIGH"],
        help="Media resolution for Gemini image processing (default: MEDIA_RESOLUTION_HIGH)",
    )
    parser.add_argument(
        "--embedding_model", default="gemini-embedding-001",
        help="Vertex embedding model for stage 7 (default: "
             "gemini-embedding-001)",
    )
    parser.add_argument(
        "--thinking_level", default="HIGH",
        choices=["OFF", "LOW", "MEDIUM", "HIGH"],
        help="Thinking level for Gemini (default: HIGH)",
    )
    parser.add_argument(
        "--cost_limit", type=float, default=50.0,
        help="Refuse to submit if the estimated cost of this extraction "
             "exceeds this many USD until approved (default: 50)",
    )
    parser.add_argument(
        "--approve_cost", action="store_true",
        help="Approve an extraction that exceeds --cost_limit",
    )
    parser.add_argument(
        "--allow_incomplete", action="store_true",
        help="Build embeddings and write a manifest even when some panoramas "
             "have no usable response (the gap is recorded in the manifest)",
    )
    parser.add_argument(
        "--retry_failed", action="store_true",
        help="Re-run only the requests with no usable response and stop. "
             "Results are written as an additional predictions file that "
             "supersedes the failed attempt; nothing is overwritten.",
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Report response coverage for an existing extraction and stop",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip confirmation prompts",
    )

    args = parser.parse_args()

    if args.end_stage is None:
        args.end_stage = Stage.EMBEDDINGS if args.with_embeddings \
            else Stage.DOWNLOAD

    # Validate stage range
    if not (1 <= args.start_stage <= 7):
        parser.error("--start_stage must be between 1 and 7")
    if not (1 <= args.end_stage <= 7):
        parser.error("--end_stage must be between 1 and 7")
    if args.start_stage > args.end_stage:
        parser.error("--start_stage must be <= --end_stage")

    config = build_config(args, parser)

    if args.validate_only:
        print_validation(config, validate_predictions(config))
        return
    if args.retry_failed:
        retry_failed(config)
        return

    run_pipeline(config)


def build_config(args, parser) -> PipelineConfig:
    """Resolve the two modes down to one config.

    Farfield mode (`--dataset`) resolves panoramas, the pinhole artifact and the
    frame_landmarks artifact from the disk layout. Legacy mode
    (`--panorama_dir` + `--output_base`) keeps the flat `<output_base>/<name>`
    layout that the pre-artifact VIGOR sets are stored in; it writes no
    manifests, because nothing reads one there.
    """
    if args.dataset:
        if args.output_base:
            parser.error(
                "--output_base is the legacy flat layout and cannot be combined "
                "with --dataset; drop it to write into the frame_landmarks "
                "artifact lane, or drop --dataset to use it.")
        paths = farfield_paths.from_args(args)
        panorama_dir = args.panorama_dir or paths.panorama_dir
        pinhole_dir = paths.pinhole_images
        artifact_dir = paths.frame_landmarks
        name = args.name or paths.dataset
        try:
            paths.require("dataset_base", "panorama_dir")
        except farfield_paths.MissingInput as exc:
            parser.error(str(exc))
    else:
        if not (args.name and args.panorama_dir and args.output_base):
            parser.error(
                "pass --dataset (farfield lanes), or all of --name, "
                "--panorama_dir and --output_base (legacy flat layout)")
        paths = None
        panorama_dir = args.panorama_dir
        # Legacy pinhole location: a sibling of the flat output root, so a set
        # outside the farfield lanes stays self-contained instead of writing
        # into a global pinhole directory keyed only by name.
        pinhole_dir = (getattr(args, "pinhole_dir", None) or
                       args.output_base / args.name / "pinhole_images")
        artifact_dir = args.output_base / args.name
        name = args.name

    if args.start_stage == Stage.PINHOLE and not Path(panorama_dir).is_dir():
        parser.error(f"Panorama directory does not exist: {panorama_dir}")

    # A resume must reach the staging the earlier invocation uploaded to; a fresh
    # run must not be able to land on anyone else's.
    recorded = Path(artifact_dir) / "gcs_prefix.txt"
    if args.gcs_prefix:
        gcs_prefix = args.gcs_prefix
    elif args.start_stage > Stage.UPLOAD and recorded.exists():
        gcs_prefix = recorded.read_text().strip()
        print(f"resuming against recorded staging prefix {gcs_prefix}")
    elif args.start_stage > Stage.UPLOAD:
        parser.error(
            f"--start_stage {args.start_stage} needs the staging prefix the "
            f"upload stage used, but {recorded} does not exist. Pass "
            f"--gcs_prefix explicitly (see gs://{args.gcs_bucket}/ for the "
            f"run's requests/ and results/ directories).")
    else:
        gcs_prefix = new_gcs_prefix(name, paths)

    return PipelineConfig(
        name=name,
        panorama_dir=Path(panorama_dir),
        pinhole_dir=Path(pinhole_dir),
        artifact_dir=Path(artifact_dir),
        prompt_type=args.prompt_type,
        model=args.model,
        gcs_bucket=args.gcs_bucket,
        num_pinhole_workers=args.num_pinhole_workers,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
        force=args.force,
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        pinhole_resolution=args.pinhole_resolution,
        media_resolution=args.media_resolution,
        thinking_level=args.thinking_level,
        embedding_model=args.embedding_model,
        allow_incomplete=args.allow_incomplete,
        gcs_prefix=gcs_prefix,
        cost_limit=args.cost_limit,
        approve_cost=args.approve_cost,
        paths=paths,
    )


if __name__ == "__main__":
    main()
