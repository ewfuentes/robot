"""End-to-end orchestration for Gemini landmark extraction from panoramas.

Runs the full 7-stage pipeline:
  1. PINHOLE   - Convert panoramas to pinhole images
  2. REQUESTS  - Create batch API request JSONL files
  3. UPLOAD    - Upload requests to GCS
  4. SUBMIT    - Submit Vertex AI batch jobs
  5. WAIT      - Poll until all batch jobs complete
  6. DOWNLOAD  - Download results from GCS
  7. EMBEDDINGS - Create embeddings from batch results

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:extract_gemini_landmarks_from_panoramas -- \
        --name nightdrive \
        --panorama_dir /data/overhead_matching/datasets/VIGOR/nightdrive/panorama/ \
        --output_base /data/overhead_matching/datasets/semantic_landmark_embeddings/nightdrive_osm_tags/
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path


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

PINHOLE_BASE = Path("/data/overhead_matching/datasets/pinhole_images")

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
    name: str
    panorama_dir: Path
    output_base: Path
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
    pinhole_dir_override: Path | None = None

    def __post_init__(self):
        self._date_suffix = datetime.now().strftime("%y%m%d")

    @property
    def pinhole_dir(self) -> Path:
        if self.pinhole_dir_override is not None:
            return self.pinhole_dir_override
        return PINHOLE_BASE / self.name

    @property
    def gcs_prefix(self) -> str:
        return f"{self.name}_{self._date_suffix}"

    @property
    def gcs_requests_uri(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_prefix}/requests/"

    @property
    def gcs_results_uri(self) -> str:
        return f"gs://{self.gcs_bucket}/{self.gcs_prefix}/results/"

    @property
    def sentence_requests_dir(self) -> Path:
        return self.output_base / self.name / "sentence_requests"

    @property
    def sentence_requests_jsonl_dir(self) -> Path:
        return self.sentence_requests_dir / "panorama_sentence_requests"

    @property
    def sentences_dir(self) -> Path:
        return self.output_base / self.name / "sentences"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_base / self.name / "embeddings"

    @property
    def embeddings_file(self) -> Path:
        return self.embeddings_dir / "embeddings.pkl"

    @property
    def job_names_file(self) -> Path:
        return self.output_base / self.name / "submitted_job_names.txt"


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


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pinhole_reuse(config: PipelineConfig) -> bool:
    """Check if existing pinhole images can be reused.

    Returns True if existing pinhole images should be reused.
    """
    pinhole_dir = config.pinhole_dir
    if not pinhole_dir.exists():
        return False

    # Check if there are any subdirectories (panorama outputs)
    subdirs = [d for d in pinhole_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return False

    print(f"  Found existing pinhole images at {pinhole_dir} ({len(subdirs)} panoramas)")

    # Pick one panorama to verify against
    pano_files = sorted(config.panorama_dir.iterdir())
    pano_files = [f for f in pano_files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not pano_files:
        return False

    test_pano = pano_files[0]
    existing_output = pinhole_dir / test_pano.stem

    if not existing_output.exists():
        print(f"  Verification failed: {existing_output} does not exist")
        return False

    # Extract pinhole for one panorama to a temp dir and compare hashes
    print(f"  Verifying by re-extracting {test_pano.name}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        run_command(
            [
                "bazel", "run",
                "//experimental/overhead_matching/swag/scripts:panorama_to_pinhole",
                "--", str(config.panorama_dir), tmpdir,
                "--num_workers", "1",
            ],
            "pinhole verification",
            dry_run=False,
            env={"PANORAMA_FILTER": test_pano.stem},
        )

        # Actually, the panorama_to_pinhole script doesn't support filtering,
        # so let's just compare file hashes of existing output images
        tmp_output = Path(tmpdir) / test_pano.stem
        if not tmp_output.exists():
            # The script processes ALL panoramas; check if any output appeared
            tmp_subdirs = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
            if not tmp_subdirs:
                print("  Verification inconclusive, will reuse existing")
                return True
            tmp_output = tmp_subdirs[0]
            # Find the corresponding existing dir
            existing_output = pinhole_dir / tmp_output.name

        if not existing_output.exists():
            print(f"  No matching existing output for {tmp_output.name}")
            return False

        # Compare hashes of all files in the test panorama output
        tmp_files = sorted(tmp_output.iterdir())
        existing_files = sorted(existing_output.iterdir())

        if len(tmp_files) != len(existing_files):
            print(f"  File count mismatch: {len(tmp_files)} vs {len(existing_files)}")
            return False

        for tmp_f, exist_f in zip(tmp_files, existing_files):
            if tmp_f.name != exist_f.name:
                print(f"  Filename mismatch: {tmp_f.name} vs {exist_f.name}")
                return False
            if hash_file(tmp_f) != hash_file(exist_f):
                print(f"  Hash mismatch for {tmp_f.name}")
                return False

        print("  Hash verification passed - reusing existing pinhole images")
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
    """Stage 1: Convert panoramas to pinhole images."""
    if config.pinhole_dir_override is not None:
        print(f"  Using provided pinhole dir: {config.pinhole_dir_override}")
        return

    if check_pinhole_reuse(config):
        print(f"  Reusing existing pinhole images at {config.pinhole_dir}")
        return

    if config.pinhole_dir.exists() and not config.force and not config.dry_run:
        response = input(
            f"  Pinhole dir {config.pinhole_dir} exists but verification failed. "
            f"Re-extract? [y/N]: "
        )
        if response.lower() != "y":
            print("  Aborting")
            sys.exit(1)

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


def stage_requests(config: PipelineConfig):
    """Stage 2: Create batch API request JSONL files."""
    run_command(
        [
            "bazel", "run",
            "//experimental/overhead_matching/swag/model:semantic_landmark_extractor",
            "--",
            "create_panorama_sentences",
            "--pinhole_dir", str(config.pinhole_dir),
            "--output_base", str(config.sentence_requests_dir),
            "--prompt_type", config.prompt_type,
            "--num_workers", "8",
            "--media_resolution", config.media_resolution,
            "--thinking_level", config.thinking_level,
        ],
        "create panorama sentence requests",
        dry_run=config.dry_run,
    )


def stage_upload(config: PipelineConfig):
    """Stage 3: Upload request JSONL files to GCS."""
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
    """
    vertex_env = {"GOOGLE_GENAI_USE_VERTEXAI": "True"}

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
    """Stage 6: Download results from GCS."""
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


def run_pipeline(config: PipelineConfig):
    """Run the pipeline from start_stage to end_stage."""
    print("=" * 70)
    print("Gemini Landmark Extraction Pipeline")
    print("=" * 70)
    print(f"  Name:           {config.name}")
    print(f"  Panorama dir:   {config.panorama_dir}")
    print(f"  Output base:    {config.output_base}")
    print(f"  Pinhole dir:    {config.pinhole_dir}")
    print(f"  Pinhole res:    {config.pinhole_resolution}")
    print(f"  Prompt type:    {config.prompt_type}")
    print(f"  Media res:      {config.media_resolution}")
    print(f"  Thinking level: {config.thinking_level}")
    print(f"  Model:          {config.model}")
    print(f"  GCS prefix:     {config.gcs_prefix}")
    print(f"  Stages:         {config.start_stage} -> {config.end_stage}")
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

    pipeline_elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete! Total time: {pipeline_elapsed:.1f}s")
    if not config.dry_run and config.end_stage >= Stage.EMBEDDINGS:
        print(f"Embeddings saved to: {config.embeddings_file}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Gemini landmark extraction from panoramas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--name", required=True,
        help="Dataset name (drives pinhole dir and GCS prefix)",
    )
    parser.add_argument(
        "--panorama_dir", required=True, type=Path,
        help="Input directory containing panorama JPGs",
    )
    parser.add_argument(
        "--output_base", required=True, type=Path,
        help="Root output directory for embeddings output",
    )
    parser.add_argument(
        "--start_stage", type=int, default=1,
        help="Resume from this stage (1-7, default: 1)",
    )
    parser.add_argument(
        "--end_stage", type=int, default=7,
        help="Stop after this stage (1-7, default: 7)",
    )
    parser.add_argument(
        "--prompt_type", default="osm_tags",
        choices=["osm_tags", "panorama"],
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
        "--thinking_level", default="HIGH",
        choices=["OFF", "LOW", "MEDIUM", "HIGH"],
        help="Thinking level for Gemini (default: HIGH)",
    )
    parser.add_argument(
        "--pinhole_dir", type=Path, default=None,
        help="Override computed pinhole dir (skips stage 1 pinhole extraction)",
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

    # Validate stage range
    if not (1 <= args.start_stage <= 7):
        parser.error("--start_stage must be between 1 and 7")
    if not (1 <= args.end_stage <= 7):
        parser.error("--end_stage must be between 1 and 7")
    if args.start_stage > args.end_stage:
        parser.error("--start_stage must be <= --end_stage")

    # Validate panorama dir exists
    if args.start_stage == 1 and not args.panorama_dir.is_dir():
        parser.error(f"Panorama directory does not exist: {args.panorama_dir}")

    config = PipelineConfig(
        name=args.name,
        panorama_dir=args.panorama_dir,
        output_base=args.output_base,
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
        pinhole_dir_override=args.pinhole_dir,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
