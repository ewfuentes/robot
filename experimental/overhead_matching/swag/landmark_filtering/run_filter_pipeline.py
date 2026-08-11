"""CLI driver for the landmark filtering pipeline.

Every --stage prefix emits a complete, valid run artifact (later-stage fields
empty), so each pipeline milestone can be inspected in the viewer.

Example:
    bazel run //experimental/overhead_matching/swag/landmark_filtering:run_filter_pipeline -- \\
        --dataset_base /data/overhead_matching/datasets/walk_along_river_7_1_26 \\
        --pinhole_base /data/overhead_matching/datasets/pinhole_images/walk_along_river_7_1_26 \\
        --landmark_base /data/overhead_matching/datasets/semantic_landmark_embeddings/panov2_tuned_prompt/walk_along_river_7_1_26 \\
        --output /tmp/filter_runs/walk_river_ingest.json --stage ingest
"""

import argparse
import datetime
import os
import subprocess
from pathlib import Path

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    ingest,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    FilterPipelineConfig,
    load_config,
)

STAGE_ORDER = ["ingest", "a", "track", "all"]


def get_git_hash() -> str:
    # bazel run sets BUILD_WORKSPACE_DIRECTORY to the source workspace.
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    try:
        return subprocess.check_output(
            ["git", "-C", workspace or ".", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, TypeError):
        return "unknown"


def apply_yaw_offset(artifact: schema.RunArtifact, offset_deg: float,
                     method: str, drift_deg_per_frame: float = 0.0) -> None:
    artifact.yaw_offset_deg = offset_deg
    artifact.yaw_drift_deg_per_frame = drift_deg_per_frame
    artifact.yaw_offset_method = method
    sign = artifact.config.yaw_offset.bearing_sign
    for obs in artifact.observations:
        obs.bearing_global_deg = (
            sign * obs.bearing_camera_deg + offset_deg
            + drift_deg_per_frame * obs.frame_idx) % 360.0


def run_semantic_diagnostics(artifact: schema.RunArtifact,
                             config: FilterPipelineConfig,
                             landmark_base: Path,
                             missing_report_path: Path | None,
                             device: str) -> None:
    from experimental.overhead_matching.swag.landmark_filtering import (
        semantic_similarity,
    )

    sem_config = config.semantic_similarity
    backends = []
    for name in sem_config.diagnostic_backends:
        try:
            backends.append(semantic_similarity.make_backend(
                name, landmark_base, sem_config, artifact.observations,
                missing_report_path, device))
        except semantic_similarity.MissingTextEmbeddingsError as err:
            print("=" * 70)
            print(f"SEMANTIC BACKEND '{name}' SKIPPED: {err}")
            print("=" * 70)
            artifact.semantic_diagnostics.append(schema.SemanticDiagnostics(
                backend=name,
                missing_embedding_values=err.missing_values))
    for backend in backends:
        artifact.semantic_diagnostics.append(
            semantic_similarity.compute_diagnostics(
                artifact.observations, backend,
                sem_config.num_example_pairs,
                sem_config.max_diagnostic_obs))
    for i in range(len(backends)):
        for j in range(i + 1, len(backends)):
            artifact.backend_agreements.append(
                semantic_similarity.compute_backend_agreement(
                    artifact.observations, backends[i], backends[j],
                    sem_config.num_example_pairs,
                    sem_config.max_diagnostic_obs))


def run_pipeline(config: FilterPipelineConfig, dataset_base: Path,
                 pinhole_base: Path, landmark_base: Path, stage: str,
                 yaw_offset_override: float | None = None,
                 device: str = "cpu",
                 missing_report_path: Path | None = None) -> schema.RunArtifact:
    ingest_result = ingest.run_ingest(
        dataset_base, landmark_base, config.ingest)
    artifact = schema.RunArtifact(
        schema_version=schema.SCHEMA_VERSION,
        created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        git_hash=get_git_hash(),
        dataset_base=str(dataset_base),
        pinhole_base=str(pinhole_base),
        landmark_base=str(landmark_base),
        config=config,
        stages_run=["ingest"],
        anchor_lat=ingest_result.anchor_lat,
        anchor_lon=ingest_result.anchor_lon,
        frames=ingest_result.frames,
        observations=ingest_result.observations,
        stats=ingest_result.stats,
    )

    if stage in ("a", "track", "all"):
        from experimental.overhead_matching.swag.landmark_filtering import (
            heuristic_filters,
        )
        heuristic_filters.run_stage_a(artifact, config.heuristic)
        artifact.stages_run.append("heuristic")

    if stage in ("track", "all"):
        from experimental.overhead_matching.swag.landmark_filtering import (
            tracking,
        )
        tracking.run_tracking(artifact, config, device=device)
        artifact.stages_run.append("tracking")
        run_semantic_diagnostics(
            artifact, config, landmark_base, missing_report_path, device)
        artifact.stages_run.append("semantic_diagnostics")

    if stage == "all":
        from experimental.overhead_matching.swag.landmark_filtering import (
            triangulation,
            yaw_offset,
        )
        offset_deg, drift, method, details = yaw_offset.estimate_yaw_offset(
            artifact, config, yaw_offset_override)
        apply_yaw_offset(artifact, offset_deg, method, drift)
        artifact.yaw_offset_details = details
        triangulation.run_triangulation(artifact, config.triangulation)
        artifact.stages_run.extend(["yaw_offset", "triangulation"])
    else:
        offset = (yaw_offset_override if yaw_offset_override is not None
                  else config.yaw_offset.fixed_offset_deg)
        apply_yaw_offset(artifact, offset, "fixed")

    return artifact


def print_summary(artifact: schema.RunArtifact) -> None:
    stats = artifact.stats
    print(f"stages run:        {', '.join(artifact.stages_run)}")
    print(f"frames:            {stats.n_frames}")
    print(f"raw landmarks:     {stats.n_raw_landmark_entries} "
          f"(parse failures: {stats.n_parse_failures}, "
          f"invalid-yaw boxes: {stats.n_boxes_invalid_yaw}, "
          f"boxless landmarks: {stats.n_landmarks_without_valid_boxes})")
    print(f"observations:      {stats.n_observations} "
          f"(kept: {stats.n_kept}, filtered: {stats.n_filtered})")
    n_seam = sum(1 for o in artifact.observations if o.seam_merged)
    print(f"seam-merged obs:   {n_seam}")
    if stats.filtered_by_reason:
        print("filtered by reason:")
        for reason, count in sorted(stats.filtered_by_reason.items(),
                                    key=lambda kv: -kv[1]):
            print(f"  {reason:40s} {count}")
    if stats.n_tracks:
        print(f"tracks:            {stats.n_tracks} "
              f"(by observability: {stats.tracks_by_observability})")
    print(f"yaw offset:        {artifact.yaw_offset_deg:.1f} deg "
          f"+ {artifact.yaw_drift_deg_per_frame:.3f} deg/frame "
          f"({artifact.yaw_offset_method})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_base", type=Path, required=True)
    parser.add_argument("--pinhole_base", type=Path, required=True)
    parser.add_argument("--landmark_base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML FilterPipelineConfig; defaults used if omitted")
    parser.add_argument("--stage", choices=STAGE_ORDER, default="all")
    parser.add_argument("--yaw_offset", type=float, default=None,
                        help="Override yaw offset in degrees (forces fixed)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = (load_config(args.config) if args.config
              else FilterPipelineConfig())
    artifact = run_pipeline(
        config, args.dataset_base, args.pinhole_base, args.landmark_base,
        args.stage, args.yaw_offset, args.device,
        missing_report_path=args.output.with_suffix(".missing_values.txt"))
    schema.save_artifact(artifact, args.output)
    print_summary(artifact)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
