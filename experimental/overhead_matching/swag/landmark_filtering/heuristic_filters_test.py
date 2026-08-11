import unittest

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
    heuristic_filters,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    FilterPipelineConfig,
    HeuristicConfig,
    IntraFrameDedupConfig,
)


def make_obs(obs_id="f0000__lm0__box0", frame_idx=0, primary=("building",
             "yes"), confidence="high", bearing=10.0, elevation=0.0,
             width=10.0, boxes=None, seam_merged=False):
    if boxes is None:
        boxes = [schema.BBox(face_yaw_deg=0, xmin=400, ymin=400, xmax=600,
                             ymax=600)]
    return schema.Observation(
        obs_id=obs_id, pano_id=obs_id.split("__")[0], frame_idx=frame_idx,
        landmark_idx=int(obs_id.split("__lm")[1].split("__")[0]),
        embedding_id=None, primary_tag_key=primary[0],
        primary_tag_value=primary[1], additional_tags=[],
        confidence=confidence, description="", boxes=boxes,
        seam_merged=seam_merged, bearing_camera_deg=bearing,
        bearing_global_deg=bearing, elevation_deg=elevation,
        angular_width_deg=width, decisions=[])


def make_artifact(observations):
    frames = [schema.Frame(
        frame_idx=i, pano_id=f"f{i:04d}", pano_stem=f"f{i:04d},42,-71,",
        lat=42.0, lon=-71.0, x_m=0.0, y_m=0.0)
        for i in range(max((o.frame_idx for o in observations), default=0)
                       + 1)]
    return schema.RunArtifact(
        schema_version=schema.SCHEMA_VERSION, created_at="", git_hash="",
        dataset_base="", pinhole_base="", landmark_base="",
        config=FilterPipelineConfig(), stages_run=["ingest"],
        anchor_lat=42.0, anchor_lon=-71.0, frames=frames,
        observations=observations,
        stats=schema.SummaryStats(n_observations=len(observations)))


class HeuristicFiltersTest(unittest.TestCase):
    def test_annotate_not_delete(self):
        artifact = make_artifact([
            make_obs(confidence="low"),
            make_obs(obs_id="f0000__lm1__box0", confidence="high"),
        ])
        heuristic_filters.run_stage_a(artifact, HeuristicConfig())
        self.assertEqual(len(artifact.observations), 2)
        self.assertEqual(artifact.observations[0].final_disposition,
                         schema.FILTERED)
        self.assertEqual(artifact.observations[0].final_reason,
                         "confidence_low")
        self.assertEqual(artifact.observations[1].final_disposition,
                         schema.KEPT)
        self.assertEqual(artifact.stats.n_kept, 1)
        self.assertEqual(artifact.stats.n_filtered, 1)

    def test_full_trail_recorded_for_all_filters(self):
        artifact = make_artifact([make_obs(confidence="low", width=90.0)])
        heuristic_filters.run_stage_a(artifact, HeuristicConfig())
        obs = artifact.observations[0]
        # Every enabled filter left a decision, in registry order.
        names = [d.filter_name for d in obs.decisions]
        self.assertEqual(names, [
            "confidence_gate", "angular_width_gate", "tag_blocklist",
            "elevation_gate", "edge_truncation", "intra_frame_dedup"])
        # First filtering reason wins.
        self.assertEqual(obs.final_reason, "confidence_low")
        # Both filtering filters counted.
        self.assertEqual(artifact.stats.filtered_by_filter, {
            "confidence_gate": 1, "angular_width_gate": 1})

    def test_tag_blocklist_key_and_exact(self):
        config = HeuristicConfig()
        artifact = make_artifact([
            make_obs(primary=("highway", "primary")),
            make_obs(obs_id="f0000__lm1__box0", primary=("natural", "tree")),
        ])
        # Add an exact-tag block via msgspec struct replacement (frozen).
        import msgspec
        config = msgspec.structs.replace(
            config, tag_blocklist=msgspec.structs.replace(
                config.tag_blocklist,
                blocked_primary_tags=[["natural", "tree"]]))
        heuristic_filters.run_stage_a(artifact, config)
        self.assertEqual(artifact.observations[0].final_reason,
                         "blocked_tag:highway")
        self.assertEqual(artifact.observations[1].final_reason,
                         "blocked_tag:natural=tree")

    def test_elevation_gate(self):
        artifact = make_artifact([make_obs(elevation=-40.0)])
        heuristic_filters.run_stage_a(artifact, HeuristicConfig())
        self.assertEqual(artifact.observations[0].final_reason,
                         "elevation_too_low")

    def test_edge_truncation_only_unmerged(self):
        edge_box = [schema.BBox(face_yaw_deg=0, xmin=980, ymin=400, xmax=1000,
                                ymax=600)]
        artifact = make_artifact([
            make_obs(boxes=edge_box, seam_merged=False),
            make_obs(obs_id="f0000__lm1__box0", boxes=edge_box,
                     seam_merged=True, bearing=44.0),
        ])
        heuristic_filters.run_stage_a(artifact, HeuristicConfig())
        self.assertEqual(artifact.observations[0].final_reason,
                         "unmatched_edge_truncation")
        self.assertEqual(artifact.observations[1].final_disposition,
                         schema.KEPT)

    def test_intra_frame_dedup_keeps_best(self):
        artifact = make_artifact([
            make_obs(obs_id="f0000__lm0__box0", confidence="medium",
                     bearing=10.0),
            make_obs(obs_id="f0000__lm1__box0", confidence="high",
                     bearing=12.0),
            # Same tag but far away in bearing: kept.
            make_obs(obs_id="f0000__lm2__box0", confidence="low",
                     bearing=200.0),
        ])
        config = HeuristicConfig(
            confidence_gate=msgspec_replace_enabled(False))
        heuristic_filters.run_stage_a(artifact, config)
        by_id = {o.obs_id: o for o in artifact.observations}
        self.assertEqual(by_id["f0000__lm0__box0"].final_reason,
                         "intra_frame_duplicate")
        self.assertEqual(by_id["f0000__lm1__box0"].final_disposition,
                         schema.KEPT)
        self.assertEqual(by_id["f0000__lm2__box0"].final_disposition,
                         schema.KEPT)
        # Details point at the winner.
        dup_decision = [d for d in by_id["f0000__lm0__box0"].decisions
                        if d.filter_name == "intra_frame_dedup"][0]
        self.assertEqual(
            dup_decision.details["duplicate_of_landmark_idx"], 1.0)

    def test_disabled_filters_leave_no_decisions(self):
        artifact = make_artifact([make_obs(confidence="low")])
        config = HeuristicConfig(
            confidence_gate=msgspec_replace_enabled(False),
            intra_frame_dedup=IntraFrameDedupConfig(enabled=False))
        heuristic_filters.run_stage_a(artifact, config)
        obs = artifact.observations[0]
        names = {d.filter_name for d in obs.decisions}
        self.assertNotIn("confidence_gate", names)
        self.assertNotIn("intra_frame_dedup", names)
        self.assertEqual(obs.final_disposition, schema.KEPT)


def msgspec_replace_enabled(enabled):
    from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
        ConfidenceGateConfig,
    )
    return ConfidenceGateConfig(enabled=enabled)


if __name__ == "__main__":
    unittest.main()
