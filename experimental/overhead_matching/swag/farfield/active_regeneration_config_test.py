import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield import pipeline


CONFIG_DIR = Path(__file__).parent / "configs" / "active_regeneration"
EXPERIMENT = "260824_stage3_active_regeneration"
ADOPTED_VERSION = "stage3_7b88e81_adopted_v1"
REGEN_VERSION = "stage3_7b88e81_regen_v1"
CATALOG_VERSION = "stage3_7b88e81_trim_v1"
REVERSE_SOURCE = (
    "human_review_no_reverse_segments_user_certified_2026-08-24")
DATASETS = {
    "boston_harbor_leg1": (378, "gemini-3.1-pro-preview"),
    "boston_harbor_leg2": (235, "gemini-3.1-pro-preview"),
    "boston_harbor_leg3": (733, "gemini-3.1-pro-preview"),
    "charles_river_20260727": (512, "gemini-3.1-pro-preview"),
    "mount_washington_20260815_leg1": (
        133, "gemini-3.1-pro-preview"),
    "mount_washington_20260815_leg2": (
        264, "gemini-3.1-pro-preview"),
    "mount_washington_20260815_leg3": (
        397, "gemini-3.1-pro-preview"),
    "pohang_canal_04": (1449, "gemini-3.7-flash"),
}


class ActiveRegenerationConfigTest(unittest.TestCase):
    def test_frozen_recipes_are_complete_and_valid(self):
        self.assertEqual(
            {path.stem for path in CONFIG_DIR.glob("*.yaml")},
            set(DATASETS))
        expected_artifacts = {
            "frame_landmarks_version": ADOPTED_VERSION,
            "pinhole_images_version": ADOPTED_VERSION,
            "object_tracks_version": REGEN_VERSION,
            "semantic_audits_version": REGEN_VERSION,
            "bearing_observations_version": REGEN_VERSION,
            "landmark_matches_version": REGEN_VERSION,
            "alignment_diagnostics_version": REGEN_VERSION,
            "localization_inputs_version": REGEN_VERSION,
            "catalogs_version": CATALOG_VERSION,
        }
        for dataset, (k_end, model) in DATASETS.items():
            with self.subTest(dataset=dataset):
                config = pipeline.load_pipeline_config(
                    CONFIG_DIR / f"{dataset}.yaml")
                pipeline.validate_pipeline_config(config)
                self.assertEqual(config["experiment"]["name"], EXPERIMENT)
                self.assertEqual(config["artifacts"], expected_artifacts)
                self.assertEqual(config["extraction"]["model"], model)
                self.assertEqual(
                    config["execution"], {
                        "llm_transport": "batch",
                        "batch_gcs_prefix": (
                            "gs://crossview/farfield/stage3_7b88e81/"
                            f"{dataset}"),
                        "approve_cost": True,
                    })
                self.assertEqual(config["cost"]["limit_usd"], 50.0)
                self.assertEqual(
                    config["tracking"]["range"],
                    {"k_start": 0, "k_end": k_end})
                self.assertEqual(
                    config["tracking"]["sam2_checkpoint"],
                    "sam2/sam2.1_hiera_large.pt")
                inputs = config["localization_inputs"]
                self.assertEqual(inputs["motion_source"], "frames_gps.csv")
                self.assertEqual(
                    inputs["nominal_forward_calibration"],
                    "nominal_forward.json")
                self.assertEqual(
                    inputs["identity_review_dir"],
                    f"identity_reviews/{dataset}/{REGEN_VERSION}")
                self.assertEqual(inputs["reverse_keyframe_ranges"], [])
                self.assertEqual(
                    inputs["reverse_annotation_source"], REVERSE_SOURCE)
                self.assertEqual(
                    inputs["landmark_position_sigma_m"], 25.0)
                self.assertEqual(
                    config["localization"]["run_name"],
                    f"{dataset}_{REGEN_VERSION}")


if __name__ == "__main__":
    unittest.main()
