import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common.torch.load_torch_deps  # noqa: F401
import msgspec
import numpy as np
from pyproj import Proj, Transformer
import torch

from experimental.overhead_matching.swag.farfield import artifact, geometry
from experimental.overhead_matching.swag.farfield.localization import (
    distance_episodes, export_ingest, grid_filter, retrieval_grid, structs,
)


class Meta(msgspec.Struct):
    n_keyframes: int
    prior_region: object
    nominal_forward: dict


class RetrievalGridTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.frame = geometry.RegionFrame(42, -71)
        x, y = Transformer.from_crs(4326, 32619, always_xy=True).transform(-71, 42)
        self.lon, self.lat = Transformer.from_crs(32619, 4326, always_xy=True).transform(
            x + np.array([0, 100, 0, 100]), y + np.array([0, 0, 100, 100]))
        south, west = self.frame.latlon_from_enu(-50, -50)
        north, east = self.frame.latlon_from_enu(260, 160)
        prior = export_ingest.PriorRegion("catalog", [west, south, east, north],
                                         -50, 260, -50, 160)
        ref = artifact.ArtifactRef(kind="localization_inputs", dataset="test",
                                   version="v1", path="inputs",
                                   content_digest="a" * 64, manifest_digest="b" * 64)
        self.data = export_ingest.ExportData(
            ref, SimpleNamespace(), Meta(5, prior, {"bearing_camera_cw_deg": 45.0}),
            self.frame, SimpleNamespace(position_sigma_m=[10]), [], [], [], {},
            [structs.TruthPose(i, i * 10, 0, 0) for i in range(5)])
        self.csv = self.root / "frames.csv"
        self.ids = [f"p{i},42,-71," for i in range(5)]
        with self.csv.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["idx", "latitude", "longitude", "frame_file"])
            for p in self.data.truth:
                lat, lon = self.frame.latlon_from_enu(p.east_m, p.north_m)
                writer.writerow([p.keyframe_idx, lat, lon, self.ids[p.keyframe_idx] + ".jpg"])
        self.meta = dict(schema_version="0.7", dataset="test", n_keyframes=5,
                         n_nodes=4, n_heading_bins=4, node_spacing_m=100.,
                         scorer="synthetic", db_manifest_sha256="c" * 64)
        (self.raw / "retrieval_meta.json").write_text(json.dumps(self.meta))
        self.scores = np.arange(80, dtype=np.float16).reshape(5, 4, 4) / 10
        self.write_scores()
        self.grid = grid_filter.Grid(-50, 260, -50, 160, 100)

    def write_scores(self, ids=None):
        np.savez_compressed(self.raw / "retrieval_fields.npz", lat_deg=self.lat,
                            lon_deg=self.lon, scores=self.scores,
                            keyframe_idx=np.arange(5),
                            pano_ids=np.array(ids or self.ids))

    def observation(self, **kwargs):
        options = dict(render_crs="EPSG:32619", temperature=0.3,
                       outlier_epsilon=0.1, device="cpu")
        options.update(kwargs)
        return retrieval_grid.RetrievalGridObservation(
            self.raw, self.csv, self.root / "cache", self.data, self.grid, 8, **options)

    def test_mapping_heading_wrap_floor_and_local_cache(self):
        observation = self.observation()
        self.assertIsInstance(observation.scores, np.memmap)
        before = sorted(p.name for p in self.raw.iterdir())
        actual = observation.log_likelihood(self.ids[2]).numpy()
        self.assertAlmostEqual(np.exp(actual).sum(), 1)
        self.assertTrue(np.isneginf(actual[:, ~observation.region_mask.numpy()]).all())
        unsupported = observation.region_mask & ~observation.support_mask
        self.assertTrue(bool(unsupported.any()))
        np.testing.assert_allclose(np.exp(actual[:, unsupported.numpy()]),
                                   0.1 / (8 * observation.region_mask.sum().item()))
        # At the first cell, true nominal zero means camera -45 degrees, then
        # subtract the (negative here) meridian convergence: interpolate 3->0.
        convergence = Proj(32619).get_factors(self.lon[0], self.lat[0]).meridian_convergence
        position = ((-45 - convergence) % 360) / 90
        self.assertEqual(observation._lo[0, 0], 3)
        self.assertAlmostEqual(observation._fraction[0, 0], position - 3)
        def interpolated(angle):
            value = ((angle - 45 - convergence) % 360) / 90
            lo = int(value)
            return ((1 - (value - lo)) * float(self.scores[2, 0, lo])
                    + (value - lo) * float(self.scores[2, 0, (lo + 1) % 4]))
        floor = 0.1 / (8 * observation.region_mask.sum().item())
        ratio = (np.exp(actual[0, 0, 0]) - floor) / (np.exp(actual[1, 0, 0]) - floor)
        self.assertAlmostEqual(ratio, np.exp((interpolated(0) - interpolated(45)) / 0.3))
        self.assertEqual(sorted(p.name for p in self.raw.iterdir()), before)
        cache = next((self.root / "cache").glob("*.npy"))
        timestamp = cache.stat().st_mtime_ns
        np.testing.assert_array_equal(self.observation().log_likelihood(self.ids[2]).numpy(), actual)
        self.assertEqual(cache.stat().st_mtime_ns, timestamp)

    def test_invalid_inputs_and_nonfinite_scores(self):
        for options in (dict(temperature=0), dict(temperature=float("nan")),
                        dict(outlier_epsilon=1), dict(render_crs="EPSG:4326"),
                        dict(render_crs="EPSG:32618")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.observation(**options)
        self.write_scores(ids=[self.ids[1], self.ids[0], *self.ids[2:]])
        with self.assertRaisesRegex(ValueError, "CSV order"):
            self.observation()
        self.scores[2, 0, 0] = np.nan
        self.write_scores()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.observation().log_likelihood(self.ids[2])
        (self.raw / "retrieval_fields.npz").write_bytes(b"incomplete archive")
        with self.assertRaises((ValueError, OSError)):
            self.observation()

    def test_real_causal_loop_uses_parent_panorama_ids_for_episode_and_prefix(self):
        plan = {"schema": distance_episodes.SCHEMA,
                "localization_inputs": self.data.artifact_ref.to_dict(),
                "windows": [[0, 2], [1, 3], [2, 4]],
                "boundary_policy": "trajectory_only"}
        plan_path = self.root / "episodes.json"
        plan_path.write_text(json.dumps(plan))
        output = self.root / "result.json"
        seen = []
        original = retrieval_grid.RetrievalGridObservation.log_likelihood
        def observe(instance, pano_id):
            seen.append(pano_id)
            return original(instance, pano_id)
        odometry = [structs.OdometryDelta(i, 10, 0, 0, 0, 0) for i in (1, 2)]
        with patch.object(grid_filter.odometry_profiles, "derive",
                          return_value=(odometry, {})) as derive, patch.object(
                retrieval_grid.RetrievalGridObservation, "log_likelihood", observe):
            grid_filter.main([
                "--input_dir", "inputs", "--observation_source", "crosslocate",
                "--availability", "immediate", "--margin_m", "0",
                "--retrieval_dir", str(self.raw), "--retrieval_frames_csv", str(self.csv),
                "--retrieval_cache_dir", str(self.root / "cache"),
                "--retrieval_render_crs", "EPSG:32619", "--device", "cpu",
                "--cell_m", "100", "--n_heading", "8", "--top_modes", "0",
                "--episode_plan", str(plan_path), "--episode_index", "1",
                "--output_end", "1", "--out", str(output)],
                load_input=lambda _: self.data)
        self.assertEqual(seen, self.ids[1:3])
        self.assertEqual(derive.call_args.kwargs["keyframe_range"], (1, 3))
        result = json.loads(output.read_text())
        self.assertEqual(result["schema"], "farfield_causal_grid/v1")
        self.assertEqual(result["episode"]["parent_keyframe_start"], 1)
        self.assertEqual(len(result["map_error_m_by_keyframe"]), 2)
        self.assertFalse(result["crosslocate"]["calibration_frozen"])
        self.assertFalse(result["availability"]["track_inputs_used"])


if __name__ == "__main__":
    unittest.main()
