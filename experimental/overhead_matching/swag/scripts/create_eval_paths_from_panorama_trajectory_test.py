import csv
import tempfile
import unittest
from pathlib import Path

from common.gps.web_mercator import EARTH_RADIUS_M
from common.math.haversine import find_d_on_unit_circle
from experimental.overhead_matching.swag.scripts.create_eval_paths_from_panorama_trajectory import (
    load_trajectory,
)
from experimental.overhead_matching.swag.scripts.dataset_statistics import (
    compute_trajectory_km,
)


def _write_mapping(dir_path: Path, rows: list[dict]) -> None:
    mapping = dir_path / "pano_id_mapping.csv"
    with open(mapping, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pano_id", "lat", "lon"])
        writer.writeheader()
        writer.writerows(rows)


class CreateEvalPathsTrajectoryTest(unittest.TestCase):

    def test_load_trajectory_matches_haversine(self):
        rows = [
            {"pano_id": "a", "lat": "42.3601", "lon": "-71.0589"},  # Boston
            {"pano_id": "b", "lat": "42.3611", "lon": "-71.0592"},
            {"pano_id": "c", "lat": "42.3615", "lon": "-71.0600"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            _write_mapping(dataset, rows)

            pano_ids, cum_dist = load_trajectory(dataset)

            self.assertEqual(pano_ids, ["a", "b", "c"])
            self.assertEqual(cum_dist[0], 0.0)

            # Cumulative distance must be monotone non-decreasing.
            for prev, curr in zip(cum_dist, cum_dist[1:]):
                self.assertGreaterEqual(curr, prev)

            # Final distance matches a direct haversine computation (pins
            # the (lat, lon) argument order and the EARTH_RADIUS_M scaling).
            expected = 0.0
            for i in range(1, len(rows)):
                p1 = (float(rows[i - 1]["lat"]), float(rows[i - 1]["lon"]))
                p2 = (float(rows[i]["lat"]), float(rows[i]["lon"]))
                expected += EARTH_RADIUS_M * find_d_on_unit_circle(p1, p2)
            self.assertAlmostEqual(cum_dist[-1], expected, places=6)

    def test_load_trajectory_short_csv_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            # Zero data rows (header only): must not touch any columns.
            _write_mapping(dataset, [])
            self.assertEqual(load_trajectory(dataset), ([], []))

            # Single data row: still can't form a trajectory segment.
            _write_mapping(dataset, [{"pano_id": "a", "lat": "1.0", "lon": "2.0"}])
            self.assertEqual(load_trajectory(dataset), ([], []))

    def test_load_trajectory_short_csv_tolerates_missing_columns(self):
        # < 2 rows must bail before dereferencing columns, so a CSV that
        # has a valid header structure but is missing the pano_id column
        # should not raise.
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            mapping = dataset / "pano_id_mapping.csv"
            with open(mapping, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["lat", "lon"])
                writer.writeheader()
                writer.writerow({"lat": "1.0", "lon": "2.0"})
            self.assertEqual(load_trajectory(dataset), ([], []))

    def test_compute_trajectory_km_matches_cum_dist(self):
        rows = [
            {"pano_id": "a", "lat": "42.3601", "lon": "-71.0589"},
            {"pano_id": "b", "lat": "42.3650", "lon": "-71.0700"},
            {"pano_id": "c", "lat": "42.3700", "lon": "-71.0800"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            _write_mapping(dataset, rows)
            _, cum_dist = load_trajectory(dataset)
            self.assertAlmostEqual(
                compute_trajectory_km(dataset), cum_dist[-1] / 1000.0, places=6
            )

    def test_compute_trajectory_km_returns_none_on_missing_or_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            # Missing file.
            self.assertIsNone(compute_trajectory_km(dataset))
            # < 2 rows.
            _write_mapping(dataset, [{"pano_id": "a", "lat": "1", "lon": "2"}])
            self.assertIsNone(compute_trajectory_km(dataset))


if __name__ == "__main__":
    unittest.main()
