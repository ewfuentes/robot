import csv
import tempfile
import unittest
from pathlib import Path

from common.gps.web_mercator import EARTH_RADIUS_M
from common.math.haversine import find_d_on_unit_circle
from experimental.overhead_matching.swag.scripts.create_eval_paths_from_panorama_trajectory import (
    generate_paths,
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


def _write_uniform_linear_trajectory(
    dir_path: Path, n_points: int, step_m: float, base_lat: float = 0.0
) -> None:
    """Write a mapping CSV of n_points points spaced step_m apart along
    a constant-latitude line (so each segment is approximately step_m meters
    at the equator — uniform spacing is what the generate_paths tests rely on)."""
    # At the equator a small longitude step Δlon (radians) traces an arc of
    # length EARTH_RADIUS_M * Δlon on the unit sphere, so Δlon = step_m / R.
    import math

    lon_step_deg = math.degrees(step_m / EARTH_RADIUS_M)
    rows = [
        {
            "pano_id": f"p{i:06d}",
            "lat": f"{base_lat}",
            "lon": f"{i * lon_step_deg}",
        }
        for i in range(n_points)
    ]
    _write_mapping(dir_path, rows)


class LoadTrajectoryTest(unittest.TestCase):

    def test_matches_haversine(self):
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
            for prev, curr in zip(cum_dist, cum_dist[1:]):
                self.assertGreaterEqual(curr, prev)

            # Pin the (lat, lon) argument order and EARTH_RADIUS_M scaling
            # by recomputing with the shared haversine utility.
            expected = 0.0
            for i in range(1, len(rows)):
                p1 = (float(rows[i - 1]["lat"]), float(rows[i - 1]["lon"]))
                p2 = (float(rows[i]["lat"]), float(rows[i]["lon"]))
                expected += EARTH_RADIUS_M * find_d_on_unit_circle(p1, p2)
            self.assertAlmostEqual(cum_dist[-1], expected, places=6)

    def test_raises_on_short_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)

            _write_mapping(dataset, [])  # header only
            with self.assertRaises(ValueError):
                load_trajectory(dataset)

            _write_mapping(dataset, [{"pano_id": "a", "lat": "1.0", "lon": "2.0"}])
            with self.assertRaises(ValueError):
                load_trajectory(dataset)

    def test_raises_on_short_csv_before_touching_columns(self):
        # If the CSV is short, we must raise before dereferencing any column,
        # so missing columns produce a ValueError from our check rather than
        # a less-informative KeyError from the row parsing.
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            mapping = dataset / "pano_id_mapping.csv"
            with open(mapping, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["lat", "lon"])
                writer.writeheader()
                writer.writerow({"lat": "1.0", "lon": "2.0"})
            with self.assertRaises(ValueError):
                load_trajectory(dataset)


class ComputeTrajectoryKmTest(unittest.TestCase):

    def test_matches_cum_dist(self):
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

    def test_returns_none_when_csv_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(compute_trajectory_km(Path(tmp)))

    def test_propagates_short_csv_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            _write_mapping(dataset, [{"pano_id": "a", "lat": "1", "lon": "2"}])
            with self.assertRaises(ValueError):
                compute_trajectory_km(dataset)


class GeneratePathsTest(unittest.TestCase):

    # Uniform synthetic trajectory: 10001 points spaced ~1m apart ⇒ ~10 km.
    N_POINTS = 10001
    STEP_M = 1.0
    TARGET_M = 1000.0

    def _load_uniform(self, tmp: Path) -> tuple[list[str], list[float]]:
        _write_uniform_linear_trajectory(tmp, self.N_POINTS, self.STEP_M)
        pano_ids, cum_dist = load_trajectory(tmp)
        # Sanity: ~10 km total.
        self.assertAlmostEqual(cum_dist[-1], (self.N_POINTS - 1) * self.STEP_M, delta=0.5)
        return pano_ids, cum_dist

    def test_half_forward_half_backward_with_correct_ordering(self):
        num_paths = 10
        with tempfile.TemporaryDirectory() as tmp:
            pano_ids, cum_dist = self._load_uniform(Path(tmp))
            paths = generate_paths(pano_ids, cum_dist, self.TARGET_M, num_paths)

        self.assertEqual(len(paths), num_paths)
        num_fwd = num_paths // 2
        num_bwd = num_paths - num_fwd
        fwd, bwd = paths[:num_fwd], paths[num_fwd:]
        self.assertEqual(len(fwd), num_bwd)  # equal halves (num_paths is even)

        pano_idx = {pid: i for i, pid in enumerate(pano_ids)}
        for path in fwd:
            idx = [pano_idx[p] for p in path]
            self.assertEqual(idx, sorted(idx), "forward path not in ascending order")
        for path in bwd:
            idx = [pano_idx[p] for p in path]
            self.assertEqual(idx, sorted(idx, reverse=True),
                             "backward path not in descending order")

    def test_all_paths_cover_target_distance(self):
        num_paths = 10
        with tempfile.TemporaryDirectory() as tmp:
            pano_ids, cum_dist = self._load_uniform(Path(tmp))
            paths = generate_paths(pano_ids, cum_dist, self.TARGET_M, num_paths)

        pano_idx = {pid: i for i, pid in enumerate(pano_ids)}
        num_fwd = num_paths // 2

        for path in paths[:num_fwd]:
            span = cum_dist[pano_idx[path[-1]]] - cum_dist[pano_idx[path[0]]]
            self.assertGreaterEqual(span, self.TARGET_M - 1e-6)
        for path in paths[num_fwd:]:
            # Backward paths are stored reversed, so path[0] is the later pano.
            span = cum_dist[pano_idx[path[0]]] - cum_dist[pano_idx[path[-1]]]
            self.assertGreaterEqual(span, self.TARGET_M - 1e-6)

        # With uniform ~1m spacing, every path should be about target_m + 1
        # panos long (one pano per meter, inclusive endpoints). Accumulated
        # floating-point drift in cum_dist can push the end index one step
        # further, so allow a single extra pano.
        base_len = int(self.TARGET_M / self.STEP_M) + 1
        for path in paths:
            self.assertIn(len(path), (base_len, base_len + 1))

    def test_path_starts_uniformly_spaced(self):
        num_paths = 10
        with tempfile.TemporaryDirectory() as tmp:
            pano_ids, cum_dist = self._load_uniform(Path(tmp))
            paths = generate_paths(pano_ids, cum_dist, self.TARGET_M, num_paths)

        pano_idx = {pid: i for i, pid in enumerate(pano_ids)}
        total_dist = cum_dist[-1]
        num_fwd = num_paths // 2
        num_bwd = num_paths - num_fwd

        # Forward starts: uniformly spaced over [0, total - target].
        fwd_starts = [cum_dist[pano_idx[p[0]]] for p in paths[:num_fwd]]
        max_fwd_start = total_dist - self.TARGET_M
        expected_fwd = [max_fwd_start * i / (num_fwd - 1) for i in range(num_fwd)]
        for actual, want in zip(fwd_starts, expected_fwd):
            self.assertAlmostEqual(actual, want, delta=self.STEP_M)

        # Backward starts: uniformly spaced over [target, total].
        # path[0] for backward paths is the later pano (lists are reversed).
        bwd_starts = [cum_dist[pano_idx[p[0]]] for p in paths[num_fwd:]]
        expected_bwd = [
            self.TARGET_M + (total_dist - self.TARGET_M) * i / (num_bwd - 1)
            for i in range(num_bwd)
        ]
        for actual, want in zip(bwd_starts, expected_bwd):
            self.assertAlmostEqual(actual, want, delta=self.STEP_M)


if __name__ == "__main__":
    unittest.main()
