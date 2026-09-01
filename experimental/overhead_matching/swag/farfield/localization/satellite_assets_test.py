import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import (
    satellite_assets,
    satellite_underlay,
)


ANCHOR_LAT = 36.037149268965514
ANCHOR_LON = 129.37857389517242


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def _make_dataset(root: Path, dataset: str, date: str | None = "2021-07-16"):
    directory = root / "datasets" / dataset
    directory.mkdir(parents=True)
    metadata = {"dataset_name": dataset}
    if date is not None:
        metadata["capture_date"] = date
    _write_json(directory / "pipeline_metadata.json", metadata)


def _make_run(root: Path, experiment: str, name: str, *, dataset: str = "ds",
              prior_bounds=(-1000.0, 1000.0, -500.0, 500.0),
              catalog_bounds=(-10000.0, 12000.0, -8000.0, 9000.0),
              trajectory_bounds=(-200.0, 200.0, -100.0, 100.0)) -> Path:
    run = root / "runs" / experiment / name
    run.mkdir(parents=True)
    _write_json(run / artifact.MANIFEST_NAME, {
        "kind": "localization_run", "dataset": dataset,
    })
    e0, e1, n0, n1 = catalog_bounds
    pe0, pe1, pn0, pn1 = prior_bounds
    _write_json(run / "run_manifest.json", {
        "dataset": dataset,
        "anchor_lat_deg": ANCHOR_LAT,
        "anchor_lon_deg": ANCHOR_LON,
        "filter_config": {"init": {
            "kind": "UniformBoxInit",
            "east_min_m": pe0,
            "east_max_m": pe1,
            "north_min_m": pn0,
            "north_max_m": pn1,
        }},
        "landmarks": [{
            "lat_deg": ANCHOR_LAT,
            "lon_deg": ANCHOR_LON,
            "hull_east_m": [e0, e1],
            "hull_north_m": [n0, n1],
        }],
    })
    te0, te1, tn0, tn1 = trajectory_bounds
    (run / "truth.jsonl").write_text("\n".join(json.dumps(record) for record in (
        {"east_m": te0, "north_m": tn0},
        {"east_m": te1, "north_m": tn1},
    )) + "\n")
    return run


def _make_underlay(directory: Path, source_run: Path, *, dataset: str | None = None,
                   date: str = "2021-07-16",
                   wide=(-1000.0, 1000.0, -500.0, 500.0),
                   fine=(-400.0, 400.0, -300.0, 300.0),
                   extent_kind: str | None = satellite_assets.WIDE_EXTENT_KIND
                   ) -> Path:
    directory.mkdir(parents=True)
    (directory / "wide.jpg").write_bytes(b"wide")
    (directory / "fine.jpg").write_bytes(b"fine")
    spec = {
        "anchor_lat_deg": ANCHOR_LAT,
        "anchor_lon_deg": ANCHOR_LON,
        "layers": [
            {"image": "wide.jpg", "zoom": 12,
             "east_min": wide[0], "east_max": wide[1],
             "north_min": wide[2], "north_max": wide[3],
             "n_tiles": 10, "n_failed": 0},
            {"image": "fine.jpg", "zoom": 17,
             "east_min": fine[0], "east_max": fine[1],
             "north_min": fine[2], "north_max": fine[3],
             "n_tiles": 10, "n_failed": 0},
        ],
    }
    if extent_kind is not None:
        spec["wide_extent_kind"] = extent_kind
    if dataset is not None:
        spec["dataset"] = dataset
        spec["capture_date"] = date
    _write_json(directory / satellite_assets.SATELLITE_MANIFEST, spec)
    _write_json(directory / artifact.MANIFEST_NAME, {
        "inputs": {"run_dir": str(source_run)},
        "config": {"date": date},
    })
    return directory


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _make_dataset(self.root, "ds")
        self.target = _make_run(self.root, "new", "target")

    def tearDown(self):
        self.temp.cleanup()

    def test_finds_legacy_underlay_from_another_run_of_the_dataset(self):
        source = _make_run(self.root, "old", "source")
        underlay = _make_underlay(
            source.with_name(source.name + ".satellite"), source)

        self.assertEqual(satellite_assets.discover(self.target), underlay)

    def test_shared_cache_wins_over_legacy_run_assets(self):
        source = _make_run(self.root, "old", "source")
        _make_underlay(source.with_name(source.name + ".satellite"), source)
        shared = (self.root / satellite_assets.SHARED_RELATIVE_ROOT
                  / "ds" / "wayback-shared")
        _make_underlay(shared, source, dataset="ds")

        self.assertEqual(satellite_assets.discover(self.target), shared)

    def test_exact_run_sibling_wins_over_shared_and_legacy(self):
        source = _make_run(self.root, "old", "source")
        _make_underlay(source.with_name(source.name + ".satellite"), source)
        shared = (self.root / satellite_assets.SHARED_RELATIVE_ROOT
                  / "ds" / "wayback-shared")
        _make_underlay(shared, source, dataset="ds")
        sibling = _make_underlay(
            self.target.with_name(self.target.name + ".satellite"), self.target)

        self.assertEqual(satellite_assets.discover(self.target), sibling)

    def test_wrong_dataset_is_not_reused_even_at_the_same_anchor(self):
        _make_dataset(self.root, "other")
        source = _make_run(self.root, "old", "source", dataset="other")
        _make_underlay(source.with_name(source.name + ".satellite"), source)

        self.assertIsNone(satellite_assets.discover(self.target))

    def test_capture_month_must_match(self):
        source = _make_run(self.root, "old", "source")
        _make_underlay(source.with_name(source.name + ".satellite"), source,
                       date="2020-03-01")

        self.assertIsNone(satellite_assets.discover(self.target))

    def test_prior_must_match_and_fine_must_cover_the_trajectory(self):
        source = _make_run(self.root, "old", "source")
        _make_underlay(source.with_name(source.name + ".satellite"), source,
                       wide=(-100.0, 100.0, -100.0, 100.0))

        self.assertIsNone(satellite_assets.discover(self.target))

    def test_imagery_outside_the_prior_and_legacy_extent_are_rejected(self):
        source = _make_run(self.root, "old", "source")
        oversized = _make_underlay(
            source.with_name(source.name + ".satellite"), source,
            wide=(-1200.0, 1200.0, -700.0, 700.0))
        self.assertIsNone(satellite_assets.discover(self.target))

        _write_json(oversized / satellite_assets.SATELLITE_MANIFEST, {
            "anchor_lat_deg": ANCHOR_LAT,
            "anchor_lon_deg": ANCHOR_LON,
            "layers": [{
                "image": "wide.jpg", "zoom": 12,
                "east_min": -1000.0, "east_max": 1000.0,
                "north_min": -500.0, "north_max": 500.0,
                "n_tiles": 10, "n_failed": 0,
            }],
        })
        self.assertIsNone(satellite_assets.discover(self.target))

    def test_an_all_missing_layer_is_rejected(self):
        source = _make_run(self.root, "old", "source")
        underlay = _make_underlay(
            source.with_name(source.name + ".satellite"), source)
        spec_path = underlay / satellite_assets.SATELLITE_MANIFEST
        spec = json.loads(spec_path.read_text())
        spec["layers"][1]["n_failed"] = spec["layers"][1]["n_tiles"]
        _write_json(spec_path, spec)

        self.assertIsNone(satellite_assets.discover(self.target))

    def test_existing_underlay_avoids_reading_or_fetching_the_run(self):
        underlay = _make_underlay(
            self.target.with_name(self.target.name + ".satellite"), self.target)
        with mock.patch.object(satellite_assets.run_io, "read_run") as read_run:
            self.assertEqual(
                satellite_assets.find_or_generate(self.target), underlay)
        read_run.assert_not_called()


class PlacementTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _make_dataset(self.root, "ds")
        self.run = _make_run(self.root, "experiment", "run")
        self.summary = satellite_assets.summarize_run(self.run)
        self.assertIsNotNone(self.summary)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def plan(n_tiles: int):
        return [{"name": "wide", "zoom": 12, "tiles": (1, 2, 3, 4),
                 "n_tiles": n_tiles}]

    def test_compact_plan_is_run_local(self):
        destination = satellite_assets.automatic_destination(
            self.summary, self.plan(satellite_assets.RUN_LOCAL_TILE_LIMIT))
        self.assertEqual(destination,
                         self.run.with_name(self.run.name + ".satellite"))

    def test_large_plan_is_content_keyed_and_shared(self):
        plans = self.plan(satellite_assets.RUN_LOCAL_TILE_LIMIT + 1)
        first = satellite_assets.automatic_destination(self.summary, plans)
        second = satellite_assets.automatic_destination(self.summary, plans)
        self.assertEqual(first, second)
        self.assertEqual(first.parent,
                         self.root / satellite_assets.SHARED_RELATIVE_ROOT / "ds")
        self.assertTrue(first.name.startswith("wayback-20210716-"))

    def test_missing_capture_date_does_not_guess_or_fetch(self):
        (self.root / "datasets" / "ds" / "pipeline_metadata.json").write_text(
            '{"dataset_name":"ds"}')
        with mock.patch.object(satellite_assets.run_io, "read_run") as read_run:
            self.assertIsNone(satellite_assets.find_or_generate(self.run))
        read_run.assert_not_called()

    def test_generation_uses_shared_destination_for_a_large_plan(self):
        plans = self.plan(satellite_assets.RUN_LOCAL_TILE_LIMIT + 1)
        fake_data = SimpleNamespace(manifest=SimpleNamespace(
            dataset="ds", anchor_lat_deg=ANCHOR_LAT,
            anchor_lon_deg=ANCHOR_LON))
        expected = satellite_assets.automatic_destination(self.summary, plans)
        with mock.patch.object(satellite_assets.run_io, "read_run",
                               return_value=fake_data), \
                mock.patch.object(satellite_underlay, "plan_underlay",
                                  return_value=plans), \
                mock.patch.object(satellite_underlay, "describe_plan"), \
                mock.patch.object(satellite_underlay, "generate_underlay",
                                  return_value=expected) as generate:
            actual = satellite_assets.find_or_generate(self.run)

        self.assertEqual(actual, expected)
        self.assertTrue(expected.parent.is_dir())
        self.assertEqual(generate.call_args.kwargs["output_dir"], expected)
        self.assertEqual(generate.call_args.kwargs["date"], "2021-07-16")

    def test_generation_uses_run_local_destination_for_a_compact_plan(self):
        plans = self.plan(satellite_assets.RUN_LOCAL_TILE_LIMIT)
        fake_data = SimpleNamespace(manifest=SimpleNamespace(
            dataset="ds", anchor_lat_deg=ANCHOR_LAT,
            anchor_lon_deg=ANCHOR_LON))
        expected = self.run.with_name(self.run.name + ".satellite")
        with mock.patch.object(satellite_assets.run_io, "read_run",
                               return_value=fake_data), \
                mock.patch.object(satellite_underlay, "plan_underlay",
                                  return_value=plans), \
                mock.patch.object(satellite_underlay, "describe_plan"), \
                mock.patch.object(satellite_underlay, "generate_underlay",
                                  return_value=expected) as generate:
            actual = satellite_assets.find_or_generate(self.run)

        self.assertEqual(actual, expected)
        self.assertEqual(generate.call_args.kwargs["output_dir"], expected)


if __name__ == "__main__":
    unittest.main()
