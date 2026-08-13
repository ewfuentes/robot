import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experimental.map_estimation.data import argoverse_catalog as ac
from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import s5cmd

SENSOR_PREFIX = "s3://argoverse/datasets/av2/sensor/val/"


def stat(num_bytes: int, num_objects: int) -> ac.ItemStat:
    return ac.ItemStat(num_bytes=num_bytes, num_objects=num_objects)


def make_entry(log_id: str, city: str | None = "PIT", *, sweeps: int = 157,
               camera_bytes: int = 91_000_000, annotations: bool = True) -> ac.LogEntry:
    items = {
        "map": stat(1_700_000, 3),
        "calibration": stat(9_916, 2),
        "poses": stat(164_650, 1),
        "lidar": stat(144_000_000, sweeps),
        "ring_front_center": stat(camera_bytes, 319),
    }
    if annotations:
        items["annotations"] = stat(285_474, 1)
    return ac.LogEntry(log_id=log_id, city=city, items=items)


def make_catalog(entries, *, dataset=al.Dataset.SENSOR, split="val",
                 strategy=ac.BuildStrategy.PER_LOG_DETAIL) -> ac.Catalog:
    return ac.Catalog(
        schema_version=ac.SCHEMA_VERSION,
        dataset=dataset,
        split=split,
        built_at=datetime.datetime(2026, 8, 13, tzinfo=datetime.timezone.utc),
        s5cmd_version="v2.3.0-991c9fb",
        strategy=strategy,
        logs=tuple(entries),
    )


class LogEntryTest(unittest.TestCase):
    def test_stat_returns_zero_for_absent_items(self):
        entry = make_entry("a", annotations=False)
        self.assertEqual(entry.stat(al.SensorItem.ANNOTATIONS), ac.EMPTY_STAT)
        self.assertFalse(entry.has(al.SensorItem.ANNOTATIONS))
        self.assertTrue(entry.has(al.SensorItem.LIDAR))

    def test_total_of_a_selection(self):
        entry = make_entry("a")
        total = entry.total([al.SensorItem.MAP, al.SensorItem.CALIBRATION])
        self.assertEqual(total.num_bytes, 1_700_000 + 9_916)
        self.assertEqual(total.num_objects, 5)

    def test_total_of_everything(self):
        entry = make_entry("a")
        self.assertEqual(entry.total().num_objects, 3 + 2 + 1 + 157 + 319 + 1)

    def test_lidar_sweeps(self):
        self.assertEqual(make_entry("a", sweeps=42).num_lidar_sweeps, 42)
        no_lidar = ac.LogEntry(log_id="b", city=None, items={"map": stat(1, 1)})
        self.assertEqual(no_lidar.num_lidar_sweeps, 0)


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.catalog = make_catalog([
            make_entry("aaa11111", "PIT"),
            make_entry("bbb22222", "ATX"),
            make_entry("ccc33333", "PIT"),
        ])

    def test_cities_counts_are_sorted(self):
        self.assertEqual(self.catalog.cities(), {"ATX": 1, "PIT": 2})

    def test_get_raises_with_a_near_miss_hint(self):
        with self.assertRaises(KeyError) as ctx:
            self.catalog.get("aaa11112")
        self.assertIn("aaa11111", str(ctx.exception))

    def test_spec_and_len(self):
        self.assertEqual(self.catalog.spec, "sensor/val")
        self.assertEqual(len(self.catalog), 3)

    def test_tbv_spec_has_no_split(self):
        catalog = make_catalog([make_entry("x")], dataset=al.Dataset.TBV, split=None)
        self.assertEqual(catalog.spec, "tbv")

    def test_sizes_are_inferred_only_for_prefix_only(self):
        self.assertFalse(self.catalog.sizes_are_inferred)
        sampled = make_catalog([make_entry("x")], strategy=ac.BuildStrategy.PREFIX_ONLY)
        self.assertTrue(sampled.sizes_are_inferred)


class SerializationTest(unittest.TestCase):
    def test_round_trip_through_a_file(self):
        catalog = make_catalog([make_entry("aaa11111"), make_entry("bbb22222", None)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor_val.json"
            ac.save(catalog, path)
            loaded = ac.load(path)
        self.assertEqual(loaded, catalog)
        self.assertIsNone(loaded.logs[1].city)

    def test_schema_mismatch_is_rejected(self):
        catalog = make_catalog([make_entry("a")])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            ac.save(catalog, path)
            with mock.patch.object(ac, "SCHEMA_VERSION", ac.SCHEMA_VERSION + 1):
                with self.assertRaises(ac.CatalogError):
                    ac.load(path)

    def test_corrupt_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text("{not json")
            with self.assertRaises(ac.CatalogError):
                ac.load(path)

    def test_cache_path_uses_the_slug(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL)
        self.assertEqual(ac.cache_path(request, Path("/c")).name, "sensor_val.json")
        self.assertEqual(ac.cache_path(al.TbvRequest(), Path("/c")).name, "tbv.json")


class FilterLogsTest(unittest.TestCase):
    def setUp(self):
        self.catalog = make_catalog([
            make_entry("aaa", "PIT", sweeps=100, camera_bytes=50),
            make_entry("bbb", "ATX", sweeps=300, camera_bytes=30),
            make_entry("ccc", "PIT", sweeps=200, camera_bytes=10),
        ])

    def test_city_filter_accepts_repeated_and_comma_separated(self):
        for cities in (["PIT"], ["pit"], ["PIT,PIT"]):
            with self.subTest(cities=cities):
                entries = ac.filter_logs(self.catalog, cities=cities)
                self.assertEqual([e.log_id for e in entries], ["aaa", "ccc"])
        entries = ac.filter_logs(self.catalog, cities=["PIT,ATX"])
        self.assertEqual(len(entries), 3)

    def test_unknown_city_names_the_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            ac.filter_logs(self.catalog, cities=["MIA"])
        self.assertIn("PIT", str(ctx.exception))

    def test_sort_then_limit_is_deterministic(self):
        """--limit must be reproducible, not dependent on listing order."""
        first = ac.filter_logs(self.catalog, sort_by="sweeps", limit=2)
        second = ac.filter_logs(self.catalog, sort_by="sweeps", limit=2)
        self.assertEqual([e.log_id for e in first], ["bbb", "ccc"])
        self.assertEqual(first, second)

    def test_sort_by_bytes_is_descending(self):
        entries = ac.filter_logs(self.catalog, sort_by="bytes")
        sizes = [e.total().num_bytes for e in entries]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_unknown_sort_key_is_rejected(self):
        with self.assertRaises(ValueError):
            ac.filter_logs(self.catalog, sort_by="nonsense")

    def test_literal_log_id_that_matches_nothing_raises(self):
        """A typo'd log id must not look like an empty dataset."""
        with self.assertRaises(KeyError):
            ac.filter_logs(self.catalog, log_ids=["zzz"])

    def test_glob_log_ids(self):
        entries = ac.filter_logs(self.catalog, log_ids=["a*"])
        self.assertEqual([e.log_id for e in entries], ["aaa"])

    def test_glob_matching_nothing_raises(self):
        with self.assertRaises(KeyError):
            ac.filter_logs(self.catalog, log_ids=["z*"])

    def test_none_log_ids_keeps_everything(self):
        self.assertEqual(len(ac.filter_logs(self.catalog, log_ids=None)), 3)

    def test_has_items_filter(self):
        catalog = make_catalog([
            make_entry("aaa", annotations=True),
            make_entry("bbb", annotations=False),
        ])
        entries = ac.filter_logs(catalog, has_items=[al.SensorItem.ANNOTATIONS])
        self.assertEqual([e.log_id for e in entries], ["aaa"])

    def test_min_sweeps_filter(self):
        entries = ac.filter_logs(self.catalog, min_sweeps=200)
        self.assertEqual({e.log_id for e in entries}, {"bbb", "ccc"})


class BuildTest(unittest.TestCase):
    """Building is exercised with the s5cmd layer mocked, so no network is touched."""

    LOG_A = "02678d04-cc9f-3148-9f95-1ba66347dff9"
    LOG_B = "0b86f508-5df9-4a46-bc59-5b9536dbde9f"

    def _objects_for(self, log_id: str, city: str) -> list[s5cmd.Object]:
        base = f"{SENSOR_PREFIX}{log_id}"
        keys = [
            (f"{base}/annotations.feather", 285_474),
            (f"{base}/city_SE3_egovehicle.feather", 164_650),
            (f"{base}/calibration/intrinsics.feather", 5_330),
            (f"{base}/calibration/egovehicle_SE3_sensor.feather", 4_586),
            (f"{base}/map/log_map_archive_{log_id}____{city}_city_71109.json", 127_571),
            (f"{base}/map/{log_id}_ground_height_surface____{city}.npy", 1_660_744),
            (f"{base}/sensors/lidar/315967376019741000.feather", 900_000),
            (f"{base}/sensors/lidar/315967376119741000.feather", 900_000),
            (f"{base}/sensors/cameras/ring_front_center/315967376049927216.jpg", 285_000),
            (f"{base}/README_unexpected.txt", 10),  # must be ignored, not fatal
        ]
        return [s5cmd.Object(key=key, type="file", size=size) for key, size in keys]

    def _patched_build(self, **kwargs):
        request = al.SensorRequest(split=al.SensorSplit.VAL)
        cities = {self.LOG_A: "PIT", self.LOG_B: "ATX"}

        def fake_list_objects(uri, opts=None):
            log_id = uri[len(SENSOR_PREFIX):].rstrip("/")
            return self._objects_for(log_id, cities[log_id])

        with mock.patch.object(ac.s5cmd, "list_prefixes",
                               return_value=[self.LOG_B, self.LOG_A]), \
             mock.patch.object(ac.s5cmd, "list_objects", side_effect=fake_list_objects), \
             mock.patch.object(ac.s5cmd, "version", return_value="v2.3.0-test"):
            return ac.build(request, **kwargs)

    def test_build_aggregates_items_and_city(self):
        catalog = self._patched_build()
        self.assertEqual(len(catalog), 2)
        entry = catalog.get(self.LOG_A)
        self.assertEqual(entry.city, "PIT")
        self.assertEqual(entry.stat(al.SensorItem.MAP), stat(127_571 + 1_660_744, 2))
        self.assertEqual(entry.stat(al.SensorItem.CALIBRATION), stat(5_330 + 4_586, 2))
        self.assertEqual(entry.stat(al.SensorItem.LIDAR), stat(1_800_000, 2))
        self.assertEqual(entry.stat(al.SensorItem.POSES), stat(164_650, 1))
        self.assertEqual(entry.num_lidar_sweeps, 2)

    def test_unrecognized_keys_are_ignored(self):
        catalog = self._patched_build()
        entry = catalog.get(self.LOG_A)
        # Every recorded token must be a real item of the dataset.
        for token in entry.items:
            al.SensorItem.from_token(token)

    def test_entries_are_sorted_not_completion_ordered(self):
        """Rebuilds must be byte-identical, so order cannot depend on thread completion."""
        catalog = self._patched_build()
        self.assertEqual([e.log_id for e in catalog.logs], sorted([self.LOG_A, self.LOG_B]))

    def test_records_provenance(self):
        catalog = self._patched_build()
        self.assertEqual(catalog.s5cmd_version, "v2.3.0-test")
        self.assertEqual(catalog.strategy, ac.BuildStrategy.PER_LOG_DETAIL)
        self.assertEqual(catalog.dataset, al.Dataset.SENSOR)
        self.assertEqual(catalog.split, "val")

    def test_one_failing_log_does_not_sink_the_build(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL)

        def flaky(uri, opts=None):
            log_id = uri[len(SENSOR_PREFIX):].rstrip("/")
            if log_id == self.LOG_B:
                raise s5cmd.S5cmdError("transient")
            return self._objects_for(log_id, "PIT")

        with mock.patch.object(ac.s5cmd, "list_prefixes",
                               return_value=[self.LOG_A, self.LOG_B]), \
             mock.patch.object(ac.s5cmd, "list_objects", side_effect=flaky), \
             mock.patch.object(ac.s5cmd, "version", return_value="v"):
            catalog = ac.build(request)
        self.assertEqual([e.log_id for e in catalog.logs], [self.LOG_A])

    def test_empty_listing_is_an_error(self):
        with mock.patch.object(ac.s5cmd, "list_prefixes", return_value=[]):
            with self.assertRaises(ac.CatalogError):
                ac.build(al.SensorRequest(split=al.SensorSplit.VAL))

    def test_progress_is_reported(self):
        seen = []
        self._patched_build(progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(1, 2), (2, 2)])


class PrefixOnlyBuildTest(unittest.TestCase):
    """Motion-forecasting has ~250k scenarios, so sizes are sampled rather than measured."""

    PREFIX = "s3://argoverse/datasets/av2/motion-forecasting/val/"

    def test_sampled_sizes_are_applied_to_unmeasured_logs(self):
        ids = [f"sc{index:03d}" for index in range(60)]

        def fake_list_objects(uri, opts=None):
            log_id = uri[len(self.PREFIX):].rstrip("/")
            return [
                s5cmd.Object(key=f"{uri}scenario_{log_id}.parquet", type="file", size=160_000),
                s5cmd.Object(key=f"{uri}log_map_archive_{log_id}.json", type="file", size=60_000),
            ]

        with mock.patch.object(ac.s5cmd, "list_prefixes", return_value=ids), \
             mock.patch.object(ac.s5cmd, "list_objects", side_effect=fake_list_objects) as lister, \
             mock.patch.object(ac.s5cmd, "version", return_value="v"):
            catalog = ac.build(al.MotionForecastingRequest(split=al.MotionForecastingSplit.VAL))

        self.assertEqual(len(catalog), 60)
        self.assertEqual(lister.call_count, ac.PREFIX_ONLY_SAMPLE_SIZE)
        self.assertEqual(catalog.sampled_logs, ac.PREFIX_ONLY_SAMPLE_SIZE)
        self.assertTrue(catalog.sizes_are_inferred)
        # An unmeasured scenario still carries usable sizes.
        unmeasured = catalog.get("sc059")
        self.assertEqual(unmeasured.stat(al.MotionForecastingItem.SCENARIO).num_bytes, 160_000)
        self.assertEqual(unmeasured.stat(al.MotionForecastingItem.MAP).num_bytes, 60_000)


class LoadOrBuildTest(unittest.TestCase):
    def setUp(self):
        self.request = al.SensorRequest(split=al.SensorSplit.VAL)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = Path(self.tmp.name)

    def test_cache_hit_avoids_building(self):
        ac.save(make_catalog([make_entry("a")]), ac.cache_path(self.request, self.cache_dir))
        with mock.patch.object(ac, "build") as build:
            catalog = ac.load_or_build(self.request, cache_dir=self.cache_dir)
        build.assert_not_called()
        self.assertEqual(len(catalog), 1)

    def test_refresh_forces_a_build(self):
        ac.save(make_catalog([make_entry("a")]), ac.cache_path(self.request, self.cache_dir))
        fresh = make_catalog([make_entry("a"), make_entry("b")])
        with mock.patch.object(ac, "build", return_value=fresh) as build:
            catalog = ac.load_or_build(self.request, cache_dir=self.cache_dir, refresh=True)
        build.assert_called_once()
        self.assertEqual(len(catalog), 2)

    def test_miss_builds_and_caches(self):
        fresh = make_catalog([make_entry("a")])
        path = ac.cache_path(self.request, self.cache_dir)
        with mock.patch.object(ac, "build", return_value=fresh):
            ac.load_or_build(self.request, cache_dir=self.cache_dir)
        self.assertTrue(path.exists())

    def test_unreadable_cache_is_rebuilt_rather_than_raised(self):
        """The cache is disposable, so corruption must not be fatal."""
        path = ac.cache_path(self.request, self.cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("garbage")
        fresh = make_catalog([make_entry("a")])
        with mock.patch.object(ac, "build", return_value=fresh) as build:
            catalog = ac.load_or_build(self.request, cache_dir=self.cache_dir)
        build.assert_called_once()
        self.assertEqual(len(catalog), 1)

    def test_explicit_path_bypasses_the_cache(self):
        explicit = Path(self.tmp.name) / "explicit.json"
        ac.save(make_catalog([make_entry("z")]), explicit)
        with mock.patch.object(ac, "build") as build:
            catalog = ac.load_or_build(self.request, catalog_path=explicit,
                                       cache_dir=self.cache_dir)
        build.assert_not_called()
        self.assertEqual(catalog.logs[0].log_id, "z")


class ReadLogIdFileTest(unittest.TestCase):
    def test_comments_and_blanks_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text("# header\n\naaa\nbbb  # trailing\n\n")
            self.assertEqual(ac.read_log_id_file(path), ["aaa", "bbb"])

    def test_empty_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text("# nothing but comments\n")
            with self.assertRaises(ValueError):
                ac.read_log_id_file(path)


if __name__ == "__main__":
    unittest.main()
