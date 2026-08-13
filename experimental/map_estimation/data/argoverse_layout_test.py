import unittest
from pathlib import Path

import msgspec

from experimental.map_estimation.data import argoverse_layout as al


class ItemEnumTest(unittest.TestCase):
    """The per-dataset item enums must match what is actually in the bucket."""

    def test_sensor_has_nine_cameras_and_annotations(self):
        self.assertEqual(len(al.SensorItem.cameras()), 9)
        self.assertEqual(len(al.SensorItem.ring_cameras()), 7)
        self.assertEqual(len(al.SensorItem.stereo_cameras()), 2)
        self.assertIn(al.SensorItem.ANNOTATIONS, al.SensorItem.metadata())

    def test_tbv_has_seven_ring_cameras_and_no_annotations(self):
        self.assertEqual(len(al.TbvItem.cameras()), 7)
        self.assertEqual(al.TbvItem.stereo_cameras(), ())
        self.assertFalse(hasattr(al.TbvItem, "ANNOTATIONS"))

    def test_lidar_has_no_cameras_and_no_annotations(self):
        self.assertEqual(al.LidarItem.cameras(), ())
        self.assertFalse(hasattr(al.LidarItem, "ANNOTATIONS"))
        self.assertFalse(hasattr(al.LidarItem, "RING_FRONT_CENTER"))
        self.assertEqual(len(al.LidarItem.metadata()), 3)

    def test_motion_forecasting_has_only_scenario_and_map(self):
        self.assertEqual(
            set(al.MotionForecastingItem), {al.MotionForecastingItem.SCENARIO,
                                           al.MotionForecastingItem.MAP}
        )
        self.assertEqual(al.MotionForecastingItem.sensors(), ())

    def test_metadata_and_sensors_partition_every_dataset(self):
        """Every item is either metadata or a sensor stream, never both, never neither."""
        for item_type in al.ITEM_TYPES.values():
            with self.subTest(item_type=item_type.__name__):
                metadata = set(item_type.metadata())
                sensors = set(item_type.sensors())
                self.assertEqual(metadata & sensors, set())
                self.assertEqual(metadata | sensors, set(item_type))

    def test_tokens_are_lowercase_names(self):
        self.assertEqual(al.SensorItem.RING_FRONT_CENTER.token, "ring_front_center")
        self.assertEqual(al.SensorItem.from_token("RING_front_center"),
                         al.SensorItem.RING_FRONT_CENTER)

    def test_token_matches_member_name(self):
        """The token is spelled out in each member declaration; keep it honest.

        It cannot be derived in __new__ (the member name isn't known yet) but it must equal the
        lowercased name, since from_token() looks members up by name.
        """
        for item_type in al.ITEM_TYPES.values():
            for item in item_type:
                with self.subTest(item=f"{item_type.__name__}.{item.name}"):
                    self.assertEqual(item.token, item.name.lower())
                    self.assertEqual(item.value, item.name.lower())

    def test_relpaths_are_unique_within_a_dataset(self):
        for item_type in al.ITEM_TYPES.values():
            with self.subTest(item_type=item_type.__name__):
                relpaths = [item.relpath for item in item_type]
                self.assertEqual(len(relpaths), len(set(relpaths)))

    def test_from_token_names_the_valid_items_for_the_dataset(self):
        """A camera requested of the lidar dataset must fail with an actionable message."""
        with self.assertRaises(al.UnknownItemError) as ctx:
            al.LidarItem.from_token("ring_front_center")
        message = str(ctx.exception)
        self.assertIn("ring_front_center", message)
        self.assertIn("lidar", message)
        self.assertNotIn("ring_front_left", message)


class RequestTypeSafetyTest(unittest.TestCase):
    """Invalid dataset/item combinations must be unrepresentable or rejected at construction."""

    def test_tbv_request_has_no_split_field(self):
        self.assertNotIn("split", al.TbvRequest.__struct_fields__)
        self.assertIsNone(al.TbvRequest().split_name)
        with self.assertRaises(TypeError):
            al.TbvRequest(split=al.SensorSplit.VAL)

    def test_foreign_enum_member_is_rejected(self):
        """Catches the dynamic case that a static type checker would have flagged."""
        with self.assertRaises(al.UnknownItemError):
            al.LidarRequest(split=al.LidarSplit.VAL, items=(al.SensorItem.RING_FRONT_CENTER,))
        with self.assertRaises(al.UnknownItemError):
            al.SensorRequest(split=al.SensorSplit.VAL, items=(al.TbvItem.MAP,))

    def test_explicit_annotations_rejected_for_sensor_test_split(self):
        """The one constraint the type system cannot express, since it depends on the split."""
        with self.assertRaises(al.UnknownItemError) as ctx:
            al.SensorRequest(split=al.SensorSplit.TEST, items=(al.SensorItem.ANNOTATIONS,))
        self.assertIn("annotations", str(ctx.exception))
        # ... but explicitly asking for them is fine on train and val.
        for split in (al.SensorSplit.TRAIN, al.SensorSplit.VAL):
            al.SensorRequest(split=split, items=(al.SensorItem.ANNOTATIONS,))

    def test_default_items_are_split_aware(self):
        """Defaulting into annotations on sensor/test must not fail -- it must drop them.

        Otherwise every bare `sensor/test` command would error until the user passed --items.
        """
        test_request = al.SensorRequest(split=al.SensorSplit.TEST)
        self.assertNotIn(al.SensorItem.ANNOTATIONS, test_request.items)
        self.assertEqual(len(test_request.items), 3)

        val_request = al.SensorRequest(split=al.SensorSplit.VAL)
        self.assertIn(al.SensorItem.ANNOTATIONS, val_request.items)
        self.assertEqual(len(val_request.items), 4)

    def test_default_items_are_metadata_only(self):
        """The default must never pull imagery: that is the difference between MB and TB."""
        for spec in ["sensor/val", "tbv", "lidar/val", "motion-forecasting/val"]:
            with self.subTest(spec=spec):
                request = al.make_request(spec)
                self.assertTrue(request.items)
                for item in request.items:
                    self.assertFalse(item.is_camera)
                    self.assertFalse(item.is_lidar)

    def test_items_is_always_resolved_after_construction(self):
        self.assertIsNotNone(al.TbvRequest().items)
        self.assertIsInstance(al.TbvRequest().items, tuple)

    def test_empty_item_selection_is_rejected(self):
        """An empty tuple is an explicit mistake; None is the way to ask for the default."""
        with self.assertRaises(al.UnknownItemError):
            al.TbvRequest(items=())

    def test_requests_are_frozen(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL)
        with self.assertRaises(AttributeError):
            request.split = al.SensorSplit.TRAIN

    def test_with_log_ids_returns_a_narrowed_copy(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL)
        narrowed = request.with_log_ids(["a", "b"])
        self.assertIsNone(request.log_ids)
        self.assertEqual(narrowed.log_ids, ("a", "b"))
        self.assertEqual(narrowed.items, request.items)


class SpecParsingTest(unittest.TestCase):
    def test_round_trips(self):
        for spec in ["sensor/val", "sensor/train", "lidar/test", "motion-forecasting/val", "tbv"]:
            with self.subTest(spec=spec):
                self.assertEqual(al.make_request(spec).spec(), spec)

    def test_tbv_rejects_a_split(self):
        with self.assertRaises(al.UnknownSplitError):
            al.parse_spec("tbv/val")

    def test_missing_split_is_an_error(self):
        with self.assertRaises(al.UnknownSplitError):
            al.parse_spec("sensor")

    def test_unknown_dataset_and_split_name_the_alternatives(self):
        with self.assertRaises(al.UnknownSplitError) as ctx:
            al.parse_spec("kitti/val")
        self.assertIn("sensor", str(ctx.exception))
        with self.assertRaises(al.UnknownSplitError) as ctx:
            al.parse_spec("sensor/validation")
        self.assertIn("val", str(ctx.exception))

    def test_slug_is_filesystem_safe(self):
        self.assertEqual(al.make_request("sensor/val").slug(), "sensor_val")
        self.assertEqual(al.make_request("tbv").slug(), "tbv")
        self.assertEqual(
            al.make_request("motion-forecasting/train").slug(), "motion_forecasting_train"
        )


class ResolveItemsTest(unittest.TestCase):
    def test_groups_resolve_against_the_dataset(self):
        self.assertEqual(len(al.resolve_items(al.SensorItem, ["cameras"])), 9)
        self.assertEqual(len(al.resolve_items(al.TbvItem, ["cameras"])), 7)

    def test_empty_group_for_a_dataset_is_an_error(self):
        """'cameras' is meaningless for the lidar dataset and must say so."""
        with self.assertRaises(al.UnknownItemError):
            al.resolve_items(al.LidarItem, ["cameras"])

    def test_comma_separated_and_mixed_tokens(self):
        items = al.resolve_items(al.SensorItem, ["metadata,lidar", "ring_front_center"])
        self.assertEqual(
            items,
            (
                al.SensorItem.MAP,
                al.SensorItem.CALIBRATION,
                al.SensorItem.POSES,
                al.SensorItem.ANNOTATIONS,
                al.SensorItem.LIDAR,
                al.SensorItem.RING_FRONT_CENTER,
            ),
        )

    def test_duplicates_collapse_and_order_follows_the_enum(self):
        items = al.resolve_items(al.SensorItem, ["lidar", "map", "lidar", "metadata"])
        self.assertEqual(len(items), 5)
        self.assertEqual(items[0], al.SensorItem.MAP)
        self.assertEqual(items[-1], al.SensorItem.LIDAR)


class PathBuildingTest(unittest.TestCase):
    LOG = "02678d04-cc9f-3148-9f95-1ba66347dff9"

    def test_s3_uri_wildcards_only_directory_items(self):
        request = al.SensorRequest(split=al.SensorSplit.VAL)
        self.assertEqual(
            al.s3_uri(request, self.LOG, al.SensorItem.LIDAR),
            f"s3://argoverse/datasets/av2/sensor/val/{self.LOG}/sensors/lidar/*",
        )
        self.assertEqual(
            al.s3_uri(request, self.LOG, al.SensorItem.POSES),
            f"s3://argoverse/datasets/av2/sensor/val/{self.LOG}/city_SE3_egovehicle.feather",
        )

    def test_tbv_uri_has_no_split_segment(self):
        request = al.TbvRequest()
        self.assertEqual(
            al.s3_uri(request, "abc__Summer_2019", al.TbvItem.MAP),
            "s3://argoverse/datasets/av2/tbv/abc__Summer_2019/map/*",
        )

    def test_motion_forecasting_filenames_embed_the_log_id(self):
        request = al.MotionForecastingRequest(split=al.MotionForecastingSplit.VAL)
        self.assertEqual(
            al.s3_uri(request, "sc1", al.MotionForecastingItem.SCENARIO),
            "s3://argoverse/datasets/av2/motion-forecasting/val/sc1/scenario_sc1.parquet",
        )
        self.assertEqual(
            al.s3_uri(request, "sc1", al.MotionForecastingItem.MAP),
            "s3://argoverse/datasets/av2/motion-forecasting/val/sc1/log_map_archive_sc1.json",
        )

    def test_local_layout_mirrors_s3(self):
        """The local path must be the S3 key with the prefix swapped, so the two stay in step."""
        root = Path("/data/map_estimation/datasets/argoverse")
        request = al.SensorRequest(split=al.SensorSplit.VAL)
        uri = al.s3_uri(request, self.LOG, al.SensorItem.LIDAR).removesuffix("/*")
        path = al.local_path(request, self.LOG, al.SensorItem.LIDAR, root)
        self.assertEqual(uri.removeprefix(f"{al.S3_ROOT}/"), str(path.relative_to(root)))

    def test_default_root(self):
        self.assertEqual(al.DEFAULT_ROOT, Path("/data/map_estimation/datasets/argoverse"))


class ClassifyKeyTest(unittest.TestCase):
    """Real keys captured from the bucket."""

    LOG = "02678d04-cc9f-3148-9f95-1ba66347dff9"

    def test_sensor_keys(self):
        cases = {
            "annotations.feather": al.SensorItem.ANNOTATIONS,
            "city_SE3_egovehicle.feather": al.SensorItem.POSES,
            "calibration/intrinsics.feather": al.SensorItem.CALIBRATION,
            f"map/log_map_archive_{self.LOG}____PIT_city_71109.json": al.SensorItem.MAP,
            f"map/{self.LOG}_ground_height_surface____PIT.npy": al.SensorItem.MAP,
            "sensors/lidar/315967376019741000.feather": al.SensorItem.LIDAR,
            "sensors/cameras/ring_front_center/315967376049927216.jpg":
                al.SensorItem.RING_FRONT_CENTER,
            "sensors/cameras/stereo_front_left/315967376099904000.jpg":
                al.SensorItem.STEREO_FRONT_LEFT,
        }
        for rel_key, expected in cases.items():
            with self.subTest(rel_key=rel_key):
                self.assertEqual(al.classify_key(al.SensorItem, rel_key, self.LOG), expected)

    def test_unrecognized_key_returns_none(self):
        self.assertIsNone(al.classify_key(al.SensorItem, "README.txt", self.LOG))
        # A ring camera is not a lidar-dataset item, so it must not classify.
        self.assertIsNone(
            al.classify_key(
                al.LidarItem, "sensors/cameras/ring_front_center/1.jpg", self.LOG
            )
        )

    def test_motion_forecasting_keys_use_the_log_id(self):
        self.assertEqual(
            al.classify_key(al.MotionForecastingItem, "scenario_sc1.parquet", "sc1"),
            al.MotionForecastingItem.SCENARIO,
        )
        # The same key under a different log id must not match.
        self.assertIsNone(
            al.classify_key(al.MotionForecastingItem, "scenario_sc1.parquet", "sc2")
        )


class CityFromMapKeyTest(unittest.TestCase):
    def test_sensor_and_tbv_forms(self):
        log = "02678d04-cc9f-3148-9f95-1ba66347dff9"
        self.assertEqual(
            al.city_from_map_key(f"map/log_map_archive_{log}____PIT_city_71109.json"), "PIT"
        )
        self.assertEqual(
            al.city_from_map_key(f"map/{log}_ground_height_surface____PIT.npy"), "PIT"
        )
        self.assertEqual(
            al.city_from_map_key(
                "map/log_map_archive_01bb304d-7bd8-35f8-bbef-7086b688e35e____WDC_city_38282.json"
            ),
            "WDC",
        )

    def test_lidar_form_with_a_season_segment(self):
        self.assertEqual(
            al.city_from_map_key(
                "map/log_map_archive_00tznBNqsndfkyfy4w00AxSNPmvmAK6v__Summer____ATX_city_77093.json"
            ),
            "ATX",
        )

    def test_keys_without_a_city(self):
        log = "02678d04-cc9f-3148-9f95-1ba66347dff9"
        self.assertIsNone(al.city_from_map_key(f"map/{log}___img_Sim2_city.json"))
        self.assertIsNone(al.city_from_map_key("log_map_archive_sc1.json"))
        self.assertIsNone(al.city_from_map_key("annotations.feather"))


class SerializationTest(unittest.TestCase):
    """Requests form a tagged union, so they round-trip through the config machinery."""

    def test_request_round_trip(self):
        request = al.SensorRequest(
            split=al.SensorSplit.VAL,
            items=(al.SensorItem.MAP, al.SensorItem.LIDAR),
            log_ids=("abc",),
        )
        encoded = msgspec.json.encode(request)
        decoded = msgspec.json.decode(encoded, type=al.Request)
        self.assertEqual(decoded, request)
        self.assertIsInstance(decoded, al.SensorRequest)

    def test_union_discriminates_on_dataset_type(self):
        for request in [
            al.SensorRequest(split=al.SensorSplit.TRAIN),
            al.TbvRequest(),
            al.LidarRequest(split=al.LidarSplit.VAL),
            al.MotionForecastingRequest(split=al.MotionForecastingSplit.TEST),
        ]:
            with self.subTest(request=type(request).__name__):
                decoded = msgspec.json.decode(msgspec.json.encode(request), type=al.Request)
                self.assertIsInstance(decoded, type(request))


if __name__ == "__main__":
    unittest.main()
