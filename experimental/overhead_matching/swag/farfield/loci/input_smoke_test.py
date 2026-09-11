import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield.loci import input_smoke


def _reference(kind: str, *, manifest_digest: str,
               dataset: str = "same", path: str = "/tmp/region"):
    reference = SimpleNamespace(
        kind=kind,
        dataset=dataset,
        version="v1",
        manifest_digest=manifest_digest,
        content_digest="c" * 64,
        path=path,
    )
    reference.to_dict = lambda: {
        "kind": reference.kind,
        "dataset": reference.dataset,
        "version": reference.version,
        "manifest_digest": reference.manifest_digest,
        "content_digest": reference.content_digest,
        "path": reference.path,
    }
    return reference


class InputSmokeTest(unittest.TestCase):
    def _valid_bundle(self):
        shared_region = _reference(
            "loci_regions", manifest_digest="r" * 64,
            dataset="boston_harbor_shared")
        satellite = _reference(
            "loci_satellite", manifest_digest="s" * 64,
            dataset="boston_harbor_shared")
        osm = _reference(
            "loci_osm_landmarks", manifest_digest="o" * 64,
            dataset="boston_harbor_shared")
        grid = {
            "schema": "loci_web_mercator_grid/v1",
            "zoom": 19,
            "n_patches": 3,
            "footprint_bbox_wsen": [-71.1, 42.2, -70.9, 42.4],
        }
        manifests = [
            SimpleNamespace(
                upstreams=(shared_region,),
                config={
                    "region_manifest_digest": shared_region.manifest_digest,
                    "grid": grid,
                }),
            SimpleNamespace(
                upstreams=(shared_region,),
                config={
                    "region_manifest_digest": shared_region.manifest_digest,
                    "footprint_bbox_wsen": grid["footprint_bbox_wsen"],
                }),
        ]
        plan = {
            "trajectory": {
                "datasets": [
                    "boston_harbor_leg1",
                    "boston_harbor_leg2",
                    "boston_harbor_leg3",
                ],
            },
            "grid": grid,
        }
        loaded = SimpleNamespace(
            _satellite_metadata=[object()] * grid["n_patches"],
            _panorama_metadata=[object()] * 2,
            _landmark_metadata=[object()] * 5,
            _pairs=[object()] * 2,
        )
        return {
            "region": shared_region,
            "satellite": satellite,
            "osm": osm,
            "manifests": manifests,
            "plan": plan,
            "loaded": loaded,
        }

    def _run_bundle(self, bundle):
        with mock.patch.object(
                input_smoke.artifact, "open_artifact",
                side_effect=(bundle["satellite"], bundle["osm"])), \
                mock.patch.object(
                    input_smoke.artifact, "load_manifest",
                    side_effect=bundle["manifests"]), mock.patch.object(
                        input_smoke.region, "load_region",
                        return_value=(bundle["region"], bundle["plan"])), \
                mock.patch.object(
                    input_smoke, "VigorDataset",
                    return_value=bundle["loaded"]) as constructor:
            result = input_smoke.run(
                Path("/tmp/boston_harbor_leg2"),
                Path("satellite"), Path("osm"))
        return result, constructor

    def test_rejects_artifacts_from_different_region_upstreams(self):
        satellite = _reference(
            "loci_satellite", manifest_digest="s" * 64)
        osm = _reference(
            "loci_osm_landmarks", manifest_digest="o" * 64)
        first_region = _reference(
            "loci_regions", manifest_digest="a" * 64)
        second_region = _reference(
            "loci_regions", manifest_digest="b" * 64)
        manifests = [
            SimpleNamespace(upstreams=(first_region,)),
            SimpleNamespace(upstreams=(second_region,)),
        ]
        with mock.patch.object(
                input_smoke.artifact, "open_artifact",
                side_effect=(satellite, osm)), mock.patch.object(
                    input_smoke.artifact, "load_manifest",
                    side_effect=manifests):
            with self.assertRaisesRegex(
                    ValueError, "different loci_regions"):
                input_smoke.run(
                    Path("/tmp/same"), Path("satellite"), Path("osm"))

    def test_shared_artifacts_accept_each_region_trajectory_and_use_grid_zoom(self):
        result, dataset_constructor = self._run_bundle(
            self._valid_bundle())

        self.assertEqual(result["dataset"], "boston_harbor_leg2")
        self.assertEqual(result["artifact_scope"], "boston_harbor_shared")
        self.assertEqual(result["satellite_zoom_level"], 19)
        self.assertEqual(result["n_satellites"], 3)
        config = dataset_constructor.call_args.args[1]
        self.assertEqual(config.satellite_zoom_level, 19)

    def test_relocated_tree_does_not_dereference_recorded_region_path(self):
        bundle = self._valid_bundle()
        stored_region = bundle["region"]
        stored_region.path = "/old/root/artifacts/loci_regions/" \
            "boston_harbor_shared/v1"
        live_region = _reference(
            stored_region.kind,
            manifest_digest=stored_region.manifest_digest,
            dataset=stored_region.dataset,
            path="/new/root/artifacts/loci_regions/"
                 "boston_harbor_shared/v1")
        bundle["region"] = live_region
        satellite_path = Path(
            "/new/root/artifacts/loci_satellite/"
            "boston_harbor_shared/satellite_v1")
        osm_path = Path(
            "/new/root/artifacts/loci_osm_landmarks/"
            "boston_harbor_shared/osm_v1")

        with mock.patch.object(
                input_smoke.artifact, "open_artifact",
                side_effect=(bundle["satellite"], bundle["osm"])), \
                mock.patch.object(
                    input_smoke.artifact, "load_manifest",
                    side_effect=bundle["manifests"]), mock.patch.object(
                        input_smoke.region, "load_region",
                        return_value=(live_region, bundle["plan"])
                    ) as load_region, mock.patch.object(
                        input_smoke, "VigorDataset",
                        return_value=bundle["loaded"]):
            input_smoke.run(
                Path("/tmp/boston_harbor_leg2"),
                satellite_path, osm_path)

        load_region.assert_called_once_with(Path(
            "/new/root/artifacts/loci_regions/boston_harbor_shared/v1"))

    def test_rejects_configured_region_manifest_digest_mismatch(self):
        for artifact_index, label in ((0, "satellite"), (1, "OSM")):
            with self.subTest(artifact=label):
                bundle = self._valid_bundle()
                bundle["manifests"][artifact_index].config[
                    "region_manifest_digest"] = "x" * 64
                with self.assertRaisesRegex(
                        ValueError,
                        f"{label} artifact config has a different region"):
                    self._run_bundle(bundle)

    def test_rejects_satellite_grid_mismatch(self):
        bundle = self._valid_bundle()
        bundle["manifests"][0].config["grid"] = {
            **bundle["plan"]["grid"],
            "zoom": 20,
        }
        with self.assertRaisesRegex(ValueError, "satellite artifact grid"):
            self._run_bundle(bundle)

    def test_rejects_osm_footprint_mismatch(self):
        bundle = self._valid_bundle()
        bundle["manifests"][1].config["footprint_bbox_wsen"] = [
            -71.0, 42.2, -70.9, 42.4]
        with self.assertRaisesRegex(ValueError, "OSM artifact footprint"):
            self._run_bundle(bundle)

    def test_rejects_loaded_satellite_count_mismatch(self):
        bundle = self._valid_bundle()
        bundle["loaded"]._satellite_metadata = [object()] * 2
        with self.assertRaisesRegex(
                ValueError, "loaded satellite metadata count"):
            self._run_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
