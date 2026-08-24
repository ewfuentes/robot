import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shapely.geometry import Point

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.catalog import (
    lineage,
    schema,
)
from experimental.overhead_matching.swag.farfield.collection import (
    active_catalogs as subject,
    geometry_helpers,
)


class ActiveCatalogsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "farfield"
        self.root.mkdir()
        self.poly_cache = self.base / "poly"
        self.enc_root = self.root / "enc"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_mapping(self, dataset, points):
        dataset_dir = self.root / "datasets" / dataset
        path = dataset_dir / "pano_id_mapping.csv"
        path.parent.mkdir(parents=True)
        rows = []
        for index, (lat, lon) in enumerate(points):
            pano_id = f"f{index:06d}"
            filename = f"{pano_id},{lat:.7f},{lon:.7f},.jpg"
            rows.append({
                "pano_id": pano_id,
                "lat": lat,
                "lon": lon,
                "filename": filename,
            })
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=subject.MAPPING_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with (dataset_dir / "frames_gps.csv").open(
                "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "idx", "latitude", "longitude", "dist_m", "video_t_s"))
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow({
                    "idx": index,
                    "latitude": row["lat"],
                    "longitude": row["lon"],
                    "dist_m": float(index),
                    "video_t_s": float(index),
                })
        frames = dataset_dir / "frames"
        frames.mkdir()
        for row in rows:
            (frames / row["filename"]).write_bytes(b"jpeg")
        (dataset_dir / "panorama").symlink_to(
            "frames", target_is_directory=True)
        return path

    def _write_pbf(self, spec, vintage="260824"):
        stem = spec.rsplit("/", 1)[-1].replace("latest", vintage)
        path = self.root / "pbfs" / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {spec}".encode())
        return path

    def _plan(self, scope_name, pbf_map, *, enc_root=None, poly_cache=None):
        with mock.patch.object(
                subject.pbf_coverage, "check_coverage",
                return_value=(True, "full coverage", [{"gate": "passed"}])):
            return subject.build_plan(
                farfield_root=self.root,
                scope_names=[scope_name],
                pbf_map=pbf_map,
                poly_cache_dir=poly_cache or self.poly_cache,
                enc_root=enc_root,
                catalog_version="full_v2",
                dedupe_tolerance_m=10.0)

    def test_fixed_scope_order_and_policies(self):
        self.assertEqual(
            [scope.name for scope in subject.ACTIVE_SCOPES],
            ["boston_harbor_20260712", "charles_river_20260727",
             "mount_washington_20260815", "pohang_canal_04"])
        boston, charles, washington, pohang = subject.ACTIVE_SCOPES
        self.assertEqual(boston.output_datasets, (
            "boston_harbor_leg1", "boston_harbor_leg2",
            "boston_harbor_leg3"))
        self.assertEqual(boston.output_datasets, boston.bbox_datasets)
        self.assertEqual(charles.enc_state, "MA")
        self.assertEqual(washington.osm_specs, (
            "north-america/us/new-hampshire-latest.osm.pbf",
            "north-america/us/maine-latest.osm.pbf"))
        self.assertIsNone(washington.enc_state)
        self.assertEqual(pohang.output_datasets, ("pohang_canal_04",))
        self.assertEqual(subject.BBOX_BUFFER_KM, 25.0)
        self.assertEqual(subject.ENC_BAND, 5)
        self.assertEqual(subject.OSM_GEOMETRY_INDEX_MODE,
                         "full_pbf_complete_geometry_index")

    def test_plan_uses_exact_union_table_bbox_and_binds_inputs(self):
        datasets = subject.SCOPE_BY_NAME[
            "boston_harbor_20260712"].bbox_datasets
        coordinates = {
            datasets[0]: [(42.0, -71.0), (42.1, -70.9)],
            datasets[1]: [(41.9, -70.8)],
            datasets[2]: [(42.2, -71.1)],
        }
        for dataset, points in coordinates.items():
            self._write_mapping(dataset, points)
        spec = subject.SCOPE_BY_NAME[
            "boston_harbor_20260712"].osm_specs[0]
        pbf = self._write_pbf(spec)

        plan = self._plan(
            "boston_harbor_20260712", {spec: pbf},
            enc_root=self.enc_root)

        scope = plan["scopes"][0]
        all_points = [point for points in coordinates.values()
                      for point in points]
        expected = geometry_helpers.padded_bbox_wsen(
            [lat for lat, _ in all_points],
            [lon for _, lon in all_points], 25.0)
        self.assertEqual(scope["bbox_wsen"], list(expected))
        self.assertEqual(scope["bbox_datasets"], list(datasets))
        self.assertEqual(scope["output_datasets"], list(datasets))
        self.assertEqual(scope["pbf_inputs"][0]["sha256"],
                         artifact.sha256_file(pbf))
        self.assertEqual(scope["enc_policy"], {
            "catalog_state": "MA", "band": 5,
            "explicit_cells": False, "include_buoys": True,
            "identity_phase": "materialize",
            "published_identity_binding": "selection_sha256",
        })
        self.assertEqual(plan["generator_git_commit"],
                         subject.provenance.git_commit())
        self.assertEqual(plan["report_io"], {
            "geofabrik_metadata_cache": "may_fetch_missing_poly_and_index",
            "downloads_pbf": False,
            "downloads_enc": False,
            "builds_catalog": False,
        })
        self.assertEqual(
            plan["plan_digest"],
            artifact.sha256_json({
                key: value for key, value in plan.items()
                if key != "plan_digest"
            }))

    def test_pbf_map_must_be_exact_and_regular(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]
        pbf = self._write_pbf(spec)
        extra = "asia/japan-latest.osm.pbf"
        with self.assertRaisesRegex(subject.ActiveCatalogError, "extra"):
            self._plan(dataset, {spec: pbf, extra: pbf})

        link = self.root / "pbfs" / "south-korea-link.osm.pbf"
        link.symlink_to(pbf)
        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "regular non-symlink"):
            self._plan(dataset, {spec: link})

    def test_stale_mapping_that_disagrees_with_current_gps_fails(self):
        dataset = "pohang_canal_04"
        mapping = self._write_mapping(dataset, [(36.0, 129.4)])
        with mapping.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["lat"] = "35.0"
        with mapping.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=subject.MAPPING_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "stale mapping coordinates"):
            self._plan(dataset, {spec: self._write_pbf(spec)})

    def test_canonical_gps_without_frame_file_is_accepted(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        gps = self.root / "datasets" / dataset / "frames_gps.csv"
        with gps.open(newline="") as stream:
            self.assertNotIn("frame_file", csv.DictReader(stream).fieldnames)
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        plan = self._plan(dataset, {spec: self._write_pbf(spec)})

        self.assertEqual(plan["scopes"][0]["dataset_tables"][0]["rows"], 1)

    def test_panorama_member_symlink_is_rejected(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        panorama = self.root / "datasets" / dataset / "panorama"
        member = next(panorama.glob("*.jpg"))
        target = self.base / "outside.jpg"
        target.write_bytes(b"jpeg")
        member.unlink()
        member.symlink_to(target)
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "per-JPEG symlink"):
            self._plan(dataset, {spec: self._write_pbf(spec)})

    def test_malformed_extra_panorama_is_rejected(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        panorama = self.root / "datasets" / dataset / "panorama"
        (panorama / "not-a-canonical-frame.jpg").write_bytes(b"jpeg")
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "canonical frame contract"):
            self._plan(dataset, {spec: self._write_pbf(spec)})

    def test_panorama_alternate_relative_target_is_rejected(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        dataset_dir = self.root / "datasets" / dataset
        panorama = dataset_dir / "panorama"
        (dataset_dir / "alternate").mkdir()
        panorama.unlink()
        panorama.symlink_to("alternate", target_is_directory=True)
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "text must be exactly 'frames'"):
            self._plan(dataset, {spec: self._write_pbf(spec)})

    def test_dangling_exact_panorama_link_is_rejected(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        dataset_dir = self.root / "datasets" / dataset
        (dataset_dir / "frames").rename(dataset_dir / "missing_frames")
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "dangling or cyclic"):
            self._plan(dataset, {spec: self._write_pbf(spec)})

    def test_symlinked_frames_directory_is_rejected(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        dataset_dir = self.root / "datasets" / dataset
        frames = dataset_dir / "frames"
        frames.rename(dataset_dir / "real_frames")
        frames.symlink_to("real_frames", target_is_directory=True)
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "real non-symlink directory"):
            self._plan(dataset, {spec: self._write_pbf(spec)})

    def test_report_cache_cannot_be_nested_under_farfield_root(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "outside farfield_root"):
            self._plan(
                dataset, {spec: self._write_pbf(spec)},
                poly_cache=self.root / "datasets" / "osm" / "poly")

    def test_cli_is_report_only_without_explicit_digest_gate(self):
        fake_plan = {
            "schema": subject.PLAN_SCHEMA,
            "generator": subject.GENERATOR,
            "scopes": [],
        }
        body = dict(fake_plan)
        fake_plan["plan_digest"] = artifact.sha256_json(body)
        argv = [
            "--farfield_root", str(self.root),
            "--scope", "pohang_canal_04",
            "--pbf", "asia/south-korea-latest.osm.pbf=/not/read",
            "--poly_cache_dir", str(self.poly_cache),
            "--catalog_version", "full_v2",
            "--dedupe_tolerance_m", "10",
        ]
        with mock.patch.object(subject, "build_plan", return_value=fake_plan), \
             mock.patch.object(subject, "materialize") as materialize, \
             mock.patch("builtins.print"):
            self.assertEqual(subject.main(argv), 0)
        materialize.assert_not_called()

        with self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "expected plan digest"):
            subject.materialize(
                fake_plan, expected_plan_digest="0" * 64)

    def test_materialize_rejects_plan_from_another_generator_commit(self):
        dataset = "pohang_canal_04"
        self._write_mapping(dataset, [(36.0, 129.4)])
        spec = subject.SCOPE_BY_NAME[dataset].osm_specs[0]
        plan = self._plan(dataset, {spec: self._write_pbf(spec)})

        with mock.patch.object(subject.provenance, "git_commit",
                               return_value="different-commit"), \
             self.assertRaisesRegex(subject.ActiveCatalogError,
                                    "generator commit"):
            subject.materialize(
                plan, expected_plan_digest=plan["plan_digest"])

    def test_materialize_extracts_scope_once_and_publishes_regular_copies(self):
        scope = subject.SCOPE_BY_NAME["mount_washington_20260815"]
        for index, dataset in enumerate(scope.bbox_datasets):
            self._write_mapping(dataset, [(44.2 + index * 0.01, -71.3)])
        pbf_map = {spec: self._write_pbf(spec) for spec in scope.osm_specs}
        plan = self._plan(scope.name, pbf_map)

        def write_frame(output_base, ids):
            output = Path(output_base).with_suffix(".feather")
            output.parent.mkdir(parents=True, exist_ok=True)
            frame = schema.build_frame(
                ids=ids,
                geometries=[Point(-71.3 + position * 0.01, 44.2)
                            for position in range(len(ids))],
                landmark_types=["osm"] * len(ids),
                tags=[{"man_made": "tower"}] * len(ids))
            frame.to_feather(output)

        def fake_extract(*, pbf_file, bbox, output_path):
            del bbox
            write_frame(output_path, [f"osm:node:{pbf_file.name}"])

        def fake_merge(*, inputs, output, dedupe_tolerance_m,
                       collision_radius_m):
            self.assertEqual(dedupe_tolerance_m, 10.0)
            self.assertEqual(collision_radius_m, 150.0)
            write_frame(output, [f"osm:node:{position + 1}"
                                 for position in range(len(inputs))])

        with mock.patch.object(
                subject.extract_landmarks_from_osm, "main",
                side_effect=fake_extract) as extract, \
             mock.patch.object(
                 subject.merge_landmark_feathers, "main",
                 side_effect=fake_merge) as merge, \
             mock.patch.object(
                 subject.download_enc_cells, "main") as enc_download, \
             mock.patch.object(subject.publication.indexes, "refresh"):
            refs = subject.materialize(
                plan, expected_plan_digest=plan["plan_digest"])

        self.assertEqual(extract.call_count, len(scope.osm_specs))
        self.assertEqual(merge.call_count, 1)
        enc_download.assert_not_called()
        self.assertEqual([ref.dataset for ref in refs],
                         list(scope.output_datasets))
        payloads = [Path(ref.path) / "catalog.feather" for ref in refs]
        self.assertTrue(all(path.is_file() and not path.is_symlink()
                            for path in payloads))
        self.assertEqual(len({artifact.sha256_file(path)
                              for path in payloads}), 1)
        self.assertEqual(len({path.resolve() for path in payloads}), 3)
        for ref, payload in zip(refs, payloads, strict=True):
            self.assertEqual(
                len(schema.read_frame(payload)), len(scope.osm_specs))
            self.assertEqual(
                lineage.require_passed_source_coverage(ref), ref)
            manifest = artifact.load_manifest(ref.path)
            self.assertEqual(manifest.upstreams, ())
            self.assertEqual(manifest.config["osm_geometry_index_mode"],
                             subject.OSM_GEOMETRY_INDEX_MODE)


if __name__ == "__main__":
    unittest.main()
