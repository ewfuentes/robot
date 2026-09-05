import dataclasses
import tempfile
import unittest
from pathlib import Path

import msgspec
import numpy as np

from experimental.overhead_matching.swag.farfield import artifact
from experimental.overhead_matching.swag.farfield.localization import (
    run_io,
    structs,
)


class FakeBelief:
    def __init__(self, n=8):
        rng = np.random.default_rng(0)
        self.east_m = rng.normal(0.0, 100.0, n)
        self.north_m = rng.normal(0.0, 100.0, n)
        self.heading_rad = rng.uniform(-np.pi, np.pi, n)
        self.log_weight = np.full(n, -np.log(n))
        self.proposal_event_id = np.full(n, -1, dtype=np.int64)
        self.proposal_hypothesis = np.full(n, -1, dtype=np.int64)
        self.mode_id = np.zeros(n, dtype=np.int64)


class FakeHistory:
    def __init__(self, *, n_measurements_at_1=1):
        self.health = [structs.HealthRecord(
            keyframe_idx=k, ess=8.0, resampled=False, mean_east_m=0.0,
            mean_north_m=0.0, mean_heading_deg=0.0, map_east_m=0.0,
            map_north_m=0.0, map_heading_deg=0.0, position_std_m=10.0,
            heading_std_deg=5.0,
            n_measurements=(n_measurements_at_1 if k == 1 else 0),
            proposal_event_id=(0 if k == 1 else None)) for k in range(3)]
        self.proposal_events = [structs.ProposalEvent(
            event_id=0, keyframe_idx=1, trigger="init", n_hypotheses=4,
            n_injected=0, n_tracklets_considered=3,
            n_combinations_examined=10, n_combinations_skipped=0,
            gate_passed=False)]
        self.mode_events = [structs.ModeEvent(
            keyframe_idx=1, kind="birth", mode_id=0)]
        self.checkpoints = {0: FakeBelief(), 2: FakeBelief()}


def make_manifest(**overrides):
    fields = dict(
        schema_version=structs.SCHEMA_VERSION,
        dataset="synthetic",
        scenario_name="tiny",
        run_kind="synthetic",
        initialization_kind="test",
        bearings_consumed=True,
        proposal_enabled=True,
        localization_inputs_manifest_sha256=None,
        anchor_lat_deg=42.35,
        anchor_lon_deg=-71.05,
        n_keyframes=3,
        filter_config=structs.FilterConfig(
            n_particles=8, seed=1,
            init=structs.GaussianInit(0.0, 0.0, 100.0)),
        landmarks=[structs.LandmarkEntry(
            "osm:node:1", 42.36, -71.05, "x", 10.0)],
        matcher_version="m",
        max_visible_range_m=10000.0,
        export_dir="synthetic:tiny",
        git_commit="deadbeef",
        argv=["run_export", "--flag"],
        created="2026-08-20T00:00:00+00:00",
        particle_history_sha256="0" * 64,
    )
    fields.update(overrides)
    return structs.RunManifest(**fields)


def sample_payload(*, bearings=True):
    manifest = make_manifest(
        bearings_consumed=bearings,
        ablation_tags=[] if bearings else ["no_bearings"])
    history = FakeHistory(n_measurements_at_1=int(bearings))
    truth = [structs.TruthPose(k, 0.0, 0.0, 0.0) for k in range(3)]
    odometry = [structs.OdometryDelta(k, 40.0, 0.0, 0.0, 1.0, 0.02)
                for k in (1, 2)]
    measurements = (
        [structs.TrackletMeasurement("T1", 1, 45.0, 100.0)]
        if bearings else [])
    tables = ({"T1": structs.CompatibilityTable(
        "T1", "m", [], 0.0, -4.0, 4.0, "fast")} if bearings else {})
    return manifest, truth, odometry, measurements, tables, history


def write_sample(run_dir: Path, *, bearings=True):
    values = sample_payload(bearings=bearings)
    manifest, truth, odometry, measurements, tables, history = values
    run_io.write_run(
        run_dir, manifest, truth, odometry, measurements, tables, history,
        dataset="synthetic", version=run_dir.name)
    return values


def repair_content_digest(run_dir: Path) -> None:
    outer = artifact.load_manifest(run_dir)
    outer = dataclasses.replace(
        outer, content_digest=artifact.sha256_directory(run_dir))
    artifact.atomic_write_json(
        run_dir / artifact.MANIFEST_NAME, outer.to_dict())


def replace_payload(run_dir: Path, relative: str, payload: bytes) -> None:
    artifact.atomic_write_file(run_dir / relative, payload)
    repair_content_digest(run_dir)


def stamp_retired_experiment_fields(
        run_dir: Path, *, measurement_damage_cap_nats=None,
        revival_enabled=False, revival_margin_nats=0.0,
        revival_match_radius_m=None) -> None:
    def update_filter_config(document):
        filter_config = document["filter_config"]
        filter_config["measurement_damage_cap_nats"] = \
            measurement_damage_cap_nats
        proposal = filter_config["proposal"]
        proposal["revival_enabled"] = revival_enabled
        proposal["revival_margin_nats"] = revival_margin_nats
        proposal["revival_match_radius_m"] = revival_match_radius_m

    run_manifest_path = run_dir / run_io.RUN_MANIFEST_NAME
    run_manifest = msgspec.json.decode(run_manifest_path.read_bytes())
    update_filter_config(run_manifest)
    artifact.atomic_write_json(run_manifest_path, run_manifest)

    outer = artifact.load_manifest(run_dir)
    config = dict(outer.config)
    contract = dict(config[run_io.RUN_CONTRACT_CONFIG_KEY])
    update_filter_config(contract)
    config[run_io.RUN_CONTRACT_CONFIG_KEY] = contract
    artifact.atomic_write_json(
        run_dir / artifact.MANIFEST_NAME,
        dataclasses.replace(outer, config=config).to_dict())
    repair_content_digest(run_dir)


class RoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            manifest, truth, odometry, measurements, tables, history = \
                write_sample(run_dir)
            serialized_truth = (run_dir / "truth.jsonl").read_text()
            self.assertIn('"course_world_cw_deg"', serialized_truth)
            self.assertNotIn('"heading_deg"', serialized_truth)
            loaded = run_io.read_run(run_dir)

        self.assertEqual(loaded.manifest, manifest)
        self.assertEqual(loaded.truth, truth)
        self.assertEqual(loaded.odometry, odometry)
        self.assertEqual(loaded.measurements, measurements)
        self.assertEqual(loaded.tables, tables)
        self.assertEqual(loaded.health, history.health)
        self.assertEqual(loaded.proposal_events, history.proposal_events)
        self.assertEqual(loaded.mode_events, history.mode_events)
        self.assertEqual(set(loaded.checkpoints), {0, 2})
        np.testing.assert_array_equal(
            loaded.checkpoints[0]["east_m"],
            history.checkpoints[0].east_m)

    def test_explicitly_optional_empty_payloads_round_trip(self):
        # Truth is optional as one whole sequence when GPS course abstains;
        # measurement/table emptiness is explicit when bearings are withheld.
        manifest, _, odometry, measurements, tables, history = \
            sample_payload(bearings=False)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_io.write_run(
                run_dir, manifest, [], odometry, measurements, tables,
                history, dataset="synthetic", version=run_dir.name)
            loaded = run_io.read_run(run_dir)
        self.assertEqual(loaded.truth, [])
        self.assertEqual(loaded.measurements, [])
        self.assertEqual(loaded.tables, {})

    def test_manifest_and_payload_are_validated_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, mutate in enumerate((
                    lambda values: values.__setitem__(0,
                        make_manifest(export_dir="")),
                    lambda values: values.__setitem__(2, []),
                    lambda values: setattr(values[5], "health", []),
                    lambda values: values[5].checkpoints.pop(2),
                    lambda values: values.__setitem__(4, {}),
            )):
                values = list(sample_payload())
                mutate(values)
                run_dir = Path(tmp) / f"run_{index}"
                with self.subTest(index=index), self.assertRaises(ValueError):
                    run_io.write_run(
                        run_dir, *values[0:5], values[5],
                        dataset="synthetic", version=run_dir.name)
                self.assertFalse(run_dir.exists())
                self.assertFalse(
                    run_dir.with_name(run_dir.name + ".incomplete").exists())

    def test_foreign_schema_is_rejected_by_writer_and_reader(self):
        values = sample_payload()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                run_io.write_run(
                    root / "bad", make_manifest(schema_version="0.1"),
                    *values[1:5], values[5], dataset="synthetic",
                    version="bad")

            run_dir = root / "run"
            write_sample(run_dir)
            document = msgspec.json.decode(
                (run_dir / run_io.RUN_MANIFEST_NAME).read_bytes())
            document["schema_version"] = "0.1"
            replace_payload(
                run_dir, run_io.RUN_MANIFEST_NAME,
                msgspec.json.encode(document))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                run_io.read_run(run_dir)

    def test_health_records_survive_association_payload(self):
        record = structs.HealthRecord(
            keyframe_idx=3, ess=42.0, resampled=True, mean_east_m=1.0,
            mean_north_m=2.0, mean_heading_deg=3.0, position_std_m=4.0,
            map_east_m=1.0, map_north_m=2.0, map_heading_deg=3.0,
            heading_std_deg=1.5, n_measurements=1,
            associations=[structs.AssociationPosterior(
                tracklet_id="trk_a", anchor_keyframe_idx=3, null_share=0.1,
                responsibilities={"lm_a": 0.9})])
        encoded = msgspec.json.encode(record)
        decoded = msgspec.json.decode(encoded, type=structs.HealthRecord)
        self.assertEqual(decoded, record)


class StrictReaderTest(unittest.TestCase):
    def _mutated_run(self, root: Path, relative: str, mutate) -> Path:
        run_dir = root / "run"
        write_sample(run_dir)
        payload = (run_dir / relative).read_bytes()
        replace_payload(run_dir, relative, mutate(payload))
        return run_dir

    def test_jsonl_rejects_duplicate_nonfinite_unknown_and_blank_records(self):
        mutations = {
            "duplicate": lambda raw: raw.replace(
                b'{"keyframe_idx":0,',
                b'{"keyframe_idx":0,"keyframe_idx":0,', 1),
            "nonfinite": lambda raw: raw.replace(b'"ess":8.0',
                                                   b'"ess":NaN', 1),
            "unknown": lambda raw: raw.replace(
                b'{"keyframe_idx":0,',
                b'{"keyframe_idx":0,"mystery":1,', 1),
            "blank": lambda raw: raw.replace(b"\n", b"\n\n", 1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                run_dir = self._mutated_run(
                    Path(tmp), "tier0_health.jsonl", mutate)
                with self.assertRaises(ValueError):
                    run_io.read_run(run_dir)

    def test_typed_json_rejects_unknown_and_duplicate_fields(self):
        for name, mutate in (
                ("unknown", lambda raw: raw.replace(
                    b'{"schema_version":',
                    b'{"mystery":1,"schema_version":', 1)),
                ("duplicate", lambda raw: raw.replace(
                    b'{"schema_version":',
                    b'{"dataset":"wrong","schema_version":', 1))):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                run_dir = self._mutated_run(
                    Path(tmp), run_io.RUN_MANIFEST_NAME, mutate)
                if name == "duplicate":
                    raw = (run_dir / run_io.RUN_MANIFEST_NAME).read_bytes()
                    raw = raw.replace(
                        b'"dataset":"synthetic"',
                        b'"dataset":"synthetic","dataset":"again"', 1)
                    replace_payload(run_dir, run_io.RUN_MANIFEST_NAME, raw)
                with self.assertRaises(ValueError):
                    run_io.read_run(run_dir)

    def test_retired_noop_experiment_fields_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            manifest, *_ = write_sample(run_dir)
            stamp_retired_experiment_fields(run_dir)

            loaded = run_io.read_run(run_dir)

        self.assertEqual(loaded.manifest, manifest)

    def test_measurements_recorded_before_the_range_cap_read_as_uncapped(self):
        # An `X | None = None` field the record predates reads as None, once
        # warned about per file; no per-field table is involved.
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_sample(run_dir)
            path = run_dir / "tier1_measurements.jsonl"
            stripped = []
            for line in path.read_text().splitlines():
                record = msgspec.json.decode(line)
                record.pop("range_max_m")
                stripped.append(msgspec.json.encode(record).decode())
            replace_payload(run_dir, "tier1_measurements.jsonl",
                            ("\n".join(stripped) + "\n").encode())

            with self.assertWarnsRegex(RuntimeWarning, "range_max_m") as cm:
                loaded = run_io.read_run(run_dir)

        self.assertTrue(loaded.measurements)
        self.assertTrue(all(m.range_max_m is None for m in loaded.measurements))
        self.assertEqual(len(cm.warnings), 1)

    def test_runs_recorded_before_the_range_cap_read_as_uncapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            manifest, *_ = write_sample(run_dir)
            run_manifest_path = run_dir / run_io.RUN_MANIFEST_NAME
            run_manifest = msgspec.json.decode(run_manifest_path.read_bytes())
            run_manifest["filter_config"].pop("range_cap")
            artifact.atomic_write_json(run_manifest_path, run_manifest)
            outer = artifact.load_manifest(run_dir)
            config = dict(outer.config)
            contract = dict(config[run_io.RUN_CONTRACT_CONFIG_KEY])
            contract["filter_config"].pop("range_cap")
            config[run_io.RUN_CONTRACT_CONFIG_KEY] = contract
            artifact.atomic_write_json(
                run_dir / artifact.MANIFEST_NAME,
                dataclasses.replace(outer, config=config).to_dict())
            repair_content_digest(run_dir)

            with self.assertWarnsRegex(RuntimeWarning, "range_cap"):
                loaded = run_io.read_run(run_dir)

        self.assertIsNone(loaded.manifest.filter_config.range_cap)
        self.assertEqual(
            msgspec.structs.replace(loaded.manifest.filter_config,
                                    range_cap=manifest.filter_config.range_cap),
            manifest.filter_config)

    def test_missing_field_without_a_none_default_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_sample(run_dir)
            run_manifest_path = run_dir / run_io.RUN_MANIFEST_NAME
            run_manifest = msgspec.json.decode(run_manifest_path.read_bytes())
            run_manifest["filter_config"].pop("matcher_recall")
            artifact.atomic_write_json(run_manifest_path, run_manifest)
            repair_content_digest(run_dir)

            with self.assertRaisesRegex(ValueError, "missing fields"):
                run_io.read_run(run_dir)

    def test_retired_experiment_fields_reject_active_values(self):
        cases = (
            {"measurement_damage_cap_nats": 2.0},
            {"revival_enabled": True},
            {"revival_enabled": 0},
            {"revival_margin_nats": 1.0},
            {"revival_match_radius_m": 25.0},
        )
        for fields in cases:
            with self.subTest(fields=fields), \
                    tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "run"
                write_sample(run_dir)
                stamp_retired_experiment_fields(run_dir, **fields)

                with self.assertRaisesRegex(ValueError, "retired non-noop"):
                    run_io.read_run(run_dir)

    def test_typed_json_cannot_backfill_defaulted_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_sample(run_dir)
            document = msgspec.json.decode(
                (run_dir / run_io.RUN_MANIFEST_NAME).read_bytes())
            del document["filter_config"]["matcher_recall"]
            replace_payload(
                run_dir, run_io.RUN_MANIFEST_NAME,
                msgspec.json.encode(document))
            with self.assertRaisesRegex(ValueError, "missing fields"):
                run_io.read_run(run_dir)

    def test_required_health_cannot_silently_decode_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._mutated_run(
                Path(tmp), "tier0_health.jsonl", lambda _: b"")
            with self.assertRaisesRegex(ValueError, "health keyframes"):
                run_io.read_run(run_dir)

    def test_cross_file_counts_and_checkpoint_index_must_agree(self):
        mutations = (
            ("tier0_health.jsonl", lambda raw: raw.replace(
                b'"n_measurements":1', b'"n_measurements":0', 1)),
            ("checkpoints/index.json", lambda _: b"[0]"),
        )
        for relative, mutate in mutations:
            with self.subTest(relative=relative), \
                    tempfile.TemporaryDirectory() as tmp:
                run_dir = self._mutated_run(Path(tmp), relative, mutate)
                with self.assertRaises(ValueError):
                    run_io.read_run(run_dir)

    def test_checkpoint_arrays_are_exact_and_have_particle_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_sample(run_dir)
            checkpoint = run_dir / "checkpoints/kf_00000.npz"
            with np.load(checkpoint, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            arrays["east_m"] = arrays["east_m"][:-1]
            np.savez(checkpoint, **arrays)
            repair_content_digest(run_dir)
            with self.assertRaisesRegex(ValueError, "shape"):
                run_io.read_run(run_dir)

    def test_artifact_tampering_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "tampered"
            write_sample(run_dir)
            artifact.atomic_write_file(run_dir / "tier0_health.jsonl", b"")
            with self.assertRaises(artifact.ArtifactValidationError):
                run_io.read_run(run_dir)

            linked = root / "linked"
            write_sample(linked)
            target = linked / "tier0_health.jsonl"
            target.unlink()
            target.symlink_to(linked / "truth.jsonl")
            with self.assertRaises(artifact.ArtifactValidationError):
                run_io.read_run(linked)

    def test_artifact_config_and_declared_outputs_are_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, mutate in (
                    ("config", lambda outer: dataclasses.replace(
                        outer, config={"run_kind": "synthetic",
                                       "localization_inputs_manifest_sha256":
                                       None})),
                    ("outputs", lambda outer: dataclasses.replace(
                        outer, declared_outputs=tuple(
                            value for value in outer.declared_outputs
                            if value != "tier0_health.jsonl")))):
                run_dir = root / name
                write_sample(run_dir)
                outer = mutate(artifact.load_manifest(run_dir))
                artifact.atomic_write_json(
                    run_dir / artifact.MANIFEST_NAME, outer.to_dict())
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run_io.read_run(run_dir)


if __name__ == "__main__":
    unittest.main()
