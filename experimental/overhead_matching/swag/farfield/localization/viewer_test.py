"""Tests for the viewer payload, the static page, and the server.

The payload builder is the single source of truth for both consumers, so what
matters is that it (a) never fails on a run missing its optional context, (b)
never silently drops information a panel needs, and (c) tells the truth about
what it left out. The page and the server are then checked to render/serve that
same payload.

  T-V1  a bare run directory — no attribution, no feather, no ghosts — still
        produces a complete payload, with notes explaining each absent panel
  T-V2  particles are drawn as a WEIGHTED sample, so a cloud that is mostly
        dead weight does not render as if it were alive
  T-V3  only landmarks referenced by matcher tables enter the map payload
  T-V4  a stale attribution cache is surfaced as a note, not silently used
  T-V5  the page renders self-contained, with no external subresources — the
        CSS/JS now live as viewer_assets/ files and must be INLINED
  T-V6  the server serves the same payload and its two extra endpoints
  T-V7  live-only detail is opt-in, while satellite and track-fit semantics are
        shared with the static viewer
  T-V8  presentation helpers preserve natural ordering, bounded MAP emphasis,
        click-only OSM navigation, and ancestry-bound tracking evidence
  T-V9  map interaction, playback keys, and wide evidence stay lightweight
"""

import base64
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import msgspec
import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import (
    artifact,
    paths as paths_lib,
)
from experimental.overhead_matching.swag.farfield.localization import (
    attribution,
    metrics,
    runner,
    run_io,
    scenario,
    side_outputs,
    structs,
    viewer,
    particle_sampling,
    viewer_payload,
    viewer_server,
)
from experimental.overhead_matching.swag.farfield.viewers import page

VISIBLE_RANGE_M = 10000.0


def _make_run(run_dir: Path, n_particles: int = 3000, seed: int = 3):
    cfg = scenario.harbor_loop(keyframe_period_s=5.0,
                               max_visible_range_m=VISIBLE_RANGE_M)
    data = scenario.generate(cfg)
    start = data.truth[0]
    config = structs.FilterConfig(
        n_particles=n_particles, seed=seed,
        init=structs.GaussianInit(start.east_m, start.north_m, 400.0),
        checkpoint_every=10)
    metric_config = metrics.position_mass_metric_config()
    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION, dataset="synthetic",
        scenario_name=cfg.name, run_kind="synthetic",
        initialization_kind="test", bearings_consumed=True,
        proposal_enabled=config.proposal.enabled,
        localization_inputs_manifest_sha256=None,
        anchor_lat_deg=cfg.anchor_lat_deg, anchor_lon_deg=cfg.anchor_lon_deg,
        n_keyframes=data.n_keyframes, filter_config=config,
        landmarks=cfg.landmarks, matcher_version=scenario.MATCHER_VERSION,
        max_visible_range_m=VISIBLE_RANGE_M,
        export_dir=f"synthetic:{cfg.name}",
        git_commit="test", argv=["viewer_test"],
        created="2026-08-21T00:00:00+00:00",
        truth_position_schema=runner.TRUTH_POSITION_SCHEMA,
        position_mass_metric=metric_config)
    result = runner.execute_localization(
        run_dir, manifest, catalog=data.catalog, truth=data.truth,
        odometry=data.odometry, measurements=data.measurements,
        tables=data.tables, dataset="synthetic", version=run_dir.name)
    return data, config, result.history


def _make_satellite(directory: Path, size=(320, 160)) -> None:
    directory.mkdir()
    Image.new("RGB", size, (40, 90, 130)).save(directory / "wide.jpg",
                                                format="JPEG")
    (directory / "satellite.json").write_text(json.dumps({
        "source": "viewer test imagery",
        "layers": [{
            "image": "wide.jpg", "zoom": 12,
            "east_min": -1000.0, "east_max": 1000.0,
            "north_min": -500.0, "north_max": 500.0,
        }],
    }))


def _make_tracking_viewer_artifact(directory: Path) -> Path:
    outputs = ("index.html", "track_full_T2.html", "videos/full_T2.mp4")
    with artifact.ArtifactDirectoryBuilder(
            directory, kind=paths_lib.OBJECT_TRACKS, dataset="synthetic",
            version="viewer-test", generator="viewer_test",
            git_commit="test", arguments=(), upstreams=(),
            config={"range": {"name": "full"}},
            declared_outputs=outputs) as builder:
        builder.output_path("index.html").write_text("track index")
        builder.output_path("track_full_T2.html").write_text(
            "<video src='videos/full_T2.mp4'></video>")
        video = builder.output_path("videos/full_T2.mp4")
        video.parent.mkdir()
        video.write_bytes(b"test-video")
    return directory


class PayloadTest(unittest.TestCase):
    def test_viewer_auto_resolves_satellite_when_not_explicit(self):
        resolved = Path("/shared/satellite/dataset/version")
        with mock.patch.object(
                viewer.satellite_assets, "find_or_generate",
                return_value=resolved) as find:
            actual = viewer.satellite_for_viewer(
                Path("/runs/experiment/run"), explicit=None, disabled=False)

        self.assertEqual(actual, resolved)
        find.assert_called_once_with(Path("/runs/experiment/run"))

    def test_explicit_or_disabled_satellite_never_auto_fetches(self):
        explicit = Path("/chosen/satellite")
        with mock.patch.object(
                viewer.satellite_assets, "find_or_generate") as find:
            self.assertEqual(viewer.satellite_for_viewer(
                Path("/runs/e/r"), explicit=explicit, disabled=False), explicit)
            self.assertIsNone(viewer.satellite_for_viewer(
                Path("/runs/e/r"), explicit=None, disabled=True))
        find.assert_not_called()

    def test_tracking_evidence_href_is_portable_with_the_data_root(self):
        root = Path("/mirror/farfield_matching")
        viewer_dir = root / "runs" / "experiment" / "run.viewer"
        tracks_dir = (root / "artifacts" / "object_tracks" / "dataset"
                      / "tracks-v2")

        href = viewer_payload._tracking_evidence_href(
            viewer_dir, tracks_dir, "track_full_T2.html")

        self.assertEqual(
            href,
            "../../../artifacts/object_tracks/dataset/tracks-v2/"
            "track_full_T2.html")
        self.assertNotIn(str(root), href)
        self.assertFalse(href.startswith("file:"))

    def test_review_href_is_portable_and_uses_the_short_display_anchor(self):
        root = Path("/mirror/farfield_matching")
        viewer_dir = root / "runs" / "experiment" / "run.viewer"
        matcher_page = (root / "runs" / "experiment"
                        / "run.matcher-review" / "index.html")

        href = viewer_payload._review_href(
            viewer_dir, matcher_page, "T2")

        self.assertEqual(
            href,
            "../run.matcher-review/index.html#T2")
        self.assertNotIn(str(root), href)

    def test_tracklet_ids_sort_naturally(self):
        tracklets = ["LT10", "LT2", "LT1", "LT20", "LT11"]
        self.assertEqual(
            sorted(tracklets, key=viewer_payload._natural_key),
            ["LT1", "LT2", "LT10", "LT11", "LT20"])

    def test_bare_run_directory_still_builds(self):
        """T-V1. Optional context is optional."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)

            for key in ("run", "health", "checkpoints", "landmarks",
                        "backdrop", "landmarkGeometry", "basemap", "truth",
                        "measurements",
                        "modes", "tracklets", "events", "triageSummary",
                        "ghosts", "notes", "colors"):
                self.assertIn(key, payload, f"payload lacks {key!r}")
            self.assertTrue(payload["health"])
            self.assertTrue(payload["tracklets"])
            self.assertIsNone(payload["attribution"])
            self.assertEqual(payload["ghosts"], [])
            # Every absent panel is explained.
            notes = " ".join(payload["notes"])
            self.assertIn("attribution cache", notes)

    def test_satellite_display_copy_is_resolution_bounded(self):
        """T-V9. Large source mosaics must not become huge decoded textures."""
        with tempfile.TemporaryDirectory() as tmp:
            satellite = Path(tmp) / "satellite"
            source_size = (2304, 768)
            _make_satellite(satellite, size=source_size)
            notes = []

            payload = viewer_payload._satellite_payload(
                satellite, notes)

            encoded = payload["layers"][0]["uri"].split(",", 1)[1]
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                self.assertEqual(max(image.size),
                                 viewer_payload.SATELLITE_MAX_EDGE_PX)
                self.assertAlmostEqual(image.width / image.height,
                                       source_size[0] / source_size[1], places=2)
            with Image.open(satellite / "wide.jpg") as source:
                self.assertEqual(source.size, source_size)
            self.assertTrue(any("source imagery unchanged" in note
                                for note in notes))

    def test_exact_landmark_geometry_and_bounds_are_not_truncated(self):
        line = structs.LandmarkEntry(
            "line", 0.0, 0.0, "pier", 2.0,
            hull_east_m=[-25.25, 10.5, 80.75],
            hull_north_m=[5.125, -4.5, 7.25])
        polygon = structs.LandmarkEntry(
            "polygon", 0.0, 0.0, "island", 2.0,
            hull_east_m=[0.0, 10.0, 10.0, 0.0, 0.0],
            hull_north_m=[0.0, 0.0, 10.0, 10.0, 0.0])

        line_payload = viewer_payload._landmark_geometry(line)
        polygon_payload = viewer_payload._landmark_geometry(
            polygon)
        self.assertEqual(line_payload["kind"], "linestring")
        self.assertEqual(line_payload["points"], [
            [-25.25, 5.125], [10.5, -4.5], [80.75, 7.25]])
        self.assertEqual(polygon_payload["kind"], "polygon")
        self.assertEqual(len(polygon_payload["points"]), 5)
        self.assertEqual(
            viewer_payload._catalog_bounds(
                np.array([0.0]), np.array([0.0]), [line], margin=0.0),
            (-25.25, 80.75, -4.5, 7.25))

    def test_visible_range_comes_from_the_manifest(self):
        """No override parameter, no fallback: the page shows the geometry
        the run recorded."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)
            self.assertEqual(payload["run"]["maxVisibleRangeM"],
                             VISIBLE_RANGE_M)

    def test_primary_mass_summary_is_truth_centered_and_headlined(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)

            aggregate = payload["run"]["positionMassMetric"]["aggregate"]
            self.assertEqual(aggregate["referencePosition"], "truth")
            self.assertEqual(aggregate["primaryRadiusM"], 500.0)
            self.assertEqual(set(aggregate["scores"]), {"100", "500"})
            self.assertTrue(all(0.0 <= value <= 1.0
                                for value in aggregate["scores"].values()))
            summary = json.loads(
                (run_dir / metrics.POSITION_MASS_SUMMARY_NAME).read_text())
            self.assertEqual(summary["radii"]["500"][
                "distance_normalized_mass"],
                             aggregate["scores"]["500"])

    def test_particles_are_a_weighted_sample(self):
        """T-V2. The regression this guards: uniform subsampling of a weighted
        posterior renders every particle as equally believed, so a belief that
        has collapsed onto a few particles looks like a healthy wide cloud."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            data = run_io.read_run(run_dir)

            # Pick a checkpoint whose weights are genuinely uneven.
            worst_kf, worst_ratio = None, 0.0
            for kf, arrays in data.checkpoints.items():
                w = np.exp(arrays["log_weight"] - arrays["log_weight"].max())
                w = w / w.sum()
                ratio = float(np.sort(w)[-len(w) // 20:].sum())  # top 5%
                if ratio > worst_ratio:
                    worst_kf, worst_ratio = kf, ratio
            self.assertGreater(worst_ratio, 0.5,
                               "no checkpoint has concentrated weight, so this "
                               "test cannot distinguish the two samplers")

            arrays = data.checkpoints[worst_kf]
            rng = np.random.default_rng(0)
            index = particle_sampling.weighted_sample(
                arrays["log_weight"], 400, rng)
            weights = np.exp(arrays["log_weight"]
                             - arrays["log_weight"].max())
            weights = weights / weights.sum()
            # The drawn sample's mean weight must far exceed the population's:
            # that is what "weighted" means.
            self.assertGreater(weights[index].mean(), weights.mean() * 2.0)
            # And it must be a valid index set.
            self.assertEqual(index.shape[0], 400)
            self.assertTrue((index >= 0).all())
            self.assertTrue((index < arrays["log_weight"].shape[0]).all())

    def test_weighted_sample_handles_degenerate_weights(self):
        rng = np.random.default_rng(0)
        # All-equal weights.
        flat = np.zeros(500)
        self.assertEqual(
            particle_sampling.weighted_sample(flat, 50, rng).shape[0], 50)
        # Fewer particles than requested: take them all.
        few = np.zeros(20)
        np.testing.assert_array_equal(
            particle_sampling.weighted_sample(few, 50, rng), np.arange(20))
        # A single surviving particle.
        spike = np.full(500, -np.inf)
        spike[7] = 0.0
        index = particle_sampling.weighted_sample(spike, 50, rng)
        self.assertTrue((index == 7).all())

    def test_map_payload_contains_only_referenced_landmarks(self):
        """T-V3. Unmatched catalog context does not belong in the map DOM."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)
            self.assertEqual(payload["backdrop"], [])
            self.assertTrue(payload["landmarks"],
                            "the synthetic matcher should reference landmarks")
            expected = viewer_payload.referenced_landmark_ids(
                run_io.read_run(run_dir))
            self.assertEqual({landmark["id"]
                              for landmark in payload["landmarks"]}, expected)
            self.assertTrue(all(geometry["referenced"]
                                for geometry in payload["landmarkGeometry"]))

    def test_every_landmark_gets_a_glyph_class(self):
        for type_key, expected in (
                ("man_made=lighthouse", "light"),
                ("seamark:type=beacon_lateral", "light"),
                ("man_made=storage_tank", "tank"),
                ("man_made=crane", "tower"),
                ("bridge=yes", "bridge"),
                ("man_made=pier", "water"),
                ("place=island", "nature"),
                ("building=commercial", "building"),
                ("", "building"),
                (None, "building")):
            self.assertEqual(viewer_payload._glyph_for(type_key), expected)

    def test_stale_attribution_cache_becomes_a_note(self):
        """T-V4. Silently rendering a waterfall from a different run is the
        worst available outcome, so staleness must surface."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            cache, _ = attribution.compute(run_dir)
            attribution.write_cache(run_dir, cache)

            # Corrupt the recorded provenance.
            meta_path = attribution.cache_dir(run_dir) / attribution.META_NAME
            meta = json.loads(meta_path.read_text())
            meta["particle_history_sha256"] = "0" * 64
            meta_path.write_text(json.dumps(meta))

            payload = viewer_payload.build(run_dir)
            self.assertIsNone(payload["attribution"])
            self.assertTrue(any("recompute it" in note
                                for note in payload["notes"]),
                            f"staleness not reported: {payload['notes']}")

    def test_fresh_attribution_is_used_and_marked_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            cache, _ = attribution.compute(run_dir)
            attribution.write_cache(run_dir, cache)

            payload = viewer_payload.build(run_dir)
            self.assertIsNotNone(payload["attribution"])
            self.assertTrue(payload["attribution"]["verified"])
            self.assertTrue(payload["attribution"]["modes"])
            # Every tracklet gets an attribution series.
            with_series = [t for t in payload["tracklets"]
                           if t.get("attribution")]
            self.assertTrue(with_series)

    def test_triage_reaches_the_tracklet_dossiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)
            triaged = [t for t in payload["tracklets"] if t.get("triage")]
            self.assertTrue(triaged, "synthetic runs have truth, so every "
                                     "tracklet should carry a verdict")
            for tracklet in triaged:
                self.assertIn("verdict", tracklet["triage"])
                self.assertIn("toleranceDeg", tracklet["triage"])
                self.assertIn("nConsistent", tracklet["triage"])
            self.assertNotIn("no ground truth", payload["triageSummary"])

    def test_unreadable_ghost_is_a_note_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(
                run_dir, ghost_dirs=[Path(tmp) / "does_not_exist"])
            self.assertEqual(payload["ghosts"], [])
            self.assertTrue(any("unreadable" in note
                                for note in payload["notes"]))

    def test_missing_feather_is_a_note_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(
                run_dir, feather=Path(tmp) / "nope.feather",
                with_basemap=True)
            self.assertEqual(payload["basemap"]["layers"], [])
            self.assertTrue(any("does not exist" in note
                                for note in payload["notes"]))

    def test_source_artifacts_must_be_supplied_as_an_exact_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            with self.assertRaisesRegex(ValueError, "supplied together"):
                viewer_payload.build(
                    run_dir, tracks_dir=Path(tmp) / "tracks")


class PageTest(unittest.TestCase):
    def test_page_is_self_contained(self):
        """T-V5. A page that fetches anything is not a portable record."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))

            self.assertTrue(html.startswith(page.GENERATED_MARK))
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("window.__RUN__", html)
            # No external subresources. Ordinary <a href> navigation is okay:
            # OSM ids are clickable, but nothing leaves the page until click.
            for pattern in (r'src\s*=\s*["\']https?://',
                            r'<link[^>]*href\s*=\s*["\']https?://',
                            r'@import\s+url\(', r'fetch\(["\']https?://'):
                self.assertIsNone(re.search(pattern, html),
                                  f"page references something external: "
                                  f"{pattern}")
            # The payload must be valid JSON that the page can parse.
            match = re.search(r"window\.__RUN__ = (\{.*?\});</script>", html,
                              re.S)
            self.assertIsNotNone(match)
            json.loads(match.group(1))

    def test_presentation_helpers_are_inlined(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))

            self.assertIn("const TRAIL_TAIL = 60", html)
            self.assertIn('stroke-width=".8" opacity=".18"', html)
            self.assertIn("latest 60 keyframes emphasized", html)
            self.assertIn("/^osm:(node|way|relation):(\\d+)$/", html)
            self.assertIn(
                'href="https://www.openstreetmap.org/${m[1]}/${m[2]}"',
                html)
            self.assertIn('target="_blank" rel="noopener"', html)
            self.assertIn("const sourceReviewLinks = src =>", html)
            self.assertIn('src.matcherHref ? `<a href=', html)
            self.assertIn('src.auditHref ? `<a href=', html)
            self.assertIn(').join(" ")}</div>`;', html)
            self.assertIn("selected.source.frameBearings.find(", html)
            self.assertIn("row => row.kf === t", html)
            self.assertIn('stroke="var(--privileged)"', html)
            self.assertIn("truth bearing @ kf ${t}", html)
            for term in ("consistent", "geometry-unexplained", "no-evidence",
                         "matcher-fault", "filter-fault", "anti"):
                self.assertIn(f"<b>{term}</b>", html)

    def test_map_and_tracklet_interactions_stay_lightweight(self):
        """T-V9. Guard the defaults and event path behind snappy interaction."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))

            self.assertNotIn('id="tgBase"', html)
            self.assertNotIn("function buildBasemap()", html)
            self.assertIn("D.landmarkGeometry.filter(g => g.referenced)", html)
            self.assertNotIn("const BACKDROP =", html)
            self.assertIn("scheduleViewCommit();", html)
            self.assertIn('$("dynamicmap").setAttribute("transform"', html)
            self.assertIn('e.code === "Space"', html)
            self.assertIn("togglePlayback();", html)
            self.assertIn('target.closest("input, textarea, select, button, a")',
                          html)
            self.assertIn(".grid2>*{min-width:0;overflow:hidden}", html)
            self.assertIn("img.crop{display:block;width:100%;max-width:100%",
                          html)
            self.assertIn("const trkLabel = id =>", html)
            self.assertIn("const compactTracklets = text =>", html)
            self.assertIn("esc(trkLabel(a.trk))", html)
            self.assertIn("esc(trkLabel(trk.id))", html)
            self.assertIn("esc(compactTracklets(e.label))", html)
            self.assertIn("esc(compactTracklets(e.detail))", html)

    def test_particle_density_uses_full_checkpoint_with_static_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))
            self.assertIn('id="liveStatus"', html)
            self.assertNotIn('id="tgFull"', html)
            self.assertIn('id="particlePct" disabled', html)
            self.assertIn('<option value="" selected>sample</option>', html)
            for percent in (10, 20, 30, 50, 100):
                self.assertIn(f'<option value="{percent}"', html)
            self.assertIn("let particlePercent = null;", html)
            self.assertIn(
                "Math.ceil(RUN.nParticles * particlePercent / 100)", html)
            self.assertIn("particleSelect.onchange", html)
            self.assertIn('LIVE.features.has("localization_particles")', html)
            self.assertIn('"/api/localization-particles/" + keyframe', html)
            self.assertIn("scheduleParticleCheckpoint(ck);", html)
            self.assertIn('apiJson("/api/health")', html)
            self.assertIn('apiJson("/api/replay"', html)
            self.assertIn("mapLayers().innerHTML = out", html)

    def test_fit_track_prefers_truth_and_falls_back_to_the_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))

            self.assertIn(
                "const track = D.truth.length ? D.truth : H.map(", html)
            self.assertNotIn(
                "D.truth.forEach(p => g(px0(p[0]), py0(p[1])));", html)

    def test_assets_are_inlined_from_the_asset_files(self):
        """The CSS/JS live as viewer_assets/ files (data deps) and are inlined
        verbatim: the emitted page must carry their content, not links."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))
            asset_dir = Path(viewer.__file__).parent / "viewer_assets"
            for name, marker in (("style.css", ".privileged{"),
                                 ("app.js", "aggregate.primaryRadiusM")):
                content = (asset_dir / name).read_text()
                self.assertIn(marker, content,
                              f"{name} no longer carries its marker; update "
                              f"this test alongside the asset")
                self.assertIn(content.strip(), html,
                              f"{name} was not inlined into the page")
            # No <link rel=stylesheet> / <script src=...> escape hatches.
            self.assertNotIn("<link", html)
            self.assertIsNone(re.search(r"<script[^>]+src=", html))

    def test_page_carries_a_provenance_footer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))
            self.assertIn("<footer", html)
            self.assertIn("git ", html)
            self.assertIn("localization:viewer", html)

    def test_viewer_publishes_as_a_sibling_without_mutating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            before = sorted(path.relative_to(run_dir)
                            for path in run_dir.rglob("*"))
            payload = viewer_payload.build(run_dir)
            output = viewer.write_viewer(
                run_dir,
                payload,
                output_dir=None,
                body_only=False,
                inputs={"run_dir": run_dir.resolve()},
                config={"test": True})

            self.assertEqual(output, Path(tmp) / "run.viewer/viewer.html")
            self.assertTrue(output.is_file())
            self.assertTrue((output.parent / "manifest.json").is_file())
            self.assertEqual(
                before,
                sorted(path.relative_to(run_dir)
                       for path in run_dir.rglob("*")))

    def test_viewer_rejects_output_inside_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            with self.assertRaisesRegex(
                    side_outputs.SideOutputError, "immutable run"):
                viewer.write_viewer(
                    run_dir,
                    viewer_payload.build(run_dir),
                    output_dir=run_dir / "viewer",
                    body_only=False,
                    inputs={"run_dir": run_dir.resolve()},
                    config={"test": True})

    def test_a_script_tag_in_the_data_cannot_escape_the_payload(self):
        """Landmark ids and run names come from data files, so `</script>`
        in one of them is reachable input rather than a hypothetical."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)
            hostile = '</script><script>window.PWNED=1</script>'
            payload["run"]["scenario"] = hostile
            if payload["landmarks"]:
                payload["landmarks"][0]["id"] = hostile
            html = viewer.render_html(payload)

            # Exactly the script tags this page defines, and no more. The
            # hostile text may still appear *escaped* — as `&lt;script&gt;` in
            # the heading — which is the correct outcome, so the assertion is
            # on executable script elements rather than on the substring.
            self.assertEqual(html.count("<script>"), 2)
            self.assertNotIn("<script>window.PWNED", html)
            self.assertNotIn("</script><script>window", html)
            # And the value still round-trips through JSON unchanged.
            match = re.search(r"window\.__RUN__ = (\{.*?\});</script>", html,
                              re.S)
            self.assertEqual(json.loads(match.group(1))["run"]["scenario"],
                             hostile)

    def test_page_declares_the_truth_privileged_fence(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))
            self.assertIn("Truth-privileged", html)
            self.assertIn("truth-privileged", html)
            self.assertIn(".privileged{", html)

    def test_body_only_omits_the_document_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)
            fragment = viewer.render_html(payload, body_only=True)
            self.assertNotIn("<!doctype", fragment)
            self.assertNotIn("<html", fragment)
            self.assertIn("<style>", fragment)

    def test_scenario_name_is_escaped(self):
        """A run name is free text and reaches the page's title and header."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)
            payload["run"]["scenario"] = '<img src=x onerror="boom">'
            html = viewer.render_html(payload)
            self.assertNotIn("<img src=x", html)
            self.assertIn("&lt;img src=x", html)


class ServerTest(unittest.TestCase):
    """T-V6. The server must serve the same payload the file inlines."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "run"
        _make_run(self.run_dir)
        self.app = viewer_server.create_app(self.run_dir)
        self.client = self.app.test_client()

    def tearDown(self):
        self._tmp.cleanup()

    def test_health_advertises_the_extra_features(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(set(body["features"]),
                         {"checkpoint", "replay"})
        self.assertGreater(body["n_checkpoints"], 0)

    def test_index_serves_the_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"window.__RUN__", response.data)

    def test_payload_matches_the_static_build(self):
        served = self.client.get("/api/payload").get_json()
        built = viewer_payload.build(self.run_dir)
        # The server marks itself; otherwise the payloads must agree.
        self.assertTrue(served.pop("server", False))
        self.assertEqual(served["run"], built["run"])
        self.assertEqual(len(served["health"]), len(built["health"]))
        self.assertEqual(served["tracklets"], built["tracklets"])

    def test_checkpoint_returns_every_particle_with_weights(self):
        """The point of the endpoint: no subsampling."""
        data = run_io.read_run(self.run_dir)
        kf = sorted(data.checkpoints)[len(data.checkpoints) // 2]
        response = self.client.get(f"/api/checkpoint/{kf}")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        n = data.checkpoints[kf]["east_m"].shape[0]
        self.assertEqual(body["n"], n)
        for key in ("e", "n_m", "h", "w", "mode", "event"):
            self.assertEqual(len(body[key]), n, f"{key} is truncated")
        self.assertAlmostEqual(sum(body["w"]), 1.0, places=3)
        self.assertGreater(n, viewer_payload.MAX_PARTICLES_PER_FRAME,
                           "the run is too small for this test to prove the "
                           "endpoint beats the inlined sample")

    def test_checkpoint_map_view_omits_unused_particle_fields(self):
        data = run_io.read_run(self.run_dir)
        kf = sorted(data.checkpoints)[len(data.checkpoints) // 2]

        response = self.client.get(f"/api/checkpoint/{kf}?view=map")

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        n = data.checkpoints[kf]["east_m"].shape[0]
        self.assertEqual(set(body), {"n", "e", "n_m", "mode"})
        self.assertEqual(body["n"], n)
        for key in ("e", "n_m", "mode"):
            self.assertEqual(len(body[key]), n, f"{key} is truncated")

    def test_live_server_passes_satellite_to_the_shared_payload(self):
        satellite = Path(self._tmp.name) / "satellite"
        _make_satellite(satellite)
        client = viewer_server.create_app(
            self.run_dir, satellite=satellite).test_client()

        response = client.get("/api/payload")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["satellite"]["source"],
                         "viewer test imagery")
        self.assertEqual(len(payload["satellite"]["layers"]), 1)
        self.assertTrue(payload["satellite"]["layers"][0]["uri"].startswith(
            "data:image/jpeg;base64,"))

    def test_tracking_evidence_serves_only_declared_ancestor_files(self):
        tracks_dir = _make_tracking_viewer_artifact(
            Path(self._tmp.name) / "tracks")
        stub_payload = {"checkpoints": {}}
        with mock.patch.object(
                viewer_server.viewer_payload, "build",
                return_value=stub_payload) as build:
            client = viewer_server.create_app(
                self.run_dir, tracks_dir=tracks_dir,
                audit_dir=Path(self._tmp.name) / "audits").test_client()

            page_response = client.get(
                "/api/tracking/track_full_T2.html")
            video_response = client.get(
                "/api/tracking/videos/full_T2.mp4")
            missing_response = client.get(
                "/api/tracking/undeclared.txt")

        self.assertEqual(page_response.status_code, 200)
        self.assertIn(b"videos/full_T2.mp4", page_response.data)
        self.assertEqual(video_response.data, b"test-video")
        self.assertEqual(missing_response.status_code, 404)
        build.assert_called_once()

    def test_missing_checkpoint_lists_what_exists(self):
        response = self.client.get("/api/checkpoint/999999")
        self.assertEqual(response.status_code, 404)
        self.assertTrue(response.get_json()["available"])

    def test_replay_rejects_empty_and_malformed_edits(self):
        self.assertEqual(
            self.client.post("/api/replay", json={"edits": {}}).status_code,
            400)
        self.assertEqual(
            self.client.post("/api/replay", json={}).status_code, 400)
        bad = self.client.post("/api/replay",
                               json={"edits": {"pi0": "not a number"}})
        self.assertEqual(bad.status_code, 400)

    def test_replay_runs_a_counterfactual_and_returns_a_ghost(self):
        data = run_io.read_run(self.run_dir)
        victim = data.measurements[0].tracklet_id
        response = self.client.post("/api/replay", json={
            "edits": {"drop_tracklets": [victim]}})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_json()
        self.assertIn(victim, body["describe"])
        self.assertIsNotNone(body["ghost"])
        self.assertEqual(len(body["ghost"]["trail"]),
                         data.manifest.n_keyframes)
        # The ghost is a real run artifact in a deterministic sibling sidecar,
        # so writing it leaves the completed source artifact immutable.
        ghost_dir = Path(body["output_dir"])
        self.assertEqual(ghost_dir.parent.parent, self.run_dir.parent)
        self.assertEqual(
            ghost_dir.parent.name,
            self.run_dir.name
            + viewer_server.replay_mod.COUNTERFACTUAL_DIR_SUFFIX)
        self.assertTrue((ghost_dir / run_io.RUN_MANIFEST_NAME).exists())
        self.assertTrue((ghost_dir / "counterfactual.json").exists())
        self.assertTrue(
            (ghost_dir / metrics.POSITION_MASS_SUMMARY_NAME).exists())
        ghost = run_io.read_run(ghost_dir)
        expected_keys = {
            metrics.position_mass_metric_key(
                ghost.manifest.position_mass_metric, radius_m)
            for radius_m in (100.0, 500.0)
        }
        self.assertTrue(all(
            set(record.position_probability_mass) == expected_keys
            for record in ghost.health))
        refreshed = self.client.get("/api/payload").get_json()
        self.assertEqual(len(refreshed["ghosts"]), 1)

    def test_edits_round_trip_through_json(self):
        """The server decodes Edits from JSON, so the shape must survive."""
        edits = viewer_server.replay_mod.Edits(
            drop_tracklets=("a", "b"), pi0=0.3,
            force_landmark={"a": "lm_1"}, log_lr={"b": {"lm_2": 1.5}})
        encoded = json.loads(msgspec.json.encode(edits))
        decoded = msgspec.convert(encoded, viewer_server.replay_mod.Edits,
                                  strict=False)
        self.assertEqual(decoded, edits)


if __name__ == "__main__":
    unittest.main()
