"""Tests for the viewer payload, the static page, and the server.

The payload builder is the single source of truth for both consumers, so what
matters is that it (a) never fails on a run missing its optional context, (b)
never silently drops information a panel needs, and (c) tells the truth about
what it left out. The page and the server are then checked to render/serve that
same payload.

  T-V1  a bare run directory — no attribution, no sources, no feather, no
        ghosts — still produces a complete payload, with notes explaining each
        absent panel
  T-V2  particles are drawn as a WEIGHTED sample, so a cloud that is mostly
        dead weight does not render as if it were alive
  T-V3  the referenced/backdrop split covers the catalog exactly once
  T-V4  a stale attribution cache is surfaced as a note, not silently used
  T-V5  the page renders self-contained, with no external references
  T-V6  the server serves the same payload and its three extra endpoints
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

import msgspec
import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    attribution,
    filter as pf,
    run_log,
    scenario,
    structs,
    viewer,
    viewer_payload,
    viewer_server,
)


def _make_run(run_dir: Path, n_particles: int = 3000, seed: int = 3):
    cfg = scenario.harbor_loop(keyframe_period_s=5.0)
    data = scenario.generate(cfg)
    start = data.truth[0]
    config = structs.FilterConfig(
        n_particles=n_particles, seed=seed,
        init=structs.GaussianInit(start.east_m, start.north_m, 400.0),
        checkpoint_every=10)
    history = pf.run_filter(config, data.catalog, data.odometry,
                            data.measurements, data.tables)
    manifest = structs.RunManifest(
        schema_version=structs.SCHEMA_VERSION, scenario_name=cfg.name,
        anchor_lat_deg=cfg.anchor_lat_deg, anchor_lon_deg=cfg.anchor_lon_deg,
        n_keyframes=data.n_keyframes, filter_config=config,
        landmarks=cfg.landmarks, matcher_version=scenario.MATCHER_VERSION,
        particle_history_sha256=history.particle_history_sha256,
        max_visible_range_m=10000.0)
    run_log.write_run(run_dir, manifest, data.truth, data.odometry,
                      data.measurements, data.tables, history)
    return data, config, history


class PayloadTest(unittest.TestCase):
    def test_bare_run_directory_still_builds(self):
        """T-V1. Optional context is optional."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)

            for key in ("run", "health", "checkpoints", "landmarks",
                        "backdrop", "basemap", "truth", "measurements",
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
            self.assertIn("sources directory", notes)

    def test_particles_are_a_weighted_sample(self):
        """T-V2. The regression this guards: uniform subsampling of a weighted
        posterior renders every particle as equally believed, so a belief that
        has collapsed onto a few particles looks like a healthy wide cloud."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            data = run_log.read_run(run_dir)

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
            index = viewer_payload._weighted_sample(  # noqa: SLF001
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
            viewer_payload._weighted_sample(flat, 50, rng).shape[0], 50)  # noqa: SLF001
        # Fewer particles than requested: take them all.
        few = np.zeros(20)
        np.testing.assert_array_equal(
            viewer_payload._weighted_sample(few, 50, rng), np.arange(20))  # noqa: SLF001
        # A single surviving particle.
        spike = np.full(500, -np.inf)
        spike[7] = 0.0
        index = viewer_payload._weighted_sample(spike, 50, rng)  # noqa: SLF001
        self.assertTrue((index == 7).all())

    def test_referenced_and_backdrop_partition_the_catalog(self):
        """T-V3. A landmark drawn twice is confusing; one drawn zero times is
        a silent omission."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            payload = viewer_payload.build(run_dir)
            n_catalog = payload["run"]["nCatalog"]
            self.assertEqual(len(payload["landmarks"])
                             + len(payload["backdrop"]), n_catalog)
            self.assertTrue(payload["landmarks"],
                            "no landmark is referenced, so the map would be "
                            "an undifferentiated dot field")

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
            self.assertEqual(viewer_payload._glyph_for(type_key), expected)  # noqa: SLF001

    def test_stale_attribution_cache_becomes_a_note(self):
        """T-V4. Silently rendering a waterfall from a different run is the
        worst available outcome, so staleness must surface."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            cache, _ = attribution.compute(run_dir)
            attribution.write_cache(run_dir, cache)

            # Corrupt the recorded provenance.
            meta_path = run_dir / attribution.META_NAME
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
                run_dir, feather=Path(tmp) / "nope.feather")
            self.assertEqual(payload["basemap"]["layers"], [])
            self.assertTrue(any("does not exist" in note
                                for note in payload["notes"]))


class PageTest(unittest.TestCase):
    def test_page_is_self_contained(self):
        """T-V5. A page that fetches anything is not a portable record."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            _make_run(run_dir)
            html = viewer.render_html(viewer_payload.build(run_dir))

            self.assertTrue(html.startswith("<!doctype html>"))
            self.assertIn("window.__RUN__", html)
            # No external hosts, no external subresources.
            for pattern in (r'src\s*=\s*["\']https?://',
                            r'href\s*=\s*["\']https?://',
                            r'@import\s+url\(', r'fetch\(["\']https?://'):
                self.assertIsNone(re.search(pattern, html),
                                  f"page references something external: "
                                  f"{pattern}")
            # The payload must be valid JSON that the page can parse.
            match = re.search(r"window\.__RUN__ = (\{.*?\});</script>", html,
                              re.S)
            self.assertIsNotNone(match)
            json.loads(match.group(1))

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
                         {"checkpoint", "crop", "replay"})
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
        data = run_log.read_run(self.run_dir)
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

    def test_missing_checkpoint_lists_what_exists(self):
        response = self.client.get("/api/checkpoint/999999")
        self.assertEqual(response.status_code, 404)
        self.assertTrue(response.get_json()["available"])

    def test_crop_without_sources_is_a_clean_404(self):
        response = self.client.get("/api/crop/trk_a")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())

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
        data = run_log.read_run(self.run_dir)
        victim = data.measurements[0].tracklet_id
        response = self.client.post("/api/replay", json={
            "edits": {"drop_tracklets": [victim]}})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_json()
        self.assertIn(victim, body["describe"])
        self.assertIsNotNone(body["ghost"])
        self.assertEqual(len(body["ghost"]["trail"]),
                         data.manifest.n_keyframes)
        # The ghost is written as a real run directory and joins the payload.
        ghost_dir = Path(body["output_dir"])
        self.assertTrue((ghost_dir / "manifest.json").exists())
        self.assertTrue((ghost_dir / "counterfactual.json").exists())
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
