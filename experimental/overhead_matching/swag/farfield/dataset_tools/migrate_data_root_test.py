import json
import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.farfield.dataset_tools import (
    migrate_data_root as mig,
)


def build_root(tmp: Path) -> Path:
    """A miniature of the real data root's before-state."""
    root = tmp / "farfield_matching"

    # runs/: one prior-generation leg, its newer namesake, loose logs, the
    # campaign write-up.
    final = root / "runs" / "260819_final"
    (final / "mtw_leg3" / "checkpoints").mkdir(parents=True)
    (final / "mtw_leg3" / "truth.jsonl").write_text("{}\n")
    (final / "mtw_leg3.log").write_text("console\n")
    (final / "name_support.png").write_bytes(b"png")
    old = root / "runs" / "260819_mount_washington_20260815_leg3"
    (old / "checkpoints").mkdir(parents=True)
    (old / "truth.jsonl").write_text("{}\n")
    (root / "runs" / "260819_localization_campaign.md").write_text("# camp\n")

    # A localization run hiding in the artifacts lane, plus an export that
    # must NOT move, plus the campaign's absolute symlink.
    tr = (root / "artifacts" / "object_tracks" / "pohang_canal_04" / "v1"
          / "m3_tracks" / "runs" / "r002_v5_seamfix")
    for name in ("localization_run_m9_uniform",
                 "localization_run_m9_extsigma_uniform",
                 "localization_export_base"):
        (tr / name).mkdir(parents=True)
        (tr / name / "manifest.json").write_text("{}")
    (root / "runs" / "260819_final" / "pohang_canal_04").symlink_to(
        tr / "localization_run_m9_uniform")

    # A catalogs-in-datasets lane with an identical trim pair and a cache.
    lm = root / "datasets" / "boston_harbor_leg1" / "landmarks"
    (lm / "sources").mkdir(parents=True)
    (lm / "sources" / "osm.feather").write_bytes(b"src")
    (lm / "v1_trimmed.feather").write_bytes(b"same bytes")
    (lm / "v2_trimmed.feather").write_bytes(b"same bytes")
    (lm / "v3_trimmed.feather").write_bytes(b"different bytes")
    # The shape that caused real data loss: a legacy-named REAL file with a
    # vN symlink already pointing at it.
    (lm / "legacy_osm_enc_v1.feather").write_bytes(b"legacy real bytes")
    (lm / "v1.feather").symlink_to("legacy_osm_enc_v1.feather")
    (lm / "v3_trimmed.provenance.json").write_text("{}")
    (lm / "PROVENANCE.json").write_text("{}")
    (lm / "catalog_cache").mkdir()
    (lm / "catalog_cache" / "c.pkl").write_bytes(b"cache")

    # Dead lane, inbox, models, index cruft.
    (root / "artifacts" / "landmark_matching" / "boston_harbor_leg1").mkdir(
        parents=True)
    (root / "inbox" / "washington").mkdir(parents=True)
    (root / "inbox" / "washington" / "tool.py").write_text("x")
    (root / "models" / "sam2").mkdir(parents=True)
    (root / "artifacts" / "object_tracks" / "index.html.bak").write_text("x")
    return root


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = build_root(Path(self.tmp.name))
        self.plan = mig.build_plan(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def ops(self, kind):
        return [o for o in self.plan.ops if o["kind"] == kind]

    def test_prior_generation_is_moved_not_deleted(self):
        moves = [o for o in self.ops("move")
                 if "260819_mount_washington" in o["src"]]
        self.assertEqual(len(moves), 1)
        self.assertTrue(moves[0]["dst"].endswith(
            "260819_final/mtw_leg3_pre_selection_charge"))
        # And nothing about it is a delete.
        self.assertFalse([o for o in self.ops("delete")
                          if "260819_mount_washington" in o["src"]])

    def test_hidden_runs_move_but_exports_stay(self):
        moved = [o["src"] for o in self.ops("move")
                 if "localization_" in o["src"]]
        self.assertTrue(any("localization_run_m9_extsigma_uniform" in s
                            for s in moved))
        self.assertFalse(any("localization_export" in s for s in moved))

    def test_campaign_run_lands_in_the_campaign_not_the_taxonomy(self):
        dst = next(o["dst"] for o in self.ops("move")
                   if o["src"].endswith("localization_run_m9_uniform"))
        self.assertTrue(dst.endswith("260819_final/pohang_canal_04"), dst)
        other = next(o["dst"] for o in self.ops("move")
                     if o["src"].endswith("localization_run_m9_extsigma_uniform"))
        self.assertIn("260820_pohang_taxonomy", other)

    def test_absolute_symlink_is_replaced(self):
        self.assertTrue([o for o in self.ops("delete")
                         if o["src"].endswith("260819_final/pohang_canal_04")])

    def test_moved_runs_leave_a_pointer(self):
        self.assertTrue(self.ops("pointer"))
        for op in self.ops("pointer"):
            self.assertTrue(op["src"].endswith("moved_to.txt"))

    def test_only_verified_duplicates_and_caches_are_deleted(self):
        for op in self.ops("delete"):
            reason = op["note"].lower()
            self.assertTrue(
                "byte-identical" in reason or "regenerable" in reason
                or "symlink" in reason or "generated index" in reason,
                f"unjustified delete: {op['src']} ({op['note']})")

    def test_identical_trim_collapses_to_one_real_file(self):
        deleted = [o["src"] for o in self.ops("delete")]
        self.assertTrue(any(s.endswith("v2_trimmed.feather") for s in deleted))
        moved = [o["src"] for o in self.ops("move")]
        self.assertTrue(any(s.endswith("v1_trimmed.feather") for s in moved))
        self.assertTrue(any(s.endswith("v3_trimmed.feather") for s in moved))

    def test_every_moved_feather_leaves_a_compat_symlink(self):
        links = {Path(o["src"]).name for o in self.ops("symlink")
                 if "datasets/" in o["src"]}
        self.assertEqual(links, {"v1_trimmed.feather", "v2_trimmed.feather",
                                 "v3_trimmed.feather", "v1.feather",
                                 "legacy_osm_enc_v1.feather"})

    def test_a_preexisting_symlink_is_never_the_dedup_keeper(self):
        """The data-loss bug: `v1.feather` was already a symlink to
        `<ds>_osm_enc_v1.feather`. Reading through it made the two look like
        byte-identical copies, so the symlink was moved as the keeper and the
        REAL FILE it pointed at was deleted, leaving a cycle."""
        deleted = [Path(o["src"]).name for o in self.ops("delete")]
        self.assertNotIn("legacy_osm_enc_v1.feather", deleted)
        moved = [Path(o["src"]).name for o in self.ops("move")]
        self.assertIn("legacy_osm_enc_v1.feather", moved)
        self.assertNotIn("v1.feather", moved)

    def test_experiment_notes_are_written(self):
        written = [Path(o["src"]).parent.name for o in self.ops("write")
                   if Path(o["src"]).name == "experiment.md"]
        self.assertIn("260819_final", written)
        self.assertIn("260820_pohang_taxonomy", written)


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = build_root(Path(self.tmp.name))
        mig.apply_plan(self.root, mig.build_plan(self.root))

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_lane_shape(self):
        final = self.root / "runs" / "260819_final"
        self.assertTrue((final / "experiment.md").exists())
        self.assertTrue((final / "mtw_leg3_pre_selection_charge"
                         / "truth.jsonl").exists())
        self.assertTrue((final / "logs" / "mtw_leg3.log").exists())
        self.assertTrue((final / "figures" / "name_support.png").exists())
        self.assertTrue((final / "campaign_writeup.md").exists())
        self.assertFalse((self.root / "runs"
                          / "260819_mount_washington_20260815_leg3").exists())

    def test_campaign_pohang_is_a_real_directory_now(self):
        p = self.root / "runs" / "260819_final" / "pohang_canal_04"
        self.assertTrue(p.is_dir())
        self.assertFalse(p.is_symlink())

    def test_pointer_left_in_the_artifacts_lane(self):
        tr = (self.root / "artifacts" / "object_tracks" / "pohang_canal_04"
              / "v1" / "m3_tracks" / "runs" / "r002_v5_seamfix")
        pointer = tr / "localization_run_m9_uniform" / "moved_to.txt"
        self.assertTrue(pointer.exists())
        self.assertIn("runs/260819_final/pohang_canal_04",
                      pointer.read_text())
        # The export stayed put.
        self.assertTrue((tr / "localization_export_base").is_dir())

    def test_catalogs_moved_with_working_compat_symlinks(self):
        cat = self.root / "artifacts" / "catalogs" / "boston_harbor_leg1"
        self.assertEqual((cat / "v1_trimmed.feather").read_bytes(),
                         b"same bytes")
        self.assertTrue((cat / "sources" / "osm.feather").exists())
        self.assertTrue((cat / "manifest.json").exists())
        # The identical name survives as a symlink to the one real file.
        self.assertTrue((cat / "v2_trimmed.feather").is_symlink())
        self.assertEqual((cat / "v2_trimmed.feather").read_bytes(),
                         b"same bytes")
        # And every old absolute path still resolves.
        lm = self.root / "datasets" / "boston_harbor_leg1" / "landmarks"
        for name in ("v1_trimmed.feather", "v2_trimmed.feather",
                     "v3_trimmed.feather"):
            self.assertTrue((lm / name).is_symlink(), name)
            self.assertTrue((lm / name).exists(), f"{name} dangles")
        self.assertFalse((lm / "catalog_cache").exists())

    def test_no_symlink_dangles_or_cycles(self):
        for link in self.root.rglob("*"):
            if link.is_symlink():
                self.assertTrue(link.exists(),
                                f"{link} dangles -> {link.readlink()}")
        cat = (self.root / "artifacts" / "catalogs" / "boston_harbor_leg1")
        self.assertEqual((cat / "legacy_osm_enc_v1.feather").read_bytes(),
                         b"legacy real bytes")
        self.assertTrue((cat / "v1.feather").is_symlink())
        self.assertEqual((cat / "v1.feather").read_bytes(),
                         b"legacy real bytes")

    def test_housekeeping(self):
        self.assertIn("frozen", (self.root / "ORGANIZATION.md").read_text())
        self.assertTrue((self.root / "models" / "SOURCE.md").exists())
        self.assertTrue((self.root / "STATUS.md.stale").exists())
        self.assertTrue((self.root / "raw_material" / "washington"
                         / "tool.py").exists())
        self.assertFalse((self.root / "artifacts"
                          / "landmark_matching").exists())
        self.assertTrue((self.root / "archive" / "landmark_matching_lane"
                         / "boston_harbor_leg1").is_dir())
        self.assertFalse((self.root / "artifacts" / "object_tracks"
                          / "index.html.bak").exists())

    def test_applying_twice_is_safe(self):
        mig.apply_plan(self.root, mig.build_plan(self.root))
        self.assertTrue((self.root / "runs" / "260819_final"
                         / "pohang_canal_04").is_dir())


if __name__ == "__main__":
    unittest.main()
