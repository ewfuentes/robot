"""Tests for the trajectory registry's live/rejected split.

The registry used to be one dict where rejected and duplicate entries were
tagged inline, and every selector had to remember to filter them; a selector
that forgot would silently re-collect rejected data. The split makes that
impossible: TRAJECTORIES holds only collectable entries, REJECTED_TRAJECTORIES
holds everything screened out (with a one-line reason), and the two share the
seed-pkey namespace for dedup checks.
"""

import unittest

from experimental.overhead_matching.swag.farfield.collection import (
    farfield_trajectories as reg,
)


class RegistrySplitTest(unittest.TestCase):
    def test_live_entries_carry_no_rejection_markers(self):
        for name, cfg in reg.TRAJECTORIES.items():
            self.assertNotIn("rejected", cfg, name)
            self.assertNotIn("duplicate_of", cfg, name)
            self.assertNotIn("reason", cfg, name)

    def test_collectable_is_exactly_the_live_dict(self):
        self.assertEqual(reg.collectable(), reg.TRAJECTORIES)

    def test_every_rejected_entry_states_a_reason(self):
        for name, cfg in reg.REJECTED_TRAJECTORIES.items():
            self.assertTrue(cfg.get("reason", "").strip(), name)
            self.assertTrue(cfg.get("seed_pkey", "").strip(), name)

    def test_duplicates_point_at_a_screened_entry(self):
        for name, cfg in reg.REJECTED_TRAJECTORIES.items():
            target = cfg.get("duplicate_of")
            if target is None:
                continue
            self.assertIn(target,
                          set(reg.TRAJECTORIES) | set(reg.REJECTED_TRAJECTORIES),
                          f"{name} duplicates unknown entry {target}")

    def test_no_name_appears_in_both_dicts(self):
        overlap = set(reg.TRAJECTORIES) & set(reg.REJECTED_TRAJECTORIES)
        self.assertEqual(overlap, set())

    def test_seed_pkeys_are_globally_unique(self):
        # The whole point of keeping rejected entries: a "new" seed must be
        # checkable against everything ever screened, so pkeys cannot collide.
        seen = {}
        for name, cfg in {**reg.TRAJECTORIES,
                          **reg.REJECTED_TRAJECTORIES}.items():
            pkey = cfg["seed_pkey"]
            self.assertNotIn(pkey, seen,
                             f"{name} and {seen.get(pkey)} share seed {pkey}")
            seen[pkey] = name

    def test_known_seed_pkeys_covers_both_dicts(self):
        known = reg.known_seed_pkeys()
        self.assertIn(reg.TRAJECTORIES["folkestone_dover"]["seed_pkey"], known)
        self.assertIn(
            reg.REJECTED_TRAJECTORIES["anglesey_menai"]["seed_pkey"], known)
        self.assertEqual(known[reg.REJECTED_TRAJECTORIES["anglesey_menai"]
                               ["seed_pkey"]], "anglesey_menai")

    def test_pilot_entries_are_live(self):
        for name in reg.PILOT:
            self.assertIn(name, reg.TRAJECTORIES)

    def test_live_entries_have_the_fields_the_orchestrator_reads(self):
        for name, cfg in reg.TRAJECTORIES.items():
            self.assertIsInstance(cfg["pano"], bool, name)
            self.assertTrue(cfg["osm"], name)
            self.assertIn("enc_state", cfg, name)
            self.assertTrue(cfg["note"].strip(), name)

    def test_selectors_partition_the_live_set(self):
        pano = set(reg.pano_names())
        persp = set(reg.perspective_names())
        self.assertEqual(pano | persp, set(reg.TRAJECTORIES))
        self.assertEqual(pano & persp, set())


if __name__ == "__main__":
    unittest.main()
