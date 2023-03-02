import unittest
import numpy as np

from experimental.beacon_sim import correlated_beacons as cb


class CorrelatedBeaconsTest(unittest.TestCase):
    def test_invalid_probabilities_throws(self):
        # Setup
        bc = cb.BeaconClique(p_beacon=0.5, p_all_beacons=0.75, members=[1, 2])
        # Action + Verification
        with self.assertRaises(AssertionError):
            cb.create_correlated_beacons(bc)

    def test_independent_beacons(self):
        # Setup
        p_beacon = 0.75
        p_together = p_beacon * p_beacon
        bc = cb.BeaconClique(
            p_beacon=p_beacon, p_all_beacons=p_together, members=[1, 2]
        )
        # Action
        beacon_pot = cb.create_correlated_beacons(bc)

        p_00 = np.exp(beacon_pot.log_prob({1: False, 2: False}))
        p_01 = np.exp(beacon_pot.log_prob({1: True, 2: False}))
        p_10 = np.exp(beacon_pot.log_prob({1: False, 2: True}))
        p_11 = np.exp(beacon_pot.log_prob({1: True, 2: True}))

        # Verification
        self.assertAlmostEqual(p_01 + p_11, 0.75, places=6)
        self.assertAlmostEqual(p_10 + p_11, 0.75, places=6)
        self.assertAlmostEqual(p_01, p_10, places=6)
        self.assertAlmostEqual(p_00 + p_01 + p_10 + p_11, 1.0, places=6)

    def test_correlated_beacons(self):
        # Setup
        p_beacon = 0.75
        p_together = p_beacon - 0.01
        bc = cb.BeaconClique(
            p_beacon=p_beacon, p_all_beacons=p_together, members=[1, 2]
        )
        # Action
        beacon_pot = cb.create_correlated_beacons(bc)

        p_00 = np.exp(beacon_pot.log_prob({1: False, 2: False}))
        p_01 = np.exp(beacon_pot.log_prob({1: True, 2: False}))
        p_10 = np.exp(beacon_pot.log_prob({1: False, 2: True}))
        p_11 = np.exp(beacon_pot.log_prob({1: True, 2: True}))

        # Verification
        self.assertAlmostEqual(p_01 + p_11, 0.75, places=6)
        self.assertAlmostEqual(p_10 + p_11, 0.75, places=6)
        self.assertAlmostEqual(p_01, p_10, places=6)
        self.assertAlmostEqual(p_00 + p_01 + p_10 + p_11, 1.0, places=6)

if __name__ == "__main__":
    unittest.main()
