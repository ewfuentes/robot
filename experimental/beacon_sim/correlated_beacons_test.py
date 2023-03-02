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

    def test_cannot_join_distributions_with_common_members(self):
        # Setup
        p_beacon = 0.75
        p_together = p_beacon - 0.01
        clique_1 = cb.BeaconClique(
            p_beacon=p_beacon, p_all_beacons=p_together, members=[1, 2]
        )
        clique_2 = cb.BeaconClique(
            p_beacon=p_beacon, p_all_beacons=p_together, members=[2, 3]
        )

        # Action + Verification
        pot_1 = cb.create_correlated_beacons(clique_1)
        pot_2 = cb.create_correlated_beacons(clique_2)
        with self.assertRaises(AssertionError):
            combined_pot = pot_1 * pot_2

    def test_combined_distributions(self):
        # Setup
        # Create a clique that are positively correlated
        p_beacon = 0.75
        p_together = p_beacon - 0.001
        clique_1 = cb.BeaconClique(
            p_beacon=p_beacon, p_all_beacons=p_together, members=[1, 2]
        )
        # Create a clique that is negatively correlated
        p_beacon = 0.5
        p_together = 0.001
        clique_2 = cb.BeaconClique(
            p_beacon=p_beacon, p_all_beacons=p_together, members=[10, 11]
        )

        # Action
        pot_1 = cb.create_correlated_beacons(clique_1)
        pot_2 = cb.create_correlated_beacons(clique_2)
        combined_pot = pot_1 * pot_2

        # Verification
        # Test an unlikely scenario, the first clique being different the second being the same
        self.assertAlmostEqual(
            np.exp(combined_pot.log_prob({1: True, 2: False, 10: False, 11: False})),
            0.0,
            places=3,
        )

        # Expect symmetry in the second clique
        self.assertAlmostEqual(
            np.exp(combined_pot.log_prob({1: True, 2: True, 10: True, 11: False})),
            np.exp(combined_pot.log_prob({1: True, 2: True, 10: False, 11: True})),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
