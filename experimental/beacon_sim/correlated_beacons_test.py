import unittest
import numpy as np

from experimental.beacon_sim import correlated_beacons_python as cb


class CorrelatedBeaconsTest(unittest.TestCase):
    def test_independent_beacons(self):
        # Setup
        p_beacon = 0.75
        p_no_beacons = (1 - p_beacon) ** 2
        bc = cb.BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[1, 2],
        )
        # Action
        beacon_pot = cb.create_correlated_beacons(bc)

        p_00 = np.exp(beacon_pot.log_prob({1: False, 2: False}))
        p_01 = np.exp(beacon_pot.log_prob({1: True, 2: False}))
        p_10 = np.exp(beacon_pot.log_prob({1: False, 2: True}))
        p_11 = np.exp(beacon_pot.log_prob({1: True, 2: True}))

        # Verification
        print('p_no_beacon:', p_00)
        print('p_marginal:', p_01 + p_11)
        self.assertAlmostEqual(p_00, p_no_beacons, places=6)
        self.assertAlmostEqual(p_01 + p_11, 0.75, places=6)
        self.assertAlmostEqual(p_10 + p_11, 0.75, places=6)
        self.assertAlmostEqual(p_01, p_10, places=6)
        self.assertAlmostEqual(p_00 + p_01 + p_10 + p_11, 1.0, places=6)

    def test_three_independent_beacons(self):
        # Setup
        p_beacon = 0.75
        p_no_beacons = (1 - p_beacon) ** 3
        bc = cb.BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[1, 2, 3],
        )
        # Action
        beacon_pot = cb.create_correlated_beacons(bc)

        p_000 = np.exp(beacon_pot.log_prob({1: False, 2: False, 3: False}))
        p_010 = np.exp(beacon_pot.log_prob({1: True, 2: False, 3: False}))
        p_100 = np.exp(beacon_pot.log_prob({1: False, 2: True, 3: False}))
        p_110 = np.exp(beacon_pot.log_prob({1: True, 2: True, 3: False}))
        p_001 = np.exp(beacon_pot.log_prob({1: False, 2: False, 3: True}))
        p_011 = np.exp(beacon_pot.log_prob({1: True, 2: False, 3: True}))
        p_101 = np.exp(beacon_pot.log_prob({1: False, 2: True, 3: True}))
        p_111 = np.exp(beacon_pot.log_prob({1: True, 2: True, 3: True}))

        # Verification
        self.assertAlmostEqual(
            p_000 + p_001 + p_010 + p_011 + p_100 + p_101 + p_110 + p_111, 1.0, places=6
        )
        self.assertAlmostEqual(p_000, p_no_beacons, places=6)
        self.assertAlmostEqual(p_001 + p_011 + p_101 + p_111, 0.75, places=6)
        self.assertAlmostEqual(p_010 + p_011 + p_110 + p_111, 0.75, places=6)
        self.assertAlmostEqual(p_100 + p_101 + p_110 + p_111, 0.75, places=6)
        self.assertAlmostEqual(p_010, p_100, places=6)
        self.assertAlmostEqual(p_010, p_001, places=6)
        self.assertAlmostEqual(p_111, p_beacon ** 3, places=6)
        self.assertAlmostEqual(p_001, p_beacon * (1-p_beacon)**2, places=6)

    def test_correlated_beacons(self):
        # Setup
        p_beacon = 0.75
        p_no_beacons = 0.1
        bc = cb.BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[1, 2],
        )
        # Action
        beacon_pot = cb.create_correlated_beacons(bc)

        p_00 = np.exp(beacon_pot.log_prob({1: False, 2: False}))
        p_01 = np.exp(beacon_pot.log_prob({1: True, 2: False}))
        p_10 = np.exp(beacon_pot.log_prob({1: False, 2: True}))
        p_11 = np.exp(beacon_pot.log_prob({1: True, 2: True}))

        # Verification
        self.assertAlmostEqual(p_00, p_no_beacons, places=6)
        self.assertAlmostEqual(p_00 + p_01 + p_10 + p_11, 1.0, places=6)
        self.assertAlmostEqual(p_01 + p_11, p_beacon, places=6)
        self.assertAlmostEqual(p_10 + p_11, p_beacon, places=6)
        self.assertAlmostEqual(p_01, p_10, places=6)

    def test_cannot_join_distributions_with_common_members(self):
        # Setup
        p_beacon = 0.75
        p_no_beacons = (1 - p_beacon) ** 2
        clique_1 = cb.BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[1, 2],
        )
        clique_2 = cb.BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[2, 3],
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
        p_no_beacons = 0.1
        clique_1 = cb.BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[1, 2],
        )
        # Create a clique that is negatively correlated
        p_beacon = 0.5
        p_no_beacons = 0.001
        clique_2 = cb.BeaconClique(
            p_beacon=p_beacon,
            p_no_beacons=p_no_beacons,
            members=[10, 11],
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
