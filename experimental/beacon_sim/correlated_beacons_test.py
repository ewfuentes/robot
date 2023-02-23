
import unittest

from experimental.beacon_sim import correlated_beacons as cb

class CorrelatedBeaconsTest(unittest.TestCase):

    def test_invalid_probabilities_throws(self):
        # Setup
        bc = cb.BeaconClique(p_beacon = 0.5, p_all_beacons = 0.75, members=[1, 2])
        # Action + Verification
        with self.assertRaises(AssertionError):
            cb.create_correlated_beacons(bc)

    def test_independent_beacons(self):
        # Setup
        bc = cb.BeaconClique(p_beacon = 0.5, p_all_beacons = 0.25, members=[1, 2])
        # Action
        beacon_pot = cb.create_correlated_beacons(bc)

        # Verification
        print(beacon_pot)


if __name__ == "__main__":
    unittest.main()
