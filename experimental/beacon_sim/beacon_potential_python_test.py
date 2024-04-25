
import experimental.beacon_sim.beacon_potential_python as bpp

import unittest


class BeaconPotentialTest(unittest.TestCase):
    def test_generate_sample(self):
        # Setup
        potential = bpp.BeaconPotential.correlated_beacon_potential(
            p_present=0.5,
            p_beacon_given_present=0.9,
            members=[10, 20])

        # Action
        assignment = potential.sample(0)

        # Verification
        for beacon_id in potential.members():
            self.assertIn(beacon_id, assignment)

    def test_conditioned_sample(self):
        # Setup
        eps = 1e-9
        # If we are very unlikely to be in the state where beacons are possible
        # and when beacons are possible, all beacons are present, then knowing
        # that one beacon is present means that all beacons are present
        potential = bpp.BeaconPotential.correlated_beacon_potential(
            p_present=0.0001,
            p_beacon_given_present=1-eps,
            members=[10, 20, 30])
        potential = potential.conditioned_on({10: True})

        # Action + Verification
        for seed in range(100000):
            assignment = potential.sample(seed)
            for beacon_id in potential.members():
                self.assertIn(beacon_id, assignment)
                self.assertTrue(assignment[beacon_id])


if __name__ == "__main__":
    unittest.main()
