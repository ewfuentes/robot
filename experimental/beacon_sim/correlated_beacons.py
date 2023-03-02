
import numpy as np
from typing import NamedTuple

class BeaconClique(NamedTuple):
    '''
    This represents a set of beacons that are correlated in appearance
    '''
    # The marginal probability of each beacon being present
    p_beacon: float
    # The probability of all pair of beacons being present
    p_all_beacons: float
    members: list[int]

class BeaconPotential(NamedTuple):
    members: list[int]
    covariance: np.ndarray
    bias: float

    def log_prob(self, assignments: dict[int, bool]) -> float:
        input_list = []
        for member in self.members:
            assert member in assignments
            input_list.append(int(assignments[member]))
        input_array = np.array(input_list, ndmin=2)
        print(self.covariance)
        print(self.bias)
        return (input_array @ self.covariance @ input_array.T + self.bias)[0, 0]



def create_correlated_beacons(clique: BeaconClique) -> BeaconPotential:
    assert clique.p_all_beacons < clique.p_beacon, 'Probability of all beacons must be less than marginal probability'

    n = len(clique.members)

    p_single_beacon = clique.p_beacon - clique.p_all_beacons
    p_no_beacon = 1 - (clique.p_beacon + p_single_beacon)
    off_diag = np.log(p_no_beacon * clique.p_all_beacons / p_single_beacon**2)
    diag = np.log(p_single_beacon / p_no_beacon)

    covar = np.eye(n) * diag + (np.ones((n,n)) - np.eye(n)) * off_diag / 2
    bias = np.log(p_no_beacon)

    return BeaconPotential(
        members=clique.members,
        covariance=covar,
        bias=bias,
    )

