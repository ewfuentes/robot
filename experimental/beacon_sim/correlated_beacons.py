
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
    log_normalizer: float

def create_correlated_beacons(clique: BeaconClique) -> BeaconPotential:
    assert clique.p_all_beacons <= clique.p_beacon, 'Probability of all beacons must be less than marginal probability'

    n = len(clique.members)

    p_single_beacon = clique.p_beacon - clique.p_all_beacons
    p_no_beacon = 1 - (clique.p_beacon + p_single_beacon)
    off_diag = np.log(p_no_beacon * clique.p_all_beacons / p_single_beacon**2)
    diag = np.log(p_single_beacon / p_no_beacon)

    covar = np.eye(n) * diag + (np.ones((n,n)) - np.eye(n)) * off_diag / 2
    bias = np.log(p_no_beacon)

    log_normalizer = 2**n * (bias + diag) + 2**(n-2) * off_diag

    return BeaconPotential(
        members=clique.members,
        covariance=covar,
        bias=bias,
        log_normalizer=log_normalizer,
    )

