from __future__ import annotations

import numpy as np
from typing import NamedTuple


class BeaconClique(NamedTuple):
    """
    This represents a set of beacons that are correlated in appearance
    """

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
        """Compute the log probability of the given configuration

        Note that this function will assert if one of the members isn't given an assignment
        """
        input_list = []
        for member in self.members:
            assert member in assignments
            input_list.append(int(assignments[member]))
        input_array = np.array(input_list, ndmin=2)
        return (input_array @ self.covariance @ input_array.T + self.bias)[0, 0]

    def __mul__(self, other: BeaconPotential) -> BeaconPotential:
        """Combine two beacon potentials"""
        assert set(self.members).isdisjoint(
            set(other.members)
        ), rf"members should be disjoint but self /\ other = {set(self.members) & set(other.members)}"
        n = len(self.members)
        m = len(other.members)

        new_covar = np.zeros((n + m, n + m))
        new_covar[:n, :n] = self.covariance
        new_covar[-m:, -m:] = other.covariance
        new_bias = self.bias + other.bias
        return BeaconPotential(
            members=self.members + other.members, covariance=new_covar, bias=new_bias
        )


def create_correlated_beacons(clique: BeaconClique) -> BeaconPotential:
    assert clique.p_beacon >= 0 and clique.p_beacon <= 1.0
    # For a beacon clique, all beacons have the same pairwise interactions.
    # For a beacon pair, they can either both be present, both be absent, or one of
    # the two is present. Denote these configurations as {00, 11, 01, 10}. Then
    # the marginal probability of the first beacon being present is:
    # p_beacon = p_10 + p_11
    # The probability of both beacons being present is p_11 = p_all_beacons.
    # For this reason, we must ensure that p_11 < p_beacon.
    assert (
        clique.p_all_beacons < clique.p_beacon
    ), "Probability of all beacons must be less than marginal probability"

    n = len(clique.members)

    # Expressed in exponential form, the distribution is:
    # exp(log(p_00) + x1 * log(p_10/p_00) + x2 * log(p_01/p_00) + x1 * x2 * log(p_00 * p_11 / (p_01 * p_10)))
    # In matrix form:
    # exp( [x1 x2] [[                      log(p_10 / p_00) 0.5 * log(p_00 * p_11 / (p_01 * p_10))] [[x1] + log(p_00))
    #               [0.5 * log(p_00 * p_11 / (p_01 * p_10))                       log(p_01 / p_00)]] [x2]]

    # Compute p_01 = p_10 = p_11 + p_01 - p_11
    p_single_beacon = clique.p_beacon - clique.p_all_beacons
    # compute p_00 = 1 - p_01 - p_11 - p_10
    p_no_beacon = 1 - (clique.p_beacon + p_single_beacon)
    # compute the diagonal and off diagonal terms of the matrix above
    off_diag = np.log(p_no_beacon * clique.p_all_beacons / p_single_beacon ** 2)
    diag = np.log(p_single_beacon / p_no_beacon)
    covar = np.eye(n) * diag + (np.ones((n, n)) - np.eye(n)) * off_diag / 2

    # the log probability of no beacons being present
    bias = np.log(p_no_beacon)

    # Note that this distribution is normalized by construction.
    return BeaconPotential(
        members=clique.members,
        covariance=covar,
        bias=bias,
    )
