from __future__ import annotations

import numpy as np
from typing import NamedTuple
from scipy import optimize, special
import math


class BeaconClique(NamedTuple):
    """
    This represents a set of beacons that are correlated in appearance
    """
    # The marginal probability of a beacon
    p_beacon: float
    # The probability that all beacons are gone
    p_no_beacon: float

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
    assert clique.p_no_beacon >= 0 and clique.p_no_beacon <= 1.0

    n = len(clique.members)

    # We use an ising model to model the interactions between beacons. However, it can be hard to
    # make global statements from a model that works locally. To work around this, we setup an
    # optimization problem where the parameters are tweaked until the global conditions are
    # satisfied.

    # In an ising model, the probability of a configuration can be expressed as:
    # P(x) = exp[x.T @ C + x + log(p_no_beacon)], where C is a symmetric matrix and x is a boolean
    # vector where the i'th entry expresses if the i'th beacon is present or not.

    # This compute the marginal probability of some arbitrary beacon being present.
    def log_marginal_prob(phi: float, psi: float):
        def kth_term(n: int, k: int, phi: float, psi: float) -> float:
            # compute log(nCr(n, k) * exp[x.T @ C @ x + np.log(p_no_beacon)])
            # n is the number of other beacons that exist and k is the number of other beacons that
            # present. We then have k+1 diagonal terms and (k+1)*k/2 edges where both nodes are
            # present, including the one that we are computing the probability for.
            return np.log(math.comb(n, k)) + (k+1) * phi + (k+1) * k / 2 * psi + np.log(clique.p_no_beacon)
        # Since one beacon is already present, we go through [0, n-1] other beacons being present
        return special.logsumexp([kth_term(n-1, k, phi, psi) for k in range(n)])

    # This computes the sum of the probability of all configurations
    def log_total_prob(phi: float, psi: float):
        def kth_term(n: int, k: int, phi: float, psi: float) -> float:
            return np.log(math.comb(n, k)) + k * phi + (k-1) * k / 2 * psi + np.log(clique.p_no_beacon)
        return special.logsumexp([kth_term(n, k, phi, psi) for k in range(n+1)])

    def loss(x):
        # The first term ensures that the marginal probability "constraint" is satisfied
        # The second term ensures that the total probability is log(1) = 0.
        return ((np.log(clique.p_beacon) - log_marginal_prob(x[0], x[1])) ** 2
                + log_total_prob(x[0], x[1]) ** 2)

    # I worked with the log probabilities because I thought that it would be easier to optimize,
    # but BFGS tended to fail for reasons unknown. I found that nelder-mead, a derivative free
    # worked well enough. These methods tend to fail when the number of decision variables gets to
    # be too high, but we've only got two, so no trouble here.
    result = optimize.minimize(loss, (1.0, 1.0), tol=1e-12, method='nelder-mead')
    assert result.success, f'Failed to find distribution that matches desiderata: {result}'
    # Our costs (beacon marginal and total probability = 1) are actually constraints. Only accept
    # solutions if the costs are near zero.
    assert result.fun < 1e-6, f'Can\'t satisfy total probability constraint or desired marginal probability: {result}'

    phi, psi = result.x
    off_diag = psi
    diag = phi
    off_diag_entries = np.ones((n, n)) - np.eye(n)
    covar = np.eye(n) * diag + off_diag_entries * off_diag / 2

    # the log probability of no beacons being present
    bias = np.log(clique.p_no_beacon)

    # Note that this distribution is normalized by construction.
    return BeaconPotential(
        members=clique.members,
        covariance=covar,
        bias=bias,
    )
