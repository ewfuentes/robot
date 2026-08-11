"""Landmark catalog: positions, map-accuracy classes, candidate priors.

One place that owns everything the measurement update needs to know about
landmarks, so ids and coordinates cannot drift out of sync (they used to be
three parallel arrays threaded through every call) and so the candidate
prior w_j and kappa_eff have a single home — design doc §4/§5.3.
"""

import numpy as np

from experimental.overhead_matching.swag.bearing_only_localization import (
    geodesy,
)


class LandmarkCatalog:
    """Known-position landmarks in region-frame ENU metres.

    `position_sigma_m` is the map-accuracy class of each landmark (ENC
    surveyed << OSM); it is projected into the angular domain per
    observation by `kappa_eff`. `log_prior` is the per-candidate prior
    log w_j; uniform by default.
    """

    def __init__(self, landmark_ids, east_m, north_m, position_sigma_m=None,
                 log_prior=None):
        self.landmark_ids = list(landmark_ids)
        self.east_m = np.asarray(east_m, dtype=np.float64)
        self.north_m = np.asarray(north_m, dtype=np.float64)
        n = len(self.landmark_ids)
        if self.east_m.shape != (n,) or self.north_m.shape != (n,):
            raise ValueError(
                f"catalog arrays disagree with ids: {n} ids, "
                f"{self.east_m.shape} east, {self.north_m.shape} north")
        if len(set(self.landmark_ids)) != n:
            raise ValueError("duplicate landmark_id in catalog")

        if position_sigma_m is None:
            self.position_sigma_m = np.zeros(n)
        else:
            self.position_sigma_m = np.broadcast_to(
                np.asarray(position_sigma_m, dtype=np.float64), (n,)).copy()
        if np.any(self.position_sigma_m < 0.0):
            raise ValueError("position_sigma_m must be non-negative")

        if log_prior is None:
            # Uniform over the whole catalog. NOTE: until per-particle
            # spatial gating lands (§5.3 `cand(x)`), this couples the
            # posterior to catalog size — a bigger catalog dilutes every
            # candidate against the fixed null. Documented, tested by
            # `filter_test.MeasurementDensityTest`, resolved by gating.
            self.log_prior = np.full(n, -np.log(n)) if n else np.zeros(0)
        else:
            self.log_prior = np.asarray(log_prior, dtype=np.float64)
            if self.log_prior.shape != (n,):
                raise ValueError("log_prior must have one entry per landmark")

    @property
    def n(self) -> int:
        return len(self.landmark_ids)

    def index_of(self, landmark_id: str) -> int:
        return self.landmark_ids.index(landmark_id)

    def kappa_eff(self, kappa_z: float, range_m: np.ndarray,
                  candidate_slice=slice(None)) -> np.ndarray:
        """Combine tracklet concentration with projected map error (§4).

        A position error of s metres at range r projects to ~s/r radians of
        bearing error, so variances add: 1/kappa_eff = 1/kappa_z + (s/r)^2.
        Returns kappa_z unchanged where the catalog is exact.
        """
        sigma_pos = self.position_sigma_m[candidate_slice]
        if not np.any(sigma_pos > 0.0):
            return np.full_like(range_m, float(kappa_z))
        safe_range = np.maximum(range_m, 1.0)
        var = 1.0 / kappa_z + np.square(sigma_pos / safe_range)
        return 1.0 / var

    def bearings_from(self, east_m: np.ndarray, north_m: np.ndarray,
                      candidate_slice=slice(None)):
        """World-frame bearings and ranges from each particle to each
        candidate. Returns (bearing_rad, range_m), both (n_particles, n_cand).
        """
        d_east = self.east_m[candidate_slice][None, :] - east_m[:, None]
        d_north = self.north_m[candidate_slice][None, :] - north_m[:, None]
        return (geodesy.compass_bearing_rad(d_east, d_north),
                np.hypot(d_east, d_north))

    def perturbed(self, sigma_m: float,
                  rng: np.random.Generator) -> "LandmarkCatalog":
        """Copy with positions jittered — models map error for T-F10. The
        accuracy class is set to match, so kappa_eff can absorb it."""
        return LandmarkCatalog(
            self.landmark_ids,
            self.east_m + rng.normal(0.0, sigma_m, self.n),
            self.north_m + rng.normal(0.0, sigma_m, self.n),
            position_sigma_m=sigma_m,
            log_prior=self.log_prior)
