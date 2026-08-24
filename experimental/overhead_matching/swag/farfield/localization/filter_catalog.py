"""Filter-side landmark catalog: positions, uniform uncertainty, priors.

One place that owns everything the measurement update needs to know about
landmarks, so ids and coordinates cannot drift out of sync and the candidate
prior w_j and kappa_eff have a single home (design doc §4/§5.3).

`max_visible_range_m` is REQUIRED. It used to default (10 km here, while the
replay fallback claimed 15 km), and that pair of silent numbers made every
synthetic run non-replayable: the radius decides which positions the proposal
thinks a landmark could have been seen from, so a guessed value changes the
hypotheses, the injections, and the belief. Producers pass it explicitly and
record it in the run manifest; replay reads the record.
"""

import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo


class LandmarkCatalog:
    """Known-position landmarks in region-frame ENU metres.

    `position_sigma_m` is one source-independent value applied uniformly to
    every landmark; it is projected into the angular domain per observation
    by `kappa_eff`. `log_prior` is the per-candidate prior log w_j; uniform by
    default.
    """

    def __init__(self, landmark_ids, east_m, north_m, *,
                 max_visible_range_m, position_sigma_m=None, log_prior=None):
        self.landmark_ids = list(landmark_ids)
        self.east_m = np.asarray(east_m, dtype=np.float64)
        self.north_m = np.asarray(north_m, dtype=np.float64)
        n = len(self.landmark_ids)
        if self.east_m.shape != (n,) or self.north_m.shape != (n,):
            raise ValueError(
                f"catalog arrays disagree with ids: {n} ids, "
                f"{self.east_m.shape} east, {self.north_m.shape} north")
        self._index = {lid: i for i, lid in enumerate(self.landmark_ids)}
        if len(self._index) != n:
            raise ValueError("duplicate landmark_id in catalog")

        if position_sigma_m is None:
            self.position_sigma_m = np.zeros(n)
        else:
            self.position_sigma_m = np.broadcast_to(
                np.asarray(position_sigma_m, dtype=np.float64), (n,)).copy()
        if (not np.all(np.isfinite(self.position_sigma_m))
                or np.any(self.position_sigma_m < 0.0)):
            raise ValueError("position_sigma_m must be finite and non-negative")
        if (n and np.any(
                self.position_sigma_m != self.position_sigma_m[0])):
            raise ValueError(
                "position_sigma_m must equal one uniform recorded value for "
                "all landmarks")

        if max_visible_range_m is None:
            raise ValueError(
                "max_visible_range_m is required — there is no default on "
                "purpose: a guessed radius silently changes the proposal's "
                "hypotheses. Pass the value the run records in its manifest.")
        self.max_visible_range_m = np.broadcast_to(
            np.asarray(max_visible_range_m, dtype=np.float64), (n,)).copy()
        if np.any(self.max_visible_range_m <= 0.0):
            raise ValueError("max_visible_range_m must be positive")

        if log_prior is None:
            # Uniform over the whole catalog. NOTE: until per-particle
            # spatial gating lands (§5.3 `cand(x)`), this couples the
            # posterior to catalog size — a bigger catalog dilutes every
            # candidate against the fixed null.
            self.log_prior = np.full(n, -np.log(n)) if n else np.zeros(0)
        else:
            self.log_prior = np.asarray(log_prior, dtype=np.float64)
            if self.log_prior.shape != (n,):
                raise ValueError("log_prior must have one entry per landmark")

    @property
    def n(self) -> int:
        return len(self.landmark_ids)

    def index_of(self, landmark_id: str) -> int:
        try:
            return self._index[landmark_id]
        except KeyError:
            raise ValueError(
                f"{landmark_id!r} is not in the catalog") from None

    def __contains__(self, landmark_id: str) -> bool:
        return landmark_id in self._index

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
        return (geo.compass_bearing_rad(d_east, d_north),
                np.hypot(d_east, d_north))

    def perturbed(self, sigma_m: float,
                  rng: np.random.Generator) -> "LandmarkCatalog":
        """Copy with positions jittered and one matching uniform uncertainty."""
        return LandmarkCatalog(
            self.landmark_ids,
            self.east_m + rng.normal(0.0, sigma_m, self.n),
            self.north_m + rng.normal(0.0, sigma_m, self.n),
            max_visible_range_m=self.max_visible_range_m,
            position_sigma_m=sigma_m,
            log_prior=self.log_prior)
