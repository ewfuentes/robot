"""Torch (GPU) backend for the §5.3 measurement update.

Same association-marginalized mixture as `filter.measurement_update` — the
numpy path remains the reference implementation and the golden-test target —
evaluated as one dense (n_particles, n_candidates) computation on the
configured device. It exists because the whole-map harbor catalog (13k
candidates x 20k particles x 344 measurements) made the numpy update the
iteration bottleneck at hours per run.

Equivalence is bounded by `torch_backend_test`, not bit-exact: float32 and a
different reduction order shift log-likelihoods at ~1e-4, far below any
decision the filter makes with them. The manifest records the backend
(`FilterConfig.measurement_backend`), so a run states which engine produced
it.
"""

import math

import numpy as np

import common.torch.load_torch_deps  # noqa: F401  # must precede torch
import torch

from experimental.overhead_matching.swag.bearing_only_localization import (
    filter as filter_mod,
    structs,
)

_TWO_PI = 2.0 * math.pi


class TorchMeasurementEngine:
    """Holds the catalog and per-tracklet LLR vectors on-device for a run."""

    def __init__(self, catalog, log_weight_by_tracklet: dict,
                 device: str = None, dtype=torch.float32, seed: int = 0,
                 surprise_by_tracklet: dict = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        # Own generator for association renewal draws (§5.3 persistence):
        # deterministic given the config seed, independent of the numpy
        # stream (the manifest records the backend, so replay is per-engine).
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.landmark_ids = list(catalog.landmark_ids)
        self.east = torch.as_tensor(catalog.east_m, dtype=dtype,
                                    device=self.device)
        self.north = torch.as_tensor(catalog.north_m, dtype=dtype,
                                     device=self.device)
        # Map-accuracy classes only matter when any are nonzero (kappa_eff
        # becomes per-pair); the scalar-kappa fast path skips the range map.
        if np.any(catalog.position_sigma_m > 0.0):
            self.sigma_pos = torch.as_tensor(
                catalog.position_sigma_m, dtype=dtype, device=self.device)
        else:
            self.sigma_pos = None
        # Proper identity posteriors per tracklet
        # (filter._identity_log_weights), already catalog-aligned.
        self.log_weight = {
            tid: torch.as_tensor(vec, dtype=dtype, device=self.device)
            for tid, vec in log_weight_by_tracklet.items()}
        # Identity-surprise masks (filter._surprise_mask), numpy for the
        # shared posterior helper plus an on-device copy for the mixture
        # path. Optional: absent -> surprise reported as 0.
        self.surprise_np = surprise_by_tracklet or {}
        self.surprise = {
            tid: torch.as_tensor(mask.astype(np.float64), dtype=dtype,
                                 device=self.device)
            for tid, mask in self.surprise_np.items()}
        self.surprise_bool = {
            tid: torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            for tid, mask in self.surprise_np.items()}

    def _log_terms(self, east, north, heading, meas):
        """log[(1-pi0)-less mixture terms]: identity posterior + log vM.

        The (1-pi0) and null constants are added by `update`; keeping this
        pure makes it reusable for hypothesis scoring.
        """
        kappa_z = min(float(meas.kappa), filter_mod.MAX_KAPPA)
        observed = math.radians(meas.bearing_body_deg)
        d_east = self.east[None, :] - east[:, None]
        d_north = self.north[None, :] - north[:, None]
        bearing = torch.atan2(d_east, d_north)  # compass: CW from north
        delta = bearing - heading[:, None] - observed
        delta = torch.remainder(delta + math.pi, _TWO_PI) - math.pi
        if self.sigma_pos is None:
            kappa = torch.as_tensor(kappa_z, dtype=self.dtype,
                                    device=self.device)
        else:
            rng = torch.sqrt(d_east * d_east + d_north * d_north)
            safe_range = torch.clamp(rng, min=1.0)
            kappa = 1.0 / (1.0 / kappa_z
                           + torch.square(self.sigma_pos[None, :]
                                          / safe_range))
        log_norm = math.log(_TWO_PI) + torch.log(
            torch.special.i0e(kappa)) + kappa
        log_vm = kappa * torch.cos(delta) - log_norm
        return self.log_weight[meas.tracklet_id][None, :] + log_vm

    @torch.no_grad()
    def pose_log_likelihood(self, east_m, north_m, heading_rad, meas,
                            pi0: float) -> np.ndarray:
        """Mirror of filter.pose_log_likelihood (§5.5 hypothesis scoring)."""
        east = torch.as_tensor(np.asarray(east_m, dtype=np.float64),
                               dtype=self.dtype, device=self.device)
        north = torch.as_tensor(np.asarray(north_m, dtype=np.float64),
                                dtype=self.dtype, device=self.device)
        heading = torch.as_tensor(np.asarray(heading_rad, dtype=np.float64),
                                  dtype=self.dtype, device=self.device)
        log_null = math.log(pi0) - math.log(_TWO_PI)
        terms = math.log1p(-pi0) + self._log_terms(east, north, heading, meas)
        log_landmark = torch.logsumexp(terms, dim=1)
        return torch.logaddexp(
            log_landmark,
            torch.full_like(log_landmark, log_null)).double().cpu().numpy()

    def _gumbel_like(self, tensor):
        uniform = torch.rand(tensor.shape, generator=self.generator,
                             dtype=tensor.dtype, device=self.device)
        # Explicit nesting: g = -log(E), E = -log(U), each clamped positive.
        # (An earlier version wrote `-torch.log(u).clamp_min(...)`, where
        # the method call binds before the unary minus — clamping the
        # NEGATIVE log(u) to 1e-30 and then taking log of a negative
        # number: every draw was NaN, NaN comparisons read false, and the
        # renewal sample silently degenerated to a deterministic argmax.)
        exponential = torch.clamp(-torch.log(torch.clamp(uniform, min=1e-30)),
                                  min=1e-30)
        return -torch.log(exponential)

    def _committed_keep(self, east, north, heading, assoc_t, meas,
                        outlier_rate: float):
        """(1-eps) vM(delta; kappa_eff) + eps/2pi toward each particle's own
        committed landmark; junk for particles not committed (masked by the
        caller)."""
        kappa_z = min(float(meas.kappa), filter_mod.MAX_KAPPA)
        observed = math.radians(meas.bearing_body_deg)
        j = assoc_t.clamp(min=0)
        d_east = self.east[j] - east
        d_north = self.north[j] - north
        bearing = torch.atan2(d_east, d_north)
        delta = bearing - heading - observed
        delta = torch.remainder(delta + math.pi, _TWO_PI) - math.pi
        if self.sigma_pos is None:
            kappa = torch.as_tensor(kappa_z, dtype=self.dtype,
                                    device=self.device)
        else:
            rng = torch.sqrt(d_east * d_east + d_north * d_north)
            kappa = 1.0 / (1.0 / kappa_z
                           + torch.square(self.sigma_pos[j]
                                          / torch.clamp(rng, min=1.0)))
        log_norm = math.log(_TWO_PI) + torch.log(
            torch.special.i0e(kappa)) + kappa
        vm = torch.exp(kappa * torch.cos(delta) - log_norm)
        return (1.0 - outlier_rate) * vm + outlier_rate / _TWO_PI

    @torch.no_grad()
    def update(self, belief, meas, pi0: float, per_mode: bool,
               resp_min: float, assoc: np.ndarray = None,
               renewal_rate: float = 0.1,
               outlier_rate: float = 0.1,
               draw_seed: int = None) -> list:
        """Mirror of filter.measurement_update: updates belief.log_weight
        (and `assoc`, when given) in place, returns the whole-belief
        AssociationPosterior then one per mode. `draw_seed` (see
        filter.measurement_draw_seed) makes the persistence draws
        order-invariant; without it the engine's own stream is used."""
        if draw_seed is not None:
            self.generator.manual_seed(int(draw_seed))
        if not 0.0 < pi0 < 1.0:
            raise ValueError(f"pi0 must be in (0, 1), got {pi0}")
        if not math.isfinite(meas.kappa) or meas.kappa <= 0.0:
            raise ValueError(f"kappa must be positive and finite, got "
                             f"{meas.kappa}")
        east = torch.as_tensor(belief.east_m, dtype=self.dtype,
                               device=self.device)
        north = torch.as_tensor(belief.north_m, dtype=self.dtype,
                                device=self.device)
        heading = torch.as_tensor(belief.heading_rad, dtype=self.dtype,
                                  device=self.device)
        log_null = math.log(pi0) - math.log(_TWO_PI)
        terms = math.log1p(-pi0) + self._log_terms(east, north, heading, meas)
        log_landmark = torch.logsumexp(terms, dim=1)
        log_lik = torch.logaddexp(
            log_landmark, torch.full_like(log_landmark, log_null))

        if assoc is not None:
            if not 0.0 < renewal_rate <= 1.0:
                raise ValueError(f"renewal_rate must be in (0, 1], got "
                                 f"{renewal_rate}")
            assoc_t = torch.as_tensor(assoc.astype(np.int64),
                                      device=self.device)
            keep = torch.zeros_like(log_lik)
            committed = assoc_t >= 0
            keep = torch.where(
                committed,
                self._committed_keep(east, north, heading, assoc_t, meas,
                                     outlier_rate),
                keep)
            keep = torch.where(assoc_t == filter_mod.ASSOC_NULL,
                               torch.full_like(keep, 1.0 / _TWO_PI), keep)
            uncommitted = assoc_t == filter_mod.ASSOC_UNCOMMITTED
            keep_scale = torch.where(uncommitted, torch.zeros_like(keep),
                                     torch.full_like(keep,
                                                     1.0 - renewal_rate))
            renew_scale = torch.where(uncommitted, torch.ones_like(keep),
                                      torch.full_like(keep, renewal_rate))
            renew_term = renew_scale * torch.exp(log_lik)
            likelihood = keep_scale * keep + renew_term
            belief.log_weight += torch.log(likelihood).double().cpu().numpy()

            # Gumbel-max renewal draw: endorsed candidates individually;
            # the null and every default-LLR candidate form one background
            # bucket that cannot anchor geometry (commitment requires
            # matcher endorsement — see filter._persistence_update).
            unendorsed = self.surprise_bool.get(meas.tracklet_id)
            if unendorsed is None:
                unendorsed = torch.zeros(terms.shape[1], dtype=torch.bool,
                                         device=self.device)
            gumbel_landmark = (
                terms.masked_fill(unendorsed[None, :], float("-inf"))
                + self._gumbel_like(terms))
            value, arg = gumbel_landmark.max(dim=1)
            background = torch.logaddexp(
                torch.logsumexp(
                    terms.masked_fill(~unendorsed[None, :], float("-inf")),
                    dim=1),
                torch.full_like(value, log_null))
            gumbel_null = background + self._gumbel_like(value)
            sampled = torch.where(
                gumbel_null > value,
                torch.full_like(arg, filter_mod.ASSOC_NULL), arg)
            renew_draw = torch.rand(likelihood.shape,
                                    generator=self.generator,
                                    dtype=self.dtype, device=self.device)
            renew = renew_draw < renew_term / likelihood
            assoc[:] = torch.where(renew, sampled,
                                   assoc_t).cpu().numpy().astype(np.int32)
            return filter_mod._commit_share_posteriors(
                belief, meas, assoc, self.landmark_ids, per_mode, resp_min,
                self.surprise_np.get(meas.tracklet_id))

        belief.log_weight += log_lik.double().cpu().numpy()

        groups = filter_mod._responsibility_groups(belief, per_mode)
        group_w = torch.as_tensor(
            np.stack([w for _, w in groups]), dtype=self.dtype,
            device=self.device)  # (G, n)
        resp = torch.exp(terms - log_lik[:, None])
        avg = group_w @ resp  # (G, m)
        null_shares = (group_w @ torch.exp(log_null - log_lik)).cpu().numpy()
        mask = self.surprise.get(meas.tracklet_id)
        surprise_shares = ((avg @ mask).double().cpu().numpy()
                           if mask is not None else np.zeros(len(groups)))

        responsibilities = [{} for _ in groups]
        rows, cols = torch.nonzero(avg >= resp_min, as_tuple=True)
        values = avg[rows, cols].double().cpu().numpy()
        rows = rows.cpu().numpy()
        cols = cols.cpu().numpy()
        for row, col, value in zip(rows, cols, values):
            responsibilities[row][self.landmark_ids[col]] = float(value)

        return [
            structs.AssociationPosterior(
                tracklet_id=meas.tracklet_id,
                anchor_keyframe_idx=meas.anchor_keyframe_idx,
                null_share=float(null_shares[position]),
                responsibilities=responsibilities[position],
                mode_id=mode_id,
                surprise_share=float(surprise_shares[position]))
            for position, (mode_id, _) in enumerate(groups)]
