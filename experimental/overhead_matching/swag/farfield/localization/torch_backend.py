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

from experimental.overhead_matching.swag.farfield.localization import (
    filter as filter_mod,
    structs,
)

_TWO_PI = 2.0 * math.pi
# Bounds the dense particle x landmark working set by ELEMENTS, not landmark
# count: the association path holds ~9 such blocks concurrently, so a fixed
# 4096-landmark chunk OOMs a 32 GiB accelerator once the particle count grows
# past ~10x the 50k it was sized for.  4096 * 50_000 elements is a ~0.8 GiB
# float32 block — the empirically comfortable peak — and dividing by the
# particle count keeps that peak flat as budgets scale.  At 50k particles the
# chunk is exactly the old 4096, so existing runs reproduce bit-for-bit.
_CANDIDATE_BLOCK_ELEMENTS = 4096 * 50_000
_CANDIDATE_CHUNK_MIN = 256


class TorchMeasurementEngine:
    """Holds the catalog and per-tracklet LLR vectors on-device for a run."""

    def __init__(self, catalog, log_weight_by_tracklet: dict,
                 device: str = None, dtype=torch.float32, seed: int = 0,
                 surprise_by_tracklet: dict = None,
                 range_softness: float = 0.25):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        if not (math.isfinite(range_softness) and range_softness > 0.0):
            raise ValueError(f"range_softness must be finite and positive, "
                             f"got {range_softness}")
        # Width of the one-sided range-cap tail as a fraction of the cap
        # (filter.range_cap_log_term); the cap itself rides on each
        # measurement.
        self.range_softness = float(range_softness)
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
        # The uniform map-position uncertainty matters only when nonzero
        # (kappa_eff becomes range-dependent); otherwise skip the range map.
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

    def _candidate_slices(self, n_particles: int):
        chunk = max(_CANDIDATE_CHUNK_MIN,
                    _CANDIDATE_BLOCK_ELEMENTS // max(n_particles, 1))
        for start in range(0, len(self.landmark_ids), chunk):
            yield start, min(start + chunk, len(self.landmark_ids))

    def _log_terms(self, east, north, heading, meas, start=0, end=None):
        """log[(1-pi0)-less mixture terms]: identity posterior + log vM.

        The (1-pi0) and null constants are added by `update`; keeping this
        pure makes it reusable for hypothesis scoring.
        """
        kappa_z = min(float(meas.kappa), filter_mod.MAX_KAPPA)
        observed = math.radians(meas.bearing_forward_cw_deg)
        end = len(self.landmark_ids) if end is None else end
        d_east = self.east[None, start:end] - east[:, None]
        d_north = self.north[None, start:end] - north[:, None]
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
                           + torch.square(self.sigma_pos[None, start:end]
                                          / safe_range))
        log_norm = math.log(_TWO_PI) + torch.log(
            torch.special.i0e(kappa)) + kappa
        log_vm = kappa * torch.cos(delta) - log_norm
        return (self.log_weight[meas.tracklet_id][None, start:end]
                + log_vm + self._range_cap_log_term(d_east, d_north, meas))

    def _range_cap_log_term(self, d_east, d_north, meas):
        """Mirror of filter.range_cap_log_term on the torch device."""
        cap = getattr(meas, "range_max_m", None)
        if cap is None:
            return 0.0
        if not (math.isfinite(cap) and cap > 0.0):
            raise ValueError(f"range_max_m must be finite and positive, got "
                             f"{cap}")
        rng = torch.sqrt(d_east * d_east + d_north * d_north)
        excess = torch.clamp(rng - cap, min=0.0)
        return -0.5 * torch.square(excess / (self.range_softness * cap))

    def _landmark_log_likelihood(self, east, north, heading, meas,
                                 log_scale: float):
        result = torch.full(east.shape, float("-inf"), dtype=self.dtype,
                            device=self.device)
        for start, end in self._candidate_slices(east.shape[0]):
            terms = log_scale + self._log_terms(
                east, north, heading, meas, start, end)
            result = torch.logaddexp(result, torch.logsumexp(terms, dim=1))
        return result

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
        log_landmark = self._landmark_log_likelihood(
            east, north, heading, meas, math.log1p(-pi0))
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
        observed = math.radians(meas.bearing_forward_cw_deg)
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
        vm = torch.exp(kappa * torch.cos(delta) - log_norm
                       + self._range_cap_log_term(d_east, d_north, meas))
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
        log_scale = math.log1p(-pi0)
        log_landmark = torch.full(
            east.shape, float("-inf"), dtype=self.dtype, device=self.device)
        best_value = best_arg = background_landmark = None
        unendorsed = None
        if assoc is not None:
            unendorsed = self.surprise_bool.get(meas.tracklet_id)
            if unendorsed is None:
                unendorsed = torch.zeros(
                    len(self.landmark_ids), dtype=torch.bool,
                    device=self.device)
            best_value = torch.full_like(log_landmark, float("-inf"))
            best_arg = torch.zeros(east.shape, dtype=torch.int64,
                                   device=self.device)
            background_landmark = torch.full_like(
                log_landmark, float("-inf"))

        for start, end in self._candidate_slices(east.shape[0]):
            terms = log_scale + self._log_terms(
                east, north, heading, meas, start, end)
            log_landmark = torch.logaddexp(
                log_landmark, torch.logsumexp(terms, dim=1))
            if assoc is not None:
                chunk_unendorsed = unendorsed[start:end]
                gumbel_landmark = (
                    terms.masked_fill(
                        chunk_unendorsed[None, :], float("-inf"))
                    + self._gumbel_like(terms))
                chunk_value, chunk_arg = gumbel_landmark.max(dim=1)
                better = chunk_value > best_value
                best_value = torch.where(better, chunk_value, best_value)
                best_arg = torch.where(better, chunk_arg + start, best_arg)
                chunk_background = torch.logsumexp(
                    terms.masked_fill(
                        ~chunk_unendorsed[None, :], float("-inf")), dim=1)
                background_landmark = torch.logaddexp(
                    background_landmark, chunk_background)
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
            background = torch.logaddexp(
                background_landmark,
                torch.full_like(best_value, log_null))
            gumbel_null = background + self._gumbel_like(best_value)
            sampled = torch.where(
                gumbel_null > best_value,
                torch.full_like(best_arg, filter_mod.ASSOC_NULL), best_arg)
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
        null_shares = (group_w @ torch.exp(log_null - log_lik)).cpu().numpy()
        mask = self.surprise.get(meas.tracklet_id)
        responsibilities = [{} for _ in groups]
        surprise_shares = torch.zeros(
            len(groups), dtype=self.dtype, device=self.device)
        for start, end in self._candidate_slices(east.shape[0]):
            terms = log_scale + self._log_terms(
                east, north, heading, meas, start, end)
            avg = group_w @ torch.exp(terms - log_lik[:, None])
            if mask is not None:
                surprise_shares += avg @ mask[start:end]
            rows, cols = torch.nonzero(avg >= resp_min, as_tuple=True)
            values = avg[rows, cols].double().cpu().numpy()
            rows = rows.cpu().numpy()
            cols = cols.cpu().numpy()
            for row, col, value in zip(rows, cols, values):
                responsibilities[row][self.landmark_ids[start + col]] = (
                    float(value))
        surprise_shares = surprise_shares.double().cpu().numpy()

        return [
            structs.AssociationPosterior(
                tracklet_id=meas.tracklet_id,
                anchor_keyframe_idx=meas.anchor_keyframe_idx,
                null_share=float(null_shares[position]),
                responsibilities=responsibilities[position],
                mode_id=mode_id,
                surprise_share=float(surprise_shares[position]))
            for position, (mode_id, _) in enumerate(groups)]
