"""Exact grid-HMM localization with causal delayed-track replay.

This experiment applies the current independent-epoch bearing mixture on an
exact (heading, north, east) grid and scores the current posterior at each
keyframe.

The observation input remains ``epoch_fused_compat_v1``. In ``natural``
availability mode an audited track arrives atomically at its recorded close
keyframe, its measurements retain their historical anchors, and the current
prefix is replayed from the last unaffected checkpoint. Previously emitted
current-state scores are never revised.

The full belief history is too large on Pohang, so causal replay keeps one
host checkpoint every ``--checkpoint_keyframes``.
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import msgspec
import numpy as np

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    filter as filter_lib,
    odometry_profiles,
    release_schedule as release_schedule_lib,
    structs,
)

MAX_KAPPA = filter_lib.MAX_KAPPA
RADII_M = (50.0, 100.0, 250.0, 500.0, 1000.0)


class Grid:
    """Cell geometry: index zero of each axis is at the box minimum."""

    def __init__(self, east_min, east_max, north_min, north_max, cell_m):
        self.cell_m = float(cell_m)
        self.east_min = float(east_min)
        self.north_min = float(north_min)
        self.n_east = int(math.ceil((east_max - east_min) / cell_m))
        self.n_north = int(math.ceil((north_max - north_min) / cell_m))

    def centers(self):
        east = self.east_min + (np.arange(self.n_east) + 0.5) * self.cell_m
        north = self.north_min + (np.arange(self.n_north) + 0.5) * self.cell_m
        return east, north


def shift2d(t: torch.Tensor, di: int, dj: int) -> torch.Tensor:
    """Integer zero-fill shift of the trailing (north, east) axes."""
    if di == 0 and dj == 0:
        return t
    out = torch.zeros_like(t)
    ni, nj = t.shape[-2], t.shape[-1]
    if abs(di) >= ni or abs(dj) >= nj:
        return out
    src_i = slice(max(0, -di), ni - max(0, di))
    dst_i = slice(max(0, di), ni - max(0, -di))
    src_j = slice(max(0, -dj), nj - max(0, dj))
    dst_j = slice(max(0, dj), nj - max(0, -dj))
    out[..., dst_i, dst_j] = t[..., src_i, src_j]
    return out


def heading_kernel(shift_rad: float, sigma_rad: float, n_heading: int):
    """Wrapped-Gaussian weights over integer heading-bin offsets."""
    binw = 2.0 * math.pi / n_heading
    half = min(
        n_heading // 2,
        int(math.ceil((4.0 * sigma_rad + abs(shift_rad)) / binw)) + 1)
    offsets = np.arange(-half, half + 1)
    sigma = max(sigma_rad, 1e-6)
    weights = np.zeros(offsets.shape)
    for wrap in (-1, 0, 1):
        delta = offsets * binw - shift_rad + wrap * 2.0 * math.pi
        weights += np.exp(-0.5 * (delta / sigma) ** 2)
    return offsets, weights / weights.sum()


def gaussian_taps(sigma_cells: float):
    half = max(1, int(math.ceil(3.0 * sigma_cells)))
    offsets = np.arange(-half, half + 1)
    weights = np.exp(-0.5 * (offsets / max(sigma_cells, 1e-6)) ** 2)
    return offsets, weights / weights.sum()


@dataclasses.dataclass(frozen=True)
class MotionPlan:
    """The discrete linear operations chosen for one odometry increment."""

    heading_offsets: tuple[int, ...] = ()
    heading_weights: tuple[float, ...] = ()
    cell_shifts: tuple[tuple[int, int], ...] = ()
    weighted_cell_shifts: tuple[
        tuple[tuple[int, int, float], ...], ...] = ()
    diffusion_offsets: tuple[int, ...] = ()
    diffusion_weights: tuple[float, ...] = ()


def apply_motion(t: torch.Tensor, plan: MotionPlan) -> torch.Tensor:
    """Apply a recorded forward motion operator."""
    if plan.cell_shifts:
        shifted = t.clone()
        for heading, (di, dj) in enumerate(plan.cell_shifts):
            if di or dj:
                shifted[heading] = shift2d(t[heading], di, dj)
        t = shifted

    if plan.weighted_cell_shifts:
        shifted = torch.zeros_like(t)
        for heading, shifts in enumerate(plan.weighted_cell_shifts):
            for di, dj, weight in shifts:
                shifted[heading] += weight * shift2d(
                    t[heading], di, dj)
        t = shifted

    if plan.heading_offsets:
        acc = torch.zeros_like(t)
        for offset, weight in zip(
                plan.heading_offsets, plan.heading_weights, strict=True):
            acc += weight * torch.roll(t, offset, dims=0)
        t = acc

    if plan.diffusion_offsets:
        for axis in (-2, -1):
            acc = torch.zeros_like(t)
            for offset, weight in zip(
                    plan.diffusion_offsets, plan.diffusion_weights,
                    strict=True):
                acc += weight * shift2d(
                    t,
                    offset if axis == -2 else 0,
                    offset if axis == -1 else 0)
            t = acc
    return t


def apply_motion_transposed(t: torch.Tensor, plan: MotionPlan) -> torch.Tensor:
    """Adjoint of `apply_motion`: the backward (smoothing) message operator.

    Forward is D(H(W(S(x)))) - integer shifts, weighted shifts, heading
    convolution, diffusion - so the adjoint applies D^T, H^T, W^T, S^T in that
    order. A zero-fill shift's adjoint is the opposite shift, a roll's adjoint
    is the opposite roll, and the Gaussian taps are symmetric.
    """
    if plan.diffusion_offsets:
        for axis in (-1, -2):
            acc = torch.zeros_like(t)
            for offset, weight in zip(
                    plan.diffusion_offsets, plan.diffusion_weights,
                    strict=True):
                acc += weight * shift2d(
                    t,
                    -offset if axis == -2 else 0,
                    -offset if axis == -1 else 0)
            t = acc

    if plan.heading_offsets:
        acc = torch.zeros_like(t)
        for offset, weight in zip(
                plan.heading_offsets, plan.heading_weights, strict=True):
            acc += weight * torch.roll(t, -offset, dims=0)
        t = acc

    if plan.weighted_cell_shifts:
        shifted = torch.zeros_like(t)
        for heading, shifts in enumerate(plan.weighted_cell_shifts):
            for di, dj, weight in shifts:
                shifted[heading] += weight * shift2d(
                    t[heading], -di, -dj)
        t = shifted

    if plan.cell_shifts:
        shifted = t.clone()
        for heading, (di, dj) in enumerate(plan.cell_shifts):
            if di or dj:
                shifted[heading] = shift2d(t[heading], -di, -dj)
        t = shifted
    return t


class LikelihoodCache:
    """Host-memory cache of measurement likelihood tensors.

    A likelihood depends only on the measurement, its table and the grid,
    never on the belief, so causal replay (which recomputes each prefix from
    a checkpoint) and the smoother's second forward pass can reuse it. Tensors
    are stored in pinned host memory in their original dtype, so a cached hit
    is bit-identical to a recompute; entries beyond `budget_bytes` are simply
    not cached.
    """

    def __init__(self, budget_bytes: float, device: str):
        self.budget = budget_bytes
        self.device = device
        self.store = {}
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.skipped = 0

    def get(self, key, compute):
        cached = self.store.get(key)
        if cached is not None:
            self.hits += 1
            return cached.to(self.device)
        value = compute()
        self.misses += 1
        size = value.numel() * value.element_size()
        if self.bytes + size <= self.budget:
            self.store[key] = value.detach().to("cpu")
            self.bytes += size
        else:
            self.skipped += 1
        return value

    def describe(self) -> str:
        return (f"likelihood cache: {len(self.store)} tensors, "
                f"{self.bytes / 1e9:.1f} GB, {self.hits} hits, "
                f"{self.misses} misses, {self.skipped} uncached")


def _normalized(t: torch.Tensor) -> torch.Tensor:
    total = t.sum()
    if not bool(torch.isfinite(total)) or not float(total) > 0.0:
        raise ValueError("grid message has non-finite or zero total mass")
    return t / total


class GridBelief:
    """Linear-space belief over (heading, north, east) on one device."""

    def __init__(self, grid: Grid, n_heading: int, device: str):
        self.grid = grid
        self.n_heading = n_heading
        self.device = device
        self.belief = torch.full(
            (n_heading, grid.n_north, grid.n_east),
            1.0 / (n_heading * grid.n_north * grid.n_east),
            dtype=torch.float32,
            device=device)
        east, north = grid.centers()
        jj, ii = np.meshgrid(east, north)
        self.cell_east = torch.tensor(
            jj.ravel(), dtype=torch.float32, device=device)
        self.cell_north = torch.tensor(
            ii.ravel(), dtype=torch.float32, device=device)
        self.bin_rad = 2.0 * math.pi * np.arange(n_heading) / n_heading
        self.pending_yaw = 0.0
        self.pending_yaw_var = 0.0
        self.pending_de = np.zeros(n_heading)
        self.pending_dn = np.zeros(n_heading)
        self.pending_pos_var = 0.0
        self.leaked_mass = 0.0

    def plan_motion(self, delta, yaw_sigma_scale: float,
                    heading_rw_rad: float, diffusion_m: float) -> MotionPlan:
        """Advance sub-cell accumulators and record the operations triggered."""
        binw = 2.0 * math.pi / self.n_heading
        sigma_h = math.hypot(
            delta.sigma_yaw_rad * yaw_sigma_scale, heading_rw_rad)
        self.pending_yaw += delta.delta_yaw_cw_rad
        self.pending_yaw_var += sigma_h * sigma_h
        heading_offsets = ()
        heading_weights = ()
        heading_convolution = (
            math.sqrt(self.pending_yaw_var) >= 0.35 * binw)
        heading_steps = 0
        if heading_convolution:
            offsets, weights = heading_kernel(
                self.pending_yaw, math.sqrt(self.pending_yaw_var),
                self.n_heading)
            heading_offsets = tuple(int(value) for value in offsets)
            heading_weights = tuple(float(value) for value in weights)
        elif abs(self.pending_yaw) >= 0.5 * binw:
            heading_steps = int(round(self.pending_yaw / binw))
            heading_offsets = (heading_steps,)
            heading_weights = (1.0,)

        # Accumulate translation in the source bin at its continuous post-yaw
        # heading.  Spatial motion is materialized before relabeling/mixing the
        # heading bins, which keeps its source-heading history intact.
        heading_rad = self.bin_rad + self.pending_yaw
        self.pending_de += (
            delta.forward_m * np.sin(heading_rad)
            - delta.left_m * np.cos(heading_rad))
        self.pending_dn += (
            delta.forward_m * np.cos(heading_rad)
            + delta.left_m * np.sin(heading_rad))
        cell = self.grid.cell_m
        shifts = ()
        weighted_shifts = ()
        if heading_convolution:
            materialized = []
            for heading in range(self.n_heading):
                east_cells = self.pending_de[heading] / cell
                north_cells = self.pending_dn[heading] / cell
                east_lo = math.floor(east_cells)
                north_lo = math.floor(north_cells)
                east_fraction = east_cells - east_lo
                north_fraction = north_cells - north_lo
                heading_shifts = []
                for di, north_weight in (
                        (north_lo, 1.0 - north_fraction),
                        (north_lo + 1, north_fraction)):
                    for dj, east_weight in (
                            (east_lo, 1.0 - east_fraction),
                            (east_lo + 1, east_fraction)):
                        weight = north_weight * east_weight
                        if weight > 0.0:
                            heading_shifts.append((di, dj, weight))
                materialized.append(tuple(heading_shifts))
            weighted_shifts = tuple(materialized)
            self.pending_de.fill(0.0)
            self.pending_dn.fill(0.0)
            self.pending_yaw = 0.0
            self.pending_yaw_var = 0.0
        else:
            integer_shifts = []
            for heading in range(self.n_heading):
                dj = int(round(self.pending_de[heading] / cell))
                di = int(round(self.pending_dn[heading] / cell))
                integer_shifts.append((di, dj))
                self.pending_de[heading] -= dj * cell
                self.pending_dn[heading] -= di * cell
            shifts = tuple(integer_shifts)
            if heading_steps:
                self.pending_de = np.roll(self.pending_de, heading_steps)
                self.pending_dn = np.roll(self.pending_dn, heading_steps)
                self.pending_yaw -= heading_steps * binw

        self.pending_pos_var += delta.sigma_m ** 2 + diffusion_m ** 2
        diffusion_offsets = ()
        diffusion_weights = ()
        if math.sqrt(self.pending_pos_var) >= 0.25 * cell:
            offsets, weights = gaussian_taps(
                math.sqrt(self.pending_pos_var) / cell)
            diffusion_offsets = tuple(int(value) for value in offsets)
            diffusion_weights = tuple(float(value) for value in weights)
            self.pending_pos_var = 0.0

        return MotionPlan(
            heading_offsets=heading_offsets,
            heading_weights=heading_weights,
            cell_shifts=shifts,
            weighted_cell_shifts=weighted_shifts,
            diffusion_offsets=diffusion_offsets,
            diffusion_weights=diffusion_weights)

    def motion(self, delta, yaw_sigma_scale: float, heading_rw_rad: float,
               diffusion_m: float) -> MotionPlan:
        plan = self.plan_motion(
            delta, yaw_sigma_scale, heading_rw_rad, diffusion_m)
        self.belief = apply_motion(self.belief, plan)
        return plan

    def track_likelihood(self, observed_rad: float, base_var: float,
                         cand_east: torch.Tensor,
                         cand_north: torch.Tensor,
                         cand_weight: torch.Tensor,
                         sigma_pos: float,
                         pi0: float,
                         tail_mass: float,
                         *,
                         quantization_comp: bool = True,
                         chunk: int = 256,
                         range_max_m=None,
                         range_softness: float = 0.25,
                         range_floor: float = 0.0) -> torch.Tensor:
        """PF-equivalent independent-epoch mixture likelihood on the grid.

        ``range_floor`` keeps that fraction of a candidate's weight beyond
        the range cap (a robust gate: the extractor's bucket may be wrong).
        """
        n_cells = self.cell_east.shape[0]
        two_pi = 2.0 * math.pi
        like = torch.zeros(
            (self.n_heading, n_cells), dtype=torch.float32,
            device=self.device)
        binw = two_pi / self.n_heading
        for start in range(0, cand_east.shape[0], chunk):
            sl = slice(start, min(start + chunk, cand_east.shape[0]))
            d_east = cand_east[sl][None, :] - self.cell_east[:, None]
            d_north = cand_north[sl][None, :] - self.cell_north[:, None]
            distance = torch.sqrt(d_east * d_east + d_north * d_north)
            world_bearing = torch.atan2(d_east, d_north)
            safe_distance = torch.clamp(distance, min=1.0)
            gate = cand_weight[sl][None, :].expand_as(distance)
            if range_max_m is not None:
                excess = torch.clamp(distance - range_max_m, min=0.0)
                soft = torch.exp(-0.5 * torch.square(
                    excess / (range_softness * range_max_m)))
                gate = gate * ((1.0 - range_floor) * soft + range_floor)
            quant_var = 0.0
            if quantization_comp:
                quant_var = (
                    binw * binw / 12.0
                    + (self.grid.cell_m / math.sqrt(12.0)
                       / safe_distance) ** 2)
            variance = (
                base_var + (sigma_pos / safe_distance) ** 2 + quant_var)
            kappa_eff = 1.0 / variance
            log_denom = torch.log(torch.special.i0e(kappa_eff))
            cos_world = torch.cos(world_bearing)
            sin_world = torch.sin(world_bearing)
            for heading in range(self.n_heading):
                angle = self.bin_rad[heading] + observed_rad
                cos_delta = (
                    cos_world * math.cos(angle)
                    + sin_world * math.sin(angle))
                exponent = -log_denom + kappa_eff * (cos_delta - 1.0)
                contrib = torch.exp(exponent) * gate
                like[heading] += contrib.sum(dim=1)
        # Scaled by 2*pi relative to a normalized bearing density; the common
        # factor cancels in every posterior normalization.
        like = pi0 + (1.0 - pi0) * (like + tail_mass)
        return like.view(
            self.n_heading, self.grid.n_north, self.grid.n_east)

    def track_joint_likelihood(self, epochs, cand_east, cand_north,
                               cand_weight, sigma_pos, pi0, tail_mass, *,
                               quantization_comp=True, chunk=64,
                               range_softness=0.25, range_floor=0.0,
                               temper=1.0, cap=None):
        """Whole-track mixture likelihood evaluated at the release pose.

        ``epochs`` is a list of (d_forward_m, d_left_m, d_heading_rad,
        observed_rad, base_var, range_max_m): the epoch pose expressed in the
        release pose's body frame plus that epoch's bearing.  One identity
        per track: sum_j w_j prod_e vM_e(j), so a wrong identity is one vote,
        not one per epoch.  ``temper`` scales every epoch's log-likelihood.
        """
        n_cells = self.cell_east.shape[0]
        binw = 2.0 * math.pi / self.n_heading
        mix = torch.zeros(
            (self.n_heading, n_cells), dtype=torch.float32,
            device=self.device)
        for heading in range(self.n_heading):
            h = self.bin_rad[heading]
            for start in range(0, cand_east.shape[0], chunk):
                sl = slice(start, min(start + chunk, cand_east.shape[0]))
                log_prod = torch.zeros(
                    (n_cells, sl.stop - sl.start), dtype=torch.float32,
                    device=self.device)
                for (d_fwd, d_left, d_head, observed_rad, base_var,
                     range_max_m, head_slack_var, pos_slack_m) in epochs:
                    pose_east = self.cell_east + (
                        d_fwd * math.sin(h) - d_left * math.cos(h))
                    pose_north = self.cell_north + (
                        d_fwd * math.cos(h) + d_left * math.sin(h))
                    d_east = cand_east[sl][None, :] - pose_east[:, None]
                    d_north = cand_north[sl][None, :] - pose_north[:, None]
                    distance = torch.sqrt(d_east * d_east + d_north * d_north)
                    safe_distance = torch.clamp(distance, min=1.0)
                    world_bearing = torch.atan2(d_east, d_north)
                    quant_var = 0.0
                    if quantization_comp:
                        quant_var = (
                            binw * binw / 12.0
                            + (self.grid.cell_m / math.sqrt(12.0)
                               / safe_distance) ** 2)
                    variance = (
                        base_var + (sigma_pos / safe_distance) ** 2
                        + quant_var + head_slack_var
                        + (pos_slack_m / safe_distance) ** 2)
                    kappa_eff = 1.0 / variance
                    angle = h + d_head + observed_rad
                    cos_delta = torch.cos(world_bearing - angle)
                    term = (-torch.log(torch.special.i0e(kappa_eff))
                            + kappa_eff * (cos_delta - 1.0))
                    if range_max_m is not None:
                        excess = torch.clamp(distance - range_max_m, min=0.0)
                        soft = torch.exp(-0.5 * torch.square(
                            excess / (range_softness * range_max_m)))
                        term = term + torch.log(
                            (1.0 - range_floor) * soft + range_floor + 1e-30)
                    log_prod += temper * term
                contrib = (torch.exp(torch.clamp(log_prod, max=80.0))
                           * cand_weight[sl][None, :])
                mix[heading] += contrib.sum(dim=1)
        if cap is not None:
            # robustness: one track can never be worth more than
            # (pi0 + (1-pi0)(cap + tail)) / floor odds, whatever its epochs say
            mix = torch.clamp(mix, max=cap)
        like = pi0 + (1.0 - pi0) * (mix + tail_mass)
        return like.view(
            self.n_heading, self.grid.n_north, self.grid.n_east)

    def renormalize(self):
        total = self.belief.sum()
        self.leaked_mass = 1.0 - float(total)
        self.belief = _normalized(self.belief)

    def position_marginal(self) -> torch.Tensor:
        return self.belief.sum(dim=0).reshape(-1)


def relative_epoch_poses(odometry, anchors, release_keyframe):
    """Dead-reckon each anchor pose into the release keyframe's body frame.

    Returns {anchor: (d_forward_m, d_left_m, d_heading_rad)} such that the
    anchor pose = release pose composed with that body-frame offset.
    """
    first = min(anchors)
    yaw = 0.0
    east = 0.0
    north = 0.0
    poses = {first: (0.0, 0.0, 0.0)}
    for keyframe in range(first + 1, release_keyframe + 1):
        delta = odometry[keyframe]
        yaw += delta.delta_yaw_cw_rad
        east += (delta.forward_m * math.sin(yaw)
                 - delta.left_m * math.cos(yaw))
        north += (delta.forward_m * math.cos(yaw)
                  + delta.left_m * math.sin(yaw))
        poses[keyframe] = (east, north, yaw)
    r_east, r_north, r_yaw = poses[release_keyframe]
    out = {}
    for anchor in anchors:
        a_east, a_north, a_yaw = poses[anchor]
        d_east, d_north = a_east - r_east, a_north - r_north
        # world -> release body frame (forward along r_yaw, left = +90 ccw)
        d_fwd = d_east * math.sin(r_yaw) + d_north * math.cos(r_yaw)
        d_left = -d_east * math.cos(r_yaw) + d_north * math.sin(r_yaw)
        out[anchor] = (d_fwd, d_left, a_yaw - r_yaw)
    return out


def truth_masks(grid: Grid, truth, radii, subgrid: int = 8):
    """Area-weighted truth-radius masks for coarse grid cells."""
    east, north = grid.centers()
    sub = (np.arange(subgrid) + 0.5) / subgrid - 0.5
    sub_e, sub_n = np.meshgrid(sub * grid.cell_m, sub * grid.cell_m)
    masks = {}
    for pose in truth:
        for radius in radii:
            reach = radius + 0.75 * grid.cell_m
            east_idx = np.nonzero(np.abs(east - pose.east_m) <= reach)[0]
            north_idx = np.nonzero(np.abs(north - pose.north_m) <= reach)[0]
            idx, fraction = [], []
            for i in north_idx:
                for j in east_idx:
                    de = east[j] + sub_e - pose.east_m
                    dn = north[i] + sub_n - pose.north_m
                    inside = float(
                        (de * de + dn * dn <= radius * radius).mean())
                    if inside > 0.0:
                        idx.append(i * grid.n_east + j)
                        fraction.append(inside)
            masks[(pose.keyframe_idx, radius)] = (
                np.asarray(idx, dtype=np.int64),
                np.asarray(fraction, dtype=np.float32))
    return masks


def _score_message(message, keyframe, masks, truth_by_kf, grid, device):
    marginal = message.sum(dim=0).reshape(-1)
    mass = {}
    for radius in RADII_M:
        idx, fraction = masks[(keyframe, radius)]
        if idx.size:
            cells = marginal[torch.as_tensor(idx, device=device)]
            mass[radius] = float(
                (cells * torch.as_tensor(fraction, device=device)).sum())
        else:
            mass[radius] = 0.0
    state = _online_map_state(message, marginal, grid)
    pose = truth_by_kf[keyframe]
    map_error = math.hypot(
        state["east_m"] - pose.east_m,
        state["north_m"] - pose.north_m)
    return mass, map_error, state


def _online_map_state(message, position_marginal, grid):
    """Position-marginal MAP with conditional heading at that grid cell."""
    cell_idx = int(torch.argmax(position_marginal))
    north_idx, east_idx = divmod(cell_idx, grid.n_east)
    heading_idx = int(torch.argmax(message[:, north_idx, east_idx]))
    east, north = grid.centers()
    return {
        "east_m": float(east[east_idx]),
        "north_m": float(north[north_idx]),
        "heading_world_cw_deg": 360.0 * heading_idx / message.shape[0],
    }


def _summary(series, map_errors, truth_by_kf, n_keyframes):
    distance = np.zeros(n_keyframes)
    for keyframe in range(1, n_keyframes):
        previous = truth_by_kf[keyframe - 1]
        current = truth_by_kf[keyframe]
        distance[keyframe] = distance[keyframe - 1] + math.hypot(
            current.east_m - previous.east_m,
            current.north_m - previous.north_m)
    result = {}
    for radius in RADII_M:
        values = np.asarray(series[radius])
        result[f"tn_mass_{radius:g}"] = float(
            np.sum(0.5 * (values[:-1] + values[1:]))
            / (n_keyframes - 1))
        result[f"dn_mass_{radius:g}"] = float(
            np.sum(
                0.5 * (values[:-1] + values[1:]) * np.diff(distance))
            / max(distance[-1], 1e-9))
    result["final_map_error_m"] = float(map_errors[-1])
    return result


def causal_replay(initial_message, motion_plans, releases,
                  apply_measurements, score, checkpoint_keyframes):
    """Replay delayed whole-track factors and score only the current belief.

    A release makes all of one track's historical measurements available at
    once.  The current prefix is recomputed from the last unaffected
    checkpoint; scores already published for earlier keyframes are untouched.
    """
    if checkpoint_keyframes <= 0:
        raise ValueError("checkpoint_keyframes must be positive")
    n_keyframes = len(motion_plans)
    if n_keyframes == 0 or motion_plans[0] is not None:
        raise ValueError("motion_plans must start with the keyframe-0 sentinel")

    release_by_keyframe = {}
    previous = (-1, "")
    for release in releases:
        key = (release.release_keyframe_idx, release.tracklet_id)
        if key <= previous:
            raise ValueError("track releases must be strictly sorted")
        previous = key
        if not 0 <= release.release_keyframe_idx < n_keyframes:
            raise ValueError("track release is outside the trajectory")
        if not release.measurements:
            raise ValueError("track release has no measurements")
        if any(not 0 <= item.anchor_keyframe_idx
               <= release.release_keyframe_idx
               for item in release.measurements):
            raise ValueError("track release contains an invalid anchor")
        release_by_keyframe.setdefault(
            release.release_keyframe_idx, []).append(release)

    device = initial_message.device
    available_by_anchor = {}
    checkpoints = {}
    current = initial_message
    scores = []
    replay_steps = 0
    max_rollback = 0
    released_tracks = 0

    def advance(message, start, stop):
        nonlocal replay_steps
        for keyframe in range(start, stop + 1):
            if keyframe > 0:
                message = apply_motion(message, motion_plans[keyframe])
            message = apply_measurements(
                message, keyframe,
                available_by_anchor.get(keyframe, ()))
            message = _normalized(message)
            if keyframe % checkpoint_keyframes == 0:
                checkpoints[keyframe] = message.detach().cpu().clone()
            replay_steps += 1
        return message

    for keyframe in range(n_keyframes):
        if keyframe == 0:
            current = apply_measurements(
                current, keyframe,
                available_by_anchor.get(keyframe, ()))
        else:
            current = apply_motion(current, motion_plans[keyframe])
            current = apply_measurements(
                current, keyframe,
                available_by_anchor.get(keyframe, ()))
        current = _normalized(current)
        if keyframe % checkpoint_keyframes == 0:
            checkpoints[keyframe] = current.detach().cpu().clone()

        arriving = release_by_keyframe.get(keyframe, ())
        if arriving:
            earliest_anchor = keyframe
            for release in arriving:
                released_tracks += 1
                for measurement in release.measurements:
                    earliest_anchor = min(
                        earliest_anchor, measurement.anchor_keyframe_idx)
                    available_by_anchor.setdefault(
                        measurement.anchor_keyframe_idx, []).append(measurement)
            for measurements in available_by_anchor.values():
                measurements.sort(key=lambda item: item.tracklet_id)

            candidates = [saved for saved in checkpoints
                          if saved < earliest_anchor]
            checkpoint = max(candidates, default=-1)
            start = checkpoint + 1
            max_rollback = max(max_rollback, keyframe - start + 1)
            current = (initial_message.clone() if checkpoint < 0 else
                       checkpoints[checkpoint].to(device).clone())
            current = advance(current, start, keyframe)
        scores.append(score(current, keyframe))

    return current, scores, {
        "n_track_releases": released_tracks,
        "n_release_keyframes": len(release_by_keyframe),
        "n_replay_keyframe_steps": replay_steps,
        "max_rollback_keyframes": max_rollback,
        "checkpoint_keyframes": checkpoint_keyframes,
    }


def _top_modes(message, grid, limit, position_nms_m, heading_nms_deg):
    """Highest-mass final grid states after greedy SE(2) non-max suppression."""
    if limit <= 0:
        return []
    flat = message.reshape(-1)
    pool = min(flat.numel(), max(4096, 16 * limit))
    heading_step = 2.0 * math.pi / message.shape[0]
    east_centers, north_centers = grid.centers()
    chosen = []
    while True:
        values, indices = torch.topk(flat, pool, sorted=True)
        values = values.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()
        chosen = []
        chosen_east = []
        chosen_north = []
        chosen_heading = []
        cells_per_heading = grid.n_north * grid.n_east
        for value, index in zip(values, indices, strict=True):
            heading_idx, cell_idx = divmod(int(index), cells_per_heading)
            north_idx, east_idx = divmod(cell_idx, grid.n_east)
            east = float(east_centers[east_idx])
            north = float(north_centers[north_idx])
            heading = heading_idx * heading_step
            if chosen:
                distance = np.hypot(
                    np.asarray(chosen_east) - east,
                    np.asarray(chosen_north) - north)
                angle = np.abs(
                    (np.asarray(chosen_heading) - heading + math.pi)
                    % (2.0 * math.pi) - math.pi)
                if np.any(
                        (distance <= position_nms_m)
                        & (angle <= math.radians(heading_nms_deg))):
                    continue
            chosen_east.append(east)
            chosen_north.append(north)
            chosen_heading.append(heading)
            chosen.append({
                "source_rank": len(chosen) + 1,
                "source_probability": float(value),
                "east_m": east,
                "north_m": north,
                "heading_world_cw_deg": math.degrees(heading),
                "heading_index": heading_idx,
                "north_index": north_idx,
                "east_index": east_idx,
            })
            if len(chosen) == limit:
                return chosen
        if pool == flat.numel():
            return chosen
        pool = min(flat.numel(), 4 * pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--odometry_profile", choices=odometry_profiles.PROFILE_CHOICES,
        default="recorded")
    parser.add_argument("--odometry_seed", type=int, default=0)
    parser.add_argument(
        "--availability", choices=("natural",), default="natural",
        help="audited track measurements become visible when the track "
             "closes (its release keyframe); the only supported mode")
    parser.add_argument(
        "--release_schedule", required=True,
        help="bound natural-closure sidecar "
             "(localization:build_release_schedule)")
    parser.add_argument("--cell_m", type=float, default=200.0)
    parser.add_argument("--n_heading", type=int, default=18)
    parser.add_argument("--yaw_sigma_scale", type=float, default=1.0)
    parser.add_argument("--heading_rw_deg", type=float, default=1.0)
    parser.add_argument("--diffusion_m", type=float, default=5.0)
    parser.add_argument("--pi0", type=float, default=None)
    parser.add_argument("--matcher_recall", type=float, default=None)
    parser.add_argument(
        "--tables_override", default=None,
        help="JSON list of CompatibilityTable documents (compatibility.json "
             "schema, e.g. from matching:reaggregate_tables) replacing the "
             "export's tables for the tracklets they name: an alternative "
             "aggregation of the same matcher responses, not privileged")
    parser.add_argument(
        "--track_joint", type=int, default=0,
        help="natural mode: one whole-track factor per release at the "
             "release keyframe (no replay) instead of independent epochs")
    parser.add_argument(
        "--joint_slack", type=int, default=0,
        help="joint factor: add the motion model's heading/position slack "
             "accumulated between each anchor and the release keyframe")
    parser.add_argument(
        "--joint_cap", type=float, default=None,
        help="joint factor: clamp the per-track mixture term at this value")
    parser.add_argument(
        "--joint_temper", type=float, default=1.0,
        help="per-epoch log-likelihood scale inside the joint factor")
    parser.add_argument(
        "--range_floor", type=float, default=0.0,
        help="fraction of candidate weight kept beyond the range cap")
    parser.add_argument(
        "--likelihood_cache_gb", type=float, default=12.0,
        help="pinned host memory for caching per-measurement likelihood "
             "tensors across causal replays and the smoother pass "
             "(bit-identical results; zero disables)")
    parser.add_argument(
        "--smoother", choices=("none", "fixed_interval"), default="none",
        help="after the causal run, also compute the fixed-interval smoothed "
             "trajectory (forward-backward over the same grid HMM with every "
             "factor at its anchor, or at its release keyframe in joint mode) "
             "and score it alongside; this is what an incremental smoother "
             "reports once the run has ended")
    parser.add_argument(
        "--smooth_lag", type=int, default=0,
        help="with --smoother: also score the fixed-lag estimate of pose k "
             "available at keyframe k+lag (online smoothing with a latency "
             "of lag keyframes); zero skips it")
    parser.add_argument(
        "--smooth_lags", default="",
        help="comma-separated extra lags scored in the same backward chain "
             "(the online lag curve); the largest sets the alpha window")
    parser.add_argument("--kappa_scale", type=float, default=1.0)
    parser.add_argument("--quantization_comp", type=int, default=1)
    parser.add_argument(
        "--tail", choices=("exact", "uniform"), default="exact")
    parser.add_argument("--range_cap", type=int, default=1)
    parser.add_argument("--range_softness", type=float, default=0.25)
    parser.add_argument("--margin_m", type=float, default=1000.0)
    parser.add_argument("--checkpoint_keyframes", type=int, default=32)
    parser.add_argument("--top_modes", type=int, default=400)
    parser.add_argument("--top_mode_position_nms_m", type=float, default=200.0)
    parser.add_argument("--top_mode_heading_nms_deg", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.checkpoint_keyframes <= 0:
        parser.error("--checkpoint_keyframes must be positive")
    if args.top_modes < 0:
        parser.error("--top_modes must be nonnegative")
    if args.top_mode_position_nms_m < 0.0 \
            or args.top_mode_heading_nms_deg < 0.0:
        parser.error("top-mode NMS radii must be nonnegative")
    if (args.smooth_lag > 0 or args.smooth_lags.strip()) \
            and not args.track_joint:
        parser.error(
            "--smooth_lag/--smooth_lags need --track_joint 1: the "
            "independent-epoch smoother places each epoch at its anchor "
            "before its track has closed, so a fixed-lag estimate would use "
            "tracks not yet released at k+lag and is not an online result")

    data = export_ingest.load(Path(args.input_dir))
    if len(data.truth) != data.n_keyframes:
        raise ValueError("grid localization requires truth at every keyframe")
    data.odometry, odometry_profile = odometry_profiles.derive(
        Path(args.input_dir), data, args.odometry_profile,
        noise_seed=args.odometry_seed)
    n_keyframes = len(data.truth)
    catalog = data.catalog
    sigma_pos = float(catalog.position_sigma_m[0])

    region = export_ingest.prior_box(data, args.margin_m)
    declared_box = (
        region.east_min_m, region.east_max_m,
        region.north_min_m, region.north_max_m)
    box = declared_box
    if args.pi0 is None:
        args.pi0 = 0.2
    if args.matcher_recall is None:
        args.matcher_recall = 0.5

    grid = Grid(*box, args.cell_m)
    belief = GridBelief(grid, args.n_heading, args.device)
    initial_belief = belief.belief.clone()
    n_states = args.n_heading * grid.n_north * grid.n_east
    print(
        f"grid: {grid.n_east} x {grid.n_north} x {args.n_heading} = "
        f"{n_states / 1e6:.1f}M states, cell {grid.cell_m:g} m; "
        f"pi0={args.pi0} recall={args.matcher_recall} "
        f"yaw_scale={args.yaw_sigma_scale} rw={args.heading_rw_deg} deg "
        f"tail={args.tail}")

    catalog_east = torch.tensor(
        catalog.east_m, dtype=torch.float32, device=args.device)
    catalog_north = torch.tensor(
        catalog.north_m, dtype=torch.float32, device=args.device)
    weight_cache = {}

    if args.tables_override:
        override_tables = msgspec.json.decode(
            Path(args.tables_override).read_bytes(),
            type=list[structs.CompatibilityTable])
        by_suffix = {t.tracklet_id.split("#")[-1]: t for t in override_tables}
        replaced = 0
        for tracklet_id in list(data.tables):
            forced_table = by_suffix.get(tracklet_id.split("#")[-1])
            if forced_table is not None:
                data.tables[tracklet_id] = msgspec.structs.replace(
                    forced_table, tracklet_id=tracklet_id)
                replaced += 1
        print(f"tables override: replaced {replaced} of "
              f"{len(data.tables)} tables from {args.tables_override}")

    def candidate_set(tracklet_id):
        if tracklet_id not in weight_cache:
            table = data.tables[tracklet_id]
            log_weight = filter_lib._identity_log_weights(  # noqa: SLF001
                table, catalog, args.matcher_recall)
            weights = np.exp(log_weight)
            if args.tail == "exact":
                idx = np.arange(catalog.n)
                tail_mass = 0.0
            else:
                endorsed = ~filter_lib._surprise_mask(  # noqa: SLF001
                    table,
                    filter_lib._clipped_log_lr(table, catalog))  # noqa: SLF001
                idx = np.nonzero(endorsed)[0]
                tail_mass = float(weights[~endorsed].sum())
            idx_tensor = torch.as_tensor(idx, device=args.device)
            weight_cache[tracklet_id] = (
                catalog_east[idx_tensor],
                catalog_north[idx_tensor],
                torch.tensor(
                    weights[idx], dtype=torch.float32, device=args.device),
                tail_mass)
        return weight_cache[tracklet_id]

    by_keyframe = {}
    for measurement in data.measurements:
        by_keyframe.setdefault(
            measurement.anchor_keyframe_idx, []).append(measurement)
    odometry = {item.keyframe_idx: item for item in data.odometry}

    def apply_likelihoods(message, keyframe, measurements=None):
        if measurements is None:
            measurements = by_keyframe.get(keyframe, ())
        for measurement in measurements:
            message = _normalized(
                message * epoch_likelihood(measurement, belief))
        return message

    likelihood_cache = LikelihoodCache(
        args.likelihood_cache_gb * 1e9, args.device)

    def epoch_likelihood(measurement, grid_belief):
        """One fused epoch's likelihood over the grid (cached)."""
        def compute():
            (candidate_east, candidate_north, candidate_weight,
             tail_mass) = candidate_set(measurement.tracklet_id)
            kappa = min(
                float(measurement.kappa) * args.kappa_scale, MAX_KAPPA)
            return grid_belief.track_likelihood(
                math.radians(measurement.bearing_forward_cw_deg),
                1.0 / kappa,
                candidate_east,
                candidate_north,
                candidate_weight,
                sigma_pos,
                args.pi0,
                tail_mass,
                quantization_comp=bool(args.quantization_comp),
                range_max_m=(measurement.range_max_m
                             if args.range_cap else None),
                range_softness=args.range_softness,
                range_floor=args.range_floor)
        return likelihood_cache.get(
            ("epoch", measurement.tracklet_id,
             measurement.anchor_keyframe_idx), compute)

    masks = truth_masks(grid, data.truth, RADII_M)
    truth_by_kf = {pose.keyframe_idx: pose for pose in data.truth}
    heading_rw_rad = math.radians(args.heading_rw_deg)
    smoothing_releases = None
    joint_release_likelihood = None

    def keyframe_likelihood(keyframe, grid_belief):
        """Product of every factor anchored at `keyframe`, or None."""
        factor = None
        if args.track_joint:
            releases_at = {}
            for release in smoothing_releases or ():
                releases_at.setdefault(
                    release.release_keyframe_idx, []).append(release)
            factors = [joint_release_likelihood(r, keyframe, grid_belief)
                       for r in releases_at.get(keyframe, ())]
        else:
            factors = [epoch_likelihood(measurement, grid_belief)
                       for measurement in by_keyframe.get(keyframe, ())]
        for likelihood in factors:
            factor = likelihood if factor is None else factor * likelihood
        return factor

    def smoothing_pass():
        """Fixed-interval (and optional fixed-lag) smoothing on the grid HMM.

        Forward: every factor at its anchor (independent epochs) or at its
        release keyframe (joint mode). Backward: beta_{k-1} = M_k^T (L_k *
        beta_k); smoothed_k ~ alpha_k * beta_k. Memory is bounded by
        checkpointing alpha every `stride` keyframes and recomputing each
        segment during the backward sweep; factors come from the likelihood
        cache. The fixed-lag estimate for pose k is computed online at
        keyframe k + lag from a rolling window, exactly as a causal
        incremental smoother would report it.
        """
        if args.smoother == "none":
            return None
        started = time.time()
        device = args.device
        stride = max(1, args.checkpoint_keyframes)
        lags = sorted({int(v) for v in args.smooth_lags.split(",") if v.strip()}
                      | ({args.smooth_lag} if args.smooth_lag > 0 else set()))
        lags = [v for v in lags if v > 0]
        lag = max(lags) if lags else 0
        planner = GridBelief(grid, args.n_heading, device)
        planner.belief = initial_belief.clone()
        plans = [None] * n_keyframes
        checkpoints = {}
        window = {}  # keyframe -> alpha on host, only the last `lag` + 1
        lag_series = {v: {radius: [None] * n_keyframes for radius in RADII_M}
                      for v in lags}
        lag_error = {v: [None] * n_keyframes for v in lags}
        lag_states = {v: [None] * n_keyframes for v in lags}
        message = planner.belief

        def backward_step(beta, keyframe):
            factor = keyframe_likelihood(keyframe, planner)
            incoming = beta if factor is None else beta * factor
            return _normalized(
                apply_motion_transposed(incoming, plans[keyframe]))

        for keyframe in range(n_keyframes):
            if keyframe > 0:
                plans[keyframe] = planner.plan_motion(
                    odometry[keyframe], args.yaw_sigma_scale,
                    heading_rw_rad, args.diffusion_m)
                message = apply_motion(message, plans[keyframe])
            factor = keyframe_likelihood(keyframe, planner)
            if factor is not None:
                message = message * factor
            message = _normalized(message)
            if keyframe % stride == 0:
                checkpoints[keyframe] = message.detach().cpu()
            if lag > 0:
                window[keyframe] = message.detach().cpu()
                # one backward chain from this keyframe serves every lag:
                # after `v` adjoint steps it is beta for pose keyframe - v
                beta = torch.ones_like(message)
                for steps in range(1, lag + 1):
                    target = keyframe - steps
                    if target < 0:
                        break
                    beta = backward_step(beta, target + 1)
                    if steps in lag_series:
                        smoothed = _normalized(
                            window[target].to(device) * beta)
                        mass, map_error, state = _score_message(
                            smoothed, target, masks, truth_by_kf, grid, device)
                        for radius in RADII_M:
                            lag_series[steps][radius][target] = mass[radius]
                        lag_error[steps][target] = map_error
                        lag_states[steps][target] = state
                window.pop(keyframe - lag, None)
            if keyframe % 50 == 0:
                print(f"smoother forward kf {keyframe:4d} "
                      f"({time.time() - started:.0f}s)")
        forward_seconds = time.time() - started

        smoothed_series = {radius: [0.0] * n_keyframes for radius in RADII_M}
        smoothed_error = [0.0] * n_keyframes
        smoothed_states = [None] * n_keyframes
        beta = torch.ones_like(message)
        segment_starts = sorted(checkpoints, reverse=True)
        for seg_start in segment_starts:
            seg_end = min(seg_start + stride, n_keyframes)
            # recompute this segment's alphas from its checkpoint
            alphas = []
            current = checkpoints[seg_start].to(device)
            alphas.append(current)
            for keyframe in range(seg_start + 1, seg_end):
                current = apply_motion(current, plans[keyframe])
                factor = keyframe_likelihood(keyframe, planner)
                if factor is not None:
                    current = current * factor
                current = _normalized(current)
                alphas.append(current)
            for keyframe in range(seg_end - 1, seg_start - 1, -1):
                smoothed = _normalized(alphas[keyframe - seg_start] * beta)
                mass, map_error, state = _score_message(
                    smoothed, keyframe, masks, truth_by_kf, grid, device)
                for radius in RADII_M:
                    smoothed_series[radius][keyframe] = mass[radius]
                smoothed_error[keyframe] = map_error
                smoothed_states[keyframe] = state
                if keyframe > 0:
                    beta = backward_step(beta, keyframe)
            del alphas
            if seg_start % (stride * 10) == 0:
                print(f"smoother backward kf {seg_start:4d} "
                      f"({time.time() - started:.0f}s)")
        result = {
            "method": ("fixed_interval_forward_backward_on_grid_hmm_"
                       + ("joint_release_factors" if args.track_joint
                          else "independent_epoch_factors_at_anchors")),
            "checkpoint_stride": stride,
            "summary": _summary(smoothed_series, smoothed_error,
                                truth_by_kf, n_keyframes),
            "mass_by_keyframe": {
                f"{radius:g}": smoothed_series[radius] for radius in RADII_M},
            "map_error_m_by_keyframe": smoothed_error,
            "map_state_by_keyframe": smoothed_states,
        }
        print("smoothed:", {key: round(value, 4)
                            for key, value in result["summary"].items()})
        for v in lags:
            # poses inside the last `v` keyframes carry the fixed-interval
            # value, which is what is available once the run has ended
            for radius in RADII_M:
                for keyframe in range(n_keyframes):
                    if lag_series[v][radius][keyframe] is None:
                        lag_series[v][radius][keyframe] = smoothed_series[
                            radius][keyframe]
            for keyframe in range(n_keyframes):
                if lag_error[v][keyframe] is None:
                    lag_error[v][keyframe] = smoothed_error[keyframe]
                if lag_states[v][keyframe] is None:
                    lag_states[v][keyframe] = smoothed_states[keyframe]
            entry = {
                "lag_keyframes": v,
                "computed": "online_at_keyframe_k_plus_lag",
                "tail_semantics": ("the last `lag` poses carry the "
                                   "fixed-interval value available at the end "
                                   "of the run"),
                "summary": _summary(lag_series[v], lag_error[v], truth_by_kf,
                                    n_keyframes),
                "mass_by_keyframe": {
                    f"{radius:g}": lag_series[v][radius] for radius in RADII_M},
                "map_error_m_by_keyframe": lag_error[v],
                "map_state_by_keyframe": lag_states[v],
            }
            result.setdefault("fixed_lags", {})[str(v)] = entry
            if v == (args.smooth_lag if args.smooth_lag > 0 else lags[0]):
                result["fixed_lag"] = entry
            print(f"fixed-lag {v}:",
                  {key: round(value, 4)
                   for key, value in entry["summary"].items()})
        result["runtime_seconds"] = {
            "forward_with_fixed_lag": forward_seconds,
            "total": time.time() - started}
        print(likelihood_cache.describe())
        return result

    def finish_payload(payload):
        smoothing = smoothing_pass()
        if smoothing is not None:
            payload["smoothing"] = smoothing
            for key, value in smoothing["summary"].items():
                payload["summary"][f"sm_{key}"] = value
            for v, entry in smoothing.get("fixed_lags", {}).items():
                for key, value in entry["summary"].items():
                    payload["summary"][f"lag{v}_{key}"] = value
        return payload
    filtered_series = {radius: [] for radius in RADII_M}
    filtered_map_error = []
    online_map_states = []
    motion_plans = [None] * n_keyframes

    releases = release_schedule_lib.load_sidecar(
        Path(args.release_schedule), data)
    release_counts = {}
    for release in releases:
        release_counts[release.release_keyframe_idx] = (
            release_counts.get(release.release_keyframe_idx, 0) + 1)

    def joint_release_likelihood(release, keyframe, grid_belief=None):
        grid_belief = belief if grid_belief is None else grid_belief
        return likelihood_cache.get(
            ("joint", release.tracklet_id, keyframe),
            lambda: _joint_release_likelihood(
                release, keyframe, grid_belief))

    def _joint_release_likelihood(release, keyframe, grid_belief):
        (candidate_east, candidate_north, candidate_weight,
         tail_mass) = candidate_set(release.tracklet_id)
        anchors = sorted({m.anchor_keyframe_idx
                          for m in release.measurements})
        rel = relative_epoch_poses(odometry, anchors, keyframe)
        epochs = []
        for m in release.measurements:
            d_fwd, d_left, d_head = rel[m.anchor_keyframe_idx]
            kappa = min(float(m.kappa) * args.kappa_scale, MAX_KAPPA)
            head_slack_var = 0.0
            pos_slack_var = 0.0
            if args.joint_slack:
                # the same slack the motion model grants between this
                # anchor and the release keyframe
                for step in range(m.anchor_keyframe_idx + 1, keyframe + 1):
                    delta = odometry[step]
                    head_slack_var += (
                        heading_rw_rad ** 2
                        + (delta.sigma_yaw_rad * args.yaw_sigma_scale) ** 2)
                    pos_slack_var += (
                        args.diffusion_m ** 2 + delta.sigma_m ** 2)
            epochs.append((
                d_fwd, d_left, d_head,
                math.radians(m.bearing_forward_cw_deg),
                1.0 / kappa,
                m.range_max_m if args.range_cap else None,
                head_slack_var, math.sqrt(pos_slack_var)))
        return grid_belief.track_joint_likelihood(
            epochs, candidate_east, candidate_north,
            candidate_weight, sigma_pos, args.pi0, tail_mass,
            quantization_comp=bool(args.quantization_comp),
            range_softness=args.range_softness,
            range_floor=args.range_floor,
            temper=args.joint_temper,
            cap=args.joint_cap)

    smoothing_releases = releases

    if args.track_joint:
        releases_at = {}
        for release in releases:
            releases_at.setdefault(
                release.release_keyframe_idx, []).append(release)
        started = time.time()
        for keyframe in range(n_keyframes):
            if keyframe > 0:
                belief.motion(
                    odometry[keyframe], args.yaw_sigma_scale,
                    heading_rw_rad, args.diffusion_m)
            for release in releases_at.get(keyframe, ()):
                likelihood = joint_release_likelihood(release, keyframe)
                belief.belief = _normalized(belief.belief * likelihood)
            belief.renormalize()
            mass, map_error, state = _score_message(
                belief.belief, keyframe, masks, truth_by_kf, grid,
                args.device)
            online_map_states.append(state)
            for radius in RADII_M:
                filtered_series[radius].append(mass[radius])
            filtered_map_error.append(map_error)
            if keyframe % 20 == 0 or keyframe == n_keyframes - 1:
                print(
                    f"joint kf {keyframe:4d} mass500 {mass[500.0]:.4f} "
                    f"mass100 {mass[100.0]:.4f} "
                    f"map_err {map_error:8.1f} m "
                    f"released {release_counts.get(keyframe, 0)} "
                    f"({time.time() - started:.0f}s)")
        forward_seconds = time.time() - started
        filtered_summary = _summary(
            filtered_series, filtered_map_error, truth_by_kf,
            n_keyframes)
        final_top_modes = _top_modes(
            belief.belief, grid, args.top_modes,
            args.top_mode_position_nms_m, args.top_mode_heading_nms_deg)
        print("joint:", {key: round(value, 4)
                         for key, value in filtered_summary.items()},
              f"runtime {forward_seconds:.0f}s")
        print(likelihood_cache.describe())
        if args.out:
            payload = {
                "schema": "farfield_causal_grid/v1",
                "localization_inputs": data.artifact_ref.to_dict(),
                "config": vars(args),
                "odometry_profile": odometry_profile,
                "episode": None,
                "availability": {
                    "policy": "natural_track_close_with_eof_flush",
                    "post_closure_processing_delay_s": 0.0,
                    "release_schedule": str(args.release_schedule),
                    "track_factor": "whole_track_joint_at_release",
                    "past_scores_revised_after_replay": False,
                },
                "grid": {
                    "n_east": grid.n_east,
                    "n_north": grid.n_north,
                    "n_heading": args.n_heading,
                    "cell_m": grid.cell_m,
                    "box": box,
                },
                "summary": filtered_summary,
                "mass_by_keyframe": {
                    f"{radius:g}": filtered_series[radius]
                    for radius in RADII_M
                },
                "map_error_m_by_keyframe": filtered_map_error,
                "online_map_state_by_keyframe": {
                    "source": (
                        "online_current_position_marginal_grid_cell_"
                        "argmax_with_conditional_heading"),
                    "keyframe_order": "list_index_equals_keyframe_idx",
                    "states": online_map_states,
                },
                "filtered_final_top_modes": {
                    "source": "online_current_posterior_at_final_keyframe",
                    "reference_keyframe_idx": n_keyframes - 1,
                    "pose_frame": (
                        "region_enu_heading_world_cw_from_north"),
                    "position_nms_m": args.top_mode_position_nms_m,
                    "heading_nms_deg": args.top_mode_heading_nms_deg,
                    "requested": args.top_modes,
                    "returned": len(final_top_modes),
                    "modes": final_top_modes,
                },
                "runtime_seconds": {"joint_forward": forward_seconds},
            }
            Path(args.out).write_text(
                json.dumps(finish_payload(payload), indent=1))
            print("wrote", args.out)
        return

    planner = GridBelief(grid, args.n_heading, args.device)
    for keyframe in range(1, n_keyframes):
        motion_plans[keyframe] = planner.plan_motion(
            odometry[keyframe], args.yaw_sigma_scale,
            heading_rw_rad, args.diffusion_m)

    started = time.time()
    def score_current(message, keyframe):
        mass, map_error, state = _score_message(
            message, keyframe, masks, truth_by_kf, grid, args.device)
        online_map_states.append(state)
        if keyframe % 20 == 0 or keyframe == n_keyframes - 1:
            print(
                f"causal kf {keyframe:4d} mass500 {mass[500.0]:.4f} "
                f"mass100 {mass[100.0]:.4f} "
                f"map_err {map_error:8.1f} m "
                f"released {release_counts.get(keyframe, 0)} "
                f"({time.time() - started:.0f}s)")
        return mass, map_error

    final_filtered, scored, replay_stats = causal_replay(
        belief.belief, motion_plans, releases, apply_likelihoods,
        score_current, args.checkpoint_keyframes)
    forward_seconds = time.time() - started
    for mass, map_error in scored:
        for radius in RADII_M:
            filtered_series[radius].append(mass[radius])
        filtered_map_error.append(map_error)
    filtered_summary = _summary(
        filtered_series, filtered_map_error, truth_by_kf, n_keyframes)
    final_top_modes = _top_modes(
        final_filtered, grid, args.top_modes,
        args.top_mode_position_nms_m,
        args.top_mode_heading_nms_deg)
    print(
        "causal:",
        {key: round(value, 4)
         for key, value in filtered_summary.items()})
    print("replay:", replay_stats, f"runtime {forward_seconds:.0f}s")
    print(likelihood_cache.describe())

    if args.out:
        payload = {
            "schema": "farfield_causal_grid/v1",
            "localization_inputs": data.artifact_ref.to_dict(),
            "config": vars(args),
            "odometry_profile": odometry_profile,
            "episode": None,
            "availability": {
                "policy": "natural_track_close_with_eof_flush",
                "post_closure_processing_delay_s": 0.0,
                "release_schedule": str(args.release_schedule),
                "historical_measurements_keep_original_anchors": True,
                "past_scores_revised_after_replay": False,
            },
            "grid": {
                "n_east": grid.n_east,
                "n_north": grid.n_north,
                "n_heading": args.n_heading,
                "cell_m": grid.cell_m,
                "box": box,
            },
            "summary": filtered_summary,
            "mass_by_keyframe": {
                f"{radius:g}": filtered_series[radius]
                for radius in RADII_M
            },
            "map_error_m_by_keyframe": filtered_map_error,
            "online_map_state_by_keyframe": {
                "source": (
                    "online_current_position_marginal_grid_cell_argmax_"
                    "with_conditional_heading"),
                "keyframe_order": "list_index_equals_keyframe_idx",
                "trajectory_semantics": "not_a_joint_trajectory",
                "position_semantics": (
                    "same_position_argmax_as_map_error_m_by_keyframe"),
                "heading_semantics": (
                    "conditional_heading_argmax_at_selected_position_cell"),
                "states": online_map_states,
            },
            "replay": replay_stats,
            "filtered_final_top_modes": {
                "source": "causal_current_posterior_at_final_keyframe",
                "reference_keyframe_idx": n_keyframes - 1,
                "pose_frame": (
                    "region_enu_heading_world_cw_from_north"),
                "source_rank_semantics": (
                    "one_based_probability_order_after_se2_nms"),
                "source_probability_semantics": (
                    "single_grid_state_mass_not_integrated_mode_mass"),
                "position_nms_m": args.top_mode_position_nms_m,
                "heading_nms_deg": args.top_mode_heading_nms_deg,
                "requested": args.top_modes,
                "returned": len(final_top_modes),
                "modes": final_top_modes,
            },
            "runtime_seconds": {"causal_forward_replay": forward_seconds},
        }
        Path(args.out).write_text(
            json.dumps(finish_payload(payload), indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
