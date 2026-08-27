"""Joint location-heading scoring of panorama crop rings against the
reference depth-view database (plan section 5.3).

Query crops m = 0..M-1 sit at known relative body azimuths alpha_m; the ring's
global yaw is unknown. Reference locations store N_theta views at map yaws
psi_n. For circular shift k,

    S(i, k) = mean_over_valid_m  cos( q_m , d_{i, (m+k) mod N_theta} )

which searches location and global heading jointly. With M == N_theta and the
default rings, shift k implies robot heading k * (360 / N_theta) degrees
(compass CW), since psi_{(m+k) mod N} - alpha_m == k * spacing for every m.

Descriptors are unit-norm (CrossLocateVGG16MAC output), so the cosine is a
dot product and the release's squared-Euclidean distance is 2 - 2 * cos:
rankings agree by construction.
"""

from dataclasses import dataclass

import common.torch.load_torch_deps  # noqa: F401  (must precede torch)
import torch

import numpy as np


@dataclass
class JointScores:
    """Scores over (location, heading shift)."""

    scores: torch.Tensor  # (n_locations, n_theta) float32
    heading_cw_deg: np.ndarray  # (n_theta,) heading implied by each shift

    def top_k(self, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(values, location_idx, shift_idx) of the k best joint cells."""
        flat = self.scores.reshape(-1)
        values, flat_idx = torch.topk(flat, min(k, flat.numel()))
        n_theta = self.scores.shape[1]
        return values, flat_idx // n_theta, flat_idx % n_theta


def joint_scores(query_descriptors: torch.Tensor,
                 database_descriptors: torch.Tensor,
                 valid_crops: torch.Tensor | None = None,
                 crop_top_k: int | None = None) -> JointScores:
    """Score every (location, shift) pair.

    query_descriptors: (M, D) for one panorama's crop ring, unit-norm.
    database_descriptors: (n_locations, N_theta, D), unit-norm, where view n
        is at map yaw n * (360 / N_theta) CW from north.
    valid_crops: optional (M,) bool; invalid crops (occluded, water-dominated,
        failed extraction) are excluded from the mean. All-invalid is an error
        -- the caller decides how an unusable frame enters the evaluation
        (applicability accounting, not a silent zero).
    crop_top_k: if set, S(i, k) is the mean of the k best-matching crops
        instead of all of them — robust aggregation under partial occlusion
        (canopy, people): a few clean crops carry the frame rather than being
        drowned by the occluded majority. Chosen per-cell, so different
        (location, shift) hypotheses may be supported by different crops.
        Mutually exclusive with valid_crops (the occlusion studies used
        exactly one mechanism at a time).
    """
    m, dim = query_descriptors.shape
    n_loc, n_theta, dim_db = database_descriptors.shape
    if dim != dim_db:
        raise ValueError(f"descriptor dims differ: {dim} vs {dim_db}")
    if m != n_theta:
        raise ValueError(
            f"crop ring size {m} != reference ring size {n_theta}; matched "
            "rings are what make shift k a pure heading offset")
    if valid_crops is None:
        valid = torch.ones(m, dtype=torch.bool,
                           device=query_descriptors.device)
    else:
        valid = valid_crops.to(device=query_descriptors.device,
                               dtype=torch.bool)
    if not bool(valid.any()):
        raise ValueError("no valid crops in the query ring")

    # cos[l, m, n] = q_m . d_{l, n}
    cos = torch.einsum("md,lnd->lmn", query_descriptors,
                       database_descriptors)
    # Gather n = (m + k) mod N for every (m, k).
    m_idx = torch.arange(m, device=cos.device)
    k_idx = torch.arange(n_theta, device=cos.device)
    gather = ((m_idx[:, None] + k_idx[None, :]) % n_theta)  # (M, K)
    aligned = torch.gather(
        cos, 2, gather[None].expand(n_loc, -1, -1))  # (L, M, K)
    if crop_top_k is not None:
        if valid_crops is not None:
            raise ValueError("crop_top_k and valid_crops are mutually "
                             "exclusive")
        if not 1 <= crop_top_k <= m:
            raise ValueError(f"crop_top_k must be in [1, {m}], "
                             f"got {crop_top_k}")
        scores = aligned.topk(crop_top_k, dim=1).values.mean(dim=1)
    else:
        weights = valid.float() / valid.float().sum()
        scores = torch.einsum("lmk,m->lk", aligned, weights)

    spacing = 360.0 / n_theta
    return JointScores(scores=scores,
                       heading_cw_deg=np.arange(n_theta) * spacing)
