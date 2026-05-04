"""Learned per-step fusion policy for the histogram filter.

Replaces the constant-σ EA aggregator with a small MLP that, given a row of
image and landmark similarities (plus optional belief / trajectory features),
emits per-step (σ_img, σ_lm, α). The fused log-likelihood is then

    log_p_img = log_softmax(img_sim / σ_img)
    log_p_lm  = log_softmax(lm_sim  / σ_lm)
    fused     = α · log_p_img + (1 − α) · log_p_lm

α is *not* the EA peak_sharpness ratio — that formula was σ-coupled and
caused the calibration trap we documented (small σ_lm artificially sharpens
the landmark posterior, falsely amplifying landmark weight on no-signal
rows). Here α is a free output of the policy.

State features are deliberately city-agnostic (statistics of the row /
belief) so the same policy can transfer from Seattle to Chicago.

The policy is trained by ``train_learned_aggregator.py`` end-to-end through
the differentiable histogram filter (``apply_observation`` with
``surrogate_tau`` enabled).
"""

from dataclasses import dataclass

import common.torch.load_torch_deps  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    ObservationLogLikelihoodAggregator,
    LearnedAggregatorConfig,
    _replace_nan_with_zero,
    _load_similarity_matrix,
)
import experimental.overhead_matching.swag.data.vigor_dataset as vd


# ============ Constants ============

# Floor on σ to keep ``log_softmax(sim / σ)`` numerically stable.
SIGMA_MIN: float = 0.02

# Reference σ used when computing σ-independent features (so the feature
# values do not depend on the action the policy is about to take).
SIGMA_REF_FEATURES: float = 1.0

# Number of belief features (used only when belief context is provided).
NUM_BELIEF_FEATURES: int = 4

# Number of trajectory-context features.
NUM_STEP_FEATURES: int = 3

# Compact per-step features that appear in the policy's history window.
# Each entry summarizes a single past step: belief state + how informative
# that step's observation was. Pure scalars so the buffer is small and
# city-agnostic. Order MUST match `HistoryEntry.to_list`.
NUM_HISTORY_ENTRY_FEATURES: int = 5

# Number of past steps the policy sees alongside its current features.
# Padded with zero entries on the left when fewer than HISTORY_DEPTH steps
# have elapsed at the start of a path.
HISTORY_DEPTH: int = 10


# ============ Feature extraction ============


@dataclass
class StreamRowStats:
    """Per-row statistics for one similarity stream (image OR landmark).

    All entries are scalar tensors so they can be stacked into a feature
    vector without a Python loop. Computed over the *finite* entries of the
    row to be robust to -inf / NaN values.
    """

    max: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    median: torch.Tensor
    top1_minus_top2: torch.Tensor
    top1_minus_top5: torch.Tensor
    frac_finite: torch.Tensor
    frac_nonzero: torch.Tensor
    softmax_entropy: torch.Tensor      # entropy of softmax(sim / σ_ref)
    softmax_peak_sharp: torch.Tensor   # max − mean of log_softmax(sim / σ_ref)

    NUM_FEATURES = 10

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([
            self.max, self.mean, self.std, self.median,
            self.top1_minus_top2, self.top1_minus_top5,
            self.frac_finite, self.frac_nonzero,
            self.softmax_entropy, self.softmax_peak_sharp,
        ])


def compute_row_stats(
    sim_row: torch.Tensor,
    sigma_ref: float = SIGMA_REF_FEATURES,
) -> StreamRowStats:
    """Compute σ-independent statistics of a similarity row.

    All features are scale-normalized in the sense that they're functions of
    the row's distribution rather than absolute identifiers, so they should
    be city-transferable. The only σ involved is ``sigma_ref`` for the two
    log-softmax features, which is held *fixed* across cities.
    """
    finite_mask = torch.isfinite(sim_row)
    n_total = sim_row.numel()
    n_finite = int(finite_mask.sum().item())
    device = sim_row.device
    dtype = sim_row.dtype

    if n_finite == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return StreamRowStats(
            max=zero, mean=zero, std=zero, median=zero,
            top1_minus_top2=zero, top1_minus_top5=zero,
            frac_finite=zero, frac_nonzero=zero,
            softmax_entropy=zero, softmax_peak_sharp=zero,
        )

    finite_vals = sim_row[finite_mask]
    sorted_desc, _ = torch.sort(finite_vals, descending=True)

    max_v = sorted_desc[0]
    mean_v = finite_vals.mean()
    std_v = (
        finite_vals.std(unbiased=False)
        if n_finite > 1 else torch.zeros((), device=device, dtype=dtype)
    )
    median_v = finite_vals.median()
    top2 = sorted_desc[1] if n_finite >= 2 else sorted_desc[0]
    top5 = sorted_desc[4] if n_finite >= 5 else sorted_desc[-1]

    # softmax-on-row stats at fixed σ_ref (decoupled from policy output).
    log_p = torch.log_softmax(finite_vals / sigma_ref, dim=0)
    p = log_p.exp()
    softmax_entropy = -(p * log_p).sum()
    softmax_peak_sharp = log_p.max() - log_p.mean()

    frac_finite = torch.tensor(n_finite / max(n_total, 1), device=device, dtype=dtype)
    n_nonzero = int((finite_vals != 0).sum().item())
    frac_nonzero = torch.tensor(
        n_nonzero / max(n_total, 1), device=device, dtype=dtype
    )

    return StreamRowStats(
        max=max_v, mean=mean_v, std=std_v, median=median_v,
        top1_minus_top2=(max_v - top2),
        top1_minus_top5=(max_v - top5),
        frac_finite=frac_finite, frac_nonzero=frac_nonzero,
        softmax_entropy=softmax_entropy,
        softmax_peak_sharp=softmax_peak_sharp,
    )


@dataclass
class HistoryEntry:
    """Compact per-step summary appended to the policy's history buffer.

    Captures *what the filter has seen recently* without including raw
    similarity rows (those are summarized elsewhere) — five scalars that
    describe how the belief evolved and how informative each observation
    was. ``last_obs_max_log_p`` is the max of the fused log-likelihood that
    was applied at this step (a proxy for "how peaky was the observation").
    """

    belief_entropy: float = 0.0
    belief_log_trace_var: float = 0.0
    belief_top_cell_mass: float = 0.0
    log_step_distance: float = 0.0
    last_obs_max_log_p: float = 0.0

    def to_list(self) -> list[float]:
        return [
            self.belief_entropy, self.belief_log_trace_var,
            self.belief_top_cell_mass, self.log_step_distance,
            self.last_obs_max_log_p,
        ]


@dataclass
class StepContext:
    """Optional per-step features the training loop can attach to the
    aggregator before each __call__. None → use neutral defaults.

    All fields are floats; the aggregator turns them into a tensor.
    """

    # Belief features
    belief_entropy: float = 0.0
    belief_log_trace_var: float = 0.0
    belief_top_cell_mass: float = 0.0
    belief_step_index_norm: float = 0.0
    # Trajectory context
    norm_cum_distance: float = 0.0
    log_step_distance: float = 0.0
    step_idx_over_path_len: float = 0.0
    # Sliding window of past steps. Newest at the end. None → zero-filled.
    history: list[HistoryEntry] | None = None


def history_to_tensor(
    history: list[HistoryEntry] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Flatten the history window to a fixed-length tensor.

    Pads on the left with zero entries when fewer than ``HISTORY_DEPTH``
    past steps are available, and truncates from the left if longer.
    """
    if history is None:
        history = []
    history = history[-HISTORY_DEPTH:]
    n_pad = HISTORY_DEPTH - len(history)
    flat: list[float] = [0.0] * (n_pad * NUM_HISTORY_ENTRY_FEATURES)
    for h in history:
        flat.extend(h.to_list())
    return torch.tensor(flat, device=device, dtype=dtype)


def step_context_to_tensor(
    ctx: StepContext | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pack a StepContext (current scalars + history window) into a tensor.

    Produces a zero-filled fallback if ``ctx is None``.
    """
    history_dim = HISTORY_DEPTH * NUM_HISTORY_ENTRY_FEATURES
    if ctx is None:
        return torch.zeros(
            NUM_BELIEF_FEATURES + NUM_STEP_FEATURES + history_dim,
            device=device, dtype=dtype,
        )
    current = torch.tensor([
        ctx.belief_entropy, ctx.belief_log_trace_var,
        ctx.belief_top_cell_mass, ctx.belief_step_index_norm,
        ctx.norm_cum_distance, ctx.log_step_distance,
        ctx.step_idx_over_path_len,
    ], device=device, dtype=dtype)
    history = history_to_tensor(ctx.history, device, dtype)
    return torch.cat([current, history], dim=0)


def extract_belief_features(belief) -> tuple[float, float, float]:
    """(entropy, log_trace_var, top_cell_mass) from a HistogramBelief."""
    probs = belief.get_belief()
    flat = probs.reshape(-1)
    # Numerically-safe entropy on probabilities.
    entropy = -(flat * torch.log(flat.clamp(min=1e-40))).sum().item()
    var = belief.get_variance_deg_sq().sum().item()
    log_trace_var = float(torch.log(torch.tensor(max(var, 1e-30))))
    top_cell_mass = float(flat.max().item())
    return entropy, log_trace_var, top_cell_mass


# Total feature dimensionality.
FEATURE_DIM: int = (
    2 * StreamRowStats.NUM_FEATURES
    + NUM_BELIEF_FEATURES
    + NUM_STEP_FEATURES
    + HISTORY_DEPTH * NUM_HISTORY_ENTRY_FEATURES
)


# ============ Policy ============


class FeatureStandardizer(nn.Module):
    """Per-feature running standardizer.

    Estimates running mean / std on the training set and applies frozen
    statistics at eval time. Lives inside the policy module so it serializes
    with the weights and is automatically applied to each inference.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.register_buffer("count", torch.zeros(()))
        self.frozen: bool = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if self.frozen:
            return
        # Welford's online algorithm, batched.
        x = x.detach()
        batch_n = float(x.shape[0]) if x.dim() == 2 else 1.0
        batch_mean = x.mean(0) if x.dim() == 2 else x
        batch_var = x.var(0, unbiased=False) if x.dim() == 2 else torch.zeros_like(x)
        new_count = self.count + batch_n
        delta = batch_mean - self.mean
        self.mean += delta * (batch_n / new_count.clamp(min=1.0))
        # Combine running and batch variance via parallel-algorithm formula.
        self.var = (
            self.var * (self.count / new_count.clamp(min=1.0))
            + batch_var * (batch_n / new_count.clamp(min=1.0))
            + (delta ** 2) * (self.count * batch_n / (new_count.clamp(min=1.0) ** 2))
        )
        self.count = new_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = (self.var + 1e-6).sqrt()
        return (x - self.mean) / std


class SigmaPolicy(nn.Module):
    """Tiny MLP that maps state features to (u_img, u_lm, u_α).

    Output post-processing (softplus + sigmoid + SIGMA_MIN floor) is done in
    :class:`LearnedAggregator` so this module's outputs are unconstrained
    and the policy can be queried at training time without coupling to the
    aggregator's interface.
    """

    def __init__(self, num_features: int = FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.standardizer = FeatureStandardizer(num_features)
        self.trunk = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_sigma = nn.Linear(hidden, 2)  # (u_img, u_lm)
        self.head_alpha = nn.Linear(hidden, 1)  # (u_alpha)

        # Initialize sigma head so initial outputs are near σ ≈ 0.13 — close
        # to the per-stream NLL-softmax optima — to avoid pathological starts.
        with torch.no_grad():
            self.head_sigma.weight.zero_()
            self.head_sigma.bias.fill_(_softplus_inv(0.13 - SIGMA_MIN))
            # Initial α ≈ 0.5: balanced fusion.
            self.head_alpha.weight.zero_()
            self.head_alpha.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.standardizer(features)
        h = self.trunk(x)
        u_sigma = self.head_sigma(h)        # (..., 2)
        u_alpha = self.head_alpha(h)        # (..., 1)
        return torch.cat([u_sigma, u_alpha], dim=-1)  # (..., 3)


def _softplus_inv(y: float) -> float:
    """Inverse of softplus, used to seed the σ-head bias at a target value."""
    import math
    if y <= 0:
        return -10.0
    return math.log(math.expm1(y))


# ============ Aggregator ============


class LearnedAggregator(ObservationLogLikelihoodAggregator):
    """Aggregator that consults a learned policy each step.

    Per-step interface:
      1. Optional: training loop calls :meth:`set_step_context` with the
         current belief and trajectory metadata.
      2. ``__call__(pano_id)`` extracts row stats, queries the policy for
         (σ_img, σ_lm, α), computes the fused log-softmax, and returns it.

    Production code that doesn't have belief context can simply call
    ``__call__`` directly; the policy then operates on row stats only with
    zero-filled belief / step features.
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        policy: SigmaPolicy,
        device: torch.device,
        sigma_ref: float = SIGMA_REF_FEATURES,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.policy = policy.to(device)
        self.device = device
        self.sigma_ref = sigma_ref
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])
        self._step_context: StepContext | None = None

    def set_step_context(self, ctx: StepContext | None) -> None:
        """Attach optional belief / trajectory features for the next call."""
        self._step_context = ctx

    def _compute_features(
        self,
        img_row: torch.Tensor,
        lm_row: torch.Tensor,
    ) -> torch.Tensor:
        img_stats = compute_row_stats(img_row, sigma_ref=self.sigma_ref).to_tensor()
        lm_stats = compute_row_stats(lm_row, sigma_ref=self.sigma_ref).to_tensor()
        ctx_t = step_context_to_tensor(
            self._step_context, device=img_row.device, dtype=img_row.dtype
        )
        return torch.cat([img_stats, lm_stats, ctx_t], dim=0)

    def _policy_outputs(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (σ_img, σ_lm, α) with positivity / [0,1] constraints applied."""
        u = self.policy(features.unsqueeze(0)).squeeze(0)  # (3,)
        sigma_img = SIGMA_MIN + F.softplus(u[0])
        sigma_lm = SIGMA_MIN + F.softplus(u[1])
        alpha = torch.sigmoid(u[2])
        return sigma_img, sigma_lm, alpha

    def fused_log_likelihood(
        self,
        img_row: torch.Tensor,
        lm_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fused log-likelihood and return (log_ll, σ_img, σ_lm, α).

        Exposed as a separate method so the training loop can keep gradients
        flowing through to the policy without going through the public
        ``__call__`` (which post-processes via ``_replace_nan_with_zero`` /
        ``.to(device)``).
        """
        features = self._compute_features(img_row, lm_row)
        sigma_img, sigma_lm, alpha = self._policy_outputs(features)

        log_p_img = torch.log_softmax(img_row / sigma_img, dim=0)

        # Fall-through to image-only when the landmark row is missing or
        # uninformative. log_softmax of a constant row is uniform and adds
        # only a constant to fused log-likelihoods (cancelled by the
        # subsequent normalize), but routing through image-only keeps the
        # path more numerically stable and unambiguous.
        lm_finite_mask = torch.isfinite(lm_row)
        lm_finite = lm_row[lm_finite_mask]
        if lm_finite.numel() == 0 or torch.allclose(
            lm_finite.max(), lm_finite.min()
        ):
            return log_p_img, sigma_img, sigma_lm, alpha

        log_p_lm = torch.log_softmax(lm_row / sigma_lm, dim=0)
        fused = alpha * log_p_img + (1.0 - alpha) * log_p_lm
        # Where landmark is -inf at a single entry, fall back to image-only
        # at that entry.
        fused = torch.where(lm_finite_mask, fused, log_p_img)
        return fused, sigma_img, sigma_lm, alpha

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)
        img_row = self.image_similarity_matrix[pano_index].to(self.device)
        lm_row = self.landmark_similarity_matrix[pano_index].to(self.device)
        log_ll, *_ = self.fused_log_likelihood(img_row, lm_row)
        return _replace_nan_with_zero(log_ll)

    @classmethod
    def from_config(
        cls,
        config: LearnedAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "LearnedAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        policy = SigmaPolicy()
        state_dict = torch.load(
            config.policy_weights_path, map_location="cpu", weights_only=False,
        )
        policy.load_state_dict(state_dict)
        policy.standardizer.frozen = True
        policy.eval()
        return cls(
            image_similarity_matrix=image_sim,
            landmark_similarity_matrix=landmark_sim,
            panorama_metadata=vigor_dataset._panorama_metadata,
            policy=policy,
            device=device,
        )
