"""Adaptive aggregators for computing observation log-likelihoods."""

import abc
from pathlib import Path
import msgspec
import common.torch.load_torch_deps
import torch
import pandas as pd

from common.python.serialization import MSGSPEC_STRUCT_OPTS, msgspec_dec_hook
from experimental.overhead_matching.swag.filter.particle_filter import (
    wag_observation_log_likelihood_from_similarity_matrix,
)
import experimental.overhead_matching.swag.data.vigor_dataset as vd


# ============ Config Types (msgspec) ============


class SingleSimilarityMatrixAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for single similarity matrix aggregator."""

    similarity_matrix_path: Path
    sigma: float


class ImageLandmarkPrivilegedInformationFusionConfig(
    msgspec.Struct, **MSGSPEC_STRUCT_OPTS
):
    """Config for privileged information fusion aggregator."""

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    sigma: float
    include_semipositive: bool = True


class EntropyAdaptiveAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for entropy-adaptive weighted fusion aggregator."""

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    sigma: float


class SafaPlusNormalizedLandmarkAggregatorConfig(
    msgspec.Struct, **MSGSPEC_STRUCT_OPTS
):
    """Config for ``SafaPlusNormalizedLandmarkAggregator``.

    Image stream uses SAFA-style Gaussian-on-residuals at ``image_sigma``.
    Landmark stream divides each row by its ``row_max`` and applies
    Gaussian-on-residuals to ``r_norm = 1 − sim_t / row_max`` at
    ``landmark_sigma``. All-zero / constant landmark rows fall through
    to image-only.

    When ``landmark_use_raw_residual`` is True, the landmark stream uses
    the same SAFA-form raw residual ``row_max − sim_t`` as the image
    stream. Use this when the second matrix is a similarity matrix on
    the same scale as the image stream (e.g. an OSM-tile-baseline SAFA
    matrix), where the per-row /max normalization is unnecessary.
    Note ``landmark_sigma`` then lives on the raw-cosine scale rather
    than the [0,1] normalized scale.

    The two modes also differ in their NaN contract on the landmark
    matrix:

    * Normalized-residual (default): NaN entries are legitimate "no
      landmark info for this cell" signals; the aggregator falls back
      to image-only at those positions.
    * Raw-residual: ``landmark_similarity_matrix`` is expected dense.
      Any NaN/inf entry indicates a bug (stale matrix, broken export,
      etc.) and raises eagerly.
    """

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    image_sigma: float
    landmark_sigma: float
    landmark_use_raw_residual: bool = False

    def __post_init__(self):
        if not (self.image_sigma > 0):
            raise ValueError(f"image_sigma must be positive, got {self.image_sigma}")
        if not (self.landmark_sigma > 0):
            raise ValueError(f"landmark_sigma must be positive, got {self.landmark_sigma}")


class SafaPlusPiecewiseLandmarkAggregatorConfig(
    msgspec.Struct, **MSGSPEC_STRUCT_OPTS
):
    """Config for ``SafaPlusPiecewiseLandmarkAggregator`` (prototype).

    Same image stream as ``SafaPlusNormalizedLandmarkAggregator`` (SAFA
    Gaussian-on-residuals at ``image_sigma``). The landmark stream replaces the
    single half-normal Gaussian on the normalized residual
    ``r = 1 - sim / row_max`` with a calibrated, piecewise-constant
    *log-likelihood-ratio* lookup over ``r in [0, 1]``:

      ``log_p_lm[j] = landmark_lr_scale * g(r_j)``

    where ``g`` is the per-bin value ``log p(r|true) - log p(r|negative)``
    estimated from a calibration matrix. This is the principled per-patch
    factor for the histogram filter under conditional independence of patches
    given the true location; ``0`` means "uninformative", so the natural
    fall-through for missing / constant landmark rows is ``log_p_lm == 0``
    (image-only) — consistent with the lookup baseline.

    The lookup uses uniformly-wide bins (the discrete ``r == 0`` argmax-hit and
    ``r == 1`` miss events simply land in the first / last bin — both
    populations have their atoms at the same location, so the bin ratio
    recovers them without special-casing). It is matrix-agnostic: a bimodal
    Hungarian residual and a unimodal dense residual are both fit by the same
    table.

    ``landmark_lr_scale`` is an optional temperature multiplier; ``1.0`` is the
    calibrated value. Calibration is produced by
    ``calibrate_landmark_piecewise.py`` and frozen across cities (calibrated
    once on a reference city), matching the single-``sigma_lm`` methodology of
    the Gaussian variant.
    """

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    image_sigma: float
    landmark_log_lr_edges: list[float]
    landmark_log_lr_values: list[float]
    landmark_lr_scale: float = 1.0

    def __post_init__(self):
        if not (self.image_sigma > 0):
            raise ValueError(f"image_sigma must be positive, got {self.image_sigma}")
        if len(self.landmark_log_lr_edges) != len(self.landmark_log_lr_values) + 1:
            raise ValueError(
                "landmark_log_lr_edges must have one more entry than "
                f"landmark_log_lr_values (got {len(self.landmark_log_lr_edges)} "
                f"edges vs {len(self.landmark_log_lr_values)} bins)"
            )


# Union type for polymorphic deserialization.
AggregatorConfig = (
    SingleSimilarityMatrixAggregatorConfig
    | ImageLandmarkPrivilegedInformationFusionConfig
    | EntropyAdaptiveAggregatorConfig
    | SafaPlusNormalizedLandmarkAggregatorConfig
    | SafaPlusPiecewiseLandmarkAggregatorConfig
)


# ============ Helper Functions ============


def _replace_nan_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN values with 0 (no update in log-likelihood space)."""
    tensor[torch.isnan(tensor)] = 0
    return tensor


def _raise_if_nonfinite(
    tensor: torch.Tensor, pano_id: str, name: str, cls_name: str
) -> None:
    nonfinite = ~torch.isfinite(tensor)
    if nonfinite.any():
        first_idx = int(torch.nonzero(nonfinite, as_tuple=False)[0].item())
        raise RuntimeError(
            f"{cls_name}: non-finite value in {name} "
            f"for pano_id={pano_id!r} at index {first_idx} "
            f"(value={tensor.flatten()[first_idx].item()})."
        )


def _load_similarity_matrix(path: Path) -> torch.Tensor:
    """Load a similarity matrix from a file.

    Handles both raw tensor format and dict format (with 'similarity' key).
    Keeps matrix on CPU to save GPU memory - row lookups are fast enough.
    """
    data = torch.load(path, weights_only=False, map_location="cpu")
    if isinstance(data, dict):
        if "similarity" in data:
            return data["similarity"]
        raise ValueError(
            f"Similarity matrix file {path} is a dict but has no 'similarity' key. "
            f"Available keys: {list(data.keys())}"
        )
    return data


def _assert_matrix_aligned(
    matrix: torch.Tensor,
    panorama_metadata: pd.DataFrame,
    satellite_metadata: pd.DataFrame,
    matrix_path: Path,
) -> None:
    """Verify a similarity matrix's shape matches the current dataset.

    Rows are indexed positionally via the panorama_metadata pano_id list and
    columns via the satellite_metadata satellite-patch index (e.g. through
    ``positive_satellite_idxs`` and the cell-to-patch mapping). If either
    dimension was trimmed after the matrix was generated, indexing silently
    returns valid-looking values for the wrong query — the filter still
    converges, but to the wrong cell. Bigger-than-current always silently
    succeeds (no IndexError), so we have to compare lengths explicitly.
    """
    n_pano = len(panorama_metadata)
    n_sat = len(satellite_metadata)
    n_rows, n_cols = matrix.shape[0], matrix.shape[1]
    if n_rows != n_pano:
        raise ValueError(
            f"Similarity matrix at {matrix_path} has {n_rows} rows but the "
            f"dataset's panorama_metadata has {n_pano} entries. The matrix is "
            f"misaligned with the current panorama set (likely stale relative "
            f"to a dataset trim). Regenerate it from the current panoramas."
        )
    if n_cols != n_sat:
        raise ValueError(
            f"Similarity matrix at {matrix_path} has {n_cols} columns but the "
            f"dataset's satellite_metadata has {n_sat} entries. The matrix is "
            f"misaligned with the current satellite set (likely stale relative "
            f"to a satellite trim). Regenerate it from the current satellites."
        )


# ============ Aggregator Classes ============


class ObservationLogLikelihoodAggregator(abc.ABC):
    """Base class for computing observation log-likelihoods from various inputs."""

    @abc.abstractmethod
    def __call__(self, pano_id: str) -> torch.Tensor:
        """Compute observation log-likelihoods for the given panorama.

        Args:
            pano_id: Panorama identifier string

        Returns:
            (num_patches,) tensor of log-likelihoods for each satellite patch
        """
        ...


class SingleSimilarityMatrixAggregator(ObservationLogLikelihoodAggregator):
    """Produces log-likelihoods from a single similarity matrix."""

    def __init__(
        self,
        similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        sigma: float,
        device: torch.device,
    ):
        self.similarity_matrix = similarity_matrix
        self.sigma = sigma
        self.device = device
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)
        similarity = self.similarity_matrix[pano_index]
        if similarity.numel() == 0:
            raise RuntimeError(
                f"SingleSimilarityMatrixAggregator: empty similarity slice for "
                f"pano_id={pano_id!r}. matrix shape={tuple(self.similarity_matrix.shape)}, "
                f"pano_index={pano_index!r} ({type(pano_index).__name__}), "
                f"slice shape={tuple(similarity.shape)}. Likely cause: duplicate "
                f"pano_id in panorama_metadata producing a boolean mask that selects "
                f"zero rows, or a pano_id in the path JSON not present in the dataset."
            )
        log_ll = wag_observation_log_likelihood_from_similarity_matrix(
            similarity, self.sigma
        )
        _raise_if_nonfinite(log_ll, pano_id, "log_ll", cls_name=type(self).__name__)
        return log_ll.to(self.device)

    @classmethod
    def from_config(
        cls,
        config: SingleSimilarityMatrixAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "SingleSimilarityMatrixAggregator":
        similarity_matrix = _load_similarity_matrix(config.similarity_matrix_path)
        _assert_matrix_aligned(
            similarity_matrix,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.similarity_matrix_path,
        )
        return cls(
            similarity_matrix, vigor_dataset._panorama_metadata, config.sigma, device
        )


class ImageLandmarkPrivilegedInformationFusion(ObservationLogLikelihoodAggregator):
    """Fuses image and landmark similarity matrices using privileged information.

    For true satellite patches: max(image_sim, landmark_sim)
    For all other patches: image_sim only
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        sigma: float,
        device: torch.device,
        include_semipositive: bool = True,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.panorama_metadata = panorama_metadata
        self.sigma = sigma
        self.device = device
        self.include_semipositive = include_semipositive
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        image_sim = self.image_similarity_matrix[pano_index]
        landmark_sim = self.landmark_similarity_matrix[pano_index]

        fused_similarity = image_sim.clone()

        row = self.panorama_metadata.iloc[pano_index]
        true_indices = list(row["positive_satellite_idxs"])
        if self.include_semipositive:
            true_indices.extend(row["semipositive_satellite_idxs"])

        if true_indices:
            idx = torch.tensor(
                true_indices, dtype=torch.long, device=image_sim.device
            )
            fused_similarity[idx] = torch.maximum(image_sim[idx], landmark_sim[idx])

        log_ll = wag_observation_log_likelihood_from_similarity_matrix(
            fused_similarity, self.sigma
        )
        return _replace_nan_with_zero(log_ll).to(self.device)

    @classmethod
    def from_config(
        cls,
        config: ImageLandmarkPrivilegedInformationFusionConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "ImageLandmarkPrivilegedInformationFusion":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        _assert_matrix_aligned(
            image_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.image_similarity_matrix_path,
        )
        _assert_matrix_aligned(
            landmark_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.landmark_similarity_matrix_path,
        )
        return cls(
            image_sim,
            landmark_sim,
            vigor_dataset._panorama_metadata,
            config.sigma,
            device,
            config.include_semipositive,
        )


class EntropyAdaptiveAggregator(ObservationLogLikelihoodAggregator):
    """Fuses image and landmark similarity matrices using confidence-weighted averaging.

    Computes a per-source peak sharpness confidence score (max - mean of log-probs)
    and uses it to blend the two similarity vectors before converting to log-likelihoods.
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        sigma: float,
        device: torch.device,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.sigma = sigma
        self.device = device
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def _compute_confidence(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute peak sharpness confidence: max(log_probs) - mean(log_probs).

        Only considers finite values to avoid -inf entries corrupting the result.

        Args:
            log_probs: (num_patches,) log-probability vector (from log_softmax)

        Returns:
            Scalar confidence value (higher = more confident)
        """
        finite_lp = log_probs[torch.isfinite(log_probs)]
        if len(finite_lp) < 2:
            return torch.tensor(0.0)
        return finite_lp.max() - finite_lp.mean()

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index]
        lm_sim = self.landmark_similarity_matrix[pano_index]

        # Normalize to log-probability space first (makes different scales commensurate)
        log_p_img = torch.log_softmax(img_sim / self.sigma, dim=0)

        # Fall back to image-only when landmark data is missing (all -inf)
        lm_finite_mask = torch.isfinite(lm_sim)
        if not lm_finite_mask.any():
            return _replace_nan_with_zero(log_p_img).to(self.device)

        log_p_lm = torch.log_softmax(lm_sim / self.sigma, dim=0)

        img_conf = self._compute_confidence(log_p_img)
        lm_conf = self._compute_confidence(log_p_lm)
        eps = 1e-12
        alpha = img_conf / (img_conf + lm_conf + eps)
        fused_log_p = alpha * log_p_img + (1 - alpha) * log_p_lm
        # Where landmark has -inf, use image-only to avoid eliminating patches
        fused_log_p = torch.where(lm_finite_mask, fused_log_p, log_p_img)

        return _replace_nan_with_zero(fused_log_p).to(self.device)

    @classmethod
    def from_config(
        cls,
        config: EntropyAdaptiveAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "EntropyAdaptiveAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        _assert_matrix_aligned(
            image_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.image_similarity_matrix_path,
        )
        _assert_matrix_aligned(
            landmark_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.landmark_similarity_matrix_path,
        )
        return cls(
            image_sim,
            landmark_sim,
            vigor_dataset._panorama_metadata,
            config.sigma,
            device,
        )


class SafaPlusNormalizedLandmarkAggregator(ObservationLogLikelihoodAggregator):
    """Per-patch observation likelihood by *summing* two Gaussian-on-residuals
    log-likelihoods — image stream as in SAFA, landmark stream on a per-row
    normalized residual.

    Motivation. The EA softmax fusion implicitly normalizes mass over space
    (sums to 1 across patches), which is incompatible with treating the
    aggregator's output as a true per-patch ``log p(z | patch_j)`` that the
    downstream histogram filter can multiply through its belief. This
    aggregator is properly per-patch:

      log_p_img[j] = -log(σ_img √2π) - 0.5 ((sim_max_img − img[j]) / σ_img)^2
      log_p_lm[j]  = -log(σ_lm  √2π) - 0.5 ((1 − lm[j]/sim_max_lm) / σ_lm )^2

      log_p[j] = log_p_img[j] + log_p_lm[j]   (conditional independence)

    Why divide-by-row-max for the landmark stream? Absolute landmark
    similarity scales vary across cities, but the per-row
    ``1 − sim_t / sim_max`` distribution overlaps tightly across cities,
    so a single ``σ_lm`` transfers cleanly. See PR #626 description for
    the cross-city consistency check.

    All-zero / constant landmark rows are handled with a hard fall-through
    to image-only: those rows are uninformative and we don't want them to
    contaminate the fused log-likelihood with noise from the / row_max
    division.
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        image_sigma: float,
        landmark_sigma: float,
        device: torch.device,
        landmark_use_raw_residual: bool = False,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.image_sigma = float(image_sigma)
        self.landmark_sigma = float(landmark_sigma)
        self.landmark_use_raw_residual = bool(landmark_use_raw_residual)
        self.device = device
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index].to(self.device)
        lm_sim = self.landmark_similarity_matrix[pano_index].to(self.device)

        # Image-side Gaussian-on-residuals (SAFA form).
        log_p_img = wag_observation_log_likelihood_from_similarity_matrix(
            img_sim, self.image_sigma
        )
        _raise_if_nonfinite(log_p_img, pano_id, "log_p_img", cls_name=type(self).__name__)

        # Landmark stream — branch on residual form. The two branches have
        # *different* NaN contracts:
        #
        #   * raw-residual: lm_sim is expected dense (a second SAFA-form
        #     matrix on the same scale as the image stream, e.g. an OSM-tile
        #     baseline). NaN/inf indicates a bug — stale matrix, broken
        #     export, etc. — and is raised eagerly.
        #
        #   * normalized-residual (#626): lm_sim is sparse-with-holes; NaN
        #     means "no landmark info for this cell" and we fall back to
        #     image-only at those positions.
        if self.landmark_use_raw_residual:
            _raise_if_nonfinite(lm_sim, pano_id, "lm_sim", cls_name=type(self).__name__)
            sim_max_lm = lm_sim.max()
            sim_min_lm = lm_sim.min()
            if sim_max_lm == sim_min_lm:
                # Constant row → uniform log_p_lm (no spatial information).
                # Skip the landmark contribution entirely.
                return log_p_img
            log_p_lm = wag_observation_log_likelihood_from_similarity_matrix(
                lm_sim, self.landmark_sigma
            )
        else:
            lm_finite_mask = torch.isfinite(lm_sim)
            if not lm_finite_mask.any():
                return log_p_img
            lm_finite = lm_sim[lm_finite_mask]
            sim_max_lm = lm_finite.max()
            sim_min_lm = lm_finite.min()
            if (sim_max_lm <= 0) or (sim_max_lm == sim_min_lm):
                # All-zero or constant row → landmark is uninformative.
                return log_p_img
            norm_sim = lm_sim / sim_max_lm
            r_norm = 1.0 - norm_sim
            sigma_lm = self.landmark_sigma
            log_norm_const = -torch.log(
                torch.sqrt(torch.tensor(2 * torch.pi, device=lm_sim.device))
            ) - torch.log(torch.tensor(sigma_lm, device=lm_sim.device))
            log_p_lm = log_norm_const - 0.5 * torch.square(r_norm / sigma_lm)
            # Mask non-finite landmark entries → image-only at those positions.
            log_p_lm = torch.where(lm_finite_mask, log_p_lm, torch.zeros_like(log_p_lm))

        log_p = log_p_img + log_p_lm
        _raise_if_nonfinite(log_p, pano_id, "log_p_img + log_p_lm", cls_name=type(self).__name__)
        return log_p

    @classmethod
    def from_config(
        cls,
        config: SafaPlusNormalizedLandmarkAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "SafaPlusNormalizedLandmarkAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        _assert_matrix_aligned(
            image_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.image_similarity_matrix_path,
        )
        _assert_matrix_aligned(
            landmark_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.landmark_similarity_matrix_path,
        )
        return cls(
            image_similarity_matrix=image_sim,
            landmark_similarity_matrix=landmark_sim,
            panorama_metadata=vigor_dataset._panorama_metadata,
            image_sigma=config.image_sigma,
            landmark_sigma=config.landmark_sigma,
            landmark_use_raw_residual=config.landmark_use_raw_residual,
            device=device,
        )


class SafaPlusPiecewiseLandmarkAggregator(ObservationLogLikelihoodAggregator):
    """SAFA image stream + piecewise likelihood-ratio landmark stream.

    Prototype variant of ``SafaPlusNormalizedLandmarkAggregator``. The image
    stream is identical. The landmark stream maps the per-row normalized
    residual ``r = 1 - sim / row_max`` through a frozen, calibrated
    piecewise-constant log-likelihood-ratio lookup instead of a half-normal
    Gaussian:

      ``log_p_lm[j] = landmark_lr_scale * values[bin(r_j)]``

    where ``values`` are per-bin ``log p(r|true)/p(r|neg)``. ``0`` is
    uninformative; missing / constant / all-zero landmark rows fall through to
    image-only (``log_p_lm == 0``), matching the lookup baseline. The discrete
    ``r == 0`` (argmax hit) and ``r == 1`` (miss) events land in the first /
    last bin with no special-casing.

    See ``SafaPlusPiecewiseLandmarkAggregatorConfig`` for the motivation (the
    Hungarian landmark residual is strongly bimodal — a half-normal is the
    wrong family and discards the highly-discriminative argmax-hit event while
    over-penalizing the common miss event).
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        image_sigma: float,
        landmark_log_lr_edges: list[float],
        landmark_log_lr_values: list[float],
        device: torch.device,
        landmark_lr_scale: float = 1.0,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.image_sigma = float(image_sigma)
        self.landmark_lr_scale = float(landmark_lr_scale)
        self.device = device
        # Lookup tensors live on the compute device.
        self._edges = torch.tensor(
            landmark_log_lr_edges, dtype=torch.float32, device=device
        )
        self._values = torch.tensor(
            landmark_log_lr_values, dtype=torch.float32, device=device
        )
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def _landmark_log_lr(self, r_norm: torch.Tensor) -> torch.Tensor:
        """Map normalized residuals to landmark log-LR via the frozen lookup.

        ``r_norm`` may contain non-finite entries (no landmark info); callers
        mask those to 0 afterwards. ``right=True`` so a value equal to an edge
        maps to the bin on its left; r==0 -> bin 0, r==1 -> last bin.
        """
        n_bins = self._values.numel()
        idx = torch.bucketize(r_norm, self._edges, right=True) - 1
        idx = idx.clamp(0, n_bins - 1)
        return self._values[idx] * self.landmark_lr_scale

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index].to(self.device)
        lm_sim = self.landmark_similarity_matrix[pano_index].to(self.device)

        log_p_img = wag_observation_log_likelihood_from_similarity_matrix(
            img_sim, self.image_sigma
        )
        _raise_if_nonfinite(log_p_img, pano_id, "log_p_img", cls_name=type(self).__name__)

        # Landmark stream. NaN means "no landmark info for this cell" (same
        # sparse-with-holes contract as the normalized-residual variant): fall
        # back to image-only at those positions, and image-only for whole rows
        # that are all-zero / constant / NaN.
        lm_finite_mask = torch.isfinite(lm_sim)
        if not lm_finite_mask.any():
            return log_p_img
        lm_finite = lm_sim[lm_finite_mask]
        sim_max_lm = lm_finite.max()
        sim_min_lm = lm_finite.min()
        if (sim_max_lm <= 0) or (sim_max_lm == sim_min_lm):
            return log_p_img

        r_norm = 1.0 - lm_sim / sim_max_lm
        log_p_lm = self._landmark_log_lr(r_norm)
        # Image-only (log_p_lm == 0) where the landmark entry is non-finite.
        log_p_lm = torch.where(lm_finite_mask, log_p_lm, torch.zeros_like(log_p_lm))

        log_p = log_p_img + log_p_lm
        _raise_if_nonfinite(log_p, pano_id, "log_p_img + log_p_lm", cls_name=type(self).__name__)
        return log_p

    @classmethod
    def from_config(
        cls,
        config: SafaPlusPiecewiseLandmarkAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "SafaPlusPiecewiseLandmarkAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        _assert_matrix_aligned(
            image_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.image_similarity_matrix_path,
        )
        _assert_matrix_aligned(
            landmark_sim,
            vigor_dataset._panorama_metadata,
            vigor_dataset._satellite_metadata,
            config.landmark_similarity_matrix_path,
        )
        return cls(
            image_similarity_matrix=image_sim,
            landmark_similarity_matrix=landmark_sim,
            panorama_metadata=vigor_dataset._panorama_metadata,
            image_sigma=config.image_sigma,
            landmark_log_lr_edges=config.landmark_log_lr_edges,
            landmark_log_lr_values=config.landmark_log_lr_values,
            landmark_lr_scale=config.landmark_lr_scale,
            device=device,
        )


def aggregator_from_config(
    config: AggregatorConfig,
    vigor_dataset: vd.VigorDataset,
    device: torch.device,
) -> ObservationLogLikelihoodAggregator:
    """Factory function to create aggregator from config.

    Args:
        config: Aggregator configuration
        vigor_dataset: Full VIGOR dataset (passed to inner from_config methods)
        device: Torch device for computation
    """
    if isinstance(config, SingleSimilarityMatrixAggregatorConfig):
        return SingleSimilarityMatrixAggregator.from_config(
            config, vigor_dataset, device
        )
    elif isinstance(config, ImageLandmarkPrivilegedInformationFusionConfig):
        return ImageLandmarkPrivilegedInformationFusion.from_config(
            config, vigor_dataset, device
        )
    elif isinstance(config, EntropyAdaptiveAggregatorConfig):
        return EntropyAdaptiveAggregator.from_config(config, vigor_dataset, device)
    elif isinstance(config, SafaPlusNormalizedLandmarkAggregatorConfig):
        return SafaPlusNormalizedLandmarkAggregator.from_config(
            config, vigor_dataset, device
        )
    elif isinstance(config, SafaPlusPiecewiseLandmarkAggregatorConfig):
        return SafaPlusPiecewiseLandmarkAggregator.from_config(
            config, vigor_dataset, device
        )
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def load_aggregator_config(config_path: Path) -> AggregatorConfig:
    """Load aggregator config from YAML file."""
    with open(config_path, "rb") as f:
        return msgspec.yaml.decode(
            f.read(), type=AggregatorConfig, dec_hook=msgspec_dec_hook
        )
