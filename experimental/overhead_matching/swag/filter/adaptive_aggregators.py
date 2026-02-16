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


# Union type for polymorphic deserialization
AggregatorConfig = (
    SingleSimilarityMatrixAggregatorConfig
    | ImageLandmarkPrivilegedInformationFusionConfig
    | EntropyAdaptiveAggregatorConfig
)


# ============ Helper Functions ============


def _replace_nan_with_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN values with 0 (no update in log-likelihood space)."""
    tensor[torch.isnan(tensor)] = 0
    return tensor


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
        log_ll = wag_observation_log_likelihood_from_similarity_matrix(
            similarity, self.sigma
        )
        return _replace_nan_with_zero(log_ll).to(self.device)

    @classmethod
    def from_config(
        cls,
        config: SingleSimilarityMatrixAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "SingleSimilarityMatrixAggregator":
        similarity_matrix = _load_similarity_matrix(config.similarity_matrix_path)
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
        return cls(
            image_sim,
            landmark_sim,
            vigor_dataset._panorama_metadata,
            config.sigma,
            device,
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
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def load_aggregator_config(config_path: Path) -> AggregatorConfig:
    """Load aggregator config from YAML file."""
    with open(config_path, "rb") as f:
        return msgspec.yaml.decode(
            f.read(), type=AggregatorConfig, dec_hook=msgspec_dec_hook
        )
