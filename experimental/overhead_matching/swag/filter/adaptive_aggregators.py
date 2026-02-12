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
    confidence_mode: str = "peak_sharpness"


class ProductOfExpertsAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for product-of-experts fusion aggregator."""

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    tau_img: float
    tau_lm: float


class LearnedGateAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for learned gating MLP aggregator."""

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    sigma: float
    gate_model_path: Path


# Union type for polymorphic deserialization
AggregatorConfig = (
    SingleSimilarityMatrixAggregatorConfig
    | ImageLandmarkPrivilegedInformationFusionConfig
    | EntropyAdaptiveAggregatorConfig
    | ProductOfExpertsAggregatorConfig
    | LearnedGateAggregatorConfig
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


def _build_gate_mlp() -> torch.nn.Sequential:
    """Build the gate MLP architecture: Linear(8, 32) -> ReLU -> Linear(32, 1)."""
    return torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )


def extract_gate_features(
    img_sim: torch.Tensor, lm_sim: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Extract 8 scalar features from image and landmark similarity vectors.

    Features:
        0: img_entropy - entropy of softmax(img_sim / sigma)
        1: lm_entropy - entropy of softmax(lm_sim / sigma)
        2: img_peak_sharpness - img_sim.max() - img_sim.mean()
        3: lm_peak_sharpness - lm_sim.max() - lm_sim.mean()
        4: img_max_sim - max of image similarity
        5: lm_max_sim - max of landmark similarity
        6: agreement - Pearson correlation between img_sim and lm_sim
        7: num_nonzero - fraction of landmark similarities with abs > 1e-6

    Args:
        img_sim: (num_sat_patches,) image similarity vector
        lm_sim: (num_sat_patches,) landmark similarity vector
        sigma: temperature for softmax

    Returns:
        (8,) feature tensor
    """
    # Image features
    img_log_probs = torch.nn.functional.log_softmax(img_sim / sigma, dim=0)
    img_probs = torch.exp(img_log_probs)
    img_entropy = -(img_probs * img_log_probs).sum()
    img_peak_sharpness = img_sim.max() - img_sim.mean()
    img_max_sim = img_sim.max()

    # Landmark features
    lm_log_probs = torch.nn.functional.log_softmax(lm_sim / sigma, dim=0)
    lm_probs = torch.exp(lm_log_probs)
    lm_entropy = -(lm_probs * lm_log_probs).sum()
    lm_peak_sharpness = lm_sim.max() - lm_sim.mean()
    lm_max_sim = lm_sim.max()

    # Cross features
    stacked = torch.stack([img_sim, lm_sim])
    corrcoef = torch.corrcoef(stacked)
    agreement = corrcoef[0, 1]
    # Handle NaN from corrcoef (e.g. when one vector is all zeros)
    if torch.isnan(agreement):
        agreement = torch.tensor(0.0)
    num_nonzero = (lm_sim.abs() > 1e-6).float().sum() / lm_sim.shape[0]

    return torch.tensor(
        [
            img_entropy.item(),
            lm_entropy.item(),
            img_peak_sharpness.item(),
            lm_peak_sharpness.item(),
            img_max_sim.item(),
            lm_max_sim.item(),
            agreement.item(),
            num_nonzero.item(),
        ]
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

    Computes a per-source confidence score and uses it to blend the two similarity
    vectors before converting to log-likelihoods.

    Supported confidence modes:
        - "peak_sharpness": max(sim) - mean(sim)
        - "entropy": negative entropy of softmax(sim / sigma) (higher = more confident)
        - "top2_gap": gap between top-2 similarity values
    """

    VALID_CONFIDENCE_MODES = ("peak_sharpness", "entropy", "top2_gap")

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        sigma: float,
        device: torch.device,
        confidence_mode: str = "peak_sharpness",
    ):
        if confidence_mode not in self.VALID_CONFIDENCE_MODES:
            raise ValueError(
                f"Unknown confidence_mode '{confidence_mode}'. "
                f"Must be one of {self.VALID_CONFIDENCE_MODES}"
            )
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.sigma = sigma
        self.device = device
        self.confidence_mode = confidence_mode
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def _compute_confidence(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute a scalar confidence score for a log-probability vector.

        Only considers finite values to avoid -inf entries corrupting the result.

        Args:
            log_probs: (num_patches,) log-probability vector (from log_softmax)

        Returns:
            Scalar confidence value (higher = more confident)
        """
        finite_lp = log_probs[torch.isfinite(log_probs)]
        if len(finite_lp) < 2:
            return torch.tensor(0.0)
        if self.confidence_mode == "peak_sharpness":
            return finite_lp.max() - finite_lp.mean()
        elif self.confidence_mode == "entropy":
            probs = torch.exp(finite_lp)
            entropy = -(probs * finite_lp).sum()
            return -entropy
        elif self.confidence_mode == "top2_gap":
            sorted_lp = torch.sort(finite_lp).values
            return sorted_lp[-1] - sorted_lp[-2]
        else:
            raise ValueError(f"Unknown confidence_mode: {self.confidence_mode}")

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
            config.confidence_mode,
        )


class ProductOfExpertsAggregator(ObservationLogLikelihoodAggregator):
    """Fuses image and landmark similarity matrices using a product-of-experts approach.

    Converts each similarity vector to a log-probability distribution (via
    log_softmax with per-source temperatures) and sums them in log space,
    which corresponds to multiplying the expert distributions.
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        tau_img: float,
        tau_lm: float,
        device: torch.device,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.tau_img = tau_img
        self.tau_lm = tau_lm
        self.device = device
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index]
        lm_sim = self.landmark_similarity_matrix[pano_index]

        # Use log_softmax (not wag_observation_log_likelihood_from_similarity_matrix)
        # because PoE requires normalized log-probability distributions so that
        # adding them in log space corresponds to multiplying proper distributions.
        # The Gaussian transform in wag_observation_log_likelihood is unnormalized,
        # so summing two of them wouldn't correctly implement product of experts.
        log_p_img = torch.log_softmax(img_sim / self.tau_img, dim=0)

        # Fall back to image-only when landmark data is missing (all -inf)
        lm_finite_mask = torch.isfinite(lm_sim)
        if not lm_finite_mask.any():
            combined_log_p = log_p_img
        else:
            log_p_lm = torch.log_softmax(lm_sim / self.tau_lm, dim=0)
            combined_log_p = log_p_img + log_p_lm  # product of experts in log space
            # Where landmark has -inf, use image-only to avoid eliminating patches
            combined_log_p = torch.where(lm_finite_mask, combined_log_p, log_p_img)

        return _replace_nan_with_zero(combined_log_p).to(self.device)

    @classmethod
    def from_config(
        cls,
        config: ProductOfExpertsAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "ProductOfExpertsAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        return cls(
            image_sim,
            landmark_sim,
            vigor_dataset._panorama_metadata,
            config.tau_img,
            config.tau_lm,
            device,
        )


class LearnedGateAggregator(ObservationLogLikelihoodAggregator):
    """Learned MLP gate that predicts per-panorama mixing weight.

    Uses an 8-feature vector (entropy, peak sharpness, max similarity for both
    image and landmark channels, plus cross-correlation and landmark coverage)
    to predict a scalar alpha via a small MLP. The fused similarity is then:
        fused = alpha * img_sim + (1 - alpha) * lm_sim
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        sigma: float,
        device: torch.device,
        gate_mlp: torch.nn.Sequential,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.sigma = sigma
        self.device = device
        self.gate_mlp = gate_mlp.to(device)
        self.gate_mlp.eval()
        self.feature_mean = feature_mean.to(device)
        self.feature_std = feature_std.to(device)
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])

    def _extract_features(
        self, img_sim: torch.Tensor, lm_sim: torch.Tensor
    ) -> torch.Tensor:
        """Extract and normalize 8 scalar features for the gate MLP."""
        features = extract_gate_features(img_sim, lm_sim, self.sigma)
        features = features.to(self.device)
        # Normalize using training statistics
        features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        return features

    @torch.no_grad()
    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index]
        lm_sim = self.landmark_similarity_matrix[pano_index]

        features = self._extract_features(img_sim, lm_sim)  # (8,)
        alpha = torch.sigmoid(self.gate_mlp(features))  # scalar

        # Normalize to log-probability space first (makes different scales commensurate)
        log_p_img = torch.log_softmax(img_sim.to(self.device) / self.sigma, dim=0)
        log_p_lm = torch.log_softmax(lm_sim.to(self.device) / self.sigma, dim=0)

        lm_finite_mask = torch.isfinite(lm_sim).to(self.device)
        if not lm_finite_mask.any():
            return _replace_nan_with_zero(log_p_img)

        fused_log_p = alpha * log_p_img + (1 - alpha) * log_p_lm
        fused_log_p = torch.where(lm_finite_mask, fused_log_p, log_p_img)

        return _replace_nan_with_zero(fused_log_p)

    @classmethod
    def from_config(
        cls,
        config: LearnedGateAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "LearnedGateAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)

        # Load gate model checkpoint (contains weights, feature_mean, feature_std)
        checkpoint = torch.load(
            config.gate_model_path, weights_only=False, map_location="cpu"
        )
        gate_mlp = _build_gate_mlp()
        gate_mlp.load_state_dict(checkpoint["gate_weights"])
        feature_mean = checkpoint["feature_mean"]
        feature_std = checkpoint["feature_std"]

        return cls(
            image_sim,
            landmark_sim,
            vigor_dataset._panorama_metadata,
            config.sigma,
            device,
            gate_mlp,
            feature_mean,
            feature_std,
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
    elif isinstance(config, ProductOfExpertsAggregatorConfig):
        return ProductOfExpertsAggregator.from_config(config, vigor_dataset, device)
    elif isinstance(config, LearnedGateAggregatorConfig):
        return LearnedGateAggregator.from_config(config, vigor_dataset, device)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def load_aggregator_config(config_path: Path) -> AggregatorConfig:
    """Load aggregator config from YAML file."""
    with open(config_path, "rb") as f:
        return msgspec.yaml.decode(
            f.read(), type=AggregatorConfig, dec_hook=msgspec_dec_hook
        )
