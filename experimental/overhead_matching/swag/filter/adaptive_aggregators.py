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
    use_log_softmax: bool = False


class ImageLandmarkPrivilegedInformationFusionConfig(
    msgspec.Struct, **MSGSPEC_STRUCT_OPTS
):
    """Config for privileged information fusion aggregator."""

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    sigma: float
    include_semipositive: bool = True
    use_log_softmax_landmarks: bool = False


class EntropyAdaptiveAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for entropy-adaptive weighted fusion aggregator."""

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    sigma: float
    use_log_softmax_landmarks: bool = False
    landmark_temperature_k: int = 0


class ProductOfExpertsAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for Product-of-Experts fusion: log_p_img + gamma * log_p_lm."""

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    sigma: float
    gamma: float = 0.2
    landmark_temperature_k: int = 10


class TopKRerankerAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for top-K reranking fusion.

    Landmarks only rerank within image's top-K candidates.
    All other patches keep their image-only log-likelihood unchanged.
    """

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    sigma: float
    gamma: float = 0.2
    rerank_k: int = 100
    landmark_temperature_k: int = 10


class SequentialFusionAggregatorConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Config for sequential observation fusion.

    Applies two independent Gaussian-kernel log-likelihoods and sums them.
    Each source has its own sigma. No softmax, no temperature, no gamma.
    """

    image_similarity_matrix_path: Path
    landmark_similarity_matrix_path: Path
    image_sigma: float
    landmark_sigma: float


# Union type for polymorphic deserialization
AggregatorConfig = (
    SingleSimilarityMatrixAggregatorConfig
    | ImageLandmarkPrivilegedInformationFusionConfig
    | EntropyAdaptiveAggregatorConfig
    | ProductOfExpertsAggregatorConfig
    | TopKRerankerAggregatorConfig
    | SequentialFusionAggregatorConfig
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
        use_log_softmax: bool = False,
    ):
        self.similarity_matrix = similarity_matrix
        self.sigma = sigma
        self.device = device
        self.use_log_softmax = use_log_softmax
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])
        if use_log_softmax:
            self.global_std = similarity_matrix.std().item()
            print(f"  log_softmax mode: global_std={self.global_std:.4f}")

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)
        similarity = self.similarity_matrix[pano_index]
        if self.use_log_softmax:
            log_ll = torch.log_softmax(similarity / self.global_std, dim=0)
        else:
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
            similarity_matrix,
            vigor_dataset._panorama_metadata,
            config.sigma,
            device,
            config.use_log_softmax,
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
        use_log_softmax_landmarks: bool = False,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.panorama_metadata = panorama_metadata
        self.sigma = sigma
        self.device = device
        self.include_semipositive = include_semipositive
        self.use_log_softmax_landmarks = use_log_softmax_landmarks
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])
        if use_log_softmax_landmarks:
            self.landmark_global_std = landmark_similarity_matrix.std().item()
            print(
                f"  log_softmax landmarks: global_std={self.landmark_global_std:.4f}"
            )

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        image_sim = self.image_similarity_matrix[pano_index]
        landmark_sim = self.landmark_similarity_matrix[pano_index]

        row = self.panorama_metadata.iloc[pano_index]
        true_indices = list(row["positive_satellite_idxs"])
        if self.include_semipositive:
            true_indices.extend(row["semipositive_satellite_idxs"])

        if self.use_log_softmax_landmarks:
            # Convert each source to log-probs independently, then fuse
            image_log_ll = wag_observation_log_likelihood_from_similarity_matrix(
                image_sim, self.sigma
            )
            image_log_probs = image_log_ll - torch.logsumexp(image_log_ll, dim=0)
            landmark_log_probs = torch.log_softmax(
                landmark_sim / self.landmark_global_std, dim=0
            )
            fused = image_log_probs.clone()
            if true_indices:
                idx = torch.tensor(
                    true_indices, dtype=torch.long, device=image_sim.device
                )
                fused[idx] = torch.maximum(
                    image_log_probs[idx], landmark_log_probs[idx]
                )
            return _replace_nan_with_zero(fused).to(self.device)
        else:
            fused_similarity = image_sim.clone()
            if true_indices:
                idx = torch.tensor(
                    true_indices, dtype=torch.long, device=image_sim.device
                )
                fused_similarity[idx] = torch.maximum(
                    image_sim[idx], landmark_sim[idx]
                )
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
            config.use_log_softmax_landmarks,
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
        use_log_softmax_landmarks: bool = False,
        landmark_temperature_k: int = 0,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.sigma = sigma
        self.device = device
        self.use_log_softmax_landmarks = use_log_softmax_landmarks
        self.landmark_temperature_k = landmark_temperature_k
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])
        if use_log_softmax_landmarks:
            if landmark_temperature_k > 0:
                print(
                    f"  EA log_softmax landmarks: per-row temperature (top1-top{landmark_temperature_k} gap)"
                )
            else:
                self.landmark_global_std = landmark_similarity_matrix.std().item()
                print(
                    f"  EA log_softmax landmarks: global_std={self.landmark_global_std:.4f}"
                )

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

        if self.use_log_softmax_landmarks:
            if self.landmark_temperature_k > 0:
                sorted_lm, _ = torch.sort(lm_sim[lm_finite_mask], descending=True)
                k = min(self.landmark_temperature_k, len(sorted_lm))
                T_lm = max((sorted_lm[0] - sorted_lm[k - 1]).item(), 1e-6)
                log_p_lm = torch.log_softmax(lm_sim / T_lm, dim=0)
            else:
                log_p_lm = torch.log_softmax(lm_sim / self.landmark_global_std, dim=0)
        else:
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
            config.use_log_softmax_landmarks,
            config.landmark_temperature_k,
        )


class ProductOfExpertsAggregator(ObservationLogLikelihoodAggregator):
    """Fuses image and landmark log-probs additively: log_p_img + gamma * log_p_lm.

    Image is always the base signal. Landmarks nudge the distribution with
    strength controlled by gamma. No confidence estimation needed.
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        sigma: float,
        device: torch.device,
        gamma: float = 0.2,
        landmark_temperature_k: int = 10,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.sigma = sigma
        self.device = device
        self.gamma = gamma
        self.landmark_temperature_k = landmark_temperature_k
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])
        print(f"  PoE fusion: gamma={gamma}, landmark T from top1-top{landmark_temperature_k} gap")

    def _landmark_log_probs(self, lm_sim: torch.Tensor, lm_finite_mask: torch.Tensor) -> torch.Tensor:
        """Compute landmark log-probs with per-row temperature."""
        sorted_lm, _ = torch.sort(lm_sim[lm_finite_mask], descending=True)
        k = min(self.landmark_temperature_k, len(sorted_lm))
        T_lm = max((sorted_lm[0] - sorted_lm[k - 1]).item(), 1e-6)
        return torch.log_softmax(lm_sim / T_lm, dim=0)

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index]
        lm_sim = self.landmark_similarity_matrix[pano_index]

        log_p_img = torch.log_softmax(img_sim / self.sigma, dim=0)

        lm_finite_mask = torch.isfinite(lm_sim)
        if not lm_finite_mask.any():
            return _replace_nan_with_zero(log_p_img).to(self.device)

        log_p_lm = self._landmark_log_probs(lm_sim, lm_finite_mask)

        # Additive fusion in log-space (product of experts in prob-space)
        fused = log_p_img + self.gamma * log_p_lm
        # Where landmark has -inf, use image-only
        fused = torch.where(lm_finite_mask, fused, log_p_img)

        return _replace_nan_with_zero(fused).to(self.device)

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
            config.sigma,
            device,
            config.gamma,
            config.landmark_temperature_k,
        )


class TopKRerankerAggregator(ObservationLogLikelihoodAggregator):
    """Fuses image and landmark by letting landmarks rerank only image's top-K.

    Image log-probs form the base distribution. Landmark log-probs are added
    (PoE-style, scaled by gamma) only to the top-K image candidates. All other
    patches keep their image-only log-likelihood, preventing landmarks from
    introducing noise from outside the image shortlist.
    """

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        sigma: float,
        device: torch.device,
        gamma: float = 0.2,
        rerank_k: int = 100,
        landmark_temperature_k: int = 10,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.sigma = sigma
        self.device = device
        self.gamma = gamma
        self.rerank_k = rerank_k
        self.landmark_temperature_k = landmark_temperature_k
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])
        print(
            f"  TopK reranker: gamma={gamma}, K={rerank_k}, "
            f"landmark T from top1-top{landmark_temperature_k} gap"
        )

    def _landmark_log_probs(
        self, lm_sim: torch.Tensor, lm_finite_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute landmark log-probs with per-row temperature."""
        sorted_lm, _ = torch.sort(lm_sim[lm_finite_mask], descending=True)
        k = min(self.landmark_temperature_k, len(sorted_lm))
        T_lm = max((sorted_lm[0] - sorted_lm[k - 1]).item(), 1e-6)
        return torch.log_softmax(lm_sim / T_lm, dim=0)

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index]
        lm_sim = self.landmark_similarity_matrix[pano_index]

        log_p_img = torch.log_softmax(img_sim / self.sigma, dim=0)

        lm_finite_mask = torch.isfinite(lm_sim)
        if not lm_finite_mask.any():
            return _replace_nan_with_zero(log_p_img).to(self.device)

        log_p_lm = self._landmark_log_probs(lm_sim, lm_finite_mask)

        # Only apply landmark nudge within image's top-K candidates
        _, top_k_idx = torch.topk(img_sim, self.rerank_k)
        fused = log_p_img.clone()
        fused[top_k_idx] += self.gamma * log_p_lm[top_k_idx]

        return _replace_nan_with_zero(fused).to(self.device)

    @classmethod
    def from_config(
        cls,
        config: TopKRerankerAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "TopKRerankerAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        return cls(
            image_sim,
            landmark_sim,
            vigor_dataset._panorama_metadata,
            config.sigma,
            device,
            config.gamma,
            config.rerank_k,
            config.landmark_temperature_k,
        )


class SequentialFusionAggregator(ObservationLogLikelihoodAggregator):
    """Fuses image and landmark by summing independent Gaussian-kernel log-likelihoods."""

    def __init__(
        self,
        image_similarity_matrix: torch.Tensor,
        landmark_similarity_matrix: torch.Tensor,
        panorama_metadata: pd.DataFrame,
        image_sigma: float,
        landmark_sigma: float,
        device: torch.device,
    ):
        self.image_similarity_matrix = image_similarity_matrix
        self.landmark_similarity_matrix = landmark_similarity_matrix
        self.image_sigma = image_sigma
        self.landmark_sigma = landmark_sigma
        self.device = device
        self._pano_id_index = pd.Index(panorama_metadata["pano_id"])
        print(f"  Sequential fusion: image_sigma={image_sigma}, landmark_sigma={landmark_sigma}")

    def __call__(self, pano_id: str) -> torch.Tensor:
        pano_index = self._pano_id_index.get_loc(pano_id)

        img_sim = self.image_similarity_matrix[pano_index]
        lm_sim = self.landmark_similarity_matrix[pano_index]

        log_ll_img = wag_observation_log_likelihood_from_similarity_matrix(img_sim, self.image_sigma)
        log_ll_lm = wag_observation_log_likelihood_from_similarity_matrix(lm_sim, self.landmark_sigma)

        fused = log_ll_img + log_ll_lm
        return _replace_nan_with_zero(fused).to(self.device)

    @classmethod
    def from_config(
        cls,
        config: SequentialFusionAggregatorConfig,
        vigor_dataset: vd.VigorDataset,
        device: torch.device,
    ) -> "SequentialFusionAggregator":
        image_sim = _load_similarity_matrix(config.image_similarity_matrix_path)
        landmark_sim = _load_similarity_matrix(config.landmark_similarity_matrix_path)
        return cls(
            image_sim,
            landmark_sim,
            vigor_dataset._panorama_metadata,
            config.image_sigma,
            config.landmark_sigma,
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
    elif isinstance(config, ProductOfExpertsAggregatorConfig):
        return ProductOfExpertsAggregator.from_config(config, vigor_dataset, device)
    elif isinstance(config, TopKRerankerAggregatorConfig):
        return TopKRerankerAggregator.from_config(config, vigor_dataset, device)
    elif isinstance(config, SequentialFusionAggregatorConfig):
        return SequentialFusionAggregator.from_config(config, vigor_dataset, device)
    else:
        raise ValueError(f"Unknown config type: {type(config)}")


def load_aggregator_config(config_path: Path) -> AggregatorConfig:
    """Load aggregator config from YAML file."""
    with open(config_path, "rb") as f:
        return msgspec.yaml.decode(
            f.read(), type=AggregatorConfig, dec_hook=msgspec_dec_hook
        )
