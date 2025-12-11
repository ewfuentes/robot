import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
from common.python.serialization import MSGSPEC_STRUCT_OPTS
from experimental.overhead_matching.swag.scripts.pairing import PairingDataType, PositiveAnchorSets, Pairs, collapse_anchors_to_torch
from typing import Union, Callable
from dataclasses import dataclass
import msgspec
from common.math.haversine import find_d_on_unit_circle

EARTH_RADIUS_M = 6378137.0


def compute_gps_distances(
    pano_metadata_list: list[dict],
    sat_metadata_list: list[dict],
    pairs: list[tuple[int, int]]
) -> torch.Tensor:
    """Compute GPS distances in meters for (pano_idx, sat_idx) pairs."""
    distances = []
    for pano_idx, sat_idx in pairs:
        pano_latlon = (pano_metadata_list[pano_idx]['lat'],
                       pano_metadata_list[pano_idx]['lon'])
        sat_latlon = (sat_metadata_list[sat_idx]['lat'],
                      sat_metadata_list[sat_idx]['lon'])

        distance_on_sphere = find_d_on_unit_circle(pano_latlon, sat_latlon)
        distance_m = distance_on_sphere * EARTH_RADIUS_M
        distances.append(distance_m)

    return torch.tensor(distances, dtype=torch.float32)


def compute_distance_weights(distances: torch.Tensor) -> torch.Tensor:
    """
    Convert distances to loss weights using: weight = 2 * (1 - exp(-distance/100))

    Note: This formula gives weight=0 at distance=0 and approaches weight=2
    as distance increases. Verify this behavior during testing.
    """
    return 2.0 * (1.0 - torch.exp(-distances / 100.0))


def _extract_pair_similarities(
    similarity: torch.Tensor,
    pairs: Pairs,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract pos/semipos/neg similarities from similarity matrix.

    Args:
        similarity: Similarity matrix (N_pano x N_sat)
        pairs: Pairs object containing positive, semipositive, and negative pairs
        device: Device to place empty tensors on

    Returns:
        Tuple of (pos_similarities, semipos_similarities, neg_similarities)
    """
    pos_rows = [x[0] for x in pairs.positive_pairs]
    pos_cols = [x[1] for x in pairs.positive_pairs]
    pos_similarities = similarity[pos_rows, pos_cols] if pairs.positive_pairs else \
                       torch.tensor([], device=device)

    semipos_rows = [x[0] for x in pairs.semipositive_pairs]
    semipos_cols = [x[1] for x in pairs.semipositive_pairs]
    semipos_similarities = similarity[semipos_rows, semipos_cols] if pairs.semipositive_pairs else \
                           torch.tensor([], device=device)

    neg_rows = [x[0] for x in pairs.negative_pairs]
    neg_cols = [x[1] for x in pairs.negative_pairs]
    neg_similarities = similarity[neg_rows, neg_cols] if pairs.negative_pairs else \
                       torch.tensor([], device=device)

    return pos_similarities, semipos_similarities, neg_similarities


def _compute_contrastive_term(
    similarities: torch.Tensor,
    weight: float,
    avg_similarity: float,
    device: torch.device,
    dtype: torch.dtype,
    is_positive: bool,
    distance_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute a single contrastive loss term (positive, semipositive, or negative).

    Args:
        similarities: Similarity values for this pair type
        weight: Loss weight parameter
        avg_similarity: Average similarity parameter
        device: Device for tensors
        dtype: Data type for tensors
        is_positive: True for pos/semipos pairs, False for negative pairs
        distance_weights: Optional distance weights to apply to loss (for negative pairs)

    Returns:
        Computed loss term
    """
    if len(similarities) == 0:
        return torch.tensor(0, device=device, dtype=dtype)

    sign = -1 if is_positive else 1
    raw_loss = torch.log(1 + torch.exp(sign * weight * (similarities - avg_similarity)))

    if distance_weights is not None:
        raw_loss = raw_loss * distance_weights

    return torch.mean(raw_loss) / weight


@dataclass
class LossInputs:
    similarity_matrix: torch.Tensor
    sat_embeddings_unnormalized: torch.Tensor  # NOT NORMALIZED
    pano_embeddings_unnormalized: torch.Tensor  # NOT NORMALIZED
    pairing_data: PairingDataType
    # Optional metadata for distance-based losses
    pano_metadata: list[dict] | None = None
    sat_metadata: list[dict] | None = None


class PairwiseContrastiveLossConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    positive_weight: float
    avg_positive_similarity: float

    semipositive_weight: float
    avg_semipositive_similarity: float

    negative_weight: float
    avg_negative_similarity: float

    use_distance_weighting: bool = False


class BatchUniformityLossConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    batch_uniformity_weight: float
    batch_uniformity_hinge_location: float


class InfoNCELossConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    max_num_negative_pairs: int  # if 0, uses all negative pairs
    negative_scale: float = 1.0
    scale_negative_by_num_items: bool = False
    use_pano_as_anchor: bool = False


class SphericalEmbeddingConstraintLossConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    weight_scale: float


LossConfig = Union[
    PairwiseContrastiveLossConfig,
    InfoNCELossConfig,
    SphericalEmbeddingConstraintLossConfig,
    BatchUniformityLossConfig
]


def compute_pairwise_loss(
    loss_inputs: LossInputs,
    pairwise_loss_config: PairwiseContrastiveLossConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Implementation for pairwise contrastive loss with optional distance weighting.

    Args:
        loss_inputs: Loss inputs containing similarity matrix, embeddings, and metadata
        config: Configuration for the loss
        use_distance_weighting: Whether to apply distance-based weighting to negative pairs

    Returns:
        Tuple of (total_loss, aux_data_dict)
    """
    assert isinstance(loss_inputs.pairing_data, Pairs), \
        "PairwiseContrastiveLoss requires Pairs pairing data"

    if pairwise_loss_config.use_distance_weighting:
        if loss_inputs.pano_metadata is None or loss_inputs.sat_metadata is None:
            raise ValueError(
                "Distance-weighted loss requires pano_metadata and sat_metadata in LossInputs"
            )

    similarity = loss_inputs.similarity_matrix
    pairs = loss_inputs.pairing_data
    device = similarity.device

    # Extract similarities using helper
    pos_sim, semipos_sim, neg_sim = _extract_pair_similarities(similarity, pairs, device)

    # Compute positive loss
    pos_loss = _compute_contrastive_term(
        pos_sim, pairwise_loss_config.positive_weight, pairwise_loss_config.avg_positive_similarity,
        device, similarity.dtype, is_positive=True
    )

    # Compute semipositive loss
    semipos_loss = _compute_contrastive_term(
        semipos_sim, pairwise_loss_config.semipositive_weight, pairwise_loss_config.avg_semipositive_similarity,
        device, similarity.dtype, is_positive=True
    )

    # Compute negative loss (with optional distance weighting)
    distance_weights = None
    neg_distances = None
    if pairwise_loss_config.use_distance_weighting and len(pairs.negative_pairs) > 0:
        distances = compute_gps_distances(
            loss_inputs.pano_metadata,
            loss_inputs.sat_metadata,
            pairs.negative_pairs
        ).to(device)
        distance_weights = compute_distance_weights(distances).to(device)
        neg_distances = distances.detach()

    neg_loss = _compute_contrastive_term(
        neg_sim, pairwise_loss_config.negative_weight, pairwise_loss_config.avg_negative_similarity,
        device, similarity.dtype, is_positive=False,
        distance_weights=distance_weights
    )

    # Prepare auxiliary data
    aux_data = {
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "semipos_loss": semipos_loss,
        "pos_sim": pos_sim,
        "semipos_sim": semipos_sim,
        "neg_sim": neg_sim,
    }

    if neg_distances is not None:
        aux_data["neg_distances_m"] = neg_distances
        aux_data["neg_distance_mean_m"] = neg_distances.mean()
        aux_data["neg_distance_max_m"] = neg_distances.max()

    return pos_loss + neg_loss + semipos_loss, aux_data


def compute_batch_uniformity_loss(loss_inputs: LossInputs, batch_uniformity_loss_config: BatchUniformityLossConfig):
    # Compute a batch uniformity loss, different panoramas/satellites
    # should have different embeddings
    assert loss_inputs.sat_embeddings_unnormalized.shape[
        1] == 1, f"Batch uniformity loss does not support multiple embeddings, got emb of shape {loss_inputs.sat_embeddings_unnormalized.shape}"
    assert loss_inputs.pano_embeddings_unnormalized.shape[
        1] == 1, f"Batch uniformity loss does not support multiple embeddings, got emb of shape {loss_inputs.pano_embeddings_unnormalized.shape}"
    sat_embeddings_norm = F.normalize(loss_inputs.sat_embeddings_unnormalized, dim=-1).squeeze(1)
    pano_embeddings_norm = F.normalize(loss_inputs.pano_embeddings_unnormalized, dim=-1).squeeze(1)
    rolled_sat_embeddings = torch.roll(sat_embeddings_norm, 1, dims=0)
    rolled_pano_embeddings = torch.roll(pano_embeddings_norm, 1, dims=0)

    def mean_hinge_loss(similarities):
        shifted_loss = torch.abs(similarities) - \
            batch_uniformity_loss_config.batch_uniformity_hinge_location
        relud_loss = torch.mean(torch.nn.functional.relu(shifted_loss))
        return batch_uniformity_loss_config.batch_uniformity_weight * relud_loss

    sat_similarity = torch.einsum(
        "ad,ad->a", sat_embeddings_norm, rolled_sat_embeddings)
    pano_similarity = torch.einsum(
        "ad,ad->a", pano_embeddings_norm, rolled_pano_embeddings)

    sat_uniformity_loss = mean_hinge_loss(sat_similarity)
    pano_uniformity_loss = mean_hinge_loss(pano_similarity)
    return sat_uniformity_loss + pano_uniformity_loss, {
        'sat_uniformity_loss': sat_uniformity_loss,
        'pano_uniformity_loss': pano_uniformity_loss,
    }


def compute_info_nce_loss(
    loss_inputs: LossInputs,
    info_nce_config: InfoNCELossConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert isinstance(loss_inputs.pairing_data,
                      PositiveAnchorSets), "InfoNCELoss requires PositiveAnchorSets pairing data"
    device = loss_inputs.similarity_matrix.device
    similarity_matrix = loss_inputs.similarity_matrix  # N_pano x N_sat
    if not info_nce_config.use_pano_as_anchor:
        # N_anchor x N_non_anchor
        similarity_matrix = torch.transpose(similarity_matrix, 0, 1)

    anchor_set = loss_inputs.pairing_data

    anchor_indices, pos_semipos_indices = collapse_anchors_to_torch(anchor_set)
    anchor_indices, pos_semipos_indices = anchor_indices.to(device), pos_semipos_indices.to(device)

    positive_term = similarity_matrix[anchor_indices, pos_semipos_indices]  # num_loss_terms

    # -inf will come out to log(0 + ...) in logsumexp, so it is a noop assuming at least one term, which pos guarentees
    negative_sim_matrix = torch.clone(similarity_matrix)
    negative_sim_matrix[anchor_indices, pos_semipos_indices] = -torch.inf

    negative_term = negative_sim_matrix[anchor_indices]  # num loss terms x N non anchor
    if info_nce_config.max_num_negative_pairs != 0:
        # num loss terms x max_num_negative_pairs
        negative_term = negative_term.topk(info_nce_config.max_num_negative_pairs, dim=1).values

    neg_scale = info_nce_config.negative_scale / (torch.count_nonzero(torch.isfinite(
        negative_term), dim=1) if info_nce_config.scale_negative_by_num_items else torch.tensor([1]))
    log_neg_scale = torch.log(neg_scale).to(device).unsqueeze(1)
    loss = info_nce_config.negative_scale * torch.logsumexp(torch.cat([positive_term.unsqueeze(
        1), log_neg_scale + negative_term], dim=1), dim=-1) - positive_term

    with torch.no_grad():
        aux = dict(
            num_batch_items=loss.shape[0],
            pos_sim=positive_term,
            neg_sim=negative_term[torch.isfinite(negative_term)],
        )

    return loss.mean(), aux


def compute_spherical_embedding_constraint_loss(
    loss_inputs: LossInputs,
    sec_config: SphericalEmbeddingConstraintLossConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """https://arxiv.org/pdf/2011.02785"""
    sat_norms = torch.linalg.norm(loss_inputs.sat_embeddings_unnormalized, dim=2).flatten()
    pano_norms = torch.linalg.norm(loss_inputs.pano_embeddings_unnormalized, dim=2).flatten()
    all_norms = torch.cat([sat_norms, pano_norms])
    mu = all_norms.mean()
    L_sec = torch.pow(all_norms - mu, 2).mean()
    final_loss = sec_config.weight_scale * L_sec
    return final_loss, {"sec_aux_loss": final_loss, "num_embeddings": all_norms.numel()}


LossFunctionType = Callable[[LossInputs], tuple[torch.Tensor, dict[str, torch.Tensor | int]]]


def create_losses_from_loss_config_list(
    loss_configs: list[LossConfig]
) -> list[LossFunctionType]:
    loss_functions = []
    for loss_config in loss_configs:
        if isinstance(loss_config, PairwiseContrastiveLossConfig):
            loss_functions.append(lambda x, config=loss_config: compute_pairwise_loss(x, config))
        elif isinstance(loss_config, InfoNCELossConfig):
            loss_functions.append(lambda x, config=loss_config: compute_info_nce_loss(x, config))
        elif isinstance(loss_config, SphericalEmbeddingConstraintLossConfig):
            loss_functions.append(
                lambda x, config=loss_config: compute_spherical_embedding_constraint_loss(x, config))
        elif isinstance(loss_config, BatchUniformityLossConfig):
            loss_functions.append(lambda x, config=loss_config: compute_batch_uniformity_loss(x, config))
        else:
            raise ValueError(f"Unknown loss config type: {type(loss_config)}")
    return loss_functions


def compute_loss(sat_embeddings: torch.Tensor,  # N_sat x n_emb_sat x D_emb
                 pano_embeddings: torch.Tensor,  # N_pano x n_emb_pano x D_emb
                 similarity: torch.Tensor,  # N_pano x N_sat
                 pairing_data: PairingDataType,
                 loss_functions: list[LossFunctionType],
                 pano_metadata: list[dict] | None = None,
                 sat_metadata: list[dict] | None = None,
                 ) -> dict[str, torch.Tensor | int | float]:
    loss = 0.0

    loss_input = LossInputs(
        similarity_matrix=similarity,
        sat_embeddings_unnormalized=sat_embeddings,
        pano_embeddings_unnormalized=pano_embeddings,
        pairing_data=pairing_data,
        pano_metadata=pano_metadata,
        sat_metadata=sat_metadata,
    )
    aux_info = {}
    for loss_fn in loss_functions:
        l, new_aux = loss_fn(loss_input)
        loss += l
        overlapping_keys = set(aux_info.keys()).intersection(set(new_aux.keys()))
        if len(overlapping_keys):
            print("Warning: overwriting overlapping keys in loss aux info: ", overlapping_keys)
        aux_info |= new_aux
    assert "loss" not in aux_info
    return {"loss": loss} | aux_info
