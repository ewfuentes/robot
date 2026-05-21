import common.torch.load_torch_deps
import warnings
import torch
import torch.nn.functional as F
from common.python.serialization import MSGSPEC_STRUCT_OPTS
from experimental.overhead_matching.swag.scripts.pairing import PairingDataType, PositiveAnchorSets, Pairs, collapse_anchors_to_torch
from typing import Any, Union, Callable
from dataclasses import dataclass, field
import msgspec


@dataclass
class LossInputs:
    similarity_matrix: torch.Tensor
    sat_embeddings_unnormalized: torch.Tensor  # NOT NORMALIZED
    pano_embeddings_unnormalized: torch.Tensor  # NOT NORMALIZED
    pairing_data: PairingDataType
    # Per-side extractor outputs (the second return value from a SwagPatchEmbedding
    # forward). Populated by `compute_loss` when the caller passes them; required
    # for losses that read a frozen teacher token like `DistillationLossConfig`.
    sat_extractor_outputs: dict[str, Any] = field(default_factory=dict)
    pano_extractor_outputs: dict[str, Any] = field(default_factory=dict)


class PairwiseContrastiveLossConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    positive_weight: float
    avg_positive_similarity: float

    semipositive_weight: float
    avg_semipositive_similarity: float

    negative_weight: float
    avg_negative_similarity: float


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


class CrossViewInfoNCELossConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Symmetric cross-view InfoNCE between sat and pano student embeddings.

    Multi-positive: any (pano_i, sat_j) listed in `pairs.positive_pairs` is a
    positive for both directions. Semipositives are masked out of the softmax
    denominator (treated as "neither positive nor negative"), matching the
    behavior of train_safa_distill's `_cross_view_info_nce`.

    Final loss = `weight_scale` * 0.5 * (loss_pano_to_sat + loss_sat_to_pano).
    Requires `Pairs` pairing data.
    """
    weight_scale: float = 1.0
    temperature: float = 0.07


class DistillationLossConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    """Cosine distillation loss against a frozen teacher token.

    The teacher is `{sat,pano}_extractor_outputs[extractor_name].features`, expected
    to be the per-image embedding emitted by a frozen extractor (e.g. SAFA). The
    student is the model output embedding. Both are L2-normalized along the
    feature dim before computing 1 - cosine similarity. Per-side weights apply
    on top of the shared `weight_scale` factor.
    """
    extractor_name: str
    weight_scale: float = 1.0
    weight_sat: float = 1.0
    weight_pano: float = 1.0


LossConfig = Union[
    PairwiseContrastiveLossConfig,
    InfoNCELossConfig,
    SphericalEmbeddingConstraintLossConfig,
    BatchUniformityLossConfig,
    DistillationLossConfig,
    CrossViewInfoNCELossConfig,
]


def compute_pairwise_loss(
    loss_inputs: LossInputs,
    pairwise_loss_config: PairwiseContrastiveLossConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert isinstance(loss_inputs.pairing_data,
                      Pairs), "PairwiseContrastiveLoss requires Pairs pairing data"
    similarity, pairs = loss_inputs.similarity_matrix, loss_inputs.pairing_data
    pos_rows = [x[0] for x in pairs.positive_pairs]
    pos_cols = [x[1] for x in pairs.positive_pairs]
    pos_similarities = similarity[pos_rows, pos_cols]

    semipos_rows = [x[0] for x in pairs.semipositive_pairs]
    semipos_cols = [x[1] for x in pairs.semipositive_pairs]
    semipos_similarities = similarity[semipos_rows, semipos_cols]

    neg_rows = [x[0] for x in pairs.negative_pairs]
    neg_cols = [x[1] for x in pairs.negative_pairs]
    neg_similarities = similarity[neg_rows, neg_cols]

    # Compute Loss
    POS_WEIGHT = pairwise_loss_config.positive_weight
    AVG_POS_SIMILARITY = pairwise_loss_config.avg_positive_similarity
    if len(pairs.positive_pairs):
        pos_loss = torch.log(
            1 + torch.exp(-POS_WEIGHT * (pos_similarities - AVG_POS_SIMILARITY)))
        pos_loss = torch.mean(pos_loss) / POS_WEIGHT
    else:
        pos_loss = torch.tensor(0, device=similarity.device, dtype=similarity.dtype)

    SEMIPOS_WEIGHT = pairwise_loss_config.semipositive_weight
    AVG_SEMIPOS_SIMILARITY = pairwise_loss_config.avg_semipositive_similarity
    if len(pairs.semipositive_pairs):
        semipos_loss = torch.log(
            1 + torch.exp(-SEMIPOS_WEIGHT * (
                semipos_similarities - AVG_SEMIPOS_SIMILARITY)))
        semipos_loss = torch.mean(semipos_loss) / SEMIPOS_WEIGHT
    else:
        semipos_loss = torch.tensor(0, device=similarity.device, dtype=similarity.dtype)

    NEG_WEIGHT = pairwise_loss_config.negative_weight
    AVG_NEG_SIMILARITY = pairwise_loss_config.avg_negative_similarity
    if len(pairs.negative_pairs):
        neg_loss = torch.log(
            1 + torch.exp(NEG_WEIGHT * (neg_similarities - AVG_NEG_SIMILARITY)))
        neg_loss = torch.mean(neg_loss) / NEG_WEIGHT
    else:
        neg_loss = torch.tensor(0, device=similarity.device, dtype=similarity.dtype)

    aux_data = {
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "semipos_loss": semipos_loss,
        "pos_sim": pos_similarities,
        "semipos_sim": semipos_similarities,
        "neg_sim": neg_similarities,
    }

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


def compute_cross_view_info_nce_loss(
    loss_inputs: LossInputs,
    config: CrossViewInfoNCELossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Symmetric multi-positive InfoNCE with semipositives masked from the denominator.

    Ported from train_safa_distill.py:_cross_view_info_nce. Operates directly on
    student embeddings (post-normalization) rather than on a precomputed
    similarity matrix, so it sees the cross-view (sat, pano) similarity used by
    the original.
    """
    assert isinstance(loss_inputs.pairing_data, Pairs), \
        "CrossViewInfoNCELoss requires Pairs pairing data"
    pairs = loss_inputs.pairing_data
    p = F.normalize(loss_inputs.pano_embeddings_unnormalized, dim=-1).squeeze(1)  # (B, D)
    s = F.normalize(loss_inputs.sat_embeddings_unnormalized, dim=-1).squeeze(1)   # (B, D)
    B = p.shape[0]

    logits_p2s = (p @ s.T) / config.temperature  # (B, B)
    logits_s2p = logits_p2s.T

    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=p.device)
    semipos_mask = torch.zeros(B, B, dtype=torch.bool, device=p.device)
    for pi, si in pairs.positive_pairs:
        pos_mask[pi, si] = True
    for pi, si in pairs.semipositive_pairs:
        semipos_mask[pi, si] = True

    NEG_INF = torch.finfo(logits_p2s.dtype).min
    logits_p2s_m = logits_p2s.masked_fill(semipos_mask, NEG_INF)
    logits_s2p_m = logits_s2p.masked_fill(semipos_mask.T, NEG_INF)

    log_probs_p2s = F.log_softmax(logits_p2s_m, dim=1)
    log_probs_s2p = F.log_softmax(logits_s2p_m, dim=1)

    def _multi_pos_loss(log_probs, mask):
        per_row_pos = mask.float()
        n_pos_per_row = per_row_pos.sum(dim=1)
        n_pos_per_row_safe = n_pos_per_row.clamp(min=1)
        per_row = -(per_row_pos * log_probs).sum(dim=1) / n_pos_per_row_safe
        valid = n_pos_per_row > 0
        if valid.any():
            return per_row[valid].mean()
        return torch.tensor(0.0, device=p.device)

    loss_p2s = _multi_pos_loss(log_probs_p2s, pos_mask)
    loss_s2p = _multi_pos_loss(log_probs_s2p, pos_mask.T)
    symmetric = 0.5 * (loss_p2s + loss_s2p)
    return config.weight_scale * symmetric, {
        "cross_view_loss_p2s": loss_p2s,
        "cross_view_loss_s2p": loss_s2p,
    }


def compute_distillation_loss(
    loss_inputs: LossInputs,
    distill_config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """1 - cos(student, teacher) per side, averaged. Teacher comes from a frozen extractor."""
    extractor_name = distill_config.extractor_name
    if extractor_name not in loss_inputs.sat_extractor_outputs:
        raise KeyError(
            f"DistillationLoss: extractor {extractor_name!r} missing from sat_extractor_outputs")
    if extractor_name not in loss_inputs.pano_extractor_outputs:
        raise KeyError(
            f"DistillationLoss: extractor {extractor_name!r} missing from pano_extractor_outputs")

    teacher_sat = loss_inputs.sat_extractor_outputs[extractor_name].features.detach()
    teacher_pano = loss_inputs.pano_extractor_outputs[extractor_name].features.detach()
    teacher_sat = F.normalize(teacher_sat, dim=-1)
    teacher_pano = F.normalize(teacher_pano, dim=-1)
    student_sat = F.normalize(loss_inputs.sat_embeddings_unnormalized, dim=-1)
    student_pano = F.normalize(loss_inputs.pano_embeddings_unnormalized, dim=-1)

    loss_sat = (1.0 - F.cosine_similarity(student_sat, teacher_sat, dim=-1)).mean()
    loss_pano = (1.0 - F.cosine_similarity(student_pano, teacher_pano, dim=-1)).mean()
    loss = distill_config.weight_scale * (
        distill_config.weight_sat * loss_sat + distill_config.weight_pano * loss_pano)
    return loss, {
        "distill_loss_sat": loss_sat,
        "distill_loss_pano": loss_pano,
    }


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
        elif isinstance(loss_config, DistillationLossConfig):
            loss_functions.append(lambda x, config=loss_config: compute_distillation_loss(x, config))
        elif isinstance(loss_config, CrossViewInfoNCELossConfig):
            loss_functions.append(
                lambda x, config=loss_config: compute_cross_view_info_nce_loss(x, config))
        else:
            raise ValueError(f"Unknown loss config type: {type(loss_config)}")
    return loss_functions


def compute_loss(sat_embeddings: torch.Tensor,  # N_sat x n_emb_sat x D_emb
                 pano_embeddings: torch.Tensor,  # N_pano x n_emb_pano x D_emb
                 similarity: torch.Tensor,  # N_pano x N_sat
                 pairing_data: PairingDataType,
                 loss_functions: list[LossFunctionType],
                 sat_extractor_outputs: dict[str, Any] | None = None,
                 pano_extractor_outputs: dict[str, Any] | None = None,
                 ) -> dict[str, torch.Tensor | int | float]:
    loss = 0.0

    loss_input = LossInputs(
        similarity_matrix=similarity,
        sat_embeddings_unnormalized=sat_embeddings,
        pano_embeddings_unnormalized=pano_embeddings,
        pairing_data=pairing_data,
        sat_extractor_outputs=sat_extractor_outputs or {},
        pano_extractor_outputs=pano_extractor_outputs or {},
    )
    aux_info = {}
    for loss_fn in loss_functions:
        l, new_aux = loss_fn(loss_input)
        loss += l
        overlapping_keys = set(aux_info.keys()).intersection(set(new_aux.keys()))
        if len(overlapping_keys):
            warnings.warn(f"Overwriting overlapping keys in loss aux info: {overlapping_keys}")
        aux_info |= new_aux
    assert "loss" not in aux_info
    return {"loss": loss} | aux_info
