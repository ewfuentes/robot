import common.torch.load_torch_deps
from common.torch.load_and_save_models import save_model
import torch
import torch.nn.functional as F
from common.python.serialization import dataclass_to_dict, flatten_dict
from experimental.overhead_matching.swag.scripts.distances import (
    DistanceTypes, distance_from_type
)
from experimental.overhead_matching.swag.scripts.pairing import PairingDataType, PositiveAnchorSets, Pairs, collapse_anchors_to_torch
from typing import Union
from dataclasses import dataclass
from pprint import pprint
import msgspec

STRUCT_OPTS = {
    "tag": True,
    "tag_field": "kind",
    "frozen": True
}


@dataclass
class LossInputs:
    similarity_matrix: torch.Tensor
    sat_embeddings: torch.Tensor  # NOT NORMALIZED
    pano_embeddings: torch.Tensor  # NOT NORMALIZED
    pairing_data: PairingDataType


class PairwiseContrastiveLoss(msgspec.Struct, **STRUCT_OPTS):
    positive_weight: float
    avg_positive_similarity: float

    semipositive_weight: float
    avg_semipositive_similarity: float

    negative_weight: float
    avg_negative_similarity: float

@dataclass
class BatchUniformityLossConfig:
    batch_uniformity_weight: float
    batch_uniformity_hinge_location: float

class InfoNCELoss(msgspec.Struct, **STRUCT_OPTS):
    max_num_negative_pairs: int  # if 0, uses all negative pairs
    negative_scale: float = 1.0
    scale_negative_by_num_items: bool = False
    use_pano_as_anchor: bool = False

class SphericalEmbeddingConstraintLoss(msgspec.Struct, **STRUCT_OPTS):
    weight_scale: float 


LossConfig = Union[PairwiseContrastiveLoss, InfoNCELoss, SphericalEmbeddingConstraintLoss]



def compute_pairwise_loss(
    loss_inputs: LossInputs,
    pairwise_loss_config: PairwiseContrastiveLoss
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
        "semipos_loss": semipos_loss
    }

    return pos_loss + neg_loss + semipos_loss, aux_data



def compute_batch_uniformity_loss(loss_inputs: LossInputs, batch_uniformity_loss_config: BatchUniformityLossConfig):
    # Compute a batch uniformity loss, different panoramas/satellites
    # should have different embeddings
    rolled_sat_embeddings = torch.roll(loss_inputs.sat_embeddings, 1, dims=0)
    rolled_pano_embeddings = torch.roll(loss_inputs.pano_embeddings, 1, dims=0)

    def mean_hinge_loss(similarities):
        shifted_loss = torch.abs(similarities) - loss_config.batch_uniformity_hinge_location
        relud_loss = torch.mean(torch.nn.functional.relu(shifted_loss))
        return batch_uniformity_loss_config.batch_uniformity_weight * relud_loss

    sat_similarity = torch.einsum("ad,ad->a", loss_inputs.sat_embeddings, rolled_sat_embeddings)
    pano_similarity = torch.einsum("ad,ad->a", loss_inputs.pano_embeddings, rolled_pano_embeddings)

    sat_uniformity_loss = mean_hinge_loss(sat_similarity)
    pano_uniformity_loss = mean_hinge_loss(pano_similarity)
    raise NotImplementedError("Need to update this to work after rebasing. Need to add below, factor out config, update similarity computation")
    return {
        'loss': pos_loss + neg_loss + semipos_loss + sat_uniformity_loss + pano_uniformity_loss,
        'sat_uniformity_loss': sat_uniformity_loss,
        'pano_uniformity_loss': pano_uniformity_loss,
    }

    

def compute_info_nce_loss(
    loss_inputs: LossInputs,
    info_nce_config: InfoNCELoss
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

    neg_scale = info_nce_config.negative_scale / (torch.count_nonzero(torch.isfinite(negative_term), dim=1) if info_nce_config.scale_negative_by_num_items else torch.tensor([1]))
    log_neg_scale = torch.log(neg_scale).to(device).unsqueeze(1)
    loss = info_nce_config.negative_scale * torch.logsumexp(torch.cat([positive_term.unsqueeze(
        1), log_neg_scale + negative_term], dim=1), dim=-1) - positive_term

    with torch.no_grad():
        aux = dict(
            num_batch_items=loss.shape[0],
            avg_pos_semipos_similarity=positive_term.mean(),
            avg_negative_similarity=negative_term[torch.isfinite(negative_term)].mean(),
            avg_max_negative_similarity=negative_term.max(dim=1).values.mean(),
            avg_min_negative_similarity=negative_term.min(dim=1).values.mean(),
        )

    return loss.mean(), aux

def compute_spherical_embedding_constraint_loss(
    loss_inputs: LossInputs,
    sec_config: SphericalEmbeddingConstraintLoss
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """https://arxiv.org/pdf/2011.02785"""
    sat_norms = torch.linalg.norm(loss_inputs.sat_embeddings, dim=2).flatten()
    pano_norms = torch.linalg.norm(loss_inputs.pano_embeddings, dim=2).flatten()
    all_norms = torch.cat([sat_norms, pano_norms])
    mu = all_norms.mean()
    L_sec = torch.power(all_norms - mu(), 2).mean()
    final_loss = sec_config.weight_scale * L_sec
    return final_loss, {"sec_aux_loss": final_loss, "num_embeddings": all_norms.numel()}


def compute_loss(sat_embeddings: torch.Tensor,  # N_sat x n_emb_sat x D_emb
                 pano_embeddings: torch.Tensor,  # N_pano x n_emb_pano x D_emb
                 distance_type: DistanceTypes,
                 pairing_data: PairingDataType,
                 loss_configs: list[LossConfig],
                 weight_matrix: torch.Tensor | None = None,  # N_pano x N_sat x N_emb_pano x N_emb_sat x D_emb x D_emb
                 ) -> dict[str, torch.Tensor]:
    similarity = distance_from_type(
        sat_embeddings=sat_embeddings,
        pano_embeddings=pano_embeddings,
        weight_matrix=weight_matrix,
        distance_type=distance_type,
    )  # N_pano x N_sat
    loss = 0.0

    loss_input = LossInputs(
        similarity_matrix=similarity,
        sat_embeddings=sat_embeddings,
        pano_embeddings=pano_embeddings,
        pairing_data=pairing_data
    )
    aux_info = {}
    for loss_config in loss_configs:
        if isinstance(loss_config, PairwiseContrastiveLoss):
            l, new_aux = compute_pairwise_loss(loss_input, loss_config)
        elif isinstance(loss_config, InfoNCELoss):
            l, new_aux = compute_info_nce_loss(loss_input, loss_config)
        elif isinstance(loss_config, SphericalEmbeddingConstraintLoss):
            l, new_aux = compute_spherical_embedding_constraint_loss(loss_input, loss_config)
        else:
            raise ValueError(f"Unknown loss config type: {type(loss_config)}")

        loss += l
        aux_info |= new_aux
    assert "loss" not in aux_info
    return {"loss": loss} | aux_info
