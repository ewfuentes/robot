import common.torch.load_torch_deps
from common.torch.load_and_save_models import save_model
import torch
import torch.nn.functional as F
from common.python.serialization import dataclass_to_dict, flatten_dict
from experimental.overhead_matching.swag.scripts.distances import (
    DistanceTypes, distance_from_type
)
from typing import Union
from dataclasses import dataclass
from pprint import pprint

@dataclass
class Pairs:
    positive_pairs: list[tuple[int, int]]
    negative_pairs: list[tuple[int, int]]
    semipositive_pairs: list[tuple[int, int]]

@dataclass
class LossInputs:
    similarity_matrix: torch.Tensor 
    pairs: Pairs

@dataclass
class PairwiseContrastiveLoss:
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


LossConfig = Union[PairwiseContrastiveLoss]


def compute_pairwise_loss(
    loss_inputs: LossInputs,
    pairwise_loss_config: PairwiseContrastiveLoss
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    similarity, pairs = loss_inputs.similarity_matrix, loss_inputs.pairs
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

    

def compute_loss(sat_embeddings: torch.Tensor, # B x n_emb x D_emb
                 pano_embeddings: torch.Tensor,  # B x n_emb x D_emb
                 distance_type: DistanceTypes,
                 pairs: Pairs, 
                 loss_configs: list[LossConfig],
                 weight_matrix: torch.Tensor | None = None, # B x D_emb x D_emb
                 ) -> dict[str, torch.Tensor]:
    similarity = distance_from_type(
        sat_embeddings=sat_embeddings,
        pano_embeddings=pano_embeddings,
        weight_matrix=weight_matrix,
        distance_type=distance_type,
    )
    loss = 0.0
    assert similarity.shape[2] == 1
    similarity = similarity.squeeze(2)

    loss_input = LossInputs(
        similarity_matrix=similarity,
        pairs=pairs
    )
    aux_info = {}
    for loss_config in loss_configs:
        if isinstance(loss_config, PairwiseContrastiveLoss):
            l, new_aux = compute_pairwise_loss(loss_input, loss_config)
            loss += l
            aux_info |= new_aux

    assert "loss" not in aux_info
    return {"loss": loss} | aux_info