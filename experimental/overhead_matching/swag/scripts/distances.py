from enum import StrEnum
import common.torch.load_torch_deps
import torch.nn.functional as F
import torch


class DistanceTypes(StrEnum):
    COSINE = "cosine"
    MAHALANOBIS = "mahalanobis"


def calculate_all_pairs_cosine_distance(
    sat_embeddings: torch.Tensor,  # n_sat x n_emb_sat x D_emb
    pano_embeddings: torch.Tensor  # n_pano x n_emb_pano x D_emb
) -> torch.Tensor:  # n_pano x n_sat x n_emb_pano x n_emb_sat
    return torch.einsum("aid,bjd->abij", pano_embeddings, sat_embeddings)  # n_pano x n_sat x n_emb_pano x n_emb_sat


def calculate_all_pairs_mahalanobis_distance(
    sat_embeddings: torch.Tensor,  # n_sat x n_emb_sat x D_emb
    pano_embeddings: torch.Tensor,  # n_pano x n_emb_pano x D_emb
    weight_matrix: torch.Tensor | None  # n_pano x n_sat x n_emb_pano x n_emb_sat x d_emb x d_emb
) -> torch.Tensor:  # n_pano x n_sat x n_emb_pano x n_emb_sat
    """
    If weight_matrix is none, assume identity matrix

    Returns: n_pano x n_sat x n_emb x n_emb tensor of Mahalanobis distances
    """
    # Calculate difference: n_pano x n_sat x n_emb_pano x n_emb_sat x d_emb
    emb_diff = pano_embeddings.unsqueeze(1).unsqueeze(3) - sat_embeddings.unsqueeze(0).unsqueeze(2)

    if weight_matrix is not None:
        assert weight_matrix.ndim == 6
        assert weight_matrix.shape[0] in (pano_embeddings.shape[0], 1)
        assert weight_matrix.shape[1] in (sat_embeddings.shape[0], 1)
        assert weight_matrix.shape[2] == pano_embeddings.shape[1]
        assert weight_matrix.shape[3] == sat_embeddings.shape[1]
        assert weight_matrix.shape[4] == sat_embeddings.shape[2]
        assert weight_matrix.shape[5] == sat_embeddings.shape[2]

        # Mahalanobis distance: sqrt(diff^T @ W @ diff)
        distances_squared = torch.einsum('psemd,psemdf,psemf->psem',
                                         emb_diff, weight_matrix, emb_diff)
    else:
        # When weight_matrix is None (identity), this simplifies to L2 distance squared
        distances_squared = torch.einsum('psemd,psemd->psem', emb_diff, emb_diff)

    # Return the square root for actual Mahalanobis distance
    return torch.sqrt(distances_squared)
    return distances_squared


def distance_from_type(
    sat_embeddings: torch.Tensor,  # n_sat x n_emb_sat x D_emb
    pano_embeddings: torch.Tensor,  # n_pano x n_emb_pano x D_emb
    weight_matrix: torch.Tensor | None,  # n_pano x n_sat x n_emb_pano x n_emb_sat x d_emb x d_emb
    distance_type: DistanceTypes,
) -> torch.Tensor:  # n_pano x n_sat
    sat_embeddings_norm = F.normalize(sat_embeddings, dim=-1)
    pano_embeddings_norm = F.normalize(pano_embeddings, dim=-1)
    match distance_type:
        case DistanceTypes.COSINE:
            similarity = calculate_all_pairs_cosine_distance(
                sat_embeddings=sat_embeddings_norm,
                pano_embeddings=pano_embeddings_norm,
            )
        case DistanceTypes.MAHALANOBIS:
            similarity = calculate_all_pairs_mahalanobis_distance(
                sat_embeddings=sat_embeddings_norm,
                pano_embeddings=pano_embeddings_norm,
                weight_matrix=weight_matrix
            )
        case _:
            raise RuntimeError(f"Unknown distance type {distance_type}")
        
    max_num_embeddings = max(sat_embeddings_norm.shape[1], pano_embeddings_norm.shape[1])
    if max_num_embeddings == 1:
        return similarity.squeeze(-1).squeeze(-1)
    else:
        # use sum of maxsim score from https://dl.acm.org/doi/pdf/10.1145/3397271.3401075
        # treat panorama as the query, and sat patches as the document
        return similarity.max(dim=3).values.sum(dim=2)
