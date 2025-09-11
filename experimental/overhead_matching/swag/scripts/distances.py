from enum import StrEnum
import common.torch.load_torch_deps
import torch

class DistanceTypes(StrEnum):
    COSINE = "cosine"
    MAHALANOBIS = "mahalanobis"

def calculate_all_pairs_cosine_distance(
    sat_embeddings: torch.Tensor,
    pano_embeddings: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("aid,bid->abi", pano_embeddings, sat_embeddings) # n_pano x n_sat x n_emb

def calculate_all_pairs_mahalanobis_distance(
    sat_embeddings: torch.Tensor, # n_sat x n_emb x D_emb
    pano_embeddings: torch.Tensor, # n_pano x n_emb x D_emb
    weight_matrix: torch.Tensor | None # n_pano x n_sat x n_emb x d_emb x d_emb
) -> torch.Tensor: # n_pano x n_sat x n_emb
    """
    If weight_matrix is none, assume identity matrix
    
    Returns: n_pano x n_sat x n_emb tensor of Mahalanobis distances
    """
    # Calculate difference: n_pano x n_sat x n_emb x d_emb
    emb_diff = pano_embeddings.unsqueeze(1) - sat_embeddings.unsqueeze(0)
    
    if weight_matrix is not None:
        assert weight_matrix.shape[1] in (sat_embeddings.shape[0], 1)
        assert weight_matrix.shape[0] in (pano_embeddings.shape[0], 1)
        assert weight_matrix.shape[3] == weight_matrix.shape[4]
        
        # Mahalanobis distance: sqrt(diff^T @ W @ diff)
        distances_squared = torch.einsum('psed,psedf,psef->pse', 
                                       emb_diff, weight_matrix, emb_diff)
    else:
        # When weight_matrix is None (identity), this simplifies to L2 distance squared
        distances_squared = torch.einsum('psed,psed->pse', emb_diff, emb_diff)
    
    # Return the square root for actual Mahalanobis distance
    print("min/max in distances is ", distances_squared.min(), distances_squared.max())
    return torch.sqrt(distances_squared)


def distance_from_type(
    sat_embeddings: torch.Tensor, # n_sat x n_emb x D_emb
    pano_embeddings: torch.Tensor, # n_pano x n_emb x D_emb
    weight_matrix: torch.Tensor | None, # n_pano x n_sat x n_emb x d_emb x d_emb
    distance_type: DistanceTypes,
) -> torch.Tensor: 
    match distance_type:
        case DistanceTypes.COSINE:
            similarity = calculate_all_pairs_cosine_distance(
                sat_embeddings=sat_embeddings,
                pano_embeddings=pano_embeddings,
            )
        case DistanceTypes.MAHALANOBIS:
            similarity = calculate_all_pairs_mahalanobis_distance(
                sat_embeddings=sat_embeddings,
                pano_embeddings=pano_embeddings,
                weight_matrix=weight_matrix
            )
        case _:
            raise RuntimeError(f"Unknown distance type {distance_type}")
    return similarity