import common.torch.load_torch_deps
import torch.nn.functional as F
import torch
from dataclasses import dataclass
import msgspec
from typing import Union
from common.python.serialization import MSGSPEC_STRUCT_OPTS


class CosineDistanceModelConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    ...


def normalize_embeddings(emb_1, emb_2) -> tuple[torch.Tensor, torch.Tensor]:
    norm_emb_1 = F.normalize(emb_1, dim=-1)
    norm_emb_2 = F.normalize(emb_2, dim=-1)
    return norm_emb_1, norm_emb_2


class CosineDistanceModel(torch.nn.Module):
    def __init__(self, config: CosineDistanceModelConfig):
        super().__init__()

    def forward(self,
                sat_embeddings_unnormalized: torch.Tensor,  # n_sat x n_emb_sat x D_emb
                pano_embeddings_unnormalized: torch.Tensor  # n_pano x n_emb_pano x D_emb
                ) -> torch.Tensor:  # n_pano x n_sat x n_emb_pano x n_emb_sat

        # n_pano x n_sat x n_emb_pano x n_emb_sat
        pano_embeddings_norm, sat_embeddings_norm = normalize_embeddings(
            pano_embeddings_unnormalized, sat_embeddings_unnormalized)
        return torch.einsum("aid,bjd->abij", pano_embeddings_norm, sat_embeddings_norm)


class MahalanobisDistanceModelConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    # Options include "pano", "sat" or empty, in which case a single unconditional weight matrix is learned
    input_types: list[str]
    hidden_dim: int
    output_dim: int
    use_identity: bool  # if true, just use an identity matrix (Euclidean distance)


class MahalanobisDistanceModel(torch.nn.Module):
    """
    Produce a weight matrix to be used with mahalanobis distance:
    distance = (x-x').T M(x-x')

    If no input is specified, learns a non-input-conditioned matrix

    If inputs (pano/sat) are provided, creates a matrix per pano/sat embedding. 
    If both are provided, creates a matrix per pano/sat embedding pair

    Output: 
        num_pano_embeds x num_sat_embeds x num_pano_class_tokens x num_sat_class_tokens x d_emb x d_emb
        If an embedding is not part of the input (e.g., num_sat_embeds if pano is input)
        the dimension is set to 1
    """

    def __init__(self,
                 config: MahalanobisDistanceModelConfig):
        super().__init__()
        self.config = config
        if self.config.use_identity:
            self.weight_matrix = None
        elif len(self.config.input_types) == 0:
            weight_matrix = 0.01 * \
                (torch.rand(1, 1, 1, 1, config.output_dim, config.output_dim) - 0.5)
            weight_matrix[0, 0, 0, 0].fill_diagonal_(1.0)
            self.weight_matrix = torch.nn.Parameter(weight_matrix)
        else:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(config.output_dim * len(self.config.input_types),
                                self.config.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.hidden_dim, config.output_dim**2)
            )

    def make_weight_matrix(self,
                           sat_embedding: torch.Tensor,  # n_sat x n_emb_sat x D_emb
                           pano_embedding: torch.Tensor,  # n_pano x n_emb_pano x D_emb
                           ) -> torch.Tensor | None:
        # if not conditional, return weight matrix
        if len(self.config.input_types) == 0 or self.config.use_identity:
            out_matrix = self.weight_matrix
        else:
            for item in self.config.input_types:
                if item not in ["pano", "sat"]:
                    raise RuntimeError(f"Invalid item in config: {item}")

            # otherwise, run MLP to get matrix
            embedding_dim = sat_embedding.shape[-1] if sat_embedding is not None else pano_embedding.shape[-1]
            model_input = []
            if "pano" in self.config.input_types and "sat" in self.config.input_types:
                assert pano_embedding.ndim == 3  # (num_pano, num_class_tokens, D_emb)
                assert sat_embedding.ndim == 3  # (num_sat, num_class_token, D_emb)
                assert sat_embedding.shape[1] == pano_embedding.shape[1]
                target_size = (pano_embedding.shape[0],
                               sat_embedding.shape[0],
                               sat_embedding.shape[1],
                               pano_embedding.shape[-1])
                model_input = torch.cat([
                    pano_embedding.unsqueeze(1).expand(target_size),
                    sat_embedding.unsqueeze(0).expand(target_size)
                ], dim=-1)
                out_matrix = self.mlp(model_input)
            elif "sat" in self.config.input_types:
                assert sat_embedding.ndim == 3  # (num_sat, num_embeddings, D_emb)
                out_matrix = self.mlp(sat_embedding).unsqueeze(
                    0).unsqueeze(2)  # 1, num_sat, 1, num_sat_emb, D_emb**2
            elif "pano" in self.config.input_types:
                assert pano_embedding.ndim == 3  # (num_pano, num_embeddings, D_emb)
                out_matrix = self.mlp(pano_embedding).unsqueeze(1).unsqueeze(3)
            else:
                raise RuntimeError(f"Invalid input config {self.config.input_types}")

            out_matrix = out_matrix.unflatten(-1, (embedding_dim, embedding_dim))

        # Make matrix positive semi-definite by computing A^T @ A
        if out_matrix is not None:
            out_matrix = torch.matmul(out_matrix.transpose(-2, -1), out_matrix)
        return out_matrix

    def forward(self,
                sat_embeddings_unnormalized: torch.Tensor,  # n_sat x n_emb_sat x D_emb
                pano_embeddings_unnormalized: torch.Tensor  # n_pano x n_emb_pano x D_emb
                ) -> torch.Tensor:  # n_pano x n_sat x n_emb_pano x n_emb_sat
        """
        If weight_matrix is none, assume identity matrix

        Returns: n_pano x n_sat x n_emb x n_emb tensor of SQUARED Mahalanobis distances
        """
        sat_embeddings_norm, pano_embeddings_norm = normalize_embeddings(
            sat_embeddings_unnormalized, pano_embeddings_unnormalized)
        weight_matrix = self.make_weight_matrix(
            sat_embedding=sat_embeddings_norm, pano_embedding=pano_embeddings_norm)
        # Calculate difference: n_pano x n_sat x n_emb_pano x n_emb_sat x d_emb
        emb_diff = pano_embeddings_norm.unsqueeze(1).unsqueeze(
            3) - sat_embeddings_norm.unsqueeze(0).unsqueeze(2)

        if weight_matrix is not None:
            assert weight_matrix.ndim == 6
            assert weight_matrix.shape[0] in (pano_embeddings_norm.shape[0], 1)
            assert weight_matrix.shape[1] in (sat_embeddings_norm.shape[0], 1)
            assert weight_matrix.shape[2] == pano_embeddings_norm.shape[1]
            assert weight_matrix.shape[3] == sat_embeddings_norm.shape[1]
            assert weight_matrix.shape[4] == sat_embeddings_norm.shape[2]
            assert weight_matrix.shape[5] == sat_embeddings_norm.shape[2]

            # Mahalanobis distance: sqrt(diff^T @ W @ diff)
            distances_squared = torch.einsum('psemd,psemdf,psemf->psem',
                                             emb_diff, weight_matrix, emb_diff)
        else:
            # When weight_matrix is None (identity), this simplifies to L2 distance squared
            distances_squared = torch.einsum('psemd,psemd->psem', emb_diff, emb_diff)

        # Return the square of actual Mahalanobis distance (found this improved performance)
        return distances_squared


DistanceConfig = Union[CosineDistanceModelConfig, MahalanobisDistanceModelConfig]


def create_distance_from_config(
    distance_config: DistanceConfig
) -> torch.nn.Module:
    if isinstance(distance_config, CosineDistanceModelConfig):
        distance_module = CosineDistanceModel(distance_config)
    elif isinstance(distance_config, MahalanobisDistanceModelConfig):
        distance_module = MahalanobisDistanceModel(distance_config)
    else:
        raise RuntimeError(f"Unknown distance config {distance_config}")
    return distance_module


def distance_from_model(
    sat_embeddings_unnormalized: torch.Tensor,  # n_sat x n_emb_sat x D_emb
    pano_embeddings_unnormalized: torch.Tensor,  # n_pano x n_emb_pano x D_emb
    distance_model: torch.nn.Module,  # distance module
) -> torch.Tensor:  # n_pano x n_sat

    similarity = distance_model(sat_embeddings_unnormalized=sat_embeddings_unnormalized,
                                pano_embeddings_unnormalized=pano_embeddings_unnormalized)

    max_num_embeddings = max(
        sat_embeddings_unnormalized.shape[1], pano_embeddings_unnormalized.shape[1])
    if max_num_embeddings == 1:
        return similarity.squeeze(-1).squeeze(-1)
    else:
        # use sum of maxsim score from https://dl.acm.org/doi/pdf/10.1145/3397271.3401075
        # treat panorama as the query, and sat patches as the document
        return similarity.max(dim=3).values.sum(dim=2)
