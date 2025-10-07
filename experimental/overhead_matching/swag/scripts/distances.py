import common.torch.load_torch_deps
import torch.nn.functional as F
import torch
import msgspec
from typing import Union
from common.python.serialization import MSGSPEC_STRUCT_OPTS


class CosineDistanceConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    ...


def normalize_embeddings(emb_1, emb_2) -> tuple[torch.Tensor, torch.Tensor]:
    norm_emb_1 = F.normalize(emb_1, dim=-1)
    norm_emb_2 = F.normalize(emb_2, dim=-1)
    return norm_emb_1, norm_emb_2


def compute_maxsim(similarity_matrix: torch.Tensor) -> torch.Tensor:
    """
    Similarity matrix: n_pano x n_sat x n_emb_pano x n_emb_sat
    Use sum of maxsim score from https://dl.acm.org/doi/pdf/10.1145/3397271.3401075
    Treat panorama as the query, and sat patches as the document
    """
    assert similarity_matrix.ndim == 4
    return similarity_matrix.max(dim=3).values.sum(dim=2)


class CosineDistance(torch.nn.Module):
    def __init__(self, config: CosineDistanceConfig):
        super().__init__()

    def forward(self,
                sat_embeddings_unnormalized: torch.Tensor,  # n_sat x n_emb_sat x D_emb
                pano_embeddings_unnormalized: torch.Tensor  # n_pano x n_emb_pano x D_emb
                ) -> torch.Tensor:  # n_pano x n_sat

        # n_pano x n_sat x n_emb_pano x n_emb_sat
        pano_embeddings_norm, sat_embeddings_norm = normalize_embeddings(
            pano_embeddings_unnormalized, sat_embeddings_unnormalized)
        similarity = torch.einsum("aid,bjd->abij", pano_embeddings_norm, sat_embeddings_norm)
        return compute_maxsim(similarity)


class MahalanobisDistanceConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    # Options include "pano", "sat" or empty, in which case a single unconditional weight matrix is learned
    input_types: list[str]
    hidden_dim: int
    output_dim: int
    use_identity: bool  # if true, just use an identity matrix (Euclidean distance)


class MahalanobisDistance(torch.nn.Module):
    """
    Produce a weight matrix to be used with mahalanobis distance:
    distance = (x-x').T M(x-x')

    If no input is specified, learns a non-input-conditioned matrix

    If inputs (pano/sat) are provided, creates a matrix per pano/sat embedding. 
    If both are provided, creates a matrix per pano/sat embedding pair
    """

    def __init__(self,
                 config: MahalanobisDistanceConfig):
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
                ) -> torch.Tensor:  # n_pano x n_sat
        """
        If weight_matrix is none, assume identity matrix. Uses squared mahalanobis distance. 
        If multiple embeddings are used, uses maxsim to reduce.
        Returns: n_pano x n_sat similarity tensor
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
        return compute_maxsim(distances_squared)


class LearnedDistanceFunctionConfig(msgspec.Struct, **MSGSPEC_STRUCT_OPTS):
    architecture: str  # "mlp", "attention", or "transformer_decoder"
    embedding_dim: int
    num_pano_embed: int
    num_sat_embed: int
    hidden_dim: int  # hidden embedding for transformer/attention, hidden layer dim for MLP
    num_heads: int = 8  # for attention and transformer_decoder
    num_layers: int = 1  # for transformer_decoder
    max_batch_size: int = 64  # maximum number of pano-sat pairs to process in a single batch


class LearnedDistanceFunction(torch.nn.Module):
    """
    Learned distance function with three architecture options:
    - mlp: Simple MLP on concatenated embeddings
    - attention: Multi-head attention between pano and sat embeddings
    - transformer_decoder: Transformer decoder with cross-attention
    """

    def __init__(self,
                 config: LearnedDistanceFunctionConfig):
        super().__init__()
        self.config = config

        if config.architecture == "mlp":
            num_input_dim = self.config.embedding_dim * (self.config.num_pano_embed + self.config.num_sat_embed)
            self.model = torch.nn.Sequential(
                torch.nn.Linear(num_input_dim, self.config.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config.hidden_dim, 1)
            )

        elif config.architecture == "attention":
            self.multihead_attn = torch.nn.MultiheadAttention(
                embed_dim=self.config.embedding_dim,
                num_heads=self.config.num_heads,
                batch_first=True
            )
            self.output_proj = torch.nn.Linear(self.config.embedding_dim, 1)

        elif config.architecture == "transformer_decoder":
            # CLS token for aggregating information
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.config.embedding_dim))

            decoder_layer = torch.nn.TransformerDecoderLayer(
                d_model=self.config.embedding_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.hidden_dim,
                batch_first=True
            )
            self.transformer_decoder = torch.nn.TransformerDecoder(
                decoder_layer,
                num_layers=self.config.num_layers
            )
            self.output_proj = torch.nn.Linear(self.config.embedding_dim, 1)

        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")

    def _process_attention_batch(self, pano_batch, sat_batch):
        """Process a batch of pano-sat pairs using attention."""
        batch_size = pano_batch.shape[0]
        similarities = []

        for i in range(batch_size):
            pano_emb = pano_batch[i:i+1]  # 1 x n_emb_pano x d_emb
            sat_emb = sat_batch[i:i+1]    # 1 x n_emb_sat x d_emb

            # Use pano as query, sat as key/value
            attn_output, _ = self.multihead_attn(pano_emb, sat_emb, sat_emb)
            # Pool attention output and project to similarity score
            pooled = attn_output.mean(dim=1)  # 1 x d_emb
            sim = self.output_proj(pooled)    # 1 x 1
            similarities.append(sim)

        return torch.cat(similarities, dim=0)

    def _process_transformer_batch(self, pano_batch, sat_batch):
        """Process a batch of pano-sat pairs using transformer decoder."""
        batch_size = pano_batch.shape[0]
        similarities = []

        for i in range(batch_size):
            pano_emb = pano_batch[i:i+1]  # 1 x n_emb_pano x d_emb
            sat_emb = sat_batch[i:i+1]    # 1 x n_emb_sat x d_emb

            # Use CLS token as target, sat+pano as memory
            cls_token = self.cls_token.expand(1, -1, -1)  # 1 x 1 x d_emb
            memory = torch.cat([pano_emb, sat_emb], dim=1)  # 1 x (n_emb_pano + n_emb_sat) x d_emb

            # Pass through transformer decoder
            output = self.transformer_decoder(cls_token, memory)  # 1 x 1 x d_emb
            sim = self.output_proj(output.squeeze(1))  # 1 x 1
            similarities.append(sim)

        return torch.cat(similarities, dim=0)

    def forward(self,
                sat_embeddings_unnormalized: torch.Tensor,  # n_sat x n_emb_sat x D_emb
                pano_embeddings_unnormalized: torch.Tensor  # n_pano x n_emb_pano x D_emb
                ) -> torch.Tensor:  # n_pano x n_sat
        """
        Returns: n_pano x n_sat of similarity scores
        """
        n_pano, n_emb_pano, d_emb = pano_embeddings_unnormalized.shape
        n_sat, n_emb_sat, _ = sat_embeddings_unnormalized.shape
        model_device = next(self.parameters()).device

        if self.config.architecture == "mlp":
            # Process in batches to avoid OOM with large n_pano x n_sat
            # Keep embeddings on CPU, move batches to GPU as needed
            batch_size = self.config.max_batch_size
            all_similarities = []

            for pano_start in range(0, n_pano, batch_size):
                pano_end = min(pano_start + batch_size, n_pano)
                # Move batch to model device
                pano_batch = pano_embeddings_unnormalized[pano_start:pano_end].to(model_device)
                batch_n_pano = pano_batch.shape[0]

                # Move sat embeddings to model device for this batch
                sat_batch = sat_embeddings_unnormalized.to(model_device)

                # Expand within batch
                pano_expanded = pano_batch.unsqueeze(1).expand(batch_n_pano, n_sat, n_emb_pano, d_emb)
                sat_expanded = sat_batch.unsqueeze(0).expand(batch_n_pano, n_sat, n_emb_sat, d_emb)

                # Flatten embeddings and concatenate
                pano_flat = pano_expanded.reshape(batch_n_pano, n_sat, -1)
                sat_flat = sat_expanded.reshape(batch_n_pano, n_sat, -1)
                combined = torch.cat([pano_flat, sat_flat], dim=-1)

                # Pass through MLP
                batch_similarity = self.model(combined)  # batch_size x n_sat x 1
                all_similarities.append(batch_similarity.squeeze(-1).cpu())  # batch_size x n_sat

            # Concatenate all batch results
            similarity = torch.cat(all_similarities, dim=0).to(model_device)  # n_pano x n_sat
            return similarity

        elif self.config.architecture in ["attention", "transformer_decoder"]:
            # Generate all pano-sat pairs (indices only to avoid memory issues)
            # Keep embeddings on CPU, move batches to GPU as needed
            pair_indices = []

            for p_idx in range(n_pano):
                for s_idx in range(n_sat):
                    pair_indices.append((p_idx, s_idx))

            # Process pairs in batches
            all_similarities = []
            batch_size = self.config.max_batch_size

            for i in range(0, len(pair_indices), batch_size):
                batch_indices = pair_indices[i:i+batch_size]

                # Stack pairs for batch processing and move to device
                pano_batch = torch.stack([
                    pano_embeddings_unnormalized[p_idx] for p_idx, _ in batch_indices
                ]).to(model_device)  # batch_size x n_emb_pano x d_emb
                sat_batch = torch.stack([
                    sat_embeddings_unnormalized[s_idx] for _, s_idx in batch_indices
                ]).to(model_device)  # batch_size x n_emb_sat x d_emb

                if self.config.architecture == "attention":
                    batch_similarities = self._process_attention_batch(pano_batch, sat_batch)
                else:  # transformer_decoder
                    batch_similarities = self._process_transformer_batch(pano_batch, sat_batch)

                all_similarities.append(batch_similarities.cpu())

            # Concatenate all batch results
            similarities = torch.cat(all_similarities, dim=0)  # (n_pano * n_sat) x 1

            # Reshape to n_pano x n_sat
            similarities = similarities.view(n_pano, n_sat).to(model_device)
            return similarities

        else:
            raise ValueError(f"Unknown architecture: {self.config.architecture}")




DistanceConfig = Union[CosineDistanceConfig, MahalanobisDistanceConfig, LearnedDistanceFunctionConfig]


def create_distance_from_config(
    distance_config: DistanceConfig
) -> torch.nn.Module:
    if isinstance(distance_config, CosineDistanceConfig):
        distance_module = CosineDistance(distance_config)
    elif isinstance(distance_config, MahalanobisDistanceConfig):
        distance_module = MahalanobisDistance(distance_config)
    elif isinstance(distance_config, LearnedDistanceFunctionConfig):
        distance_module = LearnedDistanceFunction(distance_config)
    else:
        raise RuntimeError(f"Unknown distance config {distance_config}")
    return distance_module
