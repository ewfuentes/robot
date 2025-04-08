import common.torch.load_torch_deps
import torch

import experimental.overhead_matching.swag.data.vigor_dataset as vig_dataset


def build_satellite_embedding_database(model: torch.nn.Module,
                                       dataloader: torch.utils.data.DataLoader,
                                       device: torch.device = "cuda:0",
                                       *,
                                       disable_dataloader_check: bool = False
                                       ) -> torch.Tensor:
    model.to(device)
    model.eval()
    inf_results = []
    all_indexes = []
    with torch.no_grad():
        for patch_dataset_item in dataloader:
            if not disable_dataloader_check:
                assert patch_dataset_item.panorama is None, "Are you sure you want to be buliding the embedding database over a dataset that isn't just overhead patches?"
            embeddings = model(patch_dataset_item.satellite.to(device))
            all_indexes.extend([x['index'] for x in patch_dataset_item.satellite_metadata])
            inf_results.append(embeddings)
    unsorted_embeddings = torch.concatenate(inf_results, dim=0)
    sorted_embeddings = torch.ones_like(unsorted_embeddings) * torch.nan
    sorted_embeddings[torch.tensor(all_indexes, dtype=torch.long)] = unsorted_embeddings
    return sorted_embeddings


def calculate_cos_similarity_against_database(embedding, embedding_database):
    """
    Compute cosine similarity between query embeddings and an embedding database.
    
    Args:
        embedding: (B x D_emb) vector (probably ego vector)
        embedding_database: N x D_emb matrix of database embeddings
        
    Returns:
        Tensor of shape (B, N) with similarities for each query-database pair
    """
    assert embedding.ndim == 2 and embedding_database.ndim == 2 and embedding.shape[1] == embedding_database.shape[1]
    
    # b: batch dimension of embeddings, n: database entries, d: embedding dimension
    # For single embedding case, b=1 and the result will be (1,N)
    similarity = torch.einsum('bd,nd->bn', embedding, embedding_database)
    similarity = similarity / (torch.norm(embedding, dim=1, keepdim=True) * torch.norm(embedding_database, dim=1, keepdim=False).unsqueeze(0))
    similarity = torch.clamp(similarity, -1.0, 1.0)  # some floating points are just over/under
        
    return similarity

