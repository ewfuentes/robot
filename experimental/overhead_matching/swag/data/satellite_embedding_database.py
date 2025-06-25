import common.torch.load_torch_deps
import torch
import tqdm

import experimental.overhead_matching.swag.data.vigor_dataset as vig_dataset


def build_embeddings_from_model(model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                model_input_from_dataloader: callable,
                                device: torch.device = "cuda:0",
                                verbose: bool = False) -> torch.Tensor:
    """Embeddings will match the order of the dataloader"""

    model.to(device)
    model.eval()
    inf_results = []
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, disable=not verbose):
            embeddings = model(model_input_from_dataloader(data).to(device))
            inf_results.append(embeddings)
    embeddings = torch.concatenate(inf_results, dim=0)
    return embeddings


def build_satellite_db(model, dataloader, **kwargs):
    return build_embeddings_from_model(model, dataloader, lambda x: x.satellite, **kwargs)


def build_panorama_db(model, dataloader, **kwargs):
    return build_embeddings_from_model(model, dataloader, lambda x: x.panorama, **kwargs)


def calculate_cos_similarity_against_database(normalized_pano_embedding, normalized_sat_embedding_db):
    """
    Compute cosine similarity between query embeddings and an embedding database.
    
    Args:
        embedding: (B x D_emb) vector (probably ego vector)
        embedding_database: N x D_emb matrix of database embeddings
        
    Returns:
        Tensor of shape (B, N) with similarities for each query-database pair
    """
    assert normalized_pano_embedding.ndim == 2 and normalized_sat_embedding_db.ndim == 2 and normalized_pano_embedding.shape[1] == normalized_sat_embedding_db.shape[1], f"Got {normalized_pano_embedding.shape=}, {normalized_sat_embedding_db.shape=}"
    # Check one of the embeddings is normalized, skipping all for speed
    assert torch.allclose(normalized_pano_embedding[0].norm(), torch.tensor(1.0), atol=1e-5), f"Got {normalized_pano_embedding[0].norm()=}, should be 1.0"
    assert torch.allclose(normalized_sat_embedding_db[0].norm(), torch.tensor(1.0), atol=1e-5), f"Got {normalized_sat_embedding_db[0].norm()=}, should be 1.0"
    
    # b: batch dimension of embeddings, n: database entries, d: embedding dimension
    # For single embedding case, b=1 and the result will be (1,N)
    similarity = torch.einsum('bd,nd->bn', normalized_pano_embedding, normalized_sat_embedding_db)
    similarity = torch.clamp(similarity, -1.0, 1.0)  # some floating points are just over/under
        
    return similarity

