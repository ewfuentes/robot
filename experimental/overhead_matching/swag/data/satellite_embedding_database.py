import common.torch.load_torch_deps
import torch

import experimental.overhead_matching.swag.data.vigor_dataset as vig_dataset

def build_satellite_embedding_database(model: torch.nn.Module,
                                       dataloader: torch.utils.data.DataLoader,
                                       device: torch.device = "cuda:0",
                                       )->torch.Tensor:
    model.to(device)
    model.eval()
    inf_results = []
    all_indexes = []
    with torch.no_grad():
        for patch_dataset_item in dataloader:
            assert patch_dataset_item.panorama is None, "Are you sure you want to be buliding the embedding database over a dataset that isn't just overhead patches?"
            embeddings = model(patch_dataset_item.satellite.to(device))
            all_indexes.extend([x['index'] for x in patch_dataset_item.satellite_metadata])
            inf_results.append(embeddings)
    unsorted_embeddings = torch.concatenate(inf_results, dim=0)
    sorted_embeddings = torch.zeros_like(unsorted_embeddings)
    sorted_embeddings[torch.tensor(all_indexes, dtype=torch.long)] = unsorted_embeddings
    return sorted_embeddings
