import common.torch.load_torch_deps
import torch

import experimental.overhead_matching.swag.data.vigor_dataset as vig_dataset

def build_satellite_embedding_database(model: torch.nn.Module,
                                       dataset: vig_dataset.VigorDataset, 
                                       dataloader_kwards: dict | None = None,
                                       device: torch.device = "cuda:0",
                                       )->torch.Tensor:
    if dataloader_kwards is None:
        dataloader_kwards = {} 
    overhead_iter = vig_dataset.get_overhead_dataloader(dataset, **dataloader_kwards)
    num_overhead_patches = dataset.num_satellite_patches
    sat_embedding_db = None

    model.to(device)
    model.eval()
    with torch.no_grad():
        for indexes, patch_dataset_item in overhead_iter:
            indexes = torch.tensor(indexes, device=device)
            embeddings = model(patch_dataset_item.satellite.to(device))
            if sat_embedding_db is None:
                sat_embedding_db = torch.zeros((num_overhead_patches, embeddings.shape[1]))
            sat_embedding_db[indexes] = embeddings
    return sat_embedding_db
