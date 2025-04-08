import common.torch.load_torch_deps
import torch
from pathlib import Path
import pandas as pd
import experimental.overhead_matching.swag.data.satellite_embedding_database as sed


# def evaluate_model_on_paths(
#     satellite embeddings, dataset, model, model config, generator, particle filter params?
# ) -> all objects created while rolling out model on several paths in a given map(too big?) Maybe stuff to make Figure 3-9:
#     pass


def evaluate_prediction_top_k(
    satellite_embedding_database: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: torch.device | str = "cuda:0",
) -> pd.DataFrame:
    model.eval()

    out_df = dict(
        panorama_index = [], # index of the ego image
        patch_cosine_similarity = [],  # cosine similarity with every overhead patch, where index in the list is the patch index (db row)
        k_value = [],  # the correct patch is the k'th highest similarity
    )

    with torch.no_grad():
        for batch in dataloader:
            batch_embedding = model(batch.panorama.to(device))
            patch_cosine_distance = sed.calculate_cos_similarity_against_database(batch_embedding, satellite_embedding_database)  # B x sat_db_size
            
            panorama_indices = [x['index'] for x in batch.panorama_metadata]
            correct_overhead_patch_indices = [x['satellite_idx'] for x in batch.panorama_metadata]

            rankings = torch.argsort(patch_cosine_distance, dim=1, descending=True)
            for i in range(len(batch.panorama_metadata)):
                out_df['k_value'].append(torch.argwhere(rankings[i] == correct_overhead_patch_indices[i]).item())
            out_df['panorama_index'].extend(panorama_indices)
            out_df['patch_cosine_similarity'].extend(patch_cosine_distance.tolist())

    df = pd.DataFrame.from_dict(out_df)
    df = df.set_index("panorama_index", drop=True)
    return df


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-m, --model_path", type=str, required=True, help="Model folder path")
#     parser.add_argument("-d, --dataset_path", type=str, required=True, help="Dataset path")

#     args = parser.parse_args()

#     args.model_path = Path(args.model_path).expanduser()
#     args.model_config_path = args.model_path.parent / Path(args.model_config_path)
#     args.dataset_path = Path(args.dataset_path)

#     # load model

#     # create dataset

#     #
