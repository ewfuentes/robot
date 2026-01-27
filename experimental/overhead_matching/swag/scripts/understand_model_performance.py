import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import common.torch.load_torch_deps
    import torch
    import torch.nn.functional as F
    import pandas as pd
    import pickle
    from pathlib import Path
    import tqdm
    import matplotlib.pyplot as plt

    import common.torch.load_and_save_models as lsm
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    import experimental.overhead_matching.swag.scripts.train as train_module
    import experimental.overhead_matching.swag.scripts.distances as distances_module

    return F, Path, lsm, mo, pickle, plt, torch, tqdm, train_module, vd


@app.cell
def _(mo):
    mo.md(
        r"""
    The goal of this notebook is to look at what models mark as similar and what they don't. The hope is to get some insight on the following questions  
     - Why do things generalize poorly to Seattle? What does it fail to learn?  
     - How "unique" are landmarks for most of these tiles?
    """
    )
    return


@app.cell
def _(vd):
    # create dataset
    _dataset_config = vd.VigorDatasetConfig(
       satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version="v4_202001",
        factor=0.3,
        # factor=1,
    )
    dataset = vd.VigorDataset(config=_dataset_config,dataset_path="/data/overhead_matching/datasets/VIGOR/Seattle/")
    # dataset = vd.VigorDataset(config=_dataset_config,dataset_path="/data/overhead_matching/datasets/VIGOR/Chicago/")
    return (dataset,)


@app.cell
def _(Path, lsm):
    # create models
    DEVICE = "cuda:0"
    _model_base_path = Path("/data/overhead_matching/training_outputs/251205_09000000_model_size_experiments/landmark_base_performance/panorama_semantic_landmark_embeddings/").expanduser()
    sat_model = lsm.load_model(_model_base_path / "0099_satellite", device=DEVICE)
    pano_model = lsm.load_model(_model_base_path / "0099_panorama", device=DEVICE)
    return DEVICE, pano_model, sat_model


@app.cell
def _(pano_model):
    pano_model
    return


@app.cell
def _(F, torch):
    def concat_list_of_3_dim_tensors_with_pad(list_of_tensors: list)-> torch.Tensor:
        _max_dims = [0, 0]
        _padded_tensors = []
        for _item in list_of_tensors:
            _max_dims[0] = max(_item.shape[1], _max_dims[0])
            _max_dims[1] = max(_item.shape[2], _max_dims[1])
        for _item in list_of_tensors:
            _padded_tensors.append(F.pad(
                _item, 
                (0, _max_dims[1] - _item.shape[2], 0, _max_dims[0] - _item.shape[1]),
                mode="constant"
            ))
        return torch.cat(_padded_tensors, dim=0)

    return (concat_list_of_3_dim_tensors_with_pad,)


@app.cell
def _(
    DEVICE,
    concat_list_of_3_dim_tensors_with_pad,
    dataset,
    pano_model,
    sat_model,
    torch,
    tqdm,
    vd,
):
    # run models on datasets
    _pano_iter = vd.get_dataloader(dataset.get_pano_view(), batch_size=128, shuffle=False)
    _sat_iter = vd.get_dataloader(dataset.get_sat_patch_view(), batch_size=128, shuffle=False)
    pano_inference_data = dict(
        input_metadata = [],
        pano_embedding_vector = [],
    )
    for batch in tqdm.tqdm(_pano_iter):
        _model_input = pano_model.model_input_from_batch(batch)
        pano_inference_data["input_metadata"].extend(_model_input.metadata)
        with torch.no_grad():
            _model_output = pano_model(_model_input.to(DEVICE))
        pano_inference_data["pano_embedding_vector"].append(_model_output[0].to("cpu"))
        for _k in _model_output[1].keys():
            if _k not in pano_inference_data:
                pano_inference_data[_k] = []
            pano_inference_data[_k].append(_model_output[1][_k].debug["sentences"].to("cpu"))
    for _k in pano_inference_data.keys():
        if _k not in ["input_metadata"]:
           pano_inference_data[_k] = concat_list_of_3_dim_tensors_with_pad(pano_inference_data[_k])

    sat_inference_data = dict(
        input_metadata = [],
        sat_embedding_vector = [],
    )

    for batch in tqdm.tqdm(_sat_iter):
        _model_input = sat_model.model_input_from_batch(batch)
        sat_inference_data["input_metadata"].extend(_model_input.metadata)
        with torch.no_grad():
            _model_output = sat_model(_model_input.to(DEVICE))
        sat_inference_data["sat_embedding_vector"].append(_model_output[0].to("cpu"))
        for _k in _model_output[1].keys():
            if _k not in sat_inference_data:
                sat_inference_data[_k] = []
            sat_inference_data[_k].append(_model_output[1][_k].debug["sentences"].to("cpu"))
    for _k in sat_inference_data.keys():
        if _k not in ["input_metadata"]:
           sat_inference_data[_k] = concat_list_of_3_dim_tensors_with_pad(sat_inference_data[_k])
    return pano_inference_data, sat_inference_data


@app.cell
def _(
    dataset,
    pano_inference_data,
    plt,
    sat_inference_data,
    torch,
    train_module,
):
    # num panos x num_sat_patches
    similarity_tensor = torch.einsum("ij,kj->ik", pano_inference_data["pano_embedding_vector"].squeeze(1), sat_inference_data["sat_embedding_vector"].squeeze(1))
    print(similarity_tensor.shape, similarity_tensor.min(), similarity_tensor.max())
    plt.figure()
    plt.hist(similarity_tensor.flatten(), bins=100)
    plt.show()

    val_metrics = train_module.validation_metrics_from_similarity("chicago", similarity_tensor, dataset._panorama_metadata)
    return similarity_tensor, val_metrics


@app.cell
def _(val_metrics):
    val_metrics
    return


@app.cell
def _(dataset):
    dataset._panorama_metadata
    return


@app.cell
def _(dataset):
    dataset._satellite_metadata
    return


@app.cell
def _(sat_inference_data):
    sat_inference_data.keys()
    return


@app.cell
def _(torch):
    def decode_sentence_tensor(sentence_tensor: torch.Tensor)-> list[str]:
        _sentences = []
        for _i in range(sentence_tensor.shape[0]):
            _sentence = bytes(sentence_tensor[_i].tolist()).decode("utf-8").rstrip("\x00")
            if _sentence:
                _sentences.append(_sentence)
        return _sentences
    return (decode_sentence_tensor,)


@app.cell
def _(
    dataset,
    decode_sentence_tensor,
    mo,
    pano_inference_data,
    sat_inference_data,
):
    pano_idx = 111
    pano_data_row = dataset._panorama_metadata.iloc[pano_idx]
    print(pano_data_row)
    _pano_metadata_from_batch = pano_inference_data["input_metadata"][pano_idx]
    assert _pano_metadata_from_batch["pano_id"] == pano_data_row["pano_id"]
    _pano_embedding_vector = pano_inference_data["pano_embedding_vector"][pano_idx]
    pano_landmark_sentences = pano_inference_data["panorama_semantic_landmark_extractor"][pano_idx]
    print(decode_sentence_tensor(pano_landmark_sentences))

    matching_sat_indexes = pano_data_row.positive_satellite_idxs + pano_data_row.semipositive_satellite_idxs
    for _sat_idx in matching_sat_indexes:
        print(f"\n\n__Sattelite index {matching_sat_indexes}__")
        _sat_data_row = dataset._satellite_metadata.iloc[_sat_idx]
        _sat_metadata = sat_inference_data["input_metadata"][_sat_idx]
        _sat_embedding_vector = sat_inference_data["sat_embedding_vector"][_sat_idx]
        print(_sat_data_row)
        for _lm in _sat_metadata["landmarks"]:
            print(_lm["id"], ": ", _lm["pruned_props"])
        print("True patch similarity: ", (_sat_embedding_vector * _pano_embedding_vector).sum())
        for _k in sat_inference_data.keys():
            if "extractor" in _k:
                print(_k, decode_sentence_tensor(sat_inference_data[_k][_sat_idx]))

    mo.image(pano_data_row.path)
    return (
        matching_sat_indexes,
        pano_data_row,
        pano_idx,
        pano_landmark_sentences,
    )


@app.cell
def _(
    dataset,
    decode_sentence_tensor,
    pano_inference_data,
    sat_inference_data,
    similarity_tensor,
    tqdm,
):
    # export

    output_dict = {
        "similarity_matrix": similarity_tensor.flatten().tolist(),  # list[float] flattened, npano * nsat
        "panorama_id": [], # list[str]
        "panorama_loc":  [], # list[tuple[float,float]] =
        "panorama_sentences":  [], # list[list[str]] =
        "panorama_embeddings": pano_inference_data["pano_embedding_vector"].squeeze(1).cpu().tolist(),  # list[list[float]], shape (npano, emb_dim)
        "sat_loc": [], # list[tuple[float,float]] =
        "sat_sentences":  [], # list[list[str]] =
        "sat_embeddings": sat_inference_data["sat_embedding_vector"].squeeze(1).cpu().tolist(),  # list[list[float]], shape (nsat, emb_dim)
        "pano_to_positive_sat_index_map":  [], # list[[list[int]] =
    }

    for _pano_idx in tqdm.tqdm(range(len(dataset._panorama_metadata))):
        _pano_data_row = dataset._panorama_metadata.iloc[_pano_idx]
        _pano_metadata_from_batch = pano_inference_data["input_metadata"][_pano_idx]
        assert _pano_metadata_from_batch["pano_id"] == _pano_data_row["pano_id"]
        _pano_landmark_sentences = pano_inference_data["panorama_semantic_landmark_extractor"][_pano_idx]
        output_dict["panorama_id"].append(_pano_data_row["pano_id"])
        output_dict["panorama_sentences"].append(decode_sentence_tensor(_pano_landmark_sentences))
        output_dict["panorama_loc"].append((_pano_data_row["lat"], _pano_data_row["lon"]))
        output_dict["pano_to_positive_sat_index_map"].append(_pano_data_row.positive_satellite_idxs + _pano_data_row.semipositive_satellite_idxs)

    for _sat_idx in tqdm.tqdm(range(len(dataset._satellite_metadata))):
        _sat_data_row = dataset._satellite_metadata.iloc[_sat_idx]
        _sat_metadata_from_batch = sat_inference_data["input_metadata"][_sat_idx]
        _sentences = []
        for _k in sat_inference_data.keys():
            if "extractor" in _k:
                _sentences.extend(decode_sentence_tensor(sat_inference_data[_k][_sat_idx]))

        output_dict["sat_sentences"].append(_sentences)
        output_dict["sat_loc"].append((_sat_metadata_from_batch["lat"], _sat_metadata_from_batch["lon"]))

    return (output_dict,)


@app.cell
def _(output_dict, pickle):
    with open("/tmp/visualizer_information_seattle.pkl", 'wb') as f:
        pickle.dump(output_dict, f)
    return


@app.cell
def _(matching_sat_indexes, pano_data_row, pano_idx, plt, similarity_tensor):
    _fig, _ax = plt.subplots()
    _ax.hist(similarity_tensor[pano_idx, :], bins=100)
    for _sat_idx in matching_sat_indexes:
        _true_sim = similarity_tensor[pano_idx, _sat_idx].item()
        _greater_than = (_true_sim > similarity_tensor[pano_idx, :]).float().mean().item()
        print("Simiarlity of true match: ", _true_sim, " greater than ", _greater_than*100, "%")
        _ax.axvline(similarity_tensor[pano_idx, _sat_idx], c="red", ls="--", label="True patch")
    _ax.set_xlabel("Sat patch similarity")
    _ax.set_ylabel("Count")
    _ax.set_title(f"Sat patch similarity against pano idx {pano_idx}, id {pano_data_row['pano_id']}")
    _ax.legend()
    plt.gca()
    return


@app.cell
def _(
    decode_sentence_tensor,
    matching_sat_indexes,
    pano_data_row,
    pano_idx,
    pano_landmark_sentences,
    sat_inference_data,
    similarity_tensor,
    torch,
):
    print(f"Panorama landmarks for {pano_idx} w/ id {pano_data_row['pano_id']}:")
    for _s in decode_sentence_tensor(pano_landmark_sentences):
        print(f"\t- {_s}")
    for _sat_idx in matching_sat_indexes:
        print(f"True match {_sat_idx} ")
        for _k in sat_inference_data.keys():
            if "extractor" in _k:
                _sentences = decode_sentence_tensor(sat_inference_data[_k][_sat_idx])
                for _s in set(_sentences):
                    print(f"\t- {_s}")

    print("\n========= Patches with HIGHEST similarity values")
    _top_k = similarity_tensor[pano_idx, :].topk(3)
    for _sim, _sat_idx in zip(_top_k.values, _top_k.indices):
        print(f"=== Sattelite patch {_sat_idx} with sim {_sim}:")
        for _k in sat_inference_data.keys():
            if "extractor" in _k:
                _sentences = decode_sentence_tensor(sat_inference_data[_k][_sat_idx])
                for _s in set(_sentences):
                    print(f"\t- {_s}")

    print("\n========= Patches sampled with range of similarity values")
    _sim_targets = torch.linspace(similarity_tensor[pano_idx, matching_sat_indexes[-1]], similarity_tensor[pano_idx].max().item(), 3)
    _closest_indexes = (similarity_tensor[pano_idx].unsqueeze(0) - _sim_targets.unsqueeze(1)).abs().argmin(dim=1)
    for _sat_idx in _closest_indexes:
        print(f"=== Sattelite patch {_sat_idx} with sim {similarity_tensor[pano_idx, _sat_idx]}:")
        for _k in sat_inference_data.keys():
            if "extractor" in _k:
                _sentences = decode_sentence_tensor(sat_inference_data[_k][_sat_idx])
                for _s in set(_sentences):
                    print(f"\t- {_s}")

    print("\n========= Patches sampled with range of LOW similarity values")
    _sim_targets = torch.linspace(similarity_tensor[pano_idx].min().item(), similarity_tensor[pano_idx, matching_sat_indexes[-1]], 3)
    _closest_indexes = (similarity_tensor[pano_idx].unsqueeze(0) - _sim_targets.unsqueeze(1)).abs().argmin(dim=1)
    for _sat_idx in _closest_indexes:
        print(f"=== Sattelite patch {_sat_idx} with sim {similarity_tensor[pano_idx, _sat_idx]}:")
        for _k in sat_inference_data.keys():
            if "extractor" in _k:
                _sentences = decode_sentence_tensor(sat_inference_data[_k][_sat_idx])
                for _s in set(_sentences):
                    print(f"\t- {_s}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
