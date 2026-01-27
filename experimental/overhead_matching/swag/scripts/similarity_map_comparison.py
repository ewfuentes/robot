import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import common.torch.load_torch_deps
    import torch
    import torch.nn.functional as F
    import common.torch.load_and_save_models as lsm
    from pathlib import Path
    return F, Path, lsm, mo, torch


@app.cell
def _():
    return


@app.cell
def _(Path, lsm):
    def get_latest_checkpoint(p: Path):
        checkpoints = []
        for dir in p.glob("[0-9]*"):
            checkpoints.append(dir.name.split('_')[0])
        sorted_checkpoints = sorted(checkpoints)
        return sorted_checkpoints[-1]

    def load_models(base_path: Path):
        checkpoint_idx = get_latest_checkpoint(base_path)
        sat_path = base_path / f"{checkpoint_idx}_satellite"
        pano_path = base_path / f"{checkpoint_idx}_panorama" 
        print('loading', sat_path)
        sat_model = lsm.load_model(sat_path, device='cuda')
        print('loading', pano_path)
        pano_model = lsm.load_model(pano_path, device='cuda')
        return pano_model, sat_model 
        ...

    osm_pano_model, osm_sat_model = load_models(Path('/data/overhead_matching/training_outputs/spoofed_generalization_experiments/spoofed_4_layers_baseline/'))
    return osm_pano_model, osm_sat_model


@app.cell
def _(osm_pano_model):
    osm_pano_model._extractor_by_name["panorama_semantic_landmark_extractor"].load_files()
    return


@app.cell
def _(osm_pano_model):

    osm_pano_model._extractor_by_name["panorama_semantic_landmark_extractor"].all_sentences.keys()
    return


@app.cell
def _(F, dataset, osm_pano_model, osm_sat_model, torch, tqdm, vd):
    # run models on datasets
    def concat_list_of_3_dim_tensors_with_pad(list_of_tensors: list)-> torch.Tensor:
        _max_dims = [0, 0]
        _padded_tensors = []
        for _item in list_of_tensors:
            _max_dims[0] = max(_item.shape[1], _max_dims[0])
            _max_dims[1] = max(_item.shape[2], _max_dims[1])
        if 0 in _max_dims:
            return torch.zeros((len(list_of_tensors), *_max_dims), dtype=list_of_tensors[0].dtype)
        for _item in list_of_tensors:
            _padded_tensors.append(F.pad(
                _item, 
                (0, _max_dims[1] - _item.shape[2], 0, _max_dims[0] - _item.shape[1]),
                mode="constant"
            ))
        return torch.cat(_padded_tensors, dim=0)

    DEVICE='cuda'
    _pano_iter = vd.get_dataloader(dataset.get_pano_view(), batch_size=128, shuffle=False)
    _sat_iter = vd.get_dataloader(dataset.get_sat_patch_view(), batch_size=128, shuffle=False)
    pano_inference_data = dict(
        input_metadata = [],
        pano_embedding_vector = [],
    )
    for batch in tqdm.tqdm(_pano_iter):
        _model_input = osm_pano_model.model_input_from_batch(batch)
        pano_inference_data["input_metadata"].extend(_model_input.metadata)
        with torch.no_grad():
            _model_output = osm_pano_model(_model_input.to(DEVICE))
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
        _model_input = osm_sat_model.model_input_from_batch(batch)
        sat_inference_data["input_metadata"].extend(_model_input.metadata)
        with torch.no_grad():
            _model_output = osm_sat_model(_model_input.to(DEVICE))
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
def _(pano_inference_data):
    pano_inference_data["panorama_semantic_landmark_extractor"].shape
    return


@app.cell
def _(sat_inference_data):
    sat_inference_data["point_semantic_landmark_extractor"].shape
    return


@app.cell
def _(pano_inference_data):
    pano_inference_data["input_metadata"][0]
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
def _(decode_sentence_tensor, pano_inference_data):
    decode_sentence_tensor(pano_inference_data["panorama_semantic_landmark_extractor"][0])
    return


@app.cell
def _(decode_sentence_tensor, sat_inference_data):
    decode_sentence_tensor(sat_inference_data["point_semantic_landmark_extractor"][3738])
    return


@app.cell
def _(pano_inferencata, sat_inference_data):
    (pano_inferencata["pano_embedding_vector"][0] * sat_inference_data["sat_embedding_vector"][3737]).sum()
    return


@app.cell
def _(Path):
    import json
    def load_sentence_file(f):
        out = {}
        with f.open() as file_in:
            for line in file_in:
                j = json.loads(line)
                pano_id = j["custom_id"].split(',')[0]
                sentences = json.loads(j["response"]['body']['choices'][0]['message']['content'])["landmarks"]
                out[pano_id] = [x["description"] for x in sentences]

        return out

    def load_sentences(base_path):
        out = {}
        for p in base_path.glob("**/sentences/*"):
            out.update(load_sentence_file(p))
        return out

    embeddings_dir = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1')
    sentences_from_pano_id = load_sentences(embeddings_dir)

    return (sentences_from_pano_id,)


@app.cell
def _(osm_pano_model, osm_sat_model, vd):
    _dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=True,
        should_load_landmarks=True,
        landmark_version="spoofed_v1",
        factor=1,
        satellite_patch_size = osm_sat_model.patch_dims,
        panorama_size=osm_pano_model.patch_dims,
    )
    dataset = vd.VigorDataset(config=_dataset_config,dataset_path="/data/overhead_matching/datasets/VIGOR/Chicago/")
    return (dataset,)


@app.cell
def _(dataset, es, osm_pano_model, osm_sat_model, pano_model, sat_model):
    osm_similarity = es.compute_cached_similarity_matrix(
        dataset=dataset, pano_model=osm_pano_model,
        sat_model=osm_sat_model, device="cuda",
        use_cached_similarity=True)
    similarity = es.compute_cached_similarity_matrix(
        dataset=dataset, pano_model=pano_model,
        sat_model=sat_model, device="cuda",
        use_cached_similarity=True)
    return osm_similarity, similarity


@app.cell
def _(dataset, np, osm_similarity, pd, similarity, torch):
    records = []

    osm_max_similarity = torch.max(osm_similarity, 1).values
    max_similarity = torch.max(similarity, 1).values

    for _pano_idx, _row in dataset._panorama_metadata.iterrows():
        for _sat_idx in _row.positive_satellite_idxs:
            records.append({
                'pano_idx': _pano_idx,
                'sat_idx': _sat_idx,
                'pano_id': _row.pano_id,
                'lat': _row.lat,
                'lon': _row.lon,
                'pano_path': _row.path,
                'sat_path': dataset._satellite_metadata.iloc[_sat_idx].path,
                'osm_sim': osm_similarity[_pano_idx, _sat_idx].item(),
                'osm_max_sim': osm_max_similarity[_pano_idx].item(),
                'sim': similarity[_pano_idx, _sat_idx].item(),
                'max_sim': max_similarity[_pano_idx].item(),
            })
        for _sat_idx in _row.semipositive_satellite_idxs:
            records.append({
                'pano_idx': _pano_idx,
                'sat_idx': _sat_idx,
                'pano_id': _row.pano_id,
                'lat': _row.lat,
                'lon': _row.lon,
                'pano_path': _row.path,
                'sat_path': dataset._satellite_metadata.iloc[_sat_idx].path,
                'osm_sim': osm_similarity[_pano_idx, _sat_idx].item(),
                'osm_max_sim': osm_max_similarity[_pano_idx].item(),
                'sim': similarity[_pano_idx, _sat_idx].item(),
                'max_sim': max_similarity[_pano_idx].item(),
            })

    df = pd.DataFrame.from_records(records)

    def likelihood(max_sim, sim, sigma):
        return -np.log(np.sqrt(2 * np.pi) * sigma) - 0.5 * ((max_sim - sim) / sigma) ** 2

    df["log_likelihood"] = likelihood(df["max_sim"], df["sim"], 0.25) 
    df["osm_log_likelihood"] = likelihood(df["osm_max_sim"], df["osm_sim"], 0.5) 

    return (df,)


@app.cell
def _(dataset):
    dataset._landmark_metadata.pruned_props
    return


@app.cell
def _(df, mo, plt):
    plt.figure()
    plt.hist(df["osm_sim"])
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(alt, df, mo):

    _chart = (alt.Chart(df)
        .mark_point()
        .encode(
            x='log_likelihood',
            y='osm_log_likelihood')
        )

    chart = mo.ui.altair_chart(_chart)
    return (chart,)


@app.cell
def _(chart):
    chart
    return


@app.cell
def _(
    chart,
    dataset,
    decode_sentence_tensor,
    mo,
    sat_inference_data,
    sentences_from_pano_id,
):
    row = chart.value.head().iloc[0]

    landmark_idxs = dataset._satellite_metadata.iloc[row.sat_idx].landmark_idxs
    landmarks = dataset._landmark_metadata.iloc[landmark_idxs].pruned_props

    input_metadata = sat_inference_data["input_metadata"][row.sat_idx]

    osm_sentences = []
    pruned_props = []
    for geom in ["point", 'linestring', 'polygon', 'multipolygon']:
        pruned_props += [x["pruned_props"] for x in input_metadata["landmarks"] if x["geometry"].geom_type.lower() == geom]
        osm_sentences += decode_sentence_tensor(sat_inference_data[f"{geom}_semantic_landmark_extractor"][row.sat_idx])

    osm_sentences = osm_sentences

    mo.vstack([
        row,
        mo.hstack([
            mo.image(row.pano_path, height=420),
            mo.image(row.sat_path, height=420)]
            , justify='start'),
        osm_sentences,
        pruned_props,
        landmarks,
        sentences_from_pano_id[row.pano_id]
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    return (vd,)


@app.cell
def _():
    import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
    return (es,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    import seaborn as sns
    return (plt,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell
def _():
    import altair as alt
    import tqdm
    return alt, tqdm


@app.cell
def _(sat_inference_data):
    sat_inference_data['input_metadata'][0]
    return


@app.cell
def _(dataset):
    places_and_things = dataset._landmark_metadata[["name", "building", "addr:housenumber", "addr:street"]].dropna()
    return (places_and_things,)


@app.cell
def _(places_and_things):
    places_and_things
    return


@app.cell
def _(places_and_things):
    things = places_and_things["name"]
    places = places_and_things["addr:street"]

    return places, things


@app.cell
def _(itertools, pd, places, things):

    _records = []
    for t, p in itertools.product(things[:1000], places[:1000]):
        _records.append({
            'thing': t,
            'place': p,
            'sentence': f"{t} on {p}"
        })

    train_set = pd.DataFrame.from_records(_records)

    _records = []
    for t, p in itertools.product(things[1000:], places[1000:]):
        _records.append({
            'thing': t,
            'place': p,
            'sentence': f"{t} on {p}"
        })

    test_set = pd.DataFrame.from_records(_records)
    return test_set, train_set


@app.cell
def _(test_set, train_set):
    import pickle
    with open('/tmp/dataset.pkl', 'wb') as file_out:
        pickle.dump({
            'test': test_set,
            'train': train_set
        }, file_out)
    return


@app.cell
def _():
    import itertools
    return (itertools,)


@app.cell
def _(things):
    things
    return


@app.cell
def _(things):
    len(set(things.values))
    return


@app.cell
def _(train_set):
    len(train_set)
    return


@app.cell
def _(test_set):
    len(test_set)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
