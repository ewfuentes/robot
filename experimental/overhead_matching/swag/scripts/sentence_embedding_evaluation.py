import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import common.torch.load_torch_deps
    import torch
    from pathlib import Path
    import itertools
    import json
    import pickle
    import tqdm
    from experimental.overhead_matching.swag.model.sentence_embedding_model import create_model_from_config
    from experimental.overhead_matching.swag.scripts.sentence_configs import load_config
    return (
        Path,
        create_model_from_config,
        itertools,
        json,
        load_config,
        pickle,
        torch,
        tqdm,
    )


@app.cell
def _():
    EMBEDDING_DIM=384
    return (EMBEDDING_DIM,)


@app.cell
def _(Path, json, load_config, torch):
    model_base = Path('/data/overhead_matching/models/sentence_embeddings/sentence_embeddings_addrname_debug_w_fix/')
    checkpoint = torch.load(model_base / 'best_model.pt')
    vocab = json.loads((model_base / 'tag_vocabs.json').read_text())
    train_config = load_config(model_base / 'config.json')
    return checkpoint, model_base, train_config, vocab


@app.cell
def _(checkpoint, create_model_from_config, train_config, vocab):

    model = create_model_from_config(
        train_config.model,
        tag_vocabs=vocab,
        classification_task_names=train_config.classification_tags,
        contrastive_task_names=train_config.contrastive_tags).cuda()
    model.load_state_dict(checkpoint)
    return (model,)


@app.cell
def _():
    return


@app.cell
def _(Path):
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    sentence_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_gemini/sentences/')
    sentence_dict, _ = slu.make_sentence_dict_from_json(slu.load_all_jsonl_from_folder(sentence_path))
    return sentence_dict, slu


@app.cell
def _(sentence_dict):
    for _x in range(10):
        _k = list(sentence_dict.keys())[_x]
        print(_k, sentence_dict[_k])
    return


@app.cell
def _():
    return


@app.cell
def _(EMBEDDING_DIM, model, pickle, sentence_dict, torch, tqdm):
    ## Export OSM Embeddings

    with open("/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_gemini/embeddings/embeddings.pkl", 'rb') as f:
        old_embeddings = pickle.load(f)
    new_embeddings_tensor = torch.ones((old_embeddings[0].shape[0], EMBEDDING_DIM)) * torch.nan
    for _custom_id, _idx in tqdm.tqdm(old_embeddings[1].items()):
        with torch.no_grad():
            _model_output = model([sentence_dict[_custom_id]])["base_embedding"].cpu()
        new_embeddings_tensor[_idx] = _model_output
    assert not torch.any(torch.isnan(new_embeddings_tensor))
    with open("/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_embeddings_from_ericks_model/embeddings/embeddings.pkl",'wb') as _f:
        pickle.dump((new_embeddings_tensor, old_embeddings[1]), _f)
    return


@app.cell
def _():
    return


@app.cell
def _(EMBEDDING_DIM, Path, itertools, model, model_base, pickle, torch):
    _old_pano_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_gemini/")
    _new_pano_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_gemini_embeddings_from_ericks_model/")

    for _city in _old_pano_base_dir.glob("*/"):
        # V2 pickle embedding
        # out = pickle.load((_city / "embeddings/embeddings.pkl").open("rb"))
        # break
        _v2_pickle_embedding = pickle.load((_city / "embeddings/embeddings.pkl").open("rb"))
        _output_v2_embedding = dict(
            version=_v2_pickle_embedding["version"],
            embedding_model=str(model_base),
            embedding_dim=EMBEDDING_DIM,
            panoramas=_v2_pickle_embedding["panoramas"],
            task_type=None,
            description_id_to_idx=_v2_pickle_embedding["description_id_to_idx"],
            location_type_to_idx=_v2_pickle_embedding["location_type_to_idx"],
            proper_noun_to_idx=_v2_pickle_embedding["proper_noun_to_idx"],
        )
        # convert location type and proper noun embeddings
        for _prefix in ["location_type", "proper_noun"]:
            _output_tensor = torch.ones((_v2_pickle_embedding[f"{_prefix}_embeddings"].shape[0], EMBEDDING_DIM)) * torch.nan
            for _key_index_pair_batch in itertools.batched(_v2_pickle_embedding[f"{_prefix}_to_idx"].items(), 50):
                _key_set = [_x[0] for _x in _key_index_pair_batch]
                _index_set = [_x[1] for _x in _key_index_pair_batch]
                with torch.no_grad():
                    _new_embeddings = model(_key_set)["base_embedding"].cpu()
                for _i, _new_index in enumerate(_index_set):
                    _output_tensor[_new_index] = _new_embeddings[_i]

            assert not torch.any(torch.isnan(_output_tensor))
            _output_v2_embedding[f"{_prefix}_embeddings"] = _output_tensor
        # convert description embeddings        
        _output_tensor = torch.ones((_v2_pickle_embedding[f"description_embeddings"].shape[0], EMBEDDING_DIM)) * torch.nan
        for _key_index_pair_batch in itertools.batched(_v2_pickle_embedding[f"description_id_to_idx"].items(), 50):
            _key_set = [_x[0] for _x in _key_index_pair_batch]
            _index_set = [_x[1] for _x in _key_index_pair_batch]
            def _get_pid_idx_from_key(key: str):
                pid = "__".join(key.split("__")[:-1])
                landmark_idx_part = key.split("__")[-1]
                landmark_id = int(landmark_idx_part.split("_")[1])
                return pid, landmark_id
            _sentence_fetching_info = [_get_pid_idx_from_key(_x) for _x in _key_set]
            _sentences = []
            for _panoid, _landmark_idx in _sentence_fetching_info:
                _pano_landmarks = _v2_pickle_embedding["panoramas"][_panoid]["landmarks"]
                _landmark = _pano_landmarks[_landmark_idx]
                assert _landmark["landmark_idx"] == _landmark_idx
                _sentences.append(_landmark["description"])

            with torch.no_grad():
                _new_embeddings = model(_sentences)["base_embedding"].cpu()
            for _i, _new_index in enumerate(_index_set):
                _output_tensor[_new_index] = _new_embeddings[_i]

        assert not torch.any(torch.isnan(_output_tensor))
        _output_v2_embedding["description_embeddings"] = _output_tensor

        pickle.dump(_output_v2_embedding, (_new_pano_base_dir / _city.name / "embeddings/embeddings.pkl").open("wb"))


        # V1 pickle embedding
        # _city_embeddings_tensor, _city_embeddings_mapping = pickle.load((_city / "embeddings/embeddings.pkl").open("rb"))
        # _new_embedding_tensor = torch.ones((_city_embeddings_tensor.shape[0], 384)) * torch.nan
        # _city_sentences, _ = slu.make_sentence_dict_from_json(slu.load_all_jsonl_from_folder(_city / "sentences/"))
        # for _panoidverbose, _props_str in tqdm.tqdm(_city_sentences.items()):
        #     _landmarks = json.loads(_props_str)["landmarks"]
        #     _sentences = [_x["description"] for _x in _landmarks]
        #     with torch.no_grad():
        #         _new_embeddings = model(_sentences)["base_embedding"].cpu()
        #     # print(_sentences)
        #     for _i, _emb in enumerate(_new_embeddings):
        #         _key = f"{_panoidverbose}__landmark_{_i}"
        #         if _key not in _city_embeddings_mapping:
        #             print(f"Failed to find key {_key}")
        #             continue
        #         _loc = _city_embeddings_mapping[_key]
        #         _new_embedding_tensor[_loc, :] = _emb
        #     # print(_props_str)

        # assert not torch.any(torch.isnan(_new_embedding_tensor))
        # pickle.dump((_new_embedding_tensor, _city_embeddings_mapping), (_new_pano_base_dir / _city.name / "embeddings/embeddings.pkl").open("wb"))
        # print(list(_city_sentences.keys())[:10])
    return


@app.cell
def _(EMBEDDING_DIM, Path, itertools, json, model, model_base, pickle, torch):
    # Process spoofed_place_location embeddings
    _spoofed_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location")
    _output_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location_embeddings_from_ericks_model")

    # --- Process pano_spoofed (Version 2.0 format) ---
    for _city_dir in (_spoofed_base / "pano_spoofed").glob("*/"):
        _v2_pickle = pickle.load((_city_dir / "embeddings/embeddings.pkl").open("rb"))

        _output_v2 = dict(
            version=_v2_pickle["version"],
            embedding_model=str(model_base),
            embedding_dim=EMBEDDING_DIM,
            panoramas=_v2_pickle["panoramas"],
            description_id_to_idx=_v2_pickle["description_id_to_idx"],
        )

        # Convert description embeddings
        _output_tensor = torch.ones((_v2_pickle["description_embeddings"].shape[0], EMBEDDING_DIM)) * torch.nan
        for _key_batch in itertools.batched(_v2_pickle["description_id_to_idx"].items(), 50):
            _keys = [x[0] for x in _key_batch]
            _indices = [x[1] for x in _key_batch]

            # Extract sentences from panoramas dict
            def _get_sentence(key):
                pid = "__".join(key.split("__")[:-1])
                landmark_idx = int(key.split("__")[-1].split("_")[1])
                return _v2_pickle["panoramas"][pid]["landmarks"][landmark_idx]["description"]

            _sentences = [_get_sentence(k) for k in _keys]
            with torch.no_grad():
                _new_emb = model(_sentences)["base_embedding"].cpu()
            for i, idx in enumerate(_indices):
                _output_tensor[idx] = _new_emb[i]

        assert not torch.any(torch.isnan(_output_tensor))
        _output_v2["description_embeddings"] = _output_tensor

        _out_dir = _output_base / "pano_spoofed" / _city_dir.name / "embeddings"
        _out_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(_output_v2, (_out_dir / "embeddings.pkl").open("wb"))
        print(f"Processed pano_spoofed/{_city_dir.name}: {_output_tensor.shape}")

    # --- Process osm_spoofed (Version 1.0 format) ---
    _osm_emb, _osm_id_to_idx = pickle.load((_spoofed_base / "osm_spoofed/embeddings/embeddings.pkl").open("rb"))

    # Load sentences from sentences.jsonl
    _osm_sentences = {}
    with open(_spoofed_base / "osm_spoofed/sentences/sentences.jsonl") as f:
        for line in f:
            _entry = json.loads(line)
            _custom_id = _entry["custom_id"]
            _sentence = _entry["response"]["body"]["choices"][0]["message"]["content"]
            _osm_sentences[_custom_id] = _sentence

    _new_osm_tensor = torch.ones((_osm_emb.shape[0], EMBEDDING_DIM)) * torch.nan
    for _key_batch in itertools.batched(_osm_id_to_idx.items(), 50):
        _keys = [x[0] for x in _key_batch]
        _indices = [x[1] for x in _key_batch]
        _sentences = [_osm_sentences[k] for k in _keys]
        with torch.no_grad():
            _new_emb = model(_sentences)["base_embedding"].cpu()
        for i, idx in enumerate(_indices):
            _new_osm_tensor[idx] = _new_emb[i]

    assert not torch.any(torch.isnan(_new_osm_tensor))

    _out_dir = _output_base / "osm_spoofed/embeddings"
    _out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump((_new_osm_tensor, _osm_id_to_idx), (_out_dir / "embeddings.pkl").open("wb"))
    print(f"Processed osm_spoofed: {_new_osm_tensor.shape}")

    return


@app.cell
def _(Path, json, pickle):
    _spoofed_pano_output_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location_embeddings_from_ericks_model/")
    _spoofed_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location")
    pano_embs = pickle.load((_spoofed_pano_output_base / "pano_spoofed/Seattle/embeddings/embeddings.pkl").open("rb"))
    osm_embs = pickle.load((_spoofed_pano_output_base / "osm_spoofed/embeddings/embeddings.pkl").open("rb"))

    osm_sentences = {}
    with open(_spoofed_base / "osm_spoofed/sentences/sentences.jsonl") as f:
        for line in f:
            _entry = json.loads(line)
            _custom_id = _entry["custom_id"]
            _sentence = _entry["response"]["body"]["choices"][0]["message"]["content"]
            osm_sentences[_custom_id] = _sentence
    return osm_sentences, pano_embs


@app.cell
def _():

    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version="spoofed_v1",
    )
    dataset = vd.VigorDataset('/data/overhead_matching/datasets/VIGOR/Seattle', _config)
    return (dataset,)


@app.cell
def _():
    from experimental.overhead_matching.swag.model.semantic_landmark_utils import custom_id_from_props, prune_landmark
    return (custom_id_from_props,)


@app.cell
def _(custom_id_from_props, dataset, osm_sentences, pano_embs):
    dataset_matching_pairs_sentences = []
    for item in iter(dataset):
        _pano_id = item.panorama_metadata["pano_id"]
        if _pano_id not in pano_embs["panoramas"]:
            continue
        _pano_landmarks = [_x["description"] for _x in pano_embs["panoramas"][_pano_id]["landmarks"]]
        _pano_embeddings = [""]
        _landmarks = item.panorama_metadata["landmarks"]
        _osm_landmarks = []
        for _l in _landmarks:
            _d = dict(
                _landmark_sentence = osm_sentences[custom_id_from_props(_l["pruned_props"])],
                _street_name = _l["addr:street"],
                _buisness_name = _l["name"],
            )
            _osm_landmarks.append(_d)
        dataset_matching_pairs_sentences.append(dict(pano_sentences=_pano_landmarks, osm_landmarks=_osm_landmarks))
    return


@app.cell
def _(model):
    model
    return


app._unparsable_cell(
    r"""
    for _item in dataset_matching_pairs_sentences:
    
    """,
    name="_"
)


@app.cell
def _(out):
    print(out["panoramas"])
    print(out["location_type_embeddings"].shape)
    print(out["proper_noun_to_idx"])
    print(out["location_type_to_idx"])
    print(out["description_id_to_idx"])
    print(len(out["description_id_to_idx"]))
    return


@app.cell
def _(dataset, model, sentence_dict, slu, torch, vocab):
    examined_ids = set()
    for _idx in range(10000):
        _pruned_props = dataset._landmark_metadata.pruned_props[_idx]
        _custom_id = slu.custom_id_from_props(_pruned_props)
        _sentence = sentence_dict[_custom_id]
        if _custom_id in examined_ids:
            continue
        examined_ids.add(_custom_id)

        _model_output = model([_sentence])
        _class_type = None
        _vocab = None
        for k, _ in _pruned_props:
            if k in _model_output["classification_logits"]:
                _class_type = k
                _vocab = {v: k for k, v in vocab[_class_type].items()}
                break
        if _class_type is None:
            print(f'{_pruned_props} No classification task found')
            continue

        _distribution = torch.nn.functional.softmax(_model_output["classification_logits"][_class_type]).detach().cpu()
        _values, _idxs = torch.sort(_distribution.squeeze(), descending=True)

        _preds = []
        for _i in range(3):
            _preds.append(f"{_vocab[_idxs[_i].item()]}: {_values[_i].item(): 0.3f}")

        print(f"{_pruned_props} {', '.join(_preds)} {_sentence=} ")

    return


@app.cell
def _(model_outpt):
    model_outpt
    return


@app.cell
def _():
    import experimental.overhead_matching.swag.data.osm_sentence_generator as osg
    return


@app.cell
def _(vocab):
    vocab.keys()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
