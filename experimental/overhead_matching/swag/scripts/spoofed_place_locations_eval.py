import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from experimental.overhead_matching.swag.data import vigor_dataset as vd
    from experimental.overhead_matching.swag.model import (
        panorama_semantic_landmark_extractor as psle,
        semantic_landmark_extractor as sle,
        swag_config_types as sct,
        swag_model_input_output as swio,
        semantic_landmark_utils as slu,
    )
    from pathlib import Path
    import json
    import pickle

    import common.torch.load_torch_deps
    import torch

    from experimental.overhead_matching.swag.scripts import sentence_configs
    from experimental.overhead_matching.swag.model import sentence_embedding_model
    from experimental.overhead_matching.swag.evaluation import evaluate_swag as es
    from experimental.overhead_matching.swag.evaluation import osm_tag_similarity as ots


    import itertools
    import tqdm

    import seaborn as sns
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.pyplot as plt

    from experimental.overhead_matching.swag.evaluation import proper_noun_matcher_python as pnm

    return (
        Path,
        es,
        itertools,
        json,
        mo,
        ots,
        pickle,
        plt,
        pnm,
        sentence_configs,
        sentence_embedding_model,
        slu,
        sns,
        torch,
        tqdm,
        vd,
    )


@app.cell
def _():
    import common.torch.load_and_save_models as lsm
    return (lsm,)


@app.cell(hide_code=True)
def _(Path, mo, pickle, slu, vd):
    @mo.persistent_cache

    def load_vigor_dataset(city):
        _config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None, 
            panorama_tensor_cache_info=None,
            should_load_images=False,
            landmark_version="v4_202001"
        )
        dataset_path = Path(f'/data/overhead_matching/datasets/VIGOR/{city}')
        return vd.VigorDataset(dataset_path, _config)

    def _load_pano_data_from_pano_id_for_city(base_path, vigor_dataset, valid_lm_pano_ids):
        sentence_from_pano_lm_id, metadata_by_pano_id, _ = slu.make_sentence_dict_from_pano_jsons(
            slu.load_all_jsonl_from_folder(base_path / 'sentences'))
        out = {}
        for _, row in vigor_dataset._panorama_metadata.iterrows():
            pano_id = row.pano_id
            k = row.path.stem
            v = metadata_by_pano_id[k]
            sat_idxs = row.positive_satellite_idxs + row.semipositive_satellite_idxs
            landmarks = []
            for lm_idx, lm_data in enumerate(v):
                if (pano_id, lm_data["landmark_idx"]) in valid_lm_pano_ids:
                    landmarks.append({
                        "landmark_idx": lm_idx,
                        "description": None, 
                        "sentence": sentence_from_pano_lm_id[lm_data["custom_id"]]
                    })
            out[pano_id] = dict(
                landmarks=landmarks,
                sat_idxs=sat_idxs
            )
        return out

    def load_sat_data_from_sat_id(vigor_dataset, valid_pruned_props):
        out = {}
        for sat_idx, row in vigor_dataset._satellite_metadata.iterrows():
            pano_idxs = row.positive_panorama_idxs + row.semipositive_panorama_idxs
            pano_ids = vigor_dataset._panorama_metadata.iloc[pano_idxs].pano_id.values

            landmark_idxs = row.landmark_idxs
            landmark_ids = set(vigor_dataset._landmark_metadata.iloc[landmark_idxs].pruned_props.values)
            landmark_ids = list(landmark_ids & valid_pruned_props)
            out[sat_idx] = dict(pano_ids = pano_ids, landmark_ids=landmark_ids)
        return out

    def _load_city(city):
        vigor_dataset = load_vigor_dataset(city)
        pano_base_path = Path(f'/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/{city}')
        pano_embeddings_path = pano_base_path / "embeddings/embeddings.pkl"
        openai_pano_emb, pkl_emb_idx_from_pano_lm_id = pickle.loads(pano_embeddings_path.read_bytes())

        osm_base_path = Path(f"/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses")
        osm_embeddings_path = osm_base_path / "embeddings/embeddings.pkl"
        openai_osm_emb, osm_emb_idx_from_custom_id = pickle.loads(osm_embeddings_path.read_bytes())
        osm_sentence_from_osm_id = pickle.loads((osm_base_path / "sentences/sentences.pkl").read_bytes())

        embedding_idx_from_pano_landmark_id = {}
        for k, v in pkl_emb_idx_from_pano_lm_id.items():
            pano_and_gps, landmark_str = k.split(",__")
            pano_id = pano_and_gps.split(',')[0]
            lm_id_for_pano = int(landmark_str.split('_')[1])
            embedding_idx_from_pano_landmark_id[(pano_id, lm_id_for_pano)] = v
        valid_pano_landmark_ids = set(embedding_idx_from_pano_landmark_id.keys())

        all_city_pruned_props = set(vigor_dataset._landmark_metadata.pruned_props)
        valid_emb_pruned_props = {x for x in all_city_pruned_props 
                                if slu.custom_id_from_props(x) in osm_emb_idx_from_custom_id}


        embedding_idx_from_osm_id = {}
        for pruned_props in valid_emb_pruned_props:
            custom_id = slu.custom_id_from_props(pruned_props)
            embedding_idx_from_osm_id[pruned_props] = osm_emb_idx_from_custom_id[custom_id]

        sentence_keys = list(osm_sentence_from_osm_id.keys())
        for osm_id in sentence_keys:
            if osm_id not in valid_emb_pruned_props:
                del osm_sentence_from_osm_id[osm_id]

        assert set(osm_sentence_from_osm_id.keys()) == set(embedding_idx_from_osm_id.keys()), (
            f"{(set(osm_sentence_from_osm_id.keys()) - set(embedding_idx_from_osm_id.keys()))=} {(set(embedding_idx_from_osm_id.keys()) - set(osm_sentence_from_osm_id.keys()))=}")

        out = dict(
            pano_data_from_pano_id = _load_pano_data_from_pano_id_for_city(pano_base_path, vigor_dataset, valid_pano_landmark_ids),
            embedding_idx_from_pano_landmark_id = embedding_idx_from_pano_landmark_id,
            embedding_idx_from_osm_id = embedding_idx_from_osm_id,
            sat_data_from_sat_id = load_sat_data_from_sat_id(vigor_dataset, valid_emb_pruned_props),
            osm_sentence_from_osm_id = osm_sentence_from_osm_id,
            thing_semipositives = None,
            place_semipositives = None,
            embeddings = dict(
                openai=dict(
                    pano_embeddings=openai_pano_emb,
                    osm_embeddings=openai_osm_emb,
                )
            )
        )
        return out


    def create_vigor_datasets(cities):
        out = {}
        for city in cities:
            out[city.lower()] = _load_city(city)
        return out 

    return create_vigor_datasets, load_sat_data_from_sat_id, load_vigor_dataset


@app.cell(hide_code=True)
def _(Path, json, mo, pickle, slu, torch, vd):
    @mo.persistent_cache
    def create_spoofed_datasets():
        _config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None, 
            panorama_tensor_cache_info=None,
            should_load_images=False,
            landmark_version="spoofed_v1"
        )
        _spoofed_base_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location/')
        _spoofed_metadata_base_path = _spoofed_base_path / 'metadata'

        _spoofed_osm_path =_spoofed_base_path / 'openai/osm_spoofed/'

        _osm_embeddings = {}
        _emb_idx_from_osm_id = {}
        _osm_embeddings["openai"], _emb_idx_from_osm_id = pickle.loads((_spoofed_osm_path / "embeddings/embeddings.pkl").read_bytes())

        _hashed_osm = pickle.loads(Path(_spoofed_base_path / "hashed/osm_spoofed/embeddings/embeddings.pkl").read_bytes())
        for kind in ["thing", "place"]:
            _osm_embeddings[f"hashed_{kind}"], _kind_emb_idx_from_osm_id, _ = _hashed_osm[kind]
            assert _emb_idx_from_osm_id == _kind_emb_idx_from_osm_id

        _osm_sentence_from_custom_id, _ = slu.make_sentence_dict_from_json(slu.load_all_jsonl_from_folder(
            _spoofed_osm_path / "sentences"))

        _osm_tuple_id_from_osm_id = {}

        datasets = {}
        for _city in ["Chicago", 'Seattle']:
            _dataset = vd.VigorDataset(Path(f'/data/overhead_matching/datasets/VIGOR/{_city}'), _config)
            _metadata = json.loads((_spoofed_metadata_base_path / f"{_city.lower()}_pairs.json").read_text())
            _embeddings_pkl = pickle.loads((_spoofed_base_path / f"openai/pano_spoofed/{_city}/embeddings/embeddings.pkl").read_bytes())

            _hashed_embeddings_pkl = pickle.loads((_spoofed_base_path / f"hashed/pano_spoofed/{_city}/embeddings/embeddings.pkl").read_bytes())
            _hashed_pano_landmark_embeddings = torch.zeros((_hashed_embeddings_pkl["description_embeddings"].shape[0],
                                                   _hashed_embeddings_pkl["proper_noun_embeddings"].shape[1]))
            assert _embeddings_pkl["description_id_to_idx"] == _hashed_embeddings_pkl["description_id_to_idx"]

            _pano_idxs_from_thing = {}
            _pano_idxs_from_place = {}

            for _idx, _item in enumerate(_metadata):
                _pano_idxs_from_thing[_item["business_name"]] = _pano_idxs_from_thing.get(_item["business_name"], []) + [_idx]
                _pano_idxs_from_place[_item["address"]] = _pano_idxs_from_place.get(_item["address"], []) + [_idx]

            _num_panos = len(_metadata)
            _thing_semipositive_matches = torch.zeros((_num_panos, _num_panos), dtype=bool)
            _place_semipositive_matches = torch.zeros((_num_panos, _num_panos), dtype=bool)
            for _idx, _item in enumerate(_metadata):
                _thing_semipositive_matches[_idx, _pano_idxs_from_thing[_item["business_name"]]] = True
                _place_semipositive_matches[_idx, _pano_idxs_from_place[_item["address"]]] = True

            _thing_semipositive_matches.fill_diagonal_(False)
            _place_semipositive_matches.fill_diagonal_(False)

            _pano_data_from_pano_id = {}
            _sat_data_from_sat_idx = {}
            _embedding_idx_from_pano_landmark_id = {}
            _embedding_idx_from_osm_id = {}
            _osm_sentence_from_osm_id = {}

            for _idx, _pano_id in enumerate(_embeddings_pkl["panoramas"]):
                _pano_metadata = _metadata[_idx]
                _pano_landmarks = _embeddings_pkl["panoramas"][_pano_id]["landmarks"]

                _osm_id = slu.custom_id_from_props({
                    "name": _pano_metadata["business_name"],
                    "addr:street": _pano_metadata["address"]})

                # Pull the correct place/thing embedding from the hashed embeddings

                # Add to embedding_idx from pano_landmark_id
                for _landmark_data in _hashed_embeddings_pkl["panoramas"][_pano_id]["landmarks"]:
                    _landmark_idx = _landmark_data["landmark_idx"]
                    _proper_noun = _landmark_data["proper_nouns"]
                    assert len(_proper_noun) == 1
                    _proper_noun = _proper_noun[0]
                    _proper_noun_emb_idx = _hashed_embeddings_pkl["proper_noun_to_idx"][_proper_noun]
                    _output_idx = _embeddings_pkl["description_id_to_idx"][f"{_pano_id}__landmark_{_landmark_idx}"]
                    _embedding_idx_from_pano_landmark_id[(_pano_id, _landmark_idx)] = _output_idx

                    _hashed_pano_landmark_embeddings[_output_idx] = (
                        _hashed_embeddings_pkl["proper_noun_embeddings"][_proper_noun_emb_idx])

                # Add to embeddings_idx from osm_id
                _osm_tuple_id = (
                    _pano_metadata["business_name"],
                    _pano_metadata["address"])
                _osm_tuple_id_from_osm_id[_osm_id] = _osm_tuple_id

                _embedding_idx_from_osm_id[_osm_tuple_id] = _emb_idx_from_osm_id[_osm_id]

                # add to osm_sentence_from_osm_id
                _osm_sentence_from_osm_id[_osm_tuple_id] = _osm_sentence_from_custom_id[_osm_id]

                # add to pano_data_from_pano_id
                _row = _dataset._panorama_metadata[_dataset._panorama_metadata.pano_id == _pano_id].iloc[0]
                _sat_idxs = (_row.positive_satellite_idxs + _row.semipositive_satellite_idxs)
                _pano_data_from_pano_id[_pano_id] = {
                    "osm_id": _osm_tuple_id,
                    'landmarks': [{
                        "landmark_idx": 0,
                        "description": _pano_metadata["business_name"],
                        "sentence":_pano_landmarks[0]["description"]},
                        {"landmark_idx": 1,
                        "description":_pano_metadata["address"],
                        "sentence":_pano_landmarks[1]["description"]}],
                    "sat_idxs": _sat_idxs
                }

                # Add to sat_data_from_sat_id
                for _sat_idx in _sat_idxs:
                    _sat_metadata = _sat_data_from_sat_idx.get(_sat_idx, {"landmark_ids": [], "pano_ids": []})
                    _sat_metadata["landmark_ids"].append(_osm_tuple_id)
                    _sat_metadata["pano_ids"].append(_pano_id)
                    _sat_data_from_sat_idx[_sat_idx] = _sat_metadata


            datasets[_city.lower()] = dict(
                pano_data_from_pano_id = _pano_data_from_pano_id,
                sat_data_from_sat_id = _sat_data_from_sat_idx,
                embedding_idx_from_pano_landmark_id = _embedding_idx_from_pano_landmark_id,
                embedding_idx_from_osm_id = _embedding_idx_from_osm_id,
                osm_sentence_from_osm_id = _osm_sentence_from_osm_id,
                thing_semipositives = _thing_semipositive_matches,
                place_semipositives = _place_semipositive_matches,
                embeddings = dict(openai={
                    "pano_embeddings": _embeddings_pkl["description_embeddings"],
                    "osm_embeddings": _osm_embeddings["openai"],
                },
                hashed_place={
                    "pano_embeddings": torch.nn.functional.normalize(_hashed_pano_landmark_embeddings),
                    "osm_embeddings": torch.nn.functional.normalize(_osm_embeddings["hashed_place"])},
                hashed_thing={
                    "pano_embeddings": torch.nn.functional.normalize(_hashed_pano_landmark_embeddings),
                    "osm_embeddings": torch.nn.functional.normalize(_osm_embeddings["hashed_thing"])})
            )

        return datasets

    # datasets = create_spoofed_datasets()
    return


@app.cell(hide_code=True)
def _(
    itertools,
    json,
    mo,
    sentence_configs,
    sentence_embedding_model,
    torch,
    tqdm,
):
    @mo.persistent_cache
    def add_model_embeddings(datasets, model_name, model_path):
        for _city in datasets:
            print(model_name, _city)
            _checkpoint = torch.load(model_path / 'best_model.pt')
            _vocab = json.loads((model_path / 'tag_vocabs.json').read_text())
            _train_config = sentence_configs.load_config(model_path / 'config.json')
            _model = sentence_embedding_model.create_model_from_config(
                _train_config.model,
                tag_vocabs=_vocab,
                classification_task_names=_train_config.classification_tags,
                contrastive_task_names=_train_config.contrastive_tags).cuda()
            _model.load_state_dict(_checkpoint)
            _EMBEDDING_DIM = _model.base_dim

            _all_embeddings = datasets[_city]["embeddings"]
            _existing_embeddings = _all_embeddings["openai"] if "openai" in _all_embeddings else _all_embeddings["gemini"]

            _pano_embeddings = torch.zeros((_existing_embeddings["pano_embeddings"].shape[0], _model.base_dim))
            _osm_embeddings = torch.zeros((_existing_embeddings["osm_embeddings"].shape[0], _model.base_dim))
            _pano_place_embeddings = torch.zeros((_existing_embeddings["pano_embeddings"].shape[0], _model.config.projection_dim))
            _osm_place_embeddings = torch.zeros((_existing_embeddings["osm_embeddings"].shape[0], _model.config.projection_dim))
            _pano_thing_embeddings = torch.zeros((_existing_embeddings["pano_embeddings"].shape[0], _model.config.projection_dim))
            _osm_thing_embeddings = torch.zeros((_existing_embeddings["osm_embeddings"].shape[0], _model.config.projection_dim))

            for _batch in tqdm.tqdm(itertools.batched(
                    datasets[_city]["pano_data_from_pano_id"].items(), n=8), desc="pano_lm_emb"):
                _sentences = []
                _emb_idxs = []
                for _pano_id, _pano_data in _batch:
                    for _lm_data in _pano_data["landmarks"]:
                        _pano_emb_idx = datasets[_city][
                            "embedding_idx_from_pano_landmark_id"][(_pano_id, _lm_data["landmark_idx"])]
                        _emb_idxs.append(_pano_emb_idx)
                        _sentences.append(_lm_data["sentence"])

                _emb_min = min(_emb_idxs)
                _emb_max = max(_emb_idxs)
                assert _emb_min < _pano_embeddings.shape[0]
                assert _emb_max < _pano_embeddings.shape[0]
                assert _emb_min < _pano_place_embeddings.shape[0]
                assert _emb_max < _pano_place_embeddings.shape[0]
                assert _emb_min < _pano_thing_embeddings.shape[0]
                assert _emb_max < _pano_thing_embeddings.shape[0]
                with torch.no_grad():
                    _model_output = _model(_sentences)
                    _pano_embeddings[_emb_idxs] = _model_output["base_embedding"].cpu()
                    _pano_place_embeddings[_emb_idxs] = _model_output["contrastive_embeddings"]["addr:street"].cpu()
                    _pano_thing_embeddings[_emb_idxs] = _model_output["contrastive_embeddings"]["name"].cpu()


            for _batch in tqdm.tqdm(itertools.batched(
                    datasets[_city]["osm_sentence_from_osm_id"].items(), n=256), desc="osm_emb"):
                _sentences = []
                _emb_idxs = []
                for (_osm_id, _osm_sentence) in _batch:
                    _osm_emb_idx = datasets[_city]["embedding_idx_from_osm_id"][_osm_id]
                    _emb_idxs.append(_osm_emb_idx)
                    _sentences.append(_osm_sentence)

                _emb_min = min(_emb_idxs)
                _emb_max = max(_emb_idxs)
                assert _emb_min < _osm_embeddings.shape[0]
                assert _emb_max < _osm_embeddings.shape[0]
                assert _emb_min < _osm_place_embeddings.shape[0]
                assert _emb_max < _osm_place_embeddings.shape[0]
                assert _emb_min < _osm_thing_embeddings.shape[0]
                assert _emb_max < _osm_thing_embeddings.shape[0]
                with torch.no_grad():
                    _model_output = _model(_sentences)
                    _osm_embeddings[_emb_idxs] = _model_output["base_embedding"].cpu()
                    _osm_place_embeddings[_emb_idxs] = _model_output["contrastive_embeddings"]["addr:street"].cpu()
                    _osm_thing_embeddings[_emb_idxs] = _model_output["contrastive_embeddings"]["name"].cpu()

            datasets[_city]["embeddings"][model_name] = {
                "pano_embeddings": _pano_embeddings,
                "osm_embeddings": _osm_embeddings
            }
            datasets[_city]["embeddings"][f"{model_name}_place"] = {
                "pano_embeddings": _pano_place_embeddings,
                "osm_embeddings": _osm_place_embeddings
            }
            datasets[_city]["embeddings"][f"{model_name}_thing"] = {
                "pano_embeddings": _pano_thing_embeddings,
                "osm_embeddings": _osm_thing_embeddings
            }


    return (add_model_embeddings,)


@app.cell(hide_code=True)
def _(Path, load_sat_data_from_sat_id, load_vigor_dataset, mo, pickle, slu):
    def _load_pano_data_from_pano_id(pano_base_path, vigor_dataset):
        pano_metadata = vigor_dataset._panorama_metadata.set_index('pano_id')
        emb_pkl = pickle.loads((pano_base_path / "embeddings/embeddings.pkl").read_bytes())

        pano_data_from_pano_id = {}
        for k, v in emb_pkl["panoramas"].items():
            pano_id = k.split(',')[0]
            landmark_data = []
            row = pano_metadata.loc[pano_id]
            for landmark in v["landmarks"]:
                landmark_data.append({
                    'landmark_idx': landmark["landmark_idx"],
                    "sentence": landmark["description"],
                    "proper_nouns": landmark["proper_nouns"],
                }) 
            pano_data_from_pano_id[pano_id] = dict(
                landmarks=landmark_data,
                location_sentence=v["location_type"],
                latlon=(row.lat, row.lon),
                sat_idxs=row.positive_satellite_idxs + row.semipositive_satellite_idxs
            )
        pano_embeddings = emb_pkl["description_embeddings"]
        pkl_emb_idx_from_pano_lm_id = emb_pkl["description_id_to_idx"]
        emb_idx_from_pano_lm_id = {}
        for k, v in pkl_emb_idx_from_pano_lm_id.items():
            pano_id = k.split(',')[0]
            landmark_idx = int(k.split('_')[-1])
            emb_idx_from_pano_lm_id[(pano_id, landmark_idx)] = v
        return pano_data_from_pano_id, pano_embeddings, emb_idx_from_pano_lm_id

    def _filter_for_proper_nouns(d):
        to_remove = []
        for pano_id, pano_data in d["pano_data_from_pano_id"].items():
            landmarks_w_proper_nouns = list(filter(lambda x: len(x['proper_nouns']) > 0, pano_data["landmarks"]))
            if len(landmarks_w_proper_nouns) == 0:
                to_remove.append(pano_id)
            for lm in landmarks_w_proper_nouns:
                if len(lm["proper_nouns"]) == 1:
                    tag_on = f' There is a sign that reads "{lm["proper_nouns"][0]}".'
                else:
                    sign_strs = [f'"{x}"' for x in lm["proper_nouns"]]
                    tag_on = f' There are signs that read {", ".join(sign_strs[:-1])} and {sign_strs[-1]}.'
                lm["sentence"] = f"{lm["sentence"]} {tag_on}"
            d["pano_data_from_pano_id"][pano_id]["landmarks"] = landmarks_w_proper_nouns

        for pano_id in to_remove:
            del d["pano_data_from_pano_id"][pano_id]

    @mo.persistent_cache
    def _load_city(city):
        vigor_dataset = load_vigor_dataset(city)
        pano_base_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_gemini/') / city
        pano_data_from_pano_id, pano_embeddings, emb_idx_from_pano_lm_id = _load_pano_data_from_pano_id(pano_base_path, vigor_dataset)

        osm_base_path = Path(f"/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses")
        osm_embeddings_path = osm_base_path / "embeddings/embeddings.pkl"
        openai_osm_emb, osm_emb_idx_from_custom_id = pickle.loads(osm_embeddings_path.read_bytes())
        osm_sentence_from_osm_id = pickle.loads((osm_base_path / "sentences/sentences.pkl").read_bytes())

        all_city_pruned_props = set(vigor_dataset._landmark_metadata.pruned_props)
        valid_emb_pruned_props = {x for x in all_city_pruned_props 
                                if slu.custom_id_from_props(x) in osm_emb_idx_from_custom_id}


        embedding_idx_from_osm_id = {}
        for pruned_props in valid_emb_pruned_props:
            custom_id = slu.custom_id_from_props(pruned_props)
            embedding_idx_from_osm_id[pruned_props] = osm_emb_idx_from_custom_id[custom_id]

        sentence_keys = list(osm_sentence_from_osm_id.keys())
        for osm_id in sentence_keys:
            if osm_id not in valid_emb_pruned_props:
                del osm_sentence_from_osm_id[osm_id]

        all_city_pruned_props = set(vigor_dataset._landmark_metadata.pruned_props)
        valid_emb_pruned_props = {x for x in all_city_pruned_props 
                                if slu.custom_id_from_props(x) in osm_emb_idx_from_custom_id}

        assert set(osm_sentence_from_osm_id.keys()) == set(embedding_idx_from_osm_id.keys()), (
            f"{(set(osm_sentence_from_osm_id.keys()) - set(embedding_idx_from_osm_id.keys()))=} {(set(embedding_idx_from_osm_id.keys()) - set(osm_sentence_from_osm_id.keys()))=}")

        out = dict(
            pano_data_from_pano_id=pano_data_from_pano_id, 
            embedding_idx_from_pano_landmark_id=emb_idx_from_pano_lm_id,
            embedding_idx_from_osm_id=embedding_idx_from_osm_id,
            osm_sentence_from_osm_id=osm_sentence_from_osm_id,
            sat_data_from_sat_id = load_sat_data_from_sat_id(vigor_dataset, valid_emb_pruned_props),
            embeddings = dict(gemini = dict(pano_embeddings=pano_embeddings, osm_embeddings=openai_osm_emb)),
            thing_semipositives = None,
            place_semipositives = None,
        )

        print(f"before filtering: {sum([len(x["landmarks"]) for x in out["pano_data_from_pano_id"].values()])=} {len(out["osm_sentence_from_osm_id"])=}")
        _filter_for_proper_nouns(out)
        print(f"after filtering: {sum([len(x["landmarks"]) for x in out["pano_data_from_pano_id"].values()])=} {len(out["osm_sentence_from_osm_id"])=}")

        return out

    def create_proper_noun_datasets(cities):
        out = {}
        for city in cities:
            print(f'Loading {city}')
            out[city.lower()] = _load_city(city) 
        return out

    return (create_proper_noun_datasets,)


@app.cell
def _(
    Path,
    add_model_embeddings,
    create_proper_noun_datasets,
    create_vigor_datasets,
):
    datasets = create_vigor_datasets(["Chicago", "Seattle"])
    _model_base = Path('/data/overhead_matching/models/sentence_embeddings/')
    add_model_embeddings(datasets, "template_w_llm", _model_base / 'train_with_osm_sentences_v4')

    proper_noun_datasets  = create_proper_noun_datasets(["Chicago", "Seattle"])
    _model_base = Path('/data/overhead_matching/models/sentence_embeddings/')
    add_model_embeddings(proper_noun_datasets, "template_w_llm", _model_base / 'train_with_osm_sentences_v4')
    return datasets, proper_noun_datasets


@app.cell
def _(torch):
    def compute_recip_rank(similarity_matrix):
        pos_sims = torch.diag(similarity_matrix)
        ranks = torch.sum(pos_sims[:, None] <= similarity_matrix, 1).float()
        return 1 / ranks
    return (compute_recip_rank,)


@app.cell
def _():
    EVAL_CITY = "Seattle"
    return (EVAL_CITY,)


@app.cell(hide_code=True)
def _(EVAL_CITY, compute_recip_rank, datasets, itertools, pd, plt, torch):
    _d = datasets[EVAL_CITY.lower()]
    _thing_emb_idxs = [_d["embedding_idx_from_pano_landmark_id"][(_pano_id, 0)] for _pano_id in _d["pano_data_from_pano_id"]]
    _place_emb_idxs = [_d["embedding_idx_from_pano_landmark_id"][(_pano_id, 1)] for _pano_id in _d["pano_data_from_pano_id"]]
    # _osm_emb_idxs = [_d["embedding_idx_from_osm_id"][_data["osm_id"]] for _pano_id, _data in _d["pano_data_from_pano_id"].items()]
    _osm_emb_idxs = list(_d["embedding_idx_from_osm_id"].values())

    plt.figure(figsize=(12, 6))

    _records = []
    _simple_records = []
    for (_emb_type, _emb), _exclude_semipos in itertools.product(_d["embeddings"].items(), [True]):
        _thing_embs = _emb["pano_embeddings"][_thing_emb_idxs].cuda()
        _place_embs = _emb["pano_embeddings"][_place_emb_idxs].cuda()
        _osm_embs = _emb["osm_embeddings"][_osm_emb_idxs].cuda()

        _thing_sims = _thing_embs @ _osm_embs.T
        _place_sims = _place_embs @ _osm_embs.T

        if _exclude_semipos:
            _thing_sims[_d["thing_semipositives"]] = -1
            _place_sims[_d["place_semipositives"]] = -1

        _max_thing_sims, _ = torch.max(_thing_sims, 1)
        _max_place_sims, _ = torch.max(_place_sims, 1)

        _pos_thing_sims = torch.diag(_thing_sims)
        _pos_place_sims = torch.diag(_place_sims)

        _thing_to_plot = (_max_thing_sims - _pos_thing_sims).cpu()
        _place_to_plot = (_max_place_sims - _pos_place_sims).cpu()

        _thing_recip_rank = compute_recip_rank(_thing_sims)
        _place_recip_rank = compute_recip_rank(_place_sims)

        for _pano_idx in range(len(_thing_emb_idxs)):
            _records.append({
                "emb_type": _emb_type,
                "eval": "place",
                "pano_idx": _pano_idx,
                "max_sim_diff": _place_to_plot[_pano_idx].item(),
                "recip_rank": _place_recip_rank[_pano_idx].item(),
                "semipos_excluded": _exclude_semipos,
            })
            _records.append({
                "emb_type": _emb_type,
                "eval": "thing",
                "pano_idx": _pano_idx,
                "max_sim_diff": _thing_to_plot[_pano_idx].item(),
                "recip_rank": _thing_recip_rank[_pano_idx].item(),
                "semipos_excluded": _exclude_semipos,
            })

    df = pd.DataFrame.from_records(_records)

    df = df[df["emb_type"].isin(["openai", "template_w_llm", "template_w_llm_place", "template_w_llm_thing"])]

    return (df,)


@app.cell(hide_code=True)
def _(df, mo, plt, sns):
    sns.displot(df, x='max_sim_diff', hue='emb_type', col='eval', row="semipos_excluded", kind='ecdf')
    mo.mpl.interactive(plt.gcf())

    return


@app.cell(hide_code=True)
def _(df, mo, plt, sns):
    sns.displot(df, x='recip_rank', hue='emb_type', col='eval', col_order=["thing", "place"], kind='ecdf')
    # plt.tight_layout()
    mo.mpl.interactive(plt.gcf())

    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return (pd,)


@app.cell
def _(
    EVAL_CITY,
    Path,
    es,
    load_vigor_dataset,
    lsm,
    proper_noun_datasets,
    torch,
):

    # Load the similarities for the baseline

    def compute_baseline_similarity(d, pano_id_subset=None):
        _city = d["city"]
        _baseline_model_path = Path('/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512/')
        _baseline_sat_model = lsm.load_model(_baseline_model_path / "0059_satellite")
        _baseline_pano_model = lsm.load_model(_baseline_model_path / "0059_panorama")
        _vd = load_vigor_dataset(_city)
        if pano_id_subset is None:
            pano_id_subset = list(_vd._panorama_metadata.pano_id)

        _baseline_similarity = es.compute_cached_similarity_matrix(
            _baseline_sat_model, _baseline_pano_model, _vd, 'cuda', use_cached_similarity=True)

        _positives = torch.zeros(_baseline_similarity.shape, dtype=torch.bool)

        for _sat_idx, _row in _vd._satellite_metadata.iterrows():
            _pano_idxs = _row.positive_panorama_idxs + _row.semipositive_panorama_idxs
            if len(_pano_idxs) > 0:
                _positives[_pano_idxs, _sat_idx] = True

        valid_idxs = []
        for pano_idx, row in _vd._panorama_metadata.iterrows():
            if row.pano_id in pano_id_subset: 
                valid_idxs.append(pano_idx)

        _baseline_similarity = _baseline_similarity[valid_idxs]
        _positives = _positives[valid_idxs]
        return {
            "similarity": _baseline_similarity,
            "positives": _positives,
        }

    baseline_sims = compute_baseline_similarity({"city":EVAL_CITY})
    pnd_baseline_sims = compute_baseline_similarity({"city": EVAL_CITY},
                                                    pano_id_subset=list(proper_noun_datasets[EVAL_CITY.lower()]["pano_data_from_pano_id"].keys()))


    return baseline_sims, pnd_baseline_sims


@app.cell
def _(EVAL_CITY, datasets, mo, pnm, proper_noun_datasets, torch, tqdm):
    # Compute pano <-> satellite similarity
    OSM_TEXT_KEYS = ['name', 'brand', 'operator', 'addr:street', 'network']

    def get_osm_texts_from_pruned_props(pruned_props: frozenset) -> list[str]:
        """Extract text fields from OSM pruned_props for matching."""
        props_dict = dict(pruned_props)
        return [v for k, v in props_dict.items() if k in OSM_TEXT_KEYS and isinstance(v, str)]

    @mo.persistent_cache
    def compute_proper_noun_binary_similarity(dataset):
        """Compute binary similarity based on proper noun matches with OSM text fields.

        Returns dict with "similarity" and "positives" tensors, same format as compute_patch_similarity.
        """
        num_sat_patches = max(dataset['sat_data_from_sat_id'].keys()) + 1
        num_panos = len(dataset["pano_data_from_pano_id"])
        key = 'openai' if 'openai' in dataset["embeddings"] else 'gemini'
        num_osm = dataset["embeddings"][key]["osm_embeddings"].shape[0]

        out = torch.full((num_panos, num_sat_patches), -torch.inf)
        positives = torch.zeros((num_panos, num_sat_patches), dtype=torch.bool)

        # Pre-compute OSM texts indexed by embedding idx
        osm_texts_by_idx = [[] for _ in range(num_osm)]
        for osm_id, emb_idx in dataset["embedding_idx_from_osm_id"].items():
            osm_texts_by_idx[emb_idx] = get_osm_texts_from_pruned_props(osm_id)

        # Pre-collect pano landmark data
        pano_for_lm = []
        proper_nouns_for_lm = []

        for pano_idx, (pano_id, pano_data) in enumerate(dataset["pano_data_from_pano_id"].items()):
            positives[pano_idx, pano_data["sat_idxs"]] = True
            for lm in pano_data["landmarks"]:
                lm_key = (pano_id, lm["landmark_idx"])
                if lm_key in dataset["embedding_idx_from_pano_landmark_id"]:
                    pano_for_lm.append(pano_idx)
                    proper_nouns_for_lm.append(lm.get("proper_nouns", []))

        # Compute binary match matrix using C++ implementation
        pano_for_lm_t = torch.tensor(pano_for_lm, device='cpu')
        best_osm_match_from_pano = torch.zeros((num_panos, num_osm), device='cpu')

        if len(proper_nouns_for_lm) > 0:
            # Call C++ proper noun matcher
            match_matrix = pnm.compute_proper_noun_matches(
                proper_nouns_for_lm, osm_texts_by_idx
            )
            chunk_match = torch.from_numpy(match_matrix)
            best_osm_match_from_pano.index_add_(0, pano_for_lm_t, chunk_match)

        # Contract to satellite patches
        for sat_idx, sat_data in tqdm.tqdm(dataset['sat_data_from_sat_id'].items(), desc="sat_contraction"):
            osm_idxs = [dataset['embedding_idx_from_osm_id'][x] for x in sat_data["landmark_ids"]]
            if len(osm_idxs) == 0:
                out[:, sat_idx] = -torch.inf
            else:
                out[:, sat_idx] = torch.max(best_osm_match_from_pano[:, osm_idxs], 1).values

        return {"similarity": out, "positives": positives}

    def compute_similarity(embeddings, pano_idxs):
        return (embeddings["pano_embeddings"][pano_idxs].cuda() @ embeddings["osm_embeddings"].cuda().T)

    def compute_idf_weight(dataset):
        key = "openai" if "openai" in dataset["embeddings"] else "gemini"
        sat_patch_counts_from_osm_idx = torch.zeros(dataset["embeddings"][key]["osm_embeddings"].shape[0])
        for sat_idx, sat_data in dataset["sat_data_from_sat_id"].items():
            lm_ids = []
            for lm in sat_data["landmark_ids"]:
                lm_ids.append(dataset["embedding_idx_from_osm_id"][lm])
            sat_patch_counts_from_osm_idx[lm_ids] += 1
        num_sat_patches = len(dataset["sat_data_from_sat_id"])
        idf = torch.log(num_sat_patches / sat_patch_counts_from_osm_idx)
        return idf

    @mo.persistent_cache
    def compute_patch_similarity(dataset, embeddings, chunk_size=512):

        num_sat_patches = max(dataset['sat_data_from_sat_id'].keys())+1
        num_panos = len(dataset["pano_data_from_pano_id"])
        num_osm = embeddings[0]["osm_embeddings"].shape[0]
        out = torch.full((num_panos, num_sat_patches), -torch.inf)
        positives = torch.zeros((num_panos, num_sat_patches), dtype=torch.bool)
        idf_weight = compute_idf_weight(dataset).cuda()

        # Move OSM embeddings to GPU once
        gpu_osm_embeddings = [emb["osm_embeddings"].cuda() for emb in embeddings]

        # Pre-collect all pano landmark indices and their pano mapping
        all_lm_idxs = []
        pano_for_lm = []
        for pano_idx, (pano_id, pano_data) in enumerate(dataset["pano_data_from_pano_id"].items()):
            positives[pano_idx, pano_data["sat_idxs"]] = True
            for i in range(len(pano_data["landmarks"])):
                if (pano_id, i) in dataset["embedding_idx_from_pano_landmark_id"]:
                    all_lm_idxs.append(dataset["embedding_idx_from_pano_landmark_id"][(pano_id, i)])
                    pano_for_lm.append(pano_idx)

        all_lm_idxs_t = torch.tensor(all_lm_idxs)
        pano_for_lm_t = torch.tensor(pano_for_lm, device='cuda')

        # Batched matmul + scatter-add to aggregate per panorama
        best_osm_match_from_pano = torch.zeros((num_panos, num_osm), device='cpu')
        for start in tqdm.tqdm(range(0, len(all_lm_idxs), chunk_size), desc="pano_lm_contraction"):
            chunk_lm_idxs = all_lm_idxs_t[start:start + chunk_size]
            chunk_pano = pano_for_lm_t[start:start + chunk_size]

            # Sum similarity across all embedding types
            chunk_sim = torch.zeros((len(chunk_lm_idxs), num_osm), device='cuda')
            for emb, gpu_osm in zip(embeddings, gpu_osm_embeddings):
                chunk_sim += emb["pano_embeddings"][chunk_lm_idxs].cuda() @ gpu_osm.T

            chunk_sim *= idf_weight[None, :]
            best_osm_match_from_pano.index_add_(0, chunk_pano.cpu(), chunk_sim.cpu())

        # best_osm_match_from_pano = best_osm_match_from_pano.cpu()

        for sat_idx, sat_data in tqdm.tqdm(dataset['sat_data_from_sat_id'].items(), desc="sat_contraction"):
            # Get the osm landmarks that are in the current OSM patch
            osm_idxs = [dataset['embedding_idx_from_osm_id'][x] for x in sat_data["landmark_ids"]]
            if len(osm_idxs) == 0:
                out[:, sat_idx] = -torch.inf
            else:
                # Get the max similarity between a panorama landmark and the osm landmarks in this patch
                out[:, sat_idx] = torch.max(best_osm_match_from_pano[:, osm_idxs], 1).values

        return {"similarity": out, "positives": positives}

    _d = datasets[EVAL_CITY.lower()]

    _embedding_type = "template_w_llm"
    lm_sims = compute_patch_similarity(
        _d, [_d["embeddings"][_embedding_type],
             _d["embeddings"][f"{_embedding_type}_place"],
             _d["embeddings"][f"{_embedding_type}_thing"]])

    _pnd = proper_noun_datasets[EVAL_CITY.lower()]

    _embedding_type = "template_w_llm"
    pnd_lm_sims = compute_patch_similarity(
        _pnd, [_pnd["embeddings"][_embedding_type],
             _pnd["embeddings"][f"{_embedding_type}_place"],
             _pnd["embeddings"][f"{_embedding_type}_thing"]])

    pnd_proper_noun_sims = compute_proper_noun_binary_similarity(_pnd)

    return lm_sims, pnd_lm_sims, pnd_proper_noun_sims


@app.cell
def _(EVAL_CITY, Path, load_vigor_dataset, mo, ots):
    # Load and evaluate OSM tag extraction predictions
    # Add extraction paths for each city here
    osm_extraction_paths = {
        "Chicago": Path('/tmp/pano_osm_extraction/test_2_output/'),
        "Seattle": Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/Seattle/sentences/'),
    }

    @mo.persistent_cache
    def _load_osm_tag_sims(city: str, extraction_path: Path):
        osm_tag_dataset = ots.create_osm_tag_extraction_dataset(extraction_path, load_vigor_dataset(city))
        print(f"Loaded {len(osm_tag_dataset['pano_data_from_pano_id'])} panoramas with OSM tag extractions for {city}")
        return osm_tag_dataset, ots.compute_osm_tag_match_similarity(osm_tag_dataset)

    if EVAL_CITY in osm_extraction_paths:
        osm_tag_dataset, osm_tag_sims = _load_osm_tag_sims(EVAL_CITY, osm_extraction_paths[EVAL_CITY])
    else:
        print(f"No OSM extraction path configured for {EVAL_CITY}")
        osm_tag_dataset, osm_tag_sims = None, None

    return osm_tag_dataset, osm_tag_sims


@app.cell
def _(osm_tag_dataset):
    osm_tag_dataset["pano_data_from_pano_id"] if osm_tag_dataset else None
    return


@app.cell
def _(
    EVAL_CITY,
    osm_tag_dataset,
    osm_tag_sims,
    pnd_proper_noun_sims,
    proper_noun_datasets,
    torch,
):
    # Find pano-satellite pairs with largest discrepancy between proper noun and tag matching
    # Only look at positive (ground truth) satellite patches for each panorama

    if osm_tag_sims is None or osm_tag_dataset is None:
        print("OSM tag sims not available")
        discrepancy_analysis = None
    else:
        _pnd = proper_noun_datasets[EVAL_CITY.lower()]
        _tag_dataset = osm_tag_dataset

        # Find common pano_ids between the two datasets
        pnd_pano_ids = list(_pnd["pano_data_from_pano_id"].keys())
        tag_pano_ids = list(_tag_dataset["pano_data_from_pano_id"].keys())
        common_pano_ids = set(pnd_pano_ids) & set(tag_pano_ids)
        print(f"Common panos: {len(common_pano_ids)} (pnd: {len(pnd_pano_ids)}, tag: {len(tag_pano_ids)})")

        # Build index mappings
        pnd_idx_from_pano_id = {pid: i for i, pid in enumerate(pnd_pano_ids)}
        tag_idx_from_pano_id = {pid: i for i, pid in enumerate(tag_pano_ids)}

        # Get similarities for common panos
        pn_sims = pnd_proper_noun_sims["similarity"]  # (num_pnd_panos, num_sats)
        tag_sims = osm_tag_sims["max_unit_max"]["similarity"]  # (num_tag_panos, num_sats)

        num_sats = tag_sims.shape[1]

        # Build index arrays for common panos
        common_pano_list = list(common_pano_ids)
        common_pnd_idxs = torch.tensor([pnd_idx_from_pano_id[pid] for pid in common_pano_list])
        common_tag_idxs = torch.tensor([tag_idx_from_pano_id[pid] for pid in common_pano_list])

        # Build positives mask for common panos
        common_positives_mask = torch.zeros((len(common_pano_list), num_sats), dtype=torch.bool)
        for i, pano_id in enumerate(common_pano_list):
            pnd_pano_data = _pnd["pano_data_from_pano_id"][pano_id]
            common_positives_mask[i, pnd_pano_data["sat_idxs"]] = True

        # Get similarities for common panos
        common_pn_sims = pn_sims[common_pnd_idxs]  # (num_common, num_sats)
        common_tag_sims = tag_sims[common_tag_idxs]  # (num_common, num_sats)

        # Matched masks
        pn_matched_mask = common_pn_sims > 0
        tag_matched_mask = common_tag_sims > 0

        # Compute stats vectorized
        pn_tp = (common_positives_mask & pn_matched_mask).sum().item()
        pn_fp = (~common_positives_mask & pn_matched_mask).sum().item()
        tag_tp = (common_positives_mask & tag_matched_mask).sum().item()
        tag_fp = (~common_positives_mask & tag_matched_mask).sum().item()
        both_tp = (common_positives_mask & pn_matched_mask & tag_matched_mask).sum().item()
        total_positives = common_positives_mask.sum().item()
        total_negatives = (~common_positives_mask).sum().item()

        # Compute precision and recall
        pn_precision = pn_tp / (pn_tp + pn_fp) if (pn_tp + pn_fp) > 0 else 0
        tag_precision = tag_tp / (tag_tp + tag_fp) if (tag_tp + tag_fp) > 0 else 0
        pn_recall = pn_tp / total_positives if total_positives > 0 else 0
        tag_recall = tag_tp / total_positives if total_positives > 0 else 0

        # Find discrepancies: positive pairs where tag didn't match
        discrepancy_mask = common_positives_mask & ~tag_matched_mask
        discrepancy_indices = discrepancy_mask.nonzero(as_tuple=False)  # (num_discrepancies, 2)

        discrepancies = []
        for idx in range(min(len(discrepancy_indices), 100)):  # Limit to 100 for memory
            i, sat_idx = discrepancy_indices[idx].tolist()
            pano_id = common_pano_list[i]
            pnd_idx = common_pnd_idxs[i].item()
            tag_idx = common_tag_idxs[i].item()
            discrepancies.append({
                "pano_id": pano_id,
                "sat_idx": sat_idx,
                "pn_sim": common_pn_sims[i, sat_idx].item(),
                "tag_sim": common_tag_sims[i, sat_idx].item(),
                "pnd_idx": pnd_idx,
                "tag_idx": tag_idx,
                "is_positive": True,
                "pn_matched": pn_matched_mask[i, sat_idx].item(),
            })

        # Sort by proper noun similarity (highest first)
        discrepancies.sort(key=lambda x: -x["pn_sim"])

        print(f"\n{'='*70}")
        print(f"COMMON PANOS ONLY ({len(common_pano_ids)} panos)")
        print(f"{'='*70}")
        print(f"  {total_positives} positive pairs, {total_negatives} negative pairs")
        print(f"\n  Proper noun matching:")
        print(f"    Recall:    {pn_tp}/{total_positives} = {100*pn_recall:.1f}%")
        print(f"    Precision: {pn_tp}/{pn_tp + pn_fp} = {100*pn_precision:.1f}%")
        print(f"\n  Tag matching (substring):")
        print(f"    Recall:    {tag_tp}/{total_positives} = {100*tag_recall:.1f}%")
        print(f"    Precision: {tag_tp}/{tag_tp + tag_fp} = {100*tag_precision:.1f}%")
        print(f"\n  Both matched: {both_tp} ({100*both_tp/total_positives:.1f}% of positives)")

        # Compute stats on ALL OSM tag dataset panos (vectorized)
        # Build positives mask for all tag panos
        all_positives_mask = torch.zeros((len(tag_pano_ids), num_sats), dtype=torch.bool)
        for tag_idx, pano_id in enumerate(tag_pano_ids):
            tag_pano_data = _tag_dataset["pano_data_from_pano_id"][pano_id]
            all_positives_mask[tag_idx, tag_pano_data["sat_idxs"]] = True

        # Matched mask: similarity > 0
        all_matched_mask = tag_sims > 0

        # Compute TP, FP, etc. using vectorized operations
        all_tp = (all_positives_mask & all_matched_mask).sum().item()
        all_fp = (~all_positives_mask & all_matched_mask).sum().item()
        all_tag_positives = all_positives_mask.sum().item()
        all_tag_negatives = (~all_positives_mask).sum().item()

        all_tag_recall = all_tp / all_tag_positives if all_tag_positives > 0 else 0
        all_tag_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0

        print(f"\n{'='*70}")
        print(f"ALL OSM TAG PANOS ({len(tag_pano_ids)} panos)")
        print(f"{'='*70}")
        print(f"  {all_tag_positives} positive pairs, {all_tag_negatives} negative pairs")
        print(f"\n  Tag matching (substring):")
        print(f"    Recall:    {all_tp}/{all_tag_positives} = {100*all_tag_recall:.1f}%")
        print(f"    Precision: {all_tp}/{all_tp + all_fp} = {100*all_tag_precision:.1f}%")
        print(f"\nFound {len(discrepancies)} positive pairs where tag matching failed")
        print("\nPositive pairs where tag matching failed:")
        for i, d in enumerate(discrepancies[:20]):
            pano_id = d["pano_id"]
            sat_idx = d["sat_idx"]

            # Get proper nouns from pnd dataset (per landmark)
            pnd_pano_data = _pnd["pano_data_from_pano_id"][pano_id]
            proper_nouns_by_lm = [lm.get("proper_nouns", []) for lm in pnd_pano_data["landmarks"]]

            # Get extracted tags from tag dataset (per landmark)
            tag_pano_data = _tag_dataset["pano_data_from_pano_id"][pano_id]
            extracted_tags_by_lm = []
            for lm in tag_pano_data["landmarks"]:
                lm_tags = []
                primary = lm.get("primary_tag", {})
                if primary.get("key") and primary.get("value"):
                    lm_tags.append((primary["key"], primary["value"]))
                for tag in lm.get("additional_tags", []):
                    if tag.get("key") and tag.get("value"):
                        lm_tags.append((tag["key"], tag["value"]))
                if lm_tags:
                    extracted_tags_by_lm.append(lm_tags)

            # Get OSM props for this satellite patch (per OSM landmark)
            sat_data = _tag_dataset["sat_data_from_sat_id"].get(sat_idx, {})
            osm_landmarks = list(sat_data.get("landmark_ids", []))

            print(f"\n{'='*60}")
            pn_status = "PN:YES" if d.get('pn_matched') else "PN:NO"
            print(f"{i+1}. pano={pano_id}, sat={sat_idx}, {pn_status}, pn_sim={d['pn_sim']:.2f}, tag_sim={d['tag_sim']:.2f}")

            print(f"\n   PANO LANDMARKS (proper nouns):")
            for lm_idx, pns in enumerate(proper_nouns_by_lm[:5]):
                if pns:
                    print(f"     [{lm_idx}]: {pns}")

            print(f"\n   PANO LANDMARKS (extracted tags):")
            for lm_idx, tags in enumerate(extracted_tags_by_lm[:5]):
                print(f"     [{lm_idx}]: {tags}")

            print(f"\n   OSM LANDMARKS in satellite patch ({len(osm_landmarks)} total):")
            for osm_idx, osm_props in enumerate(osm_landmarks[:5]):
                # Show all tags for this OSM landmark, highlighting name-like keys
                props_list = list(osm_props)
                name_tags = [(k, v) for k, v in props_list if k in ('name', 'brand', 'operator')]
                other_tags = [(k, v) for k, v in props_list if k not in ('name', 'brand', 'operator')]
                print(f"     [{osm_idx}]: names={name_tags}, other={other_tags[:3]}{'...' if len(other_tags) > 3 else ''}")

        discrepancy_analysis = discrepancies
    return


@app.cell
def _(
    EVAL_CITY,
    baseline_sims,
    datasets,
    lm_sims,
    mo,
    osm_tag_sims,
    plt,
    pnd_baseline_sims,
    pnd_lm_sims,
    pnd_proper_noun_sims,
    sns,
    torch,
):
    _d = datasets[EVAL_CITY.lower()]
    print(f"{lm_sims["similarity"].shape=} {baseline_sims["similarity"].shape=}")
    _num_panos = len(_d["pano_data_from_pano_id"])
    _num_sats = max(_d["sat_data_from_sat_id"].keys()) + 1

    _positives = torch.zeros((_num_panos, _num_sats), dtype=torch.bool)
    for _pano_idx, (_pano_id, _pano_data) in enumerate(_d["pano_data_from_pano_id"].items()):
        _positives[_pano_idx, _pano_data["sat_idxs"]] = True

    def _compute_pano_patch_sim_recip_rank(sims_and_positives):
        sims = sims_and_positives["similarity"]
        positives = sims_and_positives["positives"]

        pos_pano_sat_sims = sims.clone()
        neg_pano_sat_sims = sims.clone()
        pos_pano_sat_sims[~positives] = -torch.inf
        max_pos_sim_per_pano = torch.max(pos_pano_sat_sims, 1).values
        rank = torch.sum(sims > max_pos_sim_per_pano[:, None], 1) + 1
        return 1.0 / rank

    baseline_recip_ranks = _compute_pano_patch_sim_recip_rank(baseline_sims)
    pnd_baseline_recip_ranks = _compute_pano_patch_sim_recip_rank(pnd_baseline_sims)
    lm_recip_ranks = _compute_pano_patch_sim_recip_rank(lm_sims)
    pnd_lm_recip_ranks = _compute_pano_patch_sim_recip_rank(pnd_lm_sims)
    pnd_proper_noun_recip_ranks = _compute_pano_patch_sim_recip_rank(pnd_proper_noun_sims)

    # Compute recip ranks for all OSM tag matching methods
    osm_tag_methods = ["max_unit_max", "max_unit_mean", "max_unit_sum", "sum_idf_max", "sum_idf_mean", "sum_idf_sum"]
    if osm_tag_sims is not None:
        osm_tag_recip_ranks = {
            method: _compute_pano_patch_sim_recip_rank(osm_tag_sims[method])
            for method in osm_tag_methods
        }
    else:
        osm_tag_recip_ranks = {}

    plt.figure(figsize=(10, 6))
    sns.set_palette(sns.color_palette('tab20'))
    sns.ecdfplot(lm_recip_ranks, label="All Landmarks")
    sns.ecdfplot(pnd_lm_recip_ranks, label="Proper Noun Subset w/ Embeddings")
    sns.ecdfplot(pnd_proper_noun_recip_ranks, label="Proper Noun Subset w/ Text Match")
    sns.ecdfplot(baseline_recip_ranks.cpu(), label="Overhead Imagery Baseline")
    sns.ecdfplot(pnd_baseline_recip_ranks.cpu(), label="Proper Noun Subset Overhead Imagery Baseline")

    # Plot all OSM tag matching methods
    for method in osm_tag_recip_ranks:
        sns.ecdfplot(osm_tag_recip_ranks[method], label=f"OSM Tag: {method}")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Recip. Rank')
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


app._unparsable_cell(
    r"""
    ========def export_similarity_kernel(sims_and_pos, dataset, vigor_dataset):
        emb_idx_from_pano_id = {k: i for i, k in enumerate(dataset[\"pano_data_from_pano_id\"].keys())}
        assert len(dataset[\"sat_data_from_sat_id\"]) == len(vigor_dataset._satellite_metadata)
        num_sats = len(dataset[\"sat_data_from_sat_id\"])
        pano_metadata = vigor_dataset._panorama_metadata

        out = torch.full((len(pano_metadata), num_sats), -torch.inf)
        for out_pano_idx, row in pano_metadata.iterrows():
            in_pano_idx = emb_idx_from_pano_id.get(row.pano_id)
            if in_pano_idx is None:
                continue

            out[out_pano_idx] = sims_and_pos[\"similarity\"][in_pano_idx]

        return {
            \"similarity\": out,
            \"pano_ids\": pano_metadata.pano_id.tolist(),
            \"sat_idxs\": list(range(num_sats)),
        }

    _out_base_path = Path(\"/tmp/new_similarities\")
    _out_base_path.mkdir(exist_ok=True)
    # _EMB_TYPES = [\"template_w_llm\", \"template_w_llm_place\", \"template_w_llm_thing\"]
    # to_export = {
    #     \"landmark_only\": (lambda  x: compute_patch_similarity(x, [x[\"embeddings\"][y] for y in _EMB_TYPES]), datasets),
    #     \"proper_noun_embedding\": (lambda  x: compute_patch_similarity(x, [x[\"embeddings\"][y] for y in _EMB_TYPES]), proper_noun_datasets),
    #     \"proper_noun_text_match\": (compute_proper_noun_binary_similarity, proper_noun_datasets),
    # }
    # 
    # for _name, (_fn, _datasets) in to_export.items():
    #     for _city in _datasets:
    #         print(_name, _city)
    #         _sims = _fn(_datasets[_city])
    #         _sim_mat = export_similarity_kernel(_sims, _datasets[_city], load_vigor_dataset(_city.title()))
    #         torch.save(_sim_mat, _out_base_path / f\"{_name}_{_city}.pt\")

    # Export OSM tag matching similarities
    _osm_extraction_paths = {
        \"chicago\": Path('/tmp/pano_osm_extraction/test_2_output/'),
        \"seattle\": Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/Seattle/sentences/'),
    }
    for _city, _extraction_path in _osm_extraction_paths.items():
        if not _extraction_path.exists():
            print(f\"Skipping OSM tag export for {_city}: {_extraction_path} does not exist\")
            continue
        print(f\"osm_tag_match {_city}\")
        _osm_dataset = ots.create_osm_tag_extraction_dataset(_extraction_path, load_vigor_dataset(_city.title()))
        _osm_sims = ots.compute_osm_tag_match_similarity(_osm_dataset)
        _vigor_dataset = load_vigor_dataset(_city.title())
        for _method, _sims_and_pos in _osm_sims.items():
            _sim_mat = export_similarity_kernel(_sims_and_pos, _osm_dataset, _vigor_dataset)
            torch.save(_sim_mat, _out_base_path / f\"osm_tag_match_{_method}_{_city}.pt\")

    _baseline_seattle_sims = compute_baseline_similarity({\"city\": \"Seattle\"})
    _baseline_chicago_sims = compute_baseline_similarity({\"city\": \"Chicago\"})
    torch.save(_baseline_seattle_sims, _out_base_path / f\"baseline_seattle.pt\")
    torch.save(_baseline_chicago_sims, _out_base_path / f\"baseline_chicago.pt\")
    """,
    name="_"
)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
