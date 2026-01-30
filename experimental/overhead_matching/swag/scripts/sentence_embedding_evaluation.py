import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import common.torch.load_torch_deps
    import copy
    import torch
    from pathlib import Path
    import itertools
    import json
    import pickle
    import tqdm
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    from experimental.overhead_matching.swag.model.sentence_embedding_model import create_model_from_config
    from experimental.overhead_matching.swag.scripts.sentence_configs import load_config, SentenceTrainConfig, msgspec_dec_hook
    return (
        Path,
        copy,
        create_model_from_config,
        itertools,
        json,
        load_config,
        mo,
        pickle,
        torch,
        tqdm,
        vd,
    )


@app.cell
def _(Path, json, load_config, torch):

    _model_name = "train_with_osm_sentences_v3"
    model_base = Path('/data/overhead_matching/models/sentence_embeddings/') / _model_name
    new_dir_name_osm = f"v4_202001_{_model_name}_with_proper_nouns"
    # new_dir_pano_name = f"pano_gemini_{_model_name}"
    import shutil
    checkpoint = torch.load(model_base / 'best_model.pt')
    vocab = json.loads((model_base / 'tag_vocabs.json').read_text())
    train_config = load_config(model_base / 'config.json')
    return (
        checkpoint,
        model_base,
        new_dir_name_osm,
        shutil,
        train_config,
        vocab,
    )


@app.cell
def _(train_config):
    train_config.model
    return


@app.cell
def _(checkpoint, create_model_from_config, train_config, vocab):

    model = create_model_from_config(
        train_config.model,
        tag_vocabs=vocab,
        classification_task_names=train_config.classification_tags,
        contrastive_task_names=train_config.contrastive_tags).cuda()
    model.load_state_dict(checkpoint)
    EMBEDDING_DIM = model.base_dim
    print(EMBEDDING_DIM)
    return EMBEDDING_DIM, model


@app.cell
def _(torch):
    import hashlib
    import numpy as np

    def string_to_hash_embedding(text: str, embedding_dim: int) -> np.ndarray:
        """Convert a string to a deterministic hash-based embedding.

        Uses SHA256 hash extended to desired dimension via repeated hashing.

        Args:
            text: Input string to embed (will be lowercased)
            embedding_dim: Desired embedding dimension

        Returns:
            numpy array of shape (embedding_dim,) with values in [-1, 1]
        """
        text = text.lower()
        # Generate enough hash bytes for the embedding
        hash_bytes = b''
        counter = 0
        while len(hash_bytes) < embedding_dim:
            hash_input = f"{text}_{counter}".encode('utf-8')
            hash_bytes += hashlib.sha256(hash_input).digest()
            counter += 1

        # Convert to float array in [-1, 1]
        byte_array = np.frombuffer(hash_bytes[:embedding_dim], dtype=np.uint8)
        embedding = (byte_array.astype(np.float32) / 127.5) - 1.0
        return embedding

    def encode_texts_hash(
        texts: list[str],
        device: torch.device,
        hash_bit_dim: int,
    ) -> torch.Tensor:
        """Encode texts using hash-bit embeddings.

        Args:
            texts: List of strings to encode (will be lowercased)
            device: Target device for output tensor
            hash_bit_dim: Dimension of hash embedding

        Returns:
            Hash embeddings tensor of shape (len(texts), hash_bit_dim)
        """
        if not texts:
            return torch.zeros((0, hash_bit_dim), device=device)

        embeddings = []
        for text in texts:
            emb = string_to_hash_embedding(text, hash_bit_dim)
            embeddings.append(torch.from_numpy(emb))
        return torch.stack(embeddings).to(device)

    # Set USE_HASH_ENCODING = True to use hash-based encoding instead of the learned model
    USE_HASH_ENCODING = True
    HASH_DIM = 256 # Embedding dimension when using hash encoding

    return HASH_DIM, USE_HASH_ENCODING, encode_texts_hash


@app.cell
def _(
    EMBEDDING_DIM,
    HASH_DIM,
    USE_HASH_ENCODING,
    encode_texts_hash,
    model,
    torch,
):
    # Optionally override model with hash-based encoder
    if USE_HASH_ENCODING:
        class HashEncoder:
            def __init__(self, dim):
                self.dim = dim
            def __call__(self, texts):
                emb = encode_texts_hash(texts, torch.device('cuda'), self.dim)
                return {"base_embedding": emb}

        encode_model = HashEncoder(HASH_DIM)
        encode_dim = HASH_DIM
        print(f"Using hash-based encoding with dim={HASH_DIM}")
    else:
        encode_model = model
        encode_dim = EMBEDDING_DIM
        print(f"Using learned model encoding with dim={EMBEDDING_DIM}")

    return encode_dim, encode_model


@app.cell
def _(mo):
    mo.md(r"""# OSM With Tags Export""")
    return


@app.cell
def _(Path, vd):
    _vd_config = vd.VigorDatasetConfig(
        panorama_tensor_cache_info=None,
        satellite_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version="v4_202001"

    )
    all_dataset = vd.VigorDataset(config=_vd_config, dataset_path=[
        Path("/data/overhead_matching/datasets/VIGOR/Seattle/"),
        Path("/data/overhead_matching/datasets/VIGOR/Chicago/"),
        Path("/data/overhead_matching/datasets/VIGOR/NewYork/"),
        Path("/data/overhead_matching/datasets/VIGOR/SanFrancisco/")
    ])
    return (all_dataset,)


@app.cell
def _(Path):
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    sentence_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/sentences/')
    sentence_dict, _ = slu.make_sentence_dict_from_json(slu.load_all_jsonl_from_folder(sentence_path))
    return sentence_dict, slu


@app.cell
def _(
    EMBEDDING_DIM,
    Path,
    all_dataset,
    itertools,
    model,
    new_dir_name_osm,
    pickle,
    sentence_dict,
    shutil,
    slu,
    torch,
    tqdm,
):

    new_output = {"version": "2.0"}
    with open("/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/embeddings/embeddings.pkl", 'rb') as _f:
        _old_embeddings = pickle.load(_f)
    _new_embeddings_tensor = torch.ones((_old_embeddings[0].shape[0], EMBEDDING_DIM)) * torch.nan
    _new_sentences = [""] * _old_embeddings[0].shape[0]
    for _batch in tqdm.tqdm(itertools.batched(_old_embeddings[1].items(), 128), total=len(_old_embeddings[1]) // 128):
        _custom_id_batch = [_x[0] for _x in _batch]
        _idx_batch = [_x[1] for _x in _batch]
        _sentences = [sentence_dict[_x] for _x in _custom_id_batch]
        with torch.no_grad():
            _model_outputs = model(_sentences)["base_embedding"].cpu()
        for _i in range(len(_sentences)):
            _new_embeddings_tensor[_idx_batch[_i]] = _model_outputs[_i]
            _new_sentences[_idx_batch[_i]] = _sentences[_i]
    assert not torch.any(torch.isnan(_new_embeddings_tensor))
    assert "" not in _new_sentences

    new_output["osm_sentences"] = (_new_embeddings_tensor, _old_embeddings[1], _new_sentences)

    _fields_to_skim = ["name", "amenity", "brand"]
    _street_things = ["addr:street", "addr:housenumber"]
    _df_copy = all_dataset._landmark_metadata[_fields_to_skim + _street_things + ["pruned_props"]].copy()
    _df_copy = _df_copy.rename(columns={
        "addr:street": "addr_street", 
        "addr:housenumber": "addr_housenumber"
    })
    _street_things = ["addr_street", "addr_housenumber"]

    for _field in _fields_to_skim:
        _out_idx_map = {}
        _out_embeddings=[]
        _out_sentences = []
        _idx = 0
        _subdf = _df_copy.dropna(subset=[_field])
        for _row in tqdm.tqdm(itertools.batched(_subdf.itertuples(), 128), total=len(_subdf)//128):
            _custom_ids = [slu.custom_id_from_props(getattr(_x, "pruned_props")) for _x in _row]
            _texts = [getattr(_x, _field) for _x in _row]
            with torch.no_grad():
                _model_output = model(_texts)["base_embedding"].cpu()
            for _i in range(len(_custom_ids)):
                if _custom_ids[_i] not in _out_idx_map:
                    _out_idx_map[_custom_ids[_i]] = _idx
                    _idx+= 1
                    _out_embeddings.append(_model_output[_i])
                    _out_sentences.append(_texts[_i])

        new_output[_field] = [torch.stack(_out_embeddings), _out_idx_map, _out_sentences]

    # do nouse number and street together
    _out_idx_map = {}
    _out_embeddings=[]
    _out_sentences = []
    _idx = 0
    _subdf = _df_copy.dropna(subset=_street_things)
    for _row in tqdm.tqdm(itertools.batched(_subdf.itertuples(), 128), total=len(_subdf)//128):
        _custom_ids = [slu.custom_id_from_props(getattr(_x, "pruned_props")) for _x in _row]
        _texts = [f"{getattr(_x, _street_things[1])} {getattr(_x, _street_things[0])}" for _x in _row]
        with torch.no_grad():
            _model_output = model(_texts)["base_embedding"].cpu()
        for _i in range(len(_custom_ids)):
            if _custom_ids[_i] not in _out_idx_map:
                _out_idx_map[_custom_ids[_i]] = _idx
                _idx+= 1
                _out_embeddings.append(_model_output[_i])
                _out_sentences.append(_texts[_i])

    new_output["street_address"] = [torch.stack(_out_embeddings), _out_idx_map, _out_sentences]



    _base_path = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/") / new_dir_name_osm
    (_base_path / "embeddings").mkdir(exist_ok=False, parents=True)
    shutil.copytree(src="/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/sentences/", dst=_base_path / "sentences")

    with open(f"/data/overhead_matching/datasets/semantic_landmark_embeddings/{new_dir_name_osm}/embeddings/embeddings.pkl",'wb') as _f:
        pickle.dump(new_output, _f)
    return


@app.cell
def _(mo):
    mo.md(r"""# Normal OSM Just Sentences Export""")
    return


@app.cell
def _(
    EMBEDDING_DIM,
    Path,
    model,
    new_dir_name_osm,
    pickle,
    sentence_dict,
    shutil,
    torch,
    tqdm,
):
    # Export OSM Embeddings

    with open("/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/embeddings/embeddings.pkl", 'rb') as _f:
        old_embeddings = pickle.load(_f)
    new_embeddings_tensor = torch.ones((old_embeddings[0].shape[0], EMBEDDING_DIM)) * torch.nan
    for _custom_id, _idx in tqdm.tqdm(old_embeddings[1].items()):
        with torch.no_grad():
            _model_output = model([sentence_dict[_custom_id]])["base_embedding"].cpu()
        new_embeddings_tensor[_idx] = _model_output
    assert not torch.any(torch.isnan(new_embeddings_tensor))

    _base_path = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/") / new_dir_name_osm
    (_base_path / "embeddings").mkdir(exist_ok=False, parents=True)
    shutil.copytree(src="/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/sentences/", dst=_base_path / "sentences")

    # with open(f"/data/overhead_matching/datasets/semantic_landmark_embeddings/{new_dir_name_osm}/embeddings/embeddings.pkl",'wb') as _f:
    #     pickle.dump((new_embeddings_tensor, old_embeddings[1]), _f)
    return (old_embeddings,)


@app.cell
def _(old_embeddings):
    print(old_embeddings)
    return


@app.cell
def _(mo):
    mo.md(r"""# Normal Pano Export""")
    return


@app.cell
def _(new_dir_pano_name):
    new_dir_pano_name
    return


@app.cell
def _(
    EMBEDDING_DIM,
    Path,
    itertools,
    json,
    model,
    model_base,
    new_dir_pano_name,
    pickle,
    shutil,
    slu,
    torch,
    tqdm,
):
    _old_pano_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_gemini_embeddings_from_ericks_model/")
    # _old_pano_base_dir = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/")
    _new_pano_base_dir = Path(f"/data/overhead_matching/datasets/semantic_landmark_embeddings/{new_dir_pano_name}/")

    for _city in _old_pano_base_dir.glob("*/"):
        # V2 pickle embedding
        # out = pickle.load((_city / "embeddings/embeddings.pkl").open("rb"))
        # break
        _v2_pickle_embedding = pickle.load((_city / "embeddings/embeddings.pkl").open("rb"))
        if "version" in _v2_pickle_embedding and _v2_pickle_embedding["version"] == "2.0":
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

            (_new_pano_base_dir / _city.name / "embeddings").mkdir(exist_ok=False, parents=True)
            shutil.copytree(src=_old_pano_base_dir / _city.name / "panorama_sentence_requests", dst=_new_pano_base_dir / _city.name / "panorama_sentence_requests")
            shutil.copytree(src=_old_pano_base_dir / _city.name / "sentences", dst=_new_pano_base_dir / _city.name / "sentences")

            pickle.dump(_output_v2_embedding, (_new_pano_base_dir / _city.name / "embeddings/embeddings.pkl").open("wb"))

        else:
            # V1 pickle embedding
            print("Using v1 pickle embeddings!")
            _city_embeddings_tensor, _city_embeddings_mapping = pickle.load((_city / "embeddings/embeddings.pkl").open("rb"))
            _new_embedding_tensor = torch.ones((_city_embeddings_tensor.shape[0], EMBEDDING_DIM)) * torch.nan
            _city_sentences, _ = slu.make_sentence_dict_from_json(slu.load_all_jsonl_from_folder(_city / "sentences/"))
            for _panoidverbose, _props_str in tqdm.tqdm(_city_sentences.items()):
                _landmarks = json.loads(_props_str)["landmarks"]
                _sentences = [_x["description"] for _x in _landmarks]
                with torch.no_grad():
                    _new_embeddings = model(_sentences)["base_embedding"].cpu()
                # print(_sentences)
                for _i, _emb in enumerate(_new_embeddings):
                    _key = f"{_panoidverbose}__landmark_{_i}"
                    if _key not in _city_embeddings_mapping:
                        print(f"Failed to find key {_key}")
                        continue
                    _loc = _city_embeddings_mapping[_key]
                    _new_embedding_tensor[_loc, :] = _emb
                # print(_props_str)

            assert not torch.any(torch.isnan(_new_embedding_tensor))
            pickle.dump((_new_embedding_tensor, _city_embeddings_mapping), (_new_pano_base_dir / _city.name / "embeddings/embeddings.pkl").open("wb"))
    return


@app.cell
def _(mo):
    mo.md(r"""# Spoofed dataset pano and normal OSM""")
    return


@app.cell
def _(EMBEDDING_DIM, Path, itertools, json, model, model_base, pickle, torch):
    # Process spoofed_place_location embeddings
    _spoofed_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location")
    _output_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location_embeddings_train_with_osm_sentences_v3")

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
    with open(_spoofed_base / "osm_spoofed/sentences/sentences.jsonl") as _f:
        for line in _f:
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
def _(mo):
    mo.md(r"""# Spoofed dataset with v2 OSM landmarks""")
    return


@app.cell
def _(
    Path,
    copy,
    encode_dim,
    encode_model,
    itertools,
    json,
    pickle,
    torch,
    tqdm,
):
    # Process spoofed_place_location embeddings - V2 OUTPUT with proper_noun format
    _spoofed_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location")
    _output_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location_v2_train_with_osm_sentences_v3_hashbit_256")

    def _parse_pano_description(description: str) -> str:
        """Parse proper noun (name or address) from pano description.

        Pano descriptions have one of two formats:
        - Name: "...with a sign for {name}"
        - Address: "...with the address {address}"

        Returns:
            The extracted proper noun string

        Raises:
            ValueError: If description doesn't contain either expected pattern
        """
        if "with a sign for " in description:
            return description.split("with a sign for ", 1)[1]
        if "with the address " in description:
            return description.split("with the address ", 1)[1]
        raise ValueError(f"Expected 'with a sign for' or 'with the address' in description: {description}")

    # --- Process pano_spoofed (Version 2.0 format with proper_noun) ---
    for _city_dir in (_spoofed_base / "pano_spoofed").glob("*/"):
        _v2_pickle = pickle.load((_city_dir / "embeddings/embeddings.pkl").open("rb"))

        # Build updated panoramas dict with proper_nouns populated
        _updated_panoramas = {}
        _all_proper_nouns = set()  # Collect unique proper nouns

        for _pid, _pano_data in _v2_pickle["panoramas"].items():
            _updated_pano = copy.deepcopy(_pano_data)
            for _landmark in _updated_pano["landmarks"]:
                _proper_noun = _parse_pano_description(_landmark["description"])
                _landmark["proper_nouns"] = [_proper_noun]
                _all_proper_nouns.add(_proper_noun)
            _updated_panoramas[_pid] = _updated_pano

        # Compute description embeddings (full sentences)
        _description_id_to_idx = _v2_pickle["description_id_to_idx"]
        _num_descriptions = len(_description_id_to_idx)
        _description_embeddings = torch.ones((_num_descriptions, encode_dim)) * torch.nan

        for _batch in tqdm.tqdm(
            itertools.batched(_description_id_to_idx.items(), 50),
            total=_num_descriptions // 50 + 1,
            desc=f"pano_spoofed/{_city_dir.name}/descriptions"
        ):
            _keys = [x[0] for x in _batch]
            _indices = [x[1] for x in _batch]

            def _get_description(_key):
                _kpid = "__".join(_key.split("__")[:-1])
                _klandmark_idx = int(_key.split("__")[-1].split("_")[1])
                return _v2_pickle["panoramas"][_kpid]["landmarks"][_klandmark_idx]["description"]

            _descriptions = [_get_description(k) for k in _keys]
            with torch.no_grad():
                _emb = encode_model(_descriptions)["base_embedding"].cpu()
            for _i, _idx in enumerate(_indices):
                _description_embeddings[_idx] = _emb[_i]

        assert not torch.any(torch.isnan(_description_embeddings))

        # Build proper_noun_to_idx and compute embeddings
        _proper_noun_list = sorted(_all_proper_nouns)
        _proper_noun_to_idx = {pn: i for i, pn in enumerate(_proper_noun_list)}
        _proper_noun_embeddings = []

        for _batch in tqdm.tqdm(
            itertools.batched(_proper_noun_list, 50),
            total=len(_proper_noun_list) // 50 + 1,
            desc=f"pano_spoofed/{_city_dir.name}/proper_nouns"
        ):
            _texts = list(_batch)
            with torch.no_grad():
                _emb = encode_model(_texts)["base_embedding"].cpu()
            _proper_noun_embeddings.append(_emb)

        _proper_noun_tensor = torch.cat(_proper_noun_embeddings, dim=0) if _proper_noun_embeddings else torch.zeros((0, encode_dim))

        # Build v2 output dict
        _pano_v2_output = {
            "version": "2.0",
            "panoramas": _updated_panoramas,
            "description_embeddings": _description_embeddings,
            "description_id_to_idx": _description_id_to_idx,
            "proper_noun_embeddings": _proper_noun_tensor,
            "proper_noun_to_idx": _proper_noun_to_idx,
        }

        _out_dir = _output_base / "pano_spoofed" / _city_dir.name / "embeddings"
        _out_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(_pano_v2_output, (_out_dir / "embeddings.pkl").open("wb"))
        print(f"Processed pano_spoofed/{_city_dir.name}:")
        print(f"  panoramas: {len(_updated_panoramas)}")
        print(f"  description_embeddings: {_description_embeddings.shape}")
        print(f"  proper_noun_embeddings: {_proper_noun_tensor.shape}")

    # --- Process osm_spoofed (Version 2.0 format with proper_noun) ---
    _osm_emb, _osm_id_to_idx = pickle.load((_spoofed_base / "osm_spoofed/embeddings/embeddings.pkl").open("rb"))

    # Load and parse sentences
    _osm_sentences = {}
    _osm_names = {}
    _osm_addresses = {}
    with open(_spoofed_base / "osm_spoofed/sentences/sentences.jsonl") as f:
        for line in f:
            _entry = json.loads(line)
            _custom_id = _entry["custom_id"]
            _sentence = _entry["response"]["body"]["choices"][0]["message"]["content"]
            _osm_sentences[_custom_id] = _sentence
            # Parse name and address from "Name location on Address" format
            if " location on " in _sentence:
                _parts = _sentence.split(" location on ", 1)
                _osm_names[_custom_id] = _parts[0]
                _osm_addresses[_custom_id] = _parts[1]

    # Build v2 output dict
    _v2_output = {"version": "2.0"}

    # Compute osm_sentences embeddings (full sentences)
    _new_osm_tensor = torch.ones((_osm_emb.shape[0], encode_dim)) * torch.nan
    _new_osm_sentence_list = [""] * _osm_emb.shape[0]
    for _key_batch in tqdm.tqdm(itertools.batched(_osm_id_to_idx.items(), 50), total=len(_osm_id_to_idx) // 50, desc="osm_spoofed/sentences"):
        _keys = [x[0] for x in _key_batch]
        _indices = [x[1] for x in _key_batch]
        _sentences = [_osm_sentences[k] for k in _keys]
        with torch.no_grad():
            _new_emb = encode_model(_sentences)["base_embedding"].cpu()
        for _i, _idx in enumerate(_indices):
            _new_osm_tensor[_idx] = _new_emb[_i]
            _new_osm_sentence_list[_idx] = _sentences[_i]

    assert not torch.any(torch.isnan(_new_osm_tensor))
    assert "" not in _new_osm_sentence_list
    _v2_output["osm_sentences"] = (_new_osm_tensor, _osm_id_to_idx, _new_osm_sentence_list)

    # Compute thing embeddings (extracted names)
    _thing_embeddings = []
    _thing_id_to_idx = {}
    _thing_sentences = []
    _idx = 0
    for _batch in tqdm.tqdm(itertools.batched(_osm_names.items(), 50), total=len(_osm_names) // 50 + 1, desc="osm_spoofed/thing"):
        _custom_ids = [x[0] for x in _batch]
        _texts = [x[1] for x in _batch]
        with torch.no_grad():
            _emb = encode_model(_texts)["base_embedding"].cpu()
        for _i, _custom_id in enumerate(_custom_ids):
            if _custom_id not in _thing_id_to_idx:
                _thing_id_to_idx[_custom_id] = _idx
                _idx += 1
                _thing_embeddings.append(_emb[_i])
                _thing_sentences.append(_texts[_i])

    # Compute place embeddings (extracted addresses)
    _place_embeddings = []
    _place_id_to_idx = {}
    _place_sentences = []
    _idx = 0
    for _batch in tqdm.tqdm(itertools.batched(_osm_addresses.items(), 50), total=len(_osm_addresses) // 50 + 1, desc="osm_spoofed/place"):
        _custom_ids = [x[0] for x in _batch]
        _texts = [x[1] for x in _batch]
        with torch.no_grad():
            _emb = encode_model(_texts)["base_embedding"].cpu()
        for _i, _custom_id in enumerate(_custom_ids):
            if _custom_id not in _place_id_to_idx:
                _place_id_to_idx[_custom_id] = _idx
                _idx += 1
                _place_embeddings.append(_emb[_i])
                _place_sentences.append(_texts[_i])

    _v2_output["thing"] = [torch.stack(_thing_embeddings), _thing_id_to_idx, _thing_sentences]
    _v2_output["place"] = [torch.stack(_place_embeddings), _place_id_to_idx, _place_sentences]

    _out_dir = _output_base / "osm_spoofed/embeddings"
    _out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(_v2_output, (_out_dir / "embeddings.pkl").open("wb"))
    print(f"Processed osm_spoofed v2:")
    print(f"  osm_sentences: {_new_osm_tensor.shape}")
    print(f"  thing: {torch.stack(_thing_embeddings).shape}")
    print(f"  place: {torch.stack(_place_embeddings).shape}")

    # Export for verification cell
    osm_names_set = set(_osm_names.values())
    osm_addresses_set = set(_osm_addresses.values())

    return osm_addresses_set, osm_names_set


@app.cell
def _(Path, copy, itertools, json, model, pickle, torch, tqdm):
    # Process spoofed embeddings with contrastive head projections
    _spoofed_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location")
    _output_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location_v2_train_with_osm_sentences_v3_contrastive")

    _proj_dim = model.contrastive_heads["name"][0].out_features
    _base_dim = model.base_dim

    # --- Process pano_spoofed ---
    for _city_dir in (_spoofed_base / "pano_spoofed").glob("*/"):
        _v2_pickle = pickle.load((_city_dir / "embeddings/embeddings.pkl").open("rb"))

        # Build updated panoramas with proper_nouns
        _updated_panoramas = {}
        for _pid, _pano_data in _v2_pickle["panoramas"].items():
            _updated_pano = copy.deepcopy(_pano_data)
            for _landmark in _updated_pano["landmarks"]:
                _desc = _landmark["description"]
                if "with a sign for " in _desc:
                    _landmark["proper_nouns"] = [_desc.split("with a sign for ", 1)[1]]
                elif "with the address " in _desc:
                    _landmark["proper_nouns"] = [_desc.split("with the address ", 1)[1]]
                else:
                    raise ValueError(f"Unknown format: {_desc}")
            _updated_panoramas[_pid] = _updated_pano

        # Compute description embeddings with appropriate contrastive projection
        _description_id_to_idx = _v2_pickle["description_id_to_idx"]
        _num_descriptions = len(_description_id_to_idx)
        _description_embeddings = torch.zeros((_num_descriptions, _proj_dim))

        for _batch in tqdm.tqdm(
            itertools.batched(_description_id_to_idx.items(), 50),
            total=_num_descriptions // 50 + 1,
            desc=f"pano/{_city_dir.name}"
        ):
            _thing_indices, _thing_descs = [], []
            _place_indices, _place_descs = [], []

            for _key, _idx in _batch:
                _pid = "__".join(_key.split("__")[:-1])
                _landmark_idx = int(_key.split("__")[-1].split("_")[1])
                _desc = _v2_pickle["panoramas"][_pid]["landmarks"][_landmark_idx]["description"]

                if "with a sign for " in _desc:
                    _thing_indices.append(_idx)
                    _thing_descs.append(_desc)
                else:
                    _place_indices.append(_idx)
                    _place_descs.append(_desc)

            if _thing_descs:
                with torch.no_grad():
                    _emb = model(_thing_descs)["contrastive_embeddings"]["name"].cpu()
                for _i, _idx in enumerate(_thing_indices):
                    _description_embeddings[_idx] = _emb[_i]

            if _place_descs:
                with torch.no_grad():
                    _emb = model(_place_descs)["contrastive_embeddings"]["addr:street"].cpu()
                for _i, _idx in enumerate(_place_indices):
                    _description_embeddings[_idx] = _emb[_i]

        _pano_output = {
            "version": "2.0",
            "panoramas": _updated_panoramas,
            "description_embeddings": _description_embeddings,
            "description_id_to_idx": _description_id_to_idx,
        }

        _out_dir = _output_base / "pano_spoofed" / _city_dir.name / "embeddings"
        _out_dir.mkdir(parents=True, exist_ok=True)
        pickle.dump(_pano_output, (_out_dir / "embeddings.pkl").open("wb"))
        print(f"pano/{_city_dir.name}: panoramas={len(_updated_panoramas)}, embeddings={_description_embeddings.shape}")

    # --- Process osm_spoofed ---
    _osm_emb, _osm_id_to_idx = pickle.load((_spoofed_base / "osm_spoofed/embeddings/embeddings.pkl").open("rb"))

    _osm_sentences = {}
    with open(_spoofed_base / "osm_spoofed/sentences/sentences.jsonl") as _f:
        for _line in _f:
            _entry = json.loads(_line)
            _osm_sentences[_entry["custom_id"]] = _entry["response"]["body"]["choices"][0]["message"]["content"]

    _num_sentences = len(_osm_id_to_idx)
    _base_embeddings = torch.zeros((_num_sentences, _base_dim))
    _thing_embeddings = torch.zeros((_num_sentences, _proj_dim))
    _place_embeddings = torch.zeros((_num_sentences, _proj_dim))
    _sentence_list = [""] * _num_sentences

    for _batch in tqdm.tqdm(itertools.batched(_osm_id_to_idx.items(), 50), total=_num_sentences // 50 + 1, desc="osm"):
        _keys = [x[0] for x in _batch]
        _indices = [x[1] for x in _batch]
        _sentences = [_osm_sentences[k] for k in _keys]
        with torch.no_grad():
            _output = model(_sentences)
            _base_emb = _output["base_embedding"].cpu()
            _thing_emb = _output["contrastive_embeddings"]["name"].cpu()
            _place_emb = _output["contrastive_embeddings"]["addr:street"].cpu()
        for _i, _idx in enumerate(_indices):
            _base_embeddings[_idx] = _base_emb[_i]
            _thing_embeddings[_idx] = _thing_emb[_i]
            _place_embeddings[_idx] = _place_emb[_i]
            _sentence_list[_idx] = _sentences[_i]

    assert "" not in _sentence_list

    _osm_output = {
        "version": "2.0",
        "osm_sentences": (_base_embeddings, _osm_id_to_idx, _sentence_list),
        "thing": [_thing_embeddings, _osm_id_to_idx, _sentence_list],
        "place": [_place_embeddings, _osm_id_to_idx, _sentence_list],
    }

    _out_dir = _output_base / "osm_spoofed/embeddings"
    _out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(_osm_output, (_out_dir / "embeddings.pkl").open("wb"))
    print(f"osm: base={_base_embeddings.shape}, thing={_thing_embeddings.shape}, place={_place_embeddings.shape}")

    return


@app.cell
def _(Path, osm_addresses_set, osm_names_set, pickle):
    # Verification: Check that each pano text (name and address) shows up in OSM extracted sentences
    _spoofed_base = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/spoofed_place_location")

    def _parse_pano_description(description: str) -> tuple[str | None, str | None]:
        """Parse name or address from pano description."""
        if "with a sign for " in description:
            return description.split("with a sign for ", 1)[1], None
        if "with the address " in description:
            return None, description.split("with the address ", 1)[1]
        raise ValueError(f"Expected 'with a sign for' or 'with the address' in description: {description}")

    _missing_names = []
    _missing_addresses = []
    _total_names = 0
    _total_addresses = 0

    for _city_dir in (_spoofed_base / "pano_spoofed").glob("*/"):
        _v2_pickle = pickle.load((_city_dir / "embeddings/embeddings.pkl").open("rb"))

        for _key, _idx in _v2_pickle["description_id_to_idx"].items():
            _pid = "__".join(_key.split("__")[:-1])
            _landmark_idx = int(_key.split("__")[-1].split("_")[1])
            _description = _v2_pickle["panoramas"][_pid]["landmarks"][_landmark_idx]["description"]
            _name, _address = _parse_pano_description(_description)

            if _name:
                _total_names += 1
                if _name not in osm_names_set:
                    _missing_names.append((_name, _city_dir.name, _key))
            if _address:
                _total_addresses += 1
                if _address not in osm_addresses_set:
                    _missing_addresses.append((_address, _city_dir.name, _key))

    print(f"=== Verification Results ===")
    print(f"Total pano names: {_total_names}")
    print(f"Total pano addresses: {_total_addresses}")
    print(f"Missing names (not in OSM): {len(_missing_names)}")
    print(f"Missing addresses (not in OSM): {len(_missing_addresses)}")

    if _missing_names:
        print(f"\nFirst 10 missing names:")
        for name, city, key in _missing_names[:10]:
            print(f"  [{city}] {name}")

    if _missing_addresses:
        print(f"\nFirst 10 missing addresses:")
        for addr, city, key in _missing_addresses[:10]:
            print(f"  [{city}] {addr}")

    if not _missing_names and not _missing_addresses:
        print("\n✓ All pano names and addresses found in OSM data!")

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
def _(vd):

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
