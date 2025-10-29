import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import openai

    from pathlib import Path

    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    return Path, openai, slu, vd


@app.cell
def _():
    import numpy as np
    import pickle
    return np, pickle


@app.cell
def _(openai):
    client = openai.OpenAI()
    return (client,)


@app.cell
def _(Path, vd):
    _dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago/')
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version='v3'
    )

    dataset = vd.VigorDataset(dataset_path=_dataset_path,
                             config=_config)
    return (dataset,)


@app.cell
def _(Path, slu):
    _sentences_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/Chicago/sentences')
    _sentence_json = slu.load_all_jsonl_from_folder(_sentences_path)
    sentence_from_pano_landmark, metadata_from_pano_id, _ = slu.make_sentence_dict_from_pano_jsons(_sentence_json)
    return metadata_from_pano_id, sentence_from_pano_landmark


@app.cell
def _(Path, file_in, pickle):
    _pano_embedding_path = Path('/data/overhead_matching//datasets/semantic_landmark_embeddings/pano_v1/Chicago/embeddings/embeddings.pkl')
    with _pano_embedding_path.open('rb') as _file_in:
        pano_embeddings, pano_lm_idx_from_lm_id = pickle.load(file_in)
    return


@app.cell
def _(metadata_from_pano_id, sentence_from_pano_landmark):
    landmark_sentences_from_pano_id = {
        k.split(',')[0]:[
            {"lm_id": info["custom_id"], "sentence": sentence_from_pano_landmark[info["custom_id"]]} 
            for info in v
        ]
        for k, v in metadata_from_pano_id.items()
    }
    return (landmark_sentences_from_pano_id,)


@app.cell
def _(Path, osm_embeddings):
    _osm_embedding_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/v3_no_addresses/embeddings/embeddings.pkl')
    with _osm_embedding_path.open('rb') as _file_in:
        osm_embeddings, 
    return


@app.cell
def _(dataset):

    _pano_id = '--z0RFQbsumsJC2wWUUKIg'
    def osm_landmarks_from_pano_id(pano_id):
        pano_info = dataset._panorama_metadata[dataset._panorama_metadata.pano_id == pano_id].iloc[0]
        sat_idxs = pano_info.positive_satellite_idxs + pano_info.semipositive_satellite_idxs
        osm_landmarks = set()
        for sat_idx in sat_idxs:
            osm_landmarks |= set(dataset._satellite_metadata.iloc[sat_idx]["landmark_idxs"])
        all_landmarks = dataset._landmark_metadata.iloc[list(osm_landmarks)][["pruned_props", "id"]]
        out = set(all_landmarks["pruned_props"].values)
        return [list(x) for x in out], all_landmarks

    osm_landmarks_from_pano_id(_pano_id)
    return (osm_landmarks_from_pano_id,)


@app.cell
def _(landmark_sentences_from_pano_id):
    _pano_id = '--z0RFQbsumsJC2wWUUKIg'
    def pano_sentences_from_pano_id(pano_id):
        return landmark_sentences_from_pano_id[pano_id]

    pano_sentences_from_pano_id(_pano_id)
    return (pano_sentences_from_pano_id,)


@app.cell
def _(osm_landmarks_from_pano_id, pano_sentences_from_pano_id):
    _pano_id = '-0KSAsfK2L8z_pO6SKbhmQ'

    def itemized_list(items):
        out = []
        for i, v in enumerate(items):
            out.append(f" {i}. {v}")
        return '\n'.join(out)

    SCHEMA = {
        "name": "landmark_correspondence",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["matches"],
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["set_1_id", "set_2_matches"],
                        "additionalProperties": False,
                        "properties": {
                            "set_1_id": {"type": "integer"},
                            "set_2_matches": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        }
    }

    DEVELOPER_PROMPT = '''You are a landmark matching wizard. I would like you to propose correspondences between the landmarks in set 1 and the landmarks in set 2.
    The landmarks in set 2 are derived from key value pairs from openstreetmap entities. For each item in set 1, provide a list of items in set 2 that correspond to it.
    Note that some landmarks in Set 2 may not have a corresponding landmark in set 1 and vice versa.
    '''

    def create_prompt(pano_id):
        return f"""Set 1:
    { itemized_list([x["sentence"] for x in pano_sentences_from_pano_id(pano_id)])}

    Set 2:
    { itemized_list( '; '.join([f"{a}={b}" for a, b in x]) for x in osm_landmarks_from_pano_id(pano_id)[0])}
    """

    print(create_prompt(_pano_id))
    return DEVELOPER_PROMPT, SCHEMA, create_prompt


@app.cell
def _(DEVELOPER_PROMPT, SCHEMA, create_prompt):
    _pano_id = '-0KSAsfK2L8z_pO6SKbhmQ'


    def create_request(pano_id, reasoning_effort="medium"):
        return dict(
            custom_id=pano_id,
            method="POST",
            url="/v1/chat/completions",
            body=dict(
                model="gpt-5",
                reasoning_effort=reasoning_effort,
                messages=[
                    dict(role="developer", content=DEVELOPER_PROMPT),
                    dict(role="user", content=create_prompt(pano_id))
                ],
                response_format=dict(type="json_schema", json_schema=SCHEMA)
            )
        )


    return (create_request,)


@app.cell
def _(create_request, landmark_sentences_from_pano_id, np):
    _rng = np.random.default_rng(1024)
    _selected_panoramas = _rng.choice(list(landmark_sentences_from_pano_id.keys()), size=(1000), replace=False)
    # _selected_panoramas = list(landmark_sentences_from_pano_id.keys())
    requests = [create_request(_p, reasoning_effort="minimal") for _p in _selected_panoramas]
    return (requests,)


@app.cell
def _(requests):
    requests[0]
    return


@app.cell
def _():
    import json
    return (json,)


@app.cell(disabled=True)
def _(Path, client, json, requests):
    batch_path = Path("/tmp/batch_request.jsonl")
    with batch_path.open('w') as file_out:
        file_out.writelines([json.dumps(_x) + "\n" for _x in requests])

    batch_file_response = client.files.create(
        file=batch_path.open('rb'),
        purpose="batch",
    )
    return (batch_file_response,)


@app.cell(disabled=True)
def _(batch_file_response, client):
    client.batches.create(
        input_file_id=batch_file_response.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"reasoning_effort": "minimal",
                 "seed":"1024"}
    )
    return


@app.cell
def _(Path, json, osm_landmarks_from_pano_id, slu):
    results = {
        'minimal': Path('/data/overhead_matching/datasets/landmark_correspondence/v1_minimal/'),
        'minimal_full': Path('/data/overhead_matching/datasets/landmark_correspondence/v4_minimal_full/'),
        'low': Path('/data/overhead_matching/datasets/landmark_correspondence/v3_low/'),
        'medium': Path('/data/overhead_matching/datasets/landmark_correspondence/v2_medium/'),
    }

    def parse_sets(text):
        """
        Parse a string containing two numbered sets and return them as lists.

        Args:
            text: String containing "Set 1:" and "Set 2:" sections with numbered items

        Returns:
            tuple: (set1_list, set2_list) where each is a list of strings
        """
        # Split on "Set 2:" to separate the two sections
        parts = text.split("Set 2:")

        if len(parts) != 2:
            raise ValueError("Expected exactly two sets in the input")

        set1_text = parts[0]
        set2_text = parts[1]

        # Helper function to extract numbered items
        def extract_items(section_text):
            items = []
            lines = section_text.strip().split('\n')

            for line in lines:
                line = line.strip()
                # Skip empty lines and section headers
                if not line or line.startswith('Set '):
                    continue

                # Remove the number prefix (e.g., "1. ", "10. ", etc.)
                # Find the first period followed by a space
                period_index = line.find('. ')
                if period_index > 0:
                    # Check if everything before the period is a number
                    prefix = line[:period_index].strip()
                    if prefix.isdigit():
                        item = line[period_index + 2:].strip()
                        items.append(item)

            return items

        set1 = extract_items(set1_text)
        set2 = extract_items(set2_text)

        return set1, set2

    def load_matches_from_folder(path):
        # Load requests
        requests_json = slu.load_all_jsonl_from_folder(path / "requests")
        landmarks = {}    
        for req in requests_json:
            lm_msg = req["body"]["messages"][-1]["content"]
            pano_lms, osm_lms = parse_sets(lm_msg)
            _, full_osm_landmarks = osm_landmarks_from_pano_id(req["custom_id"])
            pruned_lms = [
                frozenset([tuple(tag.split('=')) for tag in x.split('; ')])
                for x in osm_lms
            ]

            def parse_id(x):
                return json.loads(x.replace('(', '[')
                                 .replace(')', ']')
                                 .replace("'", '"'))
        
            landmarks[req["custom_id"]] = {
                "pano": pano_lms,
                "osm": [{
                    "tags": pruned,
                    "ids": full_osm_landmarks[full_osm_landmarks["pruned_props"] == fs].id.apply(parse_id).values.tolist()
                } for pruned, fs in zip(osm_lms, pruned_lms)],
            }

        # Load responses
        response_json = slu.load_all_jsonl_from_folder(path / "responses")
        return {
            r["custom_id"]: dict(
                pano = landmarks[r["custom_id"]]["pano"],
                osm=landmarks[r["custom_id"]]["osm"],
                matches=json.loads(r["response"]["body"]["choices"][0]["message"]["content"]),
                usage=r["response"]["body"]["usage"]
            )
            for r in response_json
        }
    m = load_matches_from_folder(Path("/data/overhead_matching/datasets/landmark_correspondence/v1_minimal/"))
    return load_matches_from_folder, m, results


@app.cell
def _(Path, json, load_matches_from_folder, results):
    for k, v in results.items():
        _matches = load_matches_from_folder(v)
        Path(f'/tmp/{k}.json').write_text(json.dumps(_matches))


    return


@app.cell
def _(m):


    m['U1d0wO63hQhrhwAZux3dKg']["osm"]
    return


@app.cell
def _(dataset):
    dataset._landmark_metadata.id.apply(lambda x: x.replace("(", "[").replace(")", "]").replace("'", '"'))
    return


@app.cell
def _(osm_landmarks_from_pano_id, pano_sentences_from_pano_id):
    def produce_matches_for_pano_id(pano_id):
        # Collect the landmark sentences
        pano_sentences = pano_sentences_from_pano_id(pano_id)
        # Collect the osm landmarks
        _, osm_landmarks = osm_landmarks_from_pano_id(pano_id)
        # Collect the matches
        # Collect the embeddings
        print(pano_id)
        print(pano_sentences)
        print(osm_landmarks)
        ...

    _pano_id = "TQi22hzwJANv-cJDd79Jng"
    produce_matches_for_pano_id(_pano_id)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
