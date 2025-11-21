import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import openai

    from pathlib import Path

    import common.torch.load_torch_deps
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    return Path, mo, openai, slu, vd


@app.cell
def _():
    import numpy as np
    import pickle
    import z3
    from collections import defaultdict
    return np, pickle, z3


@app.cell
def _():
    import experimental.overhead_matching.swag.model.semantic_landmark_extractor as sle
    return


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

    new_dataset = vd.VigorDataset(dataset_path=_dataset_path,
                             config=_config)

    _dataset_path = Path('/data/overhead_matching/datasets/VIGOR/Chicago/')
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version='v4_202001'
    )

    historical_dataset = vd.VigorDataset(dataset_path=_dataset_path,
                             config=_config)
    return historical_dataset, new_dataset


@app.cell
def _(Path, slu):
    _sentences_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/Chicago/sentences')
    _sentence_json = slu.load_all_jsonl_from_folder(_sentences_path)
    sentence_from_pano_landmark, _metadata_from_pano_id, _ = slu.make_sentence_dict_from_pano_jsons(_sentence_json)
    metadata_from_pano_id = {k.split(',')[0]: v for k, v in _metadata_from_pano_id.items()}
    return metadata_from_pano_id, sentence_from_pano_landmark


@app.cell
def _(Path, pickle):
    _pano_embedding_path = Path('/data/overhead_matching//datasets/semantic_landmark_embeddings/pano_v1/Chicago/embeddings/embeddings.pkl')
    with _pano_embedding_path.open('rb') as _file_in:
        pano_embeddings_from_lm_idx, pano_lm_idx_from_lm_id = pickle.load(_file_in)
    return pano_embeddings_from_lm_idx, pano_lm_idx_from_lm_id


@app.cell
def _(metadata_from_pano_id, sentence_from_pano_landmark):
    landmark_sentences_from_pano_id = {
        k.split(',')[0]:[
            {"lm_id": info["custom_id"], "sentence": sentence_from_pano_landmark[info["custom_id"]], "yaw_angles": info["yaw_angles"]}
            for info in v
        ]
        for k, v in metadata_from_pano_id.items()
    }
    return (landmark_sentences_from_pano_id,)


@app.cell
def _(Path, pickle):
    _osm_embedding_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/v3_no_addresses/embeddings/embeddings.pkl')
    with _osm_embedding_path.open('rb') as _file_in:
        recent_osm_embeddings, recent_osm_lm_idx_from_osm_id = pickle.load(_file_in)
    return


@app.cell
def _(Path, pickle):
    _osm_embedding_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/embeddings/embeddings.pkl')
    with _osm_embedding_path.open('rb') as _file_in:
        historical_osm_embeddings, historical_osm_lm_idx_from_osm_id = pickle.load(_file_in)
    return historical_osm_embeddings, historical_osm_lm_idx_from_osm_id


@app.cell
def _(historical_osm_lm_idx_from_osm_id):
    historical_osm_lm_idx_from_osm_id["MVdkTIc8bXC2EC/3YA8JuH7kaTXcmHc/MlSQ1qiocyg="]
    return


@app.cell
def _(historical_dataset):




    _pano_id = '--z0RFQbsumsJC2wWUUKIg'
    def osm_landmarks_from_pano_id(pano_id, dataset):
        pano_info = dataset._panorama_metadata[dataset._panorama_metadata.pano_id == pano_id].iloc[0]
        sat_idxs = pano_info.positive_satellite_idxs + pano_info.semipositive_satellite_idxs
        osm_landmarks = set()
        for sat_idx in sat_idxs:
            osm_landmarks |= set(dataset._satellite_metadata.iloc[sat_idx]["landmark_idxs"])
        all_landmarks = dataset._landmark_metadata.iloc[list(osm_landmarks)][["pruned_props", "id", "geometry", "geometry_px"]]
        out = set(all_landmarks["pruned_props"].values)
        return [list(x) for x in out], all_landmarks

    osm_landmarks_from_pano_id(_pano_id, historical_dataset)
    return (osm_landmarks_from_pano_id,)


@app.cell
def _(landmark_sentences_from_pano_id):
    _pano_id = '--z0RFQbsumsJC2wWUUKIg'
    def pano_sentences_from_pano_id(pano_id):
        return landmark_sentences_from_pano_id[pano_id]

    pano_sentences_from_pano_id(_pano_id)
    return (pano_sentences_from_pano_id,)


@app.cell
def _(
    historical_dataset,
    osm_landmarks_from_pano_id,
    pano_sentences_from_pano_id,
):
    _pano_id = 'EZd0yEWwrNSoGQhljhV1Uw'

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
    { itemized_list( '; '.join([f"{a}={b}" for a, b in x]) for x in osm_landmarks_from_pano_id(pano_id, historical_dataset)[0])}
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
    requests = [create_request(_p, reasoning_effort="medium") for _p in _selected_panoramas]
    return (requests,)


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
        metadata={"reasoning_effort": "medium",
                 "seed":"1024"}
    )
    return


@app.cell
def _(
    Path,
    historical_dataset,
    json,
    landmark_sentences_from_pano_id,
    new_dataset,
    osm_landmarks_from_pano_id,
    slu,
):
    results_and_dataset = {
        'minimal': (Path('/data/overhead_matching/datasets/landmark_correspondence/v1_minimal/'), new_dataset),
        # 'minimal_full': Path('/data/overhead_matching/datasets/landmark_correspondence/v4_minimal_full/'),
        'low': (Path('/data/overhead_matching/datasets/landmark_correspondence/v3_low/'), new_dataset),
        'medium': (Path('/data/overhead_matching/datasets/landmark_correspondence/v2_medium/'), new_dataset),
        'medium_historical': (Path('/data/overhead_matching/datasets/landmark_correspondence/v5_medium_historical/'), historical_dataset),
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
                        items.append((int(prefix), item))

            return items

        set1 = extract_items(set1_text)
        set2 = extract_items(set2_text)

        return set1, set2

    def load_matches_from_folder(path, dataset):
        # Load requests
        requests_json = slu.load_all_jsonl_from_folder(path / "requests")
        landmarks = {}    
        for req in requests_json:
            lm_msg = req["body"]["messages"][-1]["content"]
            pano_lms, osm_lms = parse_sets(lm_msg)
            pano_lms_from_req = landmark_sentences_from_pano_id[req["custom_id"]]
            _, full_osm_landmarks = osm_landmarks_from_pano_id(req["custom_id"], dataset)

            pruned_lms = [
                (idx, frozenset([tuple(tag.split('=')) for tag in tags.split('; ')]))
                for idx, tags in osm_lms
            ]

            for from_prompt, from_req in zip(pano_lms, pano_lms_from_req):
                from_req["index"] = from_prompt[0]
        
            def parse_id(x):
                return json.loads(x.replace('(', '[')
                                 .replace(')', ']')
                                 .replace("'", '"'))

            landmarks[req["custom_id"]] = {
                "pano": pano_lms_from_req,
                "osm": [{
                    "tags": pruned,
                    "ids": full_osm_landmarks[full_osm_landmarks["pruned_props"] == fs].id.apply(parse_id).values.tolist(),
                    "index": idx,
                } for (_, pruned), (idx, fs) in zip(osm_lms, pruned_lms)],
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
    m = load_matches_from_folder(Path("/data/overhead_matching/datasets/landmark_correspondence/v1_minimal/"), new_dataset)
    return load_matches_from_folder, m, results_and_dataset


@app.cell
def _(Path, json, load_matches_from_folder, results_and_dataset):
    matches = {}
    for k, v in results_and_dataset.items():
        print('Loading', k)
        matches[k] = load_matches_from_folder(*v)
        Path(f'/tmp/{k}.json').write_text(json.dumps(matches[k]))


    return (matches,)


@app.cell
def _(m):
    m['U1d0wO63hQhrhwAZux3dKg']["osm"]
    return


@app.cell
def _(historical_dataset):
    historical_dataset._landmark_metadata.id.apply(lambda x: x.replace("(", "[").replace(")", "]").replace("'", '"'))
    return


@app.cell
def _():
    import itertools
    import torch
    return itertools, torch


@app.cell
def _(
    historical_dataset,
    historical_osm_embeddings,
    historical_osm_lm_idx_from_osm_id,
    itertools,
    json,
    matches,
    metadata_from_pano_id,
    osm_landmarks_from_pano_id,
    pano_embeddings_from_lm_idx,
    pano_lm_idx_from_lm_id,
    pano_sentences_from_pano_id,
    slu,
    torch,
    z3,
):
    import math

    def pano_embeddings_for_pano_id(pano_sentences):
        idxs = [pano_lm_idx_from_lm_id[x["lm_id"]] for x in pano_sentences]
        return pano_embeddings_from_lm_idx[idxs]

    def osm_embeddings_from_osm_lms(osm_lms, osm_lm_idx_from_osm_id, osm_embeddings):
        osm_lm_ids = osm_lms.pruned_props.apply(slu.custom_id_from_props)
        idxs = []
        for _, row in osm_lms.iterrows():
            pruned_props = row.pruned_props
            osm_lm_id = slu.custom_id_from_props(pruned_props)
            idxs.append(osm_lm_idx_from_osm_id[osm_lm_id])
        return osm_embeddings[idxs]

    def create_z3_problem(pano_osm_sims, heading_agreement, match_boost):
        num_pano_lms = pano_osm_sims.shape[0]
        num_osm_lms = pano_osm_sims.shape[1]
        assignment = [[z3.Int(f"a_{i}_{j}") for j in range(num_osm_lms)] for i in range(num_pano_lms)]
        opt = z3.Optimize()

        for i in range(num_pano_lms):
            # All rows can contain at most 2 values
            opt.add(z3.Sum(assignment[i]) < 3)
            for j in range(num_osm_lms):
                # Each entry can either be a zero or a 1
                opt.add(0 <= assignment[i][j])
                opt.add(assignment[i][j] <= 1)

        for j in range(num_osm_lms):
            # each column can contain at most 2 value
            opt.add(z3.Sum([assignment[i][j] for i in range(num_pano_lms)]) < 3)

        obj = []
        for i, j in itertools.product(range(num_pano_lms), range(num_osm_lms)):
            var = assignment[i][j] 
            factor = -0.2 + pano_osm_sims[i, j].item() * heading_agreement[i, j].item() * match_boost[i, j].item()
            obj.append(z3.ToReal(var) * factor)

        obj = z3.Sum(obj)
        opt.maximize(obj)
        opt.set(timeout=2000)

        opt.check()
        return assignment, opt.model()

    def compute_bounds_for_polygon(pano_loc_px, geometry):
        pano_y, pano_x = pano_loc_px
        # We need to compute the interval that the polygon occupies.
        xs, ys = geometry.exterior.xy
        dx_in_web_mercator = torch.tensor(xs) - pano_x
        dy_in_web_mercator = torch.tensor(ys) - pano_y

        pano_from_web_mercator = torch.tensor([[0.0, -1.0],
                                               [-1.0,  0.0]])
        delta_in_pano = (
            pano_from_web_mercator @
            torch.stack([dx_in_web_mercator, dy_in_web_mercator]))

        bounds = None
        prev_theta = None
        thetas_in_pano = torch.atan2(delta_in_pano[1, :], delta_in_pano[0, :]).squeeze()
        wrap_accumulator = 0
        for theta_in_pano in thetas_in_pano:
            unwrapped_theta = theta_in_pano + wrap_accumulator
            if bounds is None:
                bounds = torch.tensor([unwrapped_theta, unwrapped_theta])
                prev_theta = unwrapped_theta

            if unwrapped_theta - prev_theta > torch.pi:
                wrap_accumulator -= 2 * torch.pi
                unwrapped_theta = theta_in_pano + wrap_accumulator
            elif unwrapped_theta - prev_theta < -torch.pi:
                wrap_accumulator += 2 * torch.pi
                unwrapped_theta = theta_in_pano + wrap_accumulator

            if unwrapped_theta < bounds[0]:
                bounds[0] = unwrapped_theta
            elif unwrapped_theta > bounds[1]:
                bounds[1] = unwrapped_theta

            prev_theta = unwrapped_theta

        return bounds


    def compute_osm_landmark_heading_intervals(pano_metadata, osm_landmarks):
        out = []
        pano_y = pano_metadata["web_mercator_y"].values[0]
        pano_x = pano_metadata["web_mercator_x"].values[0]

        for i, (_, landmark) in enumerate(osm_landmarks.iterrows()):
            geometry = landmark["geometry_px"]

            # We want to compute the range spanned by this geometry.
            if geometry.geom_type == "Point":
                # These deltas are in the web mercator frame where +x goes from west to east
                # and +y goes from north to south.
                dx_in_web_mercator = geometry.x - pano_x
                dy_in_web_mercator = geometry.y - pano_y

                # We rotate them such that +x goes from south to north and +y increases from
                # west to east. The panoramas are such that north is always the middle column.
                # An angle of -pi/+pi correspond to the left/right edge respectively
                pano_from_web_mercator = torch.tensor([[0.0, -1.0],
                                                       [-1.0,  0.0]])
                delta_in_pano = pano_from_web_mercator @ torch.tensor(
                    [[dx_in_web_mercator, dy_in_web_mercator]], dtype=torch.float32).T

                theta = math.atan2(delta_in_pano[1], delta_in_pano[0])
                bounds = [theta, theta]
                out.append(bounds)

            elif geometry.geom_type == "LineString":
                xs,  ys = geometry.xy
                dx_in_web_mercator = torch.tensor(xs) - pano_x
                dy_in_web_mercator = torch.tensor(ys) - pano_y

                pano_from_web_mercator = torch.tensor([[0.0, -1.0],
                                                       [-1.0,  0.0]])
                delta_in_pano = (
                    pano_from_web_mercator @
                    torch.stack([dx_in_web_mercator, dy_in_web_mercator]))

                thetas = torch.atan2(delta_in_pano[1, :], delta_in_pano[0, :]).squeeze()
                thetas = torch.sort(thetas).values
                if thetas[1] - thetas[0] > torch.pi:
                    thetas = torch.flip(thetas, (0,))
                frac = (thetas + torch.pi) / (2 * torch.pi)
                bounds = [thetas[0].item(), thetas[1].item()]
                out.append(bounds)

            elif geometry.geom_type == "Polygon":
                bounds = compute_bounds_for_polygon((pano_y, pano_x), geometry)
                if bounds[1] - bounds[0] > 2 * torch.pi:
                    # We're enclosed, so return the entire interval:
                    bounds = torch.tensor([-torch.pi, torch.pi])
                else:
                    bounds = torch.remainder(bounds, 2*torch.pi)
                    bounds[bounds > torch.pi] -= 2 * torch.pi

                bounds = bounds.tolist()
                out.append(bounds)

            elif geometry.geom_type == "MultiPolygon":
                bounds = torch.tensor([torch.inf, -torch.inf])
                for p in geometry.geoms:
                    new_bounds = compute_bounds_for_polygon((pano_y, pano_x), p)
                    if new_bounds[0] < bounds[0]:
                        bounds[0] = new_bounds[0]
                    if new_bounds[1] > bounds[1]:
                        bounds[1] = new_bounds[1]
                if bounds[1] - bounds[0] > 2 * torch.pi:
                    # We're enclosed, so return the entire interval:
                    bounds = torch.tensor([-torch.pi, torch.pi])
                else:
                    bounds = torch.remainder(bounds, 2*torch.pi)
                    bounds[bounds > torch.pi] -= 2 * torch.pi
                bounds = bounds.tolist()
                out.append(bounds)

            else:
                raise ValueError(f"Unrecognized geometry type: {landmark["geometry_px"].geom_type}")

        return torch.tensor(out)

    def angle_intersects(a, b):
        """
        Return True if angular intervals a and b intersect.
        Intervals are (start_deg, end_deg), modulo 360°.
        start > end indicates wrap-around.
        """

        def split_interval(start, end):
            start %= 360
            end   %= 360
            if start <= end:
                return [(start, end)]
            else:
                # wrap-around -> two piecewise intervals
                return [(start, 360), (0, end)]

        def linear_intersect(i1, i2):
            return max(i1[0], i2[0]) <= min(i1[1], i2[1])

        A = split_interval(*a)
        B = split_interval(*b)

        # Check all piecewise combinations
        for x in A:
            for y in B:
                if linear_intersect(x, y):
                    return True
        return False


    def compute_heading_agreement(pano_metadata, pano_lm_metadata, osm_landmarks):
        osm_lm_headings = compute_osm_landmark_heading_intervals(pano_metadata, osm_landmarks)
        osm_lm_headings[osm_lm_headings < 0] += 2 * torch.pi
        osm_lm_headings = osm_lm_headings / torch.pi * 180
        pano_headings = [x["yaw_angles"] for x in pano_lm_metadata]

        num_pano_lms = len(pano_lm_metadata)
        num_osm_lms = len(osm_landmarks)
        out = torch.zeros((num_pano_lms, num_osm_lms))

        for i, j in itertools.product(range(num_pano_lms), range(num_osm_lms)):
            pano_heading = pano_headings[i]
            osm_heading = osm_lm_headings[j]
            debug = False
            #  if i == 1 and j == 74:
            #      debug = True
            intersection = 0
            union = 0

            if debug:
                print(f'{i=} {j=} {pano_heading=} {osm_heading=}')
            for p in [0, 90, 180, 270]:
                pano_interval = ((p - 45) % 360, (p+45) % 360)
                has_intersection = angle_intersects(pano_interval, osm_heading)

                if debug:
                    print(f"{p=} {pano_interval=} {has_intersection=} {(p in pano_heading)=}")
                if has_intersection or p in pano_heading:
                   union += 1 
                if has_intersection and p in pano_heading:
                    intersection += 1
            if debug:
                print(f"{intersection=} {union=} {intersection/union=}")
            out[i, j] = intersection / union
        return out, osm_lm_headings

    def compute_match_boost(pano_lm_metadata, osm_landmarks, matches):
        num_pano_landmarks = len(pano_lm_metadata)
        # When asking the LLM to produce a match, we deduplicated the landmarks based on their pruned properties.
        # We build a map from the index in osm_landmarks to the index in matches["osm"]
        def parse_id(x):
            return json.loads(x.replace('(', '[')
                             .replace(')', ']')
                             .replace("'", '"'))

        osm_ids = osm_landmarks.id.apply(parse_id).reset_index(drop=True)

        out = torch.zeros((num_pano_landmarks, len(osm_landmarks)), dtype=torch.float32)

        for i, pano_matches in enumerate(matches["matches"]["matches"]):
            for osm_match_idx in pano_matches["set_2_matches"]:
                for osm_id in matches["osm"][osm_match_idx]["ids"]:
                    osm_idx = osm_ids[osm_ids.apply(lambda x: x == osm_id)].index.values[0]
                    out[i, osm_idx] += 1.0

        return out



    def produce_matches_for_pano_id(pano_id, dataset, osm_lm_idx_from_osm_id, osm_embeddings, matches_to_use):
        # Collect the landmark sentences and embeddings
        pano_sentences = pano_sentences_from_pano_id(pano_id)
        pano_embeddings = pano_embeddings_for_pano_id(pano_sentences)
        pano_metadata = dataset._panorama_metadata[dataset._panorama_metadata["pano_id"] == pano_id]
        pano_lm_metadata = metadata_from_pano_id[pano_id]

        # Collect the osm landmarks and embeddings
        _, osm_landmarks = osm_landmarks_from_pano_id(pano_id, dataset)
        osm_embeddings = osm_embeddings_from_osm_lms(osm_landmarks, osm_lm_idx_from_osm_id, osm_embeddings)

        sentence_metadata = metadata_from_pano_id[pano_id]

        pano_osm_sims = pano_embeddings @ osm_embeddings.T
        heading_agreement, osm_heading_intervals = compute_heading_agreement(pano_metadata, pano_lm_metadata, osm_landmarks)
        match_boost = compute_match_boost(pano_lm_metadata, osm_landmarks, matches[matches_to_use][pano_id])

        # print(heading_agreement)
        # print(match_boost)
    
        assignment, model = create_z3_problem(
            pano_osm_sims = pano_osm_sims,
            heading_agreement=heading_agreement,
            match_boost=match_boost)

        # for i, j in itertools.product(range(len(pano_embeddings)), range(len(osm_embeddings))):
        #     if model[assignment[i][j]] == 1:
        #         print('='*30)
        #         print(i, j, model[assignment[i][j]], f"sim: {pano_osm_sims[i, j]:0.4f} heading agreement: {heading_agreement[i, j]} match boost: {match_boost[i, j]}")
        #         print('\t', pano_sentences[i]["sentence"], pano_lm_metadata[i])
        #         print('\t', osm_landmarks.iloc[j]['pruned_props'], f"heading range: {osm_heading_intervals[j]}")

        return assignment, model, pano_osm_sims, heading_agreement, match_boost, osm_heading_intervals, osm_landmarks


    #     print(pano_id)
    #     print(sentence_metadata)
    #     print(pano_sentences)
    #     print(osm_landmarks)
    #     print(osm_embeddings.shape)
        ...

    _pano_id = "EZd0yEWwrNSoGQhljhV1Uw"
    assignment, model, pano_osm_sims, heading_agreement, match_boost, osm_heading_intervals, _ = produce_matches_for_pano_id(_pano_id, historical_dataset, historical_osm_lm_idx_from_osm_id,   historical_osm_embeddings,  matches_to_use='medium_historical')

    return (produce_matches_for_pano_id,)


@app.cell
def _(dataset, metadata_from_pano_id, mo):
    _pano_id = "SJq255LiDUv2fBTorbRUIQ"
    _row = dataset._panorama_metadata[dataset._panorama_metadata["pano_id"] == _pano_id]

    print(metadata_from_pano_id[_pano_id])

    _paths = dataset._satellite_metadata.iloc[_row.positive_satellite_idxs.values[0] + _row.semipositive_satellite_idxs.values[0]].path
    mo.vstack(
        [mo.image(src=_row.path.iloc[0]),
        mo.hstack([mo.image(src=x,width=640) for x in _paths.values])])
    return


@app.cell
def _(
    historical_dataset,
    historical_osm_embeddings,
    historical_osm_lm_idx_from_osm_id,
    json,
    matches,
    produce_matches_for_pano_id,
):
    import tqdm
    import copy
    to_fill = copy.deepcopy(matches["medium_historical"])

    for _pano_id, _v in tqdm.tqdm(to_fill.items()):
        _assignment, _model, _, _, _, _, _osm_landmarks = produce_matches_for_pano_id(_pano_id, historical_dataset, historical_osm_lm_idx_from_osm_id, historical_osm_embeddings, matches_to_use='medium_historical')
        _new_matches = []

        _osm_match_id_from_pruned_props = {}
        for _, _row in _osm_landmarks.iterrows():
            _row_id = json.loads(_row.id.replace('(', '[').replace(')', ']').replace("'", '"'))
            for _i, _deduped_osm_lm in enumerate(to_fill[_pano_id]["osm"]):
                if _row_id in _deduped_osm_lm["ids"]:
                    _osm_match_id_from_pruned_props[_row.pruned_props] = _deduped_osm_lm["index"]

        for _i in range(len(_assignment)):
            _new_matches.append({
                "set_1_id": to_fill[_pano_id]["pano"][_i]["index"],
                "set_2_matches": []
            })
            for _j in range(len(_assignment[0])):
                if _model[_assignment[_i][_j]] == 1:
                    _osm_lm = _osm_landmarks.iloc[_j]
                    _new_matches[-1]["set_2_matches"].append(_osm_match_id_from_pruned_props[_osm_lm.pruned_props])
        to_fill[_pano_id]["matches"]["matches"] = _new_matches

    return (to_fill,)


@app.cell
def _():
    return


@app.cell
def _(Path, json, to_fill):
    Path('/tmp/medium_historical_heading.json').write_text(json.dumps(to_fill))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
