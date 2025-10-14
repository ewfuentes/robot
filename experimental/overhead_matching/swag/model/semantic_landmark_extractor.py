
import common.torch.load_torch_deps
import torch
import math
from common.ollama import pyollama
import json
import hashlib
from pathlib import Path
import pandas as pd
import openai
import base64
from experimental.overhead_matching.swag.model.swag_config_types import (
    SemanticLandmarkExtractorConfig)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)

BATCH_SIZE = 49_999

def prune_landmark(props):
    to_drop = [
        "index",  # for props that come from a dataloader
        "web_mercator",
        "panorama_idxs",
        "satellite_idxs",
        "landmark_type",
        "element",
        "id",
        "geometry",
        "opening_hours",
        "website",
        "addr:city",
        "addr:state",
        'check_date',
        'checked_exists',
        'opening_date',
        'chicago:building_id',
        'survey:date',
        'payment',
        'disused',
        'time',
        'end_date']
    out = set()
    for (k, v) in props.items():
        should_add = True
        for prefix in to_drop:
            if k.startswith(prefix):
                should_add = False
                break
        if not should_add:
            continue
        if pd.isna(v):
            continue
        if isinstance(v, pd.Timestamp):
            continue
        out.add((k, v))

    return frozenset(out)


def compute_bounds_for_polygon(pano_loc_px, geometry):
    pano_y, pano_x = pano_loc_px
    # We need to compute the interval that the polygon occupies.
    xs, ys = geometry.exterior.xy
    dx_in_web_mercator = torch.tensor(xs) - pano_x
    dy_in_web_mercator = torch.tensor(ys) - pano_y

    pano_from_web_mercator = torch.tensor([[0.0, -1.0],
                                           [1.0,  0.0]])
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


def compute_landmark_pano_positions(pano_metadata, pano_shape):
    out = []
    pano_y = pano_metadata["web_mercator_y"]
    pano_x = pano_metadata["web_mercator_x"]
    for landmark in pano_metadata["landmarks"]:
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
                                                   [1.0,  0.0]])
            delta_in_pano = pano_from_web_mercator @ torch.tensor(
                [[dx_in_web_mercator, dy_in_web_mercator]]).T

            theta = math.atan2(delta_in_pano[1], delta_in_pano[0])
            frac = (theta + math.pi) / (2 * math.pi)
            out.append([
                [pano_shape[0] / 2.0, pano_shape[1] * frac],
                [pano_shape[0] / 2.0, pano_shape[1] * frac]])
        elif geometry.geom_type == "LineString":
            xs,  ys = geometry.xy
            dx_in_web_mercator = torch.tensor(xs) - pano_x
            dy_in_web_mercator = torch.tensor(ys) - pano_y

            pano_from_web_mercator = torch.tensor([[0.0, -1.0],
                                                   [1.0,  0.0]])
            delta_in_pano = (
                pano_from_web_mercator @
                torch.stack([dx_in_web_mercator, dy_in_web_mercator]))

            thetas = torch.atan2(delta_in_pano[1, :], delta_in_pano[0, :]).squeeze()
            thetas = torch.sort(thetas).values
            if thetas[1] - thetas[0] > torch.pi:
                thetas = torch.flip(thetas, (0,))
            frac = (thetas + torch.pi) / (2 * torch.pi)
            out.append([
                [pano_shape[0] / 2.0, pano_shape[1] * frac[0]],
                [pano_shape[0] / 2.0, pano_shape[1] * frac[1]]])
        elif geometry.geom_type == "Polygon":
            bounds = compute_bounds_for_polygon((pano_y, pano_x), geometry)
            if bounds[1] - bounds[0] > 2 * torch.pi:
                # We're enclosed, so return the entire interval:
                bounds = torch.tensor([-torch.pi, torch.pi])
            else:
                bounds = torch.remainder(bounds, 2*torch.pi)
                bounds[bounds > torch.pi] -= 2 * torch.pi

            frac = (bounds + torch.pi) / (2 * torch.pi)
            out.append([
                [pano_shape[0] / 2.0, pano_shape[1] * frac[0].item()],
                [pano_shape[0] / 2.0, pano_shape[1] * frac[1].item()]])
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

            frac = (bounds + torch.pi) / (2 * torch.pi)
            out.append([
                [pano_shape[0] / 2.0, pano_shape[1] * frac[0].item()],
                [pano_shape[0] / 2.0, pano_shape[1] * frac[1].item()]])

        else:
            raise ValueError(f"Unrecognized geometry type: {landmark["geometry_px"].geom_type}")

    return torch.tensor(out)


def compute_landmark_sat_positions(sat_metadata):
    out = []
    sat_y = sat_metadata["web_mercator_y"]
    sat_x = sat_metadata["web_mercator_x"]
    for landmark in sat_metadata["landmarks"]:
        geometry = landmark["geometry_px"]
        if geometry.geom_type == "Point":
            out.append([
                [geometry.y - sat_y, geometry.x - sat_x],
                [geometry.y - sat_y, geometry.x - sat_x]])
        elif landmark['geometry_px'].geom_type == "LineString":
            # Approximate a linestring by it's first and last points
            x, y = geometry.xy
            out.append([
                [y[0] - sat_y, x[0] - sat_x],
                [y[1] - sat_y, x[1] - sat_x]])
        elif landmark['geometry_px'].geom_type in ["Polygon", "MultiPolygon"]:
            # Approximate a polygon by its axis aligned bounding box
            x_min, y_min, x_max, y_max = geometry.bounds
            out.append([
                # Top left
                [y_min - sat_y, x_min - sat_x],
                # Bottom Right
                [y_max - sat_y, x_max - sat_x]])
        else:
            raise ValueError(f"Unrecognized geometry type: {landmark["geometry_px"].geom_type}")
    return torch.tensor(out)


class SemanticLandmarkExtractor(torch.nn.Module):
    def __init__(self, config: SemanticLandmarkExtractorConfig, semantic_embedding_base_path: Path):
        super().__init__()
        self.config = config
        self.semantic_embedding_base_path = Path(semantic_embedding_base_path).expanduser()
        self._description_cache = {}
        self.files_loaded = False
        self.all_sentences = None

    def load_files(self):
        # lazy setup to speed things up when we're using caching
        sentence_directory = self.semantic_embedding_base_path / self.config.embedding_version / "sentences"
        embedding_directory = self.semantic_embedding_base_path / self.config.embedding_version / "embeddings"
        if sentence_directory.exists():
            self.all_sentences, _ = make_sentence_dict_from_json(
                load_all_jsonl_from_folder(sentence_directory))

        self.all_embeddings = make_embedding_dict_from_json(
            load_all_jsonl_from_folder(embedding_directory))
        assert len(self.all_embeddings) != 0, f"Failed to load any embeddings from {embedding_directory}"
        assert len(next(iter(self.all_embeddings.values()))) >= self.config.openai_embedding_size, f"Requested an embedding length longer than the OpenAI Embeddings {len(next(iter(self.all_embeddings.values())))}, requested {self.config.openai_embedding_size}"


    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if not self.files_loaded:
            self.load_files()
            self.files_loaded = True
        # drop landmarks not used by this extractor
        # true indicates valid, false indicates not valid
        landmark_mask = [torch.tensor([1 if lm['geometry'].geom_type.lower() == self.config.landmark_type.lower() else 0 for lm in batch_item["landmarks"]], dtype=bool) for batch_item in model_input.metadata]
        valid_landmarks = [x.sum() for x in landmark_mask]
        max_num_landmarks = max(valid_landmarks)
        batch_size = len(model_input.metadata)

        is_panorama = 'pano_id' in model_input.metadata[0]

        mask = torch.ones((batch_size, max_num_landmarks), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_landmarks, self.output_dim))
        positions = torch.zeros((batch_size, max_num_landmarks, 2, 2))
        max_description_length = 0
        for i, item in enumerate(model_input.metadata):
            num_landmarks_for_item = valid_landmarks[i]
            # Compute the positions of the landmarks
            if num_landmarks_for_item > 0:
                if is_panorama:
                    positions[i, :num_landmarks_for_item] = compute_landmark_pano_positions(
                        item, model_input.image.shape[-2:])[landmark_mask[i]]
                else:
                    positions[i, :num_landmarks_for_item] = compute_landmark_sat_positions(item)[landmark_mask[i]]
            landmark_index = 0
            for landmark in item["landmarks"]:
                # skip landmarks of the wrong type
                if landmark['geometry'].geom_type.lower() != self.config.landmark_type.lower():
                    continue
                props = prune_landmark(landmark)
                landmark_id = _custom_id_from_props(props)
                if landmark_id not in self.all_embeddings:
                    print(f"Warning: missing embedding for props: {props}, ID {landmark_id}")
                    continue
                features[i, landmark_index, :] = torch.tensor(self.all_embeddings[landmark_id])[:self.output_dim]  # crop off end if requested
                mask[i, landmark_index] = False
                landmark_index += 1

                if self.all_sentences is not None:
                    max_description_length = max(max_description_length, len(
                        self.all_sentences[landmark_id].encode("utf-8")))

        ## re-normalize incase we trimmed embeddings
        features[~mask] = features[~mask] / torch.norm(features[~mask], dim=-1).unsqueeze(-1)

        sentence_debug = torch.zeros(
            (batch_size, max_num_landmarks, max_description_length), dtype=torch.uint8)

        # Store the sentences in a debug tensor
        if self.all_sentences is not None:
            for i, item in enumerate(model_input.metadata):
                num_landmarks_for_item = valid_landmarks[i]
                sentence_tensors = []
                for landmark in item["landmarks"]:
                    if landmark['geometry'].geom_type.lower() != self.config.landmark_type.lower():
                        continue
                    props = prune_landmark(landmark)
                    landmark_id = _custom_id_from_props(props)
                    if landmark_id not in self.all_sentences:
                        continue
                    sentence_tensors.append(torch.tensor(list(self.all_sentences[landmark_id].encode('utf-8')), dtype=torch.uint8))
                if len(sentence_tensors):
                    landmarks_sentences_tensor = torch.nested.nested_tensor(sentence_tensors)
                    landmarks_sentences_tensor = landmarks_sentences_tensor.to_padded_tensor(
                        padding=0, output_size=(num_landmarks_for_item, max_description_length))
                    sentence_debug[i, :num_landmarks_for_item] = landmarks_sentences_tensor

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device),
            debug={'sentences': sentence_debug.to(model_input.image.device)})

    @property
    def output_dim(self):
        return self.config.openai_embedding_size

    @property
    def num_position_outputs(self):
        return 2


def _load_landmarks(geojson_list):
    import geopandas as gpd
    import pandas as pd
    return pd.concat([gpd.read_file(p) for p in geojson_list], ignore_index=True)


def _custom_id_from_props(props: dict) -> str:
    json_props = json.dumps(dict(props), sort_keys=True)
    custom_id = base64.b64encode(hashlib.sha256(
        json_props.encode('utf-8')).digest()).decode('utf-8')
    return custom_id

SYSTEM_PROMPTS = {
    'default': "your job is to produce short natural language descriptions of openstreetmap landmarks that are helpful for visually identifying the landmark. for example, do not include information about building identifiers that are unlikely to be discernable by visual inspection. don't include any details not derived from the provided landmark information. don't include descriptions about the lack of information. do not include instructions on how to identify the landmark. do include an address if provided.",
    'no-address': "your job is to produce short natural language descriptions of openstreetmap landmarks that are helpful for visually identifying the landmark. for example, do not include information about building identifiers that are unlikely to be discernable by visual inspection. don't include any details not derived from the provided landmark information. don't include descriptions about the lack of information. do not include instructions on how to identify the landmark. DO NOT include an address or parts of an address in the description."
}

def _create_requests(landmarks, prompt_type = "default"):
    system_prompt = SYSTEM_PROMPTS[prompt_type]

    user_prompt = "Produce a short natural language description for this landmark: "

    requests = []
    for props in landmarks:
        json_props = json.dumps(dict(props), sort_keys=True)
        requests.append({
            "custom_id": _custom_id_from_props(props),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5-nano",
                "max_completion_tokens": 3000,
                "reasoning_effort": 'low',
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + json_props},
                ]
            }
        })
    return requests


def launch_batch(idx, batch_requests_file, endpoint):
    import openai
    client = openai.OpenAI()
    batch_input_file = client.files.create(
        file=batch_requests_file.open('rb'),
        purpose='batch')

    return client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=endpoint,
        completion_window="24h",
        metadata={"description": f"landmark summarization part: {idx}"}
    )


def create_description_requests(args):
    from pathlib import Path
    out_path = Path(args.output_base) / 'sentence_requests'
    out_path.mkdir(parents=True, exist_ok=True)
    prompt_type = args.prompt_type
    import itertools
    print(f'create {args}')
    landmarks = _load_landmarks(args.geojson)
    unique_landmarks = {prune_landmark(row.dropna().to_dict()) for _, row in landmarks.iterrows()}
    requests = _create_requests(unique_landmarks, prompt_type=prompt_type)
    print("num requests", len(requests))
    for idx, request_batch in enumerate(itertools.batched(requests, BATCH_SIZE)):
        batch_requests_file = out_path / f'sentence_request_{idx:03d}.jsonl'
        batch_requests_file.write_text('\n'.join(json.dumps(r) for r in request_batch))

        if args.launch:
            batch_response = launch_batch(idx, batch_requests_file, "/v1/chat/completions")
            print(batch_response.id, end=" ")
    print()


def fetch(args):
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    client = openai.OpenAI()
    print(f'fetch {args}')
    for batch_id in args.batch_ids:
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            print(
                f"Batch has non 'completed' status is '{batch.status}'. Skipping!!. Batch info: {batch}")
            continue
        file_output = client.files.content(batch.output_file_id)

        with open(output_folder / batch.output_file_id, 'w') as f:
            f.write(file_output.text)
        print(f"Downloaded output for batch id {batch_id}")


def _make_sentence_embedding_request(custom_id: str, sentence: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
            "model": "text-embedding-3-small",
            "input": sentence,
        }
    }


def load_all_jsonl_from_folder(folder: Path) -> list:

    all_json_objs = []
    for file in folder.glob("*"):
        with open(file, 'r') as f:
            for line in f:
                all_json_objs.append(json.loads(line))
    return all_json_objs


def create_sentence_embedding_batch(args):
    from pathlib import Path
    import itertools
    input_path = Path(args.sentence_output_dir)
    output_base = Path(args.output_base) / "embedding_requests"
    output_base.mkdir(parents=True, exist_ok=True)
    # read all of the files
    all_responses = load_all_jsonl_from_folder(input_path)

    # create the batch API embedding requests
    sentence_dict, _ = make_sentence_dict_from_json(all_responses)
    all_requests = [_make_sentence_embedding_request(
        custom_id, sentence) for custom_id, sentence in sentence_dict.items()]
    for idx, request_batch in enumerate(itertools.batched(all_requests, BATCH_SIZE)):
        batch_requests_file = output_base / f'embedding_requests_{idx:03d}.jsonl'
        batch_requests_file.write_text('\n'.join(json.dumps(r) for r in request_batch))

        if args.launch:
            batch_response = launch_batch(idx, batch_requests_file, "/v1/embeddings")
            print(batch_response.id, end=" ")
    print()


def make_embedding_dict_from_json(embedding_jsons: list) -> dict[str, str]:
    out = {}
    for response in embedding_jsons:
        assert response["error"] == None
        custom_id = response["custom_id"]
        embedding = response["response"]["body"]["data"][0]["embedding"]
        assert custom_id not in out
        out[custom_id] = embedding
    return out


def make_sentence_dict_from_json(sentence_jsons: list) -> dict[str, str]:
    out = {}
    output_tokens = 0
    for response in sentence_jsons:
        if len(response['response']['body']) == 0:
            print(f"GOT EMPTY RESPONSE {response}. SKIPPING")
            continue
        assert response["error"] == None and \
            response["response"]["body"]["choices"][0]["finish_reason"] == "stop" and \
            response["response"]["body"]["choices"][0]["message"]["refusal"] == None
        custom_id = response["custom_id"]
        sentence = response["response"]["body"]["choices"][0]["message"]["content"]
        assert custom_id not in out
        out[custom_id] = sentence
        output_tokens += response["response"]["body"]["usage"]["completion_tokens"]
    return out, output_tokens


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Arguments required to create a batch job
    subparsers = parser.add_subparsers()

    create_parser = subparsers.add_parser('create_sentences')
    create_parser.add_argument('--geojson', required=True, nargs="+")
    create_parser.add_argument('--output_base', type=str, default="/tmp/")
    create_parser.add_argument('--prompt_type', type=str, default="default")
    create_parser.add_argument('--launch', action='store_true')
    create_parser.set_defaults(func=create_description_requests)

    # Arguments required to fetch the result of a batch job
    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument('--batch_ids', required=True, type=str,
                              nargs="+", help="List of batch IDs from batch processing jobs")
    fetch_parser.add_argument('--output_folder', type=str, required=True,
                              help="Folder to save the batch outputs to")
    fetch_parser.set_defaults(func=fetch)

    create_parser = subparsers.add_parser('create_embed')
    create_parser.add_argument('--sentence_output_dir')
    create_parser.add_argument('--output_base', type=str, default="/tmp/")
    create_parser.add_argument('--launch', action='store_true')
    create_parser.set_defaults(func=create_sentence_embedding_batch)


    args = parser.parse_args()
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args.func(args)
