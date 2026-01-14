
import common.torch.load_torch_deps
import torch
import math
from common.ollama import pyollama
import json
import hashlib
from pathlib import Path
import pandas as pd
import pickle
import openai
import base64
import time
from experimental.overhead_matching.swag.model.swag_config_types import (
    SemanticLandmarkExtractorConfig, ExtractorDataRequirement)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    ModelInput, ExtractorOutput)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    load_all_jsonl_from_folder, make_embedding_dict_from_json, make_sentence_dict_from_json,
    prune_landmark, make_sentence_dict_from_pano_jsons, convert_embeddings_to_tensors,
    custom_id_from_props, load_embeddings)
from multiprocessing import Pool
from functools import partial
import tqdm

BATCH_SIZE = 49_999


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


def compute_landmark_pano_positions(pano_metadata, pano_shape, landmark_mask=None):
    out = []
    pano_y = pano_metadata["web_mercator_y"]
    pano_x = pano_metadata["web_mercator_x"]

    for idx, landmark in enumerate(pano_metadata["landmarks"]):
        # Skip landmarks that are filtered out by the mask
        if landmark_mask is not None and not landmark_mask[idx]:
            continue

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


def compute_landmark_sat_positions(sat_metadata, landmark_mask=None):
    # The mask entry should be true if we want to keep it
    out = []
    sat_y = sat_metadata["web_mercator_y"]
    sat_x = sat_metadata["web_mercator_x"]

    for idx, landmark in enumerate(sat_metadata["landmarks"]):
        # Skip landmarks that are filtered out by the mask
        if landmark_mask is not None and not landmark_mask[idx]:
            continue

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
        self._batch_counter = 0

    def load_files(self):
        # lazy setup to speed things up when we're using caching
        sentence_directory = self.semantic_embedding_base_path / self.config.embedding_version / "sentences"
        embedding_directory = self.semantic_embedding_base_path / self.config.embedding_version / "embeddings"
        if sentence_directory.exists():
            self.all_sentences, _ = make_sentence_dict_from_json(
                load_all_jsonl_from_folder(sentence_directory))

        self.all_embeddings_tensor, self.landmark_id_to_idx = load_embeddings(
            embedding_directory,
            output_dim=self.config.openai_embedding_size,
            normalize=True)

        # Validate embeddings
        assert len(self.landmark_id_to_idx) != 0, f"Failed to load any embeddings from {embedding_directory}"

        self.files_loaded = True

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        if not self.files_loaded:
            self.load_files()

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
                        item, model_input.image.shape[-2:], landmark_mask=landmark_mask[i])
                else:
                    positions[i, :num_landmarks_for_item] = compute_landmark_sat_positions(
                        item, landmark_mask=landmark_mask[i])

            landmark_index = 0
            for landmark in item["landmarks"]:
                # skip landmarks of the wrong type
                if landmark['geometry'].geom_type.lower() != self.config.landmark_type.lower():
                    continue

                props = landmark['pruned_props']
                landmark_id = custom_id_from_props(props)

                if landmark_id not in self.landmark_id_to_idx:
                    print(f"Warning: missing embedding for props: {props}, ID {landmark_id}")
                    continue

                emb_idx = self.landmark_id_to_idx[landmark_id]
                features[i, landmark_index, :] = self.all_embeddings_tensor[emb_idx]
                mask[i, landmark_index] = False
                landmark_index += 1
                if self.all_sentences is not None:
                    max_description_length = max(max_description_length, len(
                        self.all_sentences[landmark_id].encode("utf-8")))

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
                    props = landmark['pruned_props']
                    landmark_id = custom_id_from_props(props)
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

    @property
    def data_requirements(self) -> list[ExtractorDataRequirement]:
        return [ExtractorDataRequirement.LANDMARKS]


def _load_landmarks(geojson_list):
    import geopandas as gpd

    def load_file(path):
        if Path(path).suffix == '.feather':
            return gpd.read_feather(path)
        else:
            return gpd.read_file(path)

    return pd.concat([load_file(p) for p in geojson_list], ignore_index=True)




def encode_image_to_base64(image_path: Path) -> str:
    """Encode a single image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def encode_images_parallel(image_paths: list[Path], num_workers: int = 8, disable_tqdm: bool = False) -> list[str]:
    """Encode multiple images to base64 in parallel using multiprocessing."""
    with Pool(num_workers) as pool:
        return list(tqdm.tqdm(
            pool.imap(encode_image_to_base64, image_paths),
            total=len(image_paths),
            desc="Encoding images",
            disable=disable_tqdm
        ))


PANORAMA_LANDMARK_SCHEMAS = {
    'all': {
        "name": "landmark_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "landmarks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "yaw_angles": {
                                "type": "array",
                                "items": {"type": "integer", "enum": [0, 90, 180, 270]}
                            }
                        },
                        "required": ["description", "yaw_angles"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["landmarks"],
            "additionalProperties": False
        }
    },
    'individual': {
        "name": "landmark_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "landmarks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"}
                        },
                        "required": ["description"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["landmarks"],
            "additionalProperties": False
        }
    }
}

EXAMPLE_SENTENCES = """
A small, man-made water tap, located east of a larger non-drinking fountain.
Lit, sheltered light-rail station platform with a bench and an adjacent bus stop; canopy over the platform, with seating and bus access, reference 21852.
A small, low-rise building.
A concrete footpath that climbs uphill, with a handrail on both sides.
A nine-story building along Masonic Avenue.
A professional law office storefront with a sign reading "Verso Law Group."
A private sports pitch with bright artificial turf, marked for soccer and American football.
A bus stop at the intersection of Denny Way and Dexter Ave N, with a sign labeled 2295.
A Divvy bike rental docking station at Wentworth Ave and 24th St, with 15 docks and Divvy/Lyft branding.
A small asphalt road labeled South Bayview Street, typical of a quiet, lightly trafficked street with a 20 mph speed limit.
A multi-story rectangular building about 22 meters tall.
A plain, mid-sized urban building.
A pedestrian crosswalk with traffic lights at street level, with no traffic island in the middle, and tactile paving for the visually impaired.
A narrow one-way service alley with an asphalt surface.
A mid-rise building approximately 22 meters tall.
A fast-food spot named Mr. Cow, known for corn dogs (Corndogs by Mr. Cow).
"""
SYSTEM_PROMPTS = {
    'default': "your job is to produce short natural language descriptions of openstreetmap landmarks that are helpful for visually identifying the landmark. for example, do not include information about building identifiers that are unlikely to be discernable by visual inspection. don't include any details not derived from the provided landmark information. don't include descriptions about the lack of information. do not include instructions on how to identify the landmark. do include an address if provided.",
    'no-address': "your job is to produce short natural language descriptions of openstreetmap landmarks that are helpful for visually identifying the landmark. for example, do not include information about building identifiers that are unlikely to be discernable by visual inspection. don't include any details not derived from the provided landmark information. don't include descriptions about the lack of information. do not include instructions on how to identify the landmark. DO NOT include an address or parts of an address in the description.",
    'panorama': f"You are an expert at identifying landmarks in street-level imagery. Identify distinctive, permanent landmarks visible in these image(s) and describe them in short, natural language descriptions. Do NOT include very distant objects that are likely to be more than a few hundered meters from where the image was taken. Focus on buildings, monuments, parks, infrastructure, or other landmarks that are likely to be present in OpenStreetMaps. Avoid transient landmarks like cars and pedestrians. If you can confidently make out text (e.g., a buisnesses name on a sign, a street sign), include these as landmarks. Do not mention the location of the landmark in the image or relative to other landmarks (e.g., on the left of the image/on the right side of the street). Match the style of these examples: {EXAMPLE_SENTENCES}. DO NOT make up details that are not present (for example, if there is no street sign in the image, don't say you are on a specific street).",
}

def _create_requests(landmarks, prompt_type = "default"):
    system_prompt = SYSTEM_PROMPTS[prompt_type]

    user_prompt = "Produce a short natural language description for this landmark: "

    requests = []
    for props in landmarks:
        json_props = json.dumps(dict(props), sort_keys=True)
        requests.append({
            "custom_id": custom_id_from_props(props),
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


def create_sentence_embedding_batch(args):
    from pathlib import Path
    import itertools
    input_path = Path(args.sentence_output_dir)
    output_base = Path(args.output_base) / "embedding_requests"
    output_base.mkdir(parents=True, exist_ok=True)

    # read all of the files
    all_responses = load_all_jsonl_from_folder(input_path)

    # Parse responses based on mode
    is_panorama = args.is_panorama

    if is_panorama:
        print("Processing panorama landmark responses...")
        sentence_dict, metadata_dict, output_tokens = make_sentence_dict_from_pano_jsons(all_responses)
        print(f"Extracted {len(sentence_dict)} landmark descriptions from {len(all_responses)} panorama responses")
        print(f"Total output tokens: {output_tokens}")

        # Save metadata for later use
        metadata_file = output_base / "panorama_metadata.jsonl"
        with open(metadata_file, 'w') as f:
            for panorama_id, landmarks_list in metadata_dict.items():
                for landmark_meta in landmarks_list:
                    f.write(json.dumps({
                        "custom_id": landmark_meta["custom_id"],
                        "panorama_id": panorama_id,
                        "landmark_idx": landmark_meta["landmark_idx"],
                        "yaw_angles": landmark_meta["yaw_angles"]
                    }) + '\n')
        print(f"Saved metadata to {metadata_file}")
    else:
        print("Processing regular landmark responses...")
        sentence_dict, output_tokens = make_sentence_dict_from_json(all_responses)
        print(f"Extracted {len(sentence_dict)} sentences")
        print(f"Total output tokens: {output_tokens}")

    # create the batch API embedding requests
    all_requests = [_make_sentence_embedding_request(
        custom_id, sentence) for custom_id, sentence in sentence_dict.items()]

    print(f"Creating {len(all_requests)} embedding requests...")

    for idx, request_batch in enumerate(itertools.batched(all_requests, BATCH_SIZE)):
        batch_requests_file = output_base / f'embedding_requests_{idx:03d}.jsonl'
        batch_requests_file.write_text('\n'.join(json.dumps(r) for r in request_batch))

        if args.launch:
            batch_response = launch_batch(idx, batch_requests_file, "/v1/embeddings")
            print(batch_response.id, end=" ")
    print()


def create_panorama_description_requests(args):
    """
    Create batch API requests for panorama landmark extraction.

    Args:
        args: Argument namespace containing:
            - pinhole_dir: Path to directory with panorama subfolders containing pinhole images
            - output_base: Path for output batch request files
            - submit_mode: 'all' (all 4 images in one request) or 'individual' (1 image per request)
            - num_workers: Number of parallel workers for image encoding
            - max_requests_per_batch: Maximum requests per batch file
            - launch: Whether to launch batch jobs automatically
            - disable_tqdm: Whether to disable progress bars
    """
    from pathlib import Path
    import itertools

    pinhole_dir = Path(args.pinhole_dir)
    output_base = Path(args.output_base) / 'panorama_sentence_requests'
    output_base.mkdir(parents=True, exist_ok=True)

    submit_mode = args.submit_mode
    num_workers = args.num_workers
    max_requests_per_batch = args.max_requests_per_batch
    disable_tqdm = args.disable_tqdm

    # Load pano IDs filter if provided
    pano_ids_to_process = None
    if args.pano_ids_file is not None:
        pano_ids_file = Path(args.pano_ids_file)
        if not pano_ids_file.exists():
            print(f"Error: pano_ids_file not found: {pano_ids_file}")
            return
        with open(pano_ids_file, 'r') as f:
            pano_ids_to_process = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(pano_ids_to_process)} panorama IDs from {pano_ids_file}")

    # Find all panorama subfolders
    all_panorama_folders = [f for f in pinhole_dir.iterdir() if f.is_dir()]

    # Filter by pano IDs if provided
    if pano_ids_to_process is not None:
        # Folder names are in format "pano_id,lat,lon," so extract just the pano_id part
        panorama_folders = []
        for f in all_panorama_folders:
            # Extract pano_id by splitting on comma and taking first part
            pano_id = f.name.split(',')[0]
            if pano_id in pano_ids_to_process:
                panorama_folders.append(f)
        print(f"Found {len(all_panorama_folders)} total panorama folders, filtered to {len(panorama_folders)} based on pano_ids_file")
    else:
        panorama_folders = all_panorama_folders
        print(f"Found {len(panorama_folders)} panorama folders")

    if not panorama_folders:
        print(f"No panorama folders found to process")
        return

    print(f"Submit mode: {submit_mode}")
    print(f"Encoding workers: {num_workers}")
    print(f"Processing {len(panorama_folders)} panoramas")

    # Collect panorama metadata (don't load images yet)
    yaw_angles = [0, 90, 180, 270]
    panorama_image_map = {}  # Map from panorama stem to list of (yaw, image_path)

    for pano_folder in panorama_folders:
        pano_stem = pano_folder.name
        images_for_pano = []
        for yaw in yaw_angles:
            image_path = pano_folder / f"yaw_{yaw:03d}.jpg"
            if not image_path.exists():
                image_path = pano_folder / f"yaw_{yaw:03d}.png"
            if image_path.exists():
                images_for_pano.append((yaw, image_path))
            else:
                raise RuntimeError(f"Image not found! {image_path}")

        if len(images_for_pano) == 4:
            panorama_image_map[pano_stem] = images_for_pano
        else:
            print(f"Warning: Skipping {pano_stem}, found only {len(images_for_pano)}/4 images")

    print(f"Processing {len(panorama_image_map)} complete panoramas")

    # Get system prompt and schema for this mode
    system_prompt = SYSTEM_PROMPTS['panorama']
    schema = PANORAMA_LANDMARK_SCHEMAS[submit_mode]

    # Process in batches to avoid OOM
    # We'll encode and write requests in chunks
    PANORAMA_CHUNK_SIZE = 1000 

    panorama_items = list(panorama_image_map.items())
    total_requests_created = 0
    batch_idx = 0
    current_batch_requests = []
    current_batch_size = 0

    for chunk_start in tqdm.tqdm(range(0, len(panorama_items), PANORAMA_CHUNK_SIZE),
                                   desc="Processing chunks",
                                   disable=disable_tqdm):
        chunk_end = min(chunk_start + PANORAMA_CHUNK_SIZE, len(panorama_items))
        chunk = panorama_items[chunk_start:chunk_end]

        # Collect all images for this chunk
        chunk_image_paths = []
        for pano_stem, images_for_pano in chunk:
            for yaw, image_path in images_for_pano:
                chunk_image_paths.append(image_path)

        # Encode this chunk's images in parallel
        chunk_base64_images = encode_images_parallel(chunk_image_paths, num_workers, disable_tqdm=disable_tqdm)
        chunk_image_to_base64 = dict(zip(chunk_image_paths, chunk_base64_images))

        # Create requests for this chunk
        if submit_mode == 'all':
            # One request per panorama with all 4 images
            user_prompt = "These four images show the same location from different angles (0°, 90°, 180°, 270° yaw). Identify all distinctive landmarks visible in these images. For each landmark, specify which yaw angle(s) it is visible in. Return a JSON object with a 'landmarks' array containing objects with 'description' and 'yaw_angles' fields."
        elif submit_mode == "individual":
            user_prompt = f"Identify distinctive landmarks visible in this image that are likely to be present in OpenStreetMaps."
        else:
            raise RuntimeError(f"Unrecognized submit mode type: {submit_mode}")

        for pano_stem, images_for_pano in chunk:
            # Sort by yaw angle to ensure consistent ordering
            images_for_pano = sorted(images_for_pano, key=lambda x: x[0])
            all_content = []
            if submit_mode == 'all':
                content = [{"type": "text", "text": user_prompt}]
                for yaw, image_path in images_for_pano:
                    # Detect file extension
                    ext = image_path.suffix.lower()
                    mime_type = "image/jpeg" if ext == ".jpg" else "image/png"
                    content.append({
                        "type": "image_url",
                        "image_url": {"detail": "high", "url": f"data:{mime_type};base64,{chunk_image_to_base64[image_path]}"}
                    })
                all_content.append((content, None))
            else:  #individual
                for yaw, image_path in images_for_pano:
                    ext = image_path.suffix.lower()
                    mime_type = "image/jpeg" if ext == ".jpg" else "image/png"
                    content = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"detail": "high", "url": f"data:{mime_type};base64,{chunk_image_to_base64[image_path]}"}
                        }
                    ]
                    all_content.append((content, yaw))

            for content, yaw in all_content:
                request = {
                    "custom_id": f"{pano_stem}" + (f"_yaw_{yaw}" if yaw is not None else ""),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5-2025-08-07",
                        "response_format": {"type": "json_schema", "json_schema": schema},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": content}
                        ]
                    }
                }

                # Add to batch with size monitoring
                request_json = json.dumps(request)
                request_size = len(request_json.encode('utf-8'))

                if (current_batch_size + request_size > 190_000_000 or
                    len(current_batch_requests) >= max_requests_per_batch):
                    # Write current batch
                    _write_and_launch_batch(output_base, batch_idx, current_batch_requests, args.launch)
                    batch_idx += 1
                    current_batch_requests = []
                    current_batch_size = 0

                current_batch_requests.append(request)
                current_batch_size += request_size
                total_requests_created += 1

        # Clear chunk data to free memory
        del chunk_base64_images
        del chunk_image_to_base64

    # Write final batch if there are remaining requests
    if current_batch_requests:
        _write_and_launch_batch(output_base, batch_idx, current_batch_requests, args.launch)
        batch_idx += 1
    print()
    print(f"Created {total_requests_created} API requests")
    print(f"Wrote {batch_idx} batch file(s) to {output_base}")


def _write_and_launch_batch(output_base, batch_idx, batch_requests, should_launch):
    """Helper function to write and optionally launch a batch."""
    batch_file = output_base / f'panorama_request_{batch_idx:03d}.jsonl'
    batch_file.write_text('\n'.join(json.dumps(r) for r in batch_requests))

    if should_launch:
        batch_response = launch_batch(batch_idx, batch_file, "/v1/chat/completions")
        print(f"{batch_response.id}", end=" ")


def _process_embedding_path(dirpath, _, filenames):
    if dirpath.name != 'embeddings':
        return

    print(f"Processing: {dirpath}")
    jsonl = load_all_jsonl_from_folder(dirpath)
    embeddings = make_embedding_dict_from_json(jsonl)
    # Convert to tensors
    embeddings = convert_embeddings_to_tensors(embeddings)

    # Build tensor and index map for efficient access
    embedding_list = []
    landmark_id_to_idx = {}
    for idx, (landmark_id, emb) in enumerate(embeddings.items()):
        embedding_list.append(emb)
        landmark_id_to_idx[landmark_id] = idx

    all_embeddings_tensor = torch.stack(embedding_list)

    # Pickle the tensor and index map instead of the dict
    print(f"Writing {dirpath / 'embeddings.pkl'} with tensor shape {all_embeddings_tensor.shape}")
    with open(dirpath / "embeddings.pkl", 'wb') as file_out:
        pickle.dump((all_embeddings_tensor, landmark_id_to_idx), file_out)


def create_embedding_dict(args):
    import multiprocessing
    base_path = Path(args.embedding_dir)

    with multiprocessing.Pool(5) as p:
        p.starmap(_process_embedding_path, base_path.walk())


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
    create_parser.add_argument('--is_panorama', action='store_true',
                               help='Process panorama landmark extraction responses (extracts individual landmarks with metadata)')
    create_parser.set_defaults(func=create_sentence_embedding_batch)

    # Panorama landmark extraction
    panorama_parser = subparsers.add_parser('create_panorama_sentences',
                                            help='Create batch requests for panorama landmark extraction')
    panorama_parser.add_argument('--pinhole_dir', type=str, required=True,
                                 help='Directory containing panorama subfolders with pinhole images')
    panorama_parser.add_argument('--output_base', type=str, default="/tmp/",
                                 help='Base path for output batch request files')
    panorama_parser.add_argument('--submit_mode', type=str, default='all',
                                 choices=['all', 'individual'],
                                 help='Submit mode: "all" (all 4 images in one request) or "individual" (1 image per request)')
    panorama_parser.add_argument('--num_workers', type=int, default=8,
                                 help='Number of parallel workers for image encoding')
    panorama_parser.add_argument('--max_requests_per_batch', type=int, default=10000,
                                 help='Maximum requests per batch file')
    panorama_parser.add_argument('--launch', action='store_true',
                                 help='Automatically launch batch jobs')
    panorama_parser.add_argument('--disable_tqdm', action='store_true',
                                 help='Disable progress bars')
    panorama_parser.add_argument('--pano_ids_file', type=str, default=None,
                                 help='Optional file containing panorama IDs to process (one per line). If not provided, all panoramas are processed.')
    panorama_parser.set_defaults(func=create_panorama_description_requests)

    embedding_dict_parser = subparsers.add_parser('create_embedding_dict',
                                            help='convert jsonl files to a pickled dict')
    embedding_dict_parser.add_argument('--embedding_dir', type=str, default="/tmp/",
                                 help='Base path for output batch request files')
    embedding_dict_parser.set_defaults(func=create_embedding_dict)

    args = parser.parse_args()
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        args.func(args)
