
import common.torch.load_torch_deps
import torch
import math
from common.ollama import pyollama
import json
import hashlib
import pandas as pd

from experimental.overhead_matching.swag.model.swag_config_types import (
        SemanticLandmarkExtractorConfig)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
        ModelInput, ExtractorOutput)
from sentence_transformers import SentenceTransformer


def describe_landmark(props, ollama):
    d = dict(props)
    prompt = f"""Generate a natural language description of this openstreetmap landmark.
    Only include information relevant for visually identifying the object.
    For example, don't include payment methods accepted. Don't include any details not derived
    from the landmark information. Include no other details.

    {json.dumps(d)}"""
    description = ollama(prompt)
    return description


def prune_landmark(props):
    to_drop = [
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
        if v is None:
            continue
        if isinstance(v, pd.Timestamp):
            continue
        for prefix in to_drop:
            if k.startswith(prefix):
                should_add = False
                break
        if should_add:
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
    def __init__(self, config: SemanticLandmarkExtractorConfig):
        super().__init__()
        self._sentence_embedding_model = SentenceTransformer(config.sentence_model_str)
        self._ollama = config.llm_str
        self._description_cache = {}

        self._feature_markers = {
            "Point": torch.nn.Parameter(torch.randn(1, 1, self.output_dim)),
            "LineString": torch.nn.Parameter(torch.randn(1, 1, self.output_dim)),
            "Polygon": torch.nn.Parameter(torch.randn(1, 1, self.output_dim)),
            "MultiPolygon": torch.nn.Parameter(torch.randn(1, 1, self.output_dim)),
        }

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        max_num_landmarks = max([len(x["landmarks"]) for x in model_input.metadata])
        batch_size = len(model_input.metadata)

        is_panorama = 'pano_id' in model_input.metadata[0]

        sentences = []
        sentence_splits = [0]
        for item in model_input.metadata:
            sentence_splits.append(sentence_splits[-1] + len(item["landmarks"]))
            if isinstance(self._ollama, str):
                self._ollama = pyollama.Ollama(self._ollama)
                self._ollama.__enter__()

            for landmark in item["landmarks"]:
                props = prune_landmark(landmark)
                if props in self._description_cache:
                    sentences.append(self._description_cache[props])
                    continue

                description = describe_landmark(props, self._ollama)
                sentences.append(description)
                self._description_cache[props] = description

        with torch.no_grad():
            sentence_embedding = self._sentence_embedding_model.encode(
                    sentences,
                    convert_to_tensor=True,
                    device=model_input.image.device).reshape(-1, self.output_dim)

        mask = torch.ones((batch_size, max_num_landmarks), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_landmarks, self.output_dim))
        positions = torch.zeros((batch_size, max_num_landmarks, 2, 2))
        max_description_length = max([len(x.encode('utf-8')) for x in sentences]) if len(sentences) > 0 else 0
        sentence_debug = torch.zeros(
            (batch_size, max_num_landmarks, max_description_length), dtype=torch.uint8)

        for batch_item in range(batch_size):
            start_idx, end_idx = sentence_splits[batch_item:batch_item+2]
            num_landmarks_for_item = end_idx - start_idx
            mask[batch_item, :num_landmarks_for_item] = False
            features[batch_item, :num_landmarks_for_item] = sentence_embedding[start_idx:end_idx]

            # Compute the positions of the landmarks
            if is_panorama:
                positions[batch_item, :num_landmarks_for_item] = compute_landmark_pano_positions(
                        model_input.metadata[batch_item], model_input.image.shape[-2:])
            else:
                positions[batch_item, :num_landmarks_for_item] = compute_landmark_sat_positions(
                        model_input.metadata[batch_item])

            # Store the sentences in a debug tensor
            if num_landmarks_for_item > 0:
                curr_sentences = sentences[start_idx:end_idx]
                sentence_tensors = [torch.tensor(list(x.encode('utf-8')), dtype=torch.uint8)
                                    for x in curr_sentences]
                sentence_tensor = torch.nested.nested_tensor(
                    sentence_tensors)
                sentence_tensor = sentence_tensor.to_padded_tensor(
                    padding=0, output_size=(num_landmarks_for_item, max_description_length))
                sentence_debug[batch_item, :num_landmarks_for_item] = sentence_tensor

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device),
            debug={'sentences': sentence_debug.to(model_input.image.device)})

    @property
    def output_dim(self):
        return self._sentence_embedding_model.get_sentence_embedding_dimension()

    @property
    def num_position_outputs(self):
        return 2
