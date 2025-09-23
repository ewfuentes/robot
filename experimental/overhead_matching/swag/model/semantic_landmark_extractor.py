
import common.torch.load_torch_deps
import torch
import math
from common.ollama import pyollama
import json

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
        'survey:date']
    out = set()
    for (k, v) in props.items():
        should_add = True
        if v is None:
            continue
        for prefix in to_drop:
            if k.startswith(prefix):
                should_add = False
                break
        if should_add:
            out.add((k, v))

    return frozenset(out)


def compute_landmark_pano_positions(pano_metadata, pano_shape):
    out = []
    for landmark in pano_metadata["landmarks"]:
        # Compute dx and dy in the ENU frame.
        dx = landmark["web_mercator_x"] - pano_metadata["web_mercator_x"]
        dy = landmark["web_mercator_y"] - pano_metadata["web_mercator_y"]
        # math.atan2 return an angle in [-pi, pi]. The panoramas are such that
        # north points in the middle of the panorama, so we compute theta as
        # atan(-dx / dy) so that zero angle corresponds to the center of the panorama
        # and the angle increases as we move right in the panorama
        theta = math.atan2(dx, dy)
        frac = (theta + math.pi) / (2 * math.pi)
        out.append((pano_shape[0] / 2.0, pano_shape[1] * frac))
    return torch.tensor(out).reshape(-1, 2)


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
            import IPython
            IPython.embed()
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
