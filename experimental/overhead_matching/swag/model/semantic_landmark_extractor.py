
import common.torch.load_torch_deps
import torch
import math

from experimental.overhead_matching.swag.model.swag_config_types import (
        SemanticLandmarkExtractorConfig)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
        ModelInput, ExtractorOutput)
from sentence_transformers import SentenceTransformer


def describe_landmark(props):
    """
    Generates a detailed natural language description of a landmark.

    Args:
      exemplar: A dictionary representing a landmark, typically with a
                'properties' key.

    Returns:
      A string containing the natural language description.
    """
    landmark_type = props.get('landmark_type', 'unknown landmark type').replace('_', ' ')
    name = props.get('name')

    address_parts = [
        props.get('addr:housenumber'),
        props.get('addr:street'),
        props.get('addr:city'),
        props.get('addr:state'),
        props.get('addr:postcode')
    ]
    address = ", ".join(filter(None, address_parts))

    description_parts = []

    # Core description phrase
    if name:
        description_parts.append(f"A {landmark_type} named **{name}**")
    else:
        description_parts.append(f"An unnamed {landmark_type}")

    if address:
        description_parts.append(f"is located at {address}")

    # Add details based on landmark type
    if landmark_type == 'bus stop' or landmark_type == 't stop':
        network = props.get('network')
        operator = props.get('operator')

        if network:
            description_parts.append(f"It's serviced by the **{network}** network")
        if operator and operator != network:
            description_parts.append(f"and operated by **{operator}**")

        features = []
        if props.get('bench') == 'yes':
            features.append("a bench")
        if props.get('shelter') == 'yes':
            features.append("a shelter")
        if props.get('wheelchair') == 'yes':
            features.append("wheelchair accessible")

        if features:
            if len(features) > 1:
                features_str = f"which includes {', '.join(features[:-1])} and {features[-1]}"
            else:
                features_str = f"which has {features[0]}"
            description_parts.append(features_str)

        departures_board = props.get('departures_board')
        if departures_board == 'realtime':
            description_parts.append("and features a real-time departures board")

        public_transport = props.get('public_transport')
        train = props.get('train')
        subway = props.get('subway')
        if public_transport == 'station' and (train == 'yes' or subway == 'yes'):
            transport_types = []
            if train == 'yes':
                transport_types.append('train')
            if subway == 'yes':
                transport_types.append('subway')
            transport_str = ' and '.join(transport_types)
            description_parts.append(f"which is a **{transport_str} station**")

    elif landmark_type == 'grocery store':
        shop_type = props.get('shop')
        brand = props.get('brand')
        opening_hours = props.get('opening_hours')

        if brand:
            description_parts.append(f"The store operates under the **{brand}** brand")

        if shop_type:
            description_parts.append(f"and is a {shop_type} type of store")

        if opening_hours:
            description_parts.append(f"with business hours of {opening_hours}")

        store_features = []
        if props.get('atm') == 'yes':
            store_features.append("an ATM")
        if props.get('fast_food') == 'yes':
            store_features.append("fast food offerings")

        if store_features:
            if len(store_features) > 1:
                store_features_str = f"It also provides {', '.join(store_features[:-1])} and {store_features[-1]}"
            else:
                store_features_str = f"It also provides {store_features[0]}"
            description_parts.append(store_features_str)

    elif landmark_type == 'places of worship':
        religion = props.get('religion')
        denomination = props.get('denomination')

        if religion:
            description_parts.append(f"It is a religious building for the **{religion}** faith")
            if denomination:
                description_parts.append(f"of the **{denomination}** denomination")
        if props.get('polling_station') == 'yes':
            description_parts.append("and it also serves as a polling station")

    elif landmark_type == 'restaurants':
        cuisine = props.get('cuisine')
        amenity = props.get('amenity')
        opening_hours = props.get('opening_hours')
        takeaway = props.get('takeaway')
        indoor_seating = props.get('indoor_seating')
        outdoor_seating = props.get('outdoor_seating')

        if cuisine:
            cuisine_list = cuisine.replace(";", " and ").split(" and ")
            cuisine_str = f"serving a wide variety of cuisines, including {', and '.join(cuisine_list)}"
            description_parts.append(cuisine_str)
        if amenity:
            description_parts.append(f"It is a **{amenity}**")
        if opening_hours:
            description_parts.append(f"with typical hours of {opening_hours}")

        restaurant_features = []
        if takeaway == 'yes':
            restaurant_features.append("takeout service")
        if indoor_seating == 'yes':
            restaurant_features.append("indoor seating")
        if outdoor_seating == 'yes':
            restaurant_features.append("outdoor seating")

        if restaurant_features:
            if len(restaurant_features) > 1:
                features_str = f"It offers {', '.join(restaurant_features[:-1])} and {restaurant_features[-1]}"
            else:
                features_str = f"It offers {restaurant_features[0]}"
            description_parts.append(features_str)

    elif landmark_type == 'school':
        amenity = props.get('amenity')
        operator = props.get('operator')
        grades = props.get('grades')
        website = props.get('website')

        if amenity:
            description_parts.append(f"which is an educational facility, specifically a **{amenity}**")
        if operator:
            description_parts.append(f"operated by **{operator}**")
        if grades:
            description_parts.append(f"and serves students in grades {grades}")
        if website:
            description_parts.append(f"with more information available at their website: {website}")

    # Combine parts into a final string, ensuring proper sentence structure
    final_description = ""
    if description_parts:
        final_description = description_parts[0]
        if len(description_parts) > 1:
            for part in description_parts[1:]:
                if part.startswith(('It is', 'It offers', 'It\'s')):
                    final_description += f". {part}"
                else:
                    final_description += f" {part}"

    # Capitalize the first letter and add a period if necessary
    if final_description:
        return final_description.strip() + "."
    return "An unknown landmark."


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
    for landmark in sat_metadata["landmarks"]:
        out.append((landmark["web_mercator_y"] - sat_metadata["web_mercator_y"],
                    landmark["web_mercator_x"] - sat_metadata["web_mercator_x"]))
    return torch.tensor(out).reshape(-1, 2)


class SemanticLandmarkExtractor(torch.nn.Module):
    def __init__(self, config: SemanticLandmarkExtractorConfig):
        super().__init__()
        self._sentence_embedding_model = SentenceTransformer(config.sentence_model_str)

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        max_num_landmarks = max([len(x["landmarks"]) for x in model_input.metadata])
        batch_size = len(model_input.metadata)

        is_panorama = 'pano_id' in model_input.metadata[0]

        sentences = []
        sentence_splits = [0]
        for item in model_input.metadata:
            sentence_splits.append(sentence_splits[-1] + len(item["landmarks"]))
            for landmark in item["landmarks"]:
                sentences.append(describe_landmark(landmark))

        with torch.no_grad():
            sentence_embedding = self._sentence_embedding_model.encode(
                    sentences,
                    convert_to_tensor=True,
                    device=model_input.image.device).reshape(-1, self.output_dim)

        mask = torch.ones((batch_size, max_num_landmarks), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_landmarks, self.output_dim))
        positions = torch.zeros((batch_size, max_num_landmarks, 2))

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

        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device))

    @property
    def output_dim(self):
        return self._sentence_embedding_model.get_sentence_embedding_dimension()
