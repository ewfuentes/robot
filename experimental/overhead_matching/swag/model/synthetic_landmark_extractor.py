
import common.torch.load_torch_deps
import torch
import math
import numpy as np
import hashlib

from experimental.overhead_matching.swag.model.swag_config_types import (
        SyntheticLandmarkExtractorConfig)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
        ModelInput, ExtractorOutput)


def compute_landmark_pano_positions(landmark_locations, pano_location, pano_shape):
    # Compute dx and dy in the ENU frame.
    dx = landmark_locations[:, 1] - pano_location[1] 
    dy = landmark_locations[:, 0] - pano_location[0]
    # np.arctan2 return an angle in [-pi, pi]. The panoramas are such that
    # north points in the middle of the panorama, so we compute theta as
    # atan(-dx / dy) so that zero angle corresponds to the center of the panorama
    # and the angle increases as we move right in the panorama
    theta = np.arctan2(dx, dy)
    frac = (theta + np.pi) / (2 * np.pi)
    out = np.stack((pano_shape[0] * np.ones_like(frac) / 2.0, pano_shape[1] * frac), axis=-1)
    return torch.tensor(out).reshape(-1, 2)


def compute_landmark_sat_positions(landmark_locations, sat_location):
    out = []
    dx = landmark_locations[:, 1] - sat_location[1] 
    dy = landmark_locations[:, 0] - sat_location[0]
    out = np.stack([dy, dx], axis=-1)
    return torch.tensor(out).reshape(-1, 2)

def clear_lowest_n_bits(x: int, n: int):
    return (x >> n) << n

def compute_landmark_pixel_locations(y_px:float , x_px: float, grid_bounds_px: int, log_grid_density: int):
    half_grid_bound_px = grid_bounds_px / 2
    top_px = y_px - half_grid_bound_px + (2 ** log_grid_density) - 1
    left_px = x_px - half_grid_bound_px + (2 ** log_grid_density) - 1
    bottom_px = y_px + half_grid_bound_px 
    right_px = x_px + half_grid_bound_px 

    snap_to_grid = lambda px: clear_lowest_n_bits(round(px), log_grid_density)
    top_px = snap_to_grid(top_px)
    left_px = snap_to_grid(left_px)
    bottom_px = snap_to_grid(bottom_px)
    right_px = snap_to_grid(right_px)

    ys = np.arange(top_px, bottom_px+1, 2**log_grid_density)
    xs = np.arange(left_px, right_px+1, 2**log_grid_density)
    return np.stack(np.meshgrid(ys, xs, indexing='ij'), axis=-1).reshape(-1, 2)

def compute_landmark_embeddings(locations_px, embedding_dim):
    # out = np.zeros((locations_px.shape[0], embedding_dim), dtype=np.float32)
    hashes = [hashlib.sha256(x.tobytes()).digest() for x in locations_px]
    hashes = b''.join(hashes)
    hashes = np.frombuffer(hashes, dtype=np.uint8).reshape(-1, 32)
    bits = np.unpackbits(hashes, axis=-1, count=embedding_dim)
    return (2.0 * bits - 1.0).astype(np.float32)

class SyntheticLandmarkExtractor(torch.nn.Module):
    def __init__(self, config: SyntheticLandmarkExtractorConfig):
        super().__init__()
        self._log_grid_density = config.log_grid_density
        self._grid_bounds_px= config.grid_bounds_px
        self._should_produce_bearing_position_for_pano = config.should_produce_bearing_position_for_pano
        self._embedding_dim = config.embedding_dim

    def forward(self, model_input: ModelInput) -> ExtractorOutput:
        batch_size = len(model_input.metadata)
        num_cols = (self._grid_bounds_px // (2 ** self._log_grid_density)) + 1
        num_rows = num_cols
        max_num_landmarks = num_cols * num_rows
        
        mask = torch.ones((batch_size, max_num_landmarks), dtype=torch.bool)
        features = torch.zeros((batch_size, max_num_landmarks, self.output_dim))
        positions = torch.zeros((batch_size, max_num_landmarks, 2))

        is_panorama = "pano_id" in model_input.metadata[0]

        for batch_item in range(batch_size):
            # Compute the pixel locations of the landmarks
            y_px = model_input.metadata[batch_item]["web_mercator_y"]
            x_px = model_input.metadata[batch_item]["web_mercator_x"]
            landmark_locations_px = compute_landmark_pixel_locations(
                    y_px, x_px, self._grid_bounds_px, self._log_grid_density)

            # Compute the landmark embedding
            landmark_features = compute_landmark_embeddings(landmark_locations_px, self._embedding_dim)
            num_landmarks_for_item = landmark_locations_px.shape[0]
            mask[batch_item, :num_landmarks_for_item] = False
            features[batch_item, :num_landmarks_for_item] = torch.tensor(landmark_features)

            # Compute the positions of the landmarks
            if is_panorama and self._should_produce_bearing_position_for_pano:
                positions[batch_item, :num_landmarks_for_item] = compute_landmark_pano_positions(
                        landmark_locations_px, (y_px, x_px), model_input.image.shape[-2:])
            else:
                positions[batch_item, :num_landmarks_for_item] = compute_landmark_sat_positions(
                        landmark_locations_px, (y_px, x_px))
        return ExtractorOutput(
            features=features.to(model_input.image.device),
            mask=mask.to(model_input.image.device),
            positions=positions.to(model_input.image.device))

    @property
    def output_dim(self):
        return self._embedding_dim
