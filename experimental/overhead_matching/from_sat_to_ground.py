import numpy as np
from scipy.ndimage import distance_transform_edt


def _calculate_reward_penalty_maps_from_semantic_satellite(
    semantic_satellite: np.ndarray,  # h x w, uint8 with class
    sigma: float,
    num_classes: int = 3
) -> np.ndarray:

    h, w = semantic_satellite.shape
    final_reward_map = np.zeros((num_classes, h, w))
    final_penalty_map = np.zeros((num_classes, h, w))

    for class_i in range(num_classes):
        class_map = np.ones((h, w))
        class_map[semantic_satellite == class_i] = 0
        #  each pixel value is the euclidean distance to the nearest pixel of class class_i
        dis_l = distance_transform_edt(class_map)
        final_reward_map[class_i] = np.exp(- 1 / (2 * sigma) * dis_l)

        inv_class_map = np.zeros((h, w))
        inv_class_map[semantic_satellite == class_i] = 1
        dis_nl = distance_transform_edt(inv_class_map)
        final_penalty_map[class_i] = -1 * np.exp(-1 / (2 * sigma) * dis_nl)

    return final_reward_map, final_penalty_map


def _project_pointcloud_into_bev(semantic_points: np.ndarray,  # N x 4, where the 4 are x y z class
                                 bev_image_shape: tuple,  # h w
                                 meters_per_pixel: tuple,  # i j
                                 ):

    # x gets mapped to j (where x=0 means j=w/2)
    # y gets mapped to i (where y=0 means i=h/2)
    # z gets ignored
    out = np.zeros((semantic_points.shape[0], 3))
    out[:, 2] = semantic_points[:, 3]
    out[:, 0] = bev_image_shape[0] / 2 - semantic_points[:, 1] / meters_per_pixel[0]
    out[:, 1] = semantic_points[:, 0] / meters_per_pixel[1] + bev_image_shape[1] / 2

    return out


def _calculate_bev_from_projected_points(projected_points: np.ndarray,  # N x 3 where the 3 are i, j, class
                                         num_classes: int = 3
                                         ) -> np.ndarray:
    min_i, min_j = projected_points[:, 0].min(), projected_points[:, 1].min()

    shifted_points = np.copy(projected_points)
    shifted_points[:, :2] = shifted_points[:, :2] - np.asarray([min_i, min_j])
    shifted_points[:, :2] = shifted_points[:, :2].round(0).astype(np.int)

    classes = shifted_points[:, 2].astype(int)

    assert shifted_points[:, :2].min() == 0
    out = np.zeros((num_classes, shifted_points[:, 0].max() + 1, shifted_points[:, 1].max()))
    for class_i in range(num_classes):
        class_points = shifted_points[classes == class_i]
        np.add.at(out[class_i], (class_points[:, 0], class_points[:, 1]), 1)

    return out


def _convolve_point_cloud_bev_with_reward_maps(reward_maps: np.ndarray, bev_point_cloud: np.ndarray) -> np.ndarray:
    ...


def _compute_final_fused_probability(semantic_probability, prior_probability) -> np.ndarray:
    ...


def from_sat_to_ground(
    image_sequence: ...,
    satellite_image: np.ndarray,
) -> np.ndarray:
    # image sequence branch
    semantic_sequence = ...
    point_cloud = ...

    # satellite image branch
    semantic_sattelite_image = ...
    reward_maps = _calculate_reward_maps_from_semantic_sattelite(
        semantic_sattelite_image
    )

    ...
