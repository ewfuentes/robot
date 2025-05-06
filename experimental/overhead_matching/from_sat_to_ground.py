import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torchvision.transforms.functional import rotate, InterpolationMode


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
    shifted_points[:, :2] = shifted_points[:, :2].round(0)
    shifted_points = shifted_points.astype(int)

    assert shifted_points[:, :2].min() == 0
    out = np.zeros((num_classes, shifted_points[:, 0].max(
    ) + 1, shifted_points[:, 1].max() + 1), dtype=int)
    for class_i in range(num_classes):
        class_points = shifted_points[shifted_points[:, 2] == class_i]
        np.add.at(out[class_i], (class_points[:, 0], class_points[:, 1]), 1)

    return out


def _add_orientation_filters(semantic_bev: torch.LongTensor, n_heading_bins: int) -> torch.tensor:
    assert semantic_bev.ndim == 3 and not semantic_bev.dtype.is_floating_point
    n_classes = semantic_bev.shape[0]
    rotation_amount_deg = torch.linspace(0, 360.0, n_heading_bins+1)[:-1]

    max_shape = (0, 0)
    for angle in rotation_amount_deg:
        candidate = rotate(semantic_bev, angle.item(),
                           interpolation=InterpolationMode.NEAREST, expand=True, fill=0).shape[1:]
        max_shape = (max(max_shape[0], candidate[0]), max(max_shape[1], candidate[1]))
    i_delta = max_shape[0] - semantic_bev.shape[1]
    j_delta = max_shape[1] - semantic_bev.shape[2]
    padded_input = torch.zeros((n_classes, max_shape[0], max_shape[1]), dtype=torch.long)

    padded_input[:,
                 i_delta // 2: i_delta // 2 + semantic_bev.shape[1],
                 j_delta // 2: j_delta // 2 + semantic_bev.shape[2]] = semantic_bev
    output = torch.zeros((n_heading_bins, n_classes, *max_shape), dtype=torch.long)

    for i, theta in enumerate(rotation_amount_deg):
        output[i] = rotate(padded_input, theta.item(),
                           interpolation=InterpolationMode.NEAREST, expand=False, fill=0)

    return output


def _convolve_point_cloud_bev_with_reward_maps(reward_map: np.ndarray,
                                               penalty_map: np.ndarray,
                                               bev_point_cloud: np.ndarray,
                                               gamma: float = 0.5,
                                               num_orientation_bins: int = 128,
                                               device='cpu') -> np.ndarray:

    tensor_bev = torch.from_numpy(bev_point_cloud)
    orientation_expanded_bev = _add_orientation_filters(tensor_bev, num_orientation_bins)

    orientation_expanded_bev = orientation_expanded_bev.to(device).to(float)
    tensor_reward = torch.from_numpy(reward_map).to(device)
    tensor_penalty = torch.from_numpy(penalty_map).to(device)


    with torch.no_grad():
        reward_map = torch.nn.functional.conv2d(
            tensor_reward, orientation_expanded_bev, bias=None, padding="same")
        penalty_map = torch.nn.functional.conv2d(
            tensor_penalty, orientation_expanded_bev, bias=None, padding="same")

    unnormalized_probaility_map = reward_map + gamma * penalty_map

    return unnormalized_probaility_map.cpu().numpy()


def from_sat_to_ground(
    semantic_pointcloud: np.ndarray,  # N x (x y z class) -> building (0) vegetation (1) road (2)
    semantic_satellite: np.ndarray,  # h x w -> unclassified (0) building(1) vegetation (2) road(3)
    sigma: float,
    gamma: float = 0.5,
    num_orientation_bins: int = 128,
    num_classes: int = 3,
    # meters per pixel in i and j directions
    meters_per_pixel_satellite: tuple = (0.309186, 0.309186),
    device = "cpu"
) -> np.ndarray:

    shifted_semantic_satellite = semantic_satellite - 1  # function checks for classes 0, 1, 2
    rew_map, penalty_map = _calculate_reward_penalty_maps_from_semantic_satellite(
        shifted_semantic_satellite,
        sigma,
        num_classes,
    )

    projected_points = _project_pointcloud_into_bev(
        semantic_pointcloud, semantic_satellite.shape, meters_per_pixel_satellite)
    semantic_bev = _calculate_bev_from_projected_points(projected_points, num_classes)
    unnormalized_distribution = _convolve_point_cloud_bev_with_reward_maps(
        rew_map, penalty_map, semantic_bev, gamma, num_orientation_bins, device)

    return unnormalized_distribution

