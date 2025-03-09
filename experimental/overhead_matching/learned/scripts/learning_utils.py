import common.torch.load_torch_deps
import torch
import copy
import torch.nn as nn
import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from experimental.overhead_matching.learned.model import clevr_transformer, clevr_tokenizer
from experimental.overhead_matching.learned.data import clevr_dataset
from common.torch.load_and_save_models import load_model


def sample_ego_from_world(rng, batch_size):
    xy = rng.uniform(-2, 2, size=(batch_size, 2))
    theta = rng.uniform(0, 2 * np.pi, size=(batch_size))

    out = np.zeros((batch_size, 3, 3))
    # x axis
    out[:, 0, 0] = np.cos(theta)
    out[:, 1, 0] = np.sin(theta)
    # y axis
    out[:, 0, 1] = -np.sin(theta)
    out[:, 1, 1] = np.cos(theta)

    # translation
    out[:, :2, 2] = xy
    out[:, 2, 2] = 1.0

    return out


def project_scene_description_to_ego(scene_descriptions, ego_from_world):
    """
    This function returns a batch that has the objects in the ego frame
    It does not currently perform the projective geometry
    """
    out = []
    for idx, scene in enumerate(scene_descriptions):
        out.append(copy.deepcopy(scene))
        for obj in out[-1]:
            old_z = obj["3d_coords"][-1]
            new_coords = ego_from_world[idx] @ np.array(obj["3d_coords"][:2] + [1])
            obj["3d_coords"] = new_coords.tolist()
            obj["3d_coords"][-1] = old_z

    return out


def compute_mse_loss(output, ego_from_world, reduce=True):
    gt_x = torch.from_numpy(ego_from_world[:, 0, 2]).cuda()
    gt_y = torch.from_numpy(ego_from_world[:, 1, 2]).cuda()
    gt_cos_theta = torch.from_numpy(ego_from_world[:, 0, 0]).cuda()
    gt_sin_theta = torch.from_numpy(ego_from_world[:, 1, 0]).cuda()

    pred_x = output[:, 0]
    pred_y = output[:, 1]
    pred_cos_theta = output[:, 2]
    pred_sin_theta = output[:, 3]

    dx = pred_x - gt_x
    dy = pred_y - gt_y
    d_cos_theta = pred_cos_theta - gt_cos_theta
    d_sin_theta = pred_sin_theta - gt_sin_theta

    error = dx * dx + dy * dy + d_cos_theta * d_cos_theta + d_sin_theta * d_sin_theta
    if reduce:
        error = torch.mean(error)

    return error


def gather_clevr_model_performance_from_path(model_checkpoint_path: Path, dataset):
    model = load_model(model_checkpoint_path, skip_constient_output_check=False).cuda()
    return gather_clevr_model_performance(model, dataset)


def construct_clevr_tokens_from_batch(batch: clevr_dataset.CleverDatasetItem,
                                      rng: np.random.Generator):
    ego_from_world = sample_ego_from_world(rng, len(batch.scene_description["objects"]))

    ego_scene_descriptions = project_scene_description_to_ego(
        batch.scene_description["objects"], ego_from_world)
    model_input = clevr_transformer.SceneDescription(
        overhead_image=batch.overhead_image.cuda() if batch.overhead_image is not None else None,
        ego_image=batch.ego_image.cuda() if batch.ego_image is not None else None,
        ego_scene_description=ego_scene_descriptions,
        overhead_scene_description=None,
    )

    return model_input, ego_from_world


def gather_clevr_model_performance(model: nn.Module,
                                   loader: torch.utils.data.DataLoader,
                                   rng: np.random.Generator,
                                   disable_tqdm: bool = False) -> pd.DataFrame:

    pose_source = 'prediction'

    model = model.eval()

    # Collect the predictions
    ego_from_worlds = []
    results = []
    scenes = []
    num_objects = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, total=len(loader), disable=disable_tqdm):
            input_tokens, ego_from_world = construct_clevr_tokens_from_batch(batch, rng)
            ego_from_worlds.append(ego_from_world)
            query_tokens = None
            query_mask = None
            model_outputs = model(input_tokens, query_tokens, query_mask)
            results.append(model_outputs)
            scenes.extend(batch.scene_description['objects'])
            num_objects.extend([len(x) for x in batch.scene_description['objects']])

    results = {
        k: torch.cat([x[k] for x in results]).cpu().numpy()
        for k in results[0]
    }
    ego_from_worlds = np.concatenate(ego_from_worlds)

    # compute the errors
    pred_x = results[pose_source][:, 0]
    pred_y = results[pose_source][:, 1]
    pred_cos = results[pose_source][:, 2]
    pred_sin = results[pose_source][:, 3]
    pred_theta = np.arctan2(pred_sin, pred_cos)

    gt_x = ego_from_worlds[:, 0, 2]
    gt_y = ego_from_worlds[:, 1, 2]
    gt_cos = ego_from_worlds[:, 0, 0]
    gt_sin = ego_from_worlds[:, 1, 0]
    gt_theta = np.arctan2(gt_sin, gt_cos)

    dtheta = pred_theta - gt_theta
    dtheta[dtheta > np.pi] -= 2 * np.pi
    dtheta[dtheta < -np.pi] += 2 * np.pi

    pos_absolute_error = np.abs(pred_x - gt_x) + np.abs(pred_y - gt_y)

    df = pd.DataFrame({
        'pred_x': pred_x,
        'pred_y': pred_y,
        'pred_cos': pred_cos,
        'pred_sin': pred_sin,
        'pred_theta': pred_theta,
        'gt_x': gt_x,
        'gt_y': gt_y,
        'gt_cos': gt_cos,
        'gt_sin': gt_sin,
        'gt_theta': gt_theta,
        'dx': pred_x - gt_x,
        'dy': pred_y - gt_y,
        'dtheta': dtheta,
        'pos_absolute_error': pos_absolute_error,
        'mse': compute_mse_loss(torch.from_numpy(results[pose_source]).cuda(), ego_from_worlds, reduce=False).cpu().numpy(),
        'scene': scenes,
        'num_objects': num_objects,
    })

    return df
