import argparse

from pathlib import Path

import common.torch.load_torch_deps
import torch
import torchvision as tv
import numpy as np
import copy

from experimental.overhead_matching.learned.data import clevr_dataset
from experimental.overhead_matching.learned.model import (
    clevr_tokenizer,
    clevr_transformer,
)


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


def create_query_tokens(batch_size: int):
    return None


def compute_mse_loss(output, ego_from_world):
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

    error = torch.mean(
        dx * dx + dy * dy + d_cos_theta * d_cos_theta + d_sin_theta * d_sin_theta
    )

    return error


def compute_correspondence_loss(correspondence, obj_in_overhead, obj_in_ego, ego_from_world):
    """
    provide signal on the learned correspondences by computing the reprojection error
    using the ground truth pose
    """
    batch_size, num_objects, _ = obj_in_overhead.shape
    ego_from_world = torch.tensor(
        ego_from_world, device=obj_in_overhead.device, dtype=torch.float32)

    ones = torch.ones((batch_size, num_objects, 1), device=obj_in_overhead.device)
    obj_in_overhead = torch.concatenate([obj_in_overhead, ones], dim=-1)
    obj_in_ego = torch.concatenate([obj_in_ego, ones], dim=-1)
    projected_obj_in_ego = torch.bmm(
        ego_from_world, obj_in_overhead.transpose(-1, -2)).transpose(-1, -2)

    projected_obj_in_ego = projected_obj_in_ego.unsqueeze(-2)
    obj_in_ego = obj_in_ego.unsqueeze(-3)
    projection_errors = projected_obj_in_ego - obj_in_ego
    projection_errors = torch.sum(projection_errors ** 2, dim=-1)
    weighted_projection_errors = correspondence[..., :-1] * projection_errors
    # Note that this reduces the losses on scenes with fewer objects more heavily since
    # the invalid/unused pairings are included in the mean count
    mean_scene_errors = torch.mean(weighted_projection_errors, dim=-1)
    return torch.mean(mean_scene_errors)


def main(dataset_path: Path, output_path: Path):
    torch.manual_seed(2048)
    dataset = clevr_dataset.ClevrDataset(dataset_path)
    vocabulary = dataset.vocabulary()
    loader = clevr_dataset.get_dataloader(dataset, batch_size=64, num_workers=12)

    TOKEN_SIZE = 128
    OUTPUT_DIM = 4
    model_config = clevr_transformer.ClevrTransformerConfig(
        token_dim=TOKEN_SIZE,
        num_encoder_heads=4,
        num_encoder_layers=8,
        num_decoder_heads=4,
        num_decoder_layers=8,
        output_dim=OUTPUT_DIM,
        inference_method=clevr_transformer.InferenceMethod.MEAN,
        ego_image_tokenizer_config=None,
        overhead_image_tokenizer_config=clevr_tokenizer.ImageToTokensConfig(
            TOKEN_SIZE, (240, 320), 16),
    )

    model = clevr_transformer.ClevrTransformer(model_config, vocabulary)
    model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    NUM_EPOCHS = 1001
    rng = np.random.default_rng(1024)
    for epoch_idx in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            # print(batch["objects"])
            optim.zero_grad()
            # sample an ego pose
            ego_from_world = sample_ego_from_world(rng, len(batch.scene_description["objects"]))

            ego_scene_descriptions = project_scene_description_to_ego(
                batch.scene_description["objects"], ego_from_world)
            # print(ego_from_world)
            input = clevr_transformer.SceneDescription(
                overhead_image=batch.overhead_image,
                ego_image=None,
                ego_scene_description=ego_scene_descriptions,
                overhead_scene_description=None,
            )

            query_tokens = create_query_tokens(len(batch))
            query_mask = None

            output = model(input, query_tokens, query_mask)

            loss = compute_mse_loss(output["prediction"], ego_from_world)
            # TODO add a flag to select what kind of loss we need
            # loss += compute_correspondence_loss(
            #         output["learned_correspondence"],
            #         input.overhead_position, input.ego_position, ego_from_world)
            loss.backward()
            optim.step()

        print(f"***** Epoch: {epoch_idx}", end=" ")
        model.eval()
        eval_rng = np.random.default_rng(1024)
        batch = next(iter(loader))
        ego_from_world = sample_ego_from_world(eval_rng, len(batch.scene_description["objects"]))
        ego_scene_descriptions = project_scene_description_to_ego(
            batch.scene_description["objects"], ego_from_world)

        input = clevr_transformer.SceneDescription(
            overhead_image=batch.overhead_image,
            ego_image=None,
            ego_scene_description=ego_scene_descriptions,
            overhead_scene_description=None,
        )

        output = model(input, None, None)

        decoder_output = output["prediction"]
        # correspondences = output["learned_correspondence"]

        x_pred = decoder_output[:, :2].cpu()
        x_gt = torch.from_numpy(ego_from_world[:, :2, 2])
        error = x_pred - x_gt
        error = torch.sqrt(torch.sum(error * error, axis=1))

        # correspondence_loss = compute_correspondence_loss(
        #         correspondences, input.overhead_position, input.ego_position, ego_from_world)
        # , "correspondence_loss:", correspondence_loss.item())
        print("mae:", torch.mean(error).item())

        # correspondences_output_path = output_path / "intermediates" / "correspondences" / f"{epoch_idx:06d}.png"
        # correspondences_output_path.parent.mkdir(parents=True, exist_ok=True)
        # tv.utils.save_image(correspondences.unsqueeze(1), correspondences_output_path)

        model.train()

        if epoch_idx % 10 == 0:
            output_path.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_path / f"{epoch_idx:06d}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to:", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    main(Path(args.dataset), Path(args.output_path))
