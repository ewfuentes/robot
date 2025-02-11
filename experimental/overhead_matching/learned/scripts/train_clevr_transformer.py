import argparse

from pathlib import Path

import common.torch as torch
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

    # theta = theta * 0.0
    # xy = 0.0 * xy

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


def project_batch_to_ego(batch, ego_from_world):
    """
    This function returns a batch that has the objects in the ego frame
    It does not currently perform the projective geometry
    """
    out = []
    for idx, scene in enumerate(batch):
        out.append(copy.deepcopy(scene))
        for obj in out[-1]:
            old_z = obj["3d_coords"][-1]
            new_coords = ego_from_world[idx] @ np.array(obj["3d_coords"][:2] + [1])
            obj["3d_coords"] = new_coords.tolist()
            obj["3d_coords"][-1] = old_z

    return out


def clevr_input_from_batch(batch, vocabulary, embedding_size, ego_from_world):
    result = clevr_tokenizer.create_tokens(batch, vocabulary)
    position_embeddings = clevr_tokenizer.create_position_embeddings(
        batch, embedding_size=embedding_size, min_scale=1e-6
    )

    ego_batch = project_batch_to_ego(batch, ego_from_world)
    ego_result = clevr_tokenizer.create_tokens(ego_batch, vocabulary)
    ego_position_embeddings = clevr_tokenizer.create_position_embeddings(
        ego_batch, embedding_size=embedding_size, min_scale=1e-6
    )

    return clevr_transformer.ClevrInputTokens(
        overhead_tokens=result["tokens"].cuda(),
        overhead_position=position_embeddings.cuda(),
        overhead_mask=result["mask"].cuda(),
        ego_tokens=ego_result["tokens"].cuda(),
        ego_position=ego_position_embeddings.cuda(),
        ego_mask=ego_result["mask"].cuda(),
    )


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
    pred_sin_theta = output[:, 2]

    dx = pred_x - gt_x
    dy = pred_y - gt_y
    d_cos_theta = pred_cos_theta - gt_cos_theta
    d_sin_theta = pred_sin_theta - gt_sin_theta

    error = torch.mean(
        dx * dx + dy * dy + d_cos_theta * d_cos_theta + d_sin_theta * d_sin_theta
    )

    return error


def main(dataset_path: Path, output_path: Path):
    torch.manual_seed(2048)
    dataset = clevr_dataset.ClevrDataset(dataset_path)
    vocabulary = dataset.vocabulary()
    loader = clevr_dataset.get_dataloader(dataset, batch_size=64)

    vocabulary_size = int(np.prod([len(x) for x in vocabulary.values()]))

    TOKEN_SIZE = 128
    OUTPUT_DIM = 4
    model_config = clevr_transformer.ClevrTransformerConfig(
        token_dim=TOKEN_SIZE,
        vocabulary_size=vocabulary_size,
        num_encoder_heads=4,
        num_encoder_layers=8,
        num_decoder_heads=4,
        num_decoder_layers=8,
        output_dim=OUTPUT_DIM,
        predict_gaussian=True,
    )

    model = clevr_transformer.ClevrTransformer(model_config)
    model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    NUM_EPOCHS = 1001
    rng = np.random.default_rng(1024)
    for epoch_idx in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            # print(batch["objects"])
            optim.zero_grad()
            # sample an ego pose
            ego_from_world = sample_ego_from_world(rng, len(batch["objects"]))
            # print(ego_from_world)
            input = clevr_input_from_batch(
                batch["objects"], vocabulary, TOKEN_SIZE, ego_from_world
            )
            query_tokens = create_query_tokens(len(batch))
            query_mask = None

            output = model(input, query_tokens, query_mask)

            loss = compute_mse_loss(output, ego_from_world)
            loss.backward()
            optim.step()

        print(f"***** Epoch: {epoch_idx}", end=" ")
        model.eval()
        eval_rng = np.random.default_rng(1024)
        batch = next(iter(loader))
        ego_from_world = sample_ego_from_world(eval_rng, len(batch["objects"]))
        input = clevr_input_from_batch(
            batch["objects"], vocabulary, TOKEN_SIZE, ego_from_world
        )

        output = model(input, None, None)

        x_pred = output[:, :2].cpu()
        x_gt = torch.from_numpy(ego_from_world[:, :2, 2])
        error = x_pred - x_gt
        error = torch.sqrt(torch.sum(error * error, axis=1))
        print("mae:", torch.mean(error).item())

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
