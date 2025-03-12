import argparse

from pathlib import Path

import common.torch.load_torch_deps
import shutil
from common.torch.load_and_save_models import save_model, load_model
import experimental.overhead_matching.learned.scripts.learning_utils as lu
import torch
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
import json
from dataclasses import dataclass, asdict
from experimental.overhead_matching.learned.data import clevr_dataset
from experimental.overhead_matching.learned.model import (
    clevr_tokenizer,
    clevr_transformer,
)
import enum


class LossFunctions(str, enum.Enum):
    POSE_MSE = "pose_mse"
    CORRESPONDENCE_LOSS = "correspondence_loss"


@dataclass
class TrainConfig:
    model_config: clevr_transformer.ClevrTransformerConfig
    # ego_image, overhead_image, ego_vectorized, overhead_vectorized
    model_inputs: list[lu.ModelInputs]
    loss_functions: list[LossFunctions]  # pose_mse, correspondence_loss
    num_epochs: int

    def to_file(self, file_path: Path):
        """Save the training configuration to a file."""

        # Convert model_config to dict
        config_dict = asdict(self)

        # Handle the model_config specially since it's a custom class
        config_dict['model_config'] = asdict(self.model_config)

        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_file(cls, file_path: Path):
        """Load a training configuration from a file."""
        import json

        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct the model_config from the dict
        model_config = clevr_transformer.ClevrTransformerConfig(**config_dict['model_config'])

        # Create and return the TrainConfig
        return cls(
            model_config=model_config,
            model_inputs=[lu.ModelInputs(x) for x in config_dict['model_inputs']],
            loss_functions=[LossFunctions(x) for x in config_dict['loss_functions']],
            num_epochs=config_dict['num_epochs']
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


def main(dataset_path: Path, output_path: Path, load_model_path: Path | None, train_config: TrainConfig):

    if output_path.exists():
        del_output_path = input(f"Output directory {output_path} exists. Delete (y/n): ")
        if del_output_path == "y":
            shutil.rmtree(output_path)

    output_path.mkdir(exist_ok=True, parents=True)

    train_config.to_file(output_path / "train_config.json")

    with (output_path / "dataset_path.txt").open("w") as f:
        f.write(str(dataset_path))

    writer = SummaryWriter(
        log_dir=output_path
    )
    torch.manual_seed(2048)
    dataset = clevr_dataset.ClevrDataset(dataset_path,
                                         load_overhead='overhead_image' in train_config.model_inputs,
                                         load_ego_images='ego_image' in train_config.model_inputs,
                                         )
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(1023))
    train_performance_subset = torch.utils.data.Subset(
        train_dataset, torch.arange(0, len(train_dataset) // 10))

    loader = clevr_dataset.get_dataloader(
        train_dataset, batch_size=128, num_workers=12, persistent_workers=True)
    val_loader = clevr_dataset.get_dataloader(val_dataset, batch_size=128, num_workers=12)
    train_performance_subset_loader = clevr_dataset.get_dataloader(
        train_performance_subset, batch_size=128, num_workers=12)

    best_model_weights = None
    best_val_mse = np.inf
    best_model_epoch = None

    if load_model_path is not None:
        model = load_model(load_model_path, skip_constient_output_check=False)
    else:
        vocabulary = dataset.vocabulary()
        model = clevr_transformer.ClevrTransformer(train_config.model_config, vocabulary)

    model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    rng = np.random.default_rng(1024)
    try:
        for epoch_idx in range(train_config.num_epochs):
            all_losses = []
            model.train()
            for batch in loader:
                # print(batch["objects"])
                optim.zero_grad()
                # sample an ego pose
                input_tokens, ego_from_world = lu.construct_clevr_tokens_from_batch(
                    batch, train_config.model_inputs, rng)
                query_tokens = create_query_tokens(len(batch))
                query_mask = None

                # print("input", model_input.ego_scene_description[0], model_input.overhead_image[0].sum())
                output = model(input_tokens, query_tokens, query_mask)

                # torch.save({'in': model_input, 'out': output}, "/tmp/inout.tar")
                # print("model", output['prediction'][0])
                # assert False
                loss = 0.0
                if LossFunctions.POSE_MSE in train_config.loss_functions:
                    loss += lu.compute_mse_loss(output["prediction"], ego_from_world, reduce=False)

                if LossFunctions.CORRESPONDENCE_LOSS in train_config.loss_functions:
                    loss += compute_correspondence_loss(
                        output["learned_correspondence"],
                        input.overhead_position, input.ego_position, ego_from_world)

                all_losses.append(loss.detach().cpu().numpy())
                loss = torch.mean(loss)
                loss.backward()
                optim.step()

            all_losses = np.concatenate(all_losses, axis=0)
            writer.add_scalar("loss/train", np.mean(all_losses), epoch_idx)

            print(f"***** Epoch: {epoch_idx}", end=" ")
            print(f"train loss: {np.mean(all_losses)}", end=" ")

            model.eval()
            eval_rng = np.random.default_rng(2048)

            val_df = lu.gather_clevr_model_performance(
                model, val_loader, train_config.model_inputs, eval_rng, disable_tqdm=True)

            writer.add_scalar("loss/val_mae", val_df['pos_absolute_error'].mean(), epoch_idx)
            writer.add_scalar("loss/val_mse", val_df['mse'].mean(), epoch_idx)
            val_mse = val_df['mse'].mean()
            print(f"val mse: {val_mse}")
            if np.mean(val_mse) < best_val_mse:
                best_model_weights = copy.deepcopy(model.state_dict())
                best_model_epoch = epoch_idx
                best_val_mse = val_mse

            # correspondences_output_path = output_path / "intermediates" / "correspondences" / f"{epoch_idx:06d}.png"
            # correspondences_output_path.parent.mkdir(parents=True, exist_ok=True)
            # tv.utils.save_image(correspondences.unsqueeze(1), correspondences_output_path)

            if epoch_idx % 25 == 0:
                model_save_path = output_path / f"epoch_{epoch_idx:06d}"
                output_path.mkdir(parents=True, exist_ok=True)
                save_model(model, model_save_path, (input_tokens, None, None))
                print("model saved to:", output_path / f"epoch_{epoch_idx:06d}")
                eval_rng = np.random.default_rng(2048)
                train_performance = lu.gather_clevr_model_performance(
                    model, train_performance_subset_loader, train_config.model_inputs, eval_rng, disable_tqdm=True
                )
                train_performance.to_csv(model_save_path / "train_performance.csv")
                val_df.to_csv(model_save_path / "val_performance.csv")

    except KeyboardInterrupt:
        print("Exiting (got keyboard interrupt)")

    model.load_state_dict(best_model_weights)
    save_model(model, output_path / 'best', (input_tokens, None, None), {'epoch': best_model_epoch})
    print("saved best model captured on epoch", best_model_epoch)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train_config_path", required=True)
    parser.add_argument("--load_model_path", required=False, default=None,
                        type=Path, help="Path to load initial model from")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--delete_output_if_exists", action='store_true')

    args = parser.parse_args()

    if args.delete_output_if_exists:
        if Path(args.output_path).exists():
            shutil.rmtree(args.output_path)

    args.load_model_path = None if args.load_model_path is None else Path(args.load_model_path)
    if args.load_model_path is not None:
        print("Warning: as load model path was provided the model config in train_config_path will be ignored")

    train_config = TrainConfig.from_file(args.train_config_path)

    main(Path(args.dataset), Path(args.output_path), args.load_model_path, train_config)
