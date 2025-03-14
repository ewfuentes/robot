import common.torch.load_torch_deps
import torch
import torchvision.transforms.v2 as tf
from torchvision.io import read_image, ImageReadMode

from pathlib import Path
import json
import numpy as np
from collections import defaultdict

from typing import Callable, NamedTuple

DEFAULT_CLEVR_OVERHEAD_TRANSFORM = torch.nn.Sequential(
    tf.ToImage(),
    tf.ToDtype(torch.float32, scale=True),
)
DEFAULT_CLEVR_EGO_TRANSFORM = torch.nn.Sequential(
    tf.ToImage(),
    tf.ToDtype(torch.float32, scale=True),
)

CLEVR_CAMERA_INTRINSICS = np.asarray([(350.0000,   0.0000, 160.0000),
                                      (0.0000, 350.0000, 120.0000),
                                      (0.0000,   0.0000,  1.0000)])
CLEVR_OVERHEAD_Z_METERS = 14


def project_overhead_pixels_to_ground_plane(overhead_pixels_uv: np.ndarray | torch.TensorType):
    if torch.is_tensor(overhead_pixels_uv):
        library = torch
        cam_intrinsics = torch.from_numpy(CLEVR_CAMERA_INTRINSICS).to(torch.float32)
    else:
        library = np
        cam_intrinsics = CLEVR_CAMERA_INTRINSICS
    ## overhead_pixels_uv is shape 2 x N
    assert overhead_pixels_uv.shape[0] == 2 and len(overhead_pixels_uv.shape) == 2
    n_points = overhead_pixels_uv.shape[1]
    overhead_pixels_uv = library.concatenate([overhead_pixels_uv, library.ones((1, n_points))], axis=0)
    points_in_world = library.linalg.inv(
        cam_intrinsics) @ overhead_pixels_uv * CLEVR_OVERHEAD_Z_METERS
    points_in_world[1, :] *= -1  # flip Y value to match reference frame
    points_in_world[2, :] = 0  # zero out z values
    return points_in_world


class CleverDatasetItem(NamedTuple):
    scene_description: dict | list | None = None
    overhead_image: torch.Tensor | None = None
    ego_image: torch.Tensor | None = None

    def __repr__(self):
        return f'ClevrDatasetItem(scene_description: {str(len(self.scene_description)) + " items" if self.scene_description is not None else "None"}, overhead_image: {self.overhead_image.shape if torch.is_tensor(self.overhead_image) else "None"}, ego_image: {self.ego_image.shape if torch.is_tensor(self.ego_image) else "None"})'

    def __len__(self):
        length = 0
        if self.scene_description is not None:
            # it is a list of lists of objects
            if isinstance(self.scene_description['objects'][0], list):
                length = len(self.scene_description['objects'])
            # it is a list of objects, we have one scene
            elif isinstance(self.scene_description['objects'][0], dict):
                length = 1
            else:
                raise RuntimeError(
                    f"Got a scene description with an unrecognized type: {type(self.scene_description)}")
        if self.overhead_image is not None:
            if len(self.overhead_image.shape) == 4:
                overhead_length = self.overhead_image.shape[0]
            elif len(self.overhead_image.shape) == 3:
                overhead_length = 1
            else:
                raise RuntimeError(
                    f"Got unexpected shape for overhead image: {self.overhead_image.shape}")

            if length != 0:
                assert length == overhead_length, "Got different lengths for overhead and scene length"
            else:
                length = overhead_length
        if self.ego_image is not None:
            if len(self.ego_image.shape) == 4:
                ego_length = self.ego_image.shape[0]
            elif len(self.ego_image.shape) == 3:
                ego_length = 1
            else:
                raise RuntimeError(f"Got unexpected shape for ego image: {self.ego_image.shape}")

            if length != 0:
                assert length == ego_length, "Got different lengths for ego and scene or overhead length"
            else:
                length = ego_length
        return length


def clevr_dataset_collator(objs: list[CleverDatasetItem]):
    out = {}
    first_obj = objs[0]
    if first_obj.scene_description is not None:
        collated_scene_descriptions = {}
        for key in first_obj.scene_description:
            collated_scene_descriptions[key] = []
            for obj in objs:
                if obj.scene_description is None:
                    raise RuntimeError(
                        "Encountered some scene descriptions batch items with None and some with values while collating")
                collated_scene_descriptions[key].append(obj.scene_description[key])
        out['scene_description'] = collated_scene_descriptions

    if first_obj.overhead_image is not None:
        overhead_tensor_stack = []
        for obj in objs:
            overhead_tensor_stack.append(obj.overhead_image)
            if overhead_tensor_stack[-1] is None:
                raise RuntimeError(
                    "Encountered some overhead batch items with None and some with values while collating")
        overhead_tensor_stack = torch.stack(overhead_tensor_stack, dim=0)
        out['overhead_image'] = overhead_tensor_stack

    if first_obj.ego_image is not None:
        ego_tensor_stack = []
        for obj in objs:
            ego_tensor_stack.append(obj.ego_image)
            if ego_tensor_stack[-1] is None:
                raise RuntimeError(
                    "Encountered some ego image batch items with None and some with values while collating")
        ego_tensor_stack = torch.stack(ego_tensor_stack, dim=0)
        out['ego_image'] = ego_tensor_stack

    return CleverDatasetItem(**out)


def get_dataloader(dataset, **kwargs) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, collate_fn=clevr_dataset_collator, **kwargs)


class ClevrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path,
        transform: None | Callable[[dict], dict] = None,
        load_overhead: bool = False,
        overhead_transform: None | tf.Transform = None,
        load_ego_images: bool = False,
        ego_transform: None | tf.Transform = None
    ):
        scenes_json = dataset_path / "CLEVR_scenes.json"
        with scenes_json.open() as file_in:
            scene_data = json.load(file_in)

        self._transform = transform
        self._json = scene_data["scenes"]

        self.load_overhead = load_overhead
        self.overhead_transform = overhead_transform if overhead_transform is not None else DEFAULT_CLEVR_OVERHEAD_TRANSFORM
        if self.load_overhead:
            self.overhead_file_paths = sorted((dataset_path / "images").glob("overhead_*.png"))
            assert len(self.overhead_file_paths) == len(
                self._json), "Number of overhead images and number of scenes do not match"

        self.load_ego_images = load_ego_images
        self.ego_transform = ego_transform if ego_transform is not None else DEFAULT_CLEVR_EGO_TRANSFORM
        if self.load_ego_images:
            self.ego_filepaths = sorted((dataset_path / "images").glob("CLEVR_new_*.png"))
            assert len(self.ego_filepaths) == len(
                self._json), "Number of ego images and number of scenes do not match"

        # make sure things are in the correct order 
        for i, scene in enumerate(self._json):
            def get_scene_number(path):
                return path.stem.split("_")[-1]
            scene_number = get_scene_number(Path(scene["image_filename"]))
            if self.load_overhead:
                assert get_scene_number(self.overhead_file_paths[i]) == scene_number
            if self.load_ego_images:
                assert get_scene_number(self.ego_filepaths[i]) == scene_number

    def __len__(self) -> int:
        return len(self._json)

    def __getitem__(self, idx) -> dict:
        item = self._json[idx]
        item = self._transform(item) if self._transform else item
        output = dict(
            scene_description=item
        )

        if self.load_overhead:
            overhead_image = read_image(self.overhead_file_paths[idx], mode=ImageReadMode.RGB)
            if self.overhead_transform:
                overhead_image = self.overhead_transform(overhead_image)
            output['overhead_image'] = overhead_image

        if self.load_ego_images:
            ego_image = read_image(self.ego_filepaths[idx], mode=ImageReadMode.RGB)
            if self.ego_transform:
                ego_image = self.ego_transform(ego_image)
            output['ego_image'] = ego_image

        return CleverDatasetItem(**output)

    def vocabulary(self):
        out = defaultdict(set)
        for scene in self._json:
            for obj in scene["objects"]:
                out["color"].add(obj["color"])
                out["material"].add(obj["material"])
                out["size"].add(obj["size"])
                out["shape"].add(obj["shape"])
        return {key: sorted(list(value)) for key, value in out.items()}
