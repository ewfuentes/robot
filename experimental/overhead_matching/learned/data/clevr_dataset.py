import common.torch.load_torch_deps
import torch
import torchvision.transforms.v2 as tf
from torchvision.io import read_image, ImageReadMode

from pathlib import Path
import json
from collections import defaultdict

from typing import Callable, NamedTuple

DEFAULT_CLEVR_OVERHEAD_TRANSFORM = torch.nn.Sequential(
    tf.ToImage(),
    tf.ToDtype(torch.float32, scale=True),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)
DEFAULT_CLEVR_EGO_TRANSFORM = torch.nn.Sequential(
    tf.ToImage(),
    tf.ToDtype(torch.float32, scale=True),
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
)


class CleverDatasetItem(NamedTuple):
    scene_description: dict | None = None
    overhead_image: torch.Tensor | None = None
    ego_image: torch.Tensor | None = None

    def __repr__(self):
        return f"ClevrDatasetItem(scene_description: {self.scene_description}, overhead_image: {self.overhead_image}, ego_image: {self.ego_image})"


def clevr_dataset_collator(objs: list[CleverDatasetItem]):
    out = {}
    first_obj = objs[0]
    if first_obj.scene_description is not None:
        collated_scene_descriptions = {}
        for key in first_obj.scene_description:
            collated_scene_descriptions[key] = []
            for obj in objs:
                collated_scene_descriptions[key].append(obj.scene_description[key])
                if collated_scene_descriptions[key][-1] is None:
                    raise RuntimeError(
                        "Encountered some scene descriptions batch items with None and some with values while collating")
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
