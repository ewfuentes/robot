import common.torch as torch

from pathlib import Path
import json
from collections import defaultdict

from typing import Callable


def collator(objs):
    out = {}
    for key in objs[0]:
        out[key] = [obj[key] for obj in objs]
    return out


def get_dataloader(dataset, **kwargs) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, collate_fn=collator, **kwargs)


class ClevrDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path: Path, transform: None | Callable[[dict], dict] = None
    ):
        scenes_json = dataset_path / "CLEVR_scenes.json"
        with scenes_json.open() as file_in:
            scene_data = json.load(file_in)

        self._transform = transform
        self._json = scene_data["scenes"]

    def __len__(self) -> int:
        return len(self._json)

    def __getitem__(self, idx) -> dict:
        item = self._json[idx]

        return self._transform(item) if self._transform else item

    def vocabulary(self):
        out = defaultdict(set)
        for scene in self._json:
            for obj in scene["objects"]:
                out["color"].add(obj["color"])
                out["material"].add(obj["material"])
                out["size"].add(obj["size"])
                out["shape"].add(obj["shape"])
        return {key: sorted(list(value)) for key, value in out.items()}
