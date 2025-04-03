
import common.torch.load_torch_deps
import torch

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt


def load_satellite_metadata(path: Path):
    out = []
    for p in path.iterdir():
        _, lat, lon = p.stem.split('_')
        out.append((float(lat), float(lon), p))
    return pd.DataFrame(out, columns=["lat", "lon", "path"])


def load_panorama_metadata(path: Path):
    out = []
    for p in path.iterdir():
        pano_id, lat, lon, _ = p.stem.split(',')
        out.append((pano_id, float(lat), float(lon), p))
    return pd.DataFrame(out, columns=["pano_id", "lat", "lon", "path"])


class VigorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path):
        super().__init__()
        self._dataset_path = dataset_path

        satellite_metadata = load_satellite_metadata(dataset_path / "satellite")
        panorama_metadata = load_panorama_metadata(dataset_path / "panorama")

        
