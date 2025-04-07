import common.torch.load_torch_deps
import torch
import torchvision as tv

from pathlib import Path
import pandas as pd
from scipy.spatial import cKDTree
from typing import NamedTuple


class VigorDatasetItem(NamedTuple):
    panorama_metadata: dict
    satellite_metadata: dict
    panorama: torch.Tensor
    satellite: torch.Tensor

def series_to_dict_with_index(series: pd.Series, index_key: str = "index"):
    d = series.to_dict()
    assert index_key not in d 
    d[index_key] = series.name
    return d

def load_satellite_metadata(path: Path):
    out = []
    for p in sorted(list(path.iterdir())):
        _, lat, lon = p.stem.split("_")
        out.append((float(lat), float(lon), p))
    return pd.DataFrame(out, columns=["lat", "lon", "path"])


def load_panorama_metadata(path: Path):
    out = []
    for p in sorted(list(path.iterdir())):
        pano_id, lat, lon, _ = p.stem.split(",")
        out.append((pano_id, float(lat), float(lon), p))
    return pd.DataFrame(out, columns=["pano_id", "lat", "lon", "path"])


def compute_satellite_from_panorama(sat_kdtree, pano_metadata):
    _, sat_idx_from_pano_idx = sat_kdtree.query(pano_metadata.loc[:, ["lat", "lon"]].values)

    pano_idxs_from_sat_idx = [[] for i in range(sat_kdtree.n)]
    for pano_idx, sat_idx in enumerate(sat_idx_from_pano_idx):
        pano_idxs_from_sat_idx[sat_idx].append(pano_idx)

    return sat_idx_from_pano_idx, pano_idxs_from_sat_idx

def compute_neighboring_panoramas(pano_kdtree, max_dist):
    pairs = pano_kdtree.query_pairs(max_dist)
    neighbors_by_pano_idx = [[] for i in range(pano_kdtree.n)]
    for (a, b) in pairs:
        neighbors_by_pano_idx[a].append(b)
        neighbors_by_pano_idx[b].append(a)
    return neighbors_by_pano_idx


class VigorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, panorama_neighbor_radius: float):
        super().__init__()
        self._dataset_path = dataset_path

        self._satellite_metadata = load_satellite_metadata(dataset_path / "satellite")
        self._panorama_metadata = load_panorama_metadata(dataset_path / "panorama")

        self._satellite_kdtree = cKDTree(self._satellite_metadata.loc[:, ["lat", "lon"]].values)
        self._panorama_kdtree = cKDTree(self._panorama_metadata.loc[:, ["lat", "lon"]].values)

        sat_idx_from_pano_idx, pano_idxs_from_sat_idx = compute_satellite_from_panorama(
                self._satellite_kdtree, self._panorama_metadata)

        self._satellite_metadata["panorama_idxs"] = pano_idxs_from_sat_idx
        self._panorama_metadata["satellite_idx"] = sat_idx_from_pano_idx

        self._panorama_metadata["neighbor_panorama_idxs"] = compute_neighboring_panoramas(
                self._panorama_kdtree, panorama_neighbor_radius)
        
    @property
    def num_satellite_patches(self):
        return len(self._satellite_metadata)

    def __getitem__(self, idx):
        pano_metadata = self._panorama_metadata.loc[idx]
        sat_metadata = self._satellite_metadata.loc[pano_metadata.satellite_idx]

        pano = tv.io.read_image(pano_metadata.path)
        sat = tv.io.read_image(sat_metadata.path)

        return VigorDatasetItem(
            panorama_metadata=series_to_dict_with_index(pano_metadata),
            satellite_metadata=series_to_dict_with_index(sat_metadata),
            panorama=pano,
            satellite=sat
        )

    def __len__(self):
        return len(self._panorama_metadata)
    
    def get_sat_patch_view(self)->torch.utils.data.Dataset:
        class OverheadVigorDataset(torch.utils.data.Dataset):
            def __init__(self, dataset: VigorDataset):
                super().__init__()
                self.dataset = dataset 
            def __len__(self):
                return len(self.dataset._satellite_metadata)
            def __getitem__(self, idx):
                if idx > len(self) - 1:
                    raise IndexError  # if we don't raise index error the iterator won't terminate
                sat_metadata = self.dataset._satellite_metadata.loc[idx]  # as this will throw a KeyError
                sat = tv.io.read_image(sat_metadata.path)
                return VigorDatasetItem(
                    None, 
                    series_to_dict_with_index(sat_metadata),
                    None, 
                    sat
                )
        return OverheadVigorDataset(self)


    def visualize(self, include_text_labels=False):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        plt.figure()
        ax = plt.subplot(111)
        sat_segments = []
        neighbor_segments = []
        for pano_idx, pano_meta in self._panorama_metadata.iterrows():
            sat_meta = self._satellite_metadata.loc[pano_meta.satellite_idx]
            sat_segments.append([(pano_meta.lon, pano_meta.lat), (sat_meta.lon, sat_meta.lat)])

            for neighbor_idx in pano_meta.neighbor_panorama_idxs:
                neighbor_meta = self._panorama_metadata.loc[neighbor_idx]
                if pano_idx < neighbor_idx:
                    neighbor_segments.append([(pano_meta.lon, pano_meta.lat),
                                              (neighbor_meta.lon, neighbor_meta.lat)])

        sat_collection = LineCollection(sat_segments,
                                        colors=[(0.9, 0.25, 0.25) for x in range(len(sat_segments))])
        neighbor_collection = LineCollection(neighbor_segments,
                                             colors=[(0.25, 0.9, 0.25) for x in range(len(neighbor_segments))])
        ax.add_collection(sat_collection)
        ax.add_collection(neighbor_collection)

        self._satellite_metadata.plot(x="lon", y="lat", ax=ax, kind="scatter", color="r")
        self._panorama_metadata.plot(x="lon", y="lat", ax=ax, kind="scatter", color="g")

        if include_text_labels:
            for sat_idx, sat_meta in self._satellite_metadata.iterrows():
                plt.text(sat_meta.lon, sat_meta.lat, f"{sat_idx}").set_clip_on(True)

            for pano_idx, pano_meta in self._panorama_metadata.iterrows():
                plt.text(pano_meta.lon, pano_meta.lat, f"{pano_idx}").set_clip_on(True)

        plt.axis("equal")
        plt.show(block=True)


def get_dataloader(dataset: VigorDataset, **kwargs):
    def _collate_fn(samples: list[VigorDatasetItem]):
        first_item = samples[0]
        return VigorDatasetItem(
            panorama_metadata=None if first_item.panorama_metadata is None else [x.panorama_metadata for x in samples],
            satellite_metadata=None if first_item.satellite_metadata is None else [x.satellite_metadata for x in samples],
            panorama=None if first_item.panorama is None else torch.stack([x.panorama for x in samples]),
            satellite=None if first_item.satellite is None else torch.stack([x.satellite for x in samples]),
        )

    return torch.utils.data.DataLoader(dataset, collate_fn=_collate_fn, **kwargs)