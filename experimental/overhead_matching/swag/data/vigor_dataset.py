import common.torch.load_torch_deps
import torch
import torchvision as tv

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import NamedTuple
from common.math.haversine import find_d_on_unit_circle


class VigorDatasetConfig(NamedTuple):
    panorama_neighbor_radius: float
    satellite_patch_size: None | tuple[int, int] = None
    panorama_size: None | tuple[int, int] = None
    factor: None | float = 1.0


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

def load_image(path: Path, resize_shape: None | tuple[int, int]):
    img = tv.io.read_image(path, mode=tv.io.ImageReadMode.RGB)
    img = tv.transforms.functional.convert_image_dtype(img)
    if resize_shape is not None and img.shape[1:] != resize_shape:
        img = tv.transforms.functional.resize(img, resize_shape)
    return img


class VigorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, config: VigorDatasetConfig):
        super().__init__()
        self._dataset_path = dataset_path

        self._satellite_metadata = load_satellite_metadata(dataset_path / "satellite")
        self._panorama_metadata = load_panorama_metadata(dataset_path / "panorama")

        min_lat = np.min(self._satellite_metadata.lat)
        max_lat = np.max(self._satellite_metadata.lat)
        delta_lat = max_lat - min_lat
        min_lon = np.min(self._satellite_metadata.lon)
        max_lon = np.max(self._satellite_metadata.lon)
        delta_lon = max_lon - min_lon

        sat_mask = np.logical_and(self._satellite_metadata.lat < min_lat + config.factor * delta_lat,
                                  self._satellite_metadata.lon < min_lon + config.factor * delta_lon)
        pano_mask = np.logical_and(self._panorama_metadata.lat < min_lat + config.factor * delta_lat,
                                  self._panorama_metadata.lon < min_lon + config.factor * delta_lon)
        self._satellite_metadata = self._satellite_metadata[sat_mask].reset_index(drop=True)
        self._panorama_metadata = self._panorama_metadata[pano_mask].reset_index(drop=True)

        self._satellite_kdtree = cKDTree(self._satellite_metadata.loc[:, ["lat", "lon"]].values)
        self._panorama_kdtree = cKDTree(self._panorama_metadata.loc[:, ["lat", "lon"]].values)

        sat_idx_from_pano_idx, pano_idxs_from_sat_idx = compute_satellite_from_panorama(
                self._satellite_kdtree, self._panorama_metadata)

        self._satellite_metadata["panorama_idxs"] = pano_idxs_from_sat_idx
        self._panorama_metadata["satellite_idx"] = sat_idx_from_pano_idx

        self._panorama_metadata["neighbor_panorama_idxs"] = compute_neighboring_panoramas(
                self._panorama_kdtree, config.panorama_neighbor_radius)

        self._satellite_patch_size = config.satellite_patch_size
        self._panorama_size = config.panorama_size

    @property
    def num_satellite_patches(self):
        return len(self._satellite_metadata)

    def __getitem__(self, idx):
        if idx > len(self) - 1:
            raise IndexError  # if we don't raise index error the iterator won't terminate

        if torch.is_tensor(idx):
            idx = idx.item()  # if idx is tensor, .loc[tensor] returns DataFrame instead of Series which breaks the following lines

        pano_metadata = self._panorama_metadata.loc[idx]
        sat_metadata = self._satellite_metadata.loc[pano_metadata.satellite_idx]
        pano = load_image(pano_metadata.path, self._panorama_size)
        sat = load_image(sat_metadata.path, self._satellite_patch_size)

        return VigorDatasetItem(
            panorama_metadata=series_to_dict_with_index(pano_metadata),
            satellite_metadata=series_to_dict_with_index(sat_metadata),
            panorama=pano,
            satellite=sat
        )

    def __len__(self):
        return len(self._panorama_metadata)
    
    def generate_random_path(self,
                             generator: torch.Generator,
                             max_length_m: float,
                             turn_temperature: float)->list[int]:
        """Returns a list of indices into the dataset's panoramas to define a path through the graph"""
        path = [] 
        distance_traveled_m = 0.0
        last_direction = torch.rand((2,), generator=generator)*0.5 - 1  # random initial direction
        if torch.norm(last_direction) == 0.0:  # if we somehow sampled 0, add some small offset
            last_direction += 1e-6
        # select start panorama
        last_index = torch.randint(0, len(self), (1,), generator=generator).item()

        def softmax_w_temp(x: torch.Tensor, t: float)-> torch.tensor:
            ex = torch.exp(x / t)
            return ex / torch.sum(ex)

        while distance_traveled_m < max_length_m:
            path.append(last_index)
            last_item = self._panorama_metadata.loc[last_index]
            last_coord = torch.tensor([last_item['lon'], last_item['lat']])
            last_neighbors = last_item['neighbor_panorama_idxs']

            # calculate distribution over neighbors
            neighbor_probs = torch.zeros(len(last_neighbors))
            neighbor_coords = torch.zeros(len(last_neighbors), 2)
            neighbor_directions = torch.zeros(len(last_neighbors), 2)
            for i, neighbor_pano_idx in enumerate(last_neighbors):
                neighbor_item = self._panorama_metadata.loc[neighbor_pano_idx]
                neighbor_coords[i] = torch.tensor([neighbor_item['lon'], neighbor_item['lat']])
                neighbor_directions[i] = neighbor_coords[i] - last_coord
                similarity = torch.dot(neighbor_directions[i], last_direction) / torch.norm(neighbor_directions[i]) / torch.norm(last_direction)
                neighbor_probs[i] = similarity

            neighbor_probs = softmax_w_temp(neighbor_probs, turn_temperature) 
            winning_neighbor = torch.multinomial(neighbor_probs, 1, generator=generator).item()
            
            # update current location
            last_index = last_neighbors[winning_neighbor]
            last_direction = neighbor_directions[winning_neighbor]
            EARTH_RADIUS_M = 6378137.0
            distance_traveled_m +=  EARTH_RADIUS_M * find_d_on_unit_circle(last_coord.numpy(), neighbor_coords[winning_neighbor].numpy())

        return path

    
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
                sat = load_image(sat_metadata.path, self.dataset._satellite_patch_size)
                return VigorDatasetItem(
                    None,
                    series_to_dict_with_index(sat_metadata),
                    None,
                    sat
                )
        return OverheadVigorDataset(self)

    def visualize(self, include_text_labels=False, path=None) -> "plt.Figure":
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        fig = plt.figure()
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

        if path is not None and len(path) > 2:
            path_segments = []
            get_long_lat_from_idx = lambda idx: (self._panorama_metadata.loc[idx].lon, self._panorama_metadata.loc[idx].lat)
            last_pos = get_long_lat_from_idx(path[0])
            for pano_idx in path[1:]:
                path_segments.append([last_pos, get_long_lat_from_idx(pano_idx)])
                last_pos = path_segments[-1][-1]
            path_collection = LineCollection(path_segments, colors=[(0.25, 0.25, 0.9) for x in range(len(neighbor_segments))]) 
            ax.add_collection(path_collection)


        self._satellite_metadata.plot(x="lon", y="lat", ax=ax, kind="scatter", color="r")
        self._panorama_metadata.plot(x="lon", y="lat", ax=ax, kind="scatter", color="g")

        if include_text_labels:
            for sat_idx, sat_meta in self._satellite_metadata.iterrows():
                plt.text(sat_meta.lon, sat_meta.lat, f"{sat_idx}").set_clip_on(True)

            for pano_idx, pano_meta in self._panorama_metadata.iterrows():
                plt.text(pano_meta.lon, pano_meta.lat, f"{pano_idx}").set_clip_on(True)

        plt.axis("equal")

        return fig


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
