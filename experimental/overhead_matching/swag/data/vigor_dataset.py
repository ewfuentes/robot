import common.torch.load_torch_deps
import torch
import torchvision as tv
import sys
import itertools
import math

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import NamedTuple
from common.math.haversine import find_d_on_unit_circle
from common.gps import web_mercator
from enum import StrEnum, auto

EARTH_RADIUS_M = 6378137.0

class SampleMode(StrEnum):
    # In nearest mode, when indexing into the dataset, return the nearest satellite patch
    # for the selected panorama
    NEAREST = auto()
    # In pos/semipos mode, it is possible to sample all positive and semipositive pairs
    POS_SEMIPOS = auto()


class SamplePair(NamedTuple):
    panorama_idx: int
    satellite_idx: int


class PanoramaIndexInfo(NamedTuple):
    panorama_idx: int
    nearest_satellite_idx: int
    positive_satellite_idxs: list[int]
    semipositive_satellite_idxs: list[int]


class VigorDatasetConfig(NamedTuple):
    panorama_neighbor_radius: float
    satellite_patch_size: None | tuple[int, int] = None
    panorama_size: None | tuple[int, int] = None
    factor: None | float = 1.0
    satellite_zoom_level: int = 20
    sample_mode: SampleMode = SampleMode.NEAREST


class VigorDatasetItem(NamedTuple):
    panorama_metadata: dict
    satellite_metadata: dict
    panorama: torch.Tensor
    satellite: torch.Tensor


class SatelliteFromPanoramaResult(NamedTuple):
    closest_sat_idx_from_pano_idx: list[list[int]]
    positive_sat_idxs_from_pano_idx: list[list[int]]
    semipositive_sat_idxs_from_pano_idx: list[list[int]]
    positive_pano_idxs_from_sat_idx: list[list[int]]
    semipositive_pano_idxs_from_sat_idx: list[list[int]]


def series_to_dict_with_index(series: pd.Series, index_key: str = "index"):
    d = series.to_dict()
    assert index_key not in d
    d[index_key] = series.name
    return d


def load_satellite_metadata(path: Path, zoom_level: int):
    out = []
    for p in sorted(list(path.iterdir())):
        _, lat, lon = p.stem.split("_")
        lat = float(lat)
        lon = float(lon)
        web_mercator_px = web_mercator.latlon_to_pixel_coords(lat, lon, zoom_level)
        out.append((lat, lon, *web_mercator_px, p))
    return pd.DataFrame(out, columns=["lat", "lon", "web_mercator_y", "web_mercator_x", "path"])


def load_panorama_metadata(path: Path, zoom_level: int):
    out = []
    for p in sorted(list(path.iterdir())):
        pano_id, lat, lon, _ = p.stem.split(",")
        lat = float(lat)
        lon = float(lon)
        web_mercator_px = web_mercator.latlon_to_pixel_coords(lat, lon, zoom_level)
        out.append((pano_id, lat, lon, *web_mercator_px, p))
    return pd.DataFrame(out, columns=["pano_id", "lat", "lon", "web_mercator_y", "web_mercator_x", "path"])


def compute_satellite_from_panorama(sat_kdtree, sat_metadata, pano_metadata) -> SatelliteFromPanoramaResult:
    # Get the satellite patch size:
    sat_patch = load_image(sat_metadata.iloc[0]["path"], resize_shape=None)
    sat_patch_size = sat_patch.shape[1:]
    half_width = sat_patch_size[1] / 2.0
    half_height = sat_patch_size[0] / 2.0
    max_dist = np.sqrt(half_width ** 2 + half_height ** 2)

    MAX_K = 10
    _, sat_idx_from_pano_idx = sat_kdtree.query(
            pano_metadata.loc[:, ["web_mercator_x", "web_mercator_y"]].values, k=MAX_K,
            distance_upper_bound=max_dist)

    valid_mask = sat_idx_from_pano_idx != sat_kdtree.n
    valid_idxs = sat_idx_from_pano_idx[valid_mask]

    sat_x = np.full(sat_idx_from_pano_idx.shape, np.inf)
    sat_y = np.full(sat_idx_from_pano_idx.shape, np.inf)
    sat_x[valid_mask] = sat_metadata.iloc[valid_idxs]['web_mercator_x'].values
    sat_y[valid_mask] = sat_metadata.iloc[valid_idxs]['web_mercator_y'].values

    x_semipos_lb = -half_width * np.ones_like(sat_idx_from_pano_idx) + sat_x
    x_semipos_ub = half_width * np.ones_like(sat_idx_from_pano_idx) + sat_x
    y_semipos_lb = -half_height * np.ones_like(sat_idx_from_pano_idx) + sat_y
    y_semipos_ub = half_height * np.ones_like(sat_idx_from_pano_idx) + sat_y

    x_pos_lb = -half_width/2.0 * np.ones_like(sat_idx_from_pano_idx) + sat_x
    x_pos_ub = half_width/2.0 * np.ones_like(sat_idx_from_pano_idx) + sat_x
    y_pos_lb = -half_height/2.0 * np.ones_like(sat_idx_from_pano_idx) + sat_y
    y_pos_ub = half_height/2.0 * np.ones_like(sat_idx_from_pano_idx) + sat_y

    pano_x = np.expand_dims(pano_metadata["web_mercator_x"].values, -1)
    pano_y = np.expand_dims(pano_metadata["web_mercator_y"].values, -1)

    is_semipos = np.logical_and(
        np.logical_and(pano_x >= x_semipos_lb, pano_x <= x_semipos_ub),
        np.logical_and(pano_y >= y_semipos_lb, pano_y <= y_semipos_ub))

    is_pos = np.logical_and(
        np.logical_and(pano_x >= x_pos_lb, pano_x <= x_pos_ub),
        np.logical_and(pano_y >= y_pos_lb, pano_y <= y_pos_ub))

    closest_sat_idx_from_pano_idx = [sys.maxsize for _ in range(len(pano_metadata))]
    pos_sat_idxs_from_pano_idx = [[] for _ in range(len(pano_metadata))]
    semipos_sat_idxs_from_pano_idx = [[] for _ in range(len(pano_metadata))]
    pos_pano_idxs_from_sat_idx = [[] for _ in range(len(sat_metadata))]
    semipos_pano_idxs_from_sat_idx = [[] for _ in range(len(sat_metadata))]

    for pano_idx in range(len(pano_metadata)):
        for idx in range(MAX_K):
            sat_idx = sat_idx_from_pano_idx[pano_idx, idx]
            if idx == 0:
                closest_sat_idx_from_pano_idx[pano_idx] = sat_idx

            if is_pos[pano_idx, idx] and idx == 0:
                pos_sat_idxs_from_pano_idx[pano_idx].append(sat_idx)
                pos_pano_idxs_from_sat_idx[sat_idx].append(pano_idx)
            elif is_semipos[pano_idx, idx]:
                semipos_sat_idxs_from_pano_idx[pano_idx].append(sat_idx)
                semipos_pano_idxs_from_sat_idx[sat_idx].append(pano_idx)
            else:
                break

    return SatelliteFromPanoramaResult(
        closest_sat_idx_from_pano_idx=closest_sat_idx_from_pano_idx,
        positive_sat_idxs_from_pano_idx=pos_sat_idxs_from_pano_idx,
        semipositive_sat_idxs_from_pano_idx=semipos_sat_idxs_from_pano_idx,
        positive_pano_idxs_from_sat_idx=pos_pano_idxs_from_sat_idx,
        semipositive_pano_idxs_from_sat_idx=semipos_pano_idxs_from_sat_idx,
    )


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


def populate_pairs(pano_metadata, sat_metadata, sample_mode):
    out = []
    for i, d in pano_metadata.iterrows():
        if sample_mode == SampleMode.NEAREST:
            out.append(SamplePair(panorama_idx=i, satellite_idx=d["satellite_idx"]))
            continue
        # Otherwise we are in POS_SEMIPOS
        for sat_idx in d["positive_satellite_idxs"]:
            out.append(SamplePair(panorama_idx=i, satellite_idx=sat_idx))
        for sat_idx in d["semipositive_satellite_idxs"]:
            out.append(SamplePair(panorama_idx=i, satellite_idx=sat_idx))
    return out


class VigorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path | list[Path], config: VigorDatasetConfig):
        super().__init__()
        if isinstance(dataset_path, Path):
            dataset_path = [dataset_path]
        elif isinstance(dataset_path, str):
            dataset_path = [Path(dataset_path)]

        sat_metadatas = []
        pano_metadatas = []
        for p in dataset_path:
            sat_metadata = load_satellite_metadata(p / "satellite", config.satellite_zoom_level)
            pano_metadata = load_panorama_metadata(p / "panorama", config.satellite_zoom_level)

            min_lat = np.min(sat_metadata.lat)
            max_lat = np.max(sat_metadata.lat)
            delta_lat = max_lat - min_lat
            min_lon = np.min(sat_metadata.lon)
            max_lon = np.max(sat_metadata.lon)
            delta_lon = max_lon - min_lon

            sat_mask = np.logical_and(sat_metadata.lat <= min_lat + config.factor * delta_lat,
                                      sat_metadata.lon <= min_lon + config.factor * delta_lon)
            pano_mask = np.logical_and(pano_metadata.lat <= min_lat + config.factor * delta_lat,
                                       pano_metadata.lon <= min_lon + config.factor * delta_lon)
            sat_metadatas.append(sat_metadata[sat_mask])
            pano_metadatas.append(pano_metadata[pano_mask])
        self._satellite_metadata = pd.concat(sat_metadatas).reset_index(drop=True)
        self._panorama_metadata = pd.concat(pano_metadatas).reset_index(drop=True)

        self._satellite_kdtree = cKDTree(self._satellite_metadata.loc[:, ["web_mercator_x", "web_mercator_y"]].values)
        self._panorama_kdtree = cKDTree(self._panorama_metadata.loc[:, ["lat", "lon"]].values)

        # drop rows that don't have a positive match
        correspondences = compute_satellite_from_panorama(
            self._satellite_kdtree, self._satellite_metadata, self._panorama_metadata)

        self._satellite_metadata["positive_panorama_idxs"] = correspondences.positive_pano_idxs_from_sat_idx
        self._satellite_metadata["semipositive_panorama_idxs"] = correspondences.semipositive_pano_idxs_from_sat_idx
        self._panorama_metadata["positive_satellite_idxs"] = correspondences.positive_sat_idxs_from_pano_idx
        self._panorama_metadata["semipositive_satellite_idxs"] = correspondences.semipositive_sat_idxs_from_pano_idx
        self._panorama_metadata["satellite_idx"] = correspondences.closest_sat_idx_from_pano_idx

        self._panorama_metadata["neighbor_panorama_idxs"] = compute_neighboring_panoramas(
            self._panorama_kdtree, config.panorama_neighbor_radius)

        self._satellite_patch_size = config.satellite_patch_size
        self._panorama_size = config.panorama_size

        self._pairs = populate_pairs(self._panorama_metadata, self._satellite_metadata, config.sample_mode)

    @property
    def num_satellite_patches(self):
        return len(self._satellite_metadata)

    def __getitem__(self, idx_or_pair: int | SamplePair):

        if isinstance(idx_or_pair, SamplePair):
            pair = idx_or_pair
            pano_idx, sat_idx = pair.panorama_idx, pair.satellite_idx
            if pano_idx >= len(self._panorama_metadata) or sat_idx >= len(self._satellite_metadata):
                raise IndexError
        else:
            idx = idx_or_pair
            if idx > len(self) - 1:
                raise IndexError  # if we don't raise index error the iterator won't terminate

            if torch.is_tensor(idx):
                # if idx is tensor, .loc[tensor] returns DataFrame instead of Series which breaks the following lines
                idx = idx.item()
            pano_idx, sat_idx = self._pairs[idx]

        pano_metadata = self._panorama_metadata.loc[pano_idx]
        sat_metadata = self._satellite_metadata.loc[sat_idx]
        pano = load_image(pano_metadata.path, self._panorama_size)
        sat = load_image(sat_metadata.path, self._satellite_patch_size)

        return VigorDatasetItem(
            panorama_metadata=series_to_dict_with_index(pano_metadata),
            satellite_metadata=series_to_dict_with_index(sat_metadata),
            panorama=pano,
            satellite=sat
        )

    def __len__(self):
        return len(self._pairs)

    def generate_random_path(self,
                             generator: torch.Generator,
                             max_length_m: float,
                             turn_temperature: float,
                             avoid_retrace_steps: bool = True) -> list[int]:
        """Returns a list of indices into the dataset's panoramas to define a path through the graph"""
        path = []
        distance_traveled_m = 0.0
        last_direction = torch.rand((2,), generator=generator)*0.5 - 1  # random initial direction
        if torch.norm(last_direction) == 0.0:  # if we somehow sampled 0, add some small offset
            last_direction += 1e-6
        # select start panorama
        last_index = torch.randint(0, len(self), (1,), generator=generator).item()

        def softmax_w_temp(x: torch.Tensor, t: float) -> torch.tensor:
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
                similarity = torch.dot(
                    neighbor_directions[i], last_direction) / torch.norm(neighbor_directions[i]) / torch.norm(last_direction)
                neighbor_probs[i] = similarity
                if avoid_retrace_steps and neighbor_pano_idx in path:
                    neighbor_probs[i] = -1

            neighbor_probs = softmax_w_temp(neighbor_probs, turn_temperature)
            winning_neighbor = torch.multinomial(neighbor_probs, 1, generator=generator).item()

            # update current location
            last_index = last_neighbors[winning_neighbor]
            last_direction = neighbor_directions[winning_neighbor]
            distance_traveled_m += EARTH_RADIUS_M * \
                find_d_on_unit_circle(last_coord.numpy(), neighbor_coords[winning_neighbor].numpy())

        return path

    def get_sat_patch_view(self) -> torch.utils.data.Dataset:
        class OverheadVigorDataset(torch.utils.data.Dataset):
            def __init__(self, dataset: VigorDataset):
                super().__init__()
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset._satellite_metadata)

            def __getitem__(self, idx):
                if idx > len(self) - 1:
                    raise IndexError  # if we don't raise index error the iterator won't terminate
                # as this will throw a KeyError
                sat_metadata = self.dataset._satellite_metadata.loc[idx]
                sat = load_image(sat_metadata.path, self.dataset._satellite_patch_size)
                return VigorDatasetItem(
                    panorama_metadata=None,
                    satellite_metadata=series_to_dict_with_index(sat_metadata),
                    panorama=None,
                    satellite=sat
                )
        return OverheadVigorDataset(self)

    def get_pano_view(self) -> torch.utils.data.Dataset:
        class EgoVigorDataset(torch.utils.data.Dataset):
            def __init__(self, dataset: VigorDataset):
                super().__init__()
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset._panorama_metadata)

            def __getitem__(self, idx):
                if idx > len(self) - 1:
                    raise IndexError  # if we don't raise index error the iterator won't terminate
                # as this will throw a KeyError
                pano_metadata = self.dataset._panorama_metadata.loc[idx]
                pano = load_image(pano_metadata.path, self.dataset._panorama_size)
                return VigorDatasetItem(
                    panorama_metadata=series_to_dict_with_index(pano_metadata),
                    satellite_metadata=None,
                    panorama=pano,
                    satellite=None
                )
        return EgoVigorDataset(self)

    def get_patch_positions(self) -> torch.Tensor:
        """Returns a tensor of shape (N, 2) where N is the number of satellite patches and the columns are lat, lon"""
        return torch.tensor(self._satellite_metadata.loc[:, ["lat", "lon"]].values, dtype=torch.float32)

    def get_panorama_positions(self, path: list[int] = None) -> torch.Tensor:
        """Returns a tensor of shape (N, 2) where N is the number of panoramas
        """
        if path is None:
            path = np.arange(len(self._panorama_metadata))
        return torch.tensor(self._panorama_metadata.loc[path, ["lat", "lon"]].values, dtype=torch.float32)

    def visualize(self, include_text_labels=False, path=None) -> tuple["plt.Figure", "plt.Axes"]:
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
            def get_long_lat_from_idx(idx): return (
                self._panorama_metadata.loc[idx].lon, self._panorama_metadata.loc[idx].lat)
            last_pos = get_long_lat_from_idx(path[0])
            for pano_idx in path[1:]:
                path_segments.append([last_pos, get_long_lat_from_idx(pano_idx)])
                last_pos = path_segments[-1][-1]
            path_collection = LineCollection(path_segments, colors=[(
                0.25, 0.25, 0.9) for x in range(len(neighbor_segments))])
            ax.add_collection(path_collection)

        self._satellite_metadata.plot(x="lon", y="lat", ax=ax, kind="scatter", color="r")
        self._panorama_metadata.plot(x="lon", y="lat", ax=ax, kind="scatter", color="g")

        if include_text_labels:
            for sat_idx, sat_meta in self._satellite_metadata.iterrows():
                plt.text(sat_meta.lon, sat_meta.lat, f"{sat_idx}").set_clip_on(True)

            for pano_idx, pano_meta in self._panorama_metadata.iterrows():
                plt.text(pano_meta.lon, pano_meta.lat, f"{pano_idx}").set_clip_on(True)

        plt.axis("equal")

        return fig, ax


def get_dataloader(dataset: VigorDataset, **kwargs):
    def _collate_fn(samples: list[VigorDatasetItem]):
        first_item = samples[0]
        return VigorDatasetItem(
            panorama_metadata=None if first_item.panorama_metadata is None else [
                x.panorama_metadata for x in samples],
            satellite_metadata=None if first_item.satellite_metadata is None else [
                x.satellite_metadata for x in samples],
            panorama=None if first_item.panorama is None else torch.stack(
                [x.panorama for x in samples]),
            satellite=None if first_item.satellite is None else torch.stack(
                [x.satellite for x in samples]),
        )

    return torch.utils.data.DataLoader(dataset, collate_fn=_collate_fn, **kwargs)


class HardNegativeMiner:
    '''
    This class consumes the computed embeddings and tracks the most difficult examples. The
    expected use is:

    dataset = VigorDataset(...)
    miner = HardNegativeMiner(embedding_dim, dataset, batch_size=32)

    dataloader = get_dataloader(dataset, batch_sampler=miner)

    for epoch_idx in range(num_epochs):
        if epoch_idx > hard_negative_start_idx:
            miner.set_sample_mode(HARD_NEGATIVE)

        for batch in dataloader:
            sat_embeddings = sat_model(batch.satellite)
            pano_embeddings = sat_model(batch.panorama)
            miner.consume(pano_embeddings, sat_embeddings, batch)
    '''

    class SampleMode(StrEnum):
        RANDOM = auto()
        HARD_NEGATIVE = auto()

    class RandomSampleType(StrEnum):
        NEAREST = auto()
        POS_SEMIPOS = auto()

    def __init__(self,
                 batch_size: int,
                 embedding_dimension: int,
                 random_sample_type: RandomSampleType,
                 num_panoramas: int | None = None,
                 num_satellite_patches: int | None = None,
                 panorama_info_from_pano_idx: dict[int, PanoramaIndexInfo] | None = None,
                 dataset: VigorDataset | None = None,
                 hard_negative_fraction: float = 0.5,
                 hard_negative_pool_size: int = 100,
                 generator: torch.Generator | None = None,
                 device='cpu'):

        if dataset is not None:
            assert (num_panoramas is None and
                    num_satellite_patches is None and
                    panorama_info_from_pano_idx is None)
            num_panoramas = len(dataset._panorama_metadata)
            num_satellite_patches = len(dataset._satellite_metadata)
            panorama_info_from_pano_idx = {}
            for pano_idx, row in dataset._panorama_metadata.iterrows():
                panorama_info_from_pano_idx[pano_idx] = PanoramaIndexInfo(
                    panorama_idx=pano_idx,
                    nearest_satellite_idx=row["satellite_idx"],
                    positive_satellite_idxs=row["positive_satellite_idxs"],
                    semipositive_satellite_idxs=row["semipositive_satellite_idxs"])

        assert (num_panoramas is not None and
                num_satellite_patches is not None and
                panorama_info_from_pano_idx is not None)

        self._generator = generator if generator is not None else torch.Generator()

        self._panorama_embeddings = torch.full(
                (num_panoramas, embedding_dimension), float('nan'), device=device)

        self._satellite_embeddings = torch.full(
                (num_satellite_patches, embedding_dimension), float('nan'), device=device)
        self._panorama_info_from_pano_idx = panorama_info_from_pano_idx

        self._sample_mode = self.SampleMode.RANDOM

        self._hard_negative_fraction = hard_negative_fraction
        self._hard_negative_pool_size = hard_negative_pool_size
        self._batch_size = batch_size
        self._random_sample_type = random_sample_type

    def __iter__(self):
        # To sample batches, we create a random permutation of all panoramas
        # We then assign satellite patch to each panorama. The satellite image will either be
        # a positive/semipositive satellite patch or a mined hard negative example
        permuted_panoramas = torch.randperm(self._panorama_embeddings.shape[0], generator=self._generator).tolist()

        if self._sample_mode == HardNegativeMiner.SampleMode.HARD_NEGATIVE:
            similarities = torch.einsum(
                    "nd,md->nm", self._panorama_embeddings, self._satellite_embeddings)
            # A row of this matrix contains the satellite patch similarities for a given panorama
            # sorted from least similar to most similar. When mining hard negatives, we want to
            # present the true positives and semipositives (since there are so few of them) and
            # the most similar negative matches.
            sorted_sat_idxs_from_pano_idx = torch.argsort(similarities)

            num_hard_negatives = min(self._satellite_embeddings.shape[0], self._hard_negative_pool_size)

            num_hard_negatives_per_batch = min(
                    math.ceil(self._batch_size * self._hard_negative_fraction), self._batch_size)
        else:
            num_hard_negatives_per_batch = 0

        batches = []
        for pano_batches in itertools.batched(permuted_panoramas, self._batch_size):
            batch = []
            for i, pano_idx in enumerate(pano_batches):
                if i < num_hard_negatives_per_batch:
                    # Sample hard negatives
                    hard_negative_idx = torch.randint(num_hard_negatives, (1,))
                    sat_idx = sorted_sat_idxs_from_pano_idx[pano_idx, -(hard_negative_idx+1)].item()

                    batch.append(SamplePair(panorama_idx=pano_idx, satellite_idx=sat_idx))
                else:
                    # Sample uniformly among positive and semipositive satellite patches
                    pano_info = self._panorama_info_from_pano_idx[pano_idx]
                    if self._random_sample_type == self.RandomSampleType.POS_SEMIPOS:
                        num_options = len(pano_info.positive_satellite_idxs) + len(pano_info.semipositive_satellite_idxs)
                        matching_idx = torch.randint(num_options, (1,), generator=self._generator)
                        if matching_idx < len(pano_info.positive_satellite_idxs):
                            satellite_idx = pano_info.positive_satellite_idxs[matching_idx]
                        else:
                            matching_idx -= len(pano_info.positive_satellite_idxs)
                            satellite_idx = pano_info.semipositive_satellite_idxs[matching_idx]
                    elif self._random_sample_type == self.RandomSampleType.NEAREST:
                        # Sample the nearest satellite
                        satellite_idx = pano_info.nearest_satellite_idx

                    batch.append(SamplePair(panorama_idx=pano_idx, satellite_idx=satellite_idx))
            batches.append(batch)

        for b in batches:
            yield b

    def consume(self,
                panorama_embeddings: torch.Tensor,
                satellite_embeddings: torch.Tensor,
                batch: VigorDatasetItem | None = None,
                panorama_idxs: list[int] | None = None,
                satellite_patch_idxs: list[int] | None = None):
        if batch is not None:
            panorama_idxs = [x["index"] for x in batch.panorama_metadata]
            satellite_patch_idxs = [x["index"] for x in batch.satellite_metadata]
        assert panorama_idxs is not None and satellite_patch_idxs is not None

        device = self._panorama_embeddings.device

        self._panorama_embeddings[panorama_idxs] = panorama_embeddings.to(device)
        self._satellite_embeddings[satellite_patch_idxs] = satellite_embeddings.to(device)

    def set_sample_mode(self, mode: SampleMode):
        self._sample_mode = mode
