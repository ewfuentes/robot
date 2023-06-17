from __future__ import annotations

from typing import NamedTuple, Callable

import torch
import numpy as np
import tqdm
import itertools

from experimental.beacon_dist import utils

ImageInfoDtype = np.dtype(
    [
        ("image_id", np.int64),
        ("scene_id", np.int64),
        ("world_from_camera", (np.float32, (3, 4))),
    ]
)

SceneInfoDtype = np.dtype(
    [
        ("scene_id", np.int64),
        ("object_name", (np.unicode_, 1)),
        ("world_from_object", (np.float32, (3, 4))),
    ]
)


class ImageIndexEntry(NamedTuple):
    id: int
    partition_idx: int
    start_idx: int
    end_idx: int


class SceneIndexEntry(NamedTuple):
    id: int
    image_ids: list[int]


class DatasetIndex(NamedTuple):
    scene_index: dict[int, SceneIndexEntry]
    image_index: dict[int, ImageIndexEntry]

    def update(self, other: DatasetIndex):
        self.scene_index.update(other.scene_index)
        self.image_index.update(other.image_index)


class DatasetInputs(NamedTuple):
    file_paths: list[str] | None
    index_path: str | None
    data_tables: list[dict[str, np.ndarray]] | None


class KeypointPairs(NamedTuple):
    context: utils.KeypointBatch
    query: utils.KeypointBatch

    def to(self, *args, **kwargs):
        return KeypointPairs(
            context=self.context.to(*args, **kwargs),
            query=self.query.to(*args, **kwargs),
        )


def build_partition_index(partition_idx: int, f: np.lib.npyio.NpzFile) -> DatasetIndex:
    image_ids = f["data"]["image_id"]

    # We require that the keypoints corresponding to an image are contiguous. We instead
    # check that the image ids are monotonically increasing which is a stricter check.
    assert np.all(
        np.diff(image_ids) >= 0
    ), f"image ids in {f} are not monotonically increasing"

    # Build the image index
    elements, indices = np.unique(image_ids, return_index=True)
    indices = np.concatenate([indices, [len(image_ids)]])

    image_index = {}
    for element, start, end in zip(elements, indices[:-1], indices[1:]):
        image_index[element] = ImageIndexEntry(
            id=element, partition_idx=partition_idx, start_idx=start, end_idx=end
        )

    # Build the scene index
    scene_index = {}
    image_info = f["image_info"]
    scene_ids = np.unique(image_info["scene_id"])
    for scene_id in scene_ids:
        scene_index[scene_id] = SceneIndexEntry(
            id=scene_id,
            image_ids=list(image_info[image_info["scene_id"] == scene_id]["image_id"]),
        )

    return DatasetIndex(scene_index=scene_index, image_index=image_index)


def build_index(file_handles: list[dict[str, np.ndarray]]) -> DatasetIndex:
    out = DatasetIndex(scene_index={}, image_index={})
    for partition_idx, f in tqdm.tqdm(
        enumerate(file_handles), desc="Building Index...", total=len(file_handles)
    ):
        out.update(build_partition_index(partition_idx, f))
    return out


def build_pairs_from_index(scene_index: dict[int, SceneIndexEntry]):
    out = []
    for entry in scene_index.values():
        out.extend(itertools.product(entry.image_ids, repeat=2))
    return out


def keypoint_tensor_from_array(
    arr_in: np.ndarray[utils.KeypointDescriptorDtype], num_classes: int
) -> utils.KeypointBatch:
    tensors = {}
    for field in arr_in.dtype.names:
        if field == "class_label":
            tensors[field] = utils.int_array_to_binary_tensor(arr_in[field])[:, :num_classes]
            continue
        tensors[field] = torch.from_numpy(arr_in[field].copy())

    return utils.KeypointBatch(**tensors)


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        inputs: DatasetInputs,
        sample_transform_fn: Callable[[KeypointPairs], KeypointPairs] | None = None,
    ):
        assert (inputs.file_paths is None) != (inputs.data_tables is None)
        if inputs.file_paths:
            # Load from files
            self._partitions = [np.load(f, "r") for f in inputs.file_paths]
            print(
                "num_scenes:",
                len(self._index.scene_index),
                "num images:",
                len(self._index.image_index),
            )
        else:
            # Load from the existing data tables
            self._partitions = inputs.data_tables

        for i, p in enumerate(self._partitions):
            for table in ["data", "image_info", "object_list"]:
                assert table in p, f"{table} not in partition {i}"

        if inputs.index_path:
            self._index = DatasetIndex(**np.load(inputs.index_path))
        else:
            self._index = build_index(self._partitions)

        self._pairs_from_index = build_pairs_from_index(self._index.scene_index)
        self._sample_transform_fn = (
            sample_transform_fn if sample_transform_fn is not None else lambda x: x
        )

        self._num_classes = len(self._partitions[0]["object_list"])

    def get_keypoint_array(self, entry: ImageIndexEntry) -> torch.Tensor:
        np_array = self._partitions[entry.partition_idx]["data"][
            entry.start_idx : entry.end_idx
        ]
        return np_array

    def __getitem__(self, idx: int):
        context_image_idx, query_image_idx = self._pairs_from_index[idx]

        context_entry = self._index.image_index[context_image_idx]
        context_array = self.get_keypoint_array(context_entry)
        context_tensor = keypoint_tensor_from_array(context_array, self._num_classes)

        query_entry = self._index.image_index[query_image_idx]
        query_array = self.get_keypoint_array(query_entry)
        query_tensor = keypoint_tensor_from_array(query_array, self._num_classes)

        return KeypointPairs(
            context=self._sample_transform_fn(context_tensor),
            query=self._sample_transform_fn(query_tensor),
        )

    def __len__(self) -> int:
        return len(self._pairs_from_index)