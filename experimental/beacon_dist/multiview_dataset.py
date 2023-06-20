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
            tensors[field] = utils.int_array_to_binary_tensor(
                arr_in[field])[:, :num_classes]
            continue
        elif field == "image_id":
            tensors[field] = torch.tensor(arr_in[field][0].copy())
            continue
        tensors[field] = torch.from_numpy(arr_in[field].copy())

    return utils.KeypointBatch(**tensors)


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        inputs: DatasetInputs,
        sample_transform_fn: Callable[[utils.KeypointBatch], utils.KeypointBatch]
        | None = None,
    ):
        assert (inputs.file_paths is None) != (inputs.data_tables is None)
        if inputs.file_paths:
            # Load from files
            self._partitions = [np.load(f, "r") for f in inputs.file_paths]
        else:
            # Load from the existing data tables
            self._partitions = inputs.data_tables

        for i, p in enumerate(self._partitions):
            for table in ["data", "image_info", "objects"]:
                assert table in p, f"{table} not in partition {i}"

        if inputs.index_path:
            self._index = DatasetIndex(**np.load(inputs.index_path))
        else:
            self._index = build_index(self._partitions)

        self._pairs_from_index = build_pairs_from_index(self._index.scene_index)
        self._sample_transform_fn = (
            sample_transform_fn if sample_transform_fn is not None else lambda x: x
        )

        self._num_classes = len(self._partitions[0]["objects"])

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

        return utils.KeypointPairs(
            context=self._sample_transform_fn(context_tensor),
            query=self._sample_transform_fn(query_tensor),
        )

    def __len__(self) -> int:
        return len(self._pairs_from_index)

    @staticmethod
    def from_single_view(
        data: np.ndarray | None = None,
        filename: str | None = None,
        sample_transform_fn: Callable[
            [utils.KeypointBatch], utils.KeypointBatch
        ] = None,
    ):
        assert (data is None) != (filename is None)
        if filename:
            data = np.load(filename)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data["data"]
        assert data is not None

        # Create a dummy image info table
        image_ids = np.unique(data["image_id"])
        image_info = np.array(
            [(i, i, np.eye(3, 4)) for i in image_ids],
            dtype=[
                ("image_id", np.uint64),
                ("scene_id", np.uint64),
                ("world_from_camera", np.float64, (3, 4)),
            ],
        )

        # Create a dummy object list
        class_labels = data["class_label"]
        all_class_bits = np.bitwise_or.reduce(data["class_label"], axis=0)
        classes_per_entry = class_labels.dtype.itemsize * 8
        num_entries = 1 if class_labels.ndim == 1 else class_labels.shape[1]
        max_possible_classes = num_entries * classes_per_entry

        for class_idx in range(max_possible_classes - 1, 0, -1):
            arr_idx = class_idx // classes_per_entry
            bit_idx = class_idx % classes_per_entry
            mask = np.uint64(1 << bit_idx)

            entry_under_test = np.uint64(
                all_class_bits[arr_idx] if num_entries > 1 else all_class_bits
            )
            is_class_present = np.bitwise_and(entry_under_test, mask)

            if is_class_present > 0:
                break

        object_list = [f"obj_{i}" for i in range(class_idx + 1)]

        data_tables = {
            "data": data,
            "image_info": image_info,
            "objects": object_list,
        }

        # Apply sample_transform_fn to both the context and the query
        sample_transform_fn = (
            sample_transform_fn if sample_transform_fn is not None else lambda x: x
        )

        return MultiviewDataset(
            DatasetInputs(file_paths=None, index_path=None, data_tables=[data_tables]),
            sample_transform_fn,
        )
