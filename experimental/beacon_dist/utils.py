import torch
import numpy as np
from typing import NamedTuple, Callable
from collections import defaultdict

DESCRIPTOR_SIZE = 32

KeypointDescriptorDtype = np.dtype(
    [
        ("image_id", np.int64),
        ("angle", np.float32),
        ("class_id", np.int32),
        ("octave", np.int32),
        ("x", np.float32),
        ("y", np.float32),
        ("response", np.float32),
        ("size", np.float32),
        ("descriptor", np.int16, (DESCRIPTOR_SIZE,)),
        ("class_label", np.int64),
    ]
)


class KeypointBatch(NamedTuple):
    # dimension [Batch]
    image_id: torch.Tensor

    # Note that the following tensors maybe ragged
    # Tensors with dimension [Batch, key_point_number]
    angle: torch.Tensor
    class_id: torch.Tensor
    octave: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    response: torch.Tensor
    size: torch.Tensor
    class_label: torch.Tensor

    # Tensor with dimension [Batch, key_point_number, 32]
    descriptor: torch.Tensor

    def to(self, *args, **kwargs):
        return KeypointBatch(
            **{key: value.to(*args, **kwargs) for key, value in self._asdict().items()}
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        data: None | np.ndarray = None,
        filename: None | str = None,
        sample_transform_fn: None | Callable[[KeypointBatch], KeypointBatch] = None,
    ):
        if filename:
            data = np.load(filename)
        assert data is not None

        tensors = defaultdict(list)
        _, first_idx = np.unique(data["image_id"], return_index=True)
        for subset in np.split(data, first_idx[1:]):
            for field in data.dtype.names:
                if field == "image_id":
                    tensors[field].append(
                        torch.from_numpy(np.array(subset[0]["image_id"]))
                    )
                    continue
                tensors[field].append(torch.from_numpy(subset[field].copy()))

        self._data = KeypointBatch(
            **{key: torch.nested.nested_tensor(value) for key, value in tensors.items()}
        )
        self._sample_transform_fn = (
            sample_transform_fn if sample_transform_fn is not None else lambda x: x
        )

    def __getitem__(self, idx: int) -> KeypointBatch:
        # unsqueeze to add a batch dimension
        return self._sample_transform_fn(
            KeypointBatch(
                **{f: getattr(self._data, f)[idx] for f in self._data._fields}
            )
        )

    def __len__(self) -> int:
        return self._data.image_id.size(0)

    def __repr__(self) -> str:
        return f"<Dataset: {len(self)} examples>"

    def data(self) -> KeypointBatch:
        return self._data


def sample_keypoints(
    sample: KeypointBatch, num_keypoints_to_sample: int, gen: torch.Generator
) -> KeypointBatch:
    # This function only works on single sample
    assert sample.image_id.ndim == 0

    num_keypoints = len(sample.x)
    if num_keypoints > num_keypoints_to_sample:
        # We need to sample the data
        idxs = torch.multinomial(
            torch.arange(0, num_keypoints, dtype=torch.float32),
            num_samples=num_keypoints_to_sample,
            replacement=False,
            generator=gen,
        )
        new_fields = {}
        for field_name in sample._fields:
            field = getattr(sample, field_name)
            if field.ndim > 0:
                new_fields[field_name] = field[idxs, ...]
            else:
                new_fields[field_name] = field
        return KeypointBatch(**new_fields)
    return sample


def batchify(sample: KeypointBatch) -> KeypointBatch:
    new_fields = {}
    for field_name in sample._fields:
        field = getattr(sample, field_name)
        if field.ndim == 1:
            new_fields[field_name] = torch.stack(field.unbind())
        else:
            info = torch.finfo if field.dtype.is_floating_point else torch.iinfo
            padding_value = info(field.dtype).min
            new_fields[field_name] = torch.nested.to_padded_tensor(
                field, padding=padding_value
            )

    return KeypointBatch(**new_fields)


def reconstruction_loss(
    input_batch: KeypointBatch,
    model_output: KeypointBatch,
    fields_to_compare: None | list[str] = None,
) -> torch.Tensor:
    if fields_to_compare is None:
        fields_to_compare = ["x", "y"]

    loss = torch.tensor(0.0)
    for field_name in fields_to_compare:
        assert field_name in input_batch._fields
        input_field = getattr(input_batch, field_name)
        output_field = getattr(model_output, field_name)
        loss += torch.nn.functional.mse_loss(input_field, output_field)
    return loss


def find_unique_classes(class_labels: torch.Tensor) -> torch.Tensor:
    exclusive_keypoints = torch.remainder(torch.log2(class_labels), 1.0) < 1e-6
    return [x for x in torch.unique(exclusive_keypoints * class_labels) if x != 0]


def is_valid_configuration(class_labels: torch.Tensor, query: torch.Tensor):
    # a configuration is valid if all exclusive keypoints are present
    assert class_labels.shape == query.shape
    unique_classes = find_unique_classes(class_labels)

    is_valid_mask = torch.ones(
        (query.shape[0], 1), dtype=torch.bool, device=class_labels.device
    )
    for class_name in unique_classes:
        class_mask = class_labels == class_name
        not_in_class = torch.logical_not(class_mask)
        all_from_class_present = torch.all(np.logical_or(not_in_class, query), dim=1, keepdim=True)
        all_from_class_absent = torch.all(
            np.logical_or(not_in_class, np.logical_not(query)), dim=1, keepdim=True
        )
        is_valid_for_class = np.logical_or(all_from_class_present, all_from_class_absent)

        is_valid_mask = torch.logical_and(is_valid_mask, is_valid_for_class)

    return is_valid_mask.to(torch.float32)


def valid_configuration_loss(
    class_labels: torch.Tensor,
    query: torch.Tensor,
    model_output: torch.Tensor,
):
    # Compute if this is a valid configuration
    labels = is_valid_configuration(class_labels, query)

    return torch.nn.functional.binary_cross_entropy_with_logits(model_output, labels)


def query_from_class_samples(
    class_labels: torch.Tensor, present_classes: list[int], exclusive_points_only=False
):
    assert class_labels.ndim == 1
    query = torch.zeros_like(class_labels, dtype=torch.bool)
    for class_name in present_classes:
        if exclusive_points_only:
            class_mask = class_labels == class_name
        else:
            class_mask = torch.bitwise_and(class_labels, class_name) > 0
        query = torch.logical_xor(query, class_mask)
    return query


def generate_valid_queries(
    class_labels: torch.Tensor,
    rng: torch.Generator,
    class_probability=0.9,
    exclusive_points_only=False,
) -> torch.Tensor:
    assert class_labels.ndim == 2
    # Create a [batch, keypoint_id] tensor of valid configurations
    queries = []
    for batch_idx in range(class_labels.shape[0]):
        # For each batch, find all unique class labels
        present_classes = []
        for class_name in find_unique_classes(class_labels[batch_idx, ...]):
            if torch.bernoulli(torch.tensor([0.9]), generator=rng) > 0:
                present_classes.append(class_name)
        queries.append(
            query_from_class_samples(
                class_labels[batch_idx, ...], present_classes, exclusive_points_only
            )
        )
    return torch.stack(queries)


def generate_invalid_queries(
    class_labels: torch.Tensor, valid_queries: torch.Tensor, rng: torch.Generator
):
    assert class_labels.shape == valid_queries.shape
    present_beacon_classes = class_labels * valid_queries
    queries = []
    for batch_idx in range(class_labels.shape[0]):
        unique_classes = find_unique_classes(present_beacon_classes[batch_idx, ...])
        did_update = False
        query = valid_queries[batch_idx, ...]

        while not did_update:
            if len(unique_classes) == 0:
                # No beacons have been set, set some fraction of exclusive beacons
                for class_name in find_unique_classes(class_labels[batch_idx]):
                    inclusive_class_mask = (
                        torch.bitwise_and(class_labels[batch_idx, ...], class_name) > 0
                    )
                    if torch.randint(2, size=(1,), generator=rng) < 1:
                        continue
                    # All exclusive beacons are disabled but there are beacons of this class that
                    # exist. Set some fraction of them
                    unaffected_bits = np.logical_and(
                        np.logical_not(inclusive_class_mask), query
                    )
                    update_mask = torch.randint(
                        2, inclusive_class_mask.shape, generator=rng
                    )
                    shared_beacons = np.logical_and(inclusive_class_mask, update_mask)

                    query = np.logical_or(
                        unaffected_bits,
                        shared_beacons,
                    )

            for class_name in unique_classes:
                exclusive_class_mask = present_beacon_classes[batch_idx] == class_name
                inclusive_class_mask = (
                    torch.bitwise_and(class_labels[batch_idx, ...], class_name) > 0
                )
                if torch.sum(exclusive_class_mask) > 1:
                    # If more than a single exclusive beacon for this class is present
                    # we can make it invalid by turning off some fraction of these beacons
                    update_mask = torch.randint(
                        2, exclusive_class_mask.shape, generator=rng
                    )

                    new_class_mask = np.logical_and(update_mask, exclusive_class_mask)

                    unaffected_bits = np.logical_and(
                        np.logical_not(exclusive_class_mask), query
                    )
                    query = np.logical_or(unaffected_bits, new_class_mask)
                elif torch.sum(exclusive_class_mask) > 0 and torch.sum(
                    inclusive_class_mask
                ) > torch.sum(exclusive_class_mask):
                    if torch.randint(2, size=(1,), generator=rng) < 1:
                        continue
                    # There is at least one exclusive beacon present and shared landmarks exist, we
                    # make an invalid config by turning off the exclusive beacon and turning on the
                    # shared beacon
                    unaffected_bits = np.logical_and(
                        np.logical_not(inclusive_class_mask), query
                    )
                    disable_exclusive_beacons = np.logical_and(
                        np.logical_not(exclusive_class_mask), query
                    )
                    shared_beacons = np.logical_xor(
                        exclusive_class_mask, inclusive_class_mask
                    )
                    query = np.logical_or(
                        np.logical_or(unaffected_bits, disable_exclusive_beacons),
                        shared_beacons,
                    )
            did_update = not torch.all(query == valid_queries[batch_idx, ...])
        queries.append(query)
    return torch.stack(queries).to(torch.bool)


def gen_descriptor(value: int):
    out = np.zeros((DESCRIPTOR_SIZE,), dtype=np.int16)
    out[0] = value
    return out


def get_descriptor_test_dataset() -> np.ndarray[KeypointDescriptorDtype]:
    """Generate a dataset where the points vary only in their descriptors"""
    IMAGE_ID = 0
    ANGLE = 0
    CLASS_ID = 0
    OCTAVE = 0
    X = 0.0
    Y = 0.0
    RESPONSE = 0.0
    SIZE = 0.0

    # fmt: off
    return np.array(
        [
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(1), 1),
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(2), 1),
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(3), 2),
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(4), 2),
        ],
        dtype=KeypointDescriptorDtype,
    )
    # fmt: on


def get_x_position_test_dataset() -> np.ndarray[KeypointDescriptorDtype]:
    """Generate a dataset where the points vary only in their descriptors"""
    IMAGE_ID = 0
    ANGLE = 0
    CLASS_ID = 0
    OCTAVE = 0
    RESPONSE = 0
    SIZE = 0.0

    # fmt: off
    return np.ndarray(
        [
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 1.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), 1),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 2.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), 2),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 3.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), 1),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 4.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), 2),
        ],
        dtype=KeypointDescriptorDtype,
    )
    # fmt: on


def get_y_position_test_dataset() -> np.ndarray[KeypointDescriptorDtype]:
    """Generate a dataset where the points vary only in their descriptors"""
    IMAGE_ID = 0
    ANGLE = 0
    CLASS_ID = 0
    OCTAVE = 0
    RESPONSE = 0
    SIZE = 0.0

    # fmt: off
    return np.ndarray(
        [
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 1.0, RESPONSE, SIZE, gen_descriptor(4), 1),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 2.0, RESPONSE, SIZE, gen_descriptor(4), 2),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 3.0, RESPONSE, SIZE, gen_descriptor(4), 1),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 4.0, RESPONSE, SIZE, gen_descriptor(4), 2),
        ],
        dtype=KeypointDescriptorDtype,
    )
    # fmt: on
