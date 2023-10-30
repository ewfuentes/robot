import common.torch as torch
import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Callable
from collections import defaultdict

DESCRIPTOR_SIZE = 32
CLASS_SIZE = 4

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
        ("class_label", np.uint64, (CLASS_SIZE,)),
    ]
)

ImageDescriptorDtype = np.dtype(
    [
        ("image_id", np.int64),
        ("char", (np.unicode_, 1)),
        ("x", np.float32),
        ("y", np.float32),
        ("theta", np.float32),
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


class KeypointPairs(NamedTuple):
    context: KeypointBatch
    query: KeypointBatch

    def to(self, *args, **kwargs):
        return KeypointPairs(
            context=self.context.to(*args, **kwargs),
            query=self.query.to(*args, **kwargs),
        )


def int_array_to_binary_tensor(arr: np.ndarray):
    num_bits_per_entry = arr.dtype.itemsize * 8
    mask_arr = (1 << np.arange(num_bits_per_entry)).astype(np.uint64)
    mask_arr = np.expand_dims(mask_arr, 0)
    arrs_to_concat = []
    if arr.ndim > 1:
        for col in range(arr.shape[1]):
            arrs_to_concat.append(
                np.bitwise_and(np.expand_dims(arr[:, col], 1), mask_arr) > 0
            )
    else:
        arrs_to_concat.append(np.bitwise_and(np.expand_dims(arr, 1), mask_arr) > 0)
    return torch.from_numpy(np.concatenate(arrs_to_concat, axis=1))


def trim_class_label(class_labels: list[torch.Tensor]):
    max_class_idx = 0
    for label in class_labels:
        _, col = torch.nonzero(label, as_tuple=True)
        max_class_idx = max(max_class_idx, torch.max(col).item())

    return [label[:, : max_class_idx + 1] for label in class_labels]


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


def batchify_pair(sample: KeypointPairs) -> KeypointPairs:
    return KeypointPairs(
        context=batchify(sample.context),
        query=batchify(sample.query),
    )


# Convert a keypoint batch where each field contains a nested tensor into a padded batch
def batchify(sample: KeypointBatch) -> KeypointBatch:
    new_fields = {}
    for field_name in sample._fields:
        field = getattr(sample, field_name)
        if field.ndim == 1:
            new_fields[field_name] = torch.stack(field.unbind())
        else:
            if field.dtype == torch.bool:
                padding_value = False
            else:
                info = torch.finfo if field.dtype.is_floating_point else torch.iinfo
                padding_value = info(field.dtype).min
            new_fields[field_name] = torch.nested.to_padded_tensor(
                field, padding=padding_value
            )

    return KeypointBatch(**new_fields)


def find_unique_class_idxs(class_labels: torch.Tensor) -> torch.Tensor:
    idxs = torch.nonzero(class_labels, as_tuple=True)
    return torch.unique(idxs[-1])


def is_valid_configuration(class_labels: torch.Tensor, query: torch.Tensor):
    # a configuration is valid if all exclusive keypoints are present
    assert class_labels.shape[:-1] == query.shape
    # valid_landmarks is a num_environments x num_landmarks tensor. vl[i, j]
    # is true if the j'th landmark in the i'th environment has at least one class label
    # set. A landmark may not be set if was introduced through padding.
    valid_landmarks = torch.sum(class_labels, dim=-1)
    unique_class_idxs = find_unique_class_idxs(class_labels)

    is_exclusive_valid = torch.ones(
        (query.shape[0], 1), dtype=torch.bool, device=class_labels.device
    )
    # negative class labels correspond to padded entries
    # ignore these entries for the purposes of determining validity
    is_shared_valid = torch.logical_not(valid_landmarks)
    for class_idx in unique_class_idxs:
        # Compute if the exclusive beacons have are valid
        # exclusive_class_mask is a num_environemnts x num_landmark. ecm[i, j] is true
        # if the j'th landmark in the i'th environment exclusively belongs to the class
        # identified by class_idx
        # we compute the the valid landmarks by xoring with a tensor with the desired class set
        # If the class for the beacon is set, it is cleared. If it is unset, then the bool is set
        # We then logical_or (implemented by summing) across the class dimension. The exclusive
        # landmarks are those that have no labels set.
        class_mask = torch.zeros(
            (1, 1, class_labels.shape[-1]), dtype=torch.bool, device=class_labels.device
        )
        class_mask[:, :, class_idx] = True
        class_removed = torch.logical_xor(class_labels, class_mask)
        exclusive_class_mask = torch.logical_not(
            torch.sum(class_removed, -1, dtype=torch.bool)
        )

        not_in_class = torch.logical_not(exclusive_class_mask)
        all_from_class_present = torch.all(
            torch.logical_or(not_in_class, query), dim=1, keepdim=True
        )
        all_from_class_absent = torch.all(
            torch.logical_or(not_in_class, torch.logical_not(query)),
            dim=1,
            keepdim=True,
        )
        is_exclusive_valid_for_class = torch.logical_or(
            all_from_class_present, all_from_class_absent
        )

        is_exclusive_valid = torch.logical_and(
            is_exclusive_valid, is_exclusive_valid_for_class
        )

        # compute if the shared beacons are valid
        inclusive_class_mask = torch.sum(
            torch.logical_and(class_labels, class_mask), -1, dtype=bool
        )
        shared_class_mask = torch.logical_and(
            torch.logical_not(exclusive_class_mask), inclusive_class_mask
        )

        # exclusive beacons are automatically valid
        shared_and_explained_by_class = torch.logical_and(
            shared_class_mask,
            torch.logical_or(all_from_class_present, torch.logical_not(query)),
        )
        is_shared_valid_for_class = torch.logical_or(
            exclusive_class_mask, shared_and_explained_by_class
        )
        is_shared_valid = torch.logical_or(is_shared_valid, is_shared_valid_for_class)

    is_shared_valid = torch.all(is_shared_valid, dim=1, keepdim=True)
    return torch.logical_and(is_exclusive_valid, is_shared_valid).to(torch.float32)


def valid_configuration_loss(
    class_labels: torch.Tensor,
    query: torch.Tensor,
    model_output: torch.Tensor,
    reduction: str = "mean",
):
    # Compute if this is a valid configuration
    labels = is_valid_configuration(class_labels, query)

    return torch.nn.functional.binary_cross_entropy_with_logits(
        model_output, labels, reduction=reduction
    )


def query_from_class_samples(
    class_labels: torch.Tensor, present_classes: list[int], exclusive_points_only=False
):
    assert class_labels.ndim == 2
    query = torch.zeros((class_labels.shape[0],), dtype=torch.bool)
    for present_idx in present_classes:
        present_mask = torch.zeros((1, class_labels.shape[-1]), dtype=torch.bool)
        present_mask[:, present_idx - 1] = True
        if exclusive_points_only:
            class_mask = torch.logical_not(
                torch.sum(torch.logical_xor(class_labels, present_mask), -1)
            )
        else:
            class_mask = torch.sum(torch.bitwise_and(class_labels, present_mask), -1)
        query = torch.logical_xor(query, class_mask)
    return query


def generate_valid_queries(
    class_labels: torch.Tensor,
    rng: torch.Generator,
    class_probability=0.9,
    exclusive_points_only=False,
) -> torch.Tensor:
    assert class_labels.ndim == 3
    # Create a [batch, keypoint_id] tensor of valid configurations
    queries = []
    for batch_idx in range(class_labels.shape[0]):
        # For each batch, find all unique class labels
        present_classes = []
        for class_idx in find_unique_class_idxs(class_labels[batch_idx, ...]):
            if torch.bernoulli(torch.tensor([0.9]), generator=rng) > 0:
                present_classes.append(class_idx)
        queries.append(
            query_from_class_samples(
                class_labels[batch_idx, ...], present_classes, exclusive_points_only
            )
        )
    return torch.stack(queries)


def generate_invalid_queries(
    class_labels: torch.Tensor, valid_queries: torch.Tensor, rng: torch.Generator
):
    assert class_labels.shape[:-1] == valid_queries.shape

    present_beacon_classes = torch.logical_and(
        class_labels, valid_queries.unsqueeze(-1)
    )
    queries = []
    for batch_idx in range(class_labels.shape[0]):
        unique_classes = find_unique_class_idxs(present_beacon_classes[batch_idx, ...])
        did_update = False
        query = valid_queries[batch_idx, ...]

        count = 0
        while not did_update and count < 10:
            count += 1
            if len(unique_classes) == 0:
                # No beacons have been set, set some fraction of exclusive beacons
                for class_name in find_unique_class_idxs(class_labels[batch_idx]):
                    class_mask = torch.zeros(
                        (1, class_labels.shape[-1]), dtype=torch.bool
                    )
                    class_mask[:, class_name] = True
                    inclusive_class_mask = torch.sum(
                        torch.logical_and(class_labels[batch_idx, ...], class_mask),
                        -1,
                        dtype=bool,
                    )
                    if torch.randint(2, size=(1,), generator=rng) < 1:
                        continue
                    # All exclusive beacons are disabled but there are beacons of this class that
                    # exist. Set some fraction of them
                    unaffected_bits = torch.logical_and(
                        torch.logical_not(inclusive_class_mask), query
                    )
                    update_mask = torch.randint(
                        2, inclusive_class_mask.shape, generator=rng
                    )
                    shared_beacons = torch.logical_and(
                        inclusive_class_mask, update_mask
                    )

                    query = torch.logical_or(
                        unaffected_bits,
                        shared_beacons,
                    )

            for class_name in unique_classes:
                class_mask = torch.zeros((1, class_labels.shape[-1]), dtype=torch.bool)
                class_mask[:, class_name] = True
                class_removed = torch.logical_xor(
                    present_beacon_classes[batch_idx, ...], class_mask
                )
                exclusive_class_mask = torch.logical_not(
                    torch.sum(class_removed, -1, dtype=torch.bool)
                )
                inclusive_class_mask = torch.sum(
                    torch.logical_and(class_labels[batch_idx, ...], class_mask),
                    -1,
                    dtype=bool,
                )

                if torch.sum(exclusive_class_mask) > 1:
                    # If more than a single exclusive beacon for this class is present
                    # we can make it invalid by turning off some fraction of these beacons
                    update_mask = torch.randint(
                        2, exclusive_class_mask.shape, generator=rng
                    )

                    new_class_mask = torch.logical_and(
                        update_mask, exclusive_class_mask
                    )

                    unaffected_bits = torch.logical_and(
                        torch.logical_not(exclusive_class_mask), query
                    )
                    query = torch.logical_or(unaffected_bits, new_class_mask)
                elif torch.sum(exclusive_class_mask) > 0 and torch.sum(
                    inclusive_class_mask
                ) > torch.sum(exclusive_class_mask):
                    if torch.randint(2, size=(1,), generator=rng) < 1:
                        continue
                    # There is at least one exclusive beacon present and shared landmarks exist, we
                    # make an invalid config by turning off the exclusive beacon and turning on the
                    # shared beacon
                    unaffected_bits = torch.logical_and(
                        torch.logical_not(inclusive_class_mask), query
                    )
                    disable_exclusive_beacons = torch.logical_and(
                        torch.logical_not(exclusive_class_mask), query
                    )
                    shared_beacons = torch.logical_xor(
                        exclusive_class_mask, inclusive_class_mask
                    )
                    query = torch.logical_or(
                        torch.logical_or(unaffected_bits, disable_exclusive_beacons),
                        shared_beacons,
                    )
            did_update = not torch.all(query == valid_queries[batch_idx, ...])
        queries.append(query)
    return torch.stack(queries).to(torch.bool)


def gen_descriptor(value: int):
    out = np.zeros((DESCRIPTOR_SIZE,), dtype=np.int16)
    out[0] = value
    return out


def get_descriptor_test_dataset() -> npt.NDArray[KeypointDescriptorDtype]:
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
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(1), (1, 0, 0, 0)),
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(2), (1, 0, 0, 0)),
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(3), (2, 0, 0, 0)),
             (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, X, Y, RESPONSE, SIZE, gen_descriptor(4), (2, 0, 0, 0)),
        ],
        dtype=KeypointDescriptorDtype,
    )
    # fmt: on


def get_x_position_test_dataset() -> npt.NDArray[KeypointDescriptorDtype]:
    """Generate a dataset where the points vary only in their descriptors"""
    IMAGE_ID = 0
    ANGLE = 0
    CLASS_ID = 0
    OCTAVE = 0
    RESPONSE = 0
    SIZE = 0.0

    # fmt: off
    return np.array(
        [
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 1.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), (1, 0, 0, 0)),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 3.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), (1, 0, 0, 0)),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 2.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), (2, 0, 0, 0)),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 4.0, 0.0, RESPONSE, SIZE, gen_descriptor(4), (2, 0, 0, 0)),
        ],
        dtype=KeypointDescriptorDtype,
    )
    # fmt: on


def get_y_position_test_dataset() -> npt.NDArray[KeypointDescriptorDtype]:
    """Generate a dataset where the points vary only in their descriptors"""
    IMAGE_ID = 0
    ANGLE = 0
    CLASS_ID = 0
    OCTAVE = 0
    RESPONSE = 0
    SIZE = 0.0

    # fmt: off
    return np.array(
        [
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 1.0, RESPONSE, SIZE, gen_descriptor(4), (1, 0, 0, 0)),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 3.0, RESPONSE, SIZE, gen_descriptor(4), (1, 0, 0, 0)),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 2.0, RESPONSE, SIZE, gen_descriptor(4), (2, 0, 0, 0)),
            (IMAGE_ID, ANGLE, CLASS_ID, OCTAVE, 0.0, 4.0, RESPONSE, SIZE, gen_descriptor(4), (2, 0, 0, 0)),
        ],
        dtype=KeypointDescriptorDtype,
    )
    # fmt: on


def get_test_all_queries():
    return torch.tensor(
        [
            # Valid Queries
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
            # Invalid Queries with one beacon
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            # Invalid with two beacons
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            # Invalid with three beacons
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )


def test_dataset_collator(
    samples: list[KeypointPairs],
) -> tuple[KeypointPairs, torch.Tensor]:
    assert len(samples) == 1
    queries = get_test_all_queries()
    num_queries = queries.shape[0]
    replicated = KeypointBatch(
        **{
            k: torch.nested.nested_tensor([v] * num_queries)
            for k, v in samples[0].context._asdict().items()
        }
    )
    batched = batchify(replicated)
    out = KeypointPairs(
        context=batched,
        query=batched,
    )
    return out, queries
