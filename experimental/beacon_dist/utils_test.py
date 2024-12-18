import unittest

import common.torch as torch
import numpy as np

from experimental.beacon_dist.test_helpers import get_test_data
from experimental.beacon_dist.utils import (
    sample_keypoints,
    batchify,
    valid_configuration_loss,
    generate_valid_queries,
    generate_invalid_queries,
    query_from_class_samples,
    is_valid_configuration,
    test_dataset_collator,
    get_x_position_test_dataset,
    KeypointBatch,
)

from experimental.beacon_dist.multiview_dataset import MultiviewDataset


def create_context_batch_from_dataset(dataset: MultiviewDataset) -> KeypointBatch:
    samples = list(dataset)
    fields = {}
    for field in KeypointBatch._fields:
        fields[field] = torch.nested.nested_tensor([getattr(x.context, field) for x in samples])
    return KeypointBatch(**fields)


class UtilsTest(unittest.TestCase):
    def test_sample_keypoints_subsample(self):
        # Setup
        gen = torch.manual_seed(0)
        test_data = get_test_data()
        dataset = MultiviewDataset.from_single_view(data=test_data)

        # Action
        subsampled = sample_keypoints(
            dataset[0].context, num_keypoints_to_sample=2, gen=gen
        )

        # Verification
        # There should be two keypoints even though the original sample had 3 points
        self.assertEqual(subsampled.x.shape[0], 2)

    def test_single_row_sample(self):
        test_data = get_test_data()
        dataset = MultiviewDataset.from_single_view(data=test_data)
        print(dataset[2])

    def test_batchify(self):
        # Setup
        test_data = get_test_data()
        dataset = MultiviewDataset.from_single_view(data=test_data)
        batch = create_context_batch_from_dataset(dataset)

        # Action
        batched = batchify(batch)

        # Verification
        # Check that one of the fields is set appropriately
        # There are three elements in the batch and the largest one has three key points
        self.assertEqual(batched.x.shape, (3, 3))

        # The second element has two keypoints, so the last element should be float32.min
        self.assertNotEqual(batched.x[1, 0], torch.finfo(torch.float32).min)
        self.assertNotEqual(batched.x[1, 1], torch.finfo(torch.float32).min)
        self.assertEqual(batched.x[1, 2], torch.finfo(torch.float32).min)

        # The third element has one keypoint, so the last two elements should be float32.min
        self.assertNotEqual(batched.x[2, 0], torch.finfo(torch.float32).min)
        self.assertEqual(batched.x[2, 1], torch.finfo(torch.float32).min)
        self.assertEqual(batched.x[2, 2], torch.finfo(torch.float32).min)

    def test_valid_configuration_loss_valid_queries_are_likely(self):
        # Setup
        dataset = MultiviewDataset.from_single_view(data=get_test_data())
        batch = batchify(create_context_batch_from_dataset(dataset))
        query = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_valid_configuration_loss_invalid_queries_are_unlikely(self):
        # Setup
        dataset = MultiviewDataset.from_single_view(data=get_test_data())
        batch = batchify(create_context_batch_from_dataset(dataset))
        query = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[-100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_valid_configuration_loss_common_keypoints_are_ignored(self):
        # Setup
        dataset = MultiviewDataset.from_single_view(data=get_test_data())
        batch = batchify(create_context_batch_from_dataset(dataset))
        query = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_valid_configuration_loss_bad_prediction_yields_nonzero_loss(self):
        # Setup
        dataset = MultiviewDataset.from_single_view(data=get_test_data())
        batch = batchify(create_context_batch_from_dataset(dataset))
        query = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[-100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertGreater(loss.item(), 0.0)

    def test_valid_configuration_loss_missing_exclusive_keypoints_are_valid(self):
        # Setup
        dataset = MultiviewDataset.from_single_view(data=get_test_data())
        batch = batchify(create_context_batch_from_dataset(dataset))
        query = torch.tensor([[0, 0, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_query_from_class_samples(self):
        # Setup
        class_idxs = np.expand_dims(np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8), -1)
        class_labels = torch.from_numpy(
            np.unpackbits(class_idxs, axis=-1, bitorder="little").astype(bool)
        )
        present_classes = [1, 2]

        # Action
        query = query_from_class_samples(
            class_labels, present_classes, exclusive_points_only=False
        )
        exclusive_query = query_from_class_samples(
            class_labels, present_classes, exclusive_points_only=True
        )

        # Verification
        self.assertTrue(
            torch.all(query == torch.tensor([True, True, False, False, True, True]))
        )
        self.assertTrue(
            torch.all(
                exclusive_query
                == torch.tensor([True, True, False, False, False, False])
            )
        )

    def test_valid_query_generator(self):
        # Setup
        rng = torch.Generator()
        rng.manual_seed(98764)
        class_idxs = np.expand_dims(
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [1, 1, 1, 1, 1, 1],
                    [16, 16, 8, 8, 4, 4],
                ],
                dtype=np.uint8,
            ),
            -1,
        )
        class_labels = torch.from_numpy(
            np.unpackbits(class_idxs, bitorder="little", axis=-1).astype(bool)
        )

        # Action
        queries = generate_valid_queries(class_labels, rng)

        # Verification
        self.assertEqual(class_labels.shape[:-1], queries.shape)

    def test_invalid_query_generator(self):
        # Setup
        rng = torch.Generator()
        rng.manual_seed(12345678)
        class_idxs = np.expand_dims(
            np.array(
                [
                    [1, 2, 3, 4, 5, 6],
                    [1, 1, 1, 1, 1, 1],
                    [16, 16, 8, 8, 4, 4],
                ],
                dtype=np.uint8,
            ),
            -1,
        )
        class_labels = torch.from_numpy(
            np.unpackbits(class_idxs, bitorder="little", axis=-1).astype(bool)
        )

        valid_queries = torch.tensor(
            [
                [True, True, False, False, False, False],
                [True, True, True, True, True, True],
                [True, True, False, False, True, True],
            ]
        )

        # Action
        invalid_queries = generate_invalid_queries(class_labels, valid_queries, rng)

        # Verification
        self.assertTrue(valid_queries.shape == invalid_queries.shape)
        self.assertFalse(torch.all(valid_queries == invalid_queries))

    def test_is_valid_configuration(self):
        # Setup
        queries = torch.tensor(
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
        class_labels = torch.tensor(
            [[True, False], [True, False], [False, True], [False, True]]
        ).repeat(queries.shape[0], 1, 1)

        # Action
        labels = is_valid_configuration(class_labels, queries)

        # Verification
        for i, label in enumerate(labels):
            self.assertEqual(label, 1 if i < 4 else 0)

    def test_is_valid_configuration_shared_points(self):
        # Setup
        queries = torch.tensor(
            [
                # Valid Queries
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 0],
                [1, 1, 1],
                # Invalid Queries with one beacon
                [0, 0, 1],
            ],
            dtype=torch.bool,
        )
        class_labels = torch.tensor(
            [[True, False], [False, True], [True, True]]
        ).repeat(queries.shape[0], 1, 1)

        # Action
        labels = is_valid_configuration(class_labels, queries)

        # Verification
        for i, label in enumerate(labels):
            self.assertEqual(label, 1 if i < 7 else 0)

    def test_test_dataset_collator(self):
        # Setup
        dataset = MultiviewDataset.from_single_view(data=get_x_position_test_dataset())

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=test_dataset_collator
        )

        # Action
        batch, query = next(iter(loader))

        # Verification
        self.assertEqual(batch.context.x.shape, query.shape)
        self.assertEqual(batch.query.x.shape, query.shape)


if __name__ == "__main__":
    unittest.main()
