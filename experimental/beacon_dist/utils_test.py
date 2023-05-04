import unittest

import torch

from experimental.beacon_dist.test_helpers import get_test_data
from experimental.beacon_dist.utils import (
    Dataset,
    reconstruction_loss,
    sample_keypoints,
    batchify,
    valid_configuration_loss,
    generate_valid_queries,
    generate_invalid_queries,
    query_from_class_samples,
    is_valid_configuration,
    test_dataset_collator,
    get_x_position_test_dataset,
)


class DatasetTest(unittest.TestCase):
    def test_dataset_from_filename(self):
        # Setup
        test_data = get_test_data()
        # Action
        dataset = Dataset(data=test_data)

        # Verification
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[1].descriptor[0, 0], 4)


class UtilsTest(unittest.TestCase):
    def test_sample_keypoints_subsample(self):
        # Setup
        gen = torch.manual_seed(0)
        test_data = get_test_data()
        dataset = Dataset(data=test_data)

        # Action
        subsampled = sample_keypoints(dataset[0], num_keypoints_to_sample=2, gen=gen)

        # Verification
        # There should be two keypoints even though the original sample had 3 points
        self.assertEqual(subsampled.x.shape[0], 2)

    def test_batchify(self):
        # Setup
        test_data = get_test_data()
        dataset = Dataset(data=test_data)
        batch = dataset._data

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

    def test_reconstructor_loss(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch_in = batchify(dataset._data)
        batch_out = batch_in._replace(x=batch_in.x + 0.1)
        expected_loss = torch.mean((batch_out.x - batch_in.x) ** 2)

        # Action
        loss = reconstruction_loss(batch_in, batch_out)

        # Verification
        self.assertEqual(loss, expected_loss)

    def test_valid_configuration_loss_valid_queries_are_likely(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_valid_configuration_loss_invalid_queries_are_unlikely(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[-100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_valid_configuration_loss_common_keypoints_are_ignored(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss.item(), 0.0)

    def test_valid_configuration_loss_bad_prediction_yields_nonzero_loss(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[-100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertGreater(loss, 0.0)

    def test_valid_configuration_loss_missing_exclusive_keypoints_are_valid(self):
        # Setup
        dataset = Dataset(data=get_test_data())
        batch = batchify(dataset._data)
        query = torch.tensor([[0, 0, 1], [1, 1, 0], [1, 0, 0]])
        model_output = torch.tensor([[100.0], [100.0], [100.0]])

        # Action
        loss = valid_configuration_loss(batch.class_label, query, model_output)

        # Verification
        self.assertAlmostEqual(loss, 0.0)

    def test_query_from_class_samples(self):
        # Setup
        class_labels = torch.tensor([1, 2, 3, 4, 5, 6])
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
        class_labels = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [1, 1, 1, 1, 1, 1],
                [16, 16, 8, 8, 4, 4],
            ]
        )

        # Action
        queries = generate_valid_queries(class_labels, rng)

        # Verification
        self.assertEqual(class_labels.shape, queries.shape)

    def test_invalid_query_generator(self):
        # Setup
        rng = torch.Generator()
        rng.manual_seed(12345678)
        class_labels = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [1, 1, 1, 1, 1, 1],
                [16, 16, 8, 8, 4, 4],
            ]
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
        class_labels = torch.tensor([1, 1, 2, 2]).repeat(queries.shape[0], 1)

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
        class_labels = torch.tensor([1, 2, 3]).repeat(queries.shape[0], 1)

        # Action
        labels = is_valid_configuration(class_labels, queries)

        # Verification
        for i, label in enumerate(labels):
            self.assertEqual(label, 1 if i < 7 else 0)

    def test_test_dataset_collator(self):
        # Setup
        dataset = Dataset(data=get_x_position_test_dataset())

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=test_dataset_collator
        )

        # Action
        batch, query = next(iter(loader))
        batch = batchify(batch)

        # Verification
        self.assertEqual(batch.x.shape, query.shape)


if __name__ == "__main__":
    unittest.main()
