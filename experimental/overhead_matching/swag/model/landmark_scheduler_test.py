"""Tests for landmark dropout scheduler functionality."""

import unittest
from experimental.overhead_matching.swag.model.landmark_scheduler import (
    LandmarkDropoutScheduleConfig,
    LandmarkDropoutScheduler,
)


class LandmarkDropoutScheduleTest(unittest.TestCase):
    """Tests for LandmarkDropoutSchedule configuration validation."""

    def test_valid_schedule(self):
        """Test that valid schedules are created without errors."""
        schedule = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=0.0,
            final_dropout_rate=0.8,
            extractor_names=["test_extractor"],
            min_landmarks=5
        )
        self.assertEqual(schedule.start_progress, 0.0)
        self.assertEqual(schedule.end_progress, 1.0)
        self.assertEqual(schedule.initial_dropout_rate, 0.0)
        self.assertEqual(schedule.final_dropout_rate, 0.8)
        self.assertEqual(schedule.extractor_names, ["test_extractor"])
        self.assertEqual(schedule.min_landmarks, 5)

    def test_partial_schedule(self):
        """Test schedule that only covers part of training."""
        schedule = LandmarkDropoutScheduleConfig(
            start_progress=0.2,
            end_progress=0.8,
            initial_dropout_rate=0.1,
            final_dropout_rate=0.9,
            extractor_names=["extractor1", "extractor2"]
        )
        self.assertEqual(schedule.start_progress, 0.2)
        self.assertEqual(schedule.end_progress, 0.8)


class LandmarkDropoutSchedulerTest(unittest.TestCase):
    """Tests for LandmarkDropoutScheduler computation logic."""

    def test_linear_interpolation(self):
        """Test that dropout rates are correctly interpolated."""
        schedule = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=0.0,
            final_dropout_rate=1.0,
            extractor_names=["test"]
        )
        scheduler = LandmarkDropoutScheduler([schedule], total_epochs=100)

        # At start
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=0),
            0.0,
            places=5
        )

        # At 25%
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=25),
            0.25,
            places=5
        )

        # At 50%
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=50),
            0.5,
            places=5
        )

        # At 75%
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=75),
            0.75,
            places=5
        )

        # At end
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=100),
            1.0,
            places=5
        )

    def test_partial_progress_schedule(self):
        """Test schedule that only ramps during middle of training."""
        schedule = LandmarkDropoutScheduleConfig(
            start_progress=0.25,
            end_progress=0.75,
            initial_dropout_rate=0.2,
            final_dropout_rate=0.8,
            extractor_names=["test"]
        )
        scheduler = LandmarkDropoutScheduler([schedule], total_epochs=100)

        # Before start: should be at initial rate
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=0),
            0.2,
            places=5
        )
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=20),
            0.2,
            places=5
        )

        # At start (25% progress = epoch 25)
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=25),
            0.2,
            places=5
        )

        # At midpoint (50% progress = epoch 50, halfway through schedule)
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=50),
            0.5,
            places=5
        )

        # At end (75% progress = epoch 75)
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=75),
            0.8,
            places=5
        )

        # After end: should be at final rate
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=80),
            0.8,
            places=5
        )
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=100),
            0.8,
            places=5
        )

    def test_decreasing_dropout(self):
        """Test schedule that decreases dropout rate over time."""
        schedule = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=1.0,
            final_dropout_rate=0.0,
            extractor_names=["synthetic_extractor"]
        )
        scheduler = LandmarkDropoutScheduler([schedule], total_epochs=50)

        # At start: maximum dropout
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=0),
            1.0,
            places=5
        )

        # At middle: half dropout
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=25),
            0.5,
            places=5
        )

        # At end: no dropout
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=50),
            0.0,
            places=5
        )

    def test_constant_dropout(self):
        """Test schedule with same initial and final rates (constant)."""
        schedule = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=0.5,
            final_dropout_rate=0.5,
            extractor_names=["test"]
        )
        scheduler = LandmarkDropoutScheduler([schedule], total_epochs=100)

        # Should be constant throughout
        for epoch in [0, 25, 50, 75, 100]:
            self.assertAlmostEqual(
                scheduler.compute_dropout_rate(schedule, epoch=epoch),
                0.5,
                places=5
            )

    def test_multiple_extractors(self):
        """Test get_dropout_rates_by_extractor with multiple schedules."""
        schedule1 = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=0.0,
            final_dropout_rate=0.8,
            extractor_names=["extractor1"],
            min_landmarks=5
        )
        schedule2 = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=0.0,
            final_dropout_rate=0.6,
            extractor_names=["extractor2", "extractor3"],
            min_landmarks=3
        )
        scheduler = LandmarkDropoutScheduler([schedule1, schedule2], total_epochs=100)

        # At epoch 50 (50% progress)
        rates = scheduler.get_dropout_rates_by_extractor(epoch=50)

        self.assertIn("extractor1", rates)
        self.assertIn("extractor2", rates)
        self.assertIn("extractor3", rates)

        # extractor1 should be at 0.4 (50% of 0.0->0.8)
        self.assertAlmostEqual(rates["extractor1"][0], 0.4, places=5)
        self.assertEqual(rates["extractor1"][1], 5)  # min_landmarks

        # extractor2 and extractor3 should be at 0.3 (50% of 0.0->0.6)
        self.assertAlmostEqual(rates["extractor2"][0], 0.3, places=5)
        self.assertEqual(rates["extractor2"][1], 3)  # min_landmarks
        self.assertAlmostEqual(rates["extractor3"][0], 0.3, places=5)
        self.assertEqual(rates["extractor3"][1], 3)  # min_landmarks

    def test_duplicate_extractor_validation(self):
        """Test that duplicate extractors across schedules raises error."""
        schedule1 = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=0.5,
            initial_dropout_rate=0.0,
            final_dropout_rate=0.5,
            extractor_names=["extractor1"]
        )
        schedule2 = LandmarkDropoutScheduleConfig(
            start_progress=0.5,
            end_progress=1.0,
            initial_dropout_rate=0.5,
            final_dropout_rate=1.0,
            extractor_names=["extractor1"]  # Duplicate!
        )

        with self.assertRaises(ValueError) as context:
            LandmarkDropoutScheduler([schedule1, schedule2], total_epochs=100)

        self.assertIn("Duplicate extractors", str(context.exception))

    def test_zero_total_epochs(self):
        """Test scheduler behavior with zero total epochs."""
        schedule = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=0.0,
            final_dropout_rate=1.0,
            extractor_names=["test"]
        )
        scheduler = LandmarkDropoutScheduler([schedule], total_epochs=0)

        # Should return initial dropout rate
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule, epoch=0),
            0.0,
            places=5
        )

    def test_edge_case_dropout_rates(self):
        """Test edge cases with dropout rates at 0.0 and 1.0."""
        # Test 0.0 dropout (keep all features)
        schedule_zero = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=0.0,
            final_dropout_rate=0.0,
            extractor_names=["test_zero"]
        )
        scheduler = LandmarkDropoutScheduler([schedule_zero], total_epochs=100)
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule_zero, epoch=50),
            0.0,
            places=5
        )

        # Test 1.0 dropout (drop all features)
        schedule_one = LandmarkDropoutScheduleConfig(
            start_progress=0.0,
            end_progress=1.0,
            initial_dropout_rate=1.0,
            final_dropout_rate=1.0,
            extractor_names=["test_one"]
        )
        scheduler = LandmarkDropoutScheduler([schedule_one], total_epochs=100)
        self.assertAlmostEqual(
            scheduler.compute_dropout_rate(schedule_one, epoch=50),
            1.0,
            places=5
        )


if __name__ == '__main__':
    unittest.main()
