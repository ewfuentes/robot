"""Scheduled landmark dropout for training with gradual feature ramping.

This module provides functionality to gradually ramp up or down the dropout rate
of landmarks from specific extractors during training. This enables smooth transitions
from synthetic features to noisier realistic features.

The scheduler operates at epoch boundaries and applies dropout by modifying the masks
in the extractor outputs within the model's forward pass.
"""

from typing import Dict
from experimental.overhead_matching.swag.model.swag_config_types import LandmarkDropoutSchedule


class LandmarkDropoutScheduler:
    """Applies landmark dropout schedules during training.

    This class computes the current dropout rate for each schedule based on
    training progress and can be used to apply dropout to extractor outputs.
    """

    def __init__(self, schedules: list[LandmarkDropoutSchedule], total_epochs: int):
        """Initialize the scheduler.

        Args:
            schedules: List of dropout schedules to apply
            total_epochs: Total number of training epochs
        """
        self.schedules = schedules
        self.total_epochs = total_epochs

        # Validate that no extractor appears in multiple schedules
        all_extractors = []
        for schedule in schedules:
            all_extractors.extend(schedule.extractor_names)

        if len(all_extractors) != len(set(all_extractors)):
            duplicates = [x for x in all_extractors if all_extractors.count(x) > 1]
            raise ValueError(
                f"Each extractor can only appear in one schedule. "
                f"Duplicate extractors: {set(duplicates)}")

    def compute_dropout_rate(self, schedule: LandmarkDropoutSchedule, epoch: int) -> float:
        """Compute the current dropout rate for a schedule.

        Args:
            schedule: The schedule to compute the rate for
            epoch: Current epoch number (0-indexed)

        Returns:
            Dropout rate in [0.0, 1.0] where 0.0 = keep all, 1.0 = drop all
        """
        if self.total_epochs == 0:
            return schedule.initial_dropout_rate

        # Compute current progress through training
        progress = epoch / self.total_epochs

        # Before schedule starts
        if progress < schedule.start_progress:
            return schedule.initial_dropout_rate

        # After schedule ends
        if progress >= schedule.end_progress:
            return schedule.final_dropout_rate

        # During schedule: linear interpolation
        schedule_progress = (progress - schedule.start_progress) / \
                          (schedule.end_progress - schedule.start_progress)

        dropout_rate = (schedule.initial_dropout_rate +
                       schedule_progress * (schedule.final_dropout_rate -
                                          schedule.initial_dropout_rate))

        return dropout_rate

    def get_dropout_rates_by_extractor(self, epoch: int) -> Dict[str, tuple[float, int]]:
        """Get current dropout rates for all extractors.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Dictionary mapping extractor names to (dropout_rate, min_landmarks) tuples
        """
        rates = {}
        for schedule in self.schedules:
            dropout_rate = self.compute_dropout_rate(schedule, epoch)
            for extractor_name in schedule.extractor_names:
                rates[extractor_name] = (dropout_rate, schedule.min_landmarks)
        return rates
