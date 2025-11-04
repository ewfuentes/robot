"""Scheduled landmark dropout for training with gradual feature ramping.

This module provides functionality to gradually ramp up or down the dropout rate
of landmarks from specific extractors during training. This enables smooth transitions
from synthetic features to noisier realistic features.

The scheduler operates at epoch boundaries and applies dropout by modifying the masks
in the extractor outputs within the model's forward pass.
"""

import hashlib
import common.torch.load_torch_deps
import torch
from typing import Dict

from experimental.overhead_matching.swag.model.swag_config_types import LandmarkDropoutSchedule
from experimental.overhead_matching.swag.model.swag_model_input_output import ExtractorOutput, ModelInput


class LandmarkDropoutScheduler:
    """Applies landmark dropout schedules during training.

    This class maintains state about the current epoch and computes dropout rates
    based on training progress. It can be passed to models to apply dropout to
    extractor outputs.
    """

    def __init__(self, schedules: list[LandmarkDropoutSchedule], total_epochs: int):
        """Initialize the scheduler.

        Args:
            schedules: List of dropout schedules to apply
            total_epochs: Total number of training epochs
        """
        self.schedules = schedules
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Validate that no extractor appears in multiple schedules
        all_extractors = []
        for schedule in schedules:
            all_extractors.extend(schedule.extractor_names)

        if len(all_extractors) != len(set(all_extractors)):
            duplicates = [x for x in all_extractors if all_extractors.count(x) > 1]
            raise ValueError(
                f"Each extractor can only appear in one schedule. "
                f"Duplicate extractors: {set(duplicates)}")

    def set_epoch(self, epoch: int):
        """Update the current epoch.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self.current_epoch = epoch

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

    def apply_dropout(
        self,
        extractor_outputs_by_name: dict[str, ExtractorOutput],
        model_input: ModelInput,
    ) -> dict[str, ExtractorOutput]:
        """Apply dropout to extractor outputs based on current epoch.

        This method modifies the masks in the extractor outputs to implement dropout
        according to the scheduler's current state. The dropout is:
        - Deterministic: Same sample always drops same landmarks (based on sample ID)
        - Per-extractor: Different extractors can have different dropout schedules
        - Progressive: Dropout rate changes linearly over training based on epoch progress

        Args:
            extractor_outputs_by_name: Dictionary mapping extractor names to their outputs.
                The masks in these outputs will be modified in-place.
            model_input: Model input containing metadata with sample IDs for deterministic seeding.

        Returns:
            The same extractor_outputs_by_name dictionary with modified masks. The dictionary
            is modified in-place, but returned for convenience.
        """
        return apply_landmark_dropout_schedules(
            extractor_outputs_by_name,
            self.schedules,
            self.current_epoch,
            self.total_epochs,
            model_input,
        )


def apply_landmark_dropout_schedules(
    extractor_outputs_by_name: dict[str, ExtractorOutput],
    schedules: list[LandmarkDropoutSchedule],
    current_epoch: int,
    total_epochs: int,
    model_input: ModelInput,
) -> dict[str, ExtractorOutput]:
    """Apply scheduled landmark dropout to extractor outputs.

    This function modifies the masks in the extractor outputs to implement dropout
    according to the provided schedules. The dropout is:
    - Deterministic: Same sample always drops same landmarks (based on sample ID)
    - Per-extractor: Different extractors can have different dropout schedules
    - Progressive: Dropout rate changes linearly over training based on epoch progress

    Args:
        extractor_outputs_by_name: Dictionary mapping extractor names to their outputs.
            The masks in these outputs will be modified in-place.
        schedules: List of dropout schedules to apply. Each schedule specifies which
            extractors to affect and how the dropout rate should change over training.
        current_epoch: Current training epoch (0-indexed).
        total_epochs: Total number of training epochs.
        model_input: Model input containing metadata with sample IDs for deterministic seeding.

    Returns:
        The same extractor_outputs_by_name dictionary with modified masks. The dictionary
        is modified in-place, but returned for convenience.
    """
    if len(schedules) == 0:
        return extractor_outputs_by_name

    # Determine if this is panorama input (has pano_id) vs satellite (has sat_id)
    assert len(model_input.metadata) > 0
    is_panorama = 'pano_id' in model_input.metadata[0]

    # Compute training progress as fraction of total epochs
    progress = current_epoch / total_epochs

    for schedule in schedules:
        # Compute current dropout rate for this schedule based on training progress
        if progress < schedule.start_progress:
            dropout_rate = schedule.initial_dropout_rate
        elif progress >= schedule.end_progress:
            dropout_rate = schedule.final_dropout_rate
        else:
            # Linear interpolation between start and end
            schedule_progress = (progress - schedule.start_progress) / \
                              (schedule.end_progress - schedule.start_progress)
            dropout_rate = (schedule.initial_dropout_rate +
                          schedule_progress * (schedule.final_dropout_rate -
                                              schedule.initial_dropout_rate))

        # Apply dropout to each extractor in this schedule
        for extractor_name in schedule.extractor_names:
            if extractor_name not in extractor_outputs_by_name:
                continue

            output = extractor_outputs_by_name[extractor_name]
            batch_size = output.features.shape[0]

            # Process each batch item independently with deterministic seeding
            for batch_idx in range(batch_size):
                # Get sample ID for deterministic seeding
                # For panoramas, use pano_id; for satellites, use sat_id
                if is_panorama:
                    sample_id = model_input.metadata[batch_idx].get('pano_id', batch_idx)
                else:
                    sample_id = model_input.metadata[batch_idx].get('sat_id', batch_idx)

                # Generate deterministic seed using sample_id and extractor_name
                # Use hashlib for consistency across runs (hash() is not deterministic)
                seed = _deterministic_seed(sample_id, extractor_name)
                rng = torch.Generator().manual_seed(seed)

                # Count valid (unmasked) landmarks
                valid_mask = ~output.mask[batch_idx]
                num_valid = valid_mask.sum().item()

                # Compute how many landmarks to keep after dropout
                num_to_drop = int(num_valid * dropout_rate)
                num_to_keep = max(num_valid - num_to_drop, schedule.min_landmarks)
                num_to_keep = min(num_to_keep, num_valid)  # Can't keep more than we have

                # Apply dropout by modifying the mask
                if num_to_keep < num_valid:
                    # Get indices of valid landmarks
                    valid_indices = torch.where(valid_mask)[0]
                    # Randomly select which to keep (deterministic via seed)
                    perm = torch.randperm(len(valid_indices), generator=rng)
                    keep_indices = valid_indices[perm[:num_to_keep]]
                    # Create new mask: True = masked (dropped), False = kept
                    new_mask = torch.ones_like(output.mask[batch_idx], dtype=torch.bool)
                    new_mask[keep_indices] = False
                    output.mask[batch_idx] = new_mask
                # Handle edge cases
                elif dropout_rate >= 1.0 and num_valid > 0:
                    # Drop all features (set all to masked)
                    output.mask[batch_idx] = torch.ones_like(
                        output.mask[batch_idx], dtype=torch.bool
                    )
                elif dropout_rate <= 0.0:
                    # Keep all features (leave mask unchanged)
                    pass

    return extractor_outputs_by_name


def _deterministic_seed(sample_id: str | int, extractor_name: str) -> int:
    """Generate a deterministic seed from sample ID and extractor name.

    Uses SHA256 hashing to ensure consistency across Python runs (unlike hash()).

    Args:
        sample_id: Unique identifier for the sample (pano_id or sat_id).
        extractor_name: Name of the extractor.

    Returns:
        A 32-bit unsigned integer seed suitable for torch.Generator.manual_seed().
    """
    # Combine sample_id and extractor_name into a string
    seed_string = f"{sample_id}_{extractor_name}"

    # Hash using SHA256 for deterministic, cross-run consistency
    hash_bytes = hashlib.sha256(seed_string.encode('utf-8')).digest()

    # Convert first 4 bytes to a 32-bit unsigned integer
    seed = int.from_bytes(hash_bytes[:4], byteorder='big')

    return seed
