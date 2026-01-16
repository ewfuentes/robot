"""Dataset for paired template and LLM sentences.

This module provides a dataset that yields both template-generated and
LLM-generated sentences for each landmark, enabling contrastive learning
between the two sentence types.
"""

import random
from typing import Iterator

from torch.utils.data import Dataset, Sampler

from experimental.overhead_matching.swag.data.osm_sentence_generator import (
    OSMSentenceGenerator,
)
from experimental.overhead_matching.swag.data.sentence_dataset import (
    LandmarkRecord,
    SentenceSample,
)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    prune_landmark,
)


class PairedSentenceDataset(Dataset):
    """Dataset yielding both template and LLM sentences for each landmark.

    For each landmark, returns a tuple of (template_sample, llm_sample).
    Only includes landmarks that have both template capability and LLM sentences.
    """

    def __init__(
        self,
        landmarks: list[LandmarkRecord],
        llm_sentences: dict[frozenset, str],  # pruned_tags -> sentence
        generator: OSMSentenceGenerator | None = None,
        epoch: int = 0,
    ):
        """Initialize the paired dataset.

        Args:
            landmarks: List of LandmarkRecord objects
            llm_sentences: Dictionary mapping pruned_tags to LLM sentence strings
            generator: OSMSentenceGenerator instance (created if None)
            epoch: Current epoch (used for seed variation)
        """
        self.generator = generator or OSMSentenceGenerator()
        self.llm_sentences = llm_sentences
        self.epoch = epoch

        # Filter to landmarks that have LLM sentences
        self.landmarks = []
        self.pruned_tags_list = []
        for landmark in landmarks:
            pruned = prune_landmark(landmark.tags)
            if pruned in llm_sentences:
                self.landmarks.append(landmark)
                self.pruned_tags_list.append(pruned)

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, idx: int) -> tuple[SentenceSample, SentenceSample]:
        """Get a pair of (template_sample, llm_sample) for a landmark.

        Args:
            idx: Index of the landmark

        Returns:
            Tuple of (template_sample, llm_sample)
        """
        landmark = self.landmarks[idx]
        pruned = self.pruned_tags_list[idx]

        # Generate template sentence with RNG seeded by index + epoch
        seed = idx + self.epoch * len(self.landmarks)
        rng = random.Random(seed)
        result = self.generator.generate_sentence(landmark.tags, rng=rng)

        template_sample = SentenceSample(
            sentence=result.sentence,
            used_tags=result.used_tags,
            unused_tags=result.unused_tags,
            landmark_id=landmark.landmark_id,
            all_tags=landmark.tags,
            pruned_tags=pruned,
            source="template",
        )

        # Get LLM sentence
        llm_sample = SentenceSample(
            sentence=self.llm_sentences[pruned],
            used_tags={},  # Unknown which tags LLM used
            unused_tags={},  # Unknown
            landmark_id=landmark.landmark_id,
            all_tags=landmark.tags,
            pruned_tags=pruned,
            source="llm",
        )

        return template_sample, llm_sample

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for seed variation."""
        self.epoch = epoch


class PairedBatchSampler(Sampler):
    """Batch sampler that yields indices for paired dataset.

    Each batch contains indices, and the dataloader will get pairs
    (template, llm) for each index. The collate function then flattens
    these into a single batch with alternating template/llm samples.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = False,
    ):
        """Initialize the sampler.

        Args:
            dataset_size: Number of landmarks in the dataset
            batch_size: Number of landmark pairs per batch (total samples = 2 * batch_size)
            seed: Random seed
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices."""
        rng = random.Random(self.seed + self.epoch)

        indices = list(range(self.dataset_size))
        rng.shuffle(indices)

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """Number of batches per epoch."""
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


def collate_paired_samples(
    pairs: list[tuple[SentenceSample, SentenceSample]],
) -> list[SentenceSample]:
    """Collate function that flattens paired samples.

    Takes pairs of (template, llm) samples and returns a flat list
    with all samples for further processing by collate_sentences.

    Args:
        pairs: List of (template_sample, llm_sample) tuples

    Returns:
        Flat list of SentenceSample objects
    """
    flat = []
    for template_sample, llm_sample in pairs:
        flat.append(template_sample)
        flat.append(llm_sample)
    return flat
