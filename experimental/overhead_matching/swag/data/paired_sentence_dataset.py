"""Dataset for paired template and LLM sentences.

This module provides a dataset that yields both template-generated and
LLM-generated sentences for each unique pruned_tags, enabling contrastive learning
between the two sentence types.
"""

import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Dataset, Sampler

from experimental.overhead_matching.swag.data.osm_sentence_generator import (
    OSMSentenceGenerator,
)
from experimental.overhead_matching.swag.data.sentence_dataset import (
    SentenceSample,
)


class PairedSentenceDataset(Dataset):
    """Dataset yielding both template and LLM sentences for each pruned_tags.

    For each unique pruned_tags, returns a tuple of (template_sample, llm_sample).
    Indexed by pruned_tags that have LLM sentences.
    """

    def __init__(
        self,
        llm_sentences: dict[frozenset, str],  # pruned_tags -> sentence
        generator: OSMSentenceGenerator | None = None,
        epoch: int = 0,
    ):
        """Initialize the paired dataset.

        Args:
            llm_sentences: Dictionary mapping pruned_tags to LLM sentence strings
            generator: OSMSentenceGenerator instance (created if None)
            epoch: Current epoch (used for seed variation)
        """
        self.generator = generator or OSMSentenceGenerator()
        self.llm_sentences = llm_sentences
        self.epoch = epoch

        # Index by unique pruned_tags
        self.pruned_tags_list = list(llm_sentences.keys())

    def __len__(self) -> int:
        return len(self.pruned_tags_list)

    def __getitem__(self, idx: int) -> tuple[SentenceSample, SentenceSample]:
        """Get a pair of (template_sample, llm_sample) for a pruned_tags.

        Args:
            idx: Index of the pruned_tags

        Returns:
            Tuple of (template_sample, llm_sample)
        """
        pruned_tags = self.pruned_tags_list[idx]

        # Convert frozenset to dict for sentence generation
        tags_dict = dict(pruned_tags)

        # Generate template sentence with RNG seeded by index + epoch
        seed = idx + self.epoch * len(self)
        rng = random.Random(seed)
        result = self.generator.generate_sentence(tags_dict, rng=rng)

        template_sample = SentenceSample(
            sentence=result.sentence,
            used_tags=result.used_tags,
            unused_tags=result.unused_tags,
            pruned_tags=pruned_tags,
            source="template",
        )

        # Get LLM sentence
        llm_sample = SentenceSample(
            sentence=self.llm_sentences[pruned_tags],
            used_tags={},  # Unknown which tags LLM used
            unused_tags={},  # Unknown
            pruned_tags=pruned_tags,
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
            dataset_size: Number of pruned_tags in the dataset
            batch_size: Number of pairs per batch (total samples = 2 * batch_size)
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


class PairedContrastiveBatchSampler(Sampler):
    """Batch sampler that groups pruned_tags by tag values for contrastive learning.

    Ensures batches contain multiple pruned_tags sharing the same tag value,
    enabling both base contrastive (template↔LLM) and tag-specific contrastive.
    """

    def __init__(
        self,
        pruned_tags_list: list[frozenset],
        contrastive_tags: list[str],
        batch_size: int,
        groups_per_batch: int = 32,
        samples_per_group: int = 4,
        seed: int = 42,
    ):
        """Initialize the sampler.

        Args:
            pruned_tags_list: List of unique pruned_tags (frozensets)
            contrastive_tags: Tag keys to group by (e.g., ["name", "addr:street"])
            batch_size: Total batch size (number of pairs; total samples = 2 * batch_size)
            groups_per_batch: Number of groups to sample per batch
            samples_per_group: Target samples per group (may be less if group is smaller)
            seed: Random seed
        """
        self.pruned_tags_list = pruned_tags_list
        self.batch_size = batch_size
        self.groups_per_batch = groups_per_batch
        self.samples_per_group = samples_per_group
        self.seed = seed
        self.epoch = 0

        # Build index: tag_value -> list of pruned_tags indices
        self.value_to_indices: dict[str, list[int]] = defaultdict(list)
        self.ungrouped_indices: list[int] = []

        for idx, pruned_tags in enumerate(pruned_tags_list):
            tags_dict = dict(pruned_tags)
            grouped = False
            for tag in contrastive_tags:
                if tag in tags_dict:
                    key = f"{tag}:{tags_dict[tag]}"
                    self.value_to_indices[key].append(idx)
                    grouped = True
            if not grouped:
                self.ungrouped_indices.append(idx)

        # Filter to groups with 2+ members
        self.valid_groups = [
            (key, indices)
            for key, indices in self.value_to_indices.items()
            if len(indices) >= 2
        ]

        # Count groups per contrastive tag
        groups_per_tag: dict[str, int] = defaultdict(int)
        for key, _ in self.valid_groups:
            for tag in contrastive_tags:
                if key.startswith(f"{tag}:"):
                    groups_per_tag[tag] += 1
                    break

        print(f"PairedContrastiveBatchSampler: {len(self.valid_groups)} groups with 2+ members, "
              f"{len(self.ungrouped_indices)} ungrouped items")
        print(f"  Groups per tag: {dict(groups_per_tag)}")

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices."""
        rng = random.Random(self.seed + self.epoch)

        # Shuffle groups and ungrouped
        groups = list(self.valid_groups)
        rng.shuffle(groups)
        ungrouped = list(self.ungrouped_indices)
        rng.shuffle(ungrouped)

        group_idx = 0
        ungrouped_idx = 0
        total_samples_yielded = 0

        while total_samples_yielded < len(self.pruned_tags_list):
            batch = []

            # Sample from groups
            groups_sampled = 0
            while groups_sampled < self.groups_per_batch and len(batch) < self.batch_size:
                if group_idx >= len(groups):
                    # Reshuffle and restart
                    rng.shuffle(groups)
                    group_idx = 0

                key, indices = groups[group_idx]
                group_idx += 1

                # Sample from this group
                n_samples = min(self.samples_per_group, len(indices), self.batch_size - len(batch))
                sampled = rng.sample(indices, n_samples)
                batch.extend(sampled)
                groups_sampled += 1

            # Fill remaining batch with ungrouped samples
            while len(batch) < self.batch_size:
                if ungrouped_idx >= len(ungrouped):
                    if not ungrouped:
                        break
                    rng.shuffle(ungrouped)
                    ungrouped_idx = 0

                batch.append(ungrouped[ungrouped_idx])
                ungrouped_idx += 1

            if len(batch) == 0:
                break

            yield batch
            total_samples_yielded += len(batch)

    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        return (len(self.pruned_tags_list) + self.batch_size - 1) // self.batch_size


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
