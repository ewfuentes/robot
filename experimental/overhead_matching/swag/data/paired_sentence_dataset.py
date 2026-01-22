"""Dataset for paired template and LLM sentences.

This module provides a dataset that yields both template-generated and
LLM-generated sentences for each unique pruned_tags, enabling contrastive learning
between the two sentence types.
"""

import random

from torch.utils.data import Dataset

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


def flatten_samples(
    samples: list,
) -> list[SentenceSample]:
    """Flatten a list of samples that may contain tuples.

    Takes a list where each element is either:
    - A single SentenceSample
    - A tuple of SentenceSample objects (pairs, triples, etc.)

    And returns a flat list of all samples.

    Args:
        samples: List of samples or tuples of samples

    Returns:
        Flat list of SentenceSample objects
    """
    flat = []
    for item in samples:
        if isinstance(item, tuple):
            flat.extend(item)
        else:
            flat.append(item)
    return flat
