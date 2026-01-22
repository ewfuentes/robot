"""Dataset and data loading for sentence embedding training.

Loads landmarks from SQLite database and generates sentences on-the-fly
using OSMSentenceGenerator. This provides more variety (different seeds
per epoch) and eliminates the need for pre-extracted sentence files.
"""

import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
from torch.utils.data import Dataset, Sampler

from experimental.overhead_matching.swag.data.osm_sentence_generator import (
    OSMSentenceGenerator,
)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    prune_landmark,
)


@dataclass
class LandmarkRecord:
    """A landmark loaded from the database."""

    landmark_id: int
    tags: dict[str, str]


@dataclass
class SentenceSample:
    """A generated sentence sample for training."""

    sentence: str
    used_tags: dict[str, str]
    unused_tags: dict[str, str]
    pruned_tags: frozenset  # For landmark identity matching
    source: Literal["template", "llm"]  # Source of the sentence


@dataclass
class SentenceBatch:
    """Batch with precomputed labels for efficient loss computation."""

    sentences: list[str]
    # Classification: {task_name: (mask, labels)} where mask[i]=True if task applies
    classification_labels: dict[str, tuple[torch.Tensor, torch.Tensor]]
    # Contrastive: {task_name: (mask, positive_matrix)} where positive_matrix[i,j]=1 if same value
    contrastive_labels: dict[str, tuple[torch.Tensor, torch.Tensor]]
    # Presence: {task_name: labels} where labels[i]=1 if tag is in used_tags
    presence_labels: dict[str, torch.Tensor]
    # Base contrastive: (mask, positive_matrix) for same-landmark pairing (template <-> LLM)
    base_contrastive_labels: tuple[torch.Tensor, torch.Tensor] | None
    # Template mask: mask[i]=True if sample is from template (for template-only losses)
    template_mask: torch.Tensor | None
    # Pruned tags for each sample (for global MRR computation)
    pruned_tags: list[frozenset] | None = None
    # Pre-tokenized input (if tokenizer provided to collate function)
    token_ids: dict[str, torch.Tensor] | None = None

    def to(self, device: torch.device) -> "SentenceBatch":
        """Move tensors to device."""
        base_contrastive = None
        if self.base_contrastive_labels is not None:
            base_contrastive = (
                self.base_contrastive_labels[0].to(device),
                self.base_contrastive_labels[1].to(device),
            )
        token_ids = None
        if self.token_ids is not None:
            token_ids = {k: v.to(device) for k, v in self.token_ids.items()}
        return SentenceBatch(
            sentences=self.sentences,
            classification_labels={
                task: (mask.to(device), labels.to(device))
                for task, (mask, labels) in self.classification_labels.items()
            },
            contrastive_labels={
                task: (mask.to(device), pos_matrix.to(device))
                for task, (mask, pos_matrix) in self.contrastive_labels.items()
            },
            presence_labels={
                task: labels.to(device)
                for task, labels in self.presence_labels.items()
            },
            base_contrastive_labels=base_contrastive,
            template_mask=self.template_mask.to(device) if self.template_mask is not None else None,
            pruned_tags=self.pruned_tags,  # No need to move, it's a list of frozensets
            token_ids=token_ids,
        )


class SentenceDataset(Dataset):
    """Dataset indexed by unique pruned_tags that generates sentences on-the-fly.

    Sentences are generated using OSMSentenceGenerator with a seed that combines
    the index and current epoch, providing variety across epochs while
    maintaining reproducibility within an epoch.
    """

    def __init__(
        self,
        pruned_tags_list: list[frozenset],
        generator: OSMSentenceGenerator | None = None,
        epoch: int = 0,
    ):
        """Initialize dataset.

        Args:
            pruned_tags_list: List of unique pruned_tags (frozensets)
            generator: OSMSentenceGenerator instance (created if None)
            epoch: Current epoch (used for seed variation)
        """
        self.pruned_tags_list = pruned_tags_list
        self.generator = generator or OSMSentenceGenerator()
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.pruned_tags_list)

    def __getitem__(self, idx: int) -> SentenceSample:
        pruned_tags = self.pruned_tags_list[idx]

        # Convert frozenset to dict for sentence generation
        tags_dict = dict(pruned_tags)

        # Generate sentence with RNG seeded by index + epoch for variety
        seed = idx + self.epoch * len(self)
        rng = random.Random(seed)
        result = self.generator.generate_sentence(tags_dict, rng=rng)

        return SentenceSample(
            sentence=result.sentence,
            used_tags=result.used_tags,
            unused_tags=result.unused_tags,
            pruned_tags=pruned_tags,
            source="template",
        )

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for seed variation."""
        self.epoch = epoch

    @classmethod
    def from_landmarks(
        cls,
        landmarks: list[LandmarkRecord],
        generator: OSMSentenceGenerator | None = None,
        epoch: int = 0,
    ) -> "SentenceDataset":
        """Create dataset from landmarks, extracting unique pruned_tags.

        Args:
            landmarks: List of LandmarkRecord objects
            generator: OSMSentenceGenerator instance
            epoch: Current epoch

        Returns:
            SentenceDataset instance indexed by unique pruned_tags
        """
        # Extract unique pruned_tags from landmarks
        unique_pruned_tags = list({prune_landmark(lm.tags) for lm in landmarks})
        return cls(unique_pruned_tags, generator=generator, epoch=epoch)

    @classmethod
    def from_database(
        cls,
        db_path: Path,
        generator: OSMSentenceGenerator | None = None,
        epoch: int = 0,
        limit: int | None = None,
    ) -> "SentenceDataset":
        """Load dataset from SQLite database.

        Args:
            db_path: Path to landmarks SQLite database
            generator: OSMSentenceGenerator instance
            epoch: Current epoch
            limit: Optional limit on number of landmarks to load

        Returns:
            SentenceDataset instance
        """
        landmarks = load_landmarks_from_db(db_path, limit=limit)
        return cls.from_landmarks(landmarks, generator=generator, epoch=epoch)


def load_landmarks_from_db(
    db_path: Path,
    limit: int | None = None,
) -> list[LandmarkRecord]:
    """Load all landmarks with their tags from SQLite database.

    Args:
        db_path: Path to landmarks database
        limit: Optional limit on number of landmarks

    Returns:
        List of LandmarkRecord objects
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Query to get landmarks with their tags
    # Uses GROUP_CONCAT to aggregate tags into a single string per landmark
    query = """
        SELECT
            l.id,
            GROUP_CONCAT(tk.key || '=' || tv.value, '|') as tag_pairs
        FROM landmarks l
        LEFT JOIN tags t ON l.id = t.landmark_id
        LEFT JOIN tag_keys tk ON t.key_id = tk.id
        LEFT JOIN tag_values tv ON t.value_id = tv.id
        GROUP BY l.id
    """

    if limit is not None:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query)

    landmarks = []
    for row in cursor:
        tags: dict[str, str] = {}
        if row["tag_pairs"]:
            for pair in row["tag_pairs"].split("|"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    tags[k] = v

        landmarks.append(
            LandmarkRecord(
                landmark_id=row["id"],
                tags=tags,
            )
        )

    conn.close()
    return landmarks


def load_tag_vocabularies_from_db(
    db_path: Path,
    classification_tags: list[str],
    min_count: int = 100,
) -> dict[str, dict[str, int]]:
    """Build tag vocabularies directly from database.

    Args:
        db_path: Path to landmarks database
        classification_tags: List of tag keys to build vocabularies for
        min_count: Minimum count for a value to be included

    Returns:
        Dictionary mapping tag keys to {value: class_index} dictionaries
    """
    conn = sqlite3.connect(db_path)

    vocabularies = {}

    for tag_key in classification_tags:
        # Count occurrences of each value for this tag key
        query = """
            SELECT tv.value, COUNT(*) as cnt
            FROM tags t
            JOIN tag_keys tk ON t.key_id = tk.id
            JOIN tag_values tv ON t.value_id = tv.id
            WHERE tk.key = ?
            GROUP BY tv.value
            HAVING cnt >= ?
            ORDER BY cnt DESC
        """
        cursor = conn.execute(query, (tag_key, min_count))

        values = [row[0] for row in cursor]
        if values:
            vocabularies[tag_key] = {v: i for i, v in enumerate(values)}

    conn.close()
    return vocabularies


def split_landmarks_by_id(
    landmarks: list[LandmarkRecord],
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[list[LandmarkRecord], list[LandmarkRecord]]:
    """Split landmarks into train/test sets by landmark_id.

    Args:
        landmarks: List of all landmarks
        train_fraction: Fraction of landmarks for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_landmarks, test_landmarks)
    """
    import random

    # Shuffle landmark indices
    indices = list(range(len(landmarks)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    # Split
    split_idx = int(len(indices) * train_fraction)
    train_indices = set(indices[:split_idx])

    train_landmarks = [landmarks[i] for i in range(len(landmarks)) if i in train_indices]
    test_landmarks = [landmarks[i] for i in range(len(landmarks)) if i not in train_indices]

    return train_landmarks, test_landmarks


def split_pruned_tags(
    pruned_tags_list: list[frozenset],
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[list[frozenset], list[frozenset]]:
    """Split pruned_tags into train/test sets.

    Args:
        pruned_tags_list: List of unique pruned_tags
        train_fraction: Fraction of items for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_pruned_tags, test_pruned_tags)
    """
    # Shuffle indices
    indices = list(range(len(pruned_tags_list)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    # Split
    split_idx = int(len(indices) * train_fraction)
    train_indices = set(indices[:split_idx])

    train_tags = [pruned_tags_list[i] for i in range(len(pruned_tags_list)) if i in train_indices]
    test_tags = [pruned_tags_list[i] for i in range(len(pruned_tags_list)) if i not in train_indices]

    return train_tags, test_tags


def get_unique_pruned_tags_from_landmarks(landmarks: list[LandmarkRecord]) -> list[frozenset]:
    """Extract unique pruned_tags from a list of landmarks.

    Args:
        landmarks: List of LandmarkRecord objects

    Returns:
        List of unique pruned_tags (frozensets)
    """
    return list({prune_landmark(lm.tags) for lm in landmarks})


class CombinedDataset(Dataset):
    """Dataset that combines multiple datasets for mixed batch sampling.

    Each index is a (source_key, idx) tuple where source_key identifies the dataset.
    This is a top-level class (not a closure) to enable pickling for DataLoader workers.
    """

    def __init__(
        self,
        datasets: dict[str, Dataset],
    ):
        """Initialize combined dataset.

        Args:
            datasets: Dictionary mapping source keys to datasets.
        """
        self.datasets = datasets

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, key: tuple[str, int]):
        """Get item from the appropriate dataset.

        Args:
            key: Tuple of (source_key, index)

        Returns:
            Item from the specified dataset at the given index.
        """
        source_key, idx = key
        return self.datasets[source_key][idx]


class ContrastiveBatchSampler(Sampler):
    """Batch sampler that ensures positive pairs for contrastive learning.

    Groups pruned_tags by their contrastive tag values and samples batches
    that contain multiple items sharing the same value.
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
            batch_size: Total batch size
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
        # An item can belong to multiple groups (one per contrastive tag it has)
        self.value_to_indices: dict[str, list[int]] = defaultdict(list)
        self.ungrouped_indices: list[int] = []

        for idx, pruned_tags in enumerate(pruned_tags_list):
            tags_dict = dict(pruned_tags)
            grouped = False
            for tag in contrastive_tags:
                if tag in tags_dict:
                    # Use tag:value as key to separate different tags
                    key = f"{tag}:{tags_dict[tag]}"
                    self.value_to_indices[key].append(idx)
                    grouped = True
                    # No break - add to all matching groups
            if not grouped:
                self.ungrouped_indices.append(idx)

        # Filter to groups with at least 2 members (for positive pairs)
        self.valid_groups = [
            (key, indices)
            for key, indices in self.value_to_indices.items()
            if len(indices) >= 2
        ]

        # Count groups per contrastive tag
        groups_per_tag: dict[str, int] = defaultdict(int)
        for key, _ in self.valid_groups:
            # Find which contrastive tag this key belongs to
            for tag in contrastive_tags:
                if key.startswith(f"{tag}:"):
                    groups_per_tag[tag] += 1
                    break

        print(f"ContrastiveBatchSampler: {len(self.valid_groups)} groups with 2+ members, "
              f"{len(self.ungrouped_indices)} ungrouped items")
        print(f"  Groups per tag: {dict(groups_per_tag)}")

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.epoch = epoch

    def __iter__(self):
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


class MixingSampler(Sampler):
    """Batch sampler that mixes samples from multiple child samplers with configurable ratios.

    Composes multiple samplers, drawing from each according to their weight.
    Yields batches of (source_key, index) tuples.
    """

    def __init__(
        self,
        samplers: dict[str, tuple[Sampler, float, int]],
        batch_size: int,
        seed: int = 42,
    ):
        """Initialize the mixing sampler.

        Args:
            samplers: Dictionary mapping source keys to (sampler, weight, samples_per_index).
                - sampler: The child batch sampler
                - weight: Relative weight for this source (will be normalized)
                - samples_per_index: Number of samples each index yields (e.g., 1 for single,
                  2 for pairs, 3 for triples)
            batch_size: Total number of samples per batch (accounts for samples_per_index)
            seed: Random seed
        """
        self.samplers = samplers
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        # Normalize weights
        total_weight = sum(w for _, w, _ in samplers.values())
        self.normalized_weights = {
            key: weight / total_weight
            for key, (_, weight, _) in samplers.items()
        }

        # Calculate indices per source based on weights and samples_per_index
        # E.g., if batch_size=64, weight=0.5, samples_per_index=2:
        #   samples_from_source = 64 * 0.5 = 32
        #   indices_from_source = 32 / 2 = 16
        self.indices_per_source: dict[str, int] = {}
        for key, (_, weight, samples_per_index) in samplers.items():
            normalized_weight = self.normalized_weights[key]
            samples_from_source = int(batch_size * normalized_weight)
            self.indices_per_source[key] = max(1, samples_from_source // samples_per_index)

        # Calculate batches per epoch: continue until ALL child samplers are exhausted
        # Each child sampler has a known length (number of batches it can yield)
        batches_needed = []
        for key, (sampler, _, _) in samplers.items():
            indices_per_batch = self.indices_per_source[key]
            total_indices = sum(len(batch) for batch in iter(sampler))
            if indices_per_batch > 0:
                batches_needed.append(total_indices / indices_per_batch)
        self.batches_per_epoch = int(max(batches_needed)) if batches_needed else 0

        # Print configuration
        print(f"MixingSampler:")
        for key, (sampler, weight, samples_per_index) in samplers.items():
            print(f"  {key}: weight={self.normalized_weights[key]:.2f}, "
                  f"samples_per_index={samples_per_index}, "
                  f"indices_per_batch={self.indices_per_source[key]}")
        print(f"  Total batch_size: {batch_size}")
        print(f"  Batches per epoch: {self.batches_per_epoch}")

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling. Propagates to all child samplers."""
        self.epoch = epoch
        for sampler, _, _ in self.samplers.values():
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __iter__(self):
        """Yield batches of (source_key, idx) tuples."""
        rng = random.Random(self.seed + self.epoch)

        # Get iterators for each child sampler
        # Each child sampler yields batches of indices
        source_indices: dict[str, list[int]] = {key: [] for key in self.samplers}

        # Create fresh iterators for each child sampler
        source_iters: dict[str, iter] = {}
        for key, (sampler, _, _) in self.samplers.items():
            source_iters[key] = iter(sampler)

        for _ in range(self.batches_per_epoch):
            batch: list[tuple[str, int]] = []

            for key in self.samplers:
                indices_needed = self.indices_per_source[key]

                # Refill source indices if needed
                while len(source_indices[key]) < indices_needed:
                    try:
                        child_batch = next(source_iters[key])
                        # Shuffle child batch indices
                        child_batch = list(child_batch)
                        rng.shuffle(child_batch)
                        source_indices[key].extend(child_batch)
                    except StopIteration:
                        # Child sampler exhausted, create new iterator
                        sampler, _, _ = self.samplers[key]
                        source_iters[key] = iter(sampler)
                        child_batch = next(source_iters[key])
                        child_batch = list(child_batch)
                        rng.shuffle(child_batch)
                        source_indices[key].extend(child_batch)

                # Take indices for this batch
                taken = source_indices[key][:indices_needed]
                source_indices[key] = source_indices[key][indices_needed:]
                batch.extend((key, idx) for idx in taken)

            # Shuffle the combined batch
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.batches_per_epoch


# Type alias for tokenizer function (e.g., SentenceTransformer.tokenize)
Tokenizer = Callable[[list[str]], dict[str, torch.Tensor]]


def collate_sentences(
    samples: list[SentenceSample],
    tag_vocabs: dict[str, dict[str, int]],
    classification_tasks: list[str],
    contrastive_tasks: list[str],
    include_base_contrastive: bool = False,
    tokenizer: Tokenizer | None = None,
    max_length: int | None = None,
) -> SentenceBatch:
    """Collate function that precomputes all label matrices.

    Args:
        samples: List of SentenceSample objects
        tag_vocabs: Tag vocabularies for classification tasks
        classification_tasks: List of tag keys for classification
        contrastive_tasks: List of tag keys for contrastive learning
        include_base_contrastive: If True, build base_contrastive_labels for
            samples with the same pruned_tags (used for template <-> LLM pairing)
        tokenizer: Optional tokenizer function to pre-tokenize sentences
            (enables parallel tokenization in DataLoader workers)
        max_length: Optional max sequence length for tokenization. If set, all
            sequences are padded/truncated to this length to avoid memory
            fragmentation from variable-sized tensors.

    Returns:
        SentenceBatch with precomputed labels
    """
    n = len(samples)
    sentences = [s.sentence for s in samples]

    # Build template mask (True if source == "template")
    template_mask = torch.tensor([s.source == "template" for s in samples], dtype=torch.bool)

    # Precompute classification labels (only for template samples)
    classification_labels = {}
    for task in classification_tasks:
        if task not in tag_vocabs:
            continue

        mask = torch.zeros(n, dtype=torch.bool)
        labels = torch.zeros(n, dtype=torch.long)

        for i, s in enumerate(samples):
            # Only include template samples for classification
            if s.source == "template" and task in s.used_tags and s.used_tags[task] in tag_vocabs[task]:
                mask[i] = True
                labels[i] = tag_vocabs[task][s.used_tags[task]]

        classification_labels[task] = (mask, labels)

    # Precompute contrastive positive matrices (projection head contrastive, template only)
    contrastive_labels = {}
    for task in contrastive_tasks:
        mask = torch.zeros(n, dtype=torch.bool)
        values: list[str | None] = []

        for i, s in enumerate(samples):
            # Only include template samples for projection head contrastive
            if s.source == "template" and task in s.used_tags:
                mask[i] = True
                values.append(s.used_tags[task])
            else:
                values.append(None)

        # Build positive matrix
        # positive_matrix[i, j] = 1 if samples i and j have the same value and i != j
        positive_matrix = torch.zeros(n, n, dtype=torch.float)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if mask[j] and values[i] == values[j] and i != j:
                    positive_matrix[i, j] = 1.0

        contrastive_labels[task] = (mask, positive_matrix)

    # Precompute presence labels (binary: is the tag in used_tags?)
    # Include both classification and contrastive tasks (template only)
    presence_labels = {}
    all_tasks = list(classification_tasks) + list(contrastive_tasks)
    for task in all_tasks:
        labels = torch.zeros(n, dtype=torch.float)
        for i, s in enumerate(samples):
            # Only include template samples for presence prediction
            if s.source == "template" and task in s.used_tags:
                labels[i] = 1.0
        presence_labels[task] = labels

    # Build base contrastive labels (samples with same pruned_tags are positives)
    base_contrastive_labels = None
    if include_base_contrastive:
        # Map pruned_tags to sample indices
        pruned_to_indices: dict[frozenset, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            pruned_to_indices[s.pruned_tags].append(i)

        # Build positive matrix: samples with same pruned_tags are positives
        base_mask = torch.ones(n, dtype=torch.bool)  # All samples participate
        base_positive_matrix = torch.zeros(n, n, dtype=torch.float)
        for indices in pruned_to_indices.values():
            if len(indices) >= 2:
                for i in indices:
                    for j in indices:
                        if i != j:
                            base_positive_matrix[i, j] = 1.0

        base_contrastive_labels = (base_mask, base_positive_matrix)

    # Extract pruned_tags for global MRR computation
    pruned_tags_list = [s.pruned_tags for s in samples]

    # Tokenize sentences if tokenizer provided (enables parallel tokenization in workers)
    token_ids = None
    if tokenizer is not None:
        if max_length is not None:
            token_ids = tokenizer(
                sentences, padding="max_length", truncation=True, max_length=max_length
            )
        else:
            token_ids = tokenizer(sentences)

    return SentenceBatch(
        sentences=sentences,
        classification_labels=classification_labels,
        contrastive_labels=contrastive_labels,
        presence_labels=presence_labels,
        base_contrastive_labels=base_contrastive_labels,
        template_mask=template_mask,
        pruned_tags=pruned_tags_list,
        token_ids=token_ids,
    )


def create_collate_fn(
    tag_vocabs: dict[str, dict[str, int]],
    classification_tasks: list[str],
    contrastive_tasks: list[str],
    include_base_contrastive: bool = False,
    tokenizer: Tokenizer | None = None,
    max_length: int | None = None,
):
    """Create a collate function with bound parameters.

    Args:
        tag_vocabs: Tag vocabularies for classification tasks
        classification_tasks: List of tag keys for classification
        contrastive_tasks: List of tag keys for contrastive learning
        include_base_contrastive: If True, build base_contrastive_labels for
            samples with the same pruned_tags
        tokenizer: Optional tokenizer function to pre-tokenize sentences
            (enables parallel tokenization in DataLoader workers)
        max_length: Optional max sequence length for tokenization. If set, all
            sequences are padded/truncated to this length to avoid memory
            fragmentation from variable-sized tensors.

    Returns:
        Collate function suitable for DataLoader
    """

    def collate_fn(samples: list[SentenceSample]) -> SentenceBatch:
        return collate_sentences(
            samples,
            tag_vocabs=tag_vocabs,
            classification_tasks=classification_tasks,
            contrastive_tasks=contrastive_tasks,
            include_base_contrastive=include_base_contrastive,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    return collate_fn
