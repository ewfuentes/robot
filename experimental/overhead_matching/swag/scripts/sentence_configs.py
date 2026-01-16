"""Configuration dataclasses for sentence embedding training."""

from dataclasses import dataclass, field
from pathlib import Path

import msgspec


@dataclass
class ClassificationTaskConfig:
    """Configuration for a classification task."""

    name: str  # Tag key (e.g., "amenity", "building")
    num_classes: int  # Number of classes in vocabulary


@dataclass
class ContrastiveTaskConfig:
    """Configuration for a contrastive task."""

    name: str  # Tag key (e.g., "name", "addr:street")


@dataclass
class LearningRateScheduleConfig:
    """Learning rate schedule configuration with differential rates.

    Supports separate learning rates for encoder vs heads:
    - encoder_lr: Learning rate for pretrained encoder (lower to preserve knowledge)
    - heads_lr: Learning rate for classification/contrastive heads (higher for random init)

    If heads_lr is None, uses encoder_lr for everything (single learning rate).
    """

    encoder_lr: float = 2e-5  # Learning rate for pretrained encoder
    heads_lr: float | None = 1e-4  # Learning rate for heads (None = use encoder_lr)
    warmup_steps: int = 1000
    decay_factor: float = 0.1
    decay_steps: list[int] = field(default_factory=lambda: [10000, 20000])


@dataclass
class SentenceEmbeddingModelConfig:
    """Configuration for the sentence embedding model."""

    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    projection_dim: int = 128  # Dimension for contrastive projection heads
    freeze_encoder: bool = False  # Whether to freeze the pretrained encoder


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 256
    num_epochs: int = 10
    lr_schedule: LearningRateScheduleConfig = field(
        default_factory=LearningRateScheduleConfig
    )
    gradient_clip_norm: float = 1.0
    temperature: float = 0.07  # Temperature for contrastive loss
    num_workers: int = 4
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    save_every_n_steps: int = 5000

    # Contrastive batch sampling parameters
    groups_per_batch: int = 32  # Number of contrastive groups to sample per batch
    samples_per_group: int = 4  # Target samples per group (for positive pairs)


# Default classification tasks (tag keys to use for classification)
DEFAULT_CLASSIFICATION_TAGS = [
    "amenity",
    "building",
    "highway",
    "shop",
    "leisure",
    "tourism",
    "landuse",
    "natural",
    "surface",
    "cuisine",
]

# Default contrastive tasks (tag keys for contrastive learning)
DEFAULT_CONTRASTIVE_TAGS = [
    "name",
    "addr:street",
]


@dataclass
class SentenceTrainConfig:
    """Full training configuration."""

    # Data paths
    db_path: Path | None = None  # Path to landmarks SQLite database
    output_dir: Path | None = None
    tag_vocabs_path: Path | None = None  # Optional, built automatically if not provided
    tensorboard_dir: Path | None = None
    llm_sentences_path: Path | None = None  # Path to LLM sentences JSONL file

    # Model config
    model: SentenceEmbeddingModelConfig = field(
        default_factory=SentenceEmbeddingModelConfig
    )

    # Training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Task selection
    classification_tags: list[str] = field(
        default_factory=lambda: list(DEFAULT_CLASSIFICATION_TAGS)
    )
    contrastive_tags: list[str] = field(
        default_factory=lambda: list(DEFAULT_CONTRASTIVE_TAGS)
    )

    # Train/test split
    train_split: float = 0.9
    seed: int = 42

    # Limit landmarks for testing
    limit: int | None = None

    # Tags that must always be included in sentences when present
    # Defaults to contrastive_tags to ensure contrastive learning has signal
    required_tags: list[str] | None = None  # None = use contrastive_tags


def msgspec_enc_hook(obj):
    """Encoder hook for msgspec serialization."""
    if isinstance(obj, Path):
        return str(obj)
    raise NotImplementedError(f"Cannot serialize {type(obj)}")


def msgspec_dec_hook(type_, obj):
    """Decoder hook for msgspec deserialization."""
    if type_ is Path:
        return Path(obj)
    raise NotImplementedError(f"Cannot deserialize {type_}")


def save_config(config: SentenceTrainConfig, path: Path) -> None:
    """Save config to YAML file."""
    yaml_bytes = msgspec.yaml.encode(config, enc_hook=msgspec_enc_hook)
    path.write_bytes(yaml_bytes)


def load_config(path: Path) -> SentenceTrainConfig:
    """Load config from YAML file."""
    yaml_bytes = path.read_bytes()
    return msgspec.yaml.decode(
        yaml_bytes, type=SentenceTrainConfig, dec_hook=msgspec_dec_hook
    )
