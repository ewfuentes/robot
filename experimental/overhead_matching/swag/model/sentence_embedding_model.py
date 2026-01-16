"""Multi-task sentence embedding model for OSM landmark descriptions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from experimental.overhead_matching.swag.scripts.sentence_configs import (
    ClassificationTaskConfig,
    ContrastiveTaskConfig,
    SentenceEmbeddingModelConfig,
)


class SentenceEmbeddingModel(nn.Module):
    """Multi-task sentence embedding model.

    Uses a pretrained sentence transformer as the base encoder,
    with task-specific heads for classification and contrastive learning.
    """

    def __init__(
        self,
        config: SentenceEmbeddingModelConfig,
        classification_tasks: list[ClassificationTaskConfig],
        contrastive_tasks: list[ContrastiveTaskConfig],
    ):
        """Initialize the model.

        Args:
            config: Model configuration
            classification_tasks: List of classification task configs
            contrastive_tasks: List of contrastive task configs
        """
        super().__init__()

        self.config = config

        # Load pretrained sentence transformer
        self.encoder = SentenceTransformer(config.encoder_name)
        self.base_dim = self.encoder.get_sentence_embedding_dimension()

        # Optionally freeze encoder
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification heads: Linear layer per task
        self.classification_heads = nn.ModuleDict(
            {
                task.name: nn.Linear(self.base_dim, task.num_classes)
                for task in classification_tasks
            }
        )

        # Contrastive projection heads: MLP per task
        self.contrastive_heads = nn.ModuleDict(
            {
                task.name: nn.Sequential(
                    nn.Linear(self.base_dim, config.projection_dim),
                    nn.ReLU(),
                    nn.Linear(config.projection_dim, config.projection_dim),
                )
                for task in contrastive_tasks
            }
        )

        # Presence prediction heads: Binary classifier for all tasks
        # Predicts whether the sentence contains the tag (e.g., has a name, has amenity)
        all_task_names = [t.name for t in classification_tasks] + [t.name for t in contrastive_tasks]
        self.presence_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.base_dim, 1)
                for name in all_task_names
            }
        )

        self._classification_task_names = [t.name for t in classification_tasks]
        self._contrastive_task_names = [t.name for t in contrastive_tasks]

    @property
    def classification_task_names(self) -> list[str]:
        """Get names of classification tasks."""
        return self._classification_task_names

    @property
    def contrastive_task_names(self) -> list[str]:
        """Get names of contrastive tasks."""
        return self._contrastive_task_names

    def encode_sentences(self, sentences: list[str]) -> torch.Tensor:
        """Encode sentences to base embeddings (training mode with gradients).

        Args:
            sentences: List of sentence strings

        Returns:
            Tensor of shape (batch_size, base_dim)
        """
        # Tokenize sentences
        features = self.encoder.tokenize(sentences)

        # Move to same device as model
        device = next(self.encoder.parameters()).device
        features = {k: v.to(device) for k, v in features.items()}

        # Forward through encoder modules (with gradients, unlike .encode())
        output = self.encoder(features)

        # Get sentence embeddings from output
        return output["sentence_embedding"]

    def forward(
        self, sentences: list[str]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Forward pass through the model.

        Args:
            sentences: List of sentence strings

        Returns:
            Dictionary containing:
                - base_embedding: (batch_size, base_dim)
                - classification_logits: {task_name: (batch_size, num_classes)}
                - contrastive_embeddings: {task_name: (batch_size, projection_dim)}
                - presence_logits: {task_name: (batch_size, 1)} - predicts if tag is present
        """
        # Encode sentences
        base_emb = self.encode_sentences(sentences)

        # Compute classification logits
        classification_logits = {
            name: head(base_emb) for name, head in self.classification_heads.items()
        }

        # Compute contrastive embeddings (normalized)
        contrastive_embeddings = {
            name: F.normalize(head(base_emb), dim=-1)
            for name, head in self.contrastive_heads.items()
        }

        # Compute presence prediction logits
        presence_logits = {
            name: head(base_emb) for name, head in self.presence_heads.items()
        }

        return {
            "base_embedding": base_emb,
            "classification_logits": classification_logits,
            "contrastive_embeddings": contrastive_embeddings,
            "presence_logits": presence_logits,
        }

    def get_base_embedding(self, sentences: list[str]) -> torch.Tensor:
        """Get base embeddings for sentences (for inference).

        Args:
            sentences: List of sentence strings

        Returns:
            Tensor of shape (batch_size, base_dim)
        """
        return self.encode_sentences(sentences)

    def get_contrastive_embedding(
        self, sentences: list[str], task: str
    ) -> torch.Tensor:
        """Get contrastive embeddings for a specific task (for inference).

        Args:
            sentences: List of sentence strings
            task: Name of the contrastive task

        Returns:
            Normalized tensor of shape (batch_size, projection_dim)
        """
        base_emb = self.encode_sentences(sentences)
        projected = self.contrastive_heads[task](base_emb)
        return F.normalize(projected, dim=-1)


def create_model_from_config(
    model_config: SentenceEmbeddingModelConfig,
    tag_vocabs: dict[str, dict[str, int]],
    classification_task_names: list[str],
    contrastive_task_names: list[str],
) -> SentenceEmbeddingModel:
    """Create model from config and vocabularies.

    Args:
        model_config: Model configuration
        tag_vocabs: Tag vocabularies (for determining num_classes)
        classification_task_names: Names of classification tasks
        contrastive_task_names: Names of contrastive tasks

    Returns:
        Initialized SentenceEmbeddingModel
    """
    # Build classification task configs
    classification_tasks = []
    for name in classification_task_names:
        if name in tag_vocabs:
            classification_tasks.append(
                ClassificationTaskConfig(
                    name=name,
                    num_classes=len(tag_vocabs[name]),
                )
            )

    # Build contrastive task configs
    contrastive_tasks = [ContrastiveTaskConfig(name=name) for name in contrastive_task_names]

    return SentenceEmbeddingModel(
        config=model_config,
        classification_tasks=classification_tasks,
        contrastive_tasks=contrastive_tasks,
    )
