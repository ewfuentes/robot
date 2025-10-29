"""Trainable sentence embedding model for landmark descriptions.

This module provides a trainable alternative to using frozen OpenAI embeddings.
It uses a Sentence-BERT style architecture with a pretrained transformer and
mean pooling, with optional fine-tuning during training.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TrainableSentenceEmbedder(nn.Module):
    """Trainable sentence embedding model using pretrained transformers.

    This class implements a Sentence-BERT style encoder that:
    1. Tokenizes input text using a pretrained tokenizer
    2. Encodes tokens using a pretrained transformer (BERT, RoBERTa, etc.)
    3. Applies mean pooling over token embeddings
    4. Projects to desired output dimension with a learned linear layer

    The transformer weights can be frozen or fine-tuned during training.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        output_dim: int,
        freeze_weights: bool = True,
        max_sequence_length: int = 128,
        model_weights_path: str | None = None,
    ):
        """Initialize the trainable sentence embedder.

        Args:
            pretrained_model_name_or_path: HuggingFace model identifier
                (e.g., "sentence-transformers/all-MiniLM-L6-v2", "bert-base-uncased")
            output_dim: Dimension of output embeddings
            freeze_weights: If True, freeze transformer weights during training
            max_sequence_length: Maximum sequence length for tokenization
            model_weights_path: Optional path to custom pretrained weights to load
        """
        super().__init__()

        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length

        # Load pretrained transformer and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path)

        # Load custom weights if provided
        if model_weights_path is not None:
            weights_path = Path(model_weights_path).expanduser()
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location='cpu')
                self.transformer.load_state_dict(state_dict)
                print(f"Loaded custom transformer weights from {weights_path}")
            else:
                raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

        # Freeze transformer weights if requested
        if freeze_weights:
            for param in self.transformer.parameters():
                param.requires_grad = False
            print(f"Frozen transformer weights for {pretrained_model_name_or_path}")

        # Learned projection to output dimension
        transformer_output_dim = self.transformer.config.hidden_size
        self.projection = nn.Linear(transformer_output_dim, output_dim)

    def mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mean pooling over token embeddings.

        Args:
            token_embeddings: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]

        Returns:
            Pooled embeddings: [batch_size, hidden_dim]
        """
        # Expand attention mask to match token embeddings shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings where attention mask is 1
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Divide by number of tokens (avoiding division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text strings to embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            Embeddings tensor of shape [batch_size, output_dim]
        """
        if len(texts) == 0:
            # Return empty tensor if no texts provided
            return torch.zeros((0, self.output_dim), device=next(self.parameters()).device)

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )

        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Forward through transformer
        transformer_output = self.transformer(**encoded)

        # Mean pool over token embeddings
        token_embeddings = transformer_output.last_hidden_state  # [batch, seq_len, hidden_dim]
        pooled_embeddings = self.mean_pool(token_embeddings, encoded['attention_mask'])

        # Project to output dimension
        projected_embeddings = self.projection(pooled_embeddings)

        # Normalize to unit length (for cosine similarity)
        normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=1)

        return normalized_embeddings

    def freeze(self):
        """Freeze all transformer parameters."""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Frozen transformer weights")

    def unfreeze(self):
        """Unfreeze all transformer parameters for fine-tuning."""
        for param in self.transformer.parameters():
            param.requires_grad = True
        print("Unfrozen transformer weights for fine-tuning")
