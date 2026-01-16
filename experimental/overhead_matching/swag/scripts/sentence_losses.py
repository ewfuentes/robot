"""Loss functions for sentence embedding training."""

import torch
import torch.nn.functional as F

from experimental.overhead_matching.swag.data.sentence_dataset import SentenceBatch


def info_nce_loss_from_matrix(
    similarity: torch.Tensor,
    positive_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute InfoNCE loss with precomputed positive matrix.

    For each anchor, the loss is:
        -mean(pos_similarities) + logsumexp(all_similarities_except_self)

    Args:
        similarity: (n, n) tensor of pairwise similarities (already temperature-scaled)
        positive_matrix: (n, n) binary tensor where [i,j]=1 if i and j are positives

    Returns:
        Scalar loss tensor
    """
    n = similarity.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=similarity.device)

    loss = torch.tensor(0.0, device=similarity.device)
    count = 0

    for i in range(n):
        pos_mask = positive_matrix[i] > 0

        # Skip if no positives for this anchor
        if pos_mask.sum() == 0:
            continue

        # Positive similarities
        pos_sim = similarity[i, pos_mask]

        # All similarities except self (for denominator)
        all_except_self = torch.cat([similarity[i, :i], similarity[i, i + 1 :]])

        # InfoNCE: -log(sum(exp(pos)) / sum(exp(all)))
        # = -mean(pos) + logsumexp(all)
        loss = loss + (-pos_sim.mean() + torch.logsumexp(all_except_self, dim=0))
        count += 1

    return loss / max(count, 1)


def compute_sentence_losses(
    model_output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    batch: SentenceBatch,
    temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Compute all losses for a batch.

    Args:
        model_output: Output from SentenceEmbeddingModel.forward()
        batch: SentenceBatch with precomputed labels
        temperature: Temperature for contrastive loss

    Returns:
        Dictionary mapping loss names to scalar loss tensors
    """
    losses = {}
    device = model_output["base_embedding"].device

    # Classification losses
    classification_logits = model_output["classification_logits"]
    for task, (mask, labels) in batch.classification_labels.items():
        if task not in classification_logits:
            continue

        mask = mask.to(device)
        labels = labels.to(device)

        if mask.sum() == 0:
            continue

        logits = classification_logits[task][mask]
        target_labels = labels[mask]

        losses[f"loss_cls_{task}"] = F.cross_entropy(logits, target_labels)

    # Contrastive losses
    contrastive_embeddings = model_output["contrastive_embeddings"]
    for task, (mask, positive_matrix) in batch.contrastive_labels.items():
        if task not in contrastive_embeddings:
            continue

        mask = mask.to(device)
        positive_matrix = positive_matrix.to(device)

        if mask.sum() < 2:
            continue

        embeddings = contrastive_embeddings[task]

        # Compute similarity matrix (temperature-scaled)
        similarity = embeddings @ embeddings.T / temperature

        # Extract only masked rows/cols
        masked_sim = similarity[mask][:, mask]
        masked_pos = positive_matrix[mask][:, mask]

        losses[f"loss_contrast_{task}"] = info_nce_loss_from_matrix(masked_sim, masked_pos)

    # Presence prediction losses (binary cross-entropy)
    presence_logits = model_output.get("presence_logits", {})
    for task, labels in batch.presence_labels.items():
        if task not in presence_logits:
            continue

        labels = labels.to(device)
        logits = presence_logits[task].squeeze(-1)  # (batch_size,)

        losses[f"loss_presence_{task}"] = F.binary_cross_entropy_with_logits(logits, labels)

    return losses


def aggregate_losses(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Aggregate multiple losses into a single loss.

    Args:
        losses: Dictionary of named losses
        weights: Optional weights per loss (default: uniform)

    Returns:
        Aggregated scalar loss
    """
    if not losses:
        return torch.tensor(0.0)

    if weights is None:
        # Uniform weighting
        return sum(losses.values()) / len(losses)

    total = torch.tensor(0.0, device=next(iter(losses.values())).device)
    for name, loss in losses.items():
        weight = weights.get(name, 1.0)
        total = total + weight * loss

    return total


def compute_classification_accuracy(
    model_output: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    batch: SentenceBatch,
) -> dict[str, float]:
    """Compute classification and presence accuracy for each task.

    Args:
        model_output: Output from SentenceEmbeddingModel.forward()
        batch: SentenceBatch with precomputed labels

    Returns:
        Dictionary mapping task names to accuracy values
    """
    accuracies = {}
    device = model_output["base_embedding"].device

    # Classification accuracy
    classification_logits = model_output["classification_logits"]
    for task, (mask, labels) in batch.classification_labels.items():
        if task not in classification_logits:
            continue

        mask = mask.to(device)
        labels = labels.to(device)

        if mask.sum() == 0:
            continue

        logits = classification_logits[task][mask]
        target_labels = labels[mask]

        predictions = logits.argmax(dim=-1)
        correct = (predictions == target_labels).float().mean().item()
        accuracies[f"acc_cls_{task}"] = correct

    # Presence prediction accuracy
    presence_logits = model_output.get("presence_logits", {})
    for task, labels in batch.presence_labels.items():
        if task not in presence_logits:
            continue

        labels = labels.to(device)
        logits = presence_logits[task].squeeze(-1)

        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct = (predictions == labels).float().mean().item()
        accuracies[f"acc_presence_{task}"] = correct

    return accuracies
