"""Retrieval metrics for cross-view geo-localization.

Provides:
- validation_metrics_from_similarity(): detailed metrics for tensorboard logging
- compute_top_k_metrics(): simple recall@k and MRR from a similarity matrix
"""

import torch

from experimental.overhead_matching.swag.data import vigor_dataset as vd


@torch.no_grad()
def validation_metrics_from_similarity(
    name: str,
    similarity: torch.Tensor,  # num_panos x num_sat
    panorama_metadata: "pd.DataFrame",
) -> dict:
    """Compute detailed retrieval metrics from a similarity matrix.

    Returns name-prefixed metrics suitable for tensorboard logging:
    positive MRR, max pos/semipos MRR, pos recall@K, any/all pos_semipos recall@K.

    Args:
        name: prefix for metric keys (e.g. city name)
        similarity: (num_panos, num_sats) similarity matrix
        panorama_metadata: DataFrame with positive_satellite_idxs and
            semipositive_satellite_idxs columns
    """
    import pandas as pd

    num_panos = similarity.shape[0]

    # Determine max columns needed (positives + semipositives)
    max_cols = max(
        len(row.positive_satellite_idxs) + len(row.semipositive_satellite_idxs)
        for _, row in panorama_metadata.iterrows()
    )
    max_cols = max(max_cols, 1)

    invalid_mask = torch.ones((num_panos, max_cols), dtype=torch.bool)
    sat_idxs = torch.zeros((num_panos, max_cols), dtype=torch.int32)
    for pano_idx, pano_metadata in panorama_metadata.iterrows():
        num_pos = len(pano_metadata.positive_satellite_idxs)
        for col_idx, sat_idx in enumerate(pano_metadata.positive_satellite_idxs):
            sat_idxs[pano_idx, col_idx] = sat_idx
            invalid_mask[pano_idx, col_idx] = False

        for col_idx, sat_idx in enumerate(pano_metadata.semipositive_satellite_idxs):
            sat_idxs[pano_idx, num_pos + col_idx] = sat_idx
            invalid_mask[pano_idx, num_pos + col_idx] = False

    row_idxs = torch.arange(num_panos).reshape(-1, 1).expand(-1, max_cols)
    pos_semipos_similarities = similarity[row_idxs, sat_idxs]
    pos_semipos_similarities[invalid_mask] = torch.nan

    ranks = (similarity[:, None, :] >= pos_semipos_similarities[:, :, None]).sum(dim=-1)

    positive_recip_ranks = 1.0 / ranks[:, 0]
    positive_recip_ranks[invalid_mask[:, 0]] = torch.nan
    positive_mean_recip_rank = torch.nanmean(positive_recip_ranks)

    max_pos_semi_pos_recip_ranks = 1.0 / ranks.max(dim=-1).values
    max_pos_semi_pos_recip_ranks = torch.nanmean(max_pos_semi_pos_recip_ranks)

    k_values = [1, 5, 10, 100]
    pos_recall = {
        f"{name}/pos_recall@{k}": (
            ranks[~(invalid_mask[:, 0]), 0] <= k
        ).float().mean().item()
        for k in k_values
    }

    invalid_mask_cuda = invalid_mask.cuda()
    ranks_cuda = ranks.cuda()
    any_pos_semipos_recall = {
        f"{name}/any pos_semipos_recall@{k}": (
            (ranks_cuda <= k) & (~invalid_mask_cuda)
        ).any(dim=-1).float().mean().item()
        for k in k_values
    }

    all_pos_semipos_recall = {
        f"{name}/all pos_semipos_recall@{k}": (
            ranks_cuda <= k
        ).all(dim=-1).float().mean().item()
        for k in k_values[1:]
    }

    out = (
        {
            f"{name}/positive_mean_recip_rank": positive_mean_recip_rank.item(),
            f"{name}/max_pos_semi_pos_recip_rank": max_pos_semi_pos_recip_ranks.item(),
        }
        | pos_recall
        | any_pos_semipos_recall
        | all_pos_semipos_recall
    )
    return out


def compute_top_k_metrics(
    similarity: torch.Tensor,
    dataset: vd.VigorDataset,
    ks: list[int] = [1, 5, 10],
) -> dict:
    """Compute recall@k and MRR from a precomputed similarity matrix.

    Args:
        similarity: (num_panos, num_sats) similarity matrix
        dataset: VigorDataset with panorama metadata containing positive_satellite_idxs
            and semipositive_satellite_idxs (both are treated as positives for ranking)
        ks: list of k values for recall@k

    Returns:
        dict with keys "recall@{k}" for each k, and "mrr"
    """
    rankings = torch.argsort(similarity, dim=1, descending=True)

    reciprocal_ranks = []
    hit_counts = {k: 0 for k in ks}
    total = 0

    for pano_idx, (_, row) in enumerate(dataset._panorama_metadata.iterrows()):
        positive_idxs = list(row.positive_satellite_idxs) + list(row.semipositive_satellite_idxs)
        if len(positive_idxs) == 0:
            continue
        total += 1

        # Find best rank among all positive/semipositive satellites
        best_rank = similarity.shape[1]  # worst case
        for sat_idx in positive_idxs:
            rank = torch.argwhere(rankings[pano_idx] == sat_idx).item()
            best_rank = min(best_rank, rank)

        reciprocal_ranks.append(1.0 / (best_rank + 1))
        for k in ks:
            if best_rank < k:
                hit_counts[k] += 1

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = hit_counts[k] / total if total > 0 else 0.0
    metrics["mrr"] = sum(reciprocal_ranks) / total if total > 0 else 0.0
    return metrics
