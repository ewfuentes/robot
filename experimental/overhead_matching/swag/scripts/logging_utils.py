import math
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from experimental.overhead_matching.swag.scripts.pairing import Pairs, PositiveAnchorSets
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime


LOG_HIST_EVERY = 100


def _dump_empty_sim_debug_info(
    loss_dict: dict,
    pairing_data,
    step_idx: int,
    epoch_idx: int,
    batch_idx: int,
    empty_tensor_name: str,
):
    """Dump diagnostic info when pos_sim or neg_sim is unexpectedly empty."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = Path(f"/tmp/empty_sim_debug_{timestamp}.pkl")

    debug_info = {
        "timestamp": timestamp,
        "empty_tensor_name": empty_tensor_name,
        "step_idx": step_idx,
        "epoch_idx": epoch_idx,
        "batch_idx": batch_idx,
        "loss_dict_summary": {},
        "pairing_data_summary": {},
    }

    # Summarize loss_dict
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            debug_info["loss_dict_summary"][k] = {
                "shape": list(v.shape),
                "numel": v.numel(),
                "dtype": str(v.dtype),
                "device": str(v.device),
                "has_nan": bool(torch.isnan(v).any()) if v.numel() > 0 else False,
                "has_inf": bool(torch.isinf(v).any()) if v.numel() > 0 else False,
                "values_sample": (v.flatten()[:10].tolist() if v.numel() > 0 else []),
            }
        else:
            debug_info["loss_dict_summary"][k] = {"value": v, "type": type(v).__name__}

    # Summarize pairing_data
    if isinstance(pairing_data, Pairs):
        debug_info["pairing_data_summary"] = {
            "type": "Pairs",
            "num_positive_pairs": len(pairing_data.positive_pairs),
            "num_semipositive_pairs": len(pairing_data.semipositive_pairs),
            "num_negative_pairs": len(pairing_data.negative_pairs),
            "positive_pairs_sample": pairing_data.positive_pairs[:10],
            "negative_pairs_sample": pairing_data.negative_pairs[:10],
        }
    elif isinstance(pairing_data, PositiveAnchorSets):
        debug_info["pairing_data_summary"] = {
            "type": "PositiveAnchorSets",
            "num_anchors": len(pairing_data.anchor),
            "positive_set_sizes": [len(x) for x in pairing_data.positive],
            "semipositive_set_sizes": [len(x) for x in pairing_data.semipositive],
        }

    # Save to file
    with open(debug_path, "wb") as f:
        pickle.dump(debug_info, f)

    # Build error message
    error_msg = f"""
UNEXPECTED EMPTY SIMILARITY TENSOR: {empty_tensor_name}

Debug info saved to: {debug_path}

Summary:
  step_idx={step_idx}, epoch_idx={epoch_idx}, batch_idx={batch_idx}

Pairing data:
  {debug_info['pairing_data_summary']}

Loss dict tensor shapes:
"""
    for k, v in debug_info["loss_dict_summary"].items():
        if isinstance(v, dict) and "shape" in v:
            error_msg += f"  {k}: shape={v['shape']}, numel={v['numel']}, has_nan={v.get('has_nan')}, has_inf={v.get('has_inf')}\n"

    return error_msg, debug_path

@torch.no_grad()
def log_batch_metrics(writer, loss_dict, lr_scheduler, pairing_data, step_idx, epoch_idx, batch_idx, quiet):
    if isinstance(pairing_data, Pairs):
        writer.add_scalar("train/num_positive_pairs", len(pairing_data.positive_pairs), global_step=step_idx)
        writer.add_scalar("train/num_semipos_pairs", len(pairing_data.semipositive_pairs), global_step=step_idx)
        writer.add_scalar("train/num_neg_pairs", len(pairing_data.negative_pairs), global_step=step_idx)
    elif isinstance(pairing_data, PositiveAnchorSets):
        num_positive = [len(x) for x in pairing_data.positive]
        num_semipositive = [len(x) for x in pairing_data.semipositive]
        writer.add_scalar("train/num_anchors", len(pairing_data.anchor), global_step=step_idx)
        writer.add_scalar("train/min_positive", min(num_positive), global_step=step_idx)
        writer.add_scalar("train/min_semipositive", min(num_semipositive), global_step=step_idx)
        writer.add_scalar("train/max_positive", max(num_positive), global_step=step_idx)
        writer.add_scalar("train/max_semipositive", max(num_semipositive), global_step=step_idx)

    writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step=step_idx)
    if "pos_sim" in loss_dict and "neg_sim" in loss_dict:
        pos_sim = loss_dict["pos_sim"]
        neg_sim = loss_dict["neg_sim"]
        semipos_sim = loss_dict["semipos_sim"] if "semipos_sim" in loss_dict else None  # may not be present
        if semipos_sim is not None and semipos_sim.numel() == 0:
            semipos_sim = None

        # Assert that pos_sim and neg_sim are non-empty - dump debug info if they are
        if pos_sim.numel() == 0:
            error_msg, debug_path = _dump_empty_sim_debug_info(
                loss_dict, pairing_data, step_idx, epoch_idx, batch_idx, "pos_sim")
            raise RuntimeError(error_msg)
        if neg_sim.numel() == 0:
            error_msg, debug_path = _dump_empty_sim_debug_info(
                loss_dict, pairing_data, step_idx, epoch_idx, batch_idx, "neg_sim")
            raise RuntimeError(error_msg)

        writer.add_scalar("train/loss_pos_sim", pos_sim.mean().item(), global_step=step_idx)
        writer.add_scalar("train/loss_neg_sim", neg_sim.mean().item(), global_step=step_idx)
        if semipos_sim is not None:
            writer.add_scalar("train/loss_semipos_sim", semipos_sim.mean().item(), global_step=step_idx)

        if step_idx % LOG_HIST_EVERY == 0:
            min_val = min(-1, pos_sim.min().item(), neg_sim.min().item(), semipos_sim.min().item() if semipos_sim is not None else 0.0)
            max_val = max(1, pos_sim.max().item(), neg_sim.max().item(), semipos_sim.max().item() if semipos_sim is not None else 0.0)
            bins = np.linspace(min_val, max_val, 101)
            fig, ax = plt.subplots()
            ax.hist(neg_sim.detach().cpu().float().numpy(), bins=bins, label="Negative", density=False, alpha=0.5)
            ax.hist(pos_sim.detach().cpu().float().numpy(), bins=bins, label="Positive", density=False, alpha=0.5)
            if semipos_sim is not None:
                ax.hist(semipos_sim.detach().cpu().float().numpy(), bins=bins, label="Semi", density=False, alpha=0.5)
            ax.set_yscale("log")
            ax.legend()
            ax.set_xlabel("Similarity")
            ax.set_ylabel("Count")
            writer.add_figure("train/interclass_sim", fig, global_step=step_idx)

    for k, v in loss_dict.items():
        if torch.is_tensor(v) and (v.numel() > 1 or v.numel() == 0):
            continue  # skip larger tensors
        writer.add_scalar(f"train/{k}", v.item() if torch.is_tensor(v) else v, global_step=step_idx)

    if not quiet:
        out_str = f"{epoch_idx=:4d} {batch_idx=:4d} lr: {lr_scheduler.get_last_lr()[0]:.2e} " + \
              f" loss: {loss_dict['loss'].item():0.6f}"
        out_str += f" pos_loss: {loss_dict['pos_loss'].item():0.6f}" if "pos_loss" in loss_dict else ""
        out_str += f" semipos_loss: {loss_dict['semipos_loss'].item():0.6f}" if 'semipos_loss' in loss_dict else ""
        out_str += f" neg_loss: {loss_dict['neg_loss'].item():0.6f}" if "neg_loss" in loss_dict else "" 
        if isinstance(pairing_data, Pairs):
            out_str += f" num_pos_pairs: {len(pairing_data.positive_pairs):3d}" + \
                f" num_semipos_pairs: {len(pairing_data.semipositive_pairs):3d}" + \
                f" num_neg_pairs: {len(pairing_data.negative_pairs):3d}"

        print(out_str, end='\r')
        if batch_idx % 50 == 0:
            print()


@torch.no_grad()
def _sample_pairwise_cosine(x: torch.Tensor, num_pairs: int = 2048) -> tuple[float, float]:
    """Return mean/std of sampled pairwise cosine similarities within a batch."""
    n = x.shape[0]
    if n < 2:
        return float('nan'), float('nan')
    num_pairs = min(num_pairs, n * (n - 1) // 2)
    # Normalize
    x_n = F.normalize(x, dim=1)
    # Sample pairs without replacement by sampling indices (approximate for speed)
    i = torch.randint(0, n, (num_pairs,), device=x.device)
    j = torch.randint(0, n, (num_pairs,), device=x.device)
    keep = i != j
    if keep.sum() == 0:
        return float('nan'), float('nan')
    i = i[keep]
    j = j[keep]
    sims = (x_n[i] * x_n[j]).sum(dim=1)
    return sims.mean().item(), sims.std().item()


@torch.no_grad()
def log_embedding_stats(writer: SummaryWriter, name: str, emb: torch.Tensor, step_idx: int):
    """Log scalar stats and occasional histograms for embeddings."""
    # emb: [B, N_emb, D]
    assert emb.ndim == 3
    norms = emb.norm(dim=2)
    writer.add_scalar(f"{name}_emb/value_mean", emb.mean().item(), step_idx)
    writer.add_scalar(f"{name}_emb/value_std", emb.std().item(), step_idx)
    writer.add_scalar(f"{name}_emb/norm_mean", norms.mean().item(), step_idx)
    writer.add_scalar(f"{name}_emb/norm_std", norms.std().item(), step_idx)

    mean_cos, std_cos = _sample_pairwise_cosine(emb, num_pairs=2048)
    if not math.isnan(mean_cos):
        writer.add_scalar(f"{name}_emb/cosine_mean", mean_cos, step_idx)
        writer.add_scalar(f"{name}_emb/cosine_std", std_cos, step_idx)


@torch.no_grad()
def log_gradient_stats(writer: SummaryWriter, model: torch.nn.Module, name: str, step_idx: int):
    """Log aggregated gradient stats (after unscale_) for a model."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.detach().flatten() for g in grads])
    abs_flat = flat.abs()
    # Global L2 norm
    global_norm = torch.sqrt(torch.sum(flat * flat))
    # Fractions
    zeros_frac = (flat == 0).float().mean()
    nan_frac = torch.isnan(flat).float().mean()
    inf_frac = torch.isinf(flat).float().mean()
    writer.add_scalar(f"grad/{name}/global_norm", global_norm.item(), step_idx)
    writer.add_scalar(f"grad/{name}/mean_abs", abs_flat.mean().item(), step_idx)
    writer.add_scalar(f"grad/{name}/max_abs", abs_flat.max().item(), step_idx)
    writer.add_scalar(f"grad/{name}/zeros_frac", zeros_frac.item(), step_idx)
    writer.add_scalar(f"grad/{name}/nan_frac", nan_frac.item(), step_idx)
    writer.add_scalar(f"grad/{name}/inf_frac", inf_frac.item(), step_idx)



def log_validation_metrics(writer, validation_metrics, epoch_idx, quiet):
    to_print = []
    for key, value in validation_metrics.items():
        to_print.append(f'{key}: {value:0.3f}')
        writer.add_scalar(f"validation/{key}", value, global_step=epoch_idx)

    if not quiet:
        print(f"epoch_idx: {epoch_idx} {' '.join(to_print)}")


def compute_feature_counts_from_extractor_outputs(
    extractor_outputs_by_name: dict[str, 'ExtractorOutput']
) -> dict[str, dict[str, float]]:
    """Compute feature count statistics from extractor outputs.

    Args:
        extractor_outputs_by_name: Dictionary mapping extractor names to their outputs

    Returns:
        Dictionary mapping extractor names to statistics dictionaries containing:
            - mean_count: Average number of unmasked features across the batch
            - total_count: Total number of unmasked features in the batch
            - batch_size: Number of items in the batch
    """
    stats_by_extractor = {}
    for extractor_name, output in extractor_outputs_by_name.items():
        unmasked_counts = (~output.mask).sum(dim=1).float()
        stats_by_extractor[extractor_name] = {
            'mean_count': unmasked_counts.mean().item(),
            'total_count': unmasked_counts.sum().item(),
            'batch_size': output.mask.shape[0],
        }
    return stats_by_extractor


@torch.no_grad()
def log_feature_counts(writer: SummaryWriter,
                       pano_extractor_outputs: dict[str, 'ExtractorOutput'],
                       sat_extractor_outputs: dict[str, 'ExtractorOutput'],
                       step_idx: int):
    """Log feature counts per extractor for both panorama and satellite models.

    Args:
        writer: TensorBoard SummaryWriter
        pano_extractor_outputs: Extractor outputs from panorama model's forward pass
        sat_extractor_outputs: Extractor outputs from satellite model's forward pass
        step_idx: Current training step for logging
    """
    # Log panorama extractor feature counts
    pano_stats = compute_feature_counts_from_extractor_outputs(pano_extractor_outputs)
    for extractor_name, stats in pano_stats.items():
        writer.add_scalar(
            f"features/pano/{extractor_name}/mean_count",
            stats['mean_count'],
            step_idx)
        writer.add_scalar(
            f"features/pano/{extractor_name}/total_count",
            stats['total_count'],
            step_idx)

    # Log satellite extractor feature counts
    sat_stats = compute_feature_counts_from_extractor_outputs(sat_extractor_outputs)
    for extractor_name, stats in sat_stats.items():
        writer.add_scalar(
            f"features/sat/{extractor_name}/mean_count",
            stats['mean_count'],
            step_idx)
        writer.add_scalar(
            f"features/sat/{extractor_name}/total_count",
            stats['total_count'],
            step_idx)

    # Log token type counts from debug dicts (for composable extractors)
    for extractor_name, output in pano_extractor_outputs.items():
        for key, value in output.debug.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(
                    f"token_types/pano/{extractor_name}/{key}",
                    value,
                    step_idx)

    for extractor_name, output in sat_extractor_outputs.items():
        for key, value in output.debug.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(
                    f"token_types/sat/{extractor_name}/{key}",
                    value,
                    step_idx)
