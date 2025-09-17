import math
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


LOG_HIST_EVERY = 100


def log_batch_metrics(writer, loss_dict, lr_scheduler, pairs, step_idx, epoch_idx, batch_idx, quiet):
    writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step=step_idx)
    if "pos_sim" in loss_dict and "neg_sim" in loss_dict:
        pos_sim = loss_dict["pos_sim"]
        neg_sim = loss_dict["neg_sim"]
        semipos_sim = loss_dict["semipos_sim"] if "semipos_sim" in loss_dict else None  # may not be present 
        if semipos_sim is not None and semipos_sim.numel() == 0: 
            semipos_sim = None
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
        writer.add_scalar(f"train/loss_{k}", v.item() if torch.is_tensor(v) else v, global_step=step_idx)

    if not quiet:
        print(f"{epoch_idx=:4d} {batch_idx=:4d} lr: {lr_scheduler.get_last_lr()[0]:.2e} " +
              f" num_pos_pairs: {len(pairs.positive_pairs):3d}" +
              f" num_semipos_pairs: {len(pairs.semipositive_pairs):3d}" +
              f" num_neg_pairs: {len(pairs.negative_pairs):3d}" +
              f" pos_loss: {loss_dict['pos_loss'].item():0.6f}" +
              f" semipos_loss: {loss_dict['semipos_loss'].item():0.6f}" +
              f" neg_loss: {loss_dict['neg_loss'].item():0.6f}" +
              f" loss: {loss_dict['loss'].item():0.6f}",
              end='\r')
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
    # emb: [B, D]
    norms = emb.norm(dim=1)
    writer.add_scalar(f"{name}_emb/value_mean", emb.mean().item(), step_idx)
    writer.add_scalar(f"{name}_emb/value_std", emb.std().item(), step_idx)
    writer.add_scalar(f"{name}_emb/norm_mean", norms.mean().item(), step_idx)
    writer.add_scalar(f"{name}_emb/norm_std", norms.std().item(), step_idx)
    writer.add_scalar(f"{name}_emb/norm_min", norms.min().item(), step_idx)
    writer.add_scalar(f"{name}_emb/norm_max", norms.max().item(), step_idx)

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

    if step_idx % LOG_HIST_EVERY == 0:
        # Downsample to keep TB light
        vals = flat
        if vals.numel() > 200_000:
            idx = torch.randint(0, vals.numel(), (200_000,), device=vals.device)
            vals = vals[idx]
        writer.add_histogram(f"grad/{name}/values", vals.float().cpu(), step_idx)


def log_validation_metrics(writer, validation_metrics, epoch_idx, quiet):
    to_print = []
    for key, value in validation_metrics.items():
        to_print.append(f'{key}: {value:0.3f}')
        writer.add_scalar(f"validation/{key}", value, global_step=epoch_idx)

    if not quiet:
        print(f"epoch_idx: {epoch_idx} {' '.join(to_print)}")
