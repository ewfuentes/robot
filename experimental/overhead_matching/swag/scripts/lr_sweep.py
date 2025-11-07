import torch
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


@dataclass
class LearningRateSweepConfig:
    start_lr: float = 1.5e-7  # 0.1x burn-in LR
    end_lr: float = 1.5e-1    # 100x burn-in LR
    num_batches: int = 1000   # 10x more batches for fine resolution
    burn_in_batches: int = 50
    burn_in_lr: float = 1.5e-5


def run_lr_sweep(
        lr_sweep_config: LearningRateSweepConfig,
        dataset,
        panorama_model,
        satellite_model,
        opt_config,
        output_dir: Path,
        compute_forward_pass_and_loss_fn,
        create_training_components_fn,
        setup_models_for_training_fn,
        quiet: bool = False) -> float:
    """
    Run learning rate sweep and return the optimal learning rate.
    Saves a plot showing loss vs learning rate.
    """
    if not quiet:
        print(f"Running learning rate sweep: burn-in ({lr_sweep_config.burn_in_batches} batches @ {lr_sweep_config.burn_in_lr:.2e}) + sweep ({lr_sweep_config.num_batches} batches: {lr_sweep_config.start_lr:.2e} → {lr_sweep_config.end_lr:.2e})")
    
    # Setup models
    panorama_model, satellite_model = setup_models_for_training_fn(panorama_model, satellite_model)
    
    # Create training components
    miner, dataloader, opt = create_training_components_fn(dataset, panorama_model, satellite_model, opt_config)
    
    # Start with burn-in learning rate
    for param_group in opt.param_groups:
        param_group['lr'] = lr_sweep_config.burn_in_lr
    
    # Calculate learning rate multiplier for exponential increase
    lr_multiplier = (lr_sweep_config.end_lr / lr_sweep_config.start_lr) ** (1.0 / lr_sweep_config.num_batches)
    
    learning_rates = []
    losses = []
    
    grad_scaler = torch.amp.GradScaler()
    
    # Phase 1: Burn-in phase with fixed learning rate
    batch_count = 0
    burn_in_loss = None
    
    if not quiet:
        print("Phase 1: Burn-in phase...")
    
    for batch in dataloader:
        if batch_count >= lr_sweep_config.burn_in_batches:
            break
            
        opt.zero_grad()
        
        # Use extracted function for forward pass and loss
        loss_dict, pairs, panorama_embeddings, sat_embeddings = compute_forward_pass_and_loss_fn(
            batch, panorama_model, satellite_model, opt_config)
        
        loss = loss_dict["loss"]
        burn_in_loss = loss.item()
        
        # Record burn-in data for plotting
        learning_rates.append(lr_sweep_config.burn_in_lr)
        losses.append(loss.item())
        
        if not quiet and batch_count % 10 == 0:
            print(f"Burn-in batch {batch_count}: lr={lr_sweep_config.burn_in_lr:.2e}, loss={loss.item():.4f}")
        
        grad_scaler.scale(loss).backward()
        grad_scaler.step(opt)
        grad_scaler.update()
        
        batch_count += 1
    
    if not quiet:
        print(f"Burn-in complete. Final loss: {burn_in_loss:.4f}")
        print("Phase 2: Learning rate sweep...")
    
    # Phase 2: Learning rate sweep - create fresh dataloader iterator
    batch_count = 0
    min_loss = burn_in_loss if burn_in_loss is not None else float('inf')
    
    # Get a fresh iterator for the sweep phase
    dataloader_iter = iter(dataloader)
    
    for _ in range(lr_sweep_config.num_batches):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # If we run out of data, restart the dataloader
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        current_lr = lr_sweep_config.start_lr * (lr_multiplier ** batch_count)
        
        # Update learning rate
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr
        
        opt.zero_grad()
        
        # Use extracted function for forward pass and loss
        loss_dict, pairs, panorama_embeddings, sat_embeddings = compute_forward_pass_and_loss_fn(
            batch, panorama_model, satellite_model, opt_config)
        
        loss = loss_dict["loss"]
        
        # Early stopping if loss explodes
        if loss > min_loss * 10 and batch_count > 10:
            if not quiet:
                print(f"Early stopping at sweep batch {batch_count} - loss exploded")
            break
        
        min_loss = min(min_loss, loss.item())
        learning_rates.append(current_lr)
        losses.append(loss.item())
        
        if not quiet and batch_count % 100 == 0:  # Less frequent logging for 1000 batches
            print(f"Sweep batch {batch_count}: lr={current_lr:.2e}, loss={loss.item():.4f}")
        
        grad_scaler.scale(loss).backward()
        grad_scaler.step(opt)
        grad_scaler.update()
        
        batch_count += 1
    
    # Find optimal learning rate (steepest descent point)
    # Focus on sweep phase only for analysis (exclude burn-in)
    sweep_start_idx = lr_sweep_config.burn_in_batches
    if len(losses) > sweep_start_idx and len(losses) - sweep_start_idx > 10:
        sweep_losses = losses[sweep_start_idx:]
        sweep_lrs = learning_rates[sweep_start_idx:]
        
        # Smooth the loss curve with smaller window for higher resolution
        window_size = max(5, len(sweep_losses) // 50)  # More aggressive smoothing
        smoothed_losses = np.convolve(sweep_losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_lrs = sweep_lrs[window_size//2:len(sweep_lrs)-window_size//2+1]
        
        # Find steepest descent (most negative gradient)
        gradients = np.gradient(smoothed_losses)
        optimal_idx = np.argmin(gradients)
        optimal_lr = smoothed_lrs[optimal_idx]
        
        # Additional analysis: find minimum loss LR as alternative
        min_loss_idx = np.argmin(smoothed_losses)
        min_loss_lr = smoothed_lrs[min_loss_idx]
        
        if not quiet:
            print(f"Analysis: Steepest descent at {optimal_lr:.2e}, Min loss at {min_loss_lr:.2e}")
    else:
        # Fallback to simple minimum if not enough data
        optimal_lr = learning_rates[np.argmin(losses)]
        if not quiet:
            print(f"Limited data - using simple minimum: {optimal_lr:.2e}")
    
    # Create and save plot
    plt.figure(figsize=(12, 8))
    
    # Split data into burn-in and sweep phases
    burn_in_end = lr_sweep_config.burn_in_batches
    burn_in_lrs = learning_rates[:burn_in_end] if burn_in_end <= len(learning_rates) else []
    burn_in_losses = losses[:burn_in_end] if burn_in_end <= len(losses) else []
    sweep_lrs = learning_rates[burn_in_end:]
    sweep_losses = losses[burn_in_end:]
    
    # Plot burn-in phase
    if burn_in_lrs:
        plt.semilogx(burn_in_lrs, burn_in_losses, 'g-', linewidth=2, 
                     alpha=0.7, label=f'Burn-in ({lr_sweep_config.burn_in_lr:.2e})')
    
    # Plot sweep phase
    if sweep_lrs:
        plt.semilogx(sweep_lrs, sweep_losses, 'b-', linewidth=2, label='LR Sweep')
    
    # Mark optimal learning rate
    plt.axvline(optimal_lr, color='r', linestyle='--', linewidth=2, 
                label=f'Suggested LR: {optimal_lr:.2e}')
    
    # Add burn-in LR reference line
    if burn_in_lrs:
        plt.axvline(lr_sweep_config.burn_in_lr, color='g', linestyle=':', linewidth=1, 
                    alpha=0.7, label=f'Burn-in LR: {lr_sweep_config.burn_in_lr:.2e}')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Sweep with Burn-in Phase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = output_dir / "lr_sweep_plot.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if not quiet:
        print(f"Learning rate sweep completed. Optimal LR: {optimal_lr:.2e}")
        print(f"Plot saved to: {plot_path}")
    
    return optimal_lr