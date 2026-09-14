"""Current-state composite evidence replacement; no truth or future inputs.

The ratio telescopes exactly for a static pose and fixed exponent. With motion
diffusion, recomputing both factors at the current pose is an approximation.
It does not replace or rescore estimates that have already been emitted.
"""
import torch

def replacement_ratio(new, old, exponent):
    if new.shape!=old.shape or not 0<=exponent<=1:
        raise ValueError('Invalid factor shapes or exponent')
    if not bool(torch.isfinite(new).all() and torch.isfinite(old).all()) or not bool((new>0).all() and (old>0).all()):
        raise ValueError('Replacement requires finite positive factors')
    ratio=new/old
    return ratio if exponent==1 else ratio.pow(exponent)
