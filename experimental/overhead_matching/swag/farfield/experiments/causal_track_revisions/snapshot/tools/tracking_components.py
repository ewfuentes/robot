"""CPU implementation of SAM2's consumed 8-connected-component semantics.

Component label numbers are arbitrary. SAM2 consumes foreground membership
and each component's area; those are preserved by OpenCV's 8-connectivity.
This avoids silently skipping hole filling when SAM2's CUDA extension is absent.
"""
import cv2
import numpy as np
import torch


def connected_components_cpu(mask):
    if mask.ndim!=4 or mask.shape[1]!=1:
        raise ValueError('expected binary mask shape (N,1,H,W)')
    binary=mask.detach().to(device='cpu',dtype=torch.uint8).contiguous().numpy()
    labels=np.empty(binary.shape,dtype=np.int32)
    areas=np.empty(binary.shape,dtype=np.int32)
    for i in range(len(binary)):
        _,component,stats,_=cv2.connectedComponentsWithStats(binary[i,0],connectivity=8)
        counts=stats[:,cv2.CC_STAT_AREA].copy()
        counts[0]=0
        labels[i,0]=component
        areas[i,0]=counts[component]
    return torch.from_numpy(labels).to(mask.device),torch.from_numpy(areas).to(mask.device)


def install_cpu_components():
    from sam2.utils import misc
    misc.get_connected_components=connected_components_cpu
