import torch
import itertools


def histogram_patch(input: torch.Tensor, *, bin_edges: torch.Tensor):
    assert len(input.shape) == 2
    counts, _ = torch.histogram(input.flatten(), bin_edges, density=False)
    return counts


def color_histogram(
    input: torch.Tensor,
    *,
    bins: int,
    size: int,
    stride: int,
    min_val: None | float = None,
    max_val: None | float = None
):
    """
    Given a [batch size, channel, height, width] tensor, return a [batch size, channel, height',
    width', bin] tensor where height' and width' are functions of `padding` and `stride`. The 
    final channel contains the histogram for that patch and channel. Also returns the bin edges.
    """
    assert len(input.shape) == 4
    assert bins > 1
    assert stride >= 1

    if min_val is None:
        min_val = torch.min(input).item()

    if max_val is None:
        max_val = torch.max(input).item()

    bin_size = (max_val - min_val) / bins
    bin_edges = torch.tensor([i * bin_size for i in range(bins)] + [max_val])

    # There are no native torch ops we can take advantage of, so we do it the old fashioned way
    n_batch = input.shape[0]
    n_channel = input.shape[1]
    row_patch_idxs = list(range(0, input.shape[2] - size + 1, stride))
    col_patch_idxs = list(range(0, input.shape[3] - size + 1, stride))

    result = torch.zeros(
        (n_batch, n_channel, len(row_patch_idxs), len(col_patch_idxs), bins)
    )
    for batch_idx, channel_idx in itertools.product(range(n_batch), range(n_channel)):
        for row_patch_idx, col_patch_idx in itertools.product(
            range(len(row_patch_idxs)), range(len(col_patch_idxs))
        ):
            top = row_patch_idxs[row_patch_idx]
            left = col_patch_idxs[col_patch_idx]

            patch = input[batch_idx, channel_idx, top : top + size, left : left + size]
            result[
                batch_idx, channel_idx, row_patch_idx, col_patch_idx, :
            ] = histogram_patch(patch, bin_edges=bin_edges)
    return result, bin_edges
