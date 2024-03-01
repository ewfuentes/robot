
import torch

import triton
import triton.language as tl


@triton.jit
def distance_kernel(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def nearest_neighbors(x: torch.Tensor):
    num_nodes = x.size(0)
    NUM_SOURCES_PER_BLOCK = 512
    workspace = torch.empty((x.size(0), NUM_SOURCES_PER_BLOCK), device='cuda')
    assert x.is_cuda and workspace.is_cuda

    for source_start in range(0, num_nodes, NUM_SOURCES_PER_BLOCK):
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), NUM_SOURCES_PER_BLOCK)
        add_kernel[grid](x, output, num_nodes, BLOCK_SIZE=512)

    return output
