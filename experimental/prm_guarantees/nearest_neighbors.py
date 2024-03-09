import torch

import triton
import triton.language as tl


@triton.jit
def distance_kernel(
    x_ptr, output_ptr, n_elements, source_ptr, num_sources, N_DIMS:tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    block_id = tl.program_id(axis=0)
    source_id = tl.program_id(axis=1)

    print("pid", block_id, source_id)

    block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_elements, N_DIMS),
        strides=(N_DIMS, 1),
        offsets=(block_id * BLOCK_SIZE, 0),
        block_shape=(BLOCK_SIZE, N_DIMS),
        order=(1, 0)
    )

    source_block_ptr = tl.make_block_ptr(
        base=source_ptr,
        shape=(num_sources, N_DIMS),
        strides=(N_DIMS, 1),
        offsets=(block_id, 0),
        block_shape=(1, N_DIMS),
        order=(1, 0)
    )
    x = tl.load(block_ptr)
    source = tl.load(source_block_ptr)
    print("data:", x)
    print("source:", source)


#     mask = offsets < n_elements
#
#     x = tl.load(x_ptr + offsets, mask=mask)
#     output = x + y
#     tl.store(output_ptr + offsets, output, mask=mask)


def nearest_neighbors(x: torch.Tensor):
    num_nodes = x.size(0)
    num_dims = x.size(1)
    MAX_NUM_SOURCES_PER_BLOCK = 2048
    BLOCK_SIZE = 512
    workspace = torch.empty((x.size(0), MAX_NUM_SOURCES_PER_BLOCK), device="cuda")
    assert x.is_cuda and workspace.is_cuda

    for source_start in range(0, num_nodes, MAX_NUM_SOURCES_PER_BLOCK):
        print("source start:", source_start)
        source_end = min(source_start + MAX_NUM_SOURCES_PER_BLOCK, num_nodes)
        num_sources = source_end - source_start
        grid = lambda meta: (
            triton.cdiv(num_nodes, meta["BLOCK_SIZE"]),
            num_sources,
        )
        result = distance_kernel[grid](
            x,
            workspace,
            num_nodes,
            x[source_start:source_end, :],
            num_sources,
            num_dims,
            BLOCK_SIZE,
        )
        ...
    #         add_kernel[grid](x, output, num_nodes, BLOCK_SIZE=512)

    return None
