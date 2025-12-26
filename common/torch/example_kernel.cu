
#include <c10/cuda/CUDAGuard.h>

#include "common/torch/example_kernel.hh"

namespace robot::torch {

// Template CUDA kernel that works with any scalar type
template <typename scalar_t>
__global__ void square_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
                              int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t val = input[idx];
        output[idx] = val * val;
    }
}

::torch::Tensor square(const ::torch::Tensor& in) {
    // Check input is on CUDA
    TORCH_CHECK(in.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(in.is_contiguous(), "Input tensor must be contiguous");

    // Create output tensor with same shape and options as input
    auto output = ::torch::empty_like(in);

    // Get number of elements
    int64_t numel = in.numel();
    if (numel == 0) return output;

    // Set CUDA device
    c10::cuda::CUDAGuard device_guard(in.device());

    // Configure kernel launch
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    // Dispatch to the right kernel based on dtype
    AT_DISPATCH_ALL_TYPES(in.scalar_type(), "square_kernel", [&] {
        square_kernel<scalar_t>
            <<<blocks, threads>>>(in.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), numel);
    });

    // Check for CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
}  // namespace robot::torch
