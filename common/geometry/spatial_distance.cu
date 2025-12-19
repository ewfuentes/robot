#include "common/geometry/spatial_distance.hh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/csrc/autograd/profiler.h>

namespace robot::geometry {

// Device function for point-to-segment distance
__device__ inline float point_to_segment_distance(
    float px, float py,
    float ax, float ay,
    float bx, float by
) {
    // Vector from A to B
    float abx = bx - ax;
    float aby = by - ay;

    // |AB|^2
    float ab_len_sq = abx * abx + aby * aby;

    // Handle degenerate case (A == B)
    if (ab_len_sq < 1e-12f) {
        float dx = px - ax;
        float dy = py - ay;
        return sqrtf(dx * dx + dy * dy);
    }

    // Vector from A to P
    float apx = px - ax;
    float apy = py - ay;

    // Compute parameter t
    float t = (apx * abx + apy * aby) / ab_len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));

    // Closest point on segment
    float closest_x = ax + t * abx;
    float closest_y = ay + t * aby;

    // Distance from P to closest point
    float dx = px - closest_x;
    float dy = py - closest_y;
    return sqrtf(dx * dx + dy * dy);
}

// Optimized kernel: one block per cell, shared memory for segments
__global__ void compute_distances_optimized_kernel(
    // Sorted particle data
    const float* __restrict__ sorted_particles,  // (N, 2) sorted by cell

    // Block assignments (indexed by blockIdx.x)
    const int64_t* __restrict__ particle_starts,  // (num_blocks,)
    const int64_t* __restrict__ particle_ends,    // (num_blocks,)
    const int64_t* __restrict__ cell_ids,         // (num_blocks,)
    const int64_t* __restrict__ num_geoms_per_block,  // (num_blocks,)

    // Geometry data (global)
    const float* __restrict__ segment_starts,  // (M_seg, 2)
    const float* __restrict__ segment_ends,    // (M_seg, 2)
    const int64_t* __restrict__ segment_to_geom,  // (M_seg,)

    // Spatial index
    const int64_t* __restrict__ cell_segment_indices,
    const int64_t* __restrict__ cell_offsets,

    // Local-to-global geometry mapping (flattened)
    const int64_t* __restrict__ local_to_global_geom,  // Concatenated for all blocks
    const int64_t* __restrict__ local_to_global_offsets,  // (num_blocks+1,)
    const int64_t* __restrict__ global_to_local_geom,  // (num_geometries,) lookup table

    // Output (dense per block)
    float* __restrict__ output_distances,  // Preallocated dense output
    const int64_t* __restrict__ output_offsets  // (num_blocks,) offset for each block
) {
    int block_id = blockIdx.x;

    // Get this block's assignment
    int64_t particle_start = particle_starts[block_id];
    int64_t particle_end = particle_ends[block_id];
    int64_t cell_id = cell_ids[block_id];
    int64_t num_geometries = num_geoms_per_block[block_id];
    int64_t output_offset = output_offsets[block_id];

    int64_t num_particles = particle_end - particle_start;

    // Get segments in this cell
    int64_t seg_start_idx = cell_offsets[cell_id];
    int64_t seg_end_idx = cell_offsets[cell_id + 1];
    int64_t num_segments = seg_end_idx - seg_start_idx;

    // Shared memory for segments (up to 512 segments)
    __shared__ float seg_starts_shared[512][2];
    __shared__ float seg_ends_shared[512][2];
    __shared__ int64_t seg_local_geom_ids[512];

    // Cooperatively load segments into shared memory
    for (int64_t i = threadIdx.x; i < num_segments && i < 512; i += blockDim.x) {
        int64_t global_seg_idx = cell_segment_indices[seg_start_idx + i];
        seg_starts_shared[i][0] = segment_starts[global_seg_idx * 2];
        seg_starts_shared[i][1] = segment_starts[global_seg_idx * 2 + 1];
        seg_ends_shared[i][0] = segment_ends[global_seg_idx * 2];
        seg_ends_shared[i][1] = segment_ends[global_seg_idx * 2 + 1];

        // Map global geom ID to local
        int64_t global_geom_id = segment_to_geom[global_seg_idx];
        seg_local_geom_ids[i] = global_to_local_geom[global_geom_id];
    }

    __syncthreads();

    // Each thread processes particles (loop if num_particles > blockDim.x)
    for (int64_t p_offset = 0; p_offset < num_particles; p_offset += blockDim.x) {
        int64_t particle_idx = particle_start + threadIdx.x + p_offset;
        if (particle_idx >= particle_end) continue;

        float px = sorted_particles[particle_idx * 2];
        float py = sorted_particles[particle_idx * 2 + 1];

        // Track min distance per local geometry ID (up to 128 geometries)
        float min_dists[128];
        for (int64_t i = 0; i < num_geometries && i < 128; ++i) {
            min_dists[i] = INFINITY;
        }

        // Iterate over segments in shared memory
        int64_t num_segs_to_process = (num_segments < 512) ? num_segments : 512;
        for (int64_t i = 0; i < num_segs_to_process; ++i) {
            float dist = point_to_segment_distance(
                px, py,
                seg_starts_shared[i][0], seg_starts_shared[i][1],
                seg_ends_shared[i][0], seg_ends_shared[i][1]
            );

            int64_t local_geom_id = seg_local_geom_ids[i];
            if (local_geom_id < 128 && local_geom_id >= 0) {
                min_dists[local_geom_id] = fminf(min_dists[local_geom_id], dist);
            }
        }

        // Write output (dense: particle × geometry)
        int64_t local_particle_idx = particle_idx - particle_start;
        int64_t output_base = output_offset + local_particle_idx * num_geometries;

        for (int64_t local_geom_id = 0; local_geom_id < num_geometries && local_geom_id < 128; ++local_geom_id) {
            output_distances[output_base + local_geom_id] = min_dists[local_geom_id];
        }
    }
}

torch::Tensor query_distances_cuda(
    const torch::Tensor& query_points,
    const torch::Tensor& segment_starts,
    const torch::Tensor& segment_ends,
    const torch::Tensor& segment_to_geom,
    const torch::Tensor& point_coords,
    const torch::Tensor& point_to_geom,
    const torch::Tensor& geometry_types,
    const torch::Tensor& polygon_vertices,
    const torch::Tensor& polygon_ranges,
    const torch::Tensor& polygon_geom_indices,
    int64_t num_geometries,
    const torch::Tensor& cell_segment_indices,
    const torch::Tensor& cell_offsets,
    const torch::Tensor& cell_point_indices,
    const torch::Tensor& cell_point_offsets,
    const torch::Tensor& grid_origin,
    float cell_size,
    const torch::Tensor& grid_dims
) {
    // Check inputs
    TORCH_CHECK(query_points.is_cuda(), "query_points must be on CUDA");
    TORCH_CHECK(query_points.dim() == 2 && query_points.size(1) == 2,
                "query_points must be (N, 2)");
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32,
                "query_points must be float32");

    int64_t num_queries = query_points.size(0);
    if (num_queries == 0) {
        return torch::empty({0, 3}, query_points.options());
    }

    auto device = query_points.device();
    c10::cuda::CUDAGuard device_guard(device);

    torch::Tensor sorted_particles;
    torch::Tensor sorted_particle_indices;
    torch::Tensor sorted_cell_ids;
    torch::Tensor unique_cells;
    torch::Tensor counts;

    // Step 1: Hash particles to cells
    {
        RECORD_FUNCTION("cuda_kernel::hash_and_sort", std::vector<c10::IValue>{});
        auto cell_coords = ((query_points - grid_origin) / cell_size).floor().to(torch::kInt64);
        auto in_bounds = (cell_coords.select(1, 0) >= 0) &
                         (cell_coords.select(1, 0) < grid_dims[0]) &
                         (cell_coords.select(1, 1) >= 0) &
                         (cell_coords.select(1, 1) < grid_dims[1]);

        auto cell_ids = cell_coords.select(1, 1) * grid_dims[0] + cell_coords.select(1, 0);

        // Step 2: Sort particles by cell ID
        auto valid_indices = torch::where(in_bounds)[0];
        if (valid_indices.numel() == 0) {
            return torch::empty({0, 3}, query_points.options());
        }

        auto valid_cell_ids = cell_ids.index_select(0, valid_indices);
        auto sorted_result = valid_cell_ids.sort();
        sorted_cell_ids = std::get<0>(sorted_result);
        auto sort_indices = std::get<1>(sorted_result);

        sorted_particle_indices = valid_indices.index_select(0, sort_indices);
        sorted_particles = query_points.index_select(0, sorted_particle_indices);
    }

    std::vector<int64_t> particle_starts_vec;
    std::vector<int64_t> particle_ends_vec;
    std::vector<int64_t> cell_ids_vec;
    std::vector<int64_t> num_geoms_vec;
    std::vector<int64_t> local_to_global_vec;
    std::vector<int64_t> local_to_global_offsets_vec;
    std::vector<int64_t> output_offsets_vec;
    int64_t cumulative_output_offset = 0;
    torch::Tensor particle_starts, particle_ends, block_cell_ids, num_geoms_per_block, output_offsets;
    torch::Tensor local_to_global, local_to_global_offsets, global_to_local;

    // Step 3: Build block assignments
    {
        RECORD_FUNCTION("cuda_kernel::build_block_assignments", std::vector<c10::IValue>{});
        // Use unique_consecutive with return_counts to get ranges efficiently
        auto unique_result = torch::unique_consecutive(sorted_cell_ids, false, true, torch::nullopt);
        unique_cells = std::get<0>(unique_result);
        counts = std::get<1>(unique_result);

        int64_t num_blocks = unique_cells.numel();
        int64_t cumulative_geom_offset = 0;

    // Move all tensors to CPU before loop
    auto unique_cells_cpu = unique_cells.cpu();
    auto counts_cpu = counts.cpu();
    auto segment_to_geom_cpu = segment_to_geom.cpu();
    auto cell_offsets_cpu = cell_offsets.cpu();
    auto cell_segment_indices_cpu = cell_segment_indices.cpu();

    // Use accessors for fast CPU tensor access
    auto unique_cells_acc = unique_cells_cpu.accessor<int64_t, 1>();
    auto counts_acc = counts_cpu.accessor<int64_t, 1>();
    auto segment_to_geom_acc = segment_to_geom_cpu.accessor<int64_t, 1>();
    auto cell_offsets_acc = cell_offsets_cpu.accessor<int64_t, 1>();
    auto cell_segment_indices_acc = cell_segment_indices_cpu.accessor<int64_t, 1>();

    int64_t p_cumulative = 0;
    for (int64_t b = 0; b < num_blocks; ++b) {
        int64_t cell_id = unique_cells_acc[b];
        int64_t num_particles_in_cell = counts_acc[b];
        int64_t p_start = p_cumulative;
        int64_t p_end = p_cumulative + num_particles_in_cell;
        p_cumulative = p_end;

        // Get segments in this cell
        int64_t seg_start = cell_offsets_acc[cell_id];
        int64_t seg_end = cell_offsets_acc[cell_id + 1];

        if (seg_start == seg_end) continue;  // Skip empty cells

        // Build local→global geometry mapping
        std::vector<int64_t> geom_ids_in_cell;
        for (int64_t s = seg_start; s < seg_end; ++s) {
            int64_t seg_idx = cell_segment_indices_acc[s];
            int64_t geom_id = segment_to_geom_acc[seg_idx];
            if (std::find(geom_ids_in_cell.begin(), geom_ids_in_cell.end(), geom_id) == geom_ids_in_cell.end()) {
                geom_ids_in_cell.push_back(geom_id);
            }
        }

        int64_t num_geoms_in_cell = geom_ids_in_cell.size();
        if (num_geoms_in_cell > 128) num_geoms_in_cell = 128;  // Limit

        particle_starts_vec.push_back(p_start);
        particle_ends_vec.push_back(p_end);
        cell_ids_vec.push_back(cell_id);
        num_geoms_vec.push_back(num_geoms_in_cell);
        local_to_global_offsets_vec.push_back(cumulative_geom_offset);
        output_offsets_vec.push_back(cumulative_output_offset);

        for (int64_t g = 0; g < num_geoms_in_cell; ++g) {
            local_to_global_vec.push_back(geom_ids_in_cell[g]);
        }

        cumulative_output_offset += num_particles_in_cell * num_geoms_in_cell;
        cumulative_geom_offset += num_geoms_in_cell;
    }

    local_to_global_offsets_vec.push_back(cumulative_geom_offset);

    if (particle_starts_vec.empty()) {
        return torch::empty({0, 3}, query_points.options());
    }

        // Convert to tensors
        particle_starts = torch::from_blob(particle_starts_vec.data(), {(int64_t)particle_starts_vec.size()},
                                               torch::kInt64).clone().to(device);
        particle_ends = torch::from_blob(particle_ends_vec.data(), {(int64_t)particle_ends_vec.size()},
                                              torch::kInt64).clone().to(device);
        block_cell_ids = torch::from_blob(cell_ids_vec.data(), {(int64_t)cell_ids_vec.size()},
                                               torch::kInt64).clone().to(device);
        num_geoms_per_block = torch::from_blob(num_geoms_vec.data(), {(int64_t)num_geoms_vec.size()},
                                                    torch::kInt64).clone().to(device);
        output_offsets = torch::from_blob(output_offsets_vec.data(), {(int64_t)output_offsets_vec.size()},
                                              torch::kInt64).clone().to(device);
        local_to_global = torch::from_blob(local_to_global_vec.data(), {(int64_t)local_to_global_vec.size()},
                                               torch::kInt64).clone().to(device);
        local_to_global_offsets = torch::from_blob(local_to_global_offsets_vec.data(),
                                                       {(int64_t)local_to_global_offsets_vec.size()},
                                                       torch::kInt64).clone().to(device);

        // Build global→local lookup table
        global_to_local = torch::full({num_geometries}, -1L, torch::TensorOptions().dtype(torch::kInt64).device(device));
        for (size_t b = 0; b < particle_starts_vec.size(); ++b) {
            int64_t offset = local_to_global_offsets_vec[b];
            int64_t next_offset = local_to_global_offsets_vec[b + 1];
            for (int64_t i = offset; i < next_offset; ++i) {
                int64_t global_geom_id = local_to_global_vec[i];
                int64_t local_geom_id = i - offset;
                global_to_local[global_geom_id] = local_geom_id;
            }
        }
    }

    // Allocate output
    auto output_distances = torch::full({cumulative_output_offset}, INFINITY,
                                       torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Launch kernel
    {
        RECORD_FUNCTION("cuda_kernel::compute_distances_kernel", std::vector<c10::IValue>{});
        int num_blocks_cuda = particle_starts_vec.size();
        const int threads = 256;

        compute_distances_optimized_kernel<<<num_blocks_cuda, threads>>>(
        sorted_particles.data_ptr<float>(),
        particle_starts.data_ptr<int64_t>(),
        particle_ends.data_ptr<int64_t>(),
        block_cell_ids.data_ptr<int64_t>(),
        num_geoms_per_block.data_ptr<int64_t>(),
        segment_starts.data_ptr<float>(),
        segment_ends.data_ptr<float>(),
        segment_to_geom.data_ptr<int64_t>(),
        cell_segment_indices.data_ptr<int64_t>(),
        cell_offsets.data_ptr<int64_t>(),
        local_to_global.data_ptr<int64_t>(),
        local_to_global_offsets.data_ptr<int64_t>(),
        global_to_local.data_ptr<int64_t>(),
        output_distances.data_ptr<float>(),
        output_offsets.data_ptr<int64_t>()
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Step 4: Convert dense output to sparse
    RECORD_FUNCTION("cuda_kernel::dense_to_sparse", std::vector<c10::IValue>{});
    std::vector<int64_t> result_particle_vec;
    std::vector<int64_t> result_geom_vec;
    std::vector<float> result_dist_vec;

    auto output_distances_cpu = output_distances.cpu();
    auto sorted_particle_indices_cpu = sorted_particle_indices.cpu();

    // Use accessors for fast access
    auto output_distances_acc = output_distances_cpu.accessor<float, 1>();
    auto sorted_particle_indices_acc = sorted_particle_indices_cpu.accessor<int64_t, 1>();

    for (size_t b = 0; b < particle_starts_vec.size(); ++b) {
        int64_t p_start = particle_starts_vec[b];
        int64_t p_end = particle_ends_vec[b];
        int64_t num_geoms = num_geoms_vec[b];
        int64_t out_offset = output_offsets_vec[b];
        int64_t geom_offset = local_to_global_offsets_vec[b];

        for (int64_t p = p_start; p < p_end; ++p) {
            int64_t global_p_idx = sorted_particle_indices_acc[p];
            int64_t local_p = p - p_start;

            for (int64_t g = 0; g < num_geoms; ++g) {
                float dist = output_distances_acc[out_offset + local_p * num_geoms + g];
                if (dist < INFINITY) {
                    int64_t global_geom_id = local_to_global_vec[geom_offset + g];
                    result_particle_vec.push_back(global_p_idx);
                    result_geom_vec.push_back(global_geom_id);
                    result_dist_vec.push_back(dist);
                }
            }
        }
    }

    if (result_particle_vec.empty()) {
        return torch::empty({0, 3}, query_points.options());
    }

    // Build result tensor
    auto result_particles = torch::from_blob(result_particle_vec.data(), {(int64_t)result_particle_vec.size()},
                                            torch::kInt64).clone().to(device).to(torch::kFloat32);
    auto result_geoms = torch::from_blob(result_geom_vec.data(), {(int64_t)result_geom_vec.size()},
                                        torch::kInt64).clone().to(device).to(torch::kFloat32);
    auto result_dists = torch::from_blob(result_dist_vec.data(), {(int64_t)result_dist_vec.size()},
                                        torch::kFloat32).clone().to(device);

    auto result = torch::stack({result_particles, result_geoms, result_dists}, 1);

    return result;
}

}  // namespace robot::geometry
