#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <torch/csrc/autograd/profiler.h>
#include "fmt/base.h"

#include <algorithm>
#include <tuple>
#include <vector>

#include "common/geometry/spatial_distance.hh"

template <>
struct fmt::formatter<::torch::Tensor> {
  bool shape_only = false;

  constexpr auto parse(format_parse_context& ctx) {
      auto it = ctx.begin();
      if (it != ctx.end() && *it == 's') {
          shape_only = true;
          ++it;
      }
      return it;
  }

  template <typename FormatContext>
  auto format(const torch::Tensor& t, FormatContext& ctx) const {
      std::ostringstream oss;
      if (shape_only) {
          oss << "Tensor(shape=" << t.sizes() << ", dtype=" << t.dtype() << ")";
      } else {
          oss << t;
      }
      return fmt::format_to(ctx.out(), "{}", oss.str());
  }
};

namespace robot::geometry {

// Data structure holding sorted particle information
struct SortedParticleData {
    torch::Tensor sorted_particles;        // (N, 2) sorted by cell
    torch::Tensor sorted_particle_indices; // Original indices
    torch::Tensor sorted_cell_ids;         // Cell ID for each sorted particle
    bool empty;
};

// Data structure holding block assignment information
struct BlockAssignmentData {
    torch::Tensor particle_starts;
    torch::Tensor particle_ends;
    torch::Tensor block_cell_ids;
    torch::Tensor num_geoms_per_block;
    torch::Tensor output_offsets;
    torch::Tensor local_to_global;
    torch::Tensor local_to_global_offsets;

    int64_t total_output_size;
    bool empty;
};

// Device function for computing winding number contribution from a single edge
// Returns +1 for upward crossing, -1 for downward crossing, 0 otherwise
__device__ inline int winding_number_edge(float px, float py, float x1, float y1, float x2,
                                          float y2) {
    // Translate to query point origin
    float v1y = y1 - py;
    float v2y = y2 - py;

    // Check edge crossing conditions
    if (v1y <= 0) {
        if (v2y > 0) {
            // Potential upward crossing - check which side of edge
            float v1x = x1 - px;
            float v2x = x2 - px;
            float cross = v1x * v2y - v1y * v2x;
            if (cross > 0) return 1;  // Upward crossing, point is left of edge
        }
    } else {
        if (v2y <= 0) {
            // Potential downward crossing - check which side of edge
            float v1x = x1 - px;
            float v2x = x2 - px;
            float cross = v1x * v2y - v1y * v2x;
            if (cross < 0) return -1;  // Downward crossing, point is right of edge
        }
    }
    return 0;
}

// Device function for point-to-segment distance
__device__ inline float point_to_segment_distance(float px, float py, float ax, float ay, float bx,
                                                  float by) {
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
// Segments are sorted by (cell_id, geom_id), so local geometry ID is computed
// on-the-fly by detecting geometry transitions.
// This kernel handles cells with >128 geometries and >512 segments via multi-pass.
// It writes particle indices, geometry indices, and distances directly.
__global__ void compute_distances_optimized_kernel(
    // Sorted particle data
    const float* __restrict__ sorted_particles,          // (N, 2) sorted by cell
    const int64_t* __restrict__ sorted_particle_indices, // (N,) global particle IDs

    // Block assignments (indexed by blockIdx.x)
    const int64_t* __restrict__ particle_starts,      // (num_blocks,)
    const int64_t* __restrict__ particle_ends,        // (num_blocks,)
    const int64_t* __restrict__ cell_ids,             // (num_blocks,)
    const int64_t* __restrict__ num_geoms_per_block,  // (num_blocks,)

    // Geometry data (global)
    const float* __restrict__ segment_starts,     // (M_seg, 2)
    const float* __restrict__ segment_ends,       // (M_seg, 2)
    const int64_t* __restrict__ segment_to_geom,  // (M_seg,)

    // Spatial index (segments sorted by cell_id, then geom_id)
    const int64_t* __restrict__ cell_segment_indices, const int64_t* __restrict__ cell_offsets,

    // Local to global geometry mapping
    const int64_t* __restrict__ local_to_global,         // Flattened local→global geom IDs
    const int64_t* __restrict__ local_to_global_offsets, // (num_blocks+1,) CSR offsets

    // Output arrays (dense per block)
    int64_t* __restrict__ output_particle_indices, // (total_output_size,)
    int64_t* __restrict__ output_geometry_indices, // (total_output_size,)
    float* __restrict__ output_distances,          // (total_output_size,)
    const int64_t* __restrict__ output_offsets     // (num_blocks,) offset for each block
) {
    int block_id = blockIdx.x;

    // Get this block's assignment
    int64_t particle_start = particle_starts[block_id];
    int64_t particle_end = particle_ends[block_id];
    int64_t cell_id = cell_ids[block_id];
    int64_t num_geometries = num_geoms_per_block[block_id];
    int64_t output_offset = output_offsets[block_id];
    int64_t geom_base_offset = local_to_global_offsets[block_id];

    int64_t num_particles = particle_end - particle_start;

    // Get segments in this cell
    int64_t cell_seg_start = cell_offsets[cell_id];
    int64_t cell_seg_end = cell_offsets[cell_id + 1];
    int64_t total_segments = cell_seg_end - cell_seg_start;

    // Shared memory for segments (up to 512 segments per chunk)
    __shared__ float seg_starts_shared[512][2];
    __shared__ float seg_ends_shared[512][2];
    __shared__ int seg_local_geom_ids[512];

    // Shared variable for segment range finding
    __shared__ int64_t seg_range_start;
    __shared__ int64_t seg_range_end;

    // Process geometries in chunks of 128
    int num_geom_passes = (num_geometries + 127) / 128;

    for (int geom_pass = 0; geom_pass < num_geom_passes; ++geom_pass) {
        int64_t local_geom_start = geom_pass * 128;
        int64_t local_geom_end = min(local_geom_start + 128, num_geometries);
        int64_t num_geoms_this_pass = local_geom_end - local_geom_start;

        // Find segment range for this geometry chunk
        // Segments are sorted by global geom ID within cell
        if (threadIdx.x == 0) {
            int64_t first_global_geom = local_to_global[geom_base_offset + local_geom_start];
            int64_t last_global_geom = local_to_global[geom_base_offset + local_geom_end - 1];

            // Binary search for first segment with geom_id >= first_global_geom
            int64_t lo = 0, hi = total_segments;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                int64_t seg_idx = cell_segment_indices[cell_seg_start + mid];
                if (segment_to_geom[seg_idx] < first_global_geom) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            seg_range_start = lo;

            // Binary search for first segment with geom_id > last_global_geom
            lo = seg_range_start;
            hi = total_segments;
            while (lo < hi) {
                int64_t mid = (lo + hi) / 2;
                int64_t seg_idx = cell_segment_indices[cell_seg_start + mid];
                if (segment_to_geom[seg_idx] <= last_global_geom) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            seg_range_end = lo;
        }
        __syncthreads();

        int64_t pass_seg_start = seg_range_start;
        int64_t pass_seg_end = seg_range_end;
        int64_t num_segments_this_pass = pass_seg_end - pass_seg_start;

        // Process segments in chunks of 512
        int num_seg_chunks = (num_segments_this_pass + 511) / 512;

        // Each thread processes particles
        for (int64_t p_offset = 0; p_offset < num_particles; p_offset += blockDim.x) {
            int64_t particle_idx = particle_start + threadIdx.x + p_offset;
            bool valid_particle = (particle_idx < particle_end);

            float px = 0, py = 0;
            if (valid_particle) {
                px = sorted_particles[particle_idx * 2];
                py = sorted_particles[particle_idx * 2 + 1];
            }

            // Track min distance per local geometry ID for this pass
            float min_dists[128];
            for (int64_t i = 0; i < num_geoms_this_pass; ++i) {
                min_dists[i] = INFINITY;
            }

            // Process segment chunks
            for (int seg_chunk = 0; seg_chunk < num_seg_chunks; ++seg_chunk) {
                int64_t chunk_start = pass_seg_start + seg_chunk * 512;
                int64_t chunk_end = min(chunk_start + 512, pass_seg_end);
                int64_t chunk_size = chunk_end - chunk_start;

                // Cooperatively load segments into shared memory
                __syncthreads();
                for (int64_t i = threadIdx.x; i < chunk_size; i += blockDim.x) {
                    int64_t global_seg_idx = cell_segment_indices[cell_seg_start + chunk_start + i];
                    seg_starts_shared[i][0] = segment_starts[global_seg_idx * 2];
                    seg_starts_shared[i][1] = segment_starts[global_seg_idx * 2 + 1];
                    seg_ends_shared[i][0] = segment_ends[global_seg_idx * 2];
                    seg_ends_shared[i][1] = segment_ends[global_seg_idx * 2 + 1];
                    // Store global geom ID temporarily
                    seg_local_geom_ids[i] = static_cast<int>(segment_to_geom[global_seg_idx]);
                }
                __syncthreads();

                // Compute local geometry IDs relative to this pass
                // Parallelize: each thread handles some segments
                for (int64_t i = threadIdx.x; i < chunk_size; i += blockDim.x) {
                    int64_t global_geom_id = seg_local_geom_ids[i];
                    // Binary search in local_to_global to find local geom ID
                    int64_t lo = local_geom_start;
                    int64_t hi = local_geom_end;
                    int local_id = -1;
                    while (lo < hi) {
                        int64_t mid = (lo + hi) / 2;
                        int64_t mid_geom = local_to_global[geom_base_offset + mid];
                        if (mid_geom < global_geom_id) {
                            lo = mid + 1;
                        } else if (mid_geom > global_geom_id) {
                            hi = mid;
                        } else {
                            local_id = mid - local_geom_start;
                            break;
                        }
                    }
                    seg_local_geom_ids[i] = local_id;
                }
                __syncthreads();

                // Compute distances for valid particles
                if (valid_particle) {
                    for (int64_t i = 0; i < chunk_size; ++i) {
                        int local_geom_id = seg_local_geom_ids[i];
                        if (local_geom_id >= 0 && local_geom_id < 128) {
                            float dist = point_to_segment_distance(
                                px, py,
                                seg_starts_shared[i][0], seg_starts_shared[i][1],
                                seg_ends_shared[i][0], seg_ends_shared[i][1]);
                            min_dists[local_geom_id] = fminf(min_dists[local_geom_id], dist);
                        }
                    }
                }
            }

            // Write output for this geometry pass
            if (valid_particle) {
                int64_t local_particle_idx = particle_idx - particle_start;
                int64_t output_base = output_offset + local_particle_idx * num_geometries + local_geom_start;
                int64_t global_particle_idx = sorted_particle_indices[particle_idx];

                for (int64_t i = 0; i < num_geoms_this_pass; ++i) {
                    int64_t output_idx = output_base + i;
                    output_particle_indices[output_idx] = global_particle_idx;
                    output_geometry_indices[output_idx] = local_to_global[geom_base_offset + local_geom_start + i];
                    output_distances[output_idx] = min_dists[i];
                }
            }
        }
    }
}

// Step 1: Hash particles to cells and sort by cell ID
SortedParticleData hash_and_sort_particles(
    const torch::Tensor& query_points,
    const GridConfig& grid) {
    RECORD_FUNCTION("cuda_kernel::hash_and_sort", std::vector<c10::IValue>{});

    SortedParticleData result;
    result.empty = false;

    auto cell_coords = ((query_points - grid.origin) / grid.cell_size).floor().to(torch::kInt64);
    auto in_bounds =
        (cell_coords.select(1, 0) >= 0) & (cell_coords.select(1, 0) < grid.dims[0]) &
        (cell_coords.select(1, 1) >= 0) & (cell_coords.select(1, 1) < grid.dims[1]);

    auto cell_ids = cell_coords.select(1, 1) * grid.dims[0] + cell_coords.select(1, 0);

    auto valid_indices = torch::where(in_bounds)[0];
    if (valid_indices.numel() == 0) {
        result.empty = true;
        return result;
    }

    auto valid_cell_ids = cell_ids.index_select(0, valid_indices);
    auto sorted_result = valid_cell_ids.sort();
    result.sorted_cell_ids = std::get<0>(sorted_result);
    auto sort_indices = std::get<1>(sorted_result);

    result.sorted_particle_indices = valid_indices.index_select(0, sort_indices);
    result.sorted_particles = query_points.index_select(0, result.sorted_particle_indices);

    return result;
}

// Step 2: Build block assignments from sorted particles
// Uses GPU tensor operations instead of CPU loop
BlockAssignmentData build_block_assignments(
    const SortedParticleData& sorted_data,
    const torch::Tensor& cell_offsets,
    const torch::Tensor& cell_geom_indices,
    const torch::Tensor& cell_geom_offsets) {
    RECORD_FUNCTION("cuda_kernel::build_block_assignments", std::vector<c10::IValue>{});
    BlockAssignmentData result;
    result.empty = false;
    result.total_output_size = 0;

    // Step 1: Get unique cells and particle counts (on GPU)
    auto unique_result = torch::unique_consecutive(
            sorted_data.sorted_cell_ids, false, true, torch::nullopt);
    auto unique_cells = std::get<0>(unique_result);  // (num_unique_cells,)
    auto counts = std::get<2>(unique_result);         // (num_unique_cells,)

    if (unique_cells.numel() == 0) {
        result.empty = true;
        return result;
    }

    // Step 2: Compute particle ranges (cumulative sum on GPU)
    auto particle_ends_all = counts.cumsum(0);
    auto particle_starts_all = particle_ends_all - counts;

    // Step 3: Look up segment counts per cell to filter empty cells
    auto seg_starts = cell_offsets.index_select(0, unique_cells);
    auto seg_ends = cell_offsets.index_select(0, unique_cells + 1);
    auto has_segments = seg_ends > seg_starts;

    // Step 4: Filter to non-empty cells
    auto valid_indices = torch::where(has_segments)[0];
    if (valid_indices.numel() == 0) {
        result.empty = true;
        return result;
    }

    auto block_cell_ids = unique_cells.index_select(0, valid_indices);
    auto particle_starts = particle_starts_all.index_select(0, valid_indices);
    auto particle_ends = particle_ends_all.index_select(0, valid_indices);
    auto particle_counts = counts.index_select(0, valid_indices);

    // Step 5: Look up geometry counts from precomputed CSR
    auto geom_starts = cell_geom_offsets.index_select(0, block_cell_ids);
    auto geom_ends = cell_geom_offsets.index_select(0, block_cell_ids + 1);
    auto num_geoms_per_block = geom_ends - geom_starts;

    // Note: kernel now handles >128 geometries via multi-pass, no clamping needed

    // Step 6: Compute output offsets
    auto output_sizes = particle_counts * num_geoms_per_block;
    auto output_offsets = output_sizes.cumsum(0) - output_sizes;
    result.total_output_size = output_sizes.sum().item<int64_t>();

    // Step 7: Build local_to_global mapping using GPU operations
    // Compute CSR offsets for local_to_global
    auto local_to_global_offsets = torch::cat({
        torch::zeros({1}, num_geoms_per_block.options()),
        num_geoms_per_block.cumsum(0)
    });

    // Expand block indices by num_geoms_per_block to gather geometry IDs
    int64_t num_blocks = block_cell_ids.numel();
    auto block_indices = torch::arange(num_blocks, block_cell_ids.options());
    auto block_indices_expanded = block_indices.repeat_interleave(num_geoms_per_block);

    // For each expanded index, compute the local geom offset within the block
    int64_t total_geoms = block_indices_expanded.size(0);
    auto flat_indices = torch::arange(total_geoms, block_cell_ids.options());
    auto offsets_expanded = local_to_global_offsets.index_select(0, block_indices_expanded);
    auto local_geom_indices = flat_indices - offsets_expanded;

    // Compute global indices into cell_geom_indices
    auto geom_starts_expanded = geom_starts.index_select(0, block_indices_expanded);
    auto global_geom_indices = geom_starts_expanded + local_geom_indices;

    // Gather the actual geometry IDs
    auto local_to_global = cell_geom_indices.index_select(0, global_geom_indices);

    // Store GPU tensors
    result.particle_starts = particle_starts;
    result.particle_ends = particle_ends;
    result.block_cell_ids = block_cell_ids;
    result.num_geoms_per_block = num_geoms_per_block;
    result.output_offsets = output_offsets;
    result.local_to_global = local_to_global;
    result.local_to_global_offsets = local_to_global_offsets;

    return result;
}

// Step 3: Launch the CUDA kernel and return three output tensors
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> launch_distance_kernel(
    const SortedParticleData& sorted_data,
    const BlockAssignmentData& block_data,
    const SegmentGeometry& segments,
    const torch::Tensor& cell_segment_indices,
    const torch::Tensor& cell_offsets) {
    RECORD_FUNCTION("cuda_kernel::compute_distances_kernel", std::vector<c10::IValue>{});

    auto device = sorted_data.sorted_particles.device();

    // Allocate three output tensors
    auto output_particle_indices =
        torch::empty({block_data.total_output_size},
                    torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto output_geometry_indices =
        torch::empty({block_data.total_output_size},
                    torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto output_distances =
        torch::full({block_data.total_output_size}, INFINITY,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device));

    int num_blocks_cuda = block_data.particle_starts.size(0);
    const int threads = 256;

    compute_distances_optimized_kernel<<<num_blocks_cuda, threads>>>(
        sorted_data.sorted_particles.data_ptr<float>(),
        sorted_data.sorted_particle_indices.data_ptr<int64_t>(),
        block_data.particle_starts.data_ptr<int64_t>(),
        block_data.particle_ends.data_ptr<int64_t>(),
        block_data.block_cell_ids.data_ptr<int64_t>(),
        block_data.num_geoms_per_block.data_ptr<int64_t>(),
        segments.starts.data_ptr<float>(),
        segments.ends.data_ptr<float>(),
        segments.to_geom.data_ptr<int64_t>(),
        cell_segment_indices.data_ptr<int64_t>(),
        cell_offsets.data_ptr<int64_t>(),
        block_data.local_to_global.data_ptr<int64_t>(),
        block_data.local_to_global_offsets.data_ptr<int64_t>(),
        output_particle_indices.data_ptr<int64_t>(),
        output_geometry_indices.data_ptr<int64_t>(),
        output_distances.data_ptr<float>(),
        block_data.output_offsets.data_ptr<int64_t>());

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(output_particle_indices, output_geometry_indices, output_distances);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_segment_distances(
        const SortedParticleData& sorted_data,
        const SegmentGeometry& segments,
        const SegmentSpatialIndex& seg_idx) {
    // Step 2: Build block assignments using precomputed geometry info
    auto block_data = build_block_assignments(
        sorted_data, seg_idx.cell_offsets, seg_idx.geom_indices, seg_idx.geom_offsets);
    if (block_data.empty) {
        auto opts = sorted_data.sorted_particles.options();
        return std::make_tuple(
            torch::empty({0}, opts.dtype(torch::kInt64)),
            torch::empty({0}, opts.dtype(torch::kInt64)),
            torch::empty({0}, opts.dtype(torch::kFloat32)));
    }

    // Step 3: Launch CUDA kernel (returns particle indices, geometry indices, distances)
    return launch_distance_kernel(
        sorted_data, block_data, segments, seg_idx.segment_indices, seg_idx.cell_offsets);
}

// Build candidate (point, polygon) pairs from sorted particle data
// Returns (candidate_point_idx, candidate_poly_idx)
std::tuple<torch::Tensor, torch::Tensor> build_polygon_candidates(
    const SortedParticleData& sorted_data,
    const PolygonSpatialIndex& poly_idx) {
    RECORD_FUNCTION("cuda_kernel::build_polygon_candidates", std::vector<c10::IValue>{});

    auto device = sorted_data.sorted_particles.device();

    if (sorted_data.empty || poly_idx.geom_indices.numel() == 0) {
        return std::make_tuple(
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device)),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }

    // Get polygon candidates per particle using CSR lookup
    auto poly_starts = poly_idx.cell_offsets.index_select(0, sorted_data.sorted_cell_ids);
    auto poly_ends = poly_idx.cell_offsets.index_select(0, sorted_data.sorted_cell_ids + 1);
    auto num_candidates_per_point = poly_ends - poly_starts;
    int64_t total_candidates = num_candidates_per_point.sum().item<int64_t>();

    if (total_candidates == 0) {
        return std::make_tuple(
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device)),
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device)));
    }

    // Expand particles by number of candidates
    auto candidate_point_idx = sorted_data.sorted_particle_indices.repeat_interleave(num_candidates_per_point);

    // Generate polygon indices using CSR expansion
    auto offsets = torch::cat({
        torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(device)),
        num_candidates_per_point.cumsum(0)
    });

    auto flat_indices = torch::arange(total_candidates, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto sorted_particle_expanded = torch::arange(sorted_data.sorted_particle_indices.size(0),
        torch::TensorOptions().dtype(torch::kInt64).device(device)).repeat_interleave(num_candidates_per_point);
    auto local_poly_offset = flat_indices - offsets.index_select(0, sorted_particle_expanded);

    // Look up the actual polygon geometry indices
    auto poly_starts_expanded = poly_starts.index_select(0, sorted_particle_expanded);
    auto candidate_poly_indices = poly_idx.geom_indices.index_select(0, poly_starts_expanded + local_poly_offset);

    return std::make_tuple(candidate_point_idx, candidate_poly_indices);
}

// Merge polygon results: set dist=0 for inside points, add new (point, polygon) pairs
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_polygon_results(
    const torch::Tensor& particle_indices,
    const torch::Tensor& geometry_indices,
    const torch::Tensor& distances,
    const torch::Tensor& candidate_point_idx,
    const torch::Tensor& candidate_poly_idx,
    const torch::Tensor& is_inside,
    const torch::Tensor& polygon_geom_indices,
    int64_t num_geometries,
    torch::Device device) {
    RECORD_FUNCTION("cuda_kernel::merge_polygon_results", std::vector<c10::IValue>{});

    auto opts_long = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Get unique polygon geometry indices
    auto unique_poly_geoms = std::get<0>(torch::unique_consecutive(polygon_geom_indices));

    // Create mask for polygon geometries in existing results
    auto is_polygon_geom = torch::zeros({num_geometries}, torch::TensorOptions().dtype(torch::kBool).device(device));
    is_polygon_geom.index_fill_(0, unique_poly_geoms, true);

    // Copy distances for modification
    auto merged_distances = distances.clone();

    // Compute sparse keys for is_inside lookup
    auto sparse_keys = candidate_point_idx * num_geometries + candidate_poly_idx;

    if (particle_indices.numel() > 0) {
        auto polygon_mask = is_polygon_geom.index_select(0, geometry_indices);

        if (polygon_mask.any().item<bool>()) {
            auto poly_particle_indices = particle_indices.index({polygon_mask});
            auto poly_geom_indices = geometry_indices.index({polygon_mask});
            auto existing_keys = poly_particle_indices * num_geometries + poly_geom_indices;

            if (existing_keys.numel() > 0) {
                // Sort sparse keys for binary search
                auto sort_result = sparse_keys.sort();
                auto sparse_keys_sorted = std::get<0>(sort_result);
                auto sort_idx = std::get<1>(sort_result);
                auto is_inside_sorted = is_inside.index_select(0, sort_idx);

                // Find where existing_keys would be inserted in sorted sparse_keys
                auto match_idx = torch::searchsorted(sparse_keys_sorted, existing_keys);

                // Check if matches are valid
                auto max_idx = sparse_keys_sorted.size(0) - 1;
                auto match_idx_clamped = match_idx.clamp(0, max_idx);
                auto valid_match = (match_idx < sparse_keys_sorted.size(0)) &
                                   (sparse_keys_sorted.index_select(0, match_idx_clamped) == existing_keys);

                // Get is_inside for valid matches
                auto inside_for_existing = torch::zeros({existing_keys.size(0)},
                    torch::TensorOptions().dtype(torch::kBool).device(device));
                auto valid_indices = torch::where(valid_match)[0];
                if (valid_indices.numel() > 0) {
                    auto valid_match_idx = match_idx.index_select(0, valid_indices);
                    inside_for_existing.index_put_({valid_indices}, is_inside_sorted.index_select(0, valid_match_idx));
                }

                // Set distance to 0 for inside points
                auto inside_full_mask = torch::zeros({distances.size(0)},
                    torch::TensorOptions().dtype(torch::kBool).device(device));
                inside_full_mask.index_put_({polygon_mask}, inside_for_existing);
                merged_distances.index_put_({inside_full_mask}, 0.0f);
            }
        }
    }

    // Add entries for points inside polygons but not near edges
    auto inside_indices = torch::where(is_inside)[0];
    if (inside_indices.numel() > 0) {
        auto inside_point_idx = candidate_point_idx.index_select(0, inside_indices);
        auto inside_poly_geom_idx = candidate_poly_idx.index_select(0, inside_indices);

        // Build existing keys for deduplication
        torch::Tensor existing_keys;
        if (particle_indices.numel() > 0) {
            auto polygon_mask = is_polygon_geom.index_select(0, geometry_indices);
            existing_keys = particle_indices.index({polygon_mask}) * num_geometries +
                           geometry_indices.index({polygon_mask});
        } else {
            existing_keys = torch::empty({0}, opts_long);
        }

        // Compute new keys
        auto new_keys = inside_point_idx * num_geometries + inside_poly_geom_idx;

        // Find keys that don't exist in results yet
        auto new_mask = ~torch::isin(new_keys, existing_keys);
        auto new_indices = torch::where(new_mask)[0];

        if (new_indices.numel() > 0) {
            auto new_p = inside_point_idx.index_select(0, new_indices);
            auto new_g = inside_poly_geom_idx.index_select(0, new_indices);
            auto new_d = torch::zeros({new_p.size(0)}, opts_float);

            return std::make_tuple(
                torch::cat({particle_indices, new_p}),
                torch::cat({geometry_indices, new_g}),
                torch::cat({merged_distances, new_d}));
        }
    }

    return std::make_tuple(particle_indices, geometry_indices, merged_distances);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> query_distances_cuda(
    const torch::Tensor& query_points,
    int64_t num_geometries,
    const SegmentGeometry& segments,
    const PointGeometry& points,
    const PolygonRingData& poly_rings,
    const GridConfig& grid,
    const SegmentSpatialIndex& seg_idx,
    const PointSpatialIndex& pt_idx,
    const PolygonSpatialIndex& poly_idx) {
    // Check inputs
    TORCH_CHECK(query_points.is_cuda(), "query_points must be on CUDA");
    TORCH_CHECK(query_points.dim() == 2 && query_points.size(1) == 2,
                "query_points must be (N, 2)");
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32, "query_points must be float32");

    int64_t num_queries = query_points.size(0);
    if (num_queries == 0) {
        auto opts = query_points.options();
        return std::make_tuple(
            torch::empty({0}, opts.dtype(torch::kInt64)),
            torch::empty({0}, opts.dtype(torch::kInt64)),
            torch::empty({0}, opts.dtype(torch::kFloat32)));
    }

    auto device = query_points.device();
    c10::cuda::CUDAGuard device_guard(device);

    // Step 1: Hash particles to cells and sort
    auto sorted_data = hash_and_sort_particles(query_points, grid);
    if (sorted_data.empty) {
        auto opts = query_points.options();
        return std::make_tuple(
            torch::empty({0}, opts.dtype(torch::kInt64)),
            torch::empty({0}, opts.dtype(torch::kInt64)),
            torch::empty({0}, opts.dtype(torch::kFloat32)));
    }

    // Compute the distance to segments
    auto [seg_particles, seg_geoms, seg_dists] = query_segment_distances(
            sorted_data, segments, seg_idx);

    // Compute the distance to points (reuse segment distance code with points as degenerate segments)
    SegmentGeometry point_as_segments{points.coords, points.coords, points.to_geom};
    SegmentSpatialIndex pt_as_seg_idx{pt_idx.point_indices, pt_idx.cell_offsets,
                                       pt_idx.geom_indices, pt_idx.geom_offsets};
    auto [pt_particles, pt_geoms, pt_dists] = query_segment_distances(
            sorted_data, point_as_segments, pt_as_seg_idx);

    // Concatenate segment and point results
    auto particle_indices = torch::cat({seg_particles, pt_particles}, 0);
    auto geometry_indices = torch::cat({seg_geoms, pt_geoms}, 0);
    auto distances = torch::cat({seg_dists, pt_dists}, 0);

    // Step 4: Handle polygon point-in-polygon checks
    int64_t num_polygon_geoms = poly_rings.geom_ring_offsets.size(0) - 1;
    if (num_polygon_geoms > 0 && poly_idx.geom_indices.numel() > 0) {
        // Build polygon candidates using the sorted particle data (reuses cell hashing!)
        auto [candidate_point_idx, candidate_poly_idx] = build_polygon_candidates(
            sorted_data, poly_idx);

        if (candidate_point_idx.numel() > 0) {
            // Get unique polygon geometry indices and create mapping
            auto unique_poly_geoms = std::get<0>(torch::_unique(poly_rings.geom_indices, true));
            auto geom_to_poly_idx = torch::full({num_geometries}, -1,
                torch::TensorOptions().dtype(torch::kInt64).device(device));
            geom_to_poly_idx.index_put_(
                {unique_poly_geoms},
                torch::arange(unique_poly_geoms.size(0),
                    torch::TensorOptions().dtype(torch::kInt64).device(device)));

            // Convert global geometry indices to local polygon indices
            auto candidate_local_poly_idx = geom_to_poly_idx.index_select(0, candidate_poly_idx);

            // Call sparse point-in-polygon kernel
            auto is_inside = point_in_polygon_sparse_cuda(
                query_points,
                candidate_point_idx,
                candidate_local_poly_idx,
                segments.starts,
                poly_rings.segment_ranges,
                poly_rings.geom_ring_offsets);

            // Merge results: set dist=0 for inside, add new pairs
            std::tie(particle_indices, geometry_indices, distances) = merge_polygon_results(
                particle_indices, geometry_indices, distances,
                candidate_point_idx, candidate_poly_idx, is_inside,
                poly_rings.geom_indices, num_geometries, device);
        }
    }

    return std::make_tuple(particle_indices, geometry_indices, distances);
}

// Sparse kernel for point-in-polygon testing
// Takes explicit candidate (point, polygon) pairs instead of dense matrix
// Each thread processes one candidate pair
__global__ void point_in_polygon_sparse_kernel(
    const float* __restrict__ query_points,              // (N, 2)
    const int64_t* __restrict__ candidate_point_idx,     // (K,) query point indices
    const int64_t* __restrict__ candidate_poly_idx,      // (K,) polygon geometry indices
    const int64_t num_candidates,
    const float* __restrict__ segment_starts,            // (S, 2) - ring vertices
    const int64_t* __restrict__ polygon_segment_ranges,  // (R, 2) [start, end) per ring
    const int64_t* __restrict__ geom_ring_offsets,       // (G_poly+1,) CSR offsets
    bool* __restrict__ is_inside                         // (K,) output
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    int64_t point_idx = candidate_point_idx[idx];
    int64_t poly_idx = candidate_poly_idx[idx];

    float px = query_points[point_idx * 2];
    float py = query_points[point_idx * 2 + 1];

    // Get the range of rings for this polygon geometry
    int64_t ring_start = geom_ring_offsets[poly_idx];
    int64_t ring_end = geom_ring_offsets[poly_idx + 1];

    int total_winding = 0;

    // Sum winding numbers across all rings of this geometry
    for (int64_t r = ring_start; r < ring_end; r++) {
        int64_t seg_start = polygon_segment_ranges[r * 2];
        int64_t seg_end = polygon_segment_ranges[r * 2 + 1];
        int64_t num_verts = seg_end - seg_start;

        if (num_verts < 3) continue;

        // Compute winding number for this ring
        for (int64_t i = 0; i < num_verts; i++) {
            int64_t seg_idx1 = seg_start + i;
            int64_t seg_idx2 = seg_start + ((i + 1) % num_verts);

            float x1 = segment_starts[seg_idx1 * 2];
            float y1 = segment_starts[seg_idx1 * 2 + 1];
            float x2 = segment_starts[seg_idx2 * 2];
            float y2 = segment_starts[seg_idx2 * 2 + 1];

            total_winding += winding_number_edge(px, py, x1, y1, x2, y2);
        }
    }

    // Point is inside if total winding number is non-zero
    is_inside[idx] = (total_winding != 0);
}

torch::Tensor point_in_polygon_sparse_cuda(
    const torch::Tensor& query_points,
    const torch::Tensor& candidate_point_idx,
    const torch::Tensor& candidate_poly_idx,
    const torch::Tensor& segment_starts,
    const torch::Tensor& polygon_segment_ranges,
    const torch::Tensor& geom_ring_offsets
) {
    // Check inputs
    TORCH_CHECK(query_points.is_cuda(), "query_points must be on CUDA");
    TORCH_CHECK(query_points.dim() == 2 && query_points.size(1) == 2,
                "query_points must be (N, 2)");
    TORCH_CHECK(query_points.scalar_type() == torch::kFloat32, "query_points must be float32");
    TORCH_CHECK(candidate_point_idx.dim() == 1, "candidate_point_idx must be 1D");
    TORCH_CHECK(candidate_poly_idx.dim() == 1, "candidate_poly_idx must be 1D");
    TORCH_CHECK(candidate_point_idx.size(0) == candidate_poly_idx.size(0),
                "candidate_point_idx and candidate_poly_idx must have same length");

    int64_t num_candidates = candidate_point_idx.size(0);

    if (num_candidates == 0) {
        return torch::zeros({0},
                           torch::TensorOptions().dtype(torch::kBool).device(query_points.device()));
    }

    auto device = query_points.device();
    c10::cuda::CUDAGuard device_guard(device);

    // Allocate output
    auto is_inside = torch::zeros({num_candidates},
                                  torch::TensorOptions().dtype(torch::kBool).device(device));

    // Launch kernel: one thread per candidate pair
    const int threads = 256;
    const int blocks = (num_candidates + threads - 1) / threads;

    point_in_polygon_sparse_kernel<<<blocks, threads>>>(
        query_points.data_ptr<float>(),
        candidate_point_idx.data_ptr<int64_t>(),
        candidate_poly_idx.data_ptr<int64_t>(),
        num_candidates,
        segment_starts.data_ptr<float>(),
        polygon_segment_ranges.data_ptr<int64_t>(),
        geom_ring_offsets.data_ptr<int64_t>(),
        is_inside.data_ptr<bool>()
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return is_inside;
}

}  // namespace robot::geometry
