import common.torch.load_torch_deps
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass

from common.gps import web_mercator


@dataclass
class CellToPatchMapping:
    """Mapping from histogram cells to overlapping satellite patches.

    Stored in CSR (Compressed Sparse Row) format for efficient access.
    Cell i overlaps patches: patch_indices[cell_offsets[i]:cell_offsets[i+1]]
    """

    patch_indices: torch.Tensor  # (total_overlaps,) - flat list of patch indices
    cell_offsets: torch.Tensor  # (num_cells + 1,) - start index for each cell
    segment_ids: torch.Tensor  # (total_overlaps,) - cell index for each overlap


@dataclass
class GridSpec:
    """Defines the discretization grid in Web Mercator pixel coordinates."""

    zoom_level: int
    cell_size_px: float
    origin_row_px: float  # Top-left corner row (min y in pixels)
    origin_col_px: float  # Top-left corner col (min x in pixels)
    num_rows: int
    num_cols: int

    @classmethod
    def from_bounds_and_cell_size(
        cls,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        zoom_level: int,
        cell_size_px: float,
    ) -> "GridSpec":
        """Create GridSpec from lat/lon bounds.

        Args:
            min_lat, max_lat: Latitude bounds in degrees
            min_lon, max_lon: Longitude bounds in degrees
            zoom_level: Web Mercator zoom level (typically 20)
            cell_size_px: Size of each grid cell in pixels
        """
        # Convert bounds to pixel coordinates
        # Note: In Web Mercator, y increases downward, so max_lat -> min_row
        min_row_px, min_col_px = web_mercator.latlon_to_pixel_coords(
            max_lat, min_lon, zoom_level
        )
        max_row_px, max_col_px = web_mercator.latlon_to_pixel_coords(
            min_lat, max_lon, zoom_level
        )

        # Compute grid dimensions
        num_rows = int(math.ceil((max_row_px - min_row_px) / cell_size_px))
        num_cols = int(math.ceil((max_col_px - min_col_px) / cell_size_px))

        return cls(
            zoom_level=zoom_level,
            cell_size_px=cell_size_px,
            origin_row_px=min_row_px,
            origin_col_px=min_col_px,
            num_rows=num_rows,
            num_cols=num_cols,
        )

    def latlon_to_cell_indices(
        self, lat: torch.Tensor, lon: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert lat/lon to (row_idx, col_idx) cell indices.

        Args:
            lat: Latitude tensor in degrees
            lon: Longitude tensor in degrees

        Returns:
            (row_idx, col_idx): Integer cell indices
        """
        y_px, x_px = web_mercator.latlon_to_pixel_coords(lat, lon, self.zoom_level)
        row_idx = ((y_px - self.origin_row_px) / self.cell_size_px).long()
        col_idx = ((x_px - self.origin_col_px) / self.cell_size_px).long()
        return row_idx, col_idx

    def latlon_to_cell_indices_float(
        self, lat: torch.Tensor, lon: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert lat/lon to fractional cell indices (for subpixel operations).

        Args:
            lat: Latitude tensor in degrees
            lon: Longitude tensor in degrees

        Returns:
            (row_idx, col_idx): Floating point cell indices
        """
        y_px, x_px = web_mercator.latlon_to_pixel_coords(lat, lon, self.zoom_level)
        row_idx = (y_px - self.origin_row_px) / self.cell_size_px
        col_idx = (x_px - self.origin_col_px) / self.cell_size_px
        return row_idx, col_idx

    def cell_indices_to_latlon(
        self, row_idx: torch.Tensor, col_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert cell indices to lat/lon (cell center).

        Args:
            row_idx: Row indices (can be float for interpolation)
            col_idx: Column indices (can be float for interpolation)

        Returns:
            (lat, lon): Latitude and longitude in degrees
        """
        y_px = self.origin_row_px + (row_idx + 0.5) * self.cell_size_px
        x_px = self.origin_col_px + (col_idx + 0.5) * self.cell_size_px
        return web_mercator.pixel_coords_to_latlon(y_px, x_px, self.zoom_level)

    def get_all_cell_centers_latlon(self, device: torch.device) -> torch.Tensor:
        """Return all cell centers as (num_rows * num_cols, 2) tensor of [lat, lon].

        Args:
            device: Torch device for the output tensor

        Returns:
            Tensor of shape (num_rows * num_cols, 2) with [lat, lon] for each cell
        """
        row_indices = torch.arange(self.num_rows, device=device, dtype=torch.float32)
        col_indices = torch.arange(self.num_cols, device=device, dtype=torch.float32)
        grid_rows, grid_cols = torch.meshgrid(row_indices, col_indices, indexing="ij")

        lat, lon = self.cell_indices_to_latlon(grid_rows.flatten(), grid_cols.flatten())
        return torch.stack([lat, lon], dim=-1)

    def get_all_cell_centers_px(self, device: torch.device) -> torch.Tensor:
        """Return all cell centers as (num_rows * num_cols, 2) tensor of [row_px, col_px].

        Args:
            device: Torch device for the output tensor

        Returns:
            Tensor of shape (num_rows * num_cols, 2) with [row_px, col_px] for each cell
        """
        row_indices = torch.arange(self.num_rows, device=device, dtype=torch.float32)
        col_indices = torch.arange(self.num_cols, device=device, dtype=torch.float32)
        grid_rows, grid_cols = torch.meshgrid(row_indices, col_indices, indexing="ij")

        # Cell center in pixels
        row_px = self.origin_row_px + (grid_rows.flatten() + 0.5) * self.cell_size_px
        col_px = self.origin_col_px + (grid_cols.flatten() + 0.5) * self.cell_size_px
        return torch.stack([row_px, col_px], dim=-1)


def _make_gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 1D Gaussian kernel.

    Args:
        sigma: Standard deviation in cells
        device: Torch device

    Returns:
        Normalized 1D Gaussian kernel
    """
    if sigma < 0.1:
        # For very small sigma, return identity (no blur)
        return torch.tensor([1.0], device=device)

    # Kernel size covers ±3 sigma, ensure odd
    kernel_size = int(6 * sigma) | 1
    kernel_size = max(kernel_size, 3)

    x = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()  # Normalize
    return kernel


def _shift_grid(
    log_belief: torch.Tensor, shift_rows: float, shift_cols: float
) -> torch.Tensor:
    """Shift belief grid by fractional cell amounts using bilinear interpolation.

    Args:
        log_belief: Log probability tensor of shape (H, W)
        shift_rows: Shift amount in rows (positive = shift down)
        shift_cols: Shift amount in columns (positive = shift right)

    Returns:
        Shifted log belief tensor
    """
    H, W = log_belief.shape
    device = log_belief.device
    dtype = log_belief.dtype

    # Convert to probability space for interpolation
    belief_prob = torch.exp(log_belief)

    # Create sampling grid
    # grid_sample expects coordinates in [-1, 1] range
    y = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    # Apply shift (convert cell shift to [-1,1] range)
    # To shift content by +N cells, we sample from position -N
    grid_y = grid_y - (2 * shift_rows / (H - 1))
    grid_x = grid_x - (2 * shift_cols / (W - 1))

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    belief_prob = belief_prob.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Sample with bilinear interpolation, zeros outside boundary
    shifted = F.grid_sample(
        belief_prob, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return torch.log(shifted.squeeze() + 1e-40)


def segment_logsumexp(
    values: torch.Tensor, offsets: torch.Tensor, segment_ids: torch.Tensor
) -> torch.Tensor:
    """Compute logsumexp over variable-length segments (vectorized).

    Args:
        values: (total_elements,) flat tensor of values
        offsets: (num_segments + 1,) start indices for each segment
        segment_ids: (total_elements,) segment index for each value

    Returns:
        (num_segments,) logsumexp for each segment
    """
    num_segments = len(offsets) - 1
    device = values.device

    if len(values) == 0:
        return torch.full((num_segments,), -float("inf"), device=device)

    # Step 1: Find max per segment (for numerical stability)
    max_vals = torch.full((num_segments,), -float("inf"), device=device)
    max_vals.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)

    # Step 2: Compute sum of exp(values - max) per segment
    shifted = torch.exp(values - max_vals[segment_ids])
    sum_exp = torch.zeros(num_segments, device=device)
    sum_exp.scatter_add_(0, segment_ids, shifted)

    # Step 3: logsumexp = max + log(sum_exp)
    # Handle empty segments (sum_exp = 0 -> -inf)
    result = max_vals + torch.log(sum_exp + 1e-40)
    result = torch.where(sum_exp > 0, result, torch.full_like(result, -float("inf")))

    return result


def segment_max(
    values: torch.Tensor, offsets: torch.Tensor, segment_ids: torch.Tensor
) -> torch.Tensor:
    """Compute max over variable-length segments (vectorized).

    Args:
        values: (total_elements,) flat tensor of values
        offsets: (num_segments + 1,) start indices for each segment
        segment_ids: (total_elements,) segment index for each value

    Returns:
        (num_segments,) max value for each segment
    """
    num_segments = len(offsets) - 1
    device = values.device

    if len(values) == 0:
        return torch.full((num_segments,), -float("inf"), device=device)

    max_vals = torch.full((num_segments,), -float("inf"), device=device)
    max_vals.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)

    return max_vals


def build_cell_to_patch_mapping(
    grid_spec: GridSpec,
    patch_positions_px: torch.Tensor,
    patch_half_size_px: float,
    device: torch.device,
    chunk_size: int = 4096,
) -> CellToPatchMapping:
    """Build mapping from histogram cells to overlapping satellite patches.

    Processes cells in chunks to avoid materializing a full
    (num_cells, num_patches, 2) tensor on GPU.

    Args:
        grid_spec: The histogram grid specification
        patch_positions_px: (num_patches, 2) patch centers in pixels [row, col]
        patch_half_size_px: Half the patch size (e.g., 320 for 640px patches)
        device: Torch device
        chunk_size: Number of cells to process at once (controls peak memory)

    Returns:
        CellToPatchMapping with CSR-format overlaps
    """
    cell_centers_px = grid_spec.get_all_cell_centers_px(device)  # (num_cells, 2)
    num_cells = cell_centers_px.shape[0]
    patch_positions_px = patch_positions_px.to(device)

    # Process cells in chunks to keep peak memory bounded.
    # Each chunk materializes (chunk_size, num_patches, 2) instead of the
    # full (num_cells, num_patches, 2).
    all_cell_idxs = []
    all_patch_idxs = []
    counts = torch.zeros(num_cells, dtype=torch.long, device=device)

    for start in range(0, num_cells, chunk_size):
        end = min(start + chunk_size, num_cells)
        chunk_cells = cell_centers_px[start:end]  # (chunk, 2)

        # L∞ distance from each cell in chunk to each patch
        diff = chunk_cells.unsqueeze(1) - patch_positions_px.unsqueeze(0)  # (chunk, num_patches, 2)
        linf_dist = diff.abs().max(dim=2).values  # (chunk, num_patches)
        del diff

        # Find overlapping (cell, patch) pairs within this chunk
        local_cell_idxs, patch_idxs = torch.where(linf_dist < patch_half_size_px)
        del linf_dist

        # Offset local cell indices to global
        global_cell_idxs = local_cell_idxs + start

        all_cell_idxs.append(global_cell_idxs)
        all_patch_idxs.append(patch_idxs)
        counts[start:end] = torch.bincount(local_cell_idxs, minlength=end - start)

    # Concatenate results from all chunks (already in cell-index order)
    cell_idxs = torch.cat(all_cell_idxs)
    patch_idxs = torch.cat(all_patch_idxs)

    # Build CSR offsets from counts
    offsets = torch.zeros(num_cells + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)

    return CellToPatchMapping(
        patch_indices=patch_idxs,
        cell_offsets=offsets,
        segment_ids=cell_idxs,
    )


class HistogramBelief:
    """Grid-based probability distribution over geographic locations."""

    def __init__(self, grid_spec: GridSpec, device: torch.device):
        """Initialize an empty belief (all -inf log probability).

        Args:
            grid_spec: Grid specification defining the discretization
            device: Torch device for tensors
        """
        self.grid_spec = grid_spec
        self.device = device
        # Store log probabilities for numerical stability
        self._log_belief = torch.full(
            (grid_spec.num_rows, grid_spec.num_cols),
            -float("inf"),
            device=device,
            dtype=torch.float32,
        )

    @classmethod
    def from_uniform(cls, grid_spec: GridSpec, device: torch.device) -> "HistogramBelief":
        """Create uniform prior over all cells.

        Args:
            grid_spec: Grid specification
            device: Torch device

        Returns:
            HistogramBelief with uniform distribution
        """
        belief = cls(grid_spec, device)
        num_cells = grid_spec.num_rows * grid_spec.num_cols
        belief._log_belief = torch.full_like(
            belief._log_belief, -math.log(num_cells)
        )
        return belief

    @classmethod
    def from_gaussian(
        cls,
        grid_spec: GridSpec,
        mean_latlon: torch.Tensor,
        std_deg: float,
        device: torch.device,
    ) -> "HistogramBelief":
        """Create Gaussian prior centered at mean_latlon.

        Args:
            grid_spec: Grid specification
            mean_latlon: (2,) tensor [lat, lon] for the center
            std_deg: Standard deviation in degrees (same for both dimensions)
            device: Torch device

        Returns:
            HistogramBelief with Gaussian distribution
        """
        belief = cls(grid_spec, device)

        # Get all cell centers
        cell_centers = grid_spec.get_all_cell_centers_latlon(device)  # (N, 2)

        # Compute squared distance from mean (in degrees)
        diff = cell_centers - mean_latlon.to(device)  # (N, 2)
        dist_sq = (diff**2).sum(dim=-1)  # (N,)

        # Gaussian log probability
        log_prob = -0.5 * dist_sq / (std_deg**2)

        # Reshape to grid
        belief._log_belief = log_prob.reshape(grid_spec.num_rows, grid_spec.num_cols)

        # Normalize
        belief.normalize()

        return belief

    def clone(self) -> "HistogramBelief":
        """Create a deep copy of this belief."""
        new_belief = HistogramBelief(self.grid_spec, self.device)
        new_belief._log_belief = self._log_belief.clone()
        return new_belief

    def normalize(self) -> None:
        """Normalize belief to sum to 1 (in probability space)."""
        log_sum = self._log_belief.logsumexp(dim=(0, 1))
        self._log_belief = self._log_belief - log_sum

    def get_log_belief(self) -> torch.Tensor:
        """Return log belief tensor (num_rows, num_cols)."""
        return self._log_belief

    def get_belief(self) -> torch.Tensor:
        """Return belief tensor in probability space."""
        return torch.exp(self._log_belief)

    def get_mean_latlon(self) -> torch.Tensor:
        """Compute weighted mean position in lat/lon.

        Returns:
            (2,) tensor [lat, lon]
        """
        belief_probs = self.get_belief()

        row_indices = torch.arange(
            self.grid_spec.num_rows, device=self.device, dtype=torch.float32
        )
        col_indices = torch.arange(
            self.grid_spec.num_cols, device=self.device, dtype=torch.float32
        )

        # Weighted mean of row/col indices
        row_marginal = belief_probs.sum(dim=1)  # (num_rows,)
        col_marginal = belief_probs.sum(dim=0)  # (num_cols,)

        mean_row = (row_marginal * row_indices).sum()
        mean_col = (col_marginal * col_indices).sum()

        # Convert to lat/lon
        lat, lon = self.grid_spec.cell_indices_to_latlon(mean_row, mean_col)
        return torch.stack([lat, lon])

    def get_variance_deg_sq(self) -> torch.Tensor:
        """Compute variance of belief in degrees squared.

        Returns:
            (2,) tensor [var_lat, var_lon] in degrees squared
        """
        belief_probs = self.get_belief()
        cell_centers = self.grid_spec.get_all_cell_centers_latlon(self.device)
        cell_centers = cell_centers.reshape(
            self.grid_spec.num_rows, self.grid_spec.num_cols, 2
        )

        mean_latlon = self.get_mean_latlon()

        # Compute weighted variance
        diff_sq = (cell_centers - mean_latlon) ** 2  # (H, W, 2)
        var = (belief_probs.unsqueeze(-1) * diff_sq).sum(dim=(0, 1))  # (2,)

        return var

    def apply_motion_blur(self, sigma_cells: float) -> None:
        """Apply isotropic Gaussian blur to the belief.

        This represents the motion model uncertainty spreading.

        Args:
            sigma_cells: Standard deviation of blur in cell units
        """
        if sigma_cells < 0.1:
            return  # No blur needed for tiny sigma

        kernel = _make_gaussian_kernel_1d(sigma_cells, self.device)

        # Convert to probability space for convolution
        belief_prob = torch.exp(self._log_belief)

        # Pad for convolution (zero padding = absorbing boundary)
        pad_size = len(kernel) // 2
        belief_prob = F.pad(
            belief_prob.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size, pad_size, pad_size),
            mode="constant",
            value=0,
        )

        # Apply separable convolution
        # First blur along rows (vertical kernel)
        kernel_v = kernel.view(1, 1, -1, 1)
        belief_prob = F.conv2d(belief_prob, kernel_v)

        # Then blur along columns (horizontal kernel)
        kernel_h = kernel.view(1, 1, 1, -1)
        belief_prob = F.conv2d(belief_prob, kernel_h)

        self._log_belief = torch.log(belief_prob.squeeze() + 1e-40)

    def apply_motion(
        self,
        motion_delta_deg: torch.Tensor,
        noise_percent: float,
        reference_latlon: torch.Tensor | None = None,
    ) -> None:
        """Apply motion model: shift by delta and blur by noise.

        Args:
            motion_delta_deg: (2,) tensor [delta_lat, delta_lon] in degrees
            noise_percent: Noise as fraction of motion magnitude
            reference_latlon: Reference point for coordinate conversion.
                If None, uses current belief mean.
        """
        if reference_latlon is None:
            reference_latlon = self.get_mean_latlon()

        # Convert motion delta to pixel shift
        y0, x0 = web_mercator.latlon_to_pixel_coords(
            reference_latlon[0], reference_latlon[1], self.grid_spec.zoom_level
        )
        y1, x1 = web_mercator.latlon_to_pixel_coords(
            reference_latlon[0] + motion_delta_deg[0],
            reference_latlon[1] + motion_delta_deg[1],
            self.grid_spec.zoom_level,
        )

        delta_row_px = y1 - y0
        delta_col_px = x1 - x0

        # Convert to cell units
        shift_rows = delta_row_px / self.grid_spec.cell_size_px
        shift_cols = delta_col_px / self.grid_spec.cell_size_px

        # Compute noise sigma in cell units
        delta_magnitude_px = torch.sqrt(delta_row_px**2 + delta_col_px**2)
        sigma_px = delta_magnitude_px * noise_percent
        sigma_cells = sigma_px / self.grid_spec.cell_size_px

        # Apply shift
        self._log_belief = _shift_grid(self._log_belief, shift_rows, shift_cols)

        # Apply blur
        if isinstance(sigma_cells, torch.Tensor):
            sigma_cells = sigma_cells.item()
        self.apply_motion_blur(sigma_cells)

        # Renormalize (some probability lost at edges)
        self.normalize()

    def apply_observation(
        self,
        observation_log_likelihoods: torch.Tensor,
        mapping: CellToPatchMapping,
    ) -> None:
        """Apply observation update using pre-computed log-likelihoods.

        For each cell, uses the maximum log-likelihood among all overlapping
        patches. This is similar to how the particle filter works (each particle
        uses its nearest/best patch) and avoids artifacts from varying overlap
        counts at patch boundaries.

        Args:
            observation_log_likelihoods: (num_patches,) log-likelihoods for each satellite patch
            mapping: Precomputed cell-to-patch mapping
        """
        # Gather log-likelihoods for all overlapping patches
        all_log_ll = observation_log_likelihoods[mapping.patch_indices]  # (total_overlaps,)

        # Take max log-likelihood over overlapping patches for each cell
        cell_log_ll = segment_max(
            all_log_ll, mapping.cell_offsets, mapping.segment_ids
        )

        # Reshape to grid and apply to belief
        cell_log_ll_grid = cell_log_ll.reshape(
            self.grid_spec.num_rows, self.grid_spec.num_cols
        )
        if torch.isnan(cell_log_ll_grid).any():
            raise ValueError(
                "NaN values in observation log-likelihoods. "
                "Check similarity matrix for NaN or missing data."
            )
        self._log_belief = self._log_belief + cell_log_ll_grid
        self.normalize()
