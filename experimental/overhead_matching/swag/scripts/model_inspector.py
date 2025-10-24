"""
Model Inspector - Capture model inputs and extractor outputs for debugging

This module provides functionality to capture model inputs and intermediate
extractor outputs during training for later inspection and verification.
It's designed to be minimally invasive and easily toggled on/off via command-line flags.
"""

import torch
import gzip
from pathlib import Path
from experimental.overhead_matching.swag.model.swag_model_input_output import ModelInput, ExtractorOutput
from typing import Optional


class ModelInspector:
    """Captures model inputs and extractor outputs for debugging purposes.

    This class provides a simple interface to save model inputs and intermediate
    extractor outputs during training. It's designed to capture only the first N
    batches to avoid excessive disk usage while still providing enough data for debugging.

    Example usage:
        inspector = ModelInspector(
            output_dir=Path("/tmp/training"),
            num_batches_to_capture=10,
            use_compression=True)

        # In training loop:
        if inspector.should_capture(total_batches):
            pano_extractor_outputs = pano_model.get_extractor_outputs(pano_input)
            sat_extractor_outputs = sat_model.get_extractor_outputs(sat_input)
            inspector.capture(
                pano_input=pano_input,
                sat_input=sat_input,
                pano_extractor_outputs=pano_extractor_outputs,
                sat_extractor_outputs=sat_extractor_outputs,
                batch_idx=batch_idx,
                epoch_idx=epoch_idx,
                total_batches=total_batches
            )
    """

    def __init__(self, output_dir: Path, num_batches_to_capture: int = 10, use_compression: bool = True):
        """Initialize the ModelInspector.

        Args:
            output_dir: Base directory for training outputs
            num_batches_to_capture: Number of batches to capture (default: 10)
            use_compression: Whether to use gzip compression (default: True)
        """
        self.output_dir = output_dir
        self.num_batches_to_capture = num_batches_to_capture
        self.use_compression = use_compression
        self.capture_dir = output_dir / "debug_captures"

        # Create the capture directory
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        compression_str = " with gzip compression" if use_compression else ""
        print(f"ModelInspector initialized: capturing first {num_batches_to_capture} batches{compression_str}")
        print(f"Captures will be saved to: {self.capture_dir}")

    def should_capture(self, total_batches: int) -> bool:
        """Check if we should capture this batch.

        Args:
            total_batches: Total number of batches processed so far

        Returns:
            True if we should capture this batch, False otherwise
        """
        return total_batches < self.num_batches_to_capture

    def capture(self,
                pano_input: ModelInput,
                sat_input: ModelInput,
                pano_extractor_outputs: dict[str, ExtractorOutput],
                sat_extractor_outputs: dict[str, ExtractorOutput],
                pairing_data,
                batch_idx: int,
                epoch_idx: int,
                total_batches: int) -> None:
        """Capture and save model inputs and extractor outputs.

        Saves all inputs and extractor outputs to a .pt file (optionally compressed).
        All data is moved to CPU before saving to avoid GPU memory issues.

        Args:
            pano_input: ModelInput for panorama model
            sat_input: ModelInput for satellite model
            pano_extractor_outputs: Dict of extractor name -> ExtractorOutput for panorama
            sat_extractor_outputs: Dict of extractor name -> ExtractorOutput for satellite
            pairing_data: Pairing data (Pairs or PositiveAnchorSets) with batch-local indices
            batch_idx: Current batch index within the epoch
            epoch_idx: Current epoch index
            total_batches: Total batches processed across all epochs
        """
        # Create filename
        suffix = ".pt.gz" if self.use_compression else ".pt"
        filename = self.capture_dir / f"batch_{total_batches:06d}{suffix}"

        # Helper to move ExtractorOutput to CPU
        def extractor_output_to_cpu(output: ExtractorOutput) -> dict:
            return {
                "features": output.features.cpu(),
                "positions": output.positions.cpu(),
                "mask": output.mask.cpu(),
                "debug": {k: v.cpu() for k, v in output.debug.items()}
            }

        # Move everything to CPU and prepare for saving
        capture_data = {
            "pano_input": {
                "image": pano_input.image.cpu(),
                "metadata": pano_input.metadata,
                "cached_tensors": {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in pano_input.cached_tensors.items()
                } if pano_input.cached_tensors else {}
            },
            "sat_input": {
                "image": sat_input.image.cpu(),
                "metadata": sat_input.metadata,
                "cached_tensors": {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in sat_input.cached_tensors.items()
                } if sat_input.cached_tensors else {}
            },
            "pano_extractor_outputs": {
                name: extractor_output_to_cpu(output)
                for name, output in pano_extractor_outputs.items()
            },
            "sat_extractor_outputs": {
                name: extractor_output_to_cpu(output)
                for name, output in sat_extractor_outputs.items()
            },
            "pairing_data": pairing_data,
            "metadata": {
                "batch_idx": batch_idx,
                "epoch_idx": epoch_idx,
                "total_batches": total_batches
            }
        }

        # Save to disk with optional compression
        try:
            if self.use_compression:
                import io
                buffer = io.BytesIO()
                torch.save(capture_data, buffer)
                buffer.seek(0)
                with gzip.open(filename, 'wb', compresslevel=6) as f:
                    f.write(buffer.getvalue())
            else:
                torch.save(capture_data, filename)

            print(f"Captured batch {total_batches} -> {filename}")
        except Exception as e:
            print(f"Error capturing batch {total_batches}: {e}")
            import traceback
            traceback.print_exc()


def load_captured_batch(filepath: Path) -> dict:
    """Load a captured batch from disk.

    Utility function to load previously captured model inputs and extractor outputs.
    Automatically handles both compressed (.pt.gz) and uncompressed (.pt) files.

    Args:
        filepath: Path to the .pt or .pt.gz file containing captured data

    Returns:
        Dictionary containing:
        - pano_input: {"image", "metadata", "cached_tensors"}
        - sat_input: {"image", "metadata", "cached_tensors"}
        - pano_extractor_outputs: {extractor_name: {"features", "positions", "mask", "debug"}}
        - sat_extractor_outputs: {extractor_name: {"features", "positions", "mask", "debug"}}
        - metadata: {"batch_idx", "epoch_idx", "total_batches"}
    """
    filepath = Path(filepath)
    if filepath.suffix == '.gz':
        import io
        with gzip.open(filepath, 'rb') as f:
            buffer = io.BytesIO(f.read())
            buffer.seek(0)
            return torch.load(buffer, weights_only=False)
    else:
        return torch.load(filepath, weights_only=False)
