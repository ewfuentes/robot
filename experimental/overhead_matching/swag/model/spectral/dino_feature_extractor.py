"""
DINO feature extraction with hooks for intermediate activations.

Supports extracting features from different layers of DINOv3 ViT models.
"""

import torch
import torch.nn as nn
from typing import Literal


FeatureSource = Literal[
    "attention_keys",
    "attention_queries",
    "attention_values",
    "attention_input",
    "model_output"
]


class DinoFeatureHookExtractor:
    """
    Extracts features from DINOv3 using forward hooks.

    Supports extracting from different parts of the last transformer block:
    - attention_keys: K projection from QKV (paper's recommendation)
    - attention_queries: Q projection from QKV
    - attention_values: V projection from QKV
    - attention_input: Input to the attention block
    - model_output: Final normalized patch tokens
    """

    def __init__(
        self,
        dino_model: nn.Module,
        feature_source: FeatureSource = "attention_keys"
    ):
        """
        Args:
            dino_model: DINOv3 ViT model
            feature_source: Which features to extract
        """
        self.dino_model = dino_model
        self.feature_source = feature_source
        self.captured_features = None
        self._hook_handle = None

        # Register appropriate hook
        if feature_source == "model_output":
            # No hook needed, just use standard forward pass
            pass
        else:
            self._register_hook()

    def _register_hook(self):
        """Register forward hook on the last transformer block."""
        last_block = self.dino_model.blocks[-1]

        if self.feature_source == "attention_input":
            # Hook the input to the attention module
            def hook_fn(module, input, output):
                # Input to attention is a tuple, take first element
                self.captured_features = input[0].detach()

            self._hook_handle = last_block.attn.register_forward_hook(hook_fn)

        elif self.feature_source in ["attention_keys", "attention_queries", "attention_values"]:
            # Hook the qkv projection to capture Q, K, V
            # Note: We hook the qkv module directly, not the attention module
            def hook_fn(module, input, output):
                x = input[0]  # May be [N, C] or [B, N, C]
                qkv = output  # May be [N, 3*C] or [B, N, 3*C]

                # Handle case where x might be None
                if x is None or qkv is None:
                    return

                # Get num_heads from the attention module
                num_heads = last_block.attn.num_heads

                # DINOv3 sometimes squeezes the batch dimension during internal processing
                # Check if we need to add it back
                if qkv.dim() == 2:
                    # Shape is [N, 3*C], add batch dimension
                    qkv = qkv.unsqueeze(0)  # [1, N, 3*C]

                B, N, qkv_dim = qkv.shape
                C = qkv_dim // 3  # Original embedding dim

                # Reshape to separate q, k, v
                qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]

                q, k, v = qkv[0], qkv[1], qkv[2]  # Each [B, num_heads, N, head_dim]

                # Select which one to capture
                if self.feature_source == "attention_queries":
                    features = q
                elif self.feature_source == "attention_keys":
                    features = k
                elif self.feature_source == "attention_values":
                    features = v

                # Reshape back to [B, N, C]
                features = features.transpose(1, 2).reshape(B, N, C)
                self.captured_features = features.detach()

            self._hook_handle = last_block.attn.qkv.register_forward_hook(hook_fn)

    def extract_features(
        self,
        images: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract features from images using DINOv3.

        Args:
            images: [B, 3, H, W] image tensor (values in [0, 1])
            normalize: Whether to apply ImageNet normalization

        Returns:
            features: [B, H_patch, W_patch, D] feature tensor
        """
        B, C, H, W = images.shape

        # Apply ImageNet normalization if requested
        if normalize:
            import torchvision.transforms.functional as TF
            images = TF.normalize(
                images,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        # Clear captured features
        self.captured_features = None

        # Forward pass
        with torch.no_grad():
            if self.feature_source == "model_output":
                # Use standard forward_features to get patch tokens
                output = self.dino_model.forward_features(images)
                patch_tokens = output["x_norm_patchtokens"]  # [B, N_patch, D]
            else:
                # Trigger hook by doing forward pass
                _ = self.dino_model.forward_features(images)
                all_tokens = self.captured_features  # [B, N_all, D]

                # Check if hook captured features
                if all_tokens is None:
                    raise RuntimeError(
                        f"Failed to capture features from {self.feature_source}. "
                        f"The hook may not have fired. Check that the model has the expected structure. "
                        f"Model type: {type(self.dino_model).__name__}, "
                        f"Has blocks: {hasattr(self.dino_model, 'blocks')}, "
                        f"Hook registered: {self._hook_handle is not None}"
                    )

                # Extract only patch tokens (skip CLS and register tokens)
                # DINOv3 structure: [CLS, register_tokens..., patch_tokens...]
                patch_size = self.dino_model.patch_size
                H_patch = H // patch_size
                W_patch = W // patch_size
                num_patch_tokens = H_patch * W_patch

                # Get number of register/storage tokens from model
                num_register_tokens = getattr(self.dino_model, 'n_storage_tokens', 0)

                # Slice out patch tokens (skip CLS + register tokens)
                start_idx = 1 + num_register_tokens
                patch_tokens = all_tokens[:, start_idx:start_idx+num_patch_tokens, :]  # [B, N_patch, D]

        # Reshape to spatial grid
        patch_size = self.dino_model.patch_size
        H_patch = H // patch_size
        W_patch = W // patch_size
        D = patch_tokens.shape[-1]

        # Verify we have the right number of patch tokens
        expected_patches = H_patch * W_patch
        actual_patches = patch_tokens.shape[1]

        if expected_patches != actual_patches:
            raise RuntimeError(
                f"Patch count mismatch! "
                f"Image size: {H}x{W}, Patch size: {patch_size}, "
                f"Expected patches: {H_patch}x{W_patch}={expected_patches}, "
                f"Actual patches: {actual_patches}, "
                f"Total tokens: {all_tokens.shape[1] if 'all_tokens' in locals() else 'N/A'}, "
                f"Storage tokens: {getattr(self.dino_model, 'n_storage_tokens', 0)}"
            )

        features = patch_tokens.reshape(B, H_patch, W_patch, D)

        return features

    def remove_hook(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        """Cleanup hook on deletion."""
        self.remove_hook()
