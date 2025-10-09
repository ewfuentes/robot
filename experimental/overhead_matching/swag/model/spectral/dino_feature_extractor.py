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
            # Hook the attention module to capture Q, K, V
            def hook_fn(module, input, output):
                x = input[0]  # [B, N, C]
                B, N, C = x.shape

                # DINOv3 uses a combined qkv projection
                qkv = module.qkv(x)  # [B, N, 3 * num_heads * head_dim]

                # Reshape to separate q, k, v
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
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

            self._hook_handle = last_block.attn.register_forward_hook(hook_fn)

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

                # Extract only patch tokens (skip CLS and register tokens)
                # DINOv3 structure: [CLS, patch_tokens..., register_tokens...]
                # We need to figure out how many patch tokens there are
                patch_size = self.dino_model.patch_size
                H_patch = H // patch_size
                W_patch = W // patch_size
                num_patch_tokens = H_patch * W_patch

                # Slice out patch tokens (skip CLS token at index 0)
                patch_tokens = all_tokens[:, 1:1+num_patch_tokens, :]  # [B, N_patch, D]

        # Reshape to spatial grid
        patch_size = self.dino_model.patch_size
        H_patch = H // patch_size
        W_patch = W // patch_size
        D = patch_tokens.shape[-1]

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
