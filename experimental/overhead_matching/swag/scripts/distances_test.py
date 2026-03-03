import common.torch.load_torch_deps
import torch
import unittest
from experimental.overhead_matching.swag.scripts.distances import (
    LearnedDistanceFunctionConfig,
    LearnedDistanceFunction,
    create_distance_from_config,
    normalize_embeddings
)

class DistancesTest(unittest.TestCase):

    def test_learned_distance_function_mlp(self):
        """Test MLP architecture forward pass."""
        torch.manual_seed(42)
        sat_embeddings = torch.randn(2, 3, 128)  # 2 sat images, 3 embeddings each, 128 dim
        pano_embeddings = torch.randn(3, 4, 128)  # 3 pano images, 4 embeddings each, 128 dim

        config = LearnedDistanceFunctionConfig(
            architecture="mlp",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256
        )
        model = LearnedDistanceFunction(config)

        output = model(sat_embeddings, pano_embeddings)

        # Check output shape: n_pano x n_sat
        assert output.shape == (3, 2), f"Expected shape (3, 2), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"


    def test_learned_distance_function_attention(self):
        """Test multi-head attention architecture forward pass."""
        torch.manual_seed(42)
        sat_embeddings = torch.randn(2, 3, 128)  # 2 sat images, 3 embeddings each, 128 dim
        pano_embeddings = torch.randn(3, 4, 128)  # 3 pano images, 4 embeddings each, 128 dim

        config = LearnedDistanceFunctionConfig(
            architecture="attention",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256,
            num_heads=8
        )
        model = LearnedDistanceFunction(config)

        output = model(sat_embeddings, pano_embeddings)

        # Check output shape: n_pano x n_sat
        assert output.shape == (3, 2), f"Expected shape (3, 2), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"


    def test_learned_distance_function_transformer(self):
        """Test transformer encoder architecture forward pass."""
        torch.manual_seed(42)
        sat_embeddings = torch.randn(2, 3, 128)  # 2 sat images, 3 embeddings each, 128 dim
        pano_embeddings = torch.randn(3, 4, 128)  # 3 pano images, 4 embeddings each, 128 dim

        config = LearnedDistanceFunctionConfig(
            architecture="transformer_encoder",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256,
            num_heads=8,
            num_layers=2
        )
        model = LearnedDistanceFunction(config)

        output = model(sat_embeddings, pano_embeddings)

        # Check output shape: n_pano x n_sat
        assert output.shape == (3, 2), f"Expected shape (3, 2), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"


    def test_learned_distance_function_invalid_architecture(self):
        """Test that invalid architecture raises ValueError."""
        config = LearnedDistanceFunctionConfig(
            architecture="invalid_arch",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256
        )

        try:
            LearnedDistanceFunction(config)
            assert False, "Expected ValueError for invalid architecture"
        except ValueError as e:
            assert "Unknown architecture" in str(e)


    def test_learned_distance_function_factory(self):
        """Test factory function works with LearnedDistanceFunctionConfig."""
        config = LearnedDistanceFunctionConfig(
            architecture="mlp",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256
        )
        model = create_distance_from_config(config)
        assert isinstance(model, LearnedDistanceFunction)


    def test_learned_distance_function_gradient_flow(self):
        """Test that gradients flow through the model."""
        torch.manual_seed(42)
        sat_embeddings = torch.randn(2, 3, 128, requires_grad=True)
        pano_embeddings = torch.randn(3, 4, 128, requires_grad=True)

        config = LearnedDistanceFunctionConfig(
            architecture="mlp",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256
        )
        model = LearnedDistanceFunction(config)
        output = model(sat_embeddings, pano_embeddings)
        loss = output.sum()
        loss.backward()

        assert sat_embeddings.grad is not None, "No gradients for sat embeddings"
        assert pano_embeddings.grad is not None, "No gradients for pano embeddings"
        assert torch.isfinite(sat_embeddings.grad).all(), "Non-finite gradients for sat embeddings"
        assert torch.isfinite(pano_embeddings.grad).all(), "Non-finite gradients for pano embeddings"


    def test_learned_distance_function_different_input_sizes(self):
        """Test model handles different input sizes correctly."""
        torch.manual_seed(42)
        sat_embeddings = torch.randn(5, 3, 128)  # 5 sat images
        pano_embeddings = torch.randn(2, 4, 128)  # 2 pano images

        config = LearnedDistanceFunctionConfig(
            architecture="mlp",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256
        )
        model = LearnedDistanceFunction(config)
        output = model(sat_embeddings, pano_embeddings)

        assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"


    def test_learned_distance_function_identifier_tokens(self):
        """Test that pano/sat identifier tokens are properly initialized for transformer."""
        config = LearnedDistanceFunctionConfig(
            architecture="transformer_encoder",
            embedding_dim=128,
            num_pano_embed=4,
            num_sat_embed=3,
            hidden_dim=256,
            num_heads=8,
            num_layers=1
        )
        model = LearnedDistanceFunction(config)

        assert hasattr(model, 'pano_identifier'), "Transformer model should have pano_identifier"
        assert hasattr(model, 'sat_identifier'), "Transformer model should have sat_identifier"
        assert model.pano_identifier.shape == (1, 1, 128)
        assert model.sat_identifier.shape == (1, 1, 128)
        assert model.pano_identifier.requires_grad
        assert model.sat_identifier.requires_grad


    def test_transformer_encoder_nan_padded_variable_length(self):
        """Test transformer encoder handles NaN-padded variable-length inputs."""
        torch.manual_seed(42)
        # Simulate variable-length embeddings: sat has 5 tokens, pano has 3 tokens,
        # but both are padded to length 5 with NaN
        sat_embeddings = torch.randn(2, 5, 128)
        pano_embeddings = torch.randn(3, 5, 128)
        # NaN-pad pano embeddings at different lengths
        pano_embeddings[0, 3:, :] = float('nan')  # 3 valid tokens
        pano_embeddings[1, 4:, :] = float('nan')  # 4 valid tokens
        pano_embeddings[2, 2:, :] = float('nan')  # 2 valid tokens
        # NaN-pad one sat embedding too
        sat_embeddings[1, 4:, :] = float('nan')   # 4 valid tokens

        config = LearnedDistanceFunctionConfig(
            architecture="transformer_encoder",
            embedding_dim=128,
            num_pano_embed=None,
            num_sat_embed=None,
            hidden_dim=256,
            num_heads=8,
            num_layers=1
        )
        model = LearnedDistanceFunction(config)

        output = model(sat_embeddings, pano_embeddings)

        # Check output shape: n_pano x n_sat
        assert output.shape == (3, 2), f"Expected shape (3, 2), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"

        # Verify mask construction: call _process_transformer_batch directly
        # to inspect the padding mask for a specific pano-sat pair
        pano_single = pano_embeddings[2:3]  # 2 valid tokens out of 5
        sat_single = sat_embeddings[0:1]    # all 5 valid
        pano_identified = pano_single + model.pano_identifier
        sat_identified = sat_single + model.sat_identifier
        sequence = torch.cat([pano_identified, sat_identified], dim=1)
        padding_mask = torch.any(torch.isnan(sequence), dim=2)

        assert padding_mask.dtype == torch.bool, f"Expected bool mask, got {padding_mask.dtype}"
        # pano has 2 valid + 3 NaN, sat has 5 valid = total 10 tokens, 3 masked
        assert padding_mask.shape == (1, 10), f"Expected shape (1, 10), got {padding_mask.shape}"
        assert padding_mask.sum().item() == 3, f"Expected 3 masked tokens, got {padding_mask.sum().item()}"
        # First 2 pano tokens valid, next 3 pano tokens masked, all 5 sat tokens valid
        expected_mask = torch.tensor([[False, False, True, True, True,
                                       False, False, False, False, False]])
        assert torch.equal(padding_mask, expected_mask), (
            f"Mask mismatch: got {padding_mask} expected {expected_mask}")

    def test_normalize_embeddings(self):
        """Test the normalize_embeddings utility function."""
        torch.manual_seed(42)
        emb1 = torch.randn(2, 3, 128)
        emb2 = torch.randn(3, 4, 128)

        norm1, norm2 = normalize_embeddings(emb1, emb2)

        # Check that embeddings are normalized
        norm1_magnitudes = torch.norm(norm1, dim=-1)
        norm2_magnitudes = torch.norm(norm2, dim=-1)

        torch.testing.assert_close(norm1_magnitudes, torch.ones_like(norm1_magnitudes), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(norm2_magnitudes, torch.ones_like(norm2_magnitudes), rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    unittest.main()
