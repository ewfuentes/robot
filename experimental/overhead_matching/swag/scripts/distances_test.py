import common.torch.load_torch_deps
import torch
from experimental.overhead_matching.swag.scripts.distances import (
    LearnedDistanceFunctionConfig,
    LearnedDistanceFunction,
    create_distance_from_config,
    normalize_embeddings
)


def test_learned_distance_function_mlp():
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

    # Check output shape: n_pano x n_sat x 1 x 1
    assert output.shape == (3, 2, 1, 1), f"Expected shape (3, 2, 1, 1), got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_learned_distance_function_attention():
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

    # Check output shape: n_pano x n_sat x 1 x 1
    assert output.shape == (3, 2, 1, 1), f"Expected shape (3, 2, 1, 1), got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_learned_distance_function_transformer():
    """Test transformer decoder architecture forward pass."""
    torch.manual_seed(42)
    sat_embeddings = torch.randn(2, 3, 128)  # 2 sat images, 3 embeddings each, 128 dim
    pano_embeddings = torch.randn(3, 4, 128)  # 3 pano images, 4 embeddings each, 128 dim

    config = LearnedDistanceFunctionConfig(
        architecture="transformer_decoder",
        embedding_dim=128,
        num_pano_embed=4,
        num_sat_embed=3,
        hidden_dim=256,
        num_heads=8,
        num_layers=2
    )
    model = LearnedDistanceFunction(config)

    output = model(sat_embeddings, pano_embeddings)

    # Check output shape: n_pano x n_sat x 1 x 1
    assert output.shape == (3, 2, 1, 1), f"Expected shape (3, 2, 1, 1), got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_learned_distance_function_invalid_architecture():
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


def test_learned_distance_function_factory():
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


def test_learned_distance_function_gradient_flow():
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


def test_learned_distance_function_different_input_sizes():
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

    assert output.shape == (2, 5, 1, 1), f"Expected shape (2, 5, 1, 1), got {output.shape}"


def test_learned_distance_function_cls_token():
    """Test that CLS token is properly initialized for transformer."""
    config = LearnedDistanceFunctionConfig(
        architecture="transformer_decoder",
        embedding_dim=128,
        num_pano_embed=4,
        num_sat_embed=3,
        hidden_dim=256,
        num_heads=8,
        num_layers=1
    )
    model = LearnedDistanceFunction(config)

    assert hasattr(model, 'cls_token'), "Transformer model should have cls_token"
    assert model.cls_token.shape == (1, 1, 128), f"CLS token shape should be (1, 1, 128), got {model.cls_token.shape}"
    assert model.cls_token.requires_grad, "CLS token should be trainable"


def test_normalize_embeddings():
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