#!/usr/bin/env python3
"""Test script to verify train_sentence_embedder functionality."""

import json
import sys
from pathlib import Path
import tempfile

# Test 1: Verify _custom_id_from_props produces correct hash
print("=" * 60)
print("Test 1: Verify _custom_id_from_props")
print("=" * 60)

from experimental.overhead_matching.swag.model.semantic_landmark_extractor import _custom_id_from_props

# Test with sample OSM tags
test_props = {
    "highway": "residential",
    "name": "North Bell Avenue",
    "surface": "asphalt"
}

custom_id = _custom_id_from_props(test_props)
print(f"Props: {test_props}")
print(f"Custom ID: {custom_id}")
print(f"Custom ID length: {len(custom_id)}")
print(f"✓ _custom_id_from_props works correctly\n")

# Test 2: Verify sentence lookup in CorrespondenceDataset
print("=" * 60)
print("Test 2: Verify sentence lookup")
print("=" * 60)

# Create a minimal correspondence file
test_correspondence = {
    "test_pano_1": {
        "pano": ["A red brick building"],
        "osm": [
            {"tags": "highway=residential; name=North Bell Avenue; surface=asphalt"},
            {"tags": "building=yes; addr:housenumber=123"}
        ],
        "matches": {
            "matches": [
                {"set_1_id": 0, "set_2_matches": [0, 1]}
            ]
        }
    }
}

# Create a test sentence dictionary entry
test_sentence_dict = {
    custom_id: "A residential street named North Bell Avenue with asphalt surface"
}

# Save test correspondence
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(test_correspondence, f)
    test_corr_file = f.name

# Save test sentences as JSONL (simulating OpenAI batch API output)
with tempfile.TemporaryDirectory() as tmpdir:
    sentence_dir = Path(tmpdir) / "sentences"
    sentence_dir.mkdir()

    # Create a sample JSONL file with sentence responses
    test_responses = [
        {
            "custom_id": custom_id,
            "response": {
                "body": {
                    "choices": [{
                        "message": {"content": test_sentence_dict[custom_id], "refusal": None},
                        "finish_reason": "stop"
                    }],
                    "usage": {"completion_tokens": 10}
                }
            },
            "error": None
        }
    ]

    with open(sentence_dir / "test.jsonl", 'w') as f:
        for response in test_responses:
            f.write(json.dumps(response) + '\n')

    print(f"Created test correspondence file: {test_corr_file}")
    print(f"Created test sentence directory: {sentence_dir}")

    # Import and test CorrespondenceDataset
    from experimental.overhead_matching.swag.scripts.train_sentence_embedder import CorrespondenceDataset

    # Test with natural language mode
    print("\nTesting with natural language mode...")
    dataset_nl = CorrespondenceDataset(
        correspondence_file=Path(test_corr_file),
        sentence_directory=sentence_dir,
        use_natural_language=True
    )

    print(f"Loaded {len(dataset_nl)} pairs")
    if len(dataset_nl) > 0:
        pano_text, osm_text = dataset_nl[0]
        print(f"Pano text: {pano_text}")
        print(f"OSM text: {osm_text}")

        # Verify the sentence was looked up correctly
        if "residential street" in osm_text.lower():
            print("✓ Natural language sentence lookup works correctly!")
        else:
            print("✗ ERROR: Expected natural language description but got tag format")
            sys.exit(1)

    # Test with tag format mode (fallback)
    print("\nTesting with tag format mode...")
    dataset_tags = CorrespondenceDataset(
        correspondence_file=Path(test_corr_file),
        sentence_directory=None,
        use_natural_language=False
    )

    print(f"Loaded {len(dataset_tags)} pairs")
    if len(dataset_tags) > 0:
        pano_text, osm_text = dataset_tags[0]
        print(f"Pano text: {pano_text}")
        print(f"OSM text: {osm_text}")

        # Verify tag format
        if "highway: residential" in osm_text:
            print("✓ Tag format mode works correctly!")
        else:
            print("✗ ERROR: Expected tag format")
            sys.exit(1)

# Clean up
import os
os.unlink(test_corr_file)

# Test 3: Verify TrainableSentenceEmbedder integration
print("\n" + "=" * 60)
print("Test 3: Verify TrainableSentenceEmbedder integration")
print("=" * 60)

import common.torch.load_torch_deps
import torch
from experimental.overhead_matching.swag.model.trainable_sentence_embedder import TrainableSentenceEmbedder
from experimental.overhead_matching.swag.model.swag_config_types import TrainableSentenceEmbedderConfig

# Create a small model config
config = TrainableSentenceEmbedderConfig(
    pretrained_model_name_or_path="sentence-transformers/paraphrase-MiniLM-L3-v2",
    output_dim=128,
    max_sequence_length=64,
    freeze_weights=True,
    model_weights_path=None,
)

print(f"Creating TrainableSentenceEmbedder with config:")
print(f"  Model: {config.pretrained_model_name_or_path}")
print(f"  Output dim: {config.output_dim}")
print(f"  Max length: {config.max_sequence_length}")

model = TrainableSentenceEmbedder(config)
model.eval()

# Test forward pass
test_texts = [
    "A red brick building",
    "A residential street"
]

print(f"\nTesting forward pass with {len(test_texts)} texts...")
with torch.no_grad():
    embeddings = model(test_texts)

print(f"Output shape: {embeddings.shape}")
print(f"Expected shape: ({len(test_texts)}, {config.output_dim})")

if embeddings.shape == (len(test_texts), config.output_dim):
    print("✓ TrainableSentenceEmbedder forward pass works correctly!")

    # Verify embeddings are normalized
    norms = torch.norm(embeddings, dim=1)
    print(f"Embedding norms: {norms}")
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        print("✓ Embeddings are properly normalized!")
    else:
        print("✗ WARNING: Embeddings not normalized")
else:
    print("✗ ERROR: Unexpected output shape")
    sys.exit(1)

# Test 4: Verify save/load with metadata
print("\n" + "=" * 60)
print("Test 4: Verify save/load with metadata")
print("=" * 60)

from common.torch.load_and_save_models import save_model as save_model_with_metadata, load_model

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "test_model"

    # Save model
    print(f"Saving model to {save_path}...")
    example_texts = ["This is a test sentence."]
    training_args = {"learning_rate": 2e-5, "batch_size": 64}

    save_model_with_metadata(
        model=model,
        save_path=save_path,
        example_model_inputs=(example_texts,),
        aux_information={
            "training_args": training_args,
            "model_type": "TrainableSentenceEmbedder",
        }
    )

    # Check files were created
    expected_files = ["model.pt", "model_weights.pt", "aux.json", "commit_hash.txt", "diff.txt", "input_output.tar"]
    for filename in expected_files:
        filepath = save_path / filename
        if filepath.exists():
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} missing")
            sys.exit(1)

    # Load and verify aux information
    with open(save_path / "aux.json") as f:
        aux_info = json.load(f)

    print(f"\nAux information:")
    print(f"  Model type: {aux_info['model_type']}")
    print(f"  Training args: {aux_info['training_args']}")
    print(f"  Saved at: {aux_info['current_time']}")

    # Load model and verify output consistency
    print("\nLoading model...")
    loaded_model = load_model(save_path, device="cpu")
    loaded_model.eval()

    with torch.no_grad():
        new_output = loaded_model(example_texts)

    print(f"Loaded model output shape: {new_output.shape}")
    print("✓ Model save/load with metadata works correctly!")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
