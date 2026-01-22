# OSM Sentence Embedding Training Pipeline

This document describes how to train sentence embeddings from OpenStreetMap landmark descriptions.

## Overview

The pipeline:
1. Download US city data from SimpleMaps
2. Generate city bounding boxes
3. Download OSM data dumps from Geofabrik
4. Extract landmarks to SQLite database
5. (Optional) Generate LLM sentences and create paired dataset
6. Train sentence embedding model (vocabularies built automatically)

## Prerequisites

- Bazel build system
- ~10GB disk space for OSM dumps
- ~1GB disk space for landmarks database
- GPU recommended for training

## Step 1: Download SimpleMaps US Cities Dataset

Download the free US cities dataset from SimpleMaps:

1. Go to https://simplemaps.com/data/us-cities
2. Download the free "Basic" version (CSV format)
3. Save to `/data/overhead_matching/datasets/simplemaps/uscities.csv`

```bash
mkdir -p /data/overhead_matching/datasets/simplemaps
# Download manually from the website and save as uscities.csv
```

## Step 2: Generate City Bounding Boxes

Generate a YAML file with city bounding boxes for landmark extraction:

```bash
bazel run //experimental/overhead_matching/swag/scripts:generate_city_bboxes -- \
    --input_csv /data/overhead_matching/datasets/simplemaps/uscities.csv \
    --output_yaml /data/overhead_matching/datasets/us_city_bboxes.yaml \
    --min_population 100000 \
    --radius_km 10.0
```

Parameters:
- `--min_population`: Only include cities above this population (default: 100,000)
- `--radius_km`: Bounding box radius around city center (default: 10km)

## Step 3: Download OSM Data Dumps

Download OpenStreetMap PBF files for US states from Geofabrik:

```bash
# Option 1: Use the provided script
./experimental/overhead_matching/swag/scripts/download_osm_dumps.sh

# Option 2: Download manually
mkdir -p /data/overhead_matching/datasets/osm_dumps
cd /data/overhead_matching/datasets/osm_dumps
wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf
wget https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf
# ... repeat for other states
```

The download script downloads all US states (~6.5GB total). Files are named `{state}-200101.osm.pbf`.

## Step 4: Extract Landmarks to SQLite Database

Extract OSM landmarks from the PBF files into a SQLite database:

```bash
bazel run //experimental/overhead_matching/swag/scripts:extract_landmarks_to_sqlite -- \
    --city_bboxes_yaml /data/overhead_matching/datasets/us_city_bboxes.yaml \
    --osm_dumps_dir /data/overhead_matching/datasets/osm_dumps \
    --output_db /data/overhead_matching/datasets/us_osm_landmarks/landmarks.db
```

Options:
- `--states CA NY TX`: Only process specific states
- `--cities "Los Angeles" "New York"`: Only process specific cities
- `--no_geometry`: Skip storing full geometry (smaller database)

This creates a normalized SQLite database with:
- `landmarks`: Deduplicated landmarks with tag signatures
- `tags`: Tag key-value pairs linked to landmarks
- `tag_keys`: Unique tag keys
- `tag_values`: Unique tag values

## Step 5: Train Sentence Embedding Model

Train the multi-task sentence embedding model using a YAML config file.

### Configuration

Copy and modify the example config:

```bash
cp experimental/overhead_matching/swag/scripts/example_sentence_config.yaml my_config.yaml
# Edit my_config.yaml with your paths and settings

bazel run //experimental/overhead_matching/swag/scripts:train_sentence_embeddings -- \
    --config my_config.yaml
```

### Dataset Configuration

The config uses a `datasets` list where each dataset specifies its type via `kind`:

```yaml
datasets:
  # Template dataset: generates sentences from landmark database
  - kind: TemplateDatasetConfig
    db_path: /data/overhead_matching/datasets/us_osm_landmarks/landmarks.sqlite
    weight: 0.9
    # limit: 10000  # Optional: limit landmarks for testing

  # OSM Paired dataset: template + LLM sentence pairs for contrastive learning
  - kind: OSMPairedDatasetConfig
    sentences_path: /data/sentences.pkl
    weight: 0.1
    # limit: 5000  # Optional: limit sentence pairs for testing
```

**Dataset types:**
- `TemplateDatasetConfig`: Loads landmarks from SQLite, generates template sentences on-the-fly
- `OSMPairedDatasetConfig`: Loads pre-computed LLM sentences from a pickle file, pairs with template sentences

**Weights:** Control sampling ratio between datasets. Weights are normalized automatically.

**Limits:** Each dataset can have an optional `limit` for testing with smaller data.

### CLI Options

CLI args override config values:

```bash
bazel run //experimental/overhead_matching/swag/scripts:train_sentence_embeddings -- \
    --config my_config.yaml \
    --batch_size 128 \
    --num_epochs 3
```

Available options:
- `--config`: Path to YAML config file (required)
- `--output_dir`: Output directory for model and logs
- `--tag_vocabs`: Path to precomputed vocabularies (optional)
- `--tensorboard_dir`: TensorBoard log directory (optional)
- `--encoder_name`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--batch_size`: Batch size (default: 256)
- `--num_epochs`: Number of epochs (default: 5)
- `--encoder_lr`: Learning rate for encoder (default: 2e-5)
- `--heads_lr`: Learning rate for heads (default: 1e-4)
- `--train_split`: Fraction of data for training (default: 0.9)
- `--seed`: Random seed (default: 42)
- `--groups_per_batch`: Contrastive groups per batch (default: 32)
- `--samples_per_group`: Samples per contrastive group (default: 4)

### Train/Test Split

The split operates at the **landmark level**, not the sentence level:

1. All landmarks are loaded from the SQLite database
2. Landmarks are shuffled with a fixed seed (default: 42) for reproducibility
3. The first `train_split` fraction (default: 90%) becomes the training set
4. The remaining landmarks become the test set

This ensures:
- **No data leakage**: A landmark is in either train or test, never both
- **Reproducibility**: Same seed produces same split across runs
- **Sentence variety**: Each landmark generates sentences on-the-fly with epoch-based seeds, so different epochs see different sentence variations for the same landmarks

### Training Recommendations

**Epochs**: Start with 3-5 epochs. Monitor validation loss - stop if it plateaus or increases.

**Learning Rates**: The model uses differential learning rates by default:
- `encoder_lr: 2e-5` - Lower rate for pretrained encoder (preserves knowledge)
- `heads_lr: 1e-4` - Higher rate for randomly initialized heads (faster convergence)

**Freezing**: Set `model.freeze_encoder: true` in config to only train heads (faster, less GPU memory, but won't adapt embeddings to OSM language).

### Training Output

The training script outputs:
- `config.yaml`: Training configuration
- `tag_vocabs.json`: Tag vocabularies (if not provided)
- `best_model.pt`: Best model weights
- `checkpoint_epoch_N.pt`: Epoch checkpoints
- `tensorboard/`: TensorBoard logs

### Monitor Training

```bash
bazel run //common/torch:run_tensorboard -- \
    --logdir /data/overhead_matching/models/sentence_embeddings/tensorboard
```

## Model Architecture

The model consists of:
- **Base encoder**: Pretrained sentence transformer (fine-tuned)
- **Classification heads**: Linear layers for categorical tags (amenity, building, highway, etc.)
- **Contrastive heads**: MLP projections for high-cardinality tags (name, addr:street)
- **Presence heads**: Binary classifiers predicting if a tag was used in the sentence

### Training Tasks

**Classification (CrossEntropy loss)** - Template sentences only:
- amenity (~60 classes): restaurant, school, parking, etc.
- building (~76 classes): house, apartments, commercial, etc.
- highway (~20 classes): residential, service, footway, etc.
- And more: shop, leisure, tourism, landuse, natural, surface, cuisine

**Projection Head Contrastive (InfoNCE loss)** - Template sentences only:
- name: Sentences with same landmark name are positives
- addr:street: Sentences on same street are positives

**Presence Prediction (Binary CrossEntropy)** - Template sentences only:
- Predicts which tags were used in generating the sentence

**Base Embedding Contrastive (InfoNCE loss)** - Template + LLM sentences:
- Pairs template and LLM sentences for the **same landmark** as positives
- Uses the base encoder embedding (before projection heads)
- Only active when `OSMPairedDatasetConfig` is included

### Loss Architecture with LLM Sentences

When training with LLM sentences, the architecture computes losses as follows:

```
Template sentence ──┐                      ┌── Classification heads (template only)
                    ├── Base Embedding ────┼── Projection head contrastive (template only)
LLM sentence ───────┘         │            └── Presence heads (template only)
                              │
                              └── Base contrastive loss (cross-source: template ↔ LLM)
```

This design ensures the base embedding learns to produce similar representations for:
- Different template variations of the same landmark
- Template and LLM descriptions of the same landmark

### Key Design Features

1. **On-the-fly sentence generation**: Sentences are generated from tags during training using `OSMSentenceGenerator`, with different seeds per epoch for variety
2. **Masked loss**: Only compute loss for tags present in `used_tags` (tags actually mentioned in the sentence)
3. **Precomputed label matrices**: Collate function builds classification labels and contrastive positive matrices efficiently
4. **Source tracking**: Each sample tracks its source (`"template"` or `"llm"`) to route losses correctly

## File Structure

```
experimental/overhead_matching/swag/
├── data/
│   ├── osm_sentence_generator.py     # Template-based sentence generation
│   ├── sentence_dataset.py           # Dataset, data loading, and collation
│   └── paired_sentence_dataset.py    # Dataset for template + LLM sentence pairs
├── model/
│   ├── sentence_embedding_model.py   # Multi-task model architecture
│   ├── semantic_landmark_extractor.py # Landmark extraction and sentence pickle creation
│   └── semantic_landmark_utils.py    # Landmark pruning and custom ID generation
└── scripts/
    ├── download_osm_dumps.sh         # Download OSM data
    ├── generate_city_bboxes.py       # Generate city bounding boxes
    ├── extract_landmarks_to_sqlite.py # Extract landmarks to SQLite
    ├── build_tag_vocabularies.py     # Build classification vocabularies
    ├── train_sentence_embeddings.py  # Training script
    ├── sentence_configs.py           # Configuration dataclasses
    ├── sentence_losses.py            # Loss functions (incl. base contrastive)
    └── example_sentence_config.yaml  # Example training config
```

## Quick Start (Full Pipeline)

```bash
# 1. Download SimpleMaps data manually from https://simplemaps.com/data/us-cities

# 2. Generate bounding boxes
bazel run //experimental/overhead_matching/swag/scripts:generate_city_bboxes

# 3. Download OSM dumps
./experimental/overhead_matching/swag/scripts/download_osm_dumps.sh

# 4. Extract landmarks
bazel run //experimental/overhead_matching/swag/scripts:extract_landmarks_to_sqlite -- \
    --output_db /data/overhead_matching/datasets/us_osm_landmarks/landmarks.db

# 5. Create config file
cp experimental/overhead_matching/swag/scripts/example_sentence_config.yaml my_config.yaml
# Edit paths in my_config.yaml

# 6. Train (vocabularies built automatically)
bazel run //experimental/overhead_matching/swag/scripts:train_sentence_embeddings -- \
    --config my_config.yaml
```

## Optional Tools

### Inspect Tag Vocabularies

To inspect database statistics or pre-build vocabularies before training:

```bash
bazel run //experimental/overhead_matching/swag/scripts:build_tag_vocabularies -- \
    --db_path /data/overhead_matching/datasets/us_osm_landmarks/landmarks.db \
    --output /data/overhead_matching/datasets/us_osm_landmarks/tag_vocabs.json \
    --min_count 100 \
    --show_stats
```

Options:
- `--min_count`: Minimum count for a value to be included (default: 100)
- `--tags amenity building highway`: Only build vocabularies for specific tags
- `--show_stats`: Print database statistics

This is useful for:
- Inspecting class distributions before training
- Sharing vocabularies across multiple training runs
- Using `--tag_vocabs` flag in the training script to skip vocabulary building

### Training with LLM Sentences

To improve robustness to natural language variation, you can train with LLM-generated sentences in addition to template sentences.

#### Step 1: Generate LLM Sentences

Use the OpenAI batch API to generate sentences from landmark tags:

```bash
# Create batch API requests
bazel run //experimental/overhead_matching/swag/model:semantic_landmark_extractor -- \
    create_sentences \
    --geojson /data/landmarks.geojson \
    --output_base /tmp/sentence_generation \
    --launch  # Optional: auto-launch batch job

# Fetch results when complete
bazel run //experimental/overhead_matching/swag/model:semantic_landmark_extractor -- \
    fetch \
    --batch_ids <batch_id_1> <batch_id_2> \
    --output_folder /data/sentence_responses/
```

#### Step 2: Create Sentences Pickle File

Convert the JSONL responses to a pickle file for training:

```bash
bazel run //experimental/overhead_matching/swag/model:semantic_landmark_extractor -- \
    create_sentences_pickle \
    --geojson /data/landmarks.geojson /data/more_landmarks.feather \
    --sentences_dir /data/sentence_responses/ \
    --output /data/sentences.pkl
```

This creates a pickle file mapping `frozenset[tuple[str, str]] -> str` (pruned tags to sentence).

#### Step 3: Configure Training

Add an `OSMPairedDatasetConfig` to your training config:

```yaml
datasets:
  - kind: TemplateDatasetConfig
    db_path: /data/landmarks.sqlite
    weight: 0.9

  - kind: OSMPairedDatasetConfig
    sentences_path: /data/sentences.pkl
    weight: 0.1
```

#### How it works

1. **Landmark matching**: LLM sentences are matched to landmarks using `custom_id_from_props(prune_landmark(tags))`, which creates a hash of the pruned tag properties.

2. **Paired dataset**: `PairedSentenceDataset` yields (template, llm) pairs for each landmark that has an LLM sentence.

3. **Base contrastive loss**: The base encoder embedding (before projection heads) is trained with InfoNCE loss where template and LLM sentences for the same landmark are positives.

4. **Template-only losses**: Classification, presence, and projection head contrastive losses are only computed for template sentences (since we know which tags were used).

**Pickle file format:**

```python
# dict[frozenset[tuple[str, str]], str]
{
    frozenset({("amenity", "restaurant"), ("name", "Joe's Diner")}): "A restaurant called Joe's Diner",
    frozenset({("shop", "bakery"), ("name", "Fresh Bread")}): "Fresh Bread is a bakery shop",
    ...
}
```

The keys are pruned tags (frozensets of (key, value) tuples) and values are LLM-generated sentences.
