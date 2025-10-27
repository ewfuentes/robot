# Training Makefile for overhead matching experiments
# Usage: make train-<config-name> [OUTPUT_BASE=/path/to/output]

# Default output directory (can be overridden)
OUTPUT_BASE ?= /tmp/training_outputs

# Dataset base path
DATASET_BASE = /data/overhead_matching/datasets/VIGOR/

# Config directory
CONFIG_DIR = $(HOME)/scratch

# Base training command
TRAIN_CMD = MAX_DATALOADER_WORKERS=4 bazel run //experimental/overhead_matching/swag/scripts:train -- \
	--dataset_base $(DATASET_BASE) \
	--output_base $(OUTPUT_BASE) \
	--no_ipdb \
	--train_config

# Semantic landmark configs with varying panorama radius
train-radius-50:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_50.yaml

train-radius-100:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_100.yaml

train-radius-120:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_120.yaml

train-radius-200:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_200.yaml

train-radius-320:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_320.yaml

train-radius-400:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_400.yaml

train-radius-480:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_480.yaml

train-radius-560:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_560.yaml

train-radius-640:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_640.yaml

train-radius-720:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_720.yaml

train-radius-800:
	$(TRAIN_CMD) /tmp/semantic_landmark_pano_radius_800.yaml

# Run all panorama radius configs sequentially (continue on failure)
train-all:
	-$(MAKE) train-radius-50
	-$(MAKE) train-radius-100
	-$(MAKE) train-radius-120
	-$(MAKE) train-radius-200
	-$(MAKE) train-radius-320
	-$(MAKE) train-radius-400
	-$(MAKE) train-radius-480
	-$(MAKE) train-radius-560
	-$(MAKE) train-radius-640
	-$(MAKE) train-radius-720
	-$(MAKE) train-radius-800

# List available configs
list-configs:
	@echo "Available training configs:"
	@ls -1 $(CONFIG_DIR)/*.yaml | sed 's|.*/||' | sed 's|\.yaml||'

# Create output directory
create-output-dir:
	@mkdir -p $(OUTPUT_BASE)

# Help target
help:
	@echo "Training Makefile Usage:"
	@echo ""
	@echo "Individual configs (varying panorama landmark radius):"
	@echo "  make train-radius-50   - Train with panorama_landmark_radius_px=50"
	@echo "  make train-radius-100  - Train with panorama_landmark_radius_px=100"
	@echo "  make train-radius-120  - Train with panorama_landmark_radius_px=120"
	@echo "  make train-radius-200  - Train with panorama_landmark_radius_px=200"
	@echo "  make train-radius-320  - Train with panorama_landmark_radius_px=320"
	@echo "  make train-radius-400  - Train with panorama_landmark_radius_px=400"
	@echo "  make train-radius-480  - Train with panorama_landmark_radius_px=480"
	@echo "  make train-radius-560  - Train with panorama_landmark_radius_px=560"
	@echo "  make train-radius-640  - Train with panorama_landmark_radius_px=640 (original default)"
	@echo "  make train-radius-720  - Train with panorama_landmark_radius_px=720"
	@echo "  make train-radius-800  - Train with panorama_landmark_radius_px=800"
	@echo ""
	@echo "Batch operations:"
	@echo "  make train-all         - Run all panorama radius configs sequentially"
	@echo ""
	@echo "Utilities:"
	@echo "  make list-configs      - List available config files"
	@echo "  make create-output-dir - Create output directory"
	@echo ""
	@echo "Variables:"
	@echo "  OUTPUT_BASE            - Output directory (default: /tmp/training_outputs)"
	@echo ""
	@echo "Examples:"
	@echo "  make train-radius-640"
	@echo "  make train-radius-200 OUTPUT_BASE=/data/experiments"
	@echo "  make train-all OUTPUT_BASE=/data/batch_run"

.PHONY: train-radius-50 train-radius-100 train-radius-120 train-radius-200 train-radius-320 train-radius-400 train-radius-480 train-radius-560 train-radius-640 train-radius-720 train-radius-800 train-all list-configs create-output-dir help
