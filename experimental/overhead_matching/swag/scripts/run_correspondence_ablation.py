#!/usr/bin/env python3
"""Run feature ablation sweep for the correspondence classifier.

Trains the classifier multiple times, each with a different set of features
disabled, to identify which feature groups matter most.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:run_correspondence_ablation -- \
        --base_config /path/to/config.yaml \
        --output_base /data/overhead_matching/training_outputs/landmark_correspondence/ablation_sweep
"""

import argparse
import json
import copy
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401 - Must import before torch

from experimental.overhead_matching.swag.scripts.correspondence_configs import (
    CorrespondenceTrainConfig,
    load_config,
    save_config,
)
from experimental.overhead_matching.swag.scripts.train_landmark_correspondence import (
    train,
)


ABLATION_VARIANTS = {
    # name -> list of ablation flags
    "full": [],                                                          # baseline
    "no_cross": ["no_cross"],                                            # encoder only
    "no_text": ["no_text"],                                              # no text embeddings
    "no_numeric_bool_hn": ["no_numeric", "no_boolean", "no_housenumber"],  # text + keys + cross
    "no_encoder": ["no_encoder"],                                        # cross features only
    "no_keys": ["no_keys"],                                              # no key embeddings
    "text_cross_only": ["no_numeric", "no_boolean", "no_housenumber", "no_keys"],  # text + cross
    "keys_cross_only": ["no_text", "no_numeric", "no_boolean", "no_housenumber"],  # keys + cross
    "minimal": ["no_numeric", "no_boolean", "no_housenumber", "only_text_cross"],  # keys + text + 3 text cosine cross
    "minimal_no_keys": ["no_numeric", "no_boolean", "no_housenumber", "no_keys", "only_text_cross"],  # text + 3 text cosine cross
    "minimal_no_text": ["no_numeric", "no_boolean", "no_housenumber", "no_text", "only_text_cross"],  # keys + 3 text cosine cross
}


def run_sweep(base_config: CorrespondenceTrainConfig, output_base: Path,
              variants: list[str] | None = None) -> None:
    results = {}
    variant_names = variants if variants else list(ABLATION_VARIANTS.keys())

    for name in variant_names:
        if name not in ABLATION_VARIANTS:
            print(f"Unknown variant: {name}, skipping")
            continue

        ablation_flags = ABLATION_VARIANTS[name]
        print(f"\n{'=' * 70}")
        print(f"ABLATION: {name} (flags: {ablation_flags or 'none'})")
        print(f"{'=' * 70}\n")

        # Create a modified config for this variant
        config = copy.copy(base_config)
        config = CorrespondenceTrainConfig(
            data_dir=base_config.data_dir,
            text_embeddings_path=base_config.text_embeddings_path,
            output_dir=output_base / name,
            train_city=base_config.train_city,
            val_city=base_config.val_city,
            include_difficulties=base_config.include_difficulties,
            batch_size=base_config.batch_size,
            num_epochs=base_config.num_epochs,
            lr=base_config.lr,
            weight_decay=base_config.weight_decay,
            warmup_fraction=base_config.warmup_fraction,
            gradient_clip_norm=base_config.gradient_clip_norm,
            use_amp=base_config.use_amp,
            num_workers=base_config.num_workers,
            seed=base_config.seed,
            encoder=base_config.encoder,
            classifier=base_config.classifier,
            cosine_schedule=base_config.cosine_schedule,
            ablation=ablation_flags,
            all_text=base_config.all_text,
        )

        train(config)

        # Read back best model's val metrics from the last checkpoint
        best_model_path = config.output_dir / "best_model.pt"
        if best_model_path.exists():
            results[name] = {"ablation": ablation_flags, "model_saved": True}
        else:
            results[name] = {"ablation": ablation_flags, "model_saved": False}

    # Print summary
    print(f"\n{'=' * 70}")
    print("ABLATION SWEEP COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_base}")
    print(f"Variants run: {list(results.keys())}")
    print(f"\nCompare with tensorboard:")
    print(f"  bazel run //common/torch:run_tensorboard -- --logdir {output_base}")

    # Save results index
    with open(output_base / "sweep_index.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Correspondence classifier ablation sweep")
    parser.add_argument("--base_config", type=Path, required=True,
                        help="Base YAML config (ablation and output_dir will be overridden)")
    parser.add_argument("--output_base", type=Path, required=True,
                        help="Base output directory (each variant gets a subdirectory)")
    parser.add_argument("--variants", nargs="+", default=None,
                        help=f"Which variants to run (default: all). "
                             f"Options: {', '.join(ABLATION_VARIANTS.keys())}")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    output_base = args.output_base
    output_base.mkdir(parents=True, exist_ok=True)

    run_sweep(base_config, output_base, args.variants)


if __name__ == "__main__":
    main()
