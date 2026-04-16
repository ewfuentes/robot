#!/usr/bin/env python3
"""Evaluate ablation sweep models on held-out cities.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:evaluate_ablation_sweep -- \
        --sweep_dir /data/.../ablation_sweep_v5 \
        --text_embeddings_path /data/.../eval_text_embeddings_all_cities.pkl \
        --data_dirs /data/.../mapillary/*/responses /data/.../boston_snowy/responses ...
"""

import argparse
import json
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    LandmarkCorrespondenceDataset,
    collate_correspondence,
    load_pairs_from_directory,
    load_text_embeddings,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    TagBundleEncoderConfig,
)
from experimental.overhead_matching.swag.scripts.correspondence_configs import (
    load_config,
)


def eval_model_on_pairs(model, pairs, text_embeddings, text_input_dim, device):
    ds = LandmarkCorrespondenceDataset(
        pairs, text_embeddings, text_input_dim,
        include_difficulties=("positive", "easy", "hard"),
    )
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        collate_fn=collate_correspondence, num_workers=0)
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(
                pano_key_indices=batch.pano_key_indices,
                pano_value_type=batch.pano_value_type,
                pano_boolean_values=batch.pano_boolean_values,
                pano_numeric_values=batch.pano_numeric_values,
                pano_numeric_nan_mask=batch.pano_numeric_nan_mask,
                pano_housenumber_values=batch.pano_housenumber_values,
                pano_housenumber_nan_mask=batch.pano_housenumber_nan_mask,
                pano_text_embeddings=batch.pano_text_embeddings,
                pano_tag_mask=batch.pano_tag_mask,
                osm_key_indices=batch.osm_key_indices,
                osm_value_type=batch.osm_value_type,
                osm_boolean_values=batch.osm_boolean_values,
                osm_numeric_values=batch.osm_numeric_values,
                osm_numeric_nan_mask=batch.osm_numeric_nan_mask,
                osm_housenumber_values=batch.osm_housenumber_values,
                osm_housenumber_nan_mask=batch.osm_housenumber_nan_mask,
                osm_text_embeddings=batch.osm_text_embeddings,
                osm_tag_mask=batch.osm_tag_mask,
                cross_features=batch.cross_features,
            ).squeeze(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.labels.cpu().tolist())

    la = np.array(all_labels)
    pa = np.array(all_probs)
    try:
        auc = roc_auc_score(la, pa)
    except ValueError:
        auc = float("nan")
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=Path, required=True)
    parser.add_argument("--text_embeddings_path", type=Path, required=True)
    parser.add_argument("--data_dirs", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embeddings = load_text_embeddings(args.text_embeddings_path)
    text_input_dim = next(iter(text_embeddings.values())).shape[0]

    # Load all city pairs once
    city_pairs = {}
    for data_dir in args.data_dirs:
        city_name = data_dir.parent.name if data_dir.name == "responses" else data_dir.name
        pairs = load_pairs_from_directory(data_dir)
        if pairs:
            city_pairs[city_name] = pairs
            n_pos = sum(1 for p in pairs if p.difficulty == "positive")
            print(f"  {city_name}: {len(pairs)} pairs ({n_pos} positive)")

    # Find variant directories
    variant_dirs = sorted([
        d for d in args.sweep_dir.iterdir()
        if d.is_dir() and (d / "best_model.pt").exists()
    ])

    # Evaluate each variant
    results = {}  # variant -> {city -> auc}
    for variant_dir in variant_dirs:
        variant_name = variant_dir.name
        config = load_config(variant_dir / "config.yaml")
        ablation = frozenset(config.ablation)

        encoder_config = TagBundleEncoderConfig(
            text_input_dim=text_input_dim, text_proj_dim=128)
        classifier_config = CorrespondenceClassifierConfig(
            encoder=encoder_config, ablation=ablation)
        model = CorrespondenceClassifier(classifier_config).to(device)
        model.load_state_dict(torch.load(
            variant_dir / "best_model.pt", map_location=device, weights_only=True))
        model.eval()

        results[variant_name] = {}
        for city_name, pairs in city_pairs.items():
            auc = eval_model_on_pairs(model, pairs, text_embeddings, text_input_dim, device)
            results[variant_name][city_name] = auc

    # Print table
    cities = list(city_pairs.keys())
    header = f"{'Variant':25s}" + "".join(f" {c:>12s}" for c in cities) + f" {'MEAN':>8s}"
    print(f"\n{header}")
    print("-" * len(header))
    for variant_name in [d.name for d in variant_dirs]:
        row = f"{variant_name:25s}"
        aucs = []
        for city in cities:
            auc = results[variant_name].get(city)
            if auc is not None and not np.isnan(auc):
                row += f" {auc:12.4f}"
                aucs.append(auc)
            else:
                row += f" {'N/A':>12s}"
        mean_auc = np.mean(aucs) if aucs else float("nan")
        row += f" {mean_auc:8.4f}"
        print(row)

    # Save JSON
    out_path = args.sweep_dir / "held_out_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
