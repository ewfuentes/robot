#!/usr/bin/env python3
"""Evaluate landmark correspondence model on per-city data.

Usage:
    bazel run //experimental/overhead_matching/swag/scripts:evaluate_correspondence_model -- \
        --model_path /data/.../best_model.pt \
        --text_embeddings_path /data/.../eval_text_embeddings.pkl \
        --data_dirs /data/.../mapillary/Framingham /data/.../nightdrive
"""

import argparse
import functools
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


def eval_dataset(model, pairs, text_embeddings, text_input_dim, difficulties, device):
    ds = LandmarkCorrespondenceDataset(
        pairs, text_embeddings, text_input_dim, include_difficulties=difficulties,
    )
    if len(ds) == 0:
        return None
    collate_fn = functools.partial(
        collate_correspondence, text_input_dim=text_input_dim,
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(
                pano_key_indices=batch.pano_key_indices,
                pano_text_embeddings=batch.pano_text_embeddings,
                pano_tag_mask=batch.pano_tag_mask,
                osm_key_indices=batch.osm_key_indices,
                osm_text_embeddings=batch.osm_text_embeddings,
                osm_tag_mask=batch.osm_tag_mask,
                cross_features=batch.cross_features,
            ).squeeze(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch.labels.cpu().tolist())

    la = np.array(all_labels)
    pa = np.array(all_probs)
    preds = (pa >= 0.5).astype(float)
    tp = ((preds == 1) & (la == 1)).sum()
    fp = ((preds == 1) & (la == 0)).sum()
    fn = ((preds == 0) & (la == 1)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    try:
        auc = roc_auc_score(la, pa)
    except ValueError:
        auc = float("nan")
    return {"auc": auc, "prec": prec, "rec": rec, "f1": f1, "n": len(ds)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate correspondence model per city")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--text_embeddings_path", type=Path, required=True)
    parser.add_argument("--data_dirs", type=Path, nargs="+", required=True,
                        help="Directories containing responses/**/predictions.jsonl")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_embeddings = load_text_embeddings(args.text_embeddings_path)
    text_input_dim = next(iter(text_embeddings.values())).shape[0]

    encoder_config = TagBundleEncoderConfig(text_input_dim=text_input_dim, text_proj_dim=128)
    classifier_config = CorrespondenceClassifierConfig(encoder=encoder_config)
    model = CorrespondenceClassifier(classifier_config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from {args.model_path}, device={device}")

    header = (f"{'City':<22s} {'Total':>6s} {'Pos':>5s} {'Hard':>5s} {'Easy':>5s}"
              f" | {'AUC':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}"
              f" | {'AUC_h':>6s} {'P_h':>6s} {'R_h':>6s} {'F1_h':>6s}")
    print(header)
    print("-" * len(header))

    for data_dir in args.data_dirs:
        city_name = data_dir.name
        pairs = load_pairs_from_directory(data_dir)
        if not pairs:
            print(f"{city_name:<22s} NO PAIRS")
            continue

        n_pos = sum(1 for p in pairs if p.difficulty == "positive")
        n_hard = sum(1 for p in pairs if p.difficulty == "hard")
        n_easy = sum(1 for p in pairs if p.difficulty == "easy")

        all_m = eval_dataset(model, pairs, text_embeddings, text_input_dim,
                             ("positive", "easy", "hard"), device)
        hard_m = eval_dataset(model, pairs, text_embeddings, text_input_dim,
                              ("positive", "hard"), device)

        if all_m is None:
            print(f"{city_name:<22s} {len(pairs):6d} {n_pos:5d} {n_hard:5d} {n_easy:5d} | NO DATA")
            continue

        row = f"{city_name:<22s} {all_m['n']:6d} {n_pos:5d} {n_hard:5d} {n_easy:5d}"
        row += f" | {all_m['auc']:6.4f} {all_m['prec']:6.4f} {all_m['rec']:6.4f} {all_m['f1']:6.4f}"
        if hard_m:
            row += f" | {hard_m['auc']:6.4f} {hard_m['prec']:6.4f} {hard_m['rec']:6.4f} {hard_m['f1']:6.4f}"
        else:
            row += f" |    N/A    N/A    N/A    N/A"
        print(row)


if __name__ == "__main__":
    main()
