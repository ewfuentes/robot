import common.torch.load_torch_deps
import torch
import torchvision as tv
import numpy as np
import pandas as pd
import json
import pickle
import argparse
import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

import common.torch.load_and_save_models as lsm
import experimental.overhead_matching.swag.data.vigor_dataset as vd
import experimental.overhead_matching.swag.evaluation.evaluate_swag as es
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding

import msgspec


def load_model(path, device="cuda"):
    try:
        model = lsm.load_model(path, device=device, skip_consistent_output_check=True)
        model.patch_dims
        model.model_input_from_batch
    except Exception as e:
        print("Failed to load model via lsm, falling back to config+weights:", e)
        training_config_path = path.parent / "train_config.yaml"
        training_config_bytes = training_config_path.read_bytes()
        training_config = msgspec.yaml.decode(training_config_bytes, type=dict)
        model_config_json = (
            training_config["sat_model_config"]
            if "satellite" in path.name
            else training_config["pano_model_config"]
        )
        config = msgspec.json.decode(
            msgspec.json.encode(model_config_json),
            type=patch_embedding.WagPatchEmbeddingConfig
            | swag_patch_embedding.SwagPatchEmbeddingConfig,
        )

        model_weights = torch.load(path / "model_weights.pt", weights_only=True)
        model_type = (
            patch_embedding.WagPatchEmbedding
            if isinstance(config, patch_embedding.WagPatchEmbeddingConfig)
            else swag_patch_embedding.SwagPatchEmbedding
        )
        model = model_type(config)
        model.load_state_dict(model_weights)
        model = model.to(device)
    return model


def compute_rank_labels(top_k_df, rank_threshold):
    """Compute per-panorama binary labels based on retrieval rank.

    For each panorama, take the minimum k_value across all positive + semipositive
    matches. Label "bad" = 1 if min rank >= rank_threshold, "good" = 0 otherwise.

    Returns:
        ranks: array of min rank per panorama (indexed by pano_idx)
        labels: binary array (1 = bad, 0 = good)
        num_panos: total number of panoramas
    """
    min_rank_per_pano = top_k_df.groupby("pano_idx")["k_value"].min()
    num_panos = min_rank_per_pano.index.max() + 1

    ranks = np.full(num_panos, -1, dtype=np.int64)
    labels = np.zeros(num_panos, dtype=np.int64)
    for pano_idx, min_rank in min_rank_per_pano.items():
        ranks[pano_idx] = min_rank
        labels[pano_idx] = 1 if min_rank >= rank_threshold else 0

    return ranks, labels


def extract_dino_cls_tokens(dataset, dino_model, device, batch_size):
    """Extract raw DINO CLS tokens for all panoramas in the dataset."""
    pano_view = dataset.get_pano_view()
    dataloader = vd.get_dataloader(pano_view, batch_size=batch_size, num_workers=8, shuffle=False)

    dino_model = dino_model.to(device)
    dino_model.eval()

    all_cls = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in tqdm.tqdm(dataloader, desc="Extracting DINO CLS tokens"):
            images = batch.panorama.to(device)
            # Normalize with ImageNet stats (same as patch_embedding.py)
            images = tv.transforms.functional.normalize(
                images,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            # DINO forward returns CLS token: (batch, embedding_dim)
            cls_tokens = dino_model(images)
            all_cls.append(cls_tokens.float().cpu())

    return torch.cat(all_cls, dim=0)


def evaluate_classifier(clf, scaler, X, y, split_name):
    """Evaluate a classifier and return metrics dict."""
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "num_samples": len(y),
        "num_bad": int(y.sum()),
        "num_good": int((y == 0).sum()),
    }
    return metrics


def load_city_dataset(city_name, dataset_path):
    """Load a VIGOR dataset for one city."""
    print(f"\n{'='*60}")
    print(f"Loading {city_name}")
    print(f"{'='*60}")

    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        satellite_patch_size=(322, 322),
        panorama_size=(322, 644),
        should_load_landmarks=False,
    )
    dataset = vd.VigorDataset(dataset_path, config)
    num_panos = len(dataset._panorama_metadata)
    num_sats = len(dataset._satellite_metadata)
    print(f"{city_name}: {num_panos} panoramas, {num_sats} satellite patches")
    return dataset


def compute_city_ranks(city_name, dataset, all_similarity, rank_threshold):
    """Compute retrieval ranks and labels for one city using a precomputed similarity matrix."""
    print(f"Computing retrieval ranks for {city_name} (on CPU)...")
    rankings = torch.argsort(all_similarity, dim=1, descending=True)

    out = []
    for i, row in dataset._panorama_metadata.iterrows():
        for sat_idx in row.positive_satellite_idxs:
            k_value = torch.argwhere(rankings[i] == sat_idx)
            out.append({"pano_idx": i, "sat_idx": sat_idx,
                        "match_type": "positive", "k_value": k_value.item()})
        for sat_idx in row.semipositive_satellite_idxs:
            k_value = torch.argwhere(rankings[i] == sat_idx)
            out.append({"pano_idx": i, "sat_idx": sat_idx,
                        "match_type": "semipositive", "k_value": k_value.item()})
    top_k_df = pd.DataFrame.from_records(out)

    ranks, labels = compute_rank_labels(top_k_df, rank_threshold)
    print(f"{city_name}: {labels.sum()}/{len(labels)} panoramas labeled as 'bad' "
          f"(rank >= {rank_threshold}), "
          f"{(labels == 0).sum()} labeled as 'good'")
    return ranks, labels


def extract_and_save_city_cls(city_name, dataset, dino_model, batch_size, device,
                              output_path, ranks, labels):
    """Extract DINOv2 CLS tokens, save per-city data, return numpy arrays."""
    num_panos = len(dataset._panorama_metadata)
    print(f"Extracting DINOv2 CLS tokens for {city_name}...")
    cls_tokens = extract_dino_cls_tokens(dataset, dino_model, device, batch_size)
    print(f"{city_name}: CLS token shape = {cls_tokens.shape}")
    assert cls_tokens.shape[0] == num_panos

    city_output = output_path / city_name.lower()
    city_output.mkdir(parents=True, exist_ok=True)
    torch.save(cls_tokens, city_output / "cls_tokens.pt")
    np.save(city_output / "ranks.npy", ranks)
    np.save(city_output / "labels.npy", labels)

    return cls_tokens.numpy(), labels, ranks


def main():
    parser = argparse.ArgumentParser(
        description="Classify bad performance of DINO-based satellite-panorama matching"
    )
    parser.add_argument("--model-base-path", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--dataset-base", type=str,
                        default="/data/overhead_matching/datasets/VIGOR/",
                        help="VIGOR dataset root")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Directory for results")
    parser.add_argument("--rank-threshold", type=int, default=10,
                        help="K threshold for bad/good labeling")
    parser.add_argument("--dino-model", type=str, default="dinov2_vitb14",
                        help="DINO model name (e.g. dinov2_vitb14, dinov3_vitb16)")
    parser.add_argument("--batch-size", type=int, default=96,
                        help="Batch size for inference")
    parser.add_argument("--checkpoint", type=str, default="best",
                        help='Which checkpoint to use: "best" or epoch number')
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device")
    parser.add_argument("--train-cities", type=str, nargs="+", default=["Chicago"],
                        help="City or cities to train classifier on")
    parser.add_argument("--test-city", type=str, default="Seattle",
                        help="City to evaluate classifier on")

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    model_base = Path(args.model_base_path)
    dataset_base = Path(args.dataset_base)
    device = args.device
    train_cities = args.train_cities
    test_city = args.test_city
    train_cities_label = "+".join(train_cities)

    # Step 1: Load trained sat/pano models
    if args.checkpoint == "best":
        sat_path = model_base / "best_satellite"
        pano_path = model_base / "best_panorama"
    else:
        sat_path = model_base / f"satellite_epoch_{args.checkpoint}"
        pano_path = model_base / f"panorama_epoch_{args.checkpoint}"

    print("Loading satellite model from", sat_path)
    sat_model = load_model(sat_path, device=device)
    sat_model.eval()
    print("Loading panorama model from", pano_path)
    pano_model = load_model(pano_path, device=device)
    pano_model.eval()

    # Step 2: Load all datasets (train cities + test city)
    all_cities = list(dict.fromkeys(train_cities + [test_city]))  # deduplicate, preserve order
    datasets = {}
    for city in all_cities:
        datasets[city] = load_city_dataset(city, dataset_base / city)

    # Step 3: Compute similarity matrices on GPU, then free models
    print("\nComputing similarity matrices...")
    similarities = {}
    for city in all_cities:
        similarities[city] = es.compute_cached_similarity_matrix(
            sat_model=sat_model, pano_model=pano_model,
            dataset=datasets[city], device=device, use_cached_similarity=True)

    del sat_model, pano_model
    torch.cuda.empty_cache()

    # Compute ranks on CPU to avoid OOM from argsort
    city_ranks = {}
    city_labels = {}
    for city in all_cities:
        ranks, labels = compute_city_ranks(
            city, datasets[city], similarities[city].cpu(), args.rank_threshold)
        city_ranks[city] = ranks
        city_labels[city] = labels
        del similarities[city]

    # Step 4: Extract DINOv2 CLS tokens (separate phase to avoid OOM)
    dino_repo = "facebookresearch/dinov3" if args.dino_model.startswith("dinov3") else "facebookresearch/dinov2"
    print(f"\nLoading DINO model: {args.dino_model} from {dino_repo}")
    dino_model = torch.hub.load(dino_repo, args.dino_model)
    dino_model = dino_model.to(device)
    dino_model.eval()

    city_X = {}
    for city in all_cities:
        X, labels, ranks = extract_and_save_city_cls(
            city, datasets[city], dino_model, args.batch_size, device,
            output_path, city_ranks[city], city_labels[city])
        city_X[city] = X
        city_labels[city] = labels
        city_ranks[city] = ranks

    del dino_model
    torch.cuda.empty_cache()

    # Combine training data from all train cities
    train_X = np.concatenate([city_X[c] for c in train_cities], axis=0)
    train_y = np.concatenate([city_labels[c] for c in train_cities], axis=0)
    test_X = city_X[test_city]
    test_y = city_labels[test_city]

    # Step 5: Train classifier
    print("\n" + "=" * 60)
    print(f"Training logistic regression on {train_cities_label}")
    print(f"  Total train samples: {len(train_y)} "
          f"({int(train_y.sum())} bad, {int((train_y == 0).sum())} good)")
    print("=" * 60)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)

    # 5-fold cross-validation for AUC estimate
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    cv_scores = cross_val_score(
        clf, train_X_scaled, train_y, cv=5, scoring="roc_auc"
    )
    print(f"{train_cities_label} 5-fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on full train set
    clf.fit(train_X_scaled, train_y)

    # Step 6: Evaluate on train (in-sample) and test (out-of-sample)
    train_metrics = evaluate_classifier(clf, scaler, train_X, train_y, f"{train_cities_label}_train")
    test_metrics = evaluate_classifier(clf, scaler, test_X, test_y, f"{test_city}_test")

    # Per-city train metrics
    per_city_train_metrics = {}
    for city in train_cities:
        per_city_train_metrics[city] = evaluate_classifier(
            clf, scaler, city_X[city], city_labels[city], f"{city}_train")

    print("\n" + "=" * 60)
    print(f"{train_cities_label} (train, combined) metrics:")
    for k, v in train_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix:\n    {train_metrics['confusion_matrix']}")

    for city in train_cities:
        m = per_city_train_metrics[city]
        print(f"\n  {city} (train, individual):")
        print(f"    accuracy={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
              f"roc_auc={m['roc_auc']:.4f}  bad={m['num_bad']}/{m['num_samples']}")

    print(f"\n{test_city} (test) metrics:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"  confusion_matrix:\n    {test_metrics['confusion_matrix']}")
    print("=" * 60)

    # Step 7: Save results
    all_metrics = {
        "train_cities": train_cities,
        "test_city": test_city,
        f"{train_cities_label.lower()}_cv_auc_mean": float(cv_scores.mean()),
        f"{train_cities_label.lower()}_cv_auc_std": float(cv_scores.std()),
        f"{train_cities_label.lower()}_cv_auc_per_fold": cv_scores.tolist(),
        f"{train_cities_label.lower()}_train": train_metrics,
        f"{test_city.lower()}_test": test_metrics,
        "rank_threshold": args.rank_threshold,
    }
    for city in train_cities:
        all_metrics[f"{city.lower()}_train"] = per_city_train_metrics[city]

    with open(output_path / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    with open(output_path / "classifier.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
