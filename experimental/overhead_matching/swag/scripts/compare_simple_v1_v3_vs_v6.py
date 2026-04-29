"""One-off comparison: legacy simple_v1_hungarian_0.8 (v3 embeddings, no dustbin)
vs new simple_v1_v6_hungarian_0.8_similarity (v6 embeddings, dustbin on).

Reports recall@{1,5,10} and MRR for each city + the delta. Uses VigorDataset
GT (positive_satellite_idxs + semipositive_satellite_idxs).
"""

import argparse
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.evaluation import retrieval_metrics as rm


CITIES = [
    ("SF_mapillary", "/data/overhead_matching/datasets/VIGOR/mapillary/SanFrancisco_mapillary"),
    ("Boston", "/data/overhead_matching/datasets/VIGOR/Boston"),
    ("Framingham", "/data/overhead_matching/datasets/VIGOR/mapillary/Framingham"),
    ("Norway", "/data/overhead_matching/datasets/VIGOR/mapillary/Norway"),
    ("Seattle", "/data/overhead_matching/datasets/VIGOR/Seattle"),
    ("nightdrive", "/data/overhead_matching/datasets/VIGOR/nightdrive"),
    ("MiamiBeach", "/data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach"),
    ("Middletown", "/data/overhead_matching/datasets/VIGOR/mapillary/Middletown"),
    ("Gap", "/data/overhead_matching/datasets/VIGOR/mapillary/Gap"),
    ("post_hurricane_ian", "/data/overhead_matching/datasets/VIGOR/mapillary/post_hurricane_ian"),
]


LANDMARK_VERSIONS = {
    "/data/overhead_matching/datasets/VIGOR/Boston": None,
    "/data/overhead_matching/datasets/VIGOR/Seattle": "v4_202001",
    "/data/overhead_matching/datasets/VIGOR/nightdrive": None,
}


def load_dataset(dpath: Path, landmark_version: str | None):
    config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=False,
        landmark_version=landmark_version,
    )
    return vd.VigorDataset(dpath, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*", default=None,
                        help="Subset of cities to evaluate (default: all)")
    args = parser.parse_args()

    cities = CITIES
    if args.cities:
        cities = [(n, p) for n, p in CITIES if n in args.cities]

    rows = []
    for city_name, dpath in cities:
        dpath = Path(dpath)
        legacy = dpath / "similarity_matrices" / "simple_v1_hungarian_0.8.pt"
        new = dpath / "similarity_matrices" / "simple_v1_v6_hungarian_0.8_similarity.pt"
        if not legacy.exists() or not new.exists():
            print(f"SKIP {city_name}: legacy={legacy.exists()} new={new.exists()}")
            continue

        print(f"\n=== {city_name} ===")
        lver = LANDMARK_VERSIONS.get(str(dpath), None)
        dataset = load_dataset(dpath, lver)
        print(f"  panos={len(dataset._panorama_metadata)}, "
              f"sats={len(dataset._satellite_metadata)}")

        legacy_sim = torch.load(legacy, weights_only=False, map_location="cpu")
        new_sim = torch.load(new, weights_only=False, map_location="cpu")
        print(f"  legacy shape={tuple(legacy_sim.shape)}, new shape={tuple(new_sim.shape)}")

        legacy_m = rm.compute_top_k_metrics(legacy_sim, dataset)
        new_m = rm.compute_top_k_metrics(new_sim, dataset)
        rows.append((city_name, legacy_m, new_m))

    print("\n" + "=" * 120)
    print(f"{'city':<22} | {'r@1 legacy':>10} {'r@1 v6':>8} {'Δ':>7} | "
          f"{'r@5 legacy':>10} {'r@5 v6':>8} {'Δ':>7} | "
          f"{'r@10 legacy':>11} {'r@10 v6':>9} {'Δ':>7} | "
          f"{'mrr legacy':>10} {'mrr v6':>8} {'Δ':>7}")
    print("-" * 120)
    for city, legacy_m, new_m in rows:
        def fmt(v): return f"{v:.4f}"
        def delta(a, b): return f"{(b - a):+.4f}"
        print(
            f"{city:<22} | "
            f"{fmt(legacy_m['recall@1']):>10} {fmt(new_m['recall@1']):>8} {delta(legacy_m['recall@1'], new_m['recall@1']):>7} | "
            f"{fmt(legacy_m['recall@5']):>10} {fmt(new_m['recall@5']):>8} {delta(legacy_m['recall@5'], new_m['recall@5']):>7} | "
            f"{fmt(legacy_m['recall@10']):>11} {fmt(new_m['recall@10']):>9} {delta(legacy_m['recall@10'], new_m['recall@10']):>7} | "
            f"{fmt(legacy_m['mrr']):>10} {fmt(new_m['mrr']):>8} {delta(legacy_m['mrr'], new_m['mrr']):>7}"
        )


if __name__ == "__main__":
    main()
