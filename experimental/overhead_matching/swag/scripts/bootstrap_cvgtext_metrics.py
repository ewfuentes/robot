"""Bootstrap 95% CIs for CVG-Text retrieval metrics (R@1, R@5, R@10, MRR).

Per city, draws B resamples of N query indices with replacement and reuses
the same `(B, N)` index matrix across every method in that city. That makes
the per-iteration bootstrap array for each method paired with every other
method's array in the same city, so any Δ-CI between two methods is just
the difference of their arrays — no precomputed pairwise table required.

All methods are re-scored against CVG-Text filename-based single-positive
GT (the `CVGTextDataset` convention). Exp B similarity matrices are in
VigorDataset row order (sorted VIGOR-staged pano filenames); their per-row
ranks are computed in that order then permuted to CVGTextDataset
(annotation test.json) order so every city shares one canonical query axis.

Outputs:
- `<results_root>/bootstrap_summary.json` — per-method full-sample metrics +
  marginal bootstrap CIs.
- `<results_root>/bootstrap_samples.npz` — `names` (M,), `cities` (M,),
  and one `(M, B)` float32 array per metric. Usage:

    >>> d = np.load(npz)
    >>> i, j = list(d['names']).index(a), list(d['names']).index(b)
    >>> assert d['cities'][i] == d['cities'][j]  # shared draws only in-city
    >>> delta = d['mrr'][i] - d['mrr'][j]
    >>> ci = np.percentile(delta, [2.5, 97.5])
    >>> p_two_sided = 2 * min((delta <= 0).mean(), (delta >= 0).mean())
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import numpy as np
import torch

from experimental.overhead_matching.swag.data.cvgtext_dataset import CVGTextDataset


CITIES = ("Brisbane", "NewYork", "Tokyo")
KS = (1, 5, 10)

_CVGTEXT_FN = re.compile(
    r"^(?P<lat>-?[\d.]+),(?P<lon>-?[\d.]+)_"
    r"(?P<date>\d{4}-\d{2})_"
    r"(?P<pano_id>.+)_"
    r"d(?P<yaw>\d+)_"
    r"z(?P<zoom>\d+)"
    r"\.(?:png|jpg|jpeg)$"
)


def _canonical_key_from_cvgtext(fn: str) -> str:
    m = _CVGTEXT_FN.match(fn)
    if m is None:
        raise ValueError(f"cannot parse CVG-Text filename: {fn!r}")
    g = m.groupdict()
    return f"{g['pano_id']}_{g['date']}_d{g['yaw']}"


def _canonical_key_from_vigor(fn: str) -> str:
    # VIGOR-staged: `<disambig>,<lat>,<lon>,.<ext>` where disambig = `<panoid>_<date>_d<yaw>`.
    return Path(fn).stem.split(",")[0]


@dataclass
class Method:
    name: str
    city: str
    gallery_kind: str          # "satellite" or "OSM"
    backbone: str              # "crosstext2loc" | "expA" | "expB"
    train_city: str | None     # crosstext2loc only
    similarity_path: Path
    metrics_path: Path | None  # stored metrics.json for cross-check (None for expB)
    ordering: str              # "cvgtext" | "vigor"


def discover_methods(results_roots: list[Path]) -> list[Method]:
    """Walk one or more result roots. If a method name already exists from an
    earlier root, the later root's entry is suffixed with its root basename
    to disambiguate (e.g. `expB_Brisbane` vs `expB_Brisbane__cvgtext_repro`).
    """
    out: list[Method] = []
    seen: set[str] = set()
    for root_idx, root in enumerate(results_roots):
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            name = d.name
            suffix = "" if root_idx == 0 else f"__{root.name}"
            if m := re.match(r"^crosstext2loc_train-(\w+)_test-(\w+)_(sat|osm)$", name):
                train_city, test_city, kind = m.group(1), m.group(2), m.group(3)
                entry = Method(
                    name=name + suffix, city=test_city,
                    gallery_kind="satellite" if kind == "sat" else "OSM",
                    backbone="crosstext2loc", train_city=train_city,
                    similarity_path=d / "similarity.pt",
                    metrics_path=d / "metrics.json",
                    ordering="cvgtext",
                )
            elif m := re.match(rf"^expA_test-({'|'.join(CITIES)})_", name):
                entry = Method(
                    name=name + suffix, city=m.group(1), gallery_kind="satellite",
                    backbone="expA", train_city=None,
                    similarity_path=d / "similarity.pt",
                    metrics_path=d / "metrics.json",
                    ordering="cvgtext",
                )
            elif m := re.match(rf"^expB_({'|'.join(CITIES)})$", name):
                entry = Method(
                    name=name + suffix, city=m.group(1), gallery_kind="satellite",
                    backbone="expB", train_city=None,
                    similarity_path=d / "simple_v1_raw_similarity.pt",
                    metrics_path=None,
                    ordering="vigor",
                )
            else:
                continue
            if entry.name in seen:
                print(f"  [skip dup] {entry.name} at {d}")
                continue
            seen.add(entry.name)
            out.append(entry)
    return out


def compute_ranks_argsort(sim: torch.Tensor, gt_idxs: np.ndarray) -> np.ndarray:
    """0-indexed rank of each GT, matching `retrieval_metrics.compute_top_k_metrics`
    semantics (argsort with stable tie-breaking)."""
    rankings = torch.argsort(sim, dim=1, descending=True)
    gt = torch.as_tensor(gt_idxs, dtype=torch.long)
    mask = rankings == gt.unsqueeze(1)
    return mask.long().argmax(dim=1).cpu().numpy().astype(np.int64)


def metrics_from_ranks(ranks: np.ndarray) -> dict[str, float]:
    out = {f"recall@{k}": float((ranks < k).mean()) for k in KS}
    out["mrr"] = float((1.0 / (ranks + 1.0)).mean())
    return out


def bootstrap_samples(ranks: np.ndarray, draws: np.ndarray) -> dict[str, np.ndarray]:
    rr = ranks[draws]  # (B, N)
    out = {f"recall@{k}": (rr < k).mean(axis=1) for k in KS}
    out["mrr"] = (1.0 / (rr + 1.0)).mean(axis=1)
    return out


def percentile_ci(arr: np.ndarray, alpha: float) -> dict:
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def load_canonical_and_vigor(cvgtext_root: Path, vigor_root: Path):
    """Build per-city canonical ordering (from CVGTextDataset) plus a permutation
    that maps VigorDataset row order → canonical row order."""
    canonical_keys_by_city: dict[str, list[str]] = {}
    gt_by_city_kind: dict[tuple[str, str], np.ndarray] = {}
    for city in CITIES:
        for kind in ("satellite", "OSM"):
            ds = CVGTextDataset(root=cvgtext_root, city=city, gallery_kind=kind)
            filenames = ds._panorama_metadata["filename"].tolist()
            gt_idxs = np.array(
                [r.positive_satellite_idxs[0] for _, r in ds._panorama_metadata.iterrows()],
                dtype=np.int64,
            )
            if kind == "satellite":
                canonical_keys_by_city[city] = [_canonical_key_from_cvgtext(fn) for fn in filenames]
            gt_by_city_kind[(city, kind)] = gt_idxs

    vigor_perm_by_city: dict[str, np.ndarray] = {}
    vigor_gt_by_city: dict[str, np.ndarray] = {}
    for city in CITIES:
        staged = vigor_root / f"cvgtext_{city}"
        pano_files = sorted((staged / "panorama").iterdir())
        sat_files = sorted((staged / "satellite").iterdir())

        vigor_keys = [_canonical_key_from_vigor(p.name) for p in pano_files]
        key_to_vigor_idx = {k: i for i, k in enumerate(vigor_keys)}
        canon = canonical_keys_by_city[city]
        missing = [k for k in canon if k not in key_to_vigor_idx]
        if missing:
            raise RuntimeError(
                f"{len(missing)} canonical keys missing from VIGOR panorama dir "
                f"for {city} (e.g. {missing[:3]}). Run stage_cvgtext_for_vigor.py."
            )
        vigor_perm_by_city[city] = np.array([key_to_vigor_idx[k] for k in canon], dtype=np.int64)

        sat_latlon_to_idx: dict[tuple[str, str], int] = {}
        for i, sp in enumerate(sat_files):
            parts = sp.stem.split("_")
            sat_latlon_to_idx[(parts[-2], parts[-1])] = i
        vgt = np.zeros(len(pano_files), dtype=np.int64)
        for i, pp in enumerate(pano_files):
            parts = pp.name.split(",")
            lat_str, lon_str = parts[1], parts[2]
            if (lat_str, lon_str) not in sat_latlon_to_idx:
                raise RuntimeError(
                    f"VIGOR pano {pp.name} has no matching staged sat tile at "
                    f"lat={lat_str} lon={lon_str}."
                )
            vgt[i] = sat_latlon_to_idx[(lat_str, lon_str)]
        vigor_gt_by_city[city] = vgt

    return canonical_keys_by_city, gt_by_city_kind, vigor_perm_by_city, vigor_gt_by_city


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results_root", type=Path,
                    default=Path("/data/overhead_matching/evaluation/results/cvgtext"))
    ap.add_argument("--extra_roots", type=Path, nargs="*", default=[],
                    help="Additional results dirs to include; duplicate method names "
                         "get suffixed with the root basename.")
    ap.add_argument("--cvgtext_root", type=Path,
                    default=Path("/data/overhead_matching/datasets/cvgtext"))
    ap.add_argument("--vigor_root", type=Path,
                    default=Path("/data/overhead_matching/datasets/VIGOR"))
    ap.add_argument("--output", type=Path, default=None,
                    help="JSON summary path (default: <results_root>/bootstrap_summary.json)")
    ap.add_argument("--samples_output", type=Path, default=None,
                    help=".npz raw samples path (default: <results_root>/bootstrap_samples.npz)")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_json = args.output or (args.results_root / "bootstrap_summary.json")
    out_npz = args.samples_output or (args.results_root / "bootstrap_samples.npz")

    print("Building canonical query ordering from CVGTextDataset + VIGOR staging")
    canon_keys, gt_cvgtext, vigor_perm, vigor_gt = load_canonical_and_vigor(
        args.cvgtext_root, args.vigor_root,
    )
    n_by_city = {c: len(canon_keys[c]) for c in CITIES}
    for c in CITIES:
        print(f"  {c}: N={n_by_city[c]} canonical queries")

    methods = discover_methods([args.results_root, *args.extra_roots])
    print(f"Discovered {len(methods)} similarity matrices")

    rng = np.random.default_rng(args.seed)
    draws_by_city = {
        c: rng.integers(0, n_by_city[c], size=(args.B, n_by_city[c]), dtype=np.int64)
        for c in CITIES
    }

    names: list[str] = []
    cities: list[str] = []
    per_method_summary: dict[str, dict] = {}
    per_method_samples: dict[str, dict[str, np.ndarray]] = {}

    for m in methods:
        if not m.similarity_path.exists():
            print(f"  [skip] missing {m.similarity_path}")
            continue
        sim = torch.load(m.similarity_path, map_location="cpu", weights_only=True)
        if not isinstance(sim, torch.Tensor):
            print(f"  [skip] non-tensor similarity at {m.similarity_path}")
            continue
        sim = sim.float()

        if m.ordering == "cvgtext":
            gt = gt_cvgtext[(m.city, m.gallery_kind)]
            if sim.shape[0] != len(gt):
                print(f"  [skip] {m.name}: sim rows {sim.shape[0]} != N={len(gt)}")
                continue
            ranks = compute_ranks_argsort(sim, gt)
        else:
            gt = vigor_gt[m.city]
            if sim.shape[0] != len(gt):
                print(f"  [skip] {m.name}: sim rows {sim.shape[0]} != VIGOR N={len(gt)}")
                continue
            ranks = compute_ranks_argsort(sim, gt)[vigor_perm[m.city]]

        full = metrics_from_ranks(ranks)

        cross_check = None
        if m.metrics_path is not None and m.metrics_path.exists():
            stored = json.loads(m.metrics_path.read_text())
            deltas = {k: abs(stored[k] - full[k]) for k in full if k in stored}
            cross_check = {"stored": stored, "max_delta": max(deltas.values(), default=0.0)}
            if cross_check["max_delta"] > 1e-3:
                print(f"  [warn] {m.name}: full-sample vs stored max Δ={cross_check['max_delta']:.4f}")

        print(f"  {m.name}: R@1={full['recall@1']:.4f} R@5={full['recall@5']:.4f} "
              f"R@10={full['recall@10']:.4f} MRR={full['mrr']:.4f} (gallery={sim.shape[1]})")

        samples = bootstrap_samples(ranks, draws_by_city[m.city])
        names.append(m.name)
        cities.append(m.city)
        per_method_samples[m.name] = samples
        per_method_summary[m.name] = {
            "city": m.city, "gallery_kind": m.gallery_kind, "backbone": m.backbone,
            "train_city": m.train_city, "n_gallery": int(sim.shape[1]),
            "n_queries": n_by_city[m.city],
            "full_sample": full,
            "cross_check": cross_check,
            "bootstrap_ci": {k: percentile_ci(v, args.alpha) for k, v in samples.items()},
        }

    summary = {
        "config": {
            "B": args.B, "seed": args.seed, "alpha": args.alpha, "ks": list(KS),
            "n_per_city": n_by_city,
            "results_root": str(args.results_root),
            "cvgtext_root": str(args.cvgtext_root),
            "vigor_root": str(args.vigor_root),
        },
        "methods": per_method_summary,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}  ({len(names)} methods)")

    npz_payload: dict[str, np.ndarray] = {
        "names": np.array(names),
        "cities": np.array(cities),
    }
    for metric in (*[f"recall@{k}" for k in KS], "mrr"):
        npz_payload[metric] = np.stack(
            [per_method_samples[n][metric] for n in names]
        ).astype(np.float32)
    np.savez_compressed(out_npz, **npz_payload)
    print(f"Wrote {out_npz}  (arrays shape (M={len(names)}, B={args.B}))")


if __name__ == "__main__":
    main()
