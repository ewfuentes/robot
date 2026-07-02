"""Calibrate the piecewise log-LR landmark observation model from ALL matrix
entries (no Monte-Carlo negative sampling) and report discrimination D.

The landmark stream of ``SafaPlusPiecewiseLandmarkAggregator`` maps the
normalized residual ``r = 1 - sim/row_max`` through a piecewise-constant lookup
``g(r) = log p(r|true) - log p(r|negative)`` over uniform bins on [0, 1].

Negatives are *every* (pano, patch) residual in the matrix (true patches
subtracted out exactly), so ``p(r|negative)`` is exact rather than a sampled
estimate. The discrete r==0 (argmax hit) and r==1 (miss) events fall in the
first / last bin — no special-casing.

Discrimination  D = E_true[g] - E_neg[g]  measures the mean belief-update gap
the lookup gives true vs negative patches (bigger = better; it equals the
empirical ceiling when g is the binned empirical LR).

Reports, for every eval city: D_self (the city's own lookup) and D_transfer
(the frozen reference-city lookup applied to that city) — the latter is what a
single frozen calibration actually achieves. Writes the reference city's
calibration JSON for ``SafaPlusPiecewiseLandmarkAggregatorConfig``.
"""

from pathlib import Path
import argparse
import json

import common.torch.load_torch_deps  # noqa: F401
import torch
import numpy as np

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)

# city -> (dataset_path, landmark_version). Matrix path is derived as
# <dataset>/similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt.
CITY_INFO = {
    "Seattle": ("/data/overhead_matching/datasets/VIGOR/Seattle", "v4_202001"),
    "NewYork": ("/data/overhead_matching/datasets/VIGOR/NewYork", "v4_202001"),
    "Boston": ("/data/overhead_matching/datasets/VIGOR/Boston", "boston"),
    "Framingham": ("/data/overhead_matching/datasets/VIGOR/mapillary/Framingham", "Framingham_v1_260101"),
    "Middletown": ("/data/overhead_matching/datasets/VIGOR/mapillary/Middletown", "Middletown_v1_250101"),
    "Norway": ("/data/overhead_matching/datasets/VIGOR/mapillary/Norway", "Norway_v1_251201"),
    "nightdrive": ("/data/overhead_matching/datasets/VIGOR/nightdrive", "boston"),
    "post_hurricane_ian_sw": ("/data/overhead_matching/datasets/VIGOR/mapillary/post_hurricane_ian_sw", "post_hurricane_ian_sw_v1_220101"),
    "SanFrancisco_mapillary": ("/data/overhead_matching/datasets/VIGOR/mapillary/SanFrancisco_mapillary", "SanFrancisco_mapillary_v1_220101"),
}
MATRIX_REL = "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt"


def _bin_idx(r: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Bin residuals into 0..n_bins-1 using explicit edges (handles overflow)."""
    inner = edges[1:-1].contiguous()
    return torch.clamp(torch.bucketize(r, inner, right=False), 0, edges.numel() - 2)


def _residual(M, rmax, residual_form):
    """r = rowmax - sim  (raw, image stream)  or  1 - sim/rowmax  (normalized)."""
    if residual_form == "raw":
        return rmax[:, None] - M
    safe = torch.where(rmax > 0, rmax, torch.ones_like(rmax))
    return 1.0 - M / safe[:, None]


def compute(dataset_path, landmark_version, matrix_path, edges,
            include_semipositive, device, residual_form="normalized",
            chunk_rows=1024):
    """Return (true_hist, neg_hist) bin counts over all valid rows.

    ``edges`` is a 1-D tensor of n_bins+1 bin edges. ``residual_form`` selects
    ``raw`` (rowmax-sim, the image stream) or ``normalized`` (1-sim/rowmax, the
    landmark stream). A row is excluded if it has no finite entries, is constant,
    or (normalized only) row_max <= 0 — matching the image-only fall-through.
    neg_hist = all-finite-entry hist minus true-pair hist.
    """
    sim = _load_similarity_matrix(Path(matrix_path))
    cfg = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None, panorama_tensor_cache_info=None,
        should_load_images=False, should_load_landmarks=False,
        landmark_version=landmark_version)
    ds = vd.VigorDataset(Path(dataset_path), cfg)
    meta = ds._panorama_metadata
    assert sim.shape[0] == len(meta), (sim.shape, len(meta))
    n_rows = sim.shape[0]
    edges = torch.as_tensor(edges, dtype=torch.float64)
    n_bins = edges.numel() - 1
    edges_d = edges.to(device)

    row_max = torch.empty(n_rows, dtype=torch.float64)
    valid = torch.empty(n_rows, dtype=torch.bool)
    all_hist = torch.zeros(n_bins, dtype=torch.float64)

    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        M = sim[r0:r1].to(device).double()
        finite = torch.isfinite(M)
        Mf_hi = torch.where(finite, M, torch.full_like(M, float("-inf")))
        Mf_lo = torch.where(finite, M, torch.full_like(M, float("inf")))
        rmax = Mf_hi.max(dim=1).values
        rmin = Mf_lo.min(dim=1).values
        v = torch.isfinite(rmax) & (rmax != rmin)
        if residual_form != "raw":
            v = v & (rmax > 0)
        row_max[r0:r1] = rmax.cpu()
        valid[r0:r1] = v.cpu()
        r = _residual(M, rmax, residual_form)
        sel = finite & v[:, None]
        if sel.any():
            idx = _bin_idx(r[sel], edges_d)
            all_hist += torch.bincount(idx.cpu(), minlength=n_bins).double()

    # true pairs
    true_rows, true_cols = [], []
    for i in range(n_rows):
        row = meta.iloc[i]
        idxs = list(row["positive_satellite_idxs"])
        if include_semipositive:
            idxs += list(row["semipositive_satellite_idxs"])
        for c in idxs:
            true_rows.append(i)
            true_cols.append(int(c))
    tr = torch.tensor(true_rows)
    tc = torch.tensor(true_cols)
    tsim = sim[tr, tc].double()
    keep = valid[tr] & torch.isfinite(tsim)
    tr, tc, tsim = tr[keep], tc[keep], tsim[keep]
    t_rmax = row_max[tr]
    t_r = (t_rmax - tsim) if residual_form == "raw" else (1.0 - tsim / t_rmax)
    true_hist = torch.bincount(_bin_idx(t_r, edges), minlength=n_bins).double()

    neg_hist = torch.clamp(all_hist - true_hist, min=0.0)
    return true_hist, neg_hist


def reference_raw_upper(dataset_path, landmark_version, matrix_path,
                        include_semipositive, device, q=0.995, chunk_rows=1024):
    """Upper raw-residual bin edge = q-quantile of the reference city's TRUE
    (rowmax - sim) residuals, so bins resolve where true mass lives."""
    sim = _load_similarity_matrix(Path(matrix_path))
    cfg = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None, panorama_tensor_cache_info=None,
        should_load_images=False, should_load_landmarks=False,
        landmark_version=landmark_version)
    ds = vd.VigorDataset(Path(dataset_path), cfg)
    meta = ds._panorama_metadata
    n_rows = sim.shape[0]
    row_max = torch.empty(n_rows, dtype=torch.float64)
    valid = torch.empty(n_rows, dtype=torch.bool)
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        M = sim[r0:r1].to(device).double()
        finite = torch.isfinite(M)
        rmax = torch.where(finite, M, torch.full_like(M, float("-inf"))).max(dim=1).values
        rmin = torch.where(finite, M, torch.full_like(M, float("inf"))).min(dim=1).values
        row_max[r0:r1] = rmax.cpu()
        valid[r0:r1] = (torch.isfinite(rmax) & (rmax != rmin)).cpu()
    true_rows, true_cols = [], []
    for i in range(n_rows):
        row = meta.iloc[i]
        idxs = list(row["positive_satellite_idxs"])
        if include_semipositive:
            idxs += list(row["semipositive_satellite_idxs"])
        for c in idxs:
            true_rows.append(i)
            true_cols.append(int(c))
    tr = torch.tensor(true_rows)
    tc = torch.tensor(true_cols)
    tsim = sim[tr, tc].double()
    keep = valid[tr] & torch.isfinite(tsim)
    t_r = row_max[tr][keep] - tsim[keep]
    return float(torch.quantile(t_r, q))


def build_lookup(true_hist, neg_hist, n_bins, clip):
    nt, nn = float(true_hist.sum()), float(neg_hist.sum())
    pt = (true_hist + 1.0) / (nt + n_bins)
    pn = (neg_hist + 1.0) / (nn + n_bins)
    values = torch.clamp(torch.log(pt) - torch.log(pn), -clip, clip)
    return values.numpy()


def discrimination(values, true_hist, neg_hist):
    ft = (true_hist / true_hist.sum()).numpy()
    fn = (neg_hist / neg_hist.sum()).numpy()
    return float(np.sum((ft - fn) * values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--clip", type=float, default=6.0)
    p.add_argument("--reference-city", default="Seattle")
    p.add_argument("--include-semipositive", type=lambda s: s.lower() != "false", default=True)
    p.add_argument("--image-sigma", type=float, default=0.1809)
    p.add_argument("--matrix-rel", default=MATRIX_REL,
                   help="similarity matrix path relative to each dataset dir "
                        "(e.g. similarity_matrices/wag_no_hinge.pt)")
    p.add_argument("--residual-form", choices=["normalized", "raw"],
                   default="normalized",
                   help="raw = rowmax-sim (image stream); "
                        "normalized = 1-sim/rowmax (landmark stream)")
    p.add_argument("--raw-upper", type=float, default=None,
                   help="raw mode upper bin edge; default = 0.995 quantile of "
                        "reference-city true residuals")
    p.add_argument("--output-path", required=True, help="reference calibration JSON")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.residual_form == "raw":
        rdpath, rlv = CITY_INFO[args.reference_city]
        upper = args.raw_upper if args.raw_upper is not None else reference_raw_upper(
            rdpath, rlv, Path(rdpath) / args.matrix_rel,
            args.include_semipositive, device)
        print(f"raw residual: upper bin edge = {upper:.4f} "
              f"({'given' if args.raw_upper is not None else '0.995 quantile of ' + args.reference_city + ' true'})")
        edges = list(np.linspace(0.0, upper, args.n_bins + 1))
    else:
        edges = list(np.linspace(0.0, 1.0, args.n_bins + 1))
    edges_t = torch.as_tensor(edges, dtype=torch.float64)

    hists, lookups = {}, {}
    for city, (dpath, lv) in CITY_INFO.items():
        mpath = Path(dpath) / args.matrix_rel
        print(f"[{city}] {mpath}")
        th, nh = compute(dpath, lv, mpath, edges_t,
                         args.include_semipositive, device, args.residual_form)
        hists[city] = (th, nh)
        lookups[city] = build_lookup(th, nh, args.n_bins, args.clip)

    ref = args.reference_city
    ref_lookup = lookups[ref]

    print(f"\nReference = {ref} ({args.n_bins} uniform bins, all-entry negatives)\n")
    print(f"{'city':24s}{'D_self':>9s}{'D_transfer':>12s}{'n_true':>9s}")
    print("-" * 54)
    rows = {}
    for city in CITY_INFO:
        th, nh = hists[city]
        d_self = discrimination(lookups[city], th, nh)
        d_trans = discrimination(ref_lookup, th, nh)
        rows[city] = {"D_self": d_self, "D_transfer": d_trans,
                      "n_true": int(th.sum()), "n_neg": int(nh.sum())}
        print(f"{city:24s}{d_self:9.3f}{d_trans:12.3f}{int(th.sum()):9d}")

    out = {
        "n_bins": args.n_bins, "clip": args.clip, "reference_city": ref,
        "image_sigma": args.image_sigma, "matrix_rel": args.matrix_rel,
        "residual_form": args.residual_form,
        "calibration": {
            "landmark_log_lr_edges": [float(e) for e in edges],
            "landmark_log_lr_values": [float(v) for v in ref_lookup],
        },
        "reference_lookup_values": [float(v) for v in ref_lookup],
        "per_city_D": rows,
    }
    outp = Path(args.output_path).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
