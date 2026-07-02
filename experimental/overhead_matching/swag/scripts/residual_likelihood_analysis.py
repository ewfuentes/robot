"""Diagnose the landmark normalized-residual distribution as an OBSERVATION
LIKELIHOOD, comparing true vs negative pairs.

For the filter, what matters is not how well a density fits the true residuals
in isolation, but the *likelihood ratio* L(r) = p(r|true)/p(r|neg): that is the
factor by which a patch's belief is multiplied. We therefore:

  1. Decompose both populations into the discrete endpoint masses
     (r==0  := true patch IS the row argmax; r==1 := patch totally missed)
     and the continuous interior (0,1).
  2. Plot true vs negative densities and the empirical log-likelihood-ratio.
  3. Score candidate observation models by their discriminative separation
     D = E_true[log L] - E_neg[log L]  (mean log-LR gap; bigger = better).

Candidates scored:
  - HalfNormal(sigma_mle)         (the current landmark model)
  - Beta(a,b) method-of-moments   (natural [0,1] support, can be U-shaped)
  - empirical binned LR           (non-parametric ceiling for r as a statistic)
  - zero/one-inflated empirical   (point masses at 0/1 + uniform interior)
"""

from pathlib import Path
import argparse

import common.torch.load_torch_deps  # noqa: F401  (must precede torch import)
import torch
import numpy as np
import matplotlib.pyplot as plt

import experimental.overhead_matching.swag.data.vigor_dataset as vd
from experimental.overhead_matching.swag.filter.adaptive_aggregators import (
    _load_similarity_matrix,
)
from experimental.overhead_matching.swag.scripts.calibrate_sigma import (
    compute_samples,
)

EPS = 1e-9


def _masses(r: np.ndarray):
    """Fraction at exactly 0, exactly 1, and the interior values."""
    at0 = float(np.mean(r <= EPS))
    at1 = float(np.mean(r >= 1.0 - EPS))
    interior = r[(r > EPS) & (r < 1.0 - EPS)]
    return at0, at1, interior


def _halfnormal_logpdf(r, sigma):
    return (np.log(2.0) - 0.5 * np.log(2 * np.pi) - np.log(sigma)
            - 0.5 * (r / sigma) ** 2)


def _beta_mom(x):
    """Method-of-moments Beta(a,b) fit on x in (0,1)."""
    m, v = x.mean(), x.var()
    common = m * (1 - m) / v - 1.0
    return m * common, (1 - m) * common


def _empirical_log_lr(r_true, r_neg, edges):
    """Binned log p(r|true)/p(r|neg) with Laplace smoothing, evaluated per bin."""
    ht, _ = np.histogram(r_true, bins=edges)
    hn, _ = np.histogram(r_neg, bins=edges)
    pt = (ht + 1.0) / (ht.sum() + len(ht))
    pn = (hn + 1.0) / (hn.sum() + len(hn))
    widths = np.diff(edges)
    dt, dn = pt / widths, pn / widths
    return np.log(dt) - np.log(dn), dt, dn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--similarity-matrix-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle/"
                "similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt")
    parser.add_argument(
        "--dataset-path", type=str,
        default="/data/overhead_matching/datasets/VIGOR/Seattle")
    parser.add_argument("--city", type=str, default="Seattle")
    parser.add_argument("--landmark-version", type=str, default="v4_202001")
    parser.add_argument("--num-negative-samples-per-pano", type=int, default=200)
    parser.add_argument(
        "--output-path", type=str,
        default="/data/overhead_matching/evaluation/results/"
                "260501_v6_safa_plus_norm_lm/landmark_residual_exploration/"
                "residual_likelihood_analysis")
    args = parser.parse_args()

    sim_mat = _load_similarity_matrix(Path(args.similarity_matrix_path))
    print(f"matrix {tuple(sim_mat.shape)}")
    dataset_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None, panorama_tensor_cache_info=None,
        should_load_images=False, should_load_landmarks=False,
        landmark_version=args.landmark_version)
    dataset = vd.VigorDataset(Path(args.dataset_path), dataset_config)
    pano_metadata = dataset._panorama_metadata

    rd = compute_samples(
        sim_mat, pano_metadata, include_semipositive=True,
        num_negative_samples_per_pano=args.num_negative_samples_per_pano,
        exclude_constant_rows=True, residual_form="normalized")
    r_true = rd["residuals_pair"].numpy()
    r_neg = rd["residuals_negative"].numpy()

    t0, t1, t_int = _masses(r_true)
    n0, n1, n_int = _masses(r_neg)
    sigma_mle = float(np.sqrt(np.mean(r_true ** 2)))

    print(f"\nn_true={len(r_true):,}  n_neg={len(r_neg):,}")
    print(f"{'':12s}{'r==0':>10s}{'r==1':>10s}{'interior':>10s}")
    print(f"{'true':12s}{t0:10.3f}{t1:10.3f}{1-t0-t1:10.3f}")
    print(f"{'negative':12s}{n0:10.3f}{n1:10.3f}{1-n0-n1:10.3f}")
    print(f"\nsigma_mle (HalfNormal) = {sigma_mle:.4f}")
    if len(t_int) > 50:
        a, b = _beta_mom(t_int)
        print(f"Beta interior MoM (true): a={a:.3f} b={b:.3f}  (U-shape iff a,b<1)")

    # Discrimination score D = E_true[log L] - E_neg[log L] for each model.
    def score(log_l_fn):
        return float(log_l_fn(r_true).mean() - log_l_fn(r_neg).mean())

    hn_D = score(lambda r: _halfnormal_logpdf(np.clip(r, EPS, None), sigma_mle))

    edges = np.concatenate([[0.0, EPS],
                            np.linspace(EPS, 1 - EPS, 40),
                            [1 - EPS, 1.0]])
    edges = np.unique(edges)
    log_lr, dt, dn = _empirical_log_lr(r_true, r_neg, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def emp_log_l(r):
        idx = np.clip(np.searchsorted(edges, r, side="right") - 1,
                      0, len(log_lr) - 1)
        return log_lr[idx]
    emp_D = score(emp_log_l)

    print(f"\nDiscrimination  D = E_true[log L] - E_neg[log L]  (bigger better)")
    print(f"  HalfNormal(sigma={sigma_mle:.3f}) : {hn_D:8.3f}")
    print(f"  empirical binned LR           : {emp_D:8.3f}   <- ceiling for r")

    # ---- figure ----
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # Panel 1: densities, interior only (endpoint masses annotated)
    ax = axes[0]
    ib = np.linspace(0, 1, 41)
    ax.hist(t_int, bins=ib, density=True, color="steelblue", alpha=0.7,
            label=f"true interior")
    ax.hist(n_int, bins=ib, density=True, histtype="step", color="darkorange",
            lw=2, label="negative interior")
    xs = np.linspace(EPS, 1 - EPS, 300)
    ax.plot(xs, np.exp(_halfnormal_logpdf(xs, sigma_mle)), "C3", lw=2,
            label=f"HalfNormal(σ={sigma_mle:.2f})")
    ax.set_title("interior densities (0<r<1)")
    ax.set_xlabel(r"$r_{\rm norm}$"); ax.set_ylabel("density")
    ax.legend(fontsize=9)

    # Panel 2: endpoint masses, true vs negative
    ax = axes[1]
    x = np.arange(3); w = 0.38
    ax.bar(x - w/2, [t0, 1 - t0 - t1, t1], w, color="steelblue", label="true")
    ax.bar(x + w/2, [n0, 1 - n0 - n1, n1], w, color="darkorange", label="negative")
    ax.set_xticks(x); ax.set_xticklabels(["r=0\n(argmax hit)", "0<r<1", "r=1\n(missed)"])
    ax.set_ylabel("probability mass")
    ax.set_title("where the mass is: true vs negative")
    ax.legend()

    # Panel 3: empirical log likelihood ratio vs HalfNormal log-likelihood shape
    ax = axes[2]
    ax.plot(centers, log_lr, color="purple", lw=2, marker=".",
            label="empirical log p(r|true)/p(r|neg)")
    hn_shape = _halfnormal_logpdf(np.clip(centers, EPS, None), sigma_mle)
    ax.plot(centers, hn_shape - hn_shape.mean() + log_lr.mean(), "C3", lw=2,
            ls="--", label="HalfNormal log-lik (shape, shifted)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_title("observation log-likelihood-ratio")
    ax.set_xlabel(r"$r_{\rm norm}$"); ax.set_ylabel("log LR")
    ax.legend(fontsize=9)

    fig.suptitle(
        f"{args.city} landmark residual as observation likelihood   "
        f"(D_halfnormal={hn_D:.2f}  vs  D_empirical={emp_D:.2f})")
    fig.tight_layout()
    out = Path(args.output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
