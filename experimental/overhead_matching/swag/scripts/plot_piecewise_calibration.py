"""Plot the calibrated piecewise landmark log-LR lookup over the true / negative
normalized-residual densities it was fit to.

Single-city mode (``--city``): one figure, top panel = per-bin densities of
r = 1 - sim/row_max for true (pano,patch) pairs vs all-entry negatives, bottom
panel = the piecewise-constant log-LR g(r) = log p(r|true) - log p(r|neg).

All-cities mode (``--all-cities``): a grid with one column per city. Each
column overlays, in the bottom panel, the city's OWN calibrated log-LR
(``D_self``) against the FROZEN reference-city lookup that the filter actually
applies (``D_transfer``). The gap between those two curves -- weighted by where
the city's *negative* mass sits -- is what makes a frozen lookup over- or
under-confident on that city.

Also prints a per-city "negative leakage" diagnostic: ``E_neg[g_ref]`` (the mean
applied log-LR handed to negative/wrong patches) and the fraction of negative
mass sitting in bins where the applied lookup is positive (evidence *for* a
wrong patch). High values predict filter divergence at ``landmark_lr_scale=1``.
"""
from pathlib import Path
import argparse
import json

import common.torch.load_torch_deps  # noqa: F401
import torch
import numpy as np
import matplotlib.pyplot as plt

from experimental.overhead_matching.swag.scripts.calibrate_landmark_piecewise import (
    compute, build_lookup, discrimination, CITY_INFO, MATRIX_REL,
)


def _densities(th, nh, widths):
    dt = (th / th.sum()) / widths
    dn = (nh / nh.sum()) / widths
    return dt, dn


def _leakage(values_applied, neg_hist):
    """Diagnostics for how much the applied lookup boosts negative patches."""
    fn = (neg_hist / neg_hist.sum()).numpy()
    e_neg = float(np.sum(fn * values_applied))            # mean applied log-LR to negatives
    pos_mass = float(np.sum(fn[values_applied > 0]))      # neg mass where lookup says "true"
    return e_neg, pos_mass


def _plot_city(ax1, ax2, city, edges, widths, n_bins, ref_values, ref_city, device,
               matrix_rel=MATRIX_REL, residual_form="normalized"):
    dpath, lv = CITY_INFO[city]
    mpath = Path(dpath) / matrix_rel
    th, nh = compute(dpath, lv, str(mpath), edges, True, device, residual_form)
    own_values = build_lookup(th, nh, n_bins, clip=6.0)
    d_self = discrimination(own_values, th, nh)
    d_trans = discrimination(ref_values, th, nh)
    dt, dn = _densities(th.numpy(), nh.numpy(), widths)

    ax1.step(edges, np.concatenate([[dt[0]], dt]), where="pre", color="steelblue",
             lw=2, label=f"true (n={int(th.sum()):,})")
    ax1.step(edges, np.concatenate([[dn[0]], dn]), where="pre", color="darkorange",
             lw=2, label=f"negative (n={int(nh.sum()):,})")
    ax1.set_yscale("log")
    ax1.set_title(f"{city}", fontweight="bold")
    ax1.legend(loc="lower center", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.step(edges, np.concatenate([[own_values[0]], own_values]), where="pre",
             color="C0", lw=2.0, label=f"own  (D={d_self:.2f})")
    ax2.step(edges, np.concatenate([[ref_values[0]], ref_values]), where="pre",
             color="C3", lw=2.2, ls="--", label=f"{ref_city} applied (D={d_trans:.2f})")
    ax2.axhline(0, color="k", lw=0.9, ls=":")
    ax2.set_xlim(edges[0], edges[-1])
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25)

    e_neg, pos_mass = _leakage(ref_values, nh)
    tf = (th / th.sum()).numpy()
    lo = int(round(0.1 * n_bins))   # bins with r < 0.1
    hi = n_bins - lo                # bins with r > 0.9
    return {
        "city": city, "D_self": d_self, "D_transfer": d_trans,
        "n_true": int(th.sum()), "n_neg": int(nh.sum()),
        "E_neg_applied": e_neg, "neg_mass_in_positive_g": pos_mass,
        "true_mass_lo_r": float(tf[:lo].sum()),   # true patches the lookup rewards (g>0)
        "true_mass_hi_r": float(tf[hi:].sum()),   # true patches in the miss/penalized region
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default="Seattle")
    p.add_argument("--all-cities", action="store_true",
                   help="render a grid over every city in CITY_INFO")
    p.add_argument("--calibration", default="/tmp/zoi_proto/piecewise_calibration_seattle.json")
    p.add_argument("--output-path", default="/tmp/zoi_proto/piecewise_calibration_seattle_plot")
    p.add_argument("--matrix-rel", default=None,
                   help="override matrix path rel to dataset dir; "
                        "defaults to the calibration JSON's matrix_rel")
    args = p.parse_args()

    cal = json.load(open(args.calibration))
    edges = np.asarray(cal["calibration"]["landmark_log_lr_edges"])
    ref_values = np.asarray(cal["calibration"]["landmark_log_lr_values"])
    ref_city = cal.get("reference_city", "Seattle")
    n_bins = len(ref_values)
    widths = np.diff(edges)
    matrix_rel = args.matrix_rel or cal.get("matrix_rel", MATRIX_REL)
    residual_form = cal.get("residual_form", "normalized")
    xlabel = (r"$r = \mathrm{row\,max} - \mathrm{sim}$" if residual_form == "raw"
              else r"$r_{\mathrm{norm}} = 1 - \mathrm{sim}/\mathrm{row\,max}$")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plt.rcParams.update({"font.size": 11})

    if not args.all_cities:
        # single-city behaviour (backwards compatible)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True)
        diag = _plot_city(ax1, ax2, args.city, edges, widths, n_bins,
                          ref_values, ref_city, device, matrix_rel, residual_form)
        ax1.set_ylabel("density")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"$\log\,p(r|\mathrm{true}) - \log\,p(r|\mathrm{neg})$")
        fig.tight_layout()
        out = Path(args.output_path)
        for ext in ("png", "pdf"):
            fig.savefig(out.with_suffix(f".{ext}"), dpi=190, bbox_inches="tight")
        plt.close(fig)
        print(json.dumps(diag, indent=2))
        print(f"Wrote {out.with_suffix('.png')}")
        return

    cities = list(CITY_INFO)
    ncol = len(cities)
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.6), sharex=True,
                             squeeze=False)
    diags = []
    for j, city in enumerate(cities):
        diags.append(_plot_city(axes[0][j], axes[1][j], city, edges, widths,
                                n_bins, ref_values, ref_city, device, matrix_rel,
                                residual_form))
        if j == 0:
            axes[0][j].set_ylabel("density (log)")
            axes[1][j].set_ylabel(r"$\log$-LR  $g(r)$")
        axes[1][j].set_xlabel(xlabel, fontsize=9)
    fig.suptitle(
        f"Per-city landmark residual densities (top) and log-LR (bottom): "
        f"own vs frozen {ref_city} lookup applied by the filter",
        fontsize=13, y=1.005)
    fig.tight_layout()
    out = Path(args.output_path)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'city':24s}{'D_self':>8s}{'D_trans':>9s}"
          f"{'true%@r<.1':>11s}{'true%@r>.9':>11s}{'n_true':>8s}")
    print("-" * 71)
    for d in sorted(diags, key=lambda r: r["true_mass_hi_r"], reverse=True):
        print(f"{d['city']:24s}{d['D_self']:8.3f}{d['D_transfer']:9.3f}"
              f"{100*d['true_mass_lo_r']:10.1f}%{100*d['true_mass_hi_r']:10.1f}%"
              f"{d['n_true']:8d}")
    print("\ntrue%@r<.1 = true patches in the rewarded (g>0, up to +4) region.")
    print("true%@r>.9 = true patches in the penalized miss bin (g<0).")
    print("Low true%@r<.1 + high true%@r>.9 = lookup mostly penalizes the true "
          "patch and rewards spurious low-r negatives => divergence at scale=1.")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
