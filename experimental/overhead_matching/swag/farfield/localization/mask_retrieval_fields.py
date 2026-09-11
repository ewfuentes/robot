"""Mask a retrieval score-fields artifact to a box around the truth
trajectory (diagnostic tool).

Shrinks the declared support of a retrieval observation source so the filter
can be run on a controlled sub-region: "does the run converge when the
support excludes the far-away competitor?" is the isolation experiment for
capture-by-a-distant-mode failures. Consuming truth to build the box makes
any run on the output a DIAGNOSTIC by construction — the output directory
name records the margin, and the meta's scorer string is annotated so no
masked field set can be mistaken for the full support.

    bazel run //experimental/overhead_matching/swag/farfield/localization:mask_retrieval_fields -- \
        --retrieval_dir <artifacts>/retrieval_observations/<dataset>/<version> \
        --input_dir <artifacts>/localization_inputs/<dataset>/<version> \
        --truth_margin_m 1500 \
        --out_dir <artifacts>/retrieval_observations/<dataset>/<version>_truthbox1500
"""

import argparse
from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.localization import (
    export_ingest,
    retrieval,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval_dir", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="localization_inputs artifact supplying the "
                             "ENU frame and the truth trajectory")
    parser.add_argument("--truth_margin_m", type=float, required=True,
                        help="box = truth trajectory bounding box expanded "
                             "by this margin on every side")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    data = export_ingest.load(args.input_dir)
    if not data.truth:
        parser.error("the localization_inputs artifact carries no truth; "
                     "a truth-centered mask needs it")
    fields = retrieval.load_fields(args.retrieval_dir, data.frame)
    if fields.meta.dataset != data.meta.dataset:
        parser.error(f"retrieval fields are for {fields.meta.dataset!r}, "
                     f"inputs are {data.meta.dataset!r}")

    truth_east = np.array([p.east_m for p in data.truth])
    truth_north = np.array([p.north_m for p in data.truth])
    east_lo = truth_east.min() - args.truth_margin_m
    east_hi = truth_east.max() + args.truth_margin_m
    north_lo = truth_north.min() - args.truth_margin_m
    north_hi = truth_north.max() + args.truth_margin_m
    keep = ((fields.east_m >= east_lo) & (fields.east_m <= east_hi)
            & (fields.north_m >= north_lo) & (fields.north_m <= north_hi))
    n_kept = int(keep.sum())
    if n_kept == 0:
        parser.error("the mask keeps no nodes; check frames and margin")

    # Re-read the raw npz for the anchor-free lat/lon node coordinates the
    # artifact stores (ScoreFields only holds the ENU conversion).
    raw = np.load(args.retrieval_dir / "retrieval_fields.npz")
    meta = fields.meta
    meta.n_nodes = n_kept
    meta.scorer = (f"{meta.scorer} "
                   f"[masked: truth bbox +{args.truth_margin_m:.0f} m]")
    retrieval.write_fields(
        args.out_dir, meta,
        lat_deg=raw["lat_deg"][keep],
        lon_deg=raw["lon_deg"][keep],
        scores=raw["scores"][:, keep, :],
        keyframe_idx=raw["keyframe_idx"],
        pano_ids=[str(p) for p in raw["pano_ids"]])
    print(f"truth bbox : E [{truth_east.min():.0f}, {truth_east.max():.0f}] "
          f"N [{truth_north.min():.0f}, {truth_north.max():.0f}] m")
    print(f"mask box   : E [{east_lo:.0f}, {east_hi:.0f}] "
          f"N [{north_lo:.0f}, {north_hi:.0f}] m "
          f"({(east_hi - east_lo) / 1000:.1f} x "
          f"{(north_hi - north_lo) / 1000:.1f} km)")
    print(f"nodes      : {len(keep)} -> {n_kept}")
    print(f"written to {args.out_dir}")


if __name__ == "__main__":
    main()
