"""Decide whether a dataset's camera is bolted down, drifting, or unanchored.

An offset estimator reports failure identically for two very different causes:
a camera that genuinely panned during the run, and a camera that was fixed but
sat in front of a scene the estimator could not read. This separates them, and
in the drifting case it also says whether there is anything in frame to align
*against*.

Method. Average many frames. World content -- water, coastline, traffic --
moves between them and blurs away; anything rigidly attached to the vehicle
occupies the same pixels every time and stays sharp. So

    persistence = |d/dx mean_t I_t| / mean_t |d/dx I_t|

is ~1 where image content is fixed in the camera frame and ~0 where it is not,
with the local texture scale divided out.

Two details are load-bearing.

*Normalising by the texture.* A raw temporal difference calls flat overcast
sky "static" exactly as loudly as it calls a mast static. On nyc_east_river
that mislabels the top 20% of the frame; dividing by mean|d/dx I| abstains
there instead, because neither term has any signal.

*Using only the x-derivative.* The full gradient magnitude also flags the
horizon: with steady pitch it sits on the same image row in every frame, so
grad(mean) stays sharp along it. It is fixed in the camera frame, but because
the scene is at infinite range, not because it is part of the boat -- and it
is precisely the far-field band the mount-offset estimators need to keep. A
horizon is a purely horizontal edge with no vertical one; vehicle structure
(mast, rail stanchion, bow) carries persistent vertical edges. Restricting to
d/dx keeps the structure and drops the horizon. Masking the horizon out is
what made a static mask perform *worse* on boston_harbor_leg1, so this is not
a hypothetical.

Reading the output: compare the whole-run figure against short windows.

    global ~= windowed, both high   camera and anchor are rigid
    global ~ 0, windowed high       anchor exists but DRIFTS -- alignable by
                                    tracking it, which is the only route to a
                                    per-frame camera yaw for a handheld capture
    both ~ 0                        nothing vehicle-fixed in frame; either the
                                    camera only ever sees the world (fine, but
                                    unverifiable this way) or there is no
                                    anchor to stabilise against

Each dataset's verdict is written to `<dataset>/_manifests/vehicle_anchor.json`
(the derived triage lane), which is where `dataset_status_table` reads it.
The old tool only wrote a combined JSON when asked, with no default location,
so the status table usually found nothing.

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:detect_vehicle_anchor -- \\
        --dataset_path /data/farfield_matching/datasets/*
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from experimental.overhead_matching.swag.farfield import provenance

SIDECAR_NAME = "vehicle_anchor.json"

# A pixel counts as anchored when its mean-image edge survives at 60% of the
# per-frame edge strength. Well separated in practice: rigid mounts land near
# 0.9 and world content near 0.1, so the threshold is not a tuned knob.
ANCHOR_THRESHOLD = 0.6
# Below this mean edge strength there is no texture to judge, so abstain.
TEXTURE_FLOOR = 1.5


def box_blur(a, k=5):
    """Mean over a k x k window; one-pixel misregistration must not flip a
    pixel."""
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def resolve_frame_path(ds: Path, rows, i, listing):
    """Locate row i's image, tolerating the two layouts in use.

    The self-collected datasets symlink `panorama/` to `frames/`, so either
    works. boston_harbor's `panorama/` instead holds separately *renamed*
    copies -- its `frame_file` is `f0000_t00070.00s_d00000m.jpg`, which exists
    only under `frames/` -- so the row's own name has to be tried there first
    and index alignment kept as a last resort. Looking only in `panorama/`
    resolves nothing on that dataset, which is how this silently reported
    "0.0% anchor" for boston_harbor_leg1 when the truth was "no image was
    read".
    """
    name = rows[i]["frame_file"]
    for candidate in (ds / "frames" / name, ds / "panorama" / name):
        if candidate.exists():
            return candidate
    if i < len(listing):
        return listing[i]
    return None


def persistence_map(ds: Path, rows, index_range, n_samples, work_w):
    """(persistence, mean image), or (None, None) if no image could be read."""
    lo, hi = index_range
    count = min(n_samples, hi - lo + 1)
    if count < 4:
        return None, None
    listing = sorted((ds / "panorama").glob("*.jpg"))
    idxs = np.linspace(lo, hi, count).astype(int)
    acc_img = acc_edge = None
    for i in idxs:
        path = resolve_frame_path(ds, rows, int(i), listing)
        if path is None or not path.exists():
            continue
        im = Image.open(path).convert("L")
        height = max(1, round(work_w * im.height / im.width))
        a = np.asarray(im.resize((work_w, height), Image.BILINEAR),
                       dtype=np.float32)
        if acc_img is None:
            acc_img, acc_edge = np.zeros_like(a), np.zeros_like(a)
        acc_img += a
        acc_edge += np.abs(np.gradient(a, axis=1))
    if acc_img is None:
        return None, None
    mean_img = acc_img / len(idxs)
    numerator = box_blur(np.abs(np.gradient(mean_img, axis=1)))
    denominator = box_blur(acc_edge / len(idxs))
    persistence = np.where(denominator > TEXTURE_FLOOR,
                           numerator / np.maximum(denominator, 1e-6), np.nan)
    return persistence, mean_img


def anchor_fraction(persistence):
    if persistence is None:
        return None
    valid = np.isfinite(persistence)
    if not valid.any():
        return None
    return float((valid & (persistence > ANCHOR_THRESHOLD)).mean())


def classify(global_frac, window_fracs, rigid_ratio=0.5,
             min_anchor_frac=0.03):
    """rigid / drifting / no_anchor, from whole-run vs short-window
    persistence.

    `no_anchor` is a statement about the *imagery*, not the mount: it means
    nothing vehicle-fixed is in frame, so this method has no opinion either
    way. Several perfectly rigid captures land here simply because the camera
    looks out at open water and never sees its own vessel.

    min_anchor_frac exists because a percent or two of "persistent" pixels is
    what noise alone produces; calling that a drifting anchor would invent a
    finding out of nothing.
    """
    windows = [w for w in window_fracs if w is not None]
    best_window = max(windows) if windows else 0.0
    if global_frac is None:
        return "unknown", best_window
    if best_window < min_anchor_frac:
        return "no_anchor", best_window
    if global_frac >= rigid_ratio * best_window:
        return "rigid", best_window
    return "drifting", best_window


def analyse(ds: Path, args):
    rows = list(csv.DictReader(open(ds / "frames_gps.csv")))
    n = len(rows)
    if n < args.window + 2:
        print(f"{ds.name}: only {n} frames, skipping")
        return None
    whole, mean_img = persistence_map(ds, rows, (0, n - 1), args.samples,
                                      args.work_width)
    if whole is None:
        # Distinguish "no anchor" from "read nothing". They print almost the
        # same otherwise, and the second one silently voided a mount-offset
        # comparison on boston_harbor_leg1.
        print(f"{ds.name}: NO IMAGES RESOLVED from frames_gps.frame_file — "
              f"cannot judge the anchor")
        return {"dataset": ds.name, "n_frames": n, "verdict": "no_images"}
    global_frac = anchor_fraction(whole)

    starts = np.linspace(0, n - 1 - args.window, args.n_windows).astype(int)
    window_fracs = []
    for lo in starts:
        p, _ = persistence_map(ds, rows, (int(lo), int(lo) + args.window),
                               args.window_samples, args.work_width)
        window_fracs.append(anchor_fraction(p))

    verdict, best_window = classify(global_frac, window_fracs)
    result = {
        "dataset": ds.name,
        "n_frames": n,
        "global_anchor_frac": global_frac,
        "window_anchor_fracs": window_fracs,
        "best_window_anchor_frac": best_window,
        "window_frames": args.window,
        "verdict": verdict,
    }
    print(f"{ds.name:<24} {verdict:<9} whole-run "
          f"{100 * (global_frac or 0):>5.1f}%   "
          f"{args.window}-frame windows " +
          " ".join(f"{100 * (w or 0):>5.1f}" for w in window_fracs))

    if args.write_overlay and whole is not None and mean_img is not None:
        mask = np.isfinite(whole) & (whole > ANCHOR_THRESHOLD)
        base = np.clip(mean_img, 0, 255)
        vis = np.stack(
            [base, base * (1 - 0.75 * mask), base * (1 - 0.75 * mask)], -1)
        out = ds / "_manifests" / "vehicle_anchor.png"
        out.parent.mkdir(exist_ok=True)
        Image.fromarray(vis.astype(np.uint8)).save(out)
        result["overlay"] = str(out)
    return result


def write_sidecar(ds: Path, result: dict) -> Path:
    record = {
        "generator": "farfield/dataset_tools/detect_vehicle_anchor.py",
        "git_commit": provenance.git_commit(),
        "argv": list(sys.argv),
        "created": datetime.datetime.now(datetime.timezone.utc)
                   .isoformat(timespec="seconds"),
        **result,
    }
    out = ds / "_manifests" / SIDECAR_NAME
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", nargs="+", required=True, type=Path)
    # Sampling knobs set the compute budget, not the verdict (the persistence
    # statistic saturates well below these counts), so they keep defaults.
    p.add_argument("--samples", type=int, default=60,
                   help="frames averaged for the whole-run map")
    p.add_argument("--window", type=int, default=30,
                   help="length of each short window, in frames")
    p.add_argument("--n_windows", type=int, default=6)
    p.add_argument("--window_samples", type=int, default=25)
    p.add_argument("--work_width", type=int, default=480)
    p.add_argument("--write_overlay", action="store_true",
                   help="save _manifests/vehicle_anchor.png in each dataset")
    p.add_argument("--dry_run", action="store_true",
                   help="analyse and print, but write no sidecars")
    args = p.parse_args(argv)

    for ds in args.dataset_path:
        if not (ds / "frames_gps.csv").exists():
            continue
        result = analyse(ds, args)
        if result and not args.dry_run:
            out = write_sidecar(ds, result)
            print(f"    wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
