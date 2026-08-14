"""Calibrate a dataset's camera-to-motion yaw offset from imagery alone.

Design doc §5.2 makes position route through the heading state, so camera-frame
bearings must be de-rotated into the motion body frame before the filter sees
them (`bearing_body_deg = bearing_camera_deg - mount_offset_deg`, plan Phase 1).
That offset is therefore a required per-dataset input, and it must be *constant
over the leg* for a single number to be valid.

This estimates it **map-free and GPS-course-free** from the direction of camera
translation, which is what the offset is: the focus of expansion is the image
direction the camera is moving toward. Nothing here reads a heading field, a
compass, or a landmark catalog, so it is independent of every quantity those
datasets are unreliable about. GPS is used only to pick frame pairs a chosen
distance apart -- a scalar step length, not a direction.

Method, equirectangular: for a camera translating by d through static scene
points at horizontal range R, a point at camera azimuth θ shifts by

    Δθ = -Δψ + (d/R)·sin(θ - β)

where β is the translation direction (the mount offset) and Δψ the camera's own
yaw change. Expanding the sine makes this **linear** in three unknowns:

    Δθ = c + p·sin θ + q·cos θ,    β = atan2(-q, p),  Δψ = -c,  A = hypot(p, q)

so one robust linear fit per frame pair yields the offset *and* the visual yaw
increment, with no knowledge of any point's range. Unknown per-point R only
scales A, never the phase β -- which is why this works on a scene at mixed
depths.

Method, perspective: recover the essential matrix from matched features with the
known focal length and decompose it; the unit translation gives
β = atan2(t_x, t_z) directly.

    bazel run //experimental/overhead_matching/swag/scripts:calibrate_mount_offset -- \
        --dataset_path /data/farfield_matching/boston_harbor_dataset/processed/leg1
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# Far-field parallax is tiny over one video step: 5 m of travel against a 2 km
# coastline is 0.14 deg, well under a pixel. The estimator needs a real baseline,
# so pairs are chosen by distance travelled rather than frame index.
DEFAULT_BASELINE_M = 100.0

# Equirectangular rows to use, as a fraction of image height. Restricted to a
# band about the horizon for two reasons: the vessel's own deck and superstructure
# sit below it and are *static in the camera frame*, so they would pull the fit
# toward Δθ = 0 at every azimuth; and the sky above is textureless. The
# coastline and far-field landmarks this project cares about lie in this band.
HORIZON_BAND = (0.38, 0.53)


def circular_median_deg(values, period=360.0):
    """Median direction, via the angle minimising summed circular distance.

    A plain median is wrong on a wrapped quantity (offsets near 0/360 would
    average to 180), and the circular *mean* is not robust to the outliers this
    estimator produces on featureless water.

    `period=180` treats a direction and its opposite as the same *axis*, which is
    what the mount geometry actually fixes -- see aggregate_axis().
    """
    if not values:
        return None
    arr = np.asarray(values, dtype=float) % period
    candidates = np.concatenate([arr, arr + period / 2.0])
    best, best_cost = None, None
    for c in candidates:
        diff = np.abs((arr - c + period / 2.0) % period - period / 2.0)
        cost = float(diff.sum())
        if best_cost is None or cost < best_cost:
            best, best_cost = float(c % period), cost
    return best


def circular_mad_deg(values, centre, period=360.0):
    """Median absolute circular deviation -- the "is it constant?" statistic."""
    if not values or centre is None:
        return None
    arr = np.asarray(values, dtype=float) % period
    diff = np.abs((arr - centre + period / 2.0) % period - period / 2.0)
    return float(np.median(diff))


def aggregate_axis(betas, agree_deg=30.0):
    """Fold antipodal votes onto one axis, then choose the direction by majority.

    The estimator recovers the direction of *travel* in the camera frame, and a
    vehicle that reverses -- a ferry backing off its berth -- genuinely inverts
    that while the camera's mounting does not move. Those pairs land at β+180 and
    are correct measurements, not outliers, so an axis-then-direction aggregation
    is the honest one: `mad_deg` then answers "is the mounting fixed?" rather than
    being inflated by legitimate reversals, and the reversal share is reported
    separately instead of being hidden.

    Folding first is also what makes the median robust here. A raw circular median
    over an antipodally bimodal sample is unstable -- votes at 167° and 347° admit
    minimisers anywhere between -- which produced a nonsense 160°/MAD 8° on
    fukuoka_yumechan_a before this change.
    """
    axis = circular_median_deg(betas, period=180.0)
    mad = circular_mad_deg(betas, axis, period=180.0)
    arr = np.asarray(betas, dtype=float)
    forward = np.abs((arr - axis + 180.0) % 360.0 - 180.0) <= 90.0
    n_forward = int(forward.sum())
    centre = axis if n_forward >= len(arr) - n_forward else (axis + 180.0) % 360.0
    reversal = min(n_forward, len(arr) - n_forward) / max(1, len(arr))
    # Only count pairs that actually agree with the chosen axis as support.
    aligned = np.abs((arr - axis + 90.0) % 180.0 - 90.0) <= agree_deg
    return centre, mad, reversal, float(aligned.mean())


def load_frames(dataset: Path):
    """[(image path, cumulative distance)] in dataset order.

    `frame_file` names the image in `frames/`. The self-collected datasets make
    `panorama/` a symlink to that same directory so either resolves, but
    boston_harbor's `panorama/` holds separately *renamed* copies, so the row's
    own filename must be tried against `frames/` first and index-alignment kept
    as a last resort.
    """
    rows = list(csv.DictReader(open(dataset / "frames_gps.csv")))
    listing = sorted((dataset / "panorama").glob("*.jpg"))
    out = []
    for i, r in enumerate(rows):
        candidates = [dataset / "frames" / r["frame_file"],
                      dataset / "panorama" / r["frame_file"]]
        path = next((c for c in candidates if c.exists()), None)
        if path is None and i < len(listing):
            path = listing[i]
        if path is not None and path.exists():
            out.append((path, float(r["dist_m"])))
    return out


def pick_pairs(frames, baseline_m, max_pairs):
    """(i, j) pairs separated by ~baseline_m of travel, spread over the track."""
    pairs = []
    j = 0
    for i in range(len(frames)):
        while j < len(frames) and frames[j][1] - frames[i][1] < baseline_m:
            j += 1
        if j >= len(frames):
            break
        pairs.append((i, j))
    if not pairs:
        return []
    step = max(1, len(pairs) // max_pairs)
    return pairs[::step][:max_pairs]


def match_features(img_a, img_b, orb, ratio=0.75, min_matches=25):
    """Matched keypoint coordinates between two images, or None."""
    ka, da = orb.detectAndCompute(img_a, None)
    kb, db = orb.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < min_matches or len(kb) < min_matches:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < min_matches:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return pa, pb


def fit_equirect_offset(theta, dtheta, yaw_grid_deg=60.0, cap_deg=2.0):
    """Translation azimuth from flow *sign* structure; scale-free in range.

    Returns (beta, dpsi, quality, n_informative).

    Why signs and not a least-squares sinusoid: the amplitude of
    `(d/R)·sin(θ - β)` carries each point's unknown range, and a real scene spans
    a huge span of them. On boston_harbor_leg1 the near city structure gives 23°
    of flow (R ≈ 500 m) while the open-ocean horizon in the opposite sector gives
    0.19° (R ≈ 50 km) -- a 100x amplitude ratio. A single-amplitude fit is then
    driven entirely by the near sector, and because that sector does not span the
    full circle the phase comes out biased: it read 20° off a known-good answer.

    Signs are immune to that. Since d/R > 0 always, sign(Δθ - c) = sign(sin(θ - β))
    for every point whatever its range, so the zero crossings alone pin β. Points
    whose flow is below the matcher's ~1 px quantization contribute |Δθ - c| ≈ 0
    and self-weight out rather than voting on a sign they cannot resolve.

    The camera's own yaw change c enters every point equally, so it is searched
    jointly rather than assumed -- and -c is a visual Δyaw estimate in its own
    right, which §5.2 accepts as an alternative Δyaw producer.
    """
    yaw = np.arange(-yaw_grid_deg, yaw_grid_deg + 0.25, 0.5)
    beta = np.arange(0.0, 360.0, 1.0)
    # [n_yaw, n_pts] signed residual after removing a candidate camera rotation.
    residual = dtheta[None, :] - yaw[:, None]
    # Weight by how far the flow sits above the quantization floor, capped so one
    # very close object cannot outvote a whole sector.
    weight = np.clip(np.abs(residual), 0.0, cap_deg) * np.sign(residual)
    # [n_pts, n_beta] expected sign of the parallax term.
    expected = np.sign(np.sin(np.radians(theta[:, None] - beta[None, :])))
    score = weight @ expected                      # [n_yaw, n_beta]
    iy, ib = np.unravel_index(int(np.argmax(score)), score.shape)

    informative = int((np.abs(residual[iy]) > 360.0 / 1920.0).sum())
    peak = float(score[iy, ib])
    # Normalise the peak by the total vote mass available: 1.0 means every
    # informative point agrees, ~0 means the sign field carries no direction.
    mass = float(np.abs(weight[iy]).sum()) + 1e-9
    return float(beta[ib]), -float(yaw[iy]), peak / mass, informative


def fit_perspective_offset(pa, pb, focal_px, centre):
    """Translation azimuth from the essential matrix; returns (beta, dpsi, n)."""
    essential, mask = cv2.findEssentialMat(
        pa, pb, focal=focal_px, pp=centre, method=cv2.RANSAC,
        prob=0.999, threshold=1.0)
    if essential is None or essential.shape != (3, 3):
        return None
    inliers, rotation, translation, _ = cv2.recoverPose(
        essential, pa, pb, focal=focal_px, pp=centre, mask=mask)
    if inliers < 15:
        return None
    # Camera frame: +x right, +z along the optical axis, so the translation's
    # azimuth off the optical axis is the mount offset.
    beta = math.degrees(math.atan2(float(translation[0]), float(translation[2])))
    dpsi = math.degrees(math.atan2(float(rotation[0, 2]), float(rotation[2, 2])))
    return beta % 360.0, dpsi, int(inliers)


def build_vehicle_mask(dataset: Path):
    """Normalised mask of the parts of the frame bolted to the vehicle, or None.

    A bow, a mast, a rail: rigidly attached structure moves *with* the camera, so
    every correspondence on it is a pure-rotation observation with zero parallax.
    Feed enough of those to findEssentialMat and the translation direction it
    reports is whatever the remaining minority of world points can drag it to.
    mississippi_rural's mast and bow cover 8.5% of the frame and nyc_east_river's
    rail and cabin more, which is a large share of a RANSAC consensus set.

    Delegated to detect_vehicle_anchor so there is one definition of "fixed in
    the camera frame" -- in particular its x-derivative-only rule, without which
    the mask would also swallow the horizon, the one band this estimator most
    needs to keep.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import detect_vehicle_anchor as anchor
    except ImportError:
        return None
    rows = list(csv.DictReader(open(dataset / "frames_gps.csv")))
    persistence, _ = anchor.persistence_map(dataset, rows, (0, len(rows) - 1),
                                            n_samples=40, work_w=480)
    if persistence is None:
        return None
    mask = np.isfinite(persistence) & (persistence > anchor.ANCHOR_THRESHOLD)
    return mask if mask.mean() > 0.01 else None


def drop_masked_points(matched, mask, work_w, work_h):
    """Keep only correspondences whose first-frame point is outside the mask."""
    pa, pb = matched
    mh, mw = mask.shape
    cols = np.clip((pa[:, 0] / work_w * mw).astype(int), 0, mw - 1)
    rows_ = np.clip((pa[:, 1] / work_h * mh).astype(int), 0, mh - 1)
    keep = ~mask[rows_, cols]
    if keep.sum() < 25:
        return None
    return pa[keep], pb[keep]


def calibrate(dataset: Path, args):
    meta = json.loads((dataset / "pipeline_metadata.json").read_text()) \
        if (dataset / "pipeline_metadata.json").exists() else {}
    projection = meta.get("projection", "")
    frames = load_frames(dataset)
    if len(frames) < 3:
        print(f"{dataset.name}: only {len(frames)} frames, skipping")
        return None

    probe = cv2.imread(str(frames[0][0]), cv2.IMREAD_GRAYSCALE)
    height, width = probe.shape
    # Fall back on aspect ratio when there is no metadata (boston_harbor has none).
    is_equirect = (projection.startswith("equi") if projection
                   else abs(width / height - 2.0) < 0.05)

    pairs = pick_pairs(frames, args.baseline_m, args.max_pairs)
    if not pairs:
        print(f"{dataset.name}: track shorter than one {args.baseline_m:.0f} m "
              f"baseline, skipping")
        return None

    focal_norm = None
    if not is_equirect and (dataset / "intrinsics.csv").exists():
        rows = list(csv.DictReader(open(dataset / "intrinsics.csv")))
        focals = [float(r["focal_norm"]) for r in rows if r.get("focal_norm")]
        focal_norm = float(np.median(focals)) if focals else None
        if focal_norm is None:
            print(f"{dataset.name}: perspective but no focal_norm, skipping")
            return None

    vehicle_mask = None
    if args.mask_vehicle and not is_equirect:
        vehicle_mask = build_vehicle_mask(dataset)
        if vehicle_mask is not None:
            print(f"{dataset.name:24s} masking {100 * vehicle_mask.mean():.1f}% "
                  f"of the frame as vehicle structure")

    orb = cv2.ORB_create(nfeatures=args.features)
    betas, dpsis, quality = [], [], []
    # Equirect runs in two passes: gather the flow, find the azimuths where the
    # platform occludes itself, then fit only the moving scene. The mask is a
    # property of the leg (the vehicle does not move in the camera frame), so it
    # is computed once from all pairs rather than per pair.
    pending, static, bin_deg = [], None, 5.0
    for i, j in pairs:
        a = cv2.imread(str(frames[i][0]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(frames[j][0]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or a.shape != b.shape:
            continue
        scale = args.work_width / a.shape[1]
        if scale < 1.0:
            a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        work_h, work_w = a.shape

        if is_equirect:
            lo, hi = int(HORIZON_BAND[0] * work_h), int(HORIZON_BAND[1] * work_h)
            band_a, band_b = a[lo:hi], b[lo:hi]
            matched = match_features(band_a, band_b, orb)
            if matched is None:
                continue
            pa, pb = matched
            theta = (pa[:, 0] / work_w) * 360.0
            dtheta = (((pb[:, 0] - pa[:, 0]) / work_w) * 360.0 + 180.0) % 360.0 - 180.0
            pending.append((theta, dtheta, pa, pb, work_w))
            continue
        else:
            focal_px = focal_norm * max(work_w, work_h)
            matched = match_features(a, b, orb)
            if matched is None:
                continue
            if vehicle_mask is not None:
                matched = drop_masked_points(matched, vehicle_mask, work_w, work_h)
                if matched is None:
                    continue
            result = fit_perspective_offset(*matched, focal_px,
                                            (work_w / 2.0, work_h / 2.0))
            if result is None:
                continue
            beta, dpsi, n = result
            quality.append((float("nan"), float("nan"), n))
        betas.append(beta)
        dpsis.append(dpsi)

    if is_equirect and pending:
        for theta, dtheta, pa, pb, work_w in pending:
            beta, dpsi, agreement, informative = fit_equirect_offset(theta, dtheta)
            # A pair only votes if enough points resolve a flow sign and those
            # signs actually agree on a direction. Open water with no mid-range
            # structure fails here -- which is the correct outcome, reported as
            # INSUFFICIENT rather than averaged into a confident wrong number.
            if informative < args.min_informative or agreement < args.min_agreement:
                continue
            quality.append((agreement, float("nan"), informative))
            betas.append(beta)
            dpsis.append(dpsi)

    if len(betas) < args.min_pairs:
        print(f"{dataset.name:24s} INSUFFICIENT — only {len(betas)} usable pair(s) "
              f"of {len(pairs)} attempted ({'equirect' if is_equirect else 'perspective'})")
        return {"dataset": dataset.name, "usable_pairs": len(betas),
                "attempted_pairs": len(pairs), "status": "insufficient"}

    centre, spread, reversal, aligned = aggregate_axis(betas)
    verdict = ("constant" if spread is not None and spread <= args.constant_mad_deg
               and aligned >= args.min_aligned_frac else "NOT constant")
    print(f"{dataset.name:24s} {'equirect' if is_equirect else 'persp':9s} "
          f"offset={centre:6.1f}°  axis MAD={spread:5.1f}°  "
          f"aligned={aligned:4.0%}  rev={reversal:4.0%}  "
          f"pairs={len(betas):3d}/{len(pairs):3d}  {verdict}")
    return {
        "dataset": dataset.name,
        "projection": "equirectangular" if is_equirect else "perspective",
        "mount_offset_deg": round(centre, 2),
        "axis_mad_deg": round(spread, 2),
        "aligned_fraction": round(aligned, 3),
        "reversal_fraction": round(reversal, 3),
        "constant_over_leg": verdict == "constant",
        "usable_pairs": len(betas),
        "attempted_pairs": len(pairs),
        "baseline_m": args.baseline_m,
        "median_visual_dyaw_deg": round(float(np.median(dpsis)), 3),
        "median_sign_agreement": (
            round(float(np.median([q[0] for q in quality])), 3)
            if is_equirect and quality else None),
        "per_pair_offsets_deg": [round(b, 2) for b in betas],
        "status": "ok",
    }


def write_metadata(dataset: Path, result, args):
    """Put the calibration where a consumer will look for it, gates applied.

    Only a calibration that passes both gates carries a number a caller may
    apply. The axis MAD says whether a single offset exists at all; the reversal
    fraction says whether its *direction* is resolved, and near 50% that is a
    coin flip even when the axis is perfect. Publishing the angle with
    `usable: false` and the reason beside it is safer than publishing nothing --
    the next person re-derives it otherwise -- but it must never be readable as
    an answer.
    """
    meta_path = dataset / "pipeline_metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if result.get("status") != "ok":
        meta["mount_offset"] = {"usable": False, "status": result.get("status"),
                                "baseline_m": args.baseline_m}
        meta_path.write_text(json.dumps(meta, indent=2))
        return
    axis_ok = (result["axis_mad_deg"] <= args.constant_mad_deg
               and result["aligned_fraction"] >= args.min_aligned_frac)
    direction_ok = result["reversal_fraction"] <= 0.30
    meta["mount_offset"] = {
        "mount_offset_deg": result["mount_offset_deg"],
        "axis_mad_deg": result["axis_mad_deg"],
        "aligned_fraction": result["aligned_fraction"],
        "reversal_fraction": result["reversal_fraction"],
        "axis_constant": axis_ok,
        "direction_ambiguous": not direction_ok,
        "usable": bool(axis_ok and direction_ok),
        "method": ("focus-of-expansion from image flow "
                   "(calibrate_mount_offset.py); measures the direction of "
                   "travel in the camera frame, which is the body x-axis "
                   "gps_to_odometry declares by setting left_m=0"),
        "baseline_m": args.baseline_m,
        "pairs_used": result["usable_pairs"],
        "projection": result["projection"],
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", nargs="+", required=True, type=Path)
    parser.add_argument("--baseline_m", type=float, default=DEFAULT_BASELINE_M,
                        help="Metres of travel between paired frames "
                             f"(default: {DEFAULT_BASELINE_M})")
    parser.add_argument("--min_baseline_m", type=float, default=40.0,
                        help="Floor for the automatic baseline retry (default: 40)")
    parser.add_argument("--max_pairs", type=int, default=60)
    parser.add_argument("--min_pairs", type=int, default=8)
    parser.add_argument("--features", type=int, default=4000)
    parser.add_argument("--work_width", type=int, default=1920,
                        help="Downscale width for matching (default: 1920)")
    parser.add_argument("--min_informative", type=int, default=60,
                        help="Reject a pair with fewer points whose flow rises "
                             "above the matcher quantization floor (default: 60)")
    parser.add_argument("--min_agreement", type=float, default=0.25,
                        help="Reject a pair whose flow signs do not agree on a "
                             "direction this strongly, 0-1 (default: 0.25)")
    parser.add_argument("--min_aligned_frac", type=float, default=0.6,
                        help="Fraction of pairs that must fall within 30 deg of "
                             "the fitted axis (default: 0.6)")
    parser.add_argument("--constant_mad_deg", type=float, default=15.0,
                        help="Circular MAD at or below which the offset counts "
                             "as constant over the leg (default: 15)")
    parser.add_argument("--mask_vehicle", action="store_true",
                        help="Drop features on vehicle-fixed structure "
                             "before the essential matrix (perspective only)")
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--write_metadata", action="store_true",
                        help="Also write the result into each dataset's "
                             "pipeline_metadata.json under `mount_offset`, "
                             "replacing any earlier block")
    args = parser.parse_args()

    results = []
    for dataset in args.dataset_path:
        try:
            result = calibrate(dataset, args)
            # A short track cannot yield enough pairs at a long baseline --
            # baltimore_a covers 1.1 km total, so 200 m leaves 5 pairs and it
            # reported INSUFFICIENT despite being one of the cleanest datasets
            # at 50 m. Halve and retry rather than making the caller guess.
            requested = args.baseline_m
            while (result is not None and result.get("status") == "insufficient"
                   and args.baseline_m > args.min_baseline_m):
                args.baseline_m = max(args.min_baseline_m, args.baseline_m / 2.0)
                print(f"{'':24s} retrying at {args.baseline_m:.0f} m baseline")
                result = calibrate(dataset, args)
            args.baseline_m = requested
        except Exception as exc:  # keep a batch alive past one bad dataset
            print(f"{dataset.name:24s} ERROR: {type(exc).__name__}: {exc}")
            result = {"dataset": dataset.name, "status": "error", "error": str(exc)}
        if result:
            results.append(result)
            if args.write_metadata:
                write_metadata(dataset, result, args)

    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
