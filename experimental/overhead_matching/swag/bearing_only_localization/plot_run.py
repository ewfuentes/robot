"""Plot a bearing-only localization run directory.

Reads only the self-describing run dir written by run_log.py and produces
<run_dir>/plots/{map.png,strip.png} plus an optional particle animation
(anim.gif, --animate).

Usage:
  bazel run //experimental/overhead_matching/swag/bearing_only_localization:plot_run -- \
    --run_dir /tmp/bol_demo --animate
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
import numpy as np  # noqa: E402

from experimental.overhead_matching.swag.bearing_only_localization import (  # noqa: E402
    filter as pf,
    geodesy,
    run_log,
)

_LANDMARK_MARKERS = {"lighthouse": "*", "storage_tank": "s"}


def _subsample(arrays, max_points, seed=0):
    n = arrays[0].shape[0]
    if n <= max_points:
        return arrays
    idx = np.random.default_rng(seed).choice(n, size=max_points,
                                             replace=False)
    return [a[idx] for a in arrays]


def _landmark_positions(data):
    frame = geodesy.RegionFrame(data.manifest.anchor_lat_deg,
                                data.manifest.anchor_lon_deg)
    east, north = frame.enu_from_latlon(
        np.array([lm.lat_deg for lm in data.manifest.landmarks]),
        np.array([lm.lon_deg for lm in data.manifest.landmarks]))
    return east, north


def _draw_map(data, wedge_keyframe, max_particles, ax):
    lm_east, lm_north = _landmark_positions(data)
    for lm, e, n in zip(data.manifest.landmarks, lm_east, lm_north):
        ax.scatter([e], [n], s=250, zorder=5,
                   marker=_LANDMARK_MARKERS.get(lm.type_key, "^"),
                   edgecolors="black", facecolors="gold")
        ax.annotate(lm.landmark_id, (e, n), textcoords="offset points",
                    xytext=(8, 8), fontsize=8)

    truth_e = [t.east_m for t in data.truth]
    truth_n = [t.north_m for t in data.truth]
    ax.plot(truth_e, truth_n, color="0.4", lw=1.5, label="truth")
    ax.scatter([truth_e[0]], [truth_n[0]], marker="o", color="0.4", s=40,
               zorder=4)

    ax.plot([h.mean_east_m for h in data.health],
            [h.mean_north_m for h in data.health],
            color="tab:blue", lw=1.2, label="mean estimate")
    # The MAP trail is the one to trust when the belief is multimodal: the
    # weighted mean then sits between modes and describes no hypothesis.
    ax.plot([h.map_east_m for h in data.health],
            [h.map_north_m for h in data.health],
            color="tab:purple", lw=1.0, ls="--", label="MAP (densest mode)")

    keyframes = sorted(data.checkpoints.keys())
    if keyframes:
        first_kf, last_kf = keyframes[0], keyframes[-1]
        for kf, color, alpha, label in [
                (first_kf, "tab:orange", 0.25, f"particles kf {first_kf}"),
                (last_kf, "tab:green", 0.35, f"particles kf {last_kf}")]:
            ckpt = data.checkpoints[kf]
            east, north = _subsample(
                [ckpt["east_m"], ckpt["north_m"]], max_particles)
            ax.scatter(east, north, s=2, alpha=alpha, color=color,
                       label=label, linewidths=0)

    # Bearing wedges at the chosen keyframe, drawn from the mean pose.
    health_by_kf = {h.keyframe_idx: h for h in data.health}
    wedge_meas = [m for m in data.measurements
                  if m.anchor_keyframe_idx == wedge_keyframe]
    if wedge_meas and wedge_keyframe in health_by_kf:
        record = health_by_kf[wedge_keyframe]
        heading_rad = math.radians(record.mean_heading_deg)
        ray_len = 1.1 * float(np.max(np.hypot(lm_east, lm_north)))
        for meas in wedge_meas:
            world = heading_rad + math.radians(meas.bearing_body_deg)
            sigma = 1.0 / math.sqrt(meas.kappa)
            for offset, style, lw in [(0.0, "-", 1.2), (2 * sigma, ":", 0.8),
                                      (-2 * sigma, ":", 0.8)]:
                angle = world + offset
                ax.plot(
                    [record.mean_east_m,
                     record.mean_east_m + ray_len * math.sin(angle)],
                    [record.mean_north_m,
                     record.mean_north_m + ray_len * math.cos(angle)],
                    style, color="tab:red", lw=lw, alpha=0.7)
        ax.set_title(f"{data.manifest.scenario_name} — bearing wedges at "
                     f"keyframe {wedge_keyframe}")
    else:
        ax.set_title(data.manifest.scenario_name)

    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def _draw_strip(data, axes):
    keyframes = [h.keyframe_idx for h in data.health]
    resample_kfs = [h.keyframe_idx for h in data.health if h.resampled]

    if data.truth:
        errors = pf.position_errors_m(data.health, data.truth)
        heading_errors = pf.heading_errors_deg(data.health, data.truth)
        axes[0].plot(keyframes, errors, lw=1.0, label="mean error")
        axes[0].plot(keyframes, pf.map_position_errors_m(data.health,
                                                         data.truth),
                     lw=0.8, color="tab:purple", ls="--", label="MAP error")
        # Reported sigma next to actual error: if the band sits below the
        # error line the filter is overconfident, which is the defect that
        # accuracy-only plots hide (see consistency_test.py).
        axes[0].plot(keyframes, [h.position_std_m for h in data.health],
                     lw=0.8, color="tab:green", label="reported sigma")
        axes[0].set_ylabel("pos err (m)")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=7)
        axes[1].plot(keyframes, heading_errors, lw=1.0, label="error")
        axes[1].plot(keyframes, [h.heading_std_deg for h in data.health],
                     lw=0.8, color="tab:green", label="reported sigma")
        axes[1].set_ylabel("heading err (deg)")
        axes[1].set_yscale("log")
        axes[1].legend(fontsize=7)
    for kf in resample_kfs:
        axes[0].axvline(kf, color="0.85", lw=0.5, zorder=0)

    axes[2].plot(keyframes, [h.ess for h in data.health], lw=1.0)
    n_particles = data.manifest.filter_config.n_particles
    threshold = data.manifest.filter_config.ess_resample_frac * n_particles
    axes[2].axhline(threshold, color="tab:red", lw=0.8, ls="--",
                    label="resample threshold")
    axes[2].set_ylabel("ESS")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=7)

    assoc_kf, assoc_null = [], []
    for record in data.health:
        for assoc in record.associations:
            assoc_kf.append(record.keyframe_idx)
            assoc_null.append(assoc.null_share)
    axes[3].scatter(assoc_kf, assoc_null, s=6, alpha=0.6)
    axes[3].set_ylabel("null share")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_xlabel("keyframe")
    for ax in axes:
        ax.grid(alpha=0.3)


def _animate(data, out_path, max_particles):
    lm_east, lm_north = _landmark_positions(data)
    keyframes = sorted(data.checkpoints.keys())
    truth_by_kf = {t.keyframe_idx: t for t in data.truth}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    all_e = np.concatenate([data.checkpoints[keyframes[0]]["east_m"],
                            lm_east])
    all_n = np.concatenate([data.checkpoints[keyframes[0]]["north_m"],
                            lm_north])
    pad = 300.0
    ax.set_xlim(all_e.min() - pad, all_e.max() + pad)
    ax.set_ylim(all_n.min() - pad, all_n.max() + pad)
    ax.scatter(lm_east, lm_north, s=250, marker="*", edgecolors="black",
               facecolors="gold", zorder=5)
    ax.plot([t.east_m for t in data.truth],
            [t.north_m for t in data.truth], color="0.6", lw=1.0)
    scat = ax.scatter([], [], s=2, alpha=0.4, color="tab:green",
                      linewidths=0)
    truth_dot = ax.scatter([], [], s=60, marker="o", color="tab:red",
                           zorder=6)
    title = ax.set_title("")

    def update(frame_idx):
        kf = keyframes[frame_idx]
        ckpt = data.checkpoints[kf]
        east, north = _subsample([ckpt["east_m"], ckpt["north_m"]],
                                 max_particles, seed=kf)
        scat.set_offsets(np.column_stack([east, north]))
        if kf in truth_by_kf:
            t = truth_by_kf[kf]
            truth_dot.set_offsets([[t.east_m, t.north_m]])
        title.set_text(f"keyframe {kf}")
        return scat, truth_dot, title

    anim = animation.FuncAnimation(fig, update, frames=len(keyframes))
    anim.save(out_path, writer=animation.PillowWriter(fps=4))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--wedge_keyframe", type=int, default=None,
                        help="Keyframe for bearing wedges "
                             "(default: last keyframe with measurements)")
    parser.add_argument("--max_particles", type=int, default=3000)
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    data = run_log.read_run(args.run_dir)
    plots_dir = args.run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    wedge_kf = args.wedge_keyframe
    if wedge_kf is None and data.measurements:
        wedge_kf = max(m.anchor_keyframe_idx for m in data.measurements)

    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_map(data, wedge_kf, args.max_particles, ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "map.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    _draw_strip(data, axes)
    fig.tight_layout()
    fig.savefig(plots_dir / "strip.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {plots_dir}/map.png and strip.png")

    if args.animate:
        _animate(data, plots_dir / "anim.gif", args.max_particles)
        print(f"Wrote {plots_dir}/anim.gif")


if __name__ == "__main__":
    main()
