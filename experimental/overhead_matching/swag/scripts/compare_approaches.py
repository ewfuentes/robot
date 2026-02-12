import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import json
    import pickle
    import numpy as np
    import altair as alt
    import matplotlib.pyplot as plt
    from pathlib import Path

    import common.torch.load_torch_deps
    import torch

    return Path, alt, json, mo, np, pickle, pl, plt, torch


@app.cell
def _(mo):
    result_dir_input = mo.ui.text(
        value="/data/overhead_matching/evaluation/results/260209_seattle_1k_5km",
        label="Result directory",
        full_width=True,
    )
    result_dir_input
    return (result_dir_input,)


@app.cell
def _(Path, mo, result_dir_input):
    # Discover approach subdirectories that contain summary_statistics.json
    result_dir = Path(result_dir_input.value)
    approaches = sorted(
        p.parent.name
        for p in result_dir.glob("*/summary_statistics.json")
    )

    approach_a_dropdown = mo.ui.dropdown(
        options=approaches,
        value=approaches[0] if approaches else None,
        label="Approach A",
    )
    approach_b_dropdown = mo.ui.dropdown(
        options=approaches,
        value=approaches[1] if len(approaches) > 1 else (approaches[0] if approaches else None),
        label="Approach B",
    )
    radius_dropdown = mo.ui.dropdown(
        options=["25", "50", "100"],
        value="50",
        label="Convergence radius (m)",
    )

    mo.hstack(
        [approach_a_dropdown, approach_b_dropdown, radius_dropdown],
        justify="start",
        gap=1,
    )
    return (
        approach_a_dropdown,
        approach_b_dropdown,
        approaches,
        radius_dropdown,
        result_dir,
    )


@app.cell
def _(
    approach_a_dropdown,
    approach_b_dropdown,
    json,
    pl,
    radius_dropdown,
    result_dir,
    torch,
):
    _radius = radius_dropdown.value
    _key = f"convergence_cost_{_radius}m"

    def _load_summary(approach_name):
        path = result_dir / approach_name / "summary_statistics.json"
        with open(path) as f:
            return json.load(f)

    summary_a = _load_summary(approach_a_dropdown.value)
    summary_b = _load_summary(approach_b_dropdown.value)

    costs_a = summary_a[_key]
    costs_b = summary_b[_key]
    n_paths = min(len(costs_a), len(costs_b))

    # Load final errors for tooltips
    def _load_final_errors(approach_name, n):
        errors = []
        for i in range(n):
            p = result_dir / approach_name / f"{i:07d}" / "error.pt"
            try:
                err = torch.load(p, map_location="cpu")
                errors.append(float(err[-1]))
            except FileNotFoundError:
                errors.append(float("nan"))
        return errors

    final_errors_a = _load_final_errors(approach_a_dropdown.value, n_paths)
    final_errors_b = _load_final_errors(approach_b_dropdown.value, n_paths)

    df = pl.DataFrame({
        "path_idx": list(range(n_paths)),
        "approach_a_cost": costs_a[:n_paths],
        "approach_b_cost": costs_b[:n_paths],
        "final_error_a": final_errors_a,
        "final_error_b": final_errors_b,
    })
    df
    return (df,)


@app.cell
def _(alt, approach_a_dropdown, approach_b_dropdown, df, mo, radius_dropdown):
    df_pd = df.to_pandas()
    _max_val = max(df_pd["approach_a_cost"].max(), df_pd["approach_b_cost"].max())

    _label_a = approach_a_dropdown.value
    _label_b = approach_b_dropdown.value
    _radius = radius_dropdown.value

    # Add a y=x diagonal column so we can draw the reference line from the same data
    df_pd["diag"] = df_pd["approach_a_cost"]

    _base = alt.Chart(df_pd).encode(
        x=alt.X("approach_a_cost:Q", title=f"{_label_a} cost ({_radius}m)"),
    )

    _points = (
        _base
        .mark_point(filled=True, size=40, opacity=0.3)
        .encode(
            y=alt.Y("approach_b_cost:Q", title=f"{_label_b} cost ({_radius}m)"),
            tooltip=[
                alt.Tooltip("path_idx:Q", title="Path"),
                alt.Tooltip("approach_a_cost:Q", title=f"{_label_a} cost", format=".0f"),
                alt.Tooltip("approach_b_cost:Q", title=f"{_label_b} cost", format=".0f"),
                alt.Tooltip("final_error_a:Q", title=f"{_label_a} final err (m)", format=".1f"),
                alt.Tooltip("final_error_b:Q", title=f"{_label_b} final err (m)", format=".1f"),
            ],
        )
    )

    _diag = (
        _base
        .mark_line(strokeDash=[4, 4], color="gray", opacity=0.5)
        .encode(y="diag:Q")
    )

    _chart = (
        (_points + _diag)
        .properties(
            width=600,
            height=600,
            title=f"Per-path convergence cost comparison ({_radius}m radius)",
        )
    )

    scatter_chart = mo.ui.altair_chart(_chart)
    scatter_chart
    return df_pd, scatter_chart


@app.cell
def _(
    approach_a_dropdown,
    approach_b_dropdown,
    df,
    json,
    mo,
    np,
    plt,
    result_dir,
):
    _label_a = approach_a_dropdown.value
    _label_b = approach_b_dropdown.value
    _n_total = len(df)
    _n_worst = _n_total // 2

    # Load full summaries to get costs at all radii
    def _load_summary(name):
        with open(result_dir / name / "summary_statistics.json") as f:
            return json.load(f)

    _sum_a = _load_summary(_label_a)
    _sum_b = _load_summary(_label_b)

    # Final errors from existing df (radius-independent)
    _final_err_a = df["final_error_a"].to_numpy()
    _final_err_b = df["final_error_b"].to_numpy()

    def _delta(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        d = b - a
        return d[np.isfinite(d)]

    def _summarize(d):
        if len(d) == 0:
            return {"win": 0, "mean": 0, "median": 0, "q25": 0, "q75": 0}
        return {
            "win": float((d < 0).sum()) / len(d) * 100,
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "q25": float(np.percentile(d, 25)),
            "q75": float(np.percentile(d, 75)),
        }

    def _fmt_row(label, s, fmt="+.0f", rescue=None):
        r = f"| {label} | {s['win']:.1f}% | {s['mean']:{fmt}} | {s['median']:{fmt}} | {s['q25']:{fmt}} | {s['q75']:{fmt}}"
        if rescue is not None:
            r += f" | {rescue:.1f}%"
        return r + " |"

    # Compute stats for each radius
    _radii = ["25", "50", "100"]
    _rows_all = []
    _rows_worst = []
    _rows_best = []
    _hist_data = {}

    for _r in _radii:
        _key = f"convergence_cost_{_r}m"
        _ca = np.array(_sum_a[_key][:_n_total], dtype=float)
        _cb = np.array(_sum_b[_key][:_n_total], dtype=float)

        # All paths
        _pct_all = _delta(_ca, _cb)
        _rows_all.append(_fmt_row(f"{_r}m", _summarize(_pct_all)))

        _sorted_idx = np.argsort(_ca)[::-1]  # descending by baseline cost

        # Worst 50% by this radius's baseline cost
        _worst_idx = _sorted_idx[:_n_worst]
        _pct_w = _delta(_ca[_worst_idx], _cb[_worst_idx])
        _s_w = _summarize(_pct_w)

        _med_cost = float(np.median(_ca))
        _rescued = np.sum((_ca[_worst_idx] > _med_cost) & (_cb[_worst_idx] <= _med_cost))
        _rescue = float(_rescued) / _n_worst * 100

        _rows_worst.append(_fmt_row(f"{_r}m", _s_w, _rescue))
        _hist_data[_r] = {"d": _pct_w, "median": _s_w["median"]}

        # Best 50% (the other half)
        _best_idx = _sorted_idx[_n_worst:]
        _pct_b = _delta(_ca[_best_idx], _cb[_best_idx])
        _rows_best.append(_fmt_row(f"{_r}m", _summarize(_pct_b)))

    # Final error stats
    _pct_err_all = _delta(_final_err_a, _final_err_b)
    _se_all = _summarize(_pct_err_all)

    # Worst/best 50% final error (use 50m cost to define subsets)
    _ca_50 = np.array(_sum_a["convergence_cost_50m"][:_n_total], dtype=float)
    _sorted_50_idx = np.argsort(_ca_50)[::-1]
    _worst_50_idx = _sorted_50_idx[:_n_worst]
    _best_50_idx = _sorted_50_idx[_n_worst:]
    _pct_err_w = _delta(_final_err_a[_worst_50_idx], _final_err_b[_worst_50_idx])
    _se_w = _summarize(_pct_err_w)
    _pct_err_b = _delta(_final_err_a[_best_50_idx], _final_err_b[_best_50_idx])
    _se_b = _summarize(_pct_err_b)

    # Dump path indexes to /tmp sorted by baseline toughness (50m cost, descending)
    _path_idxs = df["path_idx"].to_numpy()
    _worst_paths = _path_idxs[_worst_50_idx].tolist()
    _best_paths = _path_idxs[_best_50_idx].tolist()
    with open("/tmp/worst_50pct_path_idxs.json", "w") as _f:
        json.dump({"description": f"Worst 50% of {_label_a} by 50m convergence cost (hardest first)",
                    "path_idxs": _worst_paths}, _f, indent=2)
    with open("/tmp/best_50pct_path_idxs.json", "w") as _f:
        json.dump({"description": f"Best 50% of {_label_a} by 50m convergence cost (hardest first)",
                    "path_idxs": _best_paths}, _f, indent=2)

    _cost_hdr = "| Radius | Win% | Mean Δ | Median Δ | Q25 Δ | Q75 Δ |"
    _cost_sep = "|--------|-----:|-------:|--------:|------:|------:|"
    _cost_hdr_r = "| Radius | Win% | Mean Δ | Median Δ | Q25 Δ | Q75 Δ | Rescue |"
    _cost_sep_r = "|--------|-----:|-------:|--------:|------:|------:|-------:|"
    _err_hdr = "| Subset | Win% | Mean Δ (m) | Median Δ (m) | Q25 Δ (m) | Q75 Δ (m) |"
    _err_sep = "|--------|-----:|-----------:|-------------:|----------:|----------:|"

    _md = "\n".join([
        f"### Δ analysis: {_label_b} vs {_label_a}",
        f"*Δ = {_label_b} − {_label_a}. Negative Δ = {_label_b} is better.*",
        "",
        "#### Convergence cost (integrated 1−P over distance, m) -- All paths",
        "",
        _cost_hdr,
        _cost_sep,
    ] + _rows_all + [
        "",
        "#### Convergence cost -- Worst 50% of baseline",
        "",
        _cost_hdr_r,
        _cost_sep_r,
    ] + _rows_worst + [
        "",
        "#### Convergence cost -- Best 50% of baseline",
        "",
        _cost_hdr,
        _cost_sep,
    ] + _rows_best + [
        "",
        "#### Final localization error -- subsets by 50m cost",
        "",
        _err_hdr,
        _err_sep,
        _fmt_row("All paths", _se_all, fmt="+.1f"),
        _fmt_row("Worst 50%", _se_w, fmt="+.1f"),
        _fmt_row("Best 50%", _se_b, fmt="+.1f"),
        "",
        f"*Rescue = fraction of worst-50% paths where {_label_b} cost falls below {_label_a} median cost.*",
    ])

    # Histograms: one per radius + final error
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
    _colors = ["#1f77b4", "#2ca02c", "#9467bd"]

    for _ax, _r, _c in zip(_axes.flat[:3], _radii, _colors):
        _d = _hist_data[_r]
        _ax.hist(_d["d"], bins=40, edgecolor="black", alpha=0.7, color=_c)
        _ax.axvline(0, color="red", linestyle="--", linewidth=1)
        _ax.axvline(_d["median"], color="orange", linewidth=2,
                    label=f"median: {_d['median']:+.0f}")
        _ax.set_xlabel(f"Δ convergence cost ({_r}m)")
        _ax.set_ylabel("Count")
        _ax.set_title(f"{_r}m radius")
        _ax.legend()

    _ax_err = _axes.flat[3]
    if len(_pct_err_w) > 0:
        _ax_err.hist(_pct_err_w, bins=40, edgecolor="black", alpha=0.7, color="orange")
        _ax_err.axvline(0, color="red", linestyle="--", linewidth=1)
        _ax_err.axvline(_se_w["median"], color="orange", linewidth=2,
                        label=f"median: {_se_w['median']:+.1f}m")
    _ax_err.set_xlabel("Δ final error (m)")
    _ax_err.set_ylabel("Count")
    _ax_err.set_title("Final localization error")
    _ax_err.legend()

    _fig.suptitle(f"Δ (B − A) on worst 50% of {_label_a} paths", fontsize=14)
    _fig.tight_layout()

    _output = mo.vstack([mo.md(_md), mo.mpl.interactive(_fig)])
    _output
    return


@app.cell
def _(mo):
    get_step_idx, set_step_idx = mo.state(0)
    return get_step_idx, set_step_idx


@app.cell
def _(
    approach_a_dropdown,
    approach_b_dropdown,
    df_pd,
    result_dir,
    scatter_chart,
    set_step_idx,
    torch,
):
    # Load per-path data when a point is selected
    _sel_df = scatter_chart.apply_selection(df_pd)

    if len(_sel_df) == 0 or len(_sel_df) == len(df_pd):
        selected_path_idx = None
        path_pano_ids = []
        res_a = None
        res_b = None
    else:
        selected_path_idx = int(_sel_df.iloc[0]["path_idx"])

        def _load_results(approach_name, idx):
            path_dir = result_dir / approach_name / f"{idx:07d}"

            def _load_cpu(p):
                t = torch.load(p, map_location="cpu")
                return t.cpu() if hasattr(t, "cpu") else t

            results = {
                "error": _load_cpu(path_dir / "error.pt"),
                "distance_traveled_m": _load_cpu(path_dir / "distance_traveled_m.pt"),
            }
            path_file = path_dir / "path.pt"
            if path_file.exists():
                results["path"] = torch.load(path_file, map_location="cpu")
            pmr_path = path_dir / "prob_mass_by_radius.pt"
            if pmr_path.exists():
                results["prob_mass_by_radius"] = torch.load(pmr_path, map_location="cpu")
            return results

        res_a = _load_results(approach_a_dropdown.value, selected_path_idx)
        res_b = _load_results(approach_b_dropdown.value, selected_path_idx)
        path_pano_ids = res_a.get("path", [])
        set_step_idx(0)
    return path_pano_ids, res_a, res_b, selected_path_idx


@app.cell
def _(mo, np, path_pano_ids, res_a, selected_path_idx, set_step_idx):
    if selected_path_idx is None or len(path_pano_ids) == 0:
        _output = mo.md("**Click a point in the scatter plot above to see per-path drill-down.**")
    else:
        _dist = res_a["distance_traveled_m"].numpy()
        _max_dist = int(_dist[-1]) if len(_dist) > 0 else 0
        _n_steps = len(path_pano_ids)

        def _on_seek(dist_value):
            idx = int(np.searchsorted(_dist, dist_value, side="right"))
            set_step_idx(min(idx, _n_steps - 1))

        _slider = mo.ui.slider(
            start=0,
            stop=_max_dist,
            value=0,
            on_change=_on_seek,
            label="Seek distance along path (m)",
            full_width=True,
        )
        _output = _slider
    _output
    return


@app.cell(hide_code=True)
def _(
    approach_a_dropdown,
    approach_b_dropdown,
    get_step_idx,
    mo,
    np,
    plt,
    res_a,
    res_b,
    selected_path_idx,
):
    if selected_path_idx is None:
        _output = mo.md("")
    else:
        _label_a = approach_a_dropdown.value
        _label_b = approach_b_dropdown.value

        _dist_a = res_a["distance_traveled_m"].numpy()
        _step_idx = min(get_step_idx(), len(_dist_a) - 1)
        _cursor_dist = float(_dist_a[_step_idx])

        # --- Convergence figure ---
        _fig_conv, (_ax_a_conv, _ax_b_conv) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for _res, _ax, _label in [(res_a, _ax_a_conv, _label_a), (res_b, _ax_b_conv, _label_b)]:
            if "prob_mass_by_radius" in _res:
                _dist = _res["distance_traveled_m"].numpy()
                _colors = plt.cm.viridis(np.linspace(0, 0.8, len(_res["prob_mass_by_radius"])))
                for (_radius, _pm), _c in zip(sorted(_res["prob_mass_by_radius"].items()), _colors):
                    _pm_np = _pm.numpy()
                    _ml = min(len(_pm_np), len(_dist))
                    _ax.plot(_dist[:_ml], _pm_np[:_ml], linewidth=2, color=_c, label=f"{_radius}m")
                _ax.legend(loc="lower right")
            _ax.axvline(_cursor_dist, color="red", linewidth=1, linestyle="--", alpha=0.7)
            _ax.set_xlabel("Distance Traveled (m)")
            _ax.set_title(f"{_label}")
            _ax.set_ylim(0, 1.05)
            _ax.grid(True, alpha=0.3)

        _ax_a_conv.set_ylabel("Probability Mass Within Radius")
        _fig_conv.suptitle(f"Path {selected_path_idx}: Convergence Curves", fontsize=14)
        _fig_conv.tight_layout()

        # --- Error figure ---
        _fig_err, (_ax_a_err, _ax_b_err) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for _res, _ax, _label in [(res_a, _ax_a_err, _label_a), (res_b, _ax_b_err, _label_b)]:
            _err = _res["error"].numpy()
            _dist = _res["distance_traveled_m"].numpy()
            _ml = min(len(_err), len(_dist))
            _ax.plot(_dist[:_ml], _err[:_ml], "b-", linewidth=2)
            _ax.scatter(_dist[:_ml], _err[:_ml], c="blue", s=20)
            _ax.axvline(_cursor_dist, color="red", linewidth=1, linestyle="--", alpha=0.7)
            _ax.set_xlabel("Distance Traveled (m)")
            _ax.set_title(f"{_label} (final: {_err[-1]:.1f}m)")
            _ax.grid(True, alpha=0.3)

        _ax_a_err.set_ylabel("Error (m)")
        _fig_err.suptitle(f"Path {selected_path_idx}: Localization Error", fontsize=14)
        _fig_err.tight_layout()

        _output = mo.vstack([
            mo.md(f"### Path {selected_path_idx} drill-down"),
            mo.mpl.interactive(_fig_conv), mo.mpl.interactive(_fig_err),
        ])
    _output
    return


@app.cell(hide_code=True)
def _(Path, np, pickle):
    # Build pano_id -> (lat, lon, filename) lookup from directory listing (once)
    _pano_dir = Path("/data/overhead_matching/datasets/VIGOR/Seattle/panorama")
    pano_latlon = {}
    _all_lats = []
    _all_lons = []
    for _f in _pano_dir.iterdir():
        if _f.suffix == ".jpg":
            _parts = _f.stem.split(",")
            if len(_parts) >= 3:
                _lat, _lon = float(_parts[1]), float(_parts[2])
                pano_latlon[_parts[0]] = (_lat, _lon, _f.name)
                _all_lats.append(_lat)
                _all_lons.append(_lon)
    all_pano_lats = np.array(_all_lats)
    all_pano_lons = np.array(_all_lons)

    # Load pano_gemini landmark data once (keyed by "pano_id,lat,lon,")
    _pkl_path = Path("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_gemini/Seattle/embeddings/embeddings.pkl")
    if _pkl_path.exists():
        with open(_pkl_path, 'rb') as _f2:
            _data = pickle.load(_f2)
        pano_landmarks = {}
        for _k, _v in _data.get("panoramas", {}).items():
            _pano_id = _k.split(",")[0]
            pano_landmarks[_pano_id] = _v
    else:
        pano_landmarks = {}
    return all_pano_lats, all_pano_lons, pano_landmarks, pano_latlon


@app.cell
def _(get_step_idx, mo, path_pano_ids, selected_path_idx, set_step_idx):
    if selected_path_idx is None or len(path_pano_ids) == 0:
        _output = mo.md("")
    else:
        _n_steps = len(path_pano_ids)
        _step = min(get_step_idx(), _n_steps - 1)
        _prev_btn = mo.ui.button(
            label="< Prev",
            on_click=lambda _: set_step_idx(max(0, get_step_idx() - 1)),
            disabled=_step <= 0,
        )
        _next_btn = mo.ui.button(
            label="Next >",
            on_click=lambda _: set_step_idx(min(_n_steps - 1, get_step_idx() + 1)),
            disabled=_step >= _n_steps - 1,
        )
        _output = mo.hstack(
            [_prev_btn, mo.md(f"**Step {_step} / {_n_steps - 1}**"), _next_btn],
            justify="center",
            gap=1,
        )
    _output
    return


@app.cell(hide_code=True)
def _(
    Path,
    approach_a_dropdown,
    approach_b_dropdown,
    get_step_idx,
    mo,
    pano_landmarks,
    pano_latlon,
    path_pano_ids,
    res_a,
    res_b,
    selected_path_idx,
):
    if selected_path_idx is None or len(path_pano_ids) == 0:
        _output = mo.md("")
    else:
        _step = min(get_step_idx(), len(path_pano_ids) - 1)
        _pano_id = path_pano_ids[_step]

        # Find the panorama image file via lookup
        _pano_dir = Path("/data/overhead_matching/datasets/VIGOR/Seattle/panorama")
        _entry = pano_latlon.get(_pano_id)
        if _entry:
            _pano_img = mo.image(src=str(_pano_dir / _entry[2]), width=1024)
        else:
            _pano_img = mo.md(f"*Panorama not found for `{_pano_id}`*")

        # Show error at this step for both approaches
        _label_a = approach_a_dropdown.value
        _label_b = approach_b_dropdown.value
        _err_a = res_a["error"].numpy()
        _err_b = res_b["error"].numpy()
        _dist_a = res_a["distance_traveled_m"].numpy()
        _err_text = ""
        if _step < len(_err_a):
            _err_text += f"**{_label_a}:** {_err_a[_step]:.1f}m"
        if _step < len(_err_b):
            _err_text += f" | **{_label_b}:** {_err_b[_step]:.1f}m"
        _dist_text = f"{_dist_a[_step]:.0f}m" if _step < len(_dist_a) else "?"

        # Look up landmarks
        _pano_entry = pano_landmarks.get(_pano_id)
        if _pano_entry:
            _loc_type = _pano_entry.get("location_type", "unknown")
            _landmarks = _pano_entry.get("landmarks", [])
            _lines = [f"**Location type:** {_loc_type}", ""]
            for _i, _lm in enumerate(_landmarks):
                _desc = _lm.get("description", "")
                _pns = _lm.get("proper_nouns", [])
                _pn_str = f' — **{", ".join(_pns)}**' if _pns else ""
                _lines.append(f"{_i + 1}. {_desc}{_pn_str}")
            _landmarks_md = "\n".join(_lines)
        else:
            _landmarks_md = f"*No landmark data found for `{_pano_id}`*"

        _output = mo.vstack([
            mo.md(f"### Step {_step} ({_dist_text} traveled): `{_pano_id}` | {_err_text}"),
            _pano_img,
            mo.md(_landmarks_md),
        ])
    _output
    return


@app.cell(hide_code=True)
def _(
    all_pano_lats,
    all_pano_lons,
    get_step_idx,
    mo,
    np,
    pano_latlon,
    path_pano_ids,
    plt,
    selected_path_idx,
):
    if selected_path_idx is None or len(path_pano_ids) == 0:
        _output = mo.md("")
    else:
        # Look up path lat/lons from pre-built dict (fast)
        _lats = np.array([pano_latlon[p][0] if p in pano_latlon else float("nan")
                          for p in path_pano_ids])
        _lons = np.array([pano_latlon[p][1] if p in pano_latlon else float("nan")
                          for p in path_pano_ids])
        _step = min(get_step_idx(), len(path_pano_ids) - 1)

        _fig, _ax = plt.subplots(1, 1, figsize=(8, 8))

        # All dataset points as background
        _ax.plot(all_pano_lons, all_pano_lats, ",", color="0.75", zorder=1,
                 rasterized=True)

        # Traversed portion (bold)
        if _step > 0:
            _ax.plot(_lons[:_step + 1], _lats[:_step + 1], "-",
                     color="#1f77b4", linewidth=3, alpha=0.9, label="Traversed")
        # Remaining portion (faint)
        if _step < len(_lons) - 1:
            _ax.plot(_lons[_step:], _lats[_step:], "-",
                     color="#1f77b4", linewidth=1.5, alpha=0.3, label="Remaining")

        _ax.scatter(_lons[0], _lats[0], c="green", s=100, zorder=10,
                    marker="o", edgecolors="black", label="Start")
        _ax.scatter(_lons[-1], _lats[-1], c="black", s=60, zorder=9,
                    marker="s", label="End")
        _ax.scatter(_lons[_step], _lats[_step], c="red", s=200, zorder=11,
                    marker="*", edgecolors="darkred", label=f"Step {_step}")

        _ax.set_xlabel("Longitude")
        _ax.set_ylabel("Latitude")
        _ax.set_title(f"Path {selected_path_idx}")
        _ax.legend(loc="best")

        # Approximate equal-distance aspect at this latitude
        _mid_lat = np.nanmean(_lats)
        _ax.set_aspect(1.0 / np.cos(np.radians(_mid_lat)))
        _ax.grid(True, alpha=0.3)
        _fig.tight_layout()

        _output = mo.mpl.interactive(_fig)
    _output
    return


@app.cell
def _(approaches, mo):
    bw_approach_dropdown = mo.ui.dropdown(
        options=approaches,
        value=approaches[0] if approaches else None,
        label="Approach",
    )
    bw_n_slider = mo.ui.slider(
        start=5, stop=50, value=10, step=5,
        label="Number of paths",
    )
    mo.hstack([
        mo.md("### Best / worst paths"),
        bw_approach_dropdown,
        bw_n_slider,
    ], justify="start", gap=1)
    return bw_approach_dropdown, bw_n_slider


@app.cell(hide_code=True)
def _(
    all_pano_lats,
    all_pano_lons,
    bw_approach_dropdown,
    bw_n_slider,
    json,
    mo,
    np,
    pano_latlon,
    plt,
    radius_dropdown,
    result_dir,
    torch,
):
    _approach = bw_approach_dropdown.value
    _radius = radius_dropdown.value
    _n = bw_n_slider.value
    _key = f"convergence_cost_{_radius}m"

    with open(result_dir / _approach / "summary_statistics.json") as _f:
        _summary = json.load(_f)
    _costs = np.array(_summary[_key], dtype=float)
    _sorted_idx = np.argsort(_costs)
    _best_idxs = _sorted_idx[:_n]
    _worst_idxs = _sorted_idx[-_n:][::-1]

    def _load_path_latlons(idx):
        _p = result_dir / _approach / f"{idx:07d}" / "path.pt"
        if not _p.exists():
            return np.array([]), np.array([])
        _pids = torch.load(_p, map_location="cpu")
        _la = [pano_latlon[p][0] for p in _pids if p in pano_latlon]
        _lo = [pano_latlon[p][1] for p in _pids if p in pano_latlon]
        return np.array(_la), np.array(_lo)

    _cmap = plt.cm.tab10
    _fig, (_ax_best, _ax_worst) = plt.subplots(1, 2, figsize=(16, 8))

    for _ax, _idxs, _title in [
        (_ax_best, _best_idxs, f"Best {_n}"),
        (_ax_worst, _worst_idxs, f"Worst {_n}"),
    ]:
        _ax.plot(all_pano_lons, all_pano_lats, ",", color="0.85", zorder=1,
                 rasterized=True)
        for _i, _idx in enumerate(_idxs):
            _la, _lo = _load_path_latlons(_idx)
            if len(_la) == 0:
                continue
            _c = _cmap(_i % 10)
            _ax.plot(_lo, _la, "-", color=_c, linewidth=2, alpha=0.8)
            _ax.scatter(_lo[0], _la[0], c=[_c], s=40, zorder=5,
                        marker="o", edgecolors="black", linewidths=0.5)
            _ax.annotate(str(_idx), (_lo[0], _la[0]), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")

        _ax.set_xlabel("Longitude")
        _ax.set_ylabel("Latitude")
        _ax.set_title(f"{_title} ({_approach}, {_radius}m cost)")
        _mid_lat = np.nanmean(all_pano_lats)
        _ax.set_aspect(1.0 / np.cos(np.radians(_mid_lat)))
        _ax.grid(True, alpha=0.3)

    _fig.tight_layout()
    _output = mo.mpl.interactive(_fig)
    _output
    return


@app.cell(hide_code=True)
def _(
    all_pano_lats,
    all_pano_lons,
    bw_approach_dropdown,
    json,
    mo,
    np,
    pano_latlon,
    plt,
    radius_dropdown,
    result_dir,
    torch,
):
    _approach = bw_approach_dropdown.value
    _radius = radius_dropdown.value
    _key = f"convergence_cost_{_radius}m"

    with open(result_dir / _approach / "summary_statistics.json") as _f:
        _summary = json.load(_f)
    _costs = np.array(_summary[_key], dtype=float)
    _n_total = len(_costs)
    _n_half = _n_total // 2
    _sorted_idx = np.argsort(_costs)
    _best_idxs = _sorted_idx[:_n_half]
    _worst_idxs = _sorted_idx[_n_half:]

    # Load start lat/lon for every path
    _start_lats = np.full(_n_total, np.nan)
    _start_lons = np.full(_n_total, np.nan)
    for _i in range(_n_total):
        _p = result_dir / _approach / f"{_i:07d}" / "path.pt"
        if not _p.exists():
            continue
        _pids = torch.load(_p, map_location="cpu")
        if len(_pids) > 0 and _pids[0] in pano_latlon:
            _start_lats[_i] = pano_latlon[_pids[0]][0]
            _start_lons[_i] = pano_latlon[_pids[0]][1]

    # Shared bin edges so both histograms are comparable
    _valid = np.isfinite(_start_lats) & np.isfinite(_start_lons)
    _lon_range = (np.nanmin(_start_lons[_valid]), np.nanmax(_start_lons[_valid]))
    _lat_range = (np.nanmin(_start_lats[_valid]), np.nanmax(_start_lats[_valid]))
    _n_bins = 20

    # Compute both histograms first to get shared vmax
    _mid_lat = np.nanmean(all_pano_lats)
    _hists = []
    for _idxs in [_best_idxs, _worst_idxs]:
        _sl = _start_lats[_idxs]
        _sn = _start_lons[_idxs]
        _m = np.isfinite(_sl) & np.isfinite(_sn)
        _h, _xe, _ye = np.histogram2d(
            _sn[_m], _sl[_m], bins=_n_bins,
            range=[_lon_range, _lat_range],
        )
        _hists.append((_h, _xe, _ye))

    _vmax = max(_hists[0][0].max(), _hists[1][0].max(), 1)
    _norm = plt.Normalize(vmin=0, vmax=_vmax)

    _fig = plt.figure(figsize=(16, 8))
    _gs = _fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.08)
    _ax_best = _fig.add_subplot(_gs[0, 0])
    _ax_worst = _fig.add_subplot(_gs[0, 1], sharey=_ax_best)
    _cax = _fig.add_subplot(_gs[0, 2])

    for _ax, (_h, _xe, _ye), _title in [
        (_ax_best, _hists[0], f"Best 50% starts"),
        (_ax_worst, _hists[1], f"Worst 50% starts"),
    ]:
        _ax.plot(all_pano_lons, all_pano_lats, ",", color="0.85", zorder=1,
                 rasterized=True)
        _h_masked = np.ma.masked_where(_h == 0, _h)
        _ax.pcolormesh(
            _xe, _ye, _h_masked.T,
            cmap="hot_r", norm=_norm, alpha=0.7, zorder=2,
        )
        _ax.set_xlabel("Longitude")
        _ax.set_title(f"{_title} ({_approach}, {_radius}m cost)")
        _ax.set_aspect(1.0 / np.cos(np.radians(_mid_lat)))
        _ax.grid(True, alpha=0.3)

    _ax_best.set_ylabel("Latitude")
    plt.setp(_ax_worst.get_yticklabels(), visible=False)

    _sm = plt.cm.ScalarMappable(cmap="hot_r", norm=_norm)
    _fig.colorbar(_sm, cax=_cax, label="Path starts per bin")
    _fig.suptitle(f"Start location density: best vs worst 50% ({_approach}, {_radius}m)",
                  fontsize=14)

    _output = mo.mpl.interactive(_fig)
    _output
    return


if __name__ == "__main__":
    app.run()
