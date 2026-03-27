import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    import common.torch.load_torch_deps
    import pickle
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path
    from PIL import Image
    from collections import defaultdict

    _args = mo.cli_args()
    EMBEDDINGS_BASE = Path(_args.get("embeddings-base", "/data/overhead_matching/datasets/semantic_landmark_embeddings"))
    PINHOLE_BASE = Path(_args.get("pinhole-base", "/data/overhead_matching/datasets/pinhole_images"))
    DEPTHS_PKL = Path(_args.get("depths-pkl", "/tmp/landmark_depths.pkl"))

    DATASETS = [
        ("mapillary/Framingham", "Framingham"),
        ("mapillary/MiamiBeach", "MiamiBeach"),
        ("mapillary/Gap", "Gap"),
        ("mapillary/Norway", "Norway"),
        ("mapillary/Middletown", "Middletown"),
        ("mapillary/SanFrancisco_mapillary", "SanFrancisco_mapillary"),
        ("mapillary/post_hurricane_ian", "post_hurricane_ian"),
        ("nightdrive", "nightdrive"),
        ("boston_snowy", "boston_snowy"),
    ]
    return (
        DATASETS,
        DEPTHS_PKL,
        EMBEDDINGS_BASE,
        Image,
        PINHOLE_BASE,
        defaultdict,
        mpatches,
        np,
        pickle,
        plt,
        shutil,
    )


@app.cell
def _(DATASETS, EMBEDDINGS_BASE, PINHOLE_BASE, mo, pickle):
    def load_v2_pickle(_pkl_path):
        if not _pkl_path.exists():
            return None
        with open(_pkl_path, "rb") as _f:
            _data = pickle.load(_f)
        if isinstance(_data, dict) and _data.get("version") != "2.0":
            raise RuntimeError(f"Not v2.0: {_pkl_path}")
        return _data

    city_data = {}
    _summary_rows = []
    for _rel_path, _pinhole_city in DATASETS:
        _pkl_path = EMBEDDINGS_BASE / _rel_path / "embeddings" / "embeddings.pkl"
        _data = load_v2_pickle(_pkl_path)
        if _data is None:
            continue
        _pinhole_dir = PINHOLE_BASE / _pinhole_city

        _panos = _data.get("panoramas", {})
        _num_landmarks = sum(len(_p.get("landmarks", [])) for _p in _panos.values())
        _num_bboxes = sum(
            len(_bbox_list)
            for _p in _panos.values()
            for _lm in _p.get("landmarks", [])
            for _bbox_list in [_lm.get("bounding_boxes", [])]
        )

        city_data[_pinhole_city] = {
            "pickle_path": _pkl_path,
            "pinhole_dir": _pinhole_dir,
            "data": _data,
        }
        _summary_rows.append({
            "City": _pinhole_city,
            "Panoramas": len(_panos),
            "Landmarks": _num_landmarks,
            "Bounding Boxes": _num_bboxes,
        })

    summary_table = mo.ui.table(_summary_rows, label="Dataset Summary")
    return (city_data,)


@app.cell
def _(DEPTHS_PKL, mo, np, pickle):
    if DEPTHS_PKL.exists():
        with open(DEPTHS_PKL, "rb") as _f:
            landmark_depths = pickle.load(_f)
        depths_loaded = True
        _all_depths = [_v["depth_m"] for _v in landmark_depths.values()]
        _output = mo.md(f"**Loaded {len(landmark_depths)} landmark depths** from `{DEPTHS_PKL}`\n\n"
                        f"Depth range: {min(_all_depths):.1f}m - {max(_all_depths):.1f}m, "
                        f"median: {np.median(_all_depths):.1f}m")
    else:
        landmark_depths = {}
        depths_loaded = False
        _output = mo.md(f"**No depths file found at `{DEPTHS_PKL}`.** "
                        "Run `compute_landmark_depths.py` first.")
    _output
    return depths_loaded, landmark_depths


@app.cell(hide_code=True)
def _(city_data, mo):
    city_dropdown = mo.ui.dropdown(
        options=sorted(city_data.keys()),
        value=sorted(city_data.keys())[0] if city_data else None,
        label="City",
    )
    pano_slider = mo.ui.slider(
        start=0, stop=20, value=0, step=1,
        label="Panorama index",
    )
    mo.vstack([mo.md("## Bounding Box Visualization"), mo.hstack([city_dropdown, pano_slider])])
    return city_dropdown, pano_slider


@app.cell
def _(
    Image,
    city_data,
    city_dropdown,
    landmark_depths,
    mo,
    mpatches,
    np,
    pano_slider,
    plt,
):
    _selected_city = city_dropdown.value
    _pano_idx = pano_slider.value

    _output = mo.md("Select a city above.")

    if _selected_city is not None and _selected_city in city_data:
        _info = city_data[_selected_city]
        _panos = _info["data"].get("panoramas", {})
        _pano_keys = sorted(_panos.keys())

        if _pano_idx >= len(_pano_keys):
            _output = mo.md(f"Panorama index {_pano_idx} out of range (max {len(_pano_keys) - 1})")
        else:
            _pano_key = _pano_keys[_pano_idx]
            _pano_data = _panos[_pano_key]
            _landmarks = _pano_data.get("landmarks", [])

            # Group bboxes by yaw angle
            _yaw_bboxes = {}
            for _lm in _landmarks:
                for _bbox in _lm.get("bounding_boxes", []):
                    _yaw = _bbox.get("yaw_angle", "")
                    if _yaw not in {"0", "90", "180", "270"}:
                        continue
                    if _yaw not in _yaw_bboxes:
                        _yaw_bboxes[_yaw] = []
                    _yaw_bboxes[_yaw].append((
                        _lm["landmark_idx"],
                        _bbox,
                        _lm.get("description", "")[:60],
                    ))

            _yaws_to_show = sorted(_yaw_bboxes.keys(), key=int)
            if not _yaws_to_show:
                _yaws_to_show = ["0"]

            _colors = plt.cm.tab10(np.linspace(0, 1, max(len(_landmarks), 1)))

            _DISPLAY_SIZE = 512
            _scale = _DISPLAY_SIZE / 2048

            _fig, _axes = plt.subplots(1, len(_yaws_to_show), figsize=(6 * len(_yaws_to_show), 6), dpi=100)
            if len(_yaws_to_show) == 1:
                _axes = [_axes]

            for _ax, _yaw_str in zip(_axes, _yaws_to_show):
                _img_path = _info["pinhole_dir"] / _pano_key / f"yaw_{int(_yaw_str):03d}.jpg"
                if _img_path.exists():
                    _img = Image.open(_img_path).resize((_DISPLAY_SIZE, _DISPLAY_SIZE))
                    _ax.imshow(_img)
                else:
                    _ax.text(0.5, 0.5, f"Image not found:\n{_img_path}",
                             ha="center", va="center", transform=_ax.transAxes)

                _ax.set_title(f"Yaw {_yaw_str}")

                for _lm_idx, _bbox, _desc in _yaw_bboxes.get(_yaw_str, []):
                    _xmin = _bbox["xmin"] * _DISPLAY_SIZE / 1000
                    _ymin = _bbox["ymin"] * _DISPLAY_SIZE / 1000
                    _xmax = _bbox["xmax"] * _DISPLAY_SIZE / 1000
                    _ymax = _bbox["ymax"] * _DISPLAY_SIZE / 1000

                    _color = _colors[_lm_idx % len(_colors)]

                    _depth_key = (_selected_city, _pano_key, _lm_idx)
                    _depth_str = ""
                    if _depth_key in landmark_depths:
                        _depth_str = f" [{landmark_depths[_depth_key]['depth_m']:.1f}m]"

                    _rect = mpatches.FancyBboxPatch(
                        (_xmin, _ymin), _xmax - _xmin, _ymax - _ymin,
                        linewidth=2, edgecolor=_color, facecolor="none",
                        boxstyle="square,pad=0",
                    )
                    _ax.add_patch(_rect)
                    _ax.text(_xmin, _ymin - 5, f"#{_lm_idx}: {_desc}{_depth_str}",
                             color=_color, fontsize=6, weight="bold",
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                _ax.axis("off")

            _fig.suptitle(f"{_selected_city} - {_pano_key}\n{len(_landmarks)} landmarks", fontsize=12)
            plt.tight_layout()
            _output = mo.vstack([
                mo.md(f"Panorama {_pano_idx}/{len(_pano_keys)-1}: `{_pano_key}`"),
                _fig,
            ])

    _output
    return


@app.cell(hide_code=True)
def _(depths_loaded, mo):
    if depths_loaded:
        threshold_slider = mo.ui.slider(
            start=5, stop=200, value=50, step=5,
            label="Depth threshold (meters)",
        )
        _output = mo.vstack([mo.md("## Depth Statistics & Threshold Selection"), threshold_slider])
    else:
        threshold_slider = None
        _output = mo.md("*Depths not loaded — skipping statistics.*")
    _output
    return (threshold_slider,)


@app.cell
def _(
    defaultdict,
    depths_loaded,
    landmark_depths,
    mo,
    np,
    plt,
    threshold_slider,
):
    _output = mo.md("")

    if depths_loaded and threshold_slider is not None:
        _threshold = threshold_slider.value
        _all_d = [_v["depth_m"] for _v in landmark_depths.values()]

        _kept = [_d for _d in _all_d if _d <= _threshold]
        _removed = [_d for _d in _all_d if _d > _threshold]

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(16, 5))

        _ax1.hist(_all_d, bins=100, alpha=0.7, color="steelblue", edgecolor="white")
        _ax1.axvline(_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold: {_threshold}m")
        _ax1.set_xlabel("Depth (meters)")
        _ax1.set_ylabel("Count")
        _ax1.set_title("All Landmark Depths")
        _ax1.legend()

        _city_depths = defaultdict(list)
        for _key, _val in landmark_depths.items():
            _city_depths[_val["city"]].append(_val["depth_m"])

        _cities = sorted(_city_depths.keys())
        _kept_counts = []
        _removed_counts = []
        for _city in _cities:
            _ds = _city_depths[_city]
            _kept_counts.append(sum(1 for _d in _ds if _d <= _threshold))
            _removed_counts.append(sum(1 for _d in _ds if _d > _threshold))

        _x = np.arange(len(_cities))
        _ax2.bar(_x, _kept_counts, label="Kept", color="steelblue", alpha=0.8)
        _ax2.bar(_x, _removed_counts, bottom=_kept_counts, label="Removed", color="salmon", alpha=0.8)
        _ax2.set_xticks(_x)
        _ax2.set_xticklabels(_cities, rotation=45, ha="right", fontsize=8)
        _ax2.set_ylabel("Landmarks")
        _ax2.set_title("Per-City Breakdown")
        _ax2.legend()

        plt.tight_layout()

        _table_rows = []
        for _city in _cities:
            _ds = _city_depths[_city]
            _k = sum(1 for _d in _ds if _d <= _threshold)
            _r = sum(1 for _d in _ds if _d > _threshold)
            _table_rows.append({
                "City": _city,
                "Total": len(_ds),
                "Kept": _k,
                "Removed": _r,
                "% Removed": f"{100*_r/len(_ds):.1f}%",
                "Median Depth": f"{np.median(_ds):.1f}m",
            })

        _tag_depths = defaultdict(list)
        for _key, _val in landmark_depths.items():
            _tag = _val.get("primary_tag", {})
            _tag_key = _tag.get("key", "unknown") if isinstance(_tag, dict) else "unknown"
            _tag_depths[_tag_key].append(_val["depth_m"])

        _tag_rows = []
        for _tag_key in sorted(_tag_depths.keys(), key=lambda k: -len(_tag_depths[k])):
            _ds = _tag_depths[_tag_key]
            _r = sum(1 for _d in _ds if _d > _threshold)
            _tag_rows.append({
                "Tag": _tag_key,
                "Count": len(_ds),
                "Removed": _r,
                "% Removed": f"{100*_r/len(_ds):.1f}%",
                "Median Depth": f"{np.median(_ds):.1f}m",
            })

        _output = mo.vstack([
            mo.md(f"### At threshold **{_threshold}m**: "
                  f"keep **{len(_kept)}** ({100*len(_kept)/len(_all_d):.1f}%), "
                  f"remove **{len(_removed)}** ({100*len(_removed)/len(_all_d):.1f}%)"),
            _fig,
            mo.ui.table(_table_rows, label="Per-City Breakdown"),
            mo.ui.table(_tag_rows, label="By Primary Tag"),
        ])

    _output
    return


@app.cell(hide_code=True)
def _(depths_loaded, landmark_depths, mo):
    if depths_loaded:
        preview_num = mo.ui.number(
            start=0, stop=max(len(landmark_depths) - 1, 1), value=0, step=1,
            label="Index",
        )
        _output = mo.vstack([
            mo.md("## Preview All Landmarks (sorted deepest first)"),
            preview_num,
        ])
    else:
        preview_num = None
        _output = mo.md("")
    _output
    return (preview_num,)


@app.cell(hide_code=True)
def _(
    Image,
    city_data,
    depths_loaded,
    landmark_depths,
    mo,
    mpatches,
    np,
    plt,
    preview_num,
    threshold_slider,
):
    _output = mo.md("")

    if depths_loaded and threshold_slider is not None and preview_num is not None:
        _threshold = threshold_slider.value
        _items = sorted(
            landmark_depths.items(),
            key=lambda kv: -kv[1]["depth_m"],
        )
        _idx = preview_num.value
        if _idx >= len(_items):
            _output = mo.md("Index out of range")
        else:
            _key, _val = _items[_idx]
            _city, _pano_key, _lm_idx = _key
            _depth_m = _val["depth_m"]
            _desc = _val.get("description", "")
            _status = "KEEP" if _depth_m <= _threshold else "REMOVE"
            _color = "green" if _depth_m <= _threshold else "red"

            _info = city_data.get(_city)
            if _info is None:
                _output = mo.md(f"City {_city} not in loaded data")
            else:
                _panos = _info["data"].get("panoramas", {})
                _pano_data = _panos.get(_pano_key)
                if _pano_data is None:
                    _pano_id = _pano_key.split(",")[0]
                    for _pk, _pd in _panos.items():
                        if _pk.startswith(_pano_id):
                            _pano_data = _pd
                            _pano_key = _pk
                            break

                if _pano_data is None:
                    _output = mo.md(f"Panorama {_pano_key} not found")
                else:
                    _landmark = None
                    for _lm in _pano_data.get("landmarks", []):
                        if _lm["landmark_idx"] == _lm_idx:
                            _landmark = _lm
                            break

                    if _landmark is None:
                        _output = mo.md(f"Landmark {_lm_idx} not found in panorama")
                    else:
                        _bboxes = _landmark.get("bounding_boxes", [])
                        _yaw_bboxes = {}
                        for _bbox in _bboxes:
                            _yaw = _bbox.get("yaw_angle", "")
                            if _yaw in {"0", "90", "180", "270"}:
                                _yaw_bboxes.setdefault(_yaw, []).append(_bbox)

                        _DISPLAY_SIZE = 512
                        _yaws = sorted(_yaw_bboxes.keys(), key=int) or ["0"]
                        _n_yaws = len(_yaws)

                        # 2 rows: top = RGB with bboxes, bottom = depth map
                        _fig, _axes = plt.subplots(2, _n_yaws, figsize=(6 * _n_yaws, 10), dpi=100)
                        if _n_yaws == 1:
                            _axes = _axes.reshape(2, 1)

                        for _col, _yaw_str in enumerate(_yaws):
                            _img_path = _info["pinhole_dir"] / _pano_key / f"yaw_{int(_yaw_str):03d}.jpg"
                            _ax_rgb = _axes[0, _col]
                            _ax_depth = _axes[1, _col]

                            if _img_path.exists():
                                _full_img = Image.open(_img_path)
                                _ax_rgb.imshow(_full_img.resize((_DISPLAY_SIZE, _DISPLAY_SIZE)))

                                # Run depth model on this image
                                from experimental.learn_descriptors.depth_models import UniDepthV2 as _UDV2
                                if not hasattr(_UDV2, '_cached_model'):
                                    _UDV2._cached_model = _UDV2(device="cuda:0")
                                _depth_map = _UDV2._cached_model.infer(_img_path)
                                _depth_map = np.asarray(_depth_map, dtype=np.float32).squeeze()
                                _depth_resized = np.array(Image.fromarray(_depth_map).resize((_DISPLAY_SIZE, _DISPLAY_SIZE)))

                                _im = _ax_depth.imshow(_depth_resized, cmap='turbo')
                                _ax_depth.set_title(f"Depth {_depth_map.min():.1f}-{_depth_map.max():.1f}m")
                                plt.colorbar(_im, ax=_ax_depth, fraction=0.046, pad=0.04)
                            else:
                                _ax_rgb.text(0.5, 0.5, "not found", ha="center", va="center", transform=_ax_rgb.transAxes)

                            _ax_rgb.set_title(f"Yaw {_yaw_str}")

                            # Draw bboxes on both RGB and depth
                            for _bbox in _yaw_bboxes.get(_yaw_str, []):
                                _xmin = _bbox["xmin"] * _DISPLAY_SIZE / 1000
                                _ymin = _bbox["ymin"] * _DISPLAY_SIZE / 1000
                                _xmax = _bbox["xmax"] * _DISPLAY_SIZE / 1000
                                _ymax = _bbox["ymax"] * _DISPLAY_SIZE / 1000
                                for _ax in [_ax_rgb, _ax_depth]:
                                    _rect = mpatches.FancyBboxPatch(
                                        (_xmin, _ymin), _xmax - _xmin, _ymax - _ymin,
                                        linewidth=3, edgecolor=_color, facecolor="none",
                                        boxstyle="square,pad=0",
                                    )
                                    _ax.add_patch(_rect)

                            _ax_rgb.axis("off")
                            _ax_depth.axis("off")

                        _fig.suptitle(
                            f"[{_status}] {_city} / #{_lm_idx}: {_desc[:80]}\n"
                            f"Depth: {_depth_m:.1f}m (threshold: {_threshold}m)",
                            fontsize=12, color=_color,
                        )
                        plt.tight_layout()
                        _output = _fig

    _output
    return


@app.cell(hide_code=True)
def _(depths_loaded, mo):
    if depths_loaded:
        backup_button = mo.ui.run_button(label="Create backups")
        apply_button = mo.ui.run_button(label="Apply filter")
        _output = mo.vstack([
            mo.md("## Backup & Apply Filter"),
            mo.hstack([backup_button, apply_button]),
        ])
    else:
        backup_button = None
        apply_button = None
        _output = mo.md("")
    _output
    return apply_button, backup_button


@app.cell
def _(backup_button, city_data, mo, shutil):
    _output = mo.md("")
    if backup_button is not None and backup_button.value:
        _backup_results = []
        for _city, _info in city_data.items():
            _pkl_path = _info["pickle_path"]
            _backup_path = _pkl_path.parent / "embeddings_backup_before_depth_filter.pkl"
            if _backup_path.exists():
                _backup_results.append(f"  {_city}: backup already exists at `{_backup_path}`")
            else:
                shutil.copy2(_pkl_path, _backup_path)
                _backup_results.append(f"  {_city}: backed up to `{_backup_path}`")
        _output = mo.md("### Backup Results\n" + "\n".join(_backup_results))
    _output
    return


@app.cell
def _(
    apply_button,
    city_data,
    depths_loaded,
    landmark_depths,
    mo,
    pickle,
    threshold_slider,
):
    _output = mo.md("")
    if not depths_loaded or apply_button is None or not apply_button.value or threshold_slider is None:
        pass
    else:
        import torch

        _threshold = threshold_slider.value
        _filter_results = []

        for _city, _info in city_data.items():
            _pkl_path = _info["pickle_path"]

            _backup_path = _pkl_path.parent / "embeddings_backup_before_depth_filter.pkl"
            if not _backup_path.exists():
                _filter_results.append(f"  **{_city}: SKIPPED** - no backup found! Create backup first.")
                continue

            _data = _info["data"]
            _panos = _data.get("panoramas", {})

            _total_before = sum(len(_p.get("landmarks", [])) for _p in _panos.values())

            _to_remove = set()
            for (_c, _pk, _li), _val in landmark_depths.items():
                if _c == _city and _val["depth_m"] > _threshold:
                    _to_remove.add((_pk, _li))

            _new_panos = {}
            _kept_custom_ids = set()

            for _pano_key, _pano_data in _panos.items():
                _old_landmarks = _pano_data.get("landmarks", [])
                _new_landmarks = []
                for _lm in _old_landmarks:
                    _lm_idx = _lm["landmark_idx"]
                    _custom_id = f"{_pano_key}__landmark_{_lm_idx}"
                    if (_pano_key, _lm_idx) not in _to_remove:
                        _lm_copy = dict(_lm)
                        _lm_copy["landmark_idx"] = len(_new_landmarks)
                        _new_landmarks.append(_lm_copy)
                        _kept_custom_ids.add(_custom_id)

                _new_pano = dict(_pano_data)
                _new_pano["landmarks"] = _new_landmarks
                _new_panos[_pano_key] = _new_pano

            _old_id_to_idx = _data.get("description_id_to_idx", {})
            _old_embeddings = _data.get("description_embeddings")

            _new_id_to_idx = {}
            _new_embedding_rows = []
            for _custom_id, _old_idx in _old_id_to_idx.items():
                if _custom_id in _kept_custom_ids:
                    _new_id_to_idx[_custom_id] = len(_new_embedding_rows)
                    _new_embedding_rows.append(_old_embeddings[_old_idx])

            if _new_embedding_rows:
                _new_embeddings = torch.stack(_new_embedding_rows)
            else:
                _new_embeddings = torch.zeros(0, _old_embeddings.shape[1])

            _new_data = dict(_data)
            _new_data["panoramas"] = _new_panos
            _new_data["description_embeddings"] = _new_embeddings
            _new_data["description_id_to_idx"] = _new_id_to_idx

            _total_after = sum(len(_p.get("landmarks", [])) for _p in _new_panos.values())

            with open(_pkl_path, "wb") as _f:
                pickle.dump(_new_data, _f)

            _info["data"] = _new_data

            _filter_results.append(
                f"  **{_city}**: {_total_before} -> {_total_after} landmarks "
                f"(removed {_total_before - _total_after}, "
                f"{100*(_total_before - _total_after)/max(_total_before, 1):.1f}%)"
            )

        _verify_results = []
        for _city, _info in city_data.items():
            _data = _info["data"]
            _n_embeddings = _data["description_embeddings"].shape[0]
            _n_idx = len(_data["description_id_to_idx"])
            _ok = _n_embeddings == _n_idx
            _verify_results.append(
                f"  {_city}: embeddings={_n_embeddings}, id_to_idx={_n_idx} "
                f"{'OK' if _ok else 'MISMATCH!'}"
            )

        _output = mo.md(
            f"### Filter Applied (threshold={_threshold}m)\n"
            + "\n".join(_filter_results)
            + "\n\n### Verification\n"
            + "\n".join(_verify_results)
        )

    _output
    return


if __name__ == "__main__":
    app.run()
