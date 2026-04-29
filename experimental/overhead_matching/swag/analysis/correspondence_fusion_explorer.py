import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import common.torch.load_torch_deps
    import torch
    import numpy as np
    import pandas as pd
    import altair as alt
    from pathlib import Path
    from scipy.optimize import linear_sum_assignment

    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    from experimental.overhead_matching.swag.evaluation.correspondence_matching import (
        RawCorrespondenceData,
        MatchingMethod,
        AggregationMode,
        match_and_aggregate,
        compute_uniqueness_weights,
        similarity_from_raw_data,
    )
    from experimental.overhead_matching.swag.evaluation import retrieval_metrics as rm

    return (
        AggregationMode,
        MatchingMethod,
        Path,
        RawCorrespondenceData,
        alt,
        compute_uniqueness_weights,
        match_and_aggregate,
        mo,
        np,
        pd,
        rm,
        torch,
        vd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Correspondence Fusion Explorer

    Explore how different ways of aggregating landmark match probabilities
    into panorama-satellite similarity scores affect retrieval performance.

    The raw data contains P(match) for every (pano_landmark, osm_landmark) pair.
    This notebook lets you try different:
    - **Matching methods**: Hungarian (optimal 1:1) vs Greedy
    - **Aggregation**: Sum, Max, Log-odds of matched probabilities
    - **Thresholds**: Minimum P(match) to include a pair
    - **Uniqueness weighting**: Down-weight generic landmarks (bridge, tree)
    - **Custom fusion**: Train a learned aggregation
    """)
    return


@app.cell
def _(mo):
    city_dropdown = mo.ui.dropdown(
        options={
            "MiamiBeach": "/data/overhead_matching/datasets/VIGOR/mapillary/MiamiBeach",
            "Framingham": "/data/overhead_matching/datasets/VIGOR/mapillary/Framingham",
            "Gap": "/data/overhead_matching/datasets/VIGOR/mapillary/Gap",
            "Middletown": "/data/overhead_matching/datasets/VIGOR/mapillary/Middletown",
            "Norway": "/data/overhead_matching/datasets/VIGOR/mapillary/Norway",
            "post_hurricane_ian": "/data/overhead_matching/datasets/VIGOR/mapillary/post_hurricane_ian",
            "SanFrancisco_mapillary": "/data/overhead_matching/datasets/VIGOR/mapillary/SanFrancisco_mapillary",
            "Boston": "/data/overhead_matching/datasets/VIGOR/Boston",
            "nightdrive": "/data/overhead_matching/datasets/VIGOR/nightdrive",
        },
        value="MiamiBeach",
        label="City",
    )
    city_dropdown
    return (city_dropdown,)


@app.cell
def _(Path, RawCorrespondenceData, city_dropdown, torch, vd):
    # Load dataset and raw correspondence data
    _city_path = Path(city_dropdown.value)
    _raw_path = _city_path / "correspondence_scores" / "v5_all_cities_raw.pt"

    _data = torch.load(_raw_path, weights_only=False)
    raw = RawCorrespondenceData(
        cost_matrix=_data["cost_matrix"],
        pano_id_to_lm_rows=_data["pano_id_to_lm_rows"],
        pano_lm_tags=_data["pano_lm_tags"],
        osm_lm_indices=_data["osm_lm_indices"],
        osm_lm_tags=_data["osm_lm_tags"],
    )

    _feather_files = list((_city_path / "landmarks").glob("*.feather"))
    _lv = _feather_files[0].stem if _feather_files else None
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None, panorama_tensor_cache_info=None,
        should_load_images=False, should_load_landmarks=True, landmark_version=_lv,
    )
    dataset = vd.VigorDataset(_city_path, _config)

    # Pre-build sat -> col positions
    _osm_idx_to_col = {idx: col for col, idx in enumerate(raw.osm_lm_indices)}
    sat_col_positions = []
    for _si in range(len(dataset._satellite_metadata)):
        _lm_idxs = dataset._satellite_metadata.iloc[_si].get("landmark_idxs", [])
        if _lm_idxs is None:
            sat_col_positions.append([])
        else:
            sat_col_positions.append([_osm_idx_to_col[i] for i in _lm_idxs if i in _osm_idx_to_col])

    f"Loaded {_city_path.name}: {raw.cost_matrix.shape[0]} pano landmarks x {raw.cost_matrix.shape[1]} OSM landmarks, {len(dataset._panorama_metadata)} panos, {len(dataset._satellite_metadata)} sats"
    return dataset, raw, sat_col_positions


@app.cell
def _(mo):
    mo.md("""
    ## Aggregation Settings
    """)
    return


@app.cell
def _(mo):
    method_select = mo.ui.dropdown(
        options={"Hungarian": "hungarian", "Greedy": "greedy"},
        value="Hungarian", label="Matching",
    )
    agg_select = mo.ui.dropdown(
        options={"Sum": "sum", "Max": "max", "Log Odds": "log_odds"},
        value="Sum", label="Aggregation",
    )
    threshold_slider = mo.ui.slider(
        start=0.0, stop=0.95, step=0.05, value=0.3, label="P(match) threshold",
    )
    uniqueness_toggle = mo.ui.checkbox(value=True, label="Uniqueness weighting")

    mo.hstack([method_select, agg_select, threshold_slider, uniqueness_toggle])
    return agg_select, method_select, threshold_slider, uniqueness_toggle


@app.cell
def _(
    AggregationMode,
    MatchingMethod,
    agg_select,
    compute_uniqueness_weights,
    dataset,
    match_and_aggregate,
    method_select,
    raw,
    rm,
    sat_col_positions,
    threshold_slider,
    torch,
    uniqueness_toggle,
):
    # Compute similarity matrix with current settings
    _method = MatchingMethod(method_select.value)
    _agg = AggregationMode(agg_select.value)
    _thresh = threshold_slider.value
    _use_uniq = uniqueness_toggle.value

    _num_panos = len(dataset._panorama_metadata)
    _num_sats = len(dataset._satellite_metadata)
    _similarity = torch.zeros(_num_panos, _num_sats)

    per_pano_stats = []  # for per-pano analysis

    for _pi in range(_num_panos):
        _pano_id = dataset._panorama_metadata.iloc[_pi]["pano_id"]
        _rows = raw.pano_id_to_lm_rows.get(_pano_id)
        if _rows is None:
            per_pano_stats.append({
                "pano_idx": _pi, "pano_id": _pano_id,
                "n_landmarks": 0, "n_matched_sats": 0,
                "best_rank": _num_sats, "mrr": 0.0,
            })
            continue

        _pano_cost = raw.cost_matrix[_rows]
        _u_weights = compute_uniqueness_weights(_pano_cost, _thresh) if _use_uniq else None

        _n_matched = 0
        for _si in range(_num_sats):
            _cols = sat_col_positions[_si]
            if not _cols:
                continue
            _sub = _pano_cost[:, _cols]
            _result = match_and_aggregate(_sub, _method, _agg, _thresh,
                                          uniqueness_weights=_u_weights)
            _similarity[_pi, _si] = _result.similarity_score
            if _result.similarity_score > 0:
                _n_matched += 1

        # Per-pano metrics
        _row = dataset._panorama_metadata.iloc[_pi]
        _pos = set(_row["positive_satellite_idxs"]) | set(
            _row.get("semipositive_satellite_idxs", []))
        _ranking = torch.argsort(_similarity[_pi], descending=True)
        _best_rank = _num_sats
        for _p in _pos:
            _r = (_ranking == _p).nonzero(as_tuple=True)[0].item()
            _best_rank = min(_best_rank, _r)

        per_pano_stats.append({
            "pano_idx": _pi, "pano_id": _pano_id,
            "n_landmarks": len(_rows),
            "n_matched_sats": _n_matched,
            "best_rank": _best_rank,
            "mrr": 1.0 / (_best_rank + 1) if _pos else 0.0,
        })

    similarity = _similarity
    metrics = rm.compute_top_k_metrics(similarity, dataset, ks=[1, 5, 10, 50, 100])
    pano_df = __import__("pandas").DataFrame(per_pano_stats)
    return metrics, pano_df


@app.cell
def _(metrics, mo, pano_df):
    _metrics_md = " | ".join(f"**{k}**: {v:.4f}" for k, v in metrics.items())
    _avg_mrr = pano_df["mrr"].mean()
    _pct_nonzero = (pano_df["n_matched_sats"] > 0).mean() * 100

    mo.md(f"""
    ## Results

    {_metrics_md}

    Average MRR: **{_avg_mrr:.4f}** | Panos with any matches: **{_pct_nonzero:.1f}%** | Median best rank: **{pano_df['best_rank'].median():.0f}**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-Panorama Analysis
    """)
    return


@app.cell
def _(alt, pano_df):
    # MRR distribution
    _mrr_chart = alt.Chart(pano_df[pano_df["mrr"] > 0]).mark_bar().encode(
        alt.X("mrr:Q", bin=alt.Bin(maxbins=30), title="MRR"),
        alt.Y("count():Q", title="Count"),
    ).properties(title="MRR Distribution (non-zero only)", width=400, height=250)

    # Best rank distribution (log scale)
    _rank_chart = alt.Chart(pano_df[pano_df["best_rank"] < pano_df["best_rank"].max()]).mark_bar().encode(
        alt.X("best_rank:Q", bin=alt.Bin(maxbins=40), title="Best Positive Rank"),
        alt.Y("count():Q", title="Count", scale=alt.Scale(type="log")),
    ).properties(title="Best Rank Distribution", width=400, height=250)

    _mrr_chart | _rank_chart
    return


@app.cell
def _(alt, pano_df):
    # Landmarks vs MRR scatter
    alt.Chart(pano_df).mark_circle(size=40, opacity=0.5).encode(
        alt.X("n_landmarks:Q", title="# Pano Landmarks"),
        alt.Y("mrr:Q", title="MRR"),
        alt.Color("n_matched_sats:Q", scale=alt.Scale(scheme="viridis"), title="# Matched Sats"),
        tooltip=["pano_id", "n_landmarks", "mrr", "best_rank", "n_matched_sats"],
    ).properties(title="Landmarks vs MRR", width=600, height=300)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Per-Landmark Uniqueness Analysis

    How many OSM landmarks does each pano landmark match above threshold?
    Generic landmarks (building, tree) match many; specific ones (named restaurant) match few.
    """)
    return


@app.cell
def _(np, pd, raw, threshold_slider):
    # Analyze per-pano-landmark match counts
    _thresh = threshold_slider.value
    _match_counts = (raw.cost_matrix >= _thresh).sum(axis=1)
    _max_probs = raw.cost_matrix.max(axis=1)

    landmark_df = pd.DataFrame({
        "lm_idx": range(len(_match_counts)),
        "tags": ["; ".join(f"{k}={v}" for k, v in tags) for tags in raw.pano_lm_tags],
        "n_matches": _match_counts,
        "max_prob": _max_probs,
        "uniqueness_weight": 1.0 / np.log2(1.0 + np.maximum(_match_counts, 1).astype(float)),
    })
    landmark_df
    return (landmark_df,)


@app.cell
def _(alt, landmark_df):
    alt.Chart(landmark_df).mark_bar().encode(
        alt.X("n_matches:Q", bin=alt.Bin(maxbins=50), title="# OSM Matches Above Threshold"),
        alt.Y("count():Q", title="Count", scale=alt.Scale(type="log")),
    ).properties(title="Landmark Match Count Distribution", width=600, height=250)
    return


@app.cell
def _(landmark_df, mo):
    # Show most generic and most unique landmarks
    _generic = landmark_df.nlargest(10, "n_matches")[["tags", "n_matches", "max_prob", "uniqueness_weight"]]
    _unique = landmark_df[landmark_df["n_matches"] > 0].nsmallest(10, "n_matches")[["tags", "n_matches", "max_prob", "uniqueness_weight"]]

    mo.vstack([
        mo.md("### Most Generic Landmarks (match many OSM landmarks)"),
        _generic,
        mo.md("### Most Unique Landmarks (match few OSM landmarks)"),
        _unique,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Compare Aggregation Methods

    Run multiple configurations and compare metrics side-by-side.
    """)
    return


@app.cell
def _(
    AggregationMode,
    MatchingMethod,
    compute_uniqueness_weights,
    dataset,
    match_and_aggregate,
    pd,
    raw,
    rm,
    sat_col_positions,
    torch,
):
    def compute_metrics_for_config(method, aggregation, threshold, use_uniqueness):
        """Compute retrieval metrics for a given configuration."""
        _m = MatchingMethod(method)
        _a = AggregationMode(aggregation)
        _num_panos = len(dataset._panorama_metadata)
        _num_sats = len(dataset._satellite_metadata)
        _sim = torch.zeros(_num_panos, _num_sats)

        for _pi in range(_num_panos):
            _pano_id = dataset._panorama_metadata.iloc[_pi]["pano_id"]
            _rows = raw.pano_id_to_lm_rows.get(_pano_id)
            if _rows is None:
                continue
            _pano_cost = raw.cost_matrix[_rows]
            _uw = compute_uniqueness_weights(_pano_cost, threshold) if use_uniqueness else None

            for _si in range(_num_sats):
                _cols = sat_col_positions[_si]
                if not _cols:
                    continue
                _sub = _pano_cost[:, _cols]
                _result = match_and_aggregate(_sub, _m, _a, threshold, uniqueness_weights=_uw)
                _sim[_pi, _si] = _result.similarity_score

        _metrics = rm.compute_top_k_metrics(_sim, dataset, ks=[1, 5, 10])
        _nonzero = (_sim > 0).any(dim=1).float().mean().item() * 100
        return {**_metrics, "pct_nonzero": _nonzero}

    # Run comparison grid
    _configs = [
        ("hungarian", "sum", 0.3, False, "H/Sum/0.3"),
        ("hungarian", "sum", 0.5, False, "H/Sum/0.5"),
        ("hungarian", "sum", 0.8, False, "H/Sum/0.8"),
        ("hungarian", "sum", 0.3, True, "H/Sum/0.3/U"),
        ("hungarian", "sum", 0.5, True, "H/Sum/0.5/U"),
        ("hungarian", "sum", 0.8, True, "H/Sum/0.8/U"),
        ("hungarian", "max", 0.3, False, "H/Max/0.3"),
        ("hungarian", "log_odds", 0.3, False, "H/LogOdds/0.3"),
        ("greedy", "sum", 0.3, False, "G/Sum/0.3"),
        ("greedy", "sum", 0.3, True, "G/Sum/0.3/U"),
    ]

    _results = []
    for _method, _agg, _thresh, _uniq, _label in _configs:
        _m = compute_metrics_for_config(_method, _agg, _thresh, _uniq)
        _m["config"] = _label
        _results.append(_m)

    comparison_df = pd.DataFrame(_results).set_index("config")
    comparison_df
    return (comparison_df,)


@app.cell
def _(alt, comparison_df):
    _df = comparison_df.reset_index()
    _chart = alt.Chart(_df).mark_bar().encode(
        alt.X("config:N", title="Configuration", sort=None),
        alt.Y("mrr:Q", title="MRR"),
    ).properties(title="MRR by Configuration", width=600, height=300)
    _chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## Pano-level Drill-down

    Select a panorama to see its landmarks, their uniqueness weights,
    and how they contribute to satellite scores.
    """)
    return


@app.cell
def _(mo, pano_df):
    pano_selector = mo.ui.dropdown(
        options={
            f"{row.pano_id} (MRR={row.mrr:.3f}, rank={row.best_rank})": row.pano_idx
            for _, row in pano_df.iterrows()
        },
        label="Select panorama",
    )
    pano_selector
    return (pano_selector,)


@app.cell
def _(
    AggregationMode,
    MatchingMethod,
    agg_select,
    compute_uniqueness_weights,
    dataset,
    match_and_aggregate,
    method_select,
    mo,
    pano_selector,
    pd,
    raw,
    sat_col_positions,
    threshold_slider,
    uniqueness_toggle,
):
    if pano_selector.value is None:
        mo.stop(True, mo.md("*Select a panorama above*"))

    _pi = pano_selector.value
    _pano_id = dataset._panorama_metadata.iloc[_pi]["pano_id"]
    _rows = raw.pano_id_to_lm_rows.get(_pano_id)

    if _rows is None:
        mo.stop(True, mo.md("*No landmarks for this panorama*"))

    _pano_cost = raw.cost_matrix[_rows]
    _thresh = threshold_slider.value
    _method = MatchingMethod(method_select.value)
    _agg = AggregationMode(agg_select.value)
    _uw = compute_uniqueness_weights(_pano_cost, _thresh) if uniqueness_toggle.value else None

    # Landmark summary
    _lm_data = []
    for _i, _row_idx in enumerate(_rows):
        _tags = "; ".join(f"{k}={v}" for k, v in raw.pano_lm_tags[_row_idx])
        _n_matches = int((_pano_cost[_i] >= _thresh).sum())
        _max_p = float(_pano_cost[_i].max())
        _weight = float(_uw[_i]) if _uw is not None else 1.0
        _lm_data.append({
            "idx": _i, "tags": _tags, "n_matches": _n_matches,
            "max_prob": round(_max_p, 3), "weight": round(_weight, 3),
        })

    # Top satellites
    _pos = set(dataset._panorama_metadata.iloc[_pi]["positive_satellite_idxs"]) | set(
        dataset._panorama_metadata.iloc[_pi].get("semipositive_satellite_idxs", []))

    _sat_scores = []
    for _si in range(len(dataset._satellite_metadata)):
        _cols = sat_col_positions[_si]
        if not _cols:
            continue
        _sub = _pano_cost[:, _cols]
        _result = match_and_aggregate(_sub, _method, _agg, _thresh, uniqueness_weights=_uw)
        if _result.similarity_score > 0:
            _sat_scores.append({
                "sat_idx": _si, "score": round(_result.similarity_score, 4),
                "n_matches": len(_result.match_probs),
                "is_positive": _si in _pos,
            })

    _sat_scores.sort(key=lambda x: x["score"], reverse=True)

    mo.vstack([
        mo.md(f"### Panorama {_pano_id}"),
        mo.md("**Landmarks:**"),
        pd.DataFrame(_lm_data),
        mo.md(f"**Top 20 Satellites** ({len(_sat_scores)} with non-zero score):"),
        pd.DataFrame(_sat_scores[:20]),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
