import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import experimental.overhead_matching.swag.evaluation.osm_tag_similarity as ost
    import experimental.overhead_matching.swag.data.vigor_dataset as vd

    from pathlib import Path
    from collections import defaultdict
    import polars as pl
    import marimo as mo
    import numpy as np

    return Path, defaultdict, mo, np, ost, pl, vd


@app.cell
def _(Path, mo, ost, vd):
    @mo.persistent_cache
    def load_city(city):
        _config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            should_load_images=False,
            should_load_landmarks=True,
            landmark_version='v4_202001'
        )
        _vd_dataset = vd.VigorDataset(
            Path('/data/overhead_matching/datasets/VIGOR/Seattle/'),
            _config
        )

        return ost.create_osm_tag_extraction_dataset(
            Path(f'/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/{city}/sentences/'),
            _vd_dataset
        )

    dataset = load_city('Seattle')
    return (dataset,)


@app.cell
def _(Path, vd):
    import shapely
    import networkx as nx
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=False,
        landmark_version='v4_202001'
    )
    _vd_dataset = vd.VigorDataset(
        Path('/data/overhead_matching/datasets/VIGOR/Seattle/'),
        _config
    )
    _PATCH_HALF_WIDTH = 320
    _sat_patch_boxes = []
    for _sat_idx, _row in _vd_dataset._satellite_metadata.iterrows():
        _sat_patch_boxes.append(shapely.box(
            _row.web_mercator_x - _PATCH_HALF_WIDTH,
            _row.web_mercator_y - _PATCH_HALF_WIDTH,
            _row.web_mercator_x + _PATCH_HALF_WIDTH,
            _row.web_mercator_y + _PATCH_HALF_WIDTH))

    _tree = shapely.STRtree(_sat_patch_boxes)

    _result = _tree.query(_sat_patch_boxes, predicate='intersects')
    PANO_SAT_PATCH_CUTOFF = 20
    sat_patch_graph = nx.Graph()
    sat_patch_graph.add_edges_from(_result.T)
    sat_patch_shortest_paths = dict(nx.all_pairs_shortest_path_length(sat_patch_graph, cutoff=PANO_SAT_PATCH_CUTOFF))
    return PANO_SAT_PATCH_CUTOFF, sat_patch_shortest_paths


@app.cell
def _(Path, dataset, mo, ost, pl, vd):
    @mo.persistent_cache
    def load_matches(dataset):
        _config = vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            should_load_images=False,
            should_load_landmarks=False,
            landmark_version='v4_202001')
        _vd_dataset = vd.VigorDataset(
            Path('/data/overhead_matching/datasets/VIGOR/Seattle/'),
            _config
        )  
        _sims, (pano_osm_matches, sat_osm_table, match_schema) = ost.compute_osm_tag_match_similarity(dataset)
        _sat_locs = pl.DataFrame([_vd_dataset._satellite_metadata.index, _vd_dataset._satellite_metadata["lat"], _vd_dataset._satellite_metadata["lon"]]).rename({"": "sat_idx"})
        sat_osm_table = sat_osm_table.join(_sat_locs, on="sat_idx")
        return pano_osm_matches, sat_osm_table, match_schema

    pano_osm_matches, sat_osm_table, match_schema = load_matches(dataset)
    return match_schema, pano_osm_matches, sat_osm_table


@app.cell
def _(dataset, defaultdict, match_schema, pl, sat_patch_shortest_paths):

    _records = []
    for _pano_id, _pano_data in dataset["pano_data_from_pano_id"].items():
        _sat_patch_distances = defaultdict(lambda: 1024)
        for _sat_idx in _pano_data["sat_idxs"]:
            for _neighbor, _dist in sat_patch_shortest_paths[_sat_idx].items():
                _sat_patch_distances[_neighbor] = min(_sat_patch_distances[_neighbor], _dist)

        _records += [(_pano_id, _sat_idx, _dist) for _sat_idx, _dist in _sat_patch_distances.items()]

    pano_distance = pl.DataFrame(_records, schema={"pano_id": match_schema.pano_id, "sat_idx": pl.Int32, "dist": pl.Int32}, orient="row")
    pano_distance

    # _records = []
    # MATCH_TYPE = pl.Enum(["negative", "positive", "semipositive"])
    # for _pano_id, _pano_data in dataset["pano_data_from_pano_id"].items():
    #     _positive_sat_idxs = set(_pano_data["sat_idxs"])
    #     _maybe_semipositive_sat_idxs = set.union(*[sat_patch_graph[x] for x in _positive_sat_idxs])
    #     _semipositive_sat_idxs = _maybe_semipositive_sat_idxs - _positive_sat_idxs
    #     for _sat_idx in _positive_sat_idxs:
    #         _records.append((_pano_id, _sat_idx, 'positive'))
    #     for _sat_idx in _semipositive_sat_idxs:
    #         _records.append((_pano_id, _sat_idx, 'semipositive'))
    # 
    # positives = pl.DataFrame(_records, schema={"pano_id": match_schema.pano_id, "sat_idx": pl.Int32, "match_type": MATCH_TYPE}, orient="row")
    return (pano_distance,)


@app.cell
def _():
    import itertools
    def head(x, n=10):
       return list(itertools.islice(x, n))

    return (itertools,)


@app.cell
def _(dataset, match_schema, np, pano_osm_matches, pl, sat_osm_table):
    _num_sat_patches = max(dataset["sat_data_from_sat_id"].keys()) + 1
    _tag_sat_counts = []
    for _tag, _group in pano_osm_matches.group_by("tag_key", "pano_lm_value"):
        _counts = _group.select("osm_idx").unique().join(sat_osm_table, on="osm_idx")["sat_idx"].n_unique()
        _tag_sat_counts.append((*_tag, _counts, np.log(_num_sat_patches / _counts)))
    tag_sat_counts = pl.DataFrame(_tag_sat_counts, schema={"tag_key":match_schema.tag_key, "pano_lm_value":pl.Utf8, "count":pl.Int64, "idf":pl.Float64}, orient='row')
    return (tag_sat_counts,)


@app.cell
def _(tag_sat_counts):
    tag_sat_counts
    return


@app.cell
def _(pl):
    def compute_pano_sat_similarity(pano_osm_matches, tag_sat_counts, sat_osm_table, lm_max_threshold, tag_agg_method):
        return (pano_osm_matches.lazy()
            .join(tag_sat_counts.lazy(), on=("tag_key", "pano_lm_value"))
            # At this point we have a pano_id x pano_lm x osm_id x tag_idx tensor
            .group_by("pano_id", "pano_lm_idx", "osm_idx")
            # Contract over tags with a selected operation
            .agg(tag_agg_method("idf"), pl.count("pano_lm_idx").alias("num_matching_tags"))
            # Exclude pano/osm matches that are not at least as informative as lm_max_threshold
            .filter(pl.col('idf') > lm_max_threshold)
            # At this point we have a pano_id x pano_lm_idx x osm_id tensor
            .join(sat_osm_table.lazy(), on="osm_idx")
            # At this point we have a pano_id x pano_lm_idx x osm_id x sat_idx tensor
            .group_by("pano_id", "pano_lm_idx", "sat_idx")
            # Contract over osm_idx by taking the max
            .agg(pl.max("idf"))
            # At this point we have a pano_id x pano_lm_idx x sat_idx tensor
            .group_by("pano_id", "sat_idx")
            .agg(pl.mean("idf")))


    return (compute_pano_sat_similarity,)


@app.cell(disabled=True)
def _(
    PANO_SAT_PATCH_CUTOFF,
    compute_pano_sat_similarity,
    itertools,
    np,
    pano_distance,
    pano_osm_matches,
    pl,
    sat_osm_table,
    tag_sat_counts,
):
    import math 
    _tag_agg_methods = {'max': pl.max, 'sum': pl.sum}
    _lm_max_thresholds = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    _match_thresholds = np.linspace(0, 30, 250)

    _pos_by_distance = [pano_distance.filter(pl.col("dist") <= x).height for x in range(PANO_SAT_PATCH_CUTOFF)]

    _records = []
    for _lm_max_threshold, (_tag_agg_name, _tag_agg_method) in itertools.product(_lm_max_thresholds, _tag_agg_methods.items()):
        print(f"{_lm_max_threshold=} {_tag_agg_name=}")
        _results = compute_pano_sat_similarity(
            pano_osm_matches, tag_sat_counts, sat_osm_table,
            lm_max_threshold=_lm_max_threshold, tag_agg_method=_tag_agg_method)

        _combined = (_results.join(
            pano_distance.lazy(),
            on=["pano_id", "sat_idx"],
            how="full", coalesce=True)
            .with_columns(pl.col("idf").fill_null(0.0))).collect(engine="streaming")

        for _match_threshold in _match_thresholds:
            _pred_positives = _combined.filter(pl.col("idf") >= _match_threshold)
            _num_pred_positives = max(_pred_positives.height, 1)

            for _sat_distance_cutoff in range(PANO_SAT_PATCH_CUTOFF):
                _num_true_positives = _pred_positives.filter(pl.col("dist") <= _sat_distance_cutoff).height
                # precision = true positives / pred_positives
                # recall = true postives / num_positives
                _precision = _num_true_positives / _num_pred_positives
                _recall = _num_true_positives / _pos_by_distance[_sat_distance_cutoff]

                _records.append((_lm_max_threshold, _match_threshold, _precision, _recall, _sat_distance_cutoff, _tag_agg_name))

    pr_curve = pl.DataFrame(_records, 
                            schema={"lm_max_threshold": pl.Float64, "match_threshold": pl.Float64, "precision": pl.Float64, "recall": pl.Float64, "sat_distance_cutoff": pl.Int32, "tag_agg": pl.Utf8},
                            orient='row')
    return (pr_curve,)


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    return plt, sns


@app.cell
def _(mo, pl, plt, pr_curve, sns):
    plt.figure(figsize=(12, 8))
    _g = sns.relplot(pr_curve.filter(pl.col("lm_max_threshold") == 4.0), x='recall', y='precision',
                hue="sat_distance_cutoff",
                col="tag_agg",
                palette='tab10', kind='line', hue_order=[0, 1, 2, 4, 8, 16])
    _g.fig.suptitle("lm_max_threshold=4.0")
    # plt.yscale('log')
    # plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(mo, pl, plt, pr_curve, sns):
    plt.figure(figsize=(12, 8))
    _g = sns.relplot(pr_curve.filter(pl.col("sat_distance_cutoff") == 0), x='recall', y='precision',
                hue="lm_max_threshold",
                col="tag_agg",
                palette='tab10', kind='line')
    _g.fig.suptitle("sat_distance_cutoff=2")
    # plt.yscale('log')
    # plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(
    compute_pano_sat_similarity,
    pano_distance,
    pano_osm_matches,
    pl,
    sat_osm_table,
    tag_sat_counts,
):
    sat_patch_similarity = compute_pano_sat_similarity(pano_osm_matches, tag_sat_counts, sat_osm_table, lm_max_threshold=0.5, tag_agg_method=pl.max).collect(engine='streaming')

    combined = (sat_patch_similarity.lazy().join(
        pano_distance.lazy(),
        on=["pano_id", "sat_idx"],
        how="full", coalesce=True)
        .with_columns(pl.col("idf").fill_null(0.0))).collect(engine="streaming")
    return (combined,)


@app.cell
def _(sat_osm_table):
    sat_osm_table
    return


@app.cell
def _(combined):
    combined["idf"].max()
    return


@app.cell
def _(combined, np, pl, sat_osm_table):
    _num_sat_patches = sat_osm_table["sat_idx"].max() + 1

    num_zero_counts = (combined
        .filter(pl.col("idf") > 0)
        .group_by("pano_id")
        .len()
        .rename({"len":"num_nonzero_counts"})
        .with_columns((_num_sat_patches - pl.col("num_nonzero_counts")).alias("bin_count"))
        .with_columns(pl.lit('(-inf, 0.0]').alias("bin").cast(pl.Categorical),
                     pl.lit(0.0).alias("breakpoint"))
        .select("pano_id", "breakpoint", "bin_count")
    )

    pano_bin = (combined
        .filter(pl.col('dist') == 0)
         .sort("idf", descending=True)
         .group_by("pano_id")
         .first()
         .with_columns(pl.col("idf").cut(np.linspace(0, 10, 51), include_breaks=True).alias("positive_bin"))
         .unnest('positive_bin')
         .rename({"category": "positive_bin", "breakpoint": "positive_break"})
         .drop("sat_idx", "dist", "idf")
    )

    pano_bin_counts = (combined
        .filter(pl.col("idf") > 0.0)
        .with_columns(pl.col("idf").cut(np.linspace(0, 10, 51), include_breaks=True).alias("bin"))
        .unnest("bin")
        .rename({"category": "bin"})
        .group_by("pano_id", "breakpoint")
        .len().rename({"len": "bin_count"})
    )

    histograms = (pl.concat([num_zero_counts, pano_bin_counts])
        .join(pano_bin, on="pano_id"))
    return (histograms,)


@app.cell
def _(histograms, mo, pl, plt, sns):
    sns.barplot(
        histograms.filter(pl.col('pano_id') == 'uHS_t2B-CzM7b7HGwGYSzg'),
        x='breakpoint',
        y='bin_count'
    )
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(combined, mo, pano_osm_matches, pl, sat_osm_table, tag_sat_counts):
    _all_data = (combined.lazy().sort("idf", descending=True)
        # .limit(10000)
         .join(sat_osm_table.lazy(), on="sat_idx")
         .rename({"idf": "sat_patch_idf"})
         .join(pano_osm_matches.lazy(), on=("pano_id", "osm_idx"))
         .join(tag_sat_counts.lazy(), on=("tag_key", "pano_lm_value"))
    )
    _landmark_idf = (_all_data
        .group_by("pano_id", "pano_lm_idx", "osm_idx", "sat_idx", "sat_lm_idx")
        .agg(pl.max("idf").alias("lm_idf"))
        .filter(pl.col("lm_idf") > 0.5)
    )

    _all_data = _all_data.join(_landmark_idf, on=("pano_id", "pano_lm_idx", "osm_idx", "sat_idx", "sat_lm_idx"))

    _all_data = _all_data.select(["dist", "sat_idx", "pano_id", "pano_lm_idx", "osm_idx", "tag_key", "pano_lm_value", "sat_lm_value", "idf", "lm_idf", "sat_patch_idf", "lat", "lon"])
    all_data = _all_data.sort("sat_patch_idf", "sat_idx", "pano_id", "osm_idx", "pano_lm_idx", descending=[True, False, True, False, False])
    all_data = all_data.collect(engine='streaming')

    _links = [
        mo.Html(
            f'<a href="http://localhost:8080/view/{pano_id}?sat_idx={sat_idx}" target="_blank">link</a>'
        )
        for pano_id, sat_idx in zip(
            all_data["pano_id"].to_list(), all_data["sat_idx"].to_list()
        )
    ]

    all_data = all_data.with_columns(pl.Series("link", _links, dtype=pl.Object))
    return (all_data,)


@app.cell
def _(all_data, pl):
    all_data.filter(pl.col("pano_id") == 'sLgDPkV59mk1f9SXrxuWKA')
    return


@app.cell
def _(mo, pano_distance, pano_osm_matches, pl, sat_osm_table, tag_sat_counts):
    _false_neg_data = (sat_osm_table.lazy()
        .join(pano_distance.lazy(), on='sat_idx')
        .filter(pl.col("dist") == 0)
        .join(pano_osm_matches.lazy(), on=["pano_id", "osm_idx"])
        .join(tag_sat_counts.lazy(), on=["tag_key", "pano_lm_value"])
    )

    _landmark_idf = (_false_neg_data
        .group_by("pano_id", "pano_lm_idx", "osm_idx", "sat_idx", "sat_lm_idx")
        .agg(pl.max("idf").alias("lm_idf"))
    )
    _sat_patch_idf = (_landmark_idf
        .group_by("pano_id", "sat_idx")
        .agg(pl.mean("lm_idf").alias("sat_patch_idf"))
        .sort("sat_patch_idf", descending=True)
        .group_by("pano_id")
        .first()
    )

    _false_neg_data = (_false_neg_data
        .join(_landmark_idf, on=["pano_id", "pano_lm_idx", "osm_idx", "sat_idx", "sat_lm_idx"])
        .join(_sat_patch_idf, on=["pano_id", "sat_idx"])
    )

    _false_neg_data = _false_neg_data.select(["dist", "sat_idx", "pano_id", "pano_lm_idx", "osm_idx", "tag_key", "pano_lm_value", "sat_lm_value", "idf", "lm_idf", "sat_patch_idf", "lat", "lon"])
    _false_neg_data = _false_neg_data.sort("sat_patch_idf", "sat_idx", "pano_id", "pano_lm_idx", "osm_idx", descending=[False, False, False, False, False])

    _false_neg_data = _false_neg_data.collect(engine="streaming")

    _links = [
        mo.Html(
            f'<a href="http://localhost:8080/view/{pano_id}?sat_idx={sat_idx}" target="_blank">link</a>'
        )
        for pano_id, sat_idx in zip(
            _false_neg_data["pano_id"].to_list(), _false_neg_data["sat_idx"].to_list()
        )
    ]

    false_neg_data = _false_neg_data.with_columns(pl.Series("link", _links, dtype=pl.Object))
    return (false_neg_data,)


@app.cell
def _(false_neg_data, pl):
    false_neg_data.filter(pl.col("sat_patch_idf") > 2.6)
    return


@app.cell
def _(false_neg_data, mo, plt, sns):
    plt.figure()
    sns.displot(false_neg_data.drop("link"), x='sat_patch_idf', kind='ecdf')
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(false_neg_data, pl):
    false_neg_data.filter(pl.col("pano_id") == 'sEmtzXaYKvhjZgHNW8WZag')
    return


@app.cell
def _(all_data, pl):
    _pano_sat_sims = all_data.select("pano_id", "sat_idx", "sat_patch_idf", "dist").unique()

    _best_pos_sims = (_pano_sat_sims
        .filter(pl.col("dist") == 0)
        .sort("sat_patch_idf", descending=True)
        .group_by("pano_id")
        .first()
        .rename({"sat_patch_idf": "best_pos_sat_patch_idf"})
    )

    _bins = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    _best_pos_sims = (_best_pos_sims
        .with_columns(_best_pos_sims["best_pos_sat_patch_idf"]
            .cut(_bins)
            .alias("bin"))
        .drop("dist", "sat_idx"))
    binned = (_pano_sat_sims
        .join(_best_pos_sims, on="pano_id"))
    return (binned,)


@app.cell
def _(binned, mo, plt, sns):
    sns.boxenplot(binned, x="bin", y="sat_patch_idf")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


@app.cell
def _(pano_distance, pano_osm_matches, pl, sat_osm_table, tag_sat_counts):
    sat_patch_sims = (pano_distance.lazy()
        # .filter(pl.col("dist") == 0)
        .select("pano_id", "sat_idx", "dist")
        .join(sat_osm_table.lazy(), on="sat_idx")
        .join(pano_osm_matches.lazy(), on=["pano_id", "osm_idx"])
        .join(tag_sat_counts.lazy(), on=["tag_key", "pano_lm_value"])
        .group_by("sat_idx", "pano_id", "osm_idx", "sat_lm_idx", "pano_lm_idx", "dist")
        .agg(pl.max("idf").alias("lm_idf"))
        .sort("lm_idf", descending=True)
        .group_by("sat_idx", "pano_id", "sat_lm_idx", "dist")
        .first()
        .group_by("sat_idx", "pano_id", "dist")
         .agg(pl.mean("lm_idf").alias("sat_patch_idf"))
    ).collect(engine="streaming")
    return (sat_patch_sims,)


@app.cell
def _(mo, pl, plt, sat_patch_sims, sns):
    best_pos_pano_patch_pairs = (sat_patch_sims
        .filter(pl.col("dist") == 0)
        .sort("sat_patch_idf", descending=True)
        .group_by("pano_id")
        .first()
        .rename({"sat_patch_idf": "best_pos_patch_idf"})
    )

    max_pano_patch_pairs = (sat_patch_sims
        .sort("sat_patch_idf", descending=True)
        .group_by("pano_id")
        .first()
        .rename({"sat_patch_idf": "max_patch_idf"})
    )

    best_max_pos = max_pano_patch_pairs.join(best_pos_pano_patch_pairs, on=["pano_id"], suffix="_best")
    sns.scatterplot(best_max_pos, x="best_pos_patch_idf", y="max_patch_idf", alpha=0.05)
    mo.mpl.interactive(plt.gcf())

    return (best_max_pos,)


@app.cell
def _(best_max_pos, pl):
    low_pos_score_high_max = best_max_pos.filter((pl.col('best_pos_patch_idf') < 1) & (pl.col("max_patch_idf") > 6))
    return (low_pos_score_high_max,)


@app.cell
def _(low_pos_score_high_max):
    low_pos_score_high_max
    return


@app.cell
def _(
    low_pos_score_high_max,
    mo,
    pano_osm_matches,
    pl,
    sat_osm_table,
    tag_sat_counts,
):
    _df = (low_pos_score_high_max
        .join(sat_osm_table, on=["sat_idx"])
        .join(pano_osm_matches, on=["pano_id", "osm_idx"])
        .join(tag_sat_counts, on=["tag_key", "pano_lm_value"])
    )

    _links = [
        mo.Html(f'<a href="http://localhost:8080/view/{_pano_id}?sat_idx={_sat_idx}" target="_blank">link</a>')
        for _pano_id, _sat_idx in _df.select("pano_id", "sat_idx").iter_rows()
    ]

    _df = _df.with_columns(pl.Series("link", _links))
    _df
    return


@app.cell
def _(pano_osm_matches, pl, sat_osm_table, tag_sat_counts):
    (pano_osm_matches
        .filter(pl.col("pano_id") == "AJyfLB-vqSI6md3CRD-fNQ")
         .join(sat_osm_table
             .filter((pl.col("sat_idx") == 8164)),
              on="osm_idx")
         .join(tag_sat_counts, on=["tag_key", "pano_lm_value"])
    )
    return


if __name__ == "__main__":
    app.run()
