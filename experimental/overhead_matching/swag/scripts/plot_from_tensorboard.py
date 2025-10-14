import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib
    matplotlib.style.use('ggplot')
    import seaborn as sns
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import numpy as np
    return EventAccumulator, Path, mo, np, pd, plt, sns


@app.cell
def _(Path):
    model_paths = [
        #Path('/tmp/blah/all_chicago_sat_abs_pos_pano_abs_pos/'),
        #Path('/tmp/absolute_position_plus_dino/'),
        # Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_landmarkv2_sat_semLandmark_pano_semLandmark'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dino_embedding_mat_pano_dino_sam'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_alphaearth_pano_dinov3'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_alphaearth_pano_dinov3_batch_256'),
        # Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_pano_dinov3'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_semLandmark_pano_dinov3_semLandmark'),
        # Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_landmarkv2_sat_dinov3_semLandmark_pano_dinov3_semLandmark/'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_semLandmark_pano_dinov3_semLandmark_autocast'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_semLandmark_pano_semLandmark'),

        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_5_outputDim_256_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_6_outputDim_256_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_8_outputDim_256_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_5_outputDim_256_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_6_outputDim_256_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_8_outputDim_256_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_5_outputDim_1024_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_6_outputDim_1024_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_1024_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_8_outputDim_1024_hinge_False/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_5_outputDim_1024_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_6_outputDim_1024_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_1024_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_8_outputDim_1024_hinge_True/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_1024_hinge_True_after_lgs_8/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_6_outputDim_1024_hinge_True_after_lgs_7_after_lgs_8/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_5_outputDim_1024_hinge_True_after_lgs_6_after_lgs_7_after_lgs_8/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_6_outputDim_256_hinge_False_after_lgs_7_od_256_h_F/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_5_outputDim_256_hinge_False_after_lgs_6_after_lgs7/'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_False_synEmbeddingDim_00'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_False_synEmbeddingDim_02'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_False_synEmbeddingDim_04'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_False_synEmbeddingDim_08'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_False_synEmbeddingDim_16'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_False_synEmbeddingDim_32'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_False_synEmbeddingDim_64'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_True_synEmbeddingDim_00'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_True_synEmbeddingDim_02'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_True_synEmbeddingDim_04'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_True_synEmbeddingDim_08'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_True_synEmbeddingDim_16'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_True_synEmbeddingDim_32'),
        Path('/data/overhead_matching/models/20250918_synthetic_experiments/all_chicago_logGridSpacing_7_outputDim_256_hinge_True_isPanoPosPlanar_True_synEmbeddingDim_64'),
    ]
    return (model_paths,)


@app.cell
def _(model_paths):
    model_paths
    return


@app.cell
def _(EventAccumulator):
    def read_tensorboard_metrics(log_dir):
        """Read metrics from tensorboard log directory."""
        ea = EventAccumulator(str(log_dir))
        ea.Reload()

        metrics_data = []

        model_name = log_dir.name

        # Get all scalar tags that start with validation/
        scalar_tags = [
            tag for tag in ea.Tags()['scalars']
            # if tag.startswith('validation/')
        ]

        fields = log_dir.name.split('_')[2:]

        notes = None
        if 'after' in fields:
            after_idx  = fields.index('after')
            notes = '_'.join(fields[after_idx:])
            fields = fields[:after_idx]
        
    
        import itertools
        model_fields = {k: v for k, v in itertools.batched(fields, 2)} | {"model": model_name, "notes": notes}

        print(model_fields)
        for tag in scalar_tags:
            # Parse tag: validation/Chicago/pos_recall@1 ->
            # dataset=Chicago, metric=pos_recall@1
            parts = tag.split('/')
            if parts[0] == "validation":
                dataset = parts[1]
                metric = '/'.join(parts[2:])  # Handle metrics with slashes
    
                scalar_events = ea.Scalars(tag)
                for event in scalar_events:
                    metrics_data.append({
                        'dataset': dataset,
                        'metric': f'{metric}',  # Add leading slash
                        'step': event.step,
                        'value': event.value,
                    } | model_fields)
            elif parts[0] in ["pano_emb", "sat_emb", 'train']:
                scalar_events = ea.Scalars(tag)
                for event in scalar_events:
                    metrics_data.append({
                        'dataset': "Chicago",
                        'metric': tag,
                        'step': event.step,
                        'value': event.value,
                    } | model_fields)

        return metrics_data
    return (read_tensorboard_metrics,)


@app.cell
def _(mo, model_paths, pd, read_tensorboard_metrics):
    @mo.persistent_cache
    def load_all_metrics(model_paths_list):
        """Load metrics from all model tensorboard logs."""
        all_records = []

        for model_path in model_paths_list:
            print(f'Loading metrics for: {model_path.name}')
            log_dir = model_path

            if not log_dir.exists():
                print(f'  Warning: {log_dir} does not exist, skipping')
                continue

            try:
                metrics_data = read_tensorboard_metrics(log_dir)
                for record in metrics_data:
                    all_records.append(record)
                print(f'  Loaded {len(metrics_data)} metric entries')
            except Exception as exc:
                print(f'  Error loading {log_dir}: {exc}')
                continue

        return pd.DataFrame(all_records)

    validation_df = load_all_metrics(model_paths)
    # validation_df = 
    validation_df = validation_df.sort_values(by=['model', 'metric', 'step', 'dataset']).reset_index(drop=True)

    # validation_df['model'] = validation_df.apply(lambda x: f"logGridSpacing_{x.log_grid_spacing}")

    return (validation_df,)


@app.cell
def _(np):
    def logical_and(*preds):
        if len(preds) == 1:
            return preds[0]

        import functools
        return functools.reduce(np.logical_and, preds)

    def logical_or(*preds):
        if len(preds) == 1:
            return preds[0]

        import functools
        return functools.reduce(np.logical_or, preds)
    return logical_and, logical_or


@app.cell
def _(logical_and, mo, plt, sns, validation_df):
    # Same plotting structure as original notebook
    # _grid_layout = [
    #     ['positive_mean_recip_rank', 'max_pos_semi_pos_recip_rank',
    #      None, None],
    #     ['pos_recall@1', 'pos_recall@5', 'pos_recall@10',
    #      'pos_recall@100'],
    #     ['any pos_semipos_recall@1', 'any pos_semipos_recall@5',
    #      'any pos_semipos_recall@10', 'any pos_semipos_recall@100'],
    #     [None, 'all pos_semipos_recall@5', 'all pos_semipos_recall@10',
    #      'all pos_semipos_recall@100'],
    # ]



    _mask = logical_and(validation_df["outputDim"] == "256",
                        validation_df["hinge"] == "False",
                        validation_df["notes"].isna()
                       )
    _valid_df = validation_df[_mask]

    _grid_layout = [['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]

    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    legend_objs = None

    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 0 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='logGridSpacing', style='dataset', ax=_ax)
                legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='logGridSpacing',
                        style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)
            _ax.set_ylim(-0.05, 1.05)

    if legend_objs:
        _fig.legend(*legend_objs, loc='lower left', bbox_to_anchor=(0.7, 0.25))
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return (legend_objs,)


@app.cell
def _(legend_objs, mo, np, plt, sns, validation_df):
    plt.figure()

    _mask = np.logical_and(
        validation_df["hinge"]=='False',
       validation_df["logGridSpacing"]=="7")
    
    _valid_df = validation_df[_mask]

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean']]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 0 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='outputDim', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='outputDim', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.75, 0.2))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(legend_objs, logical_and, mo, plt, sns, validation_df):
    plt.figure()

    _mask = logical_and(
        validation_df["logGridSpacing"]=="7",
        validation_df['notes'].isna(),
        validation_df['isPanoPosPlanar'].isna(),
    )
    
    _valid_df = validation_df[_mask]

    _valid_df["legend_title"] = _valid_df.apply(lambda x: f"outputDim_{x.outputDim}_hinge_{x.hinge}", axis='columns')

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean'],
                    ['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 1 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='legend_title', style='dataset', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='legend_title', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.75, 0.2))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(legend_objs, logical_and, mo, plt, sns, validation_df):
    plt.figure()

    _mask = logical_and(
        validation_df["outputDim"]=="256",
        validation_df["hinge"]=="False",
        validation_df["notes"].isna(),
        validation_df['isPanoPosPlanar'].isna(),
    )
    
    _valid_df = validation_df[_mask]

    _valid_df["legend_title"] = _valid_df.apply(lambda x: f"outputDim_{x.outputDim}_hinge_{x.hinge}_logGridSpacing_{x.logGridSpacing}", axis='columns')

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean'],
                    ['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 1 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='legend_title', style='dataset', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='legend_title', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.7, 0.2))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(legend_objs, logical_and, mo, plt, sns, validation_df):
    plt.figure()

    _mask = logical_and(
        validation_df["outputDim"]=="1024",
        validation_df["hinge"]=="True",
        validation_df["notes"].isna()
    )
    
    _valid_df = validation_df[_mask]

    _valid_df["legend_title"] = _valid_df.apply(lambda x: f"outputDim_{x.outputDim}_hinge_{x.hinge}_logGridSpacing_{x.logGridSpacing}", axis='columns')

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean'],
                    ['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 1 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='legend_title', style='dataset', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='legend_title', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.7, 0.2))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(legend_objs, logical_and, logical_or, mo, plt, sns, validation_df):
    plt.figure()

    _mask = logical_and(
        validation_df["outputDim"]=="256",
        validation_df["hinge"]=="False",
        logical_or(~validation_df["notes"].isna(),
        validation_df["logGridSpacing"]=="7")
    )
    
    _valid_df = validation_df[_mask]

    _valid_df["legend_title"] = _valid_df.apply(lambda x: f"outputDim_{x.outputDim}_hinge_{x.hinge}_logGridSpacing_{x.logGridSpacing}_notes_{x.notes}", axis='columns')

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean'],
                    ['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 1 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='legend_title', style='dataset', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='legend_title', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.6, 0.05))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(legend_objs, logical_and, logical_or, mo, plt, sns, validation_df):
    plt.figure()

    _mask = logical_and(
        validation_df["outputDim"]=="1024",
        validation_df["hinge"]=="True",
        logical_or(~validation_df["notes"].isna(), validation_df["logGridSpacing"]=="8")
    )
    
    _valid_df = validation_df[_mask]

    _valid_df["legend_title"] = _valid_df.apply(lambda x: f"outputDim_{x.outputDim}_hinge_{x.hinge}_logGridSpacing_{x.logGridSpacing}_notes_{x.notes}", axis='columns')

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean'],
                    ['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 1 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='legend_title', style='dataset', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='legend_title', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.55, 0.1))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(legend_objs, logical_and, mo, plt, sns, validation_df):
    plt.figure()

    _mask = logical_and(
        validation_df["outputDim"]=="256",
        validation_df["hinge"]=="True",
        validation_df["logGridSpacing"]=="7",
        validation_df["isPanoPosPlanar"]=="False",
    )
    
    _valid_df = validation_df[_mask]

    _valid_df["legend_title"] = _valid_df.apply(lambda x: f"synEmbeddingDim_{x.synEmbeddingDim}_isPanoPosPlanar_{x.isPanoPosPlanar}", axis='columns')

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean'],
                    ['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 1 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='legend_title', style='dataset', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='legend_title', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.7, 0.1))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(legend_objs, logical_and, mo, plt, sns, validation_df):
    plt.figure()

    _mask = logical_and(
        validation_df["outputDim"]=="256",
        validation_df["hinge"]=="True",
        validation_df["logGridSpacing"]=="7",
        validation_df["isPanoPosPlanar"]=="True",
    )
    
    _valid_df = validation_df[_mask]

    _valid_df["legend_title"] = _valid_df.apply(lambda x: f"synEmbeddingDim_{x.synEmbeddingDim}_isPanoPosPlanar_{x.isPanoPosPlanar}", axis='columns')

    _grid_layout = [["train/loss", 'pano_emb/cosine_mean', 'sat_emb/cosine_mean'],
                    ['positive_mean_recip_rank', 'any pos_semipos_recall@100', None]]


    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(_num_cols * 5, _num_rows * 5))

    if _axs.ndim == 1:
        _axs = _axs[None, ...]
    _legend_objs = None
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = _valid_df["metric"] == f"{_metric}"
            _to_plot_df = _valid_df[_mask]
            if _i == 1 and _j == 0:
                sns.lineplot(_to_plot_df, x='step', y='value',
                            hue='legend_title', style='dataset', ax=_ax, legend='full')
                _legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='step', y='value', hue='legend_title', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    if legend_objs:
        _fig.legend(*_legend_objs, loc='lower left', bbox_to_anchor=(0.7, 0.1))
    plt.tight_layout()
    _axs[0, 0].set_ylim(0.0, 0.04)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
