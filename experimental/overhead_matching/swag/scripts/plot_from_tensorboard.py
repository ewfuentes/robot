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
    return EventAccumulator, Path, mo, pd, plt, sns


@app.cell
def _(Path):
    model_paths = [
        #Path('/tmp/blah/all_chicago_sat_abs_pos_pano_abs_pos/'),
        #Path('/tmp/absolute_position_plus_dino/'),
        Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_landmarkv2_sat_semLandmark_pano_semLandmark'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dino_embedding_mat_pano_dino_sam'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_alphaearth_pano_dinov3'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_alphaearth_pano_dinov3_batch_256'),
        Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_pano_dinov3'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_semLandmark_pano_dinov3_semLandmark'),
        Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_landmarkv2_sat_dinov3_semLandmark_pano_dinov3_semLandmark/'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_dinov3_semLandmark_pano_dinov3_semLandmark_autocast'),
        #Path('/data/overhead_matching/models/20250815_swag_semantic/all_chicago_sat_semLandmark_pano_semLandmark'),
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

        # Get all scalar tags that start with validation/
        scalar_tags = [
            tag for tag in ea.Tags()['scalars']
            if tag.startswith('validation/')
        ]

        for tag in scalar_tags:
            # Parse tag: validation/Chicago/pos_recall@1 ->
            # dataset=Chicago, metric=pos_recall@1
            parts = tag.split('/')
            if len(parts) >= 3:
                dataset = parts[1]
                metric = '/'.join(parts[2:])  # Handle metrics with slashes

                scalar_events = ea.Scalars(tag)
                for event in scalar_events:
                    metrics_data.append({
                        'dataset': dataset,
                        'metric': f'/{metric}',  # Add leading slash
                        'epoch': event.step,
                        'value': event.value
                    })

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
            log_dir = model_path / 'logs'

            if not log_dir.exists():
                print(f'  Warning: {log_dir} does not exist, skipping')
                continue

            try:
                metrics_data = read_tensorboard_metrics(log_dir)
                for record in metrics_data:
                    record['model'] = model_path.name
                    all_records.append(record)
                print(f'  Loaded {len(metrics_data)} metric entries')
            except Exception as exc:
                print(f'  Error loading {log_dir}: {exc}')
                continue

        return pd.DataFrame(all_records)

    validation_df = load_all_metrics(model_paths).sort_values(by=['dataset', 'model', 'metric', 'epoch']).reset_index(drop=True)
    return (validation_df,)


@app.cell
def _():
    return


@app.cell
def _(mo, plt, sns, validation_df):
    # Same plotting structure as original notebook
    _grid_layout = [
        ['positive_mean_recip_rank', 'max_pos_semi_pos_recip_rank',
         None, None],
        ['pos_recall@1', 'pos_recall@5', 'pos_recall@10',
         'pos_recall@100'],
        ['any pos_semipos_recall@1', 'any pos_semipos_recall@5',
         'any pos_semipos_recall@10', 'any pos_semipos_recall@100'],
        [None, 'all pos_semipos_recall@5', 'all pos_semipos_recall@10',
         'all pos_semipos_recall@100'],
    ]

    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(16, 16))
    legend_objs = None

    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = validation_df["metric"] == f"/{_metric}"
            _to_plot_df = validation_df[_mask]
            if _i == 0 and _j == 0:
                sns.lineplot(_to_plot_df, x='epoch', y='value',
                            hue='model', style='dataset', ax=_ax)
                legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='epoch', y='value', hue='model',
                        style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    print([line.get_label() for line in _axs[0, 0].lines])
    if legend_objs:
        _fig.legend(*legend_objs, loc='upper right')
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(validation_df):
    validation_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
