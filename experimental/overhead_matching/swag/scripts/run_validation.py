import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from experimental.overhead_matching.swag.scripts import (
        train as T
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    import common.torch.load_and_save_models as lsm
    from pathlib import Path
    import pandas as pd
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.pyplot as plt
    import seaborn as sns
    return Path, T, lsm, matplotlib, mo, pd, plt, sns, vd


@app.cell
def _(Path):
    model_paths = [
        Path('/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512'),
        Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_attn_8'),
        Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5'),
    ]

    validation_set_paths = [
        Path("/data/overhead_matching/datasets/VIGOR/Chicago"),
        Path("/data/overhead_matching/datasets/VIGOR/Seattle"),
    ]
    return model_paths, validation_set_paths


@app.cell
def _(T, lsm, model_paths, pd, validation_set_paths, vd):
    _records = []
    for _dp in validation_set_paths:
        _dataset = None
        print(f'Dataset: {_dp.name}')
        for _mp in model_paths:
            _checkpoints = [int(x.name.split("_")[0]) for x in sorted(_mp.glob("[0-9]*_satellite"))]
            print(_mp.name, _checkpoints)
            for _checkpoint in _checkpoints:
                print(f'working on: {_mp.name} {_checkpoint}')
                _sat_model = lsm.load_model(_mp / f"{_checkpoint:04d}_satellite", device='cuda:0', skip_consistent_output_check=True).eval()
                _pano_model = lsm.load_model(_mp / f"{_checkpoint:04d}_panorama", device='cuda:0', skip_consistent_output_check=True).eval()

                if _dataset is None:
                    _dataset_config = vd.VigorDatasetConfig(
                        panorama_tensor_cache_info=None,
                        satellite_tensor_cache_info=None,
                        panorama_size=_pano_model.patch_dims,
                        satellite_patch_size=_sat_model.patch_dims,
                        factor=0.3
                    )
                    _dataset = vd.VigorDataset(_dp, _dataset_config)
                    print(f'loaded {len(_dataset._panorama_metadata)} panos and {len(_dataset._satellite_metadata)} sats from {_dp}')

                _metrics = T.compute_validation_metrics(
                    sat_model=_sat_model,
                    pano_model=_pano_model,
                    dataset=_dataset)
                for k, v in _metrics.items():
                    _records.append({
                        'model': _mp.name,
                        'epoch': _checkpoint,
                        'dataset': _dp.name,
                        'metric': k,
                        'value': v})

    validation_df = pd.DataFrame.from_records(_records)
    return k, v, validation_df


@app.cell
def _(validation_df):
    validation_df
    return


@app.cell
def _(mo, plt, sns, validation_df):
    # grid = sns.FacetGrid(to_plot_df, col='metric', col_wrap=4, hue='dataset_and_model')
    # grid.map(sns.lineplot, 'epoch', 'value')
    # grid.set_titles(col_template='{col_name}')
    # grid.add_legend()

    _grid_layout = [
        ['positive_mean_recip_rank', 'max_pos_semi_pos_recip_rank', None, None],
        ['pos_recall@1', 'pos_recall@5', 'pos_recall@10', 'pos_recall@100'],
        ['any pos_semipos_recall@1', 'any pos_semipos_recall@5', 'any pos_semipos_recall@10', 'any pos_semipos_recall@100'],
        [None, 'all pos_semipos_recall@5', 'all pos_semipos_recall@10', 'all pos_semipos_recall@100'],
    ]

    _num_rows = len(_grid_layout)
    _num_cols = len(_grid_layout[0])
    _fig, _axs = plt.subplots(_num_rows, _num_cols, figsize=(16, 16))
    for _i, _row in enumerate(_grid_layout):
        for _j, _metric in enumerate(_row):
            _ax = _axs[_i, _j]
            if _metric is None:
                _ax.remove()
                continue
            _mask = validation_df["metric"] == _metric
            _to_plot_df = validation_df[_mask]
            if _i == 0 and _j == 0:
                sns.lineplot(_to_plot_df, x='epoch', y='value', hue='model', style='dataset', ax=_ax)    
                legend_objs = _ax.get_legend_handles_labels()
                _ax.clear()
            sns.lineplot(_to_plot_df, x='epoch', y='value', hue='model', style='dataset', legend=False, ax=_ax)
            _ax.set_title(_metric)

    print([l.get_label() for l in _axs[0, 0].lines])
    _fig.legend(*legend_objs, loc='upper right')
    plt.tight_layout()

    mo.mpl.interactive(_fig)
    return (legend_objs,)


@app.cell
def _(validation_df):
    validation_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
