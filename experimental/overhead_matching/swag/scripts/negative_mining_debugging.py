import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from common.torch import  (
        load_torch_deps,
        load_and_save_models as lsm
    )

    import torch

    from experimental.overhead_matching.swag.evaluation import (
        evaluate_swag as es
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    from pathlib import Path

    torch.use_deterministic_algorithms(True)
    return Path, es, lsm, mo, torch, vd


@app.cell
def _():
    import pandas as pd
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.style
    matplotlib.style.use('ggplot')

    return pd, plt, sns


@app.cell
def _(Path, es, lsm, pd, torch, vd):
    model_paths = [
        Path("/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_attn_8"),
        # Path("/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20"),
        # Path("/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_batch_size_128"),
        Path("/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_batch_size_128_warmup_1e-5_lr_2p66e-5"),
        # Path("/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_hard_negative_20_batch_size_128"),
        Path("/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_batch_size_256_hard_negative_20_lr_1p4e-4_warmup_5")

    ]
    checkpoints = [0, 10, 20, 30, 40, 50, 59, 60, 70, 80, 90, 99]

    dataset_path = Path("/data/overhead_matching/datasets/VIGOR/Chicago")

    idx_by_name = {
        'p95': lambda x: int(x.shape[-1] * 0.95),
        'p99': lambda x: int(x.shape[-1] * 0.99),
        'p99.9': lambda x: int(x.shape[-1] * 0.999),
        't100': lambda x: -100,
        'max': lambda x: -1,
    }

    _records = []
    for _p in model_paths:
        print("loading statistics for:", _p)
        for _checkpoint in checkpoints:
            try:
                print(f"loading statistics for {_p} checkpoint {_checkpoint}")
                _sat_model = lsm.load_model(_p / f"{_checkpoint:04d}_satellite")
                _pano_model = lsm.load_model(_p / f"{_checkpoint:04d}_panorama")

                _dataset = vd.VigorDataset(
                    dataset_path,
                    config=vd.VigorDatasetConfig(
                        satellite_tensor_cache_info=None,
                        panorama_tensor_cache_info=None,
                        satellite_patch_size=_sat_model.patch_dims,
                        panorama_size=_pano_model.patch_dims))
                _sim_matrix = es.compute_cached_similarity_matrix(_sat_model, _pano_model, _dataset, device="cuda:0", use_cached_similarity=True)
                _sim_matrix = torch.sort(_sim_matrix, dim=-1).values.cpu()

                for _stat_name, _idx in idx_by_name.items():
                    for _pano_idx, _value in enumerate(_sim_matrix[:, _idx(_sim_matrix)]):
                        _records.append({
                            'epoch': _checkpoint,
                            'statistic': _stat_name,
                            'pano_idx': _pano_idx,
                            'value': _value.item(),
                            'model': _p.name
                        })
            except Exception as e:
                print(f"couldn't load statistics for {_p} checkpoint: {_checkpoint}")
                print(e)
    sim_stats_df = pd.DataFrame.from_records(_records)
    return model_paths, sim_stats_df


@app.cell
def _(model_paths):
    model_paths
    return


@app.cell
def _(mo, model_paths, plt, sim_stats_df, sns):
    _num_models = len(model_paths)
    plt.figure(figsize=(6*_num_models, 6))

    _ax = None
    for _plot_idx, (_model_name, _df) in enumerate(sim_stats_df.groupby('model')):
        if _ax is None:
            _ax = plt.subplot(1, _num_models, _plot_idx+1)
        else:
            plt.subplot(1, _num_models, _plot_idx+1, sharey=_ax)
        show_legend = _plot_idx == 0
        sns.boxenplot(_df, x='epoch', y='value', hue='statistic', hue_order=['p95', 'p99', 't100', 'p99.9', 'max'], legend=show_legend)
        if _plot_idx == 0:
            plt.ylabel('Cosine Similarity')
        else:
            plt.ylabel('')

        _parts = _model_name.split('_')[2:]
        _title = [_parts[0]]
        for _p in _parts[1:]:
            if len(_p) + len(_title[-1]) < 30:
                _title[-1] += "_" + _p
            else:
                _title.append(_p)
        plt.title('\n'.join(_title))
    plt.suptitle('Similarity Statistics by Epoch')
    plt.tight_layout()

    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
