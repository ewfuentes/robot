import marimo

__generated_with = "0.11.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    mpl.style.use('ggplot')

    import common.torch.load_torch_deps
    import torch

    from pathlib import Path
    return Path, common, mo, mpl, plt, sns, torch


@app.cell
def _(Path, torch):
    _path_path = Path('/data/overhead_matching/evaluation/results/20250616_8_way_experiment/all_chicago_lr_schedule_False_negative_mining_False_pos_semipos_False/0000000/')
    _particle_history_path = _path_path / 'particle_history.pt'
    _log_particle_weights_path = _path_path / 'log_particle_weights.pt'
    _particle_history_pre_move_path = _path_path / 'particle_history_pre_move.pt'
    _similarity_path = _path_path / 'similarity.pt'

    particle_history = torch.load(_particle_history_path)
    log_particle_weights = torch.load(_log_particle_weights_path)
    particle_history_pre_move = torch.load(_particle_history_pre_move_path)
    similarity = torch.load(_similarity_path)
    return (
        log_particle_weights,
        particle_history,
        particle_history_pre_move,
        similarity,
    )


@app.cell
def _(particle_history):
    particle_history.shape
    return


@app.cell
def _(log_particle_weights):
    log_particle_weights.shape
    return


@app.cell
def _(similarity):
    similarity.shape
    return


@app.cell
def _(log_particle_weights, mo, plt, sns):
    plt.figure()
    sns.histplot(log_particle_weights[0,:])
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(log_particle_weights, mo, particle_history, plt, sns):
    plt.figure(figsize=(12, 8))
    plt.subplot(1,2, 1)
    norm = plt.Normalize(-25, log_particle_weights[0, :].max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    ax = sns.scatterplot(
        x=particle_history[0, :, 1],
        y=particle_history[0, :, 0],
        hue=log_particle_weights[0, :],
        palette='viridis',
        hue_norm=norm)
    # ax.get_legend().remove()
    # ax.figure.colorbar(sm, ax=ax)
    # plt.clim(-25, -5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.subplot(1,2,2)
    sns.scatterplot(
        x=particle_history[1, :, 1],
        y=particle_history[1, :, 0])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    mo.mpl.interactive(plt.gcf())
    return ax, norm, sm


@app.cell
def _():
    import numpy as  np
    50 / 6_371_000 * 180 / np.pi
    return (np,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
