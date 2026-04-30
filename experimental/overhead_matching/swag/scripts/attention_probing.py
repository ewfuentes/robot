import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    from pathlib import Path
    import numpy as np
    import pandas as pd

    from common.torch import (
        load_torch_deps,
        load_and_save_models as lsm
    )
    import torch
    import msgspec
    from common.python.serialization import msgspec_dec_hook
    from experimental.overhead_matching.swag.model import (
        patch_embedding as pe,
        swag_patch_embedding as spe
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    from experimental.overhead_matching.swag.evaluation import (
        evaluate_swag as es
    )

    return (
        Path,
        es,
        matplotlib,
        mo,
        msgspec,
        msgspec_dec_hook,
        np,
        pd,
        pe,
        plt,
        spe,
        torch,
        vd,
    )


@app.cell
def _(Path):
    TRAINING_BASE = Path('/data/overhead_matching/training_outputs/260215_baseline_retraining')
    WAG_BASE = TRAINING_BASE / '260218_093851_all_chicago_dinov3_wag_bs18_v2'
    SWAG_BASE = TRAINING_BASE / '260218_142001_all_chicago_dinov3_swag_transformer_v2'
    DATASET_PATH = Path('/data/overhead_matching/datasets/VIGOR/Seattle')
    TOP_K_THRESHOLD = 10
    PATCH_SIZE = 16  # dinov3_vitb16
    return DATASET_PATH, PATCH_SIZE, SWAG_BASE, TOP_K_THRESHOLD, WAG_BASE


@app.cell
def _(SWAG_BASE, WAG_BASE, msgspec, msgspec_dec_hook, pe, spe, torch):
    from typing import Union
    _ModelConfig = Union[pe.WagPatchEmbeddingConfig, spe.SwagPatchEmbeddingConfig]

    def _load_model(base_path, checkpoint_name, config_key):
        """Load model from config + state dict (avoids pickle issues with dinov3)."""
        config_bytes = (base_path / 'train_config.yaml').read_bytes()
        _full = msgspec.yaml.decode(config_bytes)
        _model_cfg = msgspec.convert(_full[config_key], _ModelConfig, dec_hook=msgspec_dec_hook)
        if isinstance(_model_cfg, pe.WagPatchEmbeddingConfig):
            model = pe.WagPatchEmbedding(_model_cfg)
        else:
            model = spe.SwagPatchEmbedding(_model_cfg)
        weights = torch.load(base_path / checkpoint_name / 'model_weights.pt', map_location='cuda', weights_only=True)
        # Strip _orig_mod. prefix from torch.compile'd state dicts
        weights = {k.removeprefix('_orig_mod.'): v for k, v in weights.items()}
        model.load_state_dict(weights)
        return model.eval().cuda()

    wag_sat_model = _load_model(WAG_BASE, 'best_satellite', 'sat_model_config')
    wag_pano_model = _load_model(WAG_BASE, 'best_panorama', 'pano_model_config')
    swag_sat_model = _load_model(SWAG_BASE, 'best_satellite', 'sat_model_config')
    swag_pano_model = _load_model(SWAG_BASE, 'best_panorama', 'pano_model_config')
    return swag_pano_model, swag_sat_model, wag_pano_model, wag_sat_model


@app.cell
def _(DATASET_PATH, vd, wag_pano_model, wag_sat_model):
    dataset = vd.VigorDataset(
        DATASET_PATH,
        config=vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            satellite_patch_size=wag_sat_model.patch_dims,
            panorama_size=wag_pano_model.patch_dims))
    return (dataset,)


@app.cell
def _(
    dataset,
    es,
    swag_pano_model,
    swag_sat_model,
    wag_pano_model,
    wag_sat_model,
):
    wag_similarity = es.compute_cached_similarity_matrix(
        wag_sat_model, wag_pano_model, dataset, device='cuda:0', use_cached_similarity=True)
    swag_similarity = es.compute_cached_similarity_matrix(
        swag_sat_model, swag_pano_model, dataset, device='cuda:0', use_cached_similarity=True)
    return swag_similarity, wag_similarity


@app.cell
def _(dataset, np, pd, swag_similarity, torch, wag_similarity):
    _num_panos = wag_similarity.shape[0]
    _records = []
    for _i in range(_num_panos):
        _pano_meta = dataset._panorama_metadata.iloc[_i]
        _correct_sat = _pano_meta.satellite_idx

        _wag_ranks = torch.argsort(wag_similarity[_i], descending=True)
        _swag_ranks = torch.argsort(swag_similarity[_i], descending=True)
        _wag_rank = (_wag_ranks == _correct_sat).nonzero(as_tuple=True)[0].item()
        _swag_rank = (_swag_ranks == _correct_sat).nonzero(as_tuple=True)[0].item()

        _records.append({
            'pano_idx': _i,
            'pano_id': _pano_meta.name if hasattr(_pano_meta, 'name') else str(_i),
            'correct_sat_idx': _correct_sat,
            'wag_rank': _wag_rank,
            'swag_rank': _swag_rank,
            'wag_correct_sim': wag_similarity[_i, _correct_sat].item(),
            'swag_correct_sim': swag_similarity[_i, _correct_sat].item(),
            'wag_top1_idx': wag_similarity[_i].argmax().item(),
            'swag_top1_idx': swag_similarity[_i].argmax().item(),
        })
    rankings_df = pd.DataFrame.from_records(_records)
    # Cap for log-scale scatter
    _max_rank = max(rankings_df['wag_rank'].max(), rankings_df['swag_rank'].max())
    rankings_df['wag_rank_capped'] = np.minimum(rankings_df['wag_rank'], _max_rank)
    rankings_df['swag_rank_capped'] = np.minimum(rankings_df['swag_rank'], _max_rank)
    return (rankings_df,)


@app.cell
def _(TOP_K_THRESHOLD, rankings_df):
    _k = TOP_K_THRESHOLD

    def _categorize(row):
        wag_good = row['wag_rank'] < _k
        swag_good = row['swag_rank'] < _k
        if wag_good and swag_good:
            return 'both_good'
        elif not wag_good and not swag_good:
            return 'both_bad'
        elif wag_good:
            return 'wag_better'
        else:
            return 'swag_better'

    rankings_df['category'] = rankings_df.apply(_categorize, axis=1)
    category_counts = rankings_df['category'].value_counts()
    return (category_counts,)


@app.cell
def _(category_counts, mo, plt, rankings_df):
    _colors = {
        'both_good': '#2ecc71', 'both_bad': '#e74c3c',
        'wag_better': '#3498db', 'swag_better': '#e67e22'
    }

    _fig, _ax = plt.subplots(figsize=(8, 8))
    for _cat, _group in rankings_df.groupby('category'):
        _ax.scatter(_group['wag_rank'], _group['swag_rank'],
                    alpha=0.4, s=10, label=_cat, color=_colors.get(_cat, 'gray'))
    _ax.set_xlabel('WAG Rank')
    _ax.set_ylabel('SWAG Rank')
    _ax.set_title('WAG vs SWAG Ranking Comparison')
    _ax.set_xscale('symlog', linthresh=1)
    _ax.set_yscale('symlog', linthresh=1)
    _ax.plot([0, rankings_df['wag_rank'].max()], [0, rankings_df['wag_rank'].max()],
             'k--', alpha=0.3, label='Equal rank')
    _ax.legend()

    _counts_md = "\n".join(f"- **{k}**: {v}" for k, v in category_counts.items())
    mo.vstack([
        _fig,
        mo.md(f"### Category Counts\n{_counts_md}\n\n**Total**: {len(rankings_df)}")
    ])
    return


@app.cell
def _(mo):
    category_dropdown = mo.ui.dropdown(
        options=['both_good', 'both_bad', 'wag_better', 'swag_better'],
        value='swag_better',
        label='Category')
    topk_slider = mo.ui.slider(start=1, stop=50, value=10, step=1, label='Top-K Threshold')
    mo.hstack([category_dropdown, topk_slider])
    return category_dropdown, topk_slider


@app.cell
def _(category_dropdown, mo, rankings_df, topk_slider):
    _k = topk_slider.value

    def _recat(row):
        wag_good = row['wag_rank'] < _k
        swag_good = row['swag_rank'] < _k
        if wag_good and swag_good:
            return 'both_good'
        elif not wag_good and not swag_good:
            return 'both_bad'
        elif wag_good:
            return 'wag_better'
        else:
            return 'swag_better'

    _df = rankings_df.copy()
    _df['category_dynamic'] = _df.apply(_recat, axis=1)
    filtered_df = _df[_df['category_dynamic'] == category_dropdown.value].copy()
    filtered_df['rank_diff'] = abs(filtered_df['wag_rank'] - filtered_df['swag_rank'])
    filtered_df = filtered_df.sort_values('rank_diff', ascending=False).reset_index(drop=True)

    example_slider = mo.ui.slider(
        start=0, stop=max(0, len(filtered_df) - 1), value=0, step=1,
        label=f'Example ({len(filtered_df)} total)')

    mo.vstack([
        example_slider,
        mo.ui.table(
            filtered_df[['pano_idx', 'wag_rank', 'swag_rank', 'rank_diff',
                          'wag_correct_sim', 'swag_correct_sim']].head(20))
    ])
    return example_slider, filtered_df


@app.cell
def _(
    dataset,
    example_slider,
    filtered_df,
    swag_similarity,
    vd,
    wag_similarity,
):
    _row = filtered_df.iloc[example_slider.value]
    selected_pano_idx = int(_row['pano_idx'])
    selected_correct_sat_idx = int(_row['correct_sat_idx'])
    selected_wag_rank = int(_row['wag_rank'])
    selected_swag_rank = int(_row['swag_rank'])
    selected_wag_top1_idx = int(wag_similarity[selected_pano_idx].argmax().item())
    selected_swag_top1_idx = int(swag_similarity[selected_pano_idx].argmax().item())

    correct_pair = vd.SamplePair(panorama_idx=selected_pano_idx, satellite_idx=selected_correct_sat_idx)
    wag_top1_pair = vd.SamplePair(panorama_idx=selected_pano_idx, satellite_idx=selected_wag_top1_idx)
    swag_top1_pair = vd.SamplePair(panorama_idx=selected_pano_idx, satellite_idx=selected_swag_top1_idx)

    correct_sample = dataset[correct_pair]
    wag_top1_sample = dataset[wag_top1_pair]
    swag_top1_sample = dataset[swag_top1_pair]
    return (
        correct_sample,
        selected_correct_sat_idx,
        selected_pano_idx,
        selected_swag_rank,
        selected_swag_top1_idx,
        selected_wag_rank,
        selected_wag_top1_idx,
        swag_top1_sample,
        wag_top1_sample,
    )


@app.cell
def _(
    correct_sample,
    matplotlib,
    mo,
    plt,
    selected_swag_rank,
    selected_swag_top1_idx,
    selected_wag_rank,
    selected_wag_top1_idx,
    swag_top1_sample,
    wag_top1_sample,
):
    def _show_row(fig, gs, row, sat_img, pano_img, top1_img, model_name, rank, top1_idx):
        _ax1 = fig.add_subplot(gs[row, 0])
        _ax1.imshow(sat_img.permute(1, 2, 0).cpu().numpy())
        _ax1.set_title(f'Correct Satellite')
        _ax1.axis('off')

        _ax2 = fig.add_subplot(gs[row, 1])
        _ax2.imshow(pano_img.permute(1, 2, 0).cpu().numpy())
        _ax2.set_title('Panorama')
        _ax2.axis('off')

        _ax3 = fig.add_subplot(gs[row, 2])
        _ax3.imshow(top1_img.permute(1, 2, 0).cpu().numpy())
        _ax3.set_title(f'{model_name} Top-1 (rank={rank}, sat={top1_idx})')
        _ax3.axis('off')

    _fig = plt.figure(figsize=(18, 8))
    _gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[1, 2, 1],
                                       left=0.02, right=0.98, bottom=0.02, top=0.95, wspace=0.05, hspace=0.15)

    _show_row(_fig, _gs, 0, correct_sample.satellite, correct_sample.panorama,
              wag_top1_sample.satellite, 'WAG', selected_wag_rank, selected_wag_top1_idx)
    _show_row(_fig, _gs, 1, correct_sample.satellite, correct_sample.panorama,
              swag_top1_sample.satellite, 'SWAG', selected_swag_rank, selected_swag_top1_idx)
    _fig.suptitle(f'WAG rank={selected_wag_rank}  |  SWAG rank={selected_swag_rank}', fontsize=14)

    mo.mpl.interactive(_fig)
    return


@app.cell
def _(pe, torch):
    def create_attention_recording_hook(name, attention_map):
        def __hook__(module, args, kwargs, _):
            if kwargs["need_weights"] == False:
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                _, attn_map = module(*args, **kwargs)
                attention_map[name] = attn_map
            return None
        return __hook__

    def create_input_token_recording_hook(name, input_tokens):
        def __hook__(model, args, _):
            input_tokens[name] = args[0]
            return None
        return __hook__

    def create_safa_attention_recording_hook(name, attention_map):
        def __hook__(model, args, kwargs, _):
            features = pe.extract_features(model._backbone, *args)
            attention_map[name] = model.safa(features)
            return None
        return __hook__

    def get_attention_maps(model, model_input):
        attention_maps = {}
        input_tokens = {}
        handles = []
        try:
            for idx in range(len(model._aggregator_model._encoder.layers)):
                h = model._aggregator_model._encoder.layers[idx].self_attn.register_forward_hook(
                    create_attention_recording_hook(f"agg_layer_{idx}", attention_maps), with_kwargs=True)
                handles.append(h)
            h = model._aggregator_model.register_forward_hook(
                create_input_token_recording_hook("input_tokens", input_tokens))
            handles.append(h)
            output = model(model_input)
        finally:
            for h in handles:
                h.remove()
            torch.cuda.synchronize()
        return output, attention_maps, input_tokens["input_tokens"]

    def get_safa_attention_maps(model, model_input):
        attention_maps = {}
        handles = []
        try:
            h = model.register_forward_hook(
                create_safa_attention_recording_hook("safa", attention_maps), with_kwargs=True)
            handles.append(h)
            output = model(model_input)
        finally:
            for h in handles:
                h.remove()
            torch.cuda.synchronize()
        return output, attention_maps

    return get_attention_maps, get_safa_attention_maps


@app.cell
def _(
    correct_sample,
    get_attention_maps,
    get_safa_attention_maps,
    swag_pano_model,
    swag_sat_model,
    swag_top1_sample,
    torch,
    vd,
    wag_pano_model,
    wag_sat_model,
    wag_top1_sample,
):
    # Build WAG batch: [correct_sat, wag_top1_sat] x [pano, pano]
    _wag_batch = vd.VigorDatasetItem(
        panorama_metadata=[correct_sample.panorama_metadata, wag_top1_sample.panorama_metadata],
        satellite_metadata=[correct_sample.satellite_metadata, wag_top1_sample.satellite_metadata],
        panorama=torch.stack([correct_sample.panorama, correct_sample.panorama]),
        satellite=torch.stack([correct_sample.satellite, wag_top1_sample.satellite]),
        cached_panorama_tensors={},
        cached_satellite_tensors={},
    )

    # Build SWAG batch: [correct_sat, swag_top1_sat] x [pano, pano]
    _swag_batch = vd.VigorDatasetItem(
        panorama_metadata=[correct_sample.panorama_metadata, swag_top1_sample.panorama_metadata],
        satellite_metadata=[correct_sample.satellite_metadata, swag_top1_sample.satellite_metadata],
        panorama=torch.stack([correct_sample.panorama, correct_sample.panorama]),
        satellite=torch.stack([correct_sample.satellite, swag_top1_sample.satellite]),
        cached_panorama_tensors={},
        cached_satellite_tensors={},
    )

    # WAG SAFA attention
    wag_sat_emb, wag_sat_attn = get_safa_attention_maps(
        wag_sat_model, wag_sat_model.model_input_from_batch(_wag_batch).to('cuda'))
    wag_pano_emb, wag_pano_attn = get_safa_attention_maps(
        wag_pano_model, wag_pano_model.model_input_from_batch(_wag_batch).to('cuda'))

    # SWAG transformer attention + gradient importance
    swag_sat_model.zero_grad()
    swag_pano_model.zero_grad()

    _swag_sat_out, swag_sat_attn, swag_sat_tokens = get_attention_maps(
        swag_sat_model, swag_sat_model.model_input_from_batch(_swag_batch).to('cuda'))
    _swag_pano_out, swag_pano_attn, swag_pano_tokens = get_attention_maps(
        swag_pano_model, swag_pano_model.model_input_from_batch(_swag_batch).to('cuda'))
    # Embeddings are (batch, num_embeddings, dim) — squeeze num_embeddings
    _swag_sat_emb = _swag_sat_out[0].squeeze(1)
    _swag_pano_emb = _swag_pano_out[0].squeeze(1)

    swag_sat_tokens.retain_grad()
    swag_pano_tokens.retain_grad()
    _sim_correct = _swag_sat_emb[0] @ _swag_pano_emb[0]
    _sim_incorrect = _swag_sat_emb[1] @ _swag_pano_emb[0]
    _delta = (_sim_correct - _sim_incorrect)
    _delta.backward()

    swag_sat_importance = swag_sat_tokens.grad.norm(dim=-1)
    swag_pano_importance = swag_pano_tokens.grad.norm(dim=-1)
    return (
        swag_pano_attn,
        swag_pano_importance,
        swag_sat_attn,
        swag_sat_importance,
        wag_pano_attn,
        wag_sat_attn,
    )


@app.cell
def _(PATCH_SIZE, plt, torch):
    def resize_attn_map(attn_map, size):
        attn_map = attn_map.reshape(1, 1, *attn_map.shape)
        attn_map = torch.nn.functional.interpolate(attn_map, size, mode='bilinear')
        return attn_map.squeeze()

    def get_token_grid(img_tensor):
        _, h, w = img_tensor.shape
        return (h // PATCH_SIZE, w // PATCH_SIZE)

    def plot_transformer_attention_heads(all_attentions, batch_idx, image):
        grid_h, grid_w = get_token_grid(image)
        num_layers = len(all_attentions)
        for _layer_idx in range(num_layers):
            _attn = all_attentions[f"agg_layer_{_layer_idx}"].detach().cpu()
            num_heads = _attn.shape[1]
            for _head_idx in range(num_heads):
                _cls_attn = _attn[batch_idx, _head_idx, 0, 1:]
                _cls_attn = _cls_attn[:grid_h * grid_w].reshape(grid_h, grid_w)
                _plot_idx = _layer_idx * num_heads + _head_idx + 1
                plt.subplot(num_layers, num_heads, _plot_idx)
                plt.imshow(image.permute(1, 2, 0).cpu())
                plt.imshow(resize_attn_map(_cls_attn, image.shape[1:]), alpha=0.5)
                if _head_idx == 0:
                    plt.ylabel(f'Layer {_layer_idx}')
                if _layer_idx == 0:
                    plt.title(f'Head {_head_idx}')
                plt.xticks([])
                plt.yticks([])

    def plot_safa_attention_heads(all_attentions, batch_idx, image):
        grid_h, grid_w = get_token_grid(image)
        attn = all_attentions["safa"][batch_idx]
        num_heads = attn.shape[0]
        attn = attn.reshape(num_heads, grid_h, grid_w)
        for i in range(num_heads):
            plt.subplot(1, num_heads, i + 1)
            plt.imshow(image.permute(1, 2, 0).cpu())
            attn_to_plot = resize_attn_map(attn[i].detach().cpu(), image.shape[1:])
            attn_to_plot = torch.abs(attn_to_plot)
            attn_to_plot = attn_to_plot / attn_to_plot.max()
            plt.imshow(attn_to_plot, alpha=0.5)
            if i == 0:
                plt.ylabel('SAFA')
            plt.title(f'Head {i}')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()

    def plot_gradient_importance(importance, batch_idx, image):
        grid_h, grid_w = get_token_grid(image)
        imp = importance[batch_idx, 1:].detach().cpu()
        imp = imp[:grid_h * grid_w].reshape(grid_h, grid_w)
        plt.imshow(image.permute(1, 2, 0).cpu())
        plt.imshow(resize_attn_map(imp, image.shape[1:]), alpha=0.5)
        plt.xticks([])
        plt.yticks([])

    return (
        plot_gradient_importance,
        plot_safa_attention_heads,
        plot_transformer_attention_heads,
    )


@app.cell
def _(
    correct_sample,
    plot_safa_attention_heads,
    plt,
    wag_pano_attn,
    wag_sat_attn,
    wag_top1_sample,
):
    wag_pano_fig = plt.figure(figsize=(20, 4))
    plot_safa_attention_heads(wag_pano_attn, 0, correct_sample.panorama)
    plt.suptitle('WAG SAFA — Panorama')

    wag_correct_sat_fig = plt.figure(figsize=(10, 4))
    plot_safa_attention_heads(wag_sat_attn, 0, correct_sample.satellite)
    plt.suptitle('WAG SAFA — Correct Satellite')

    wag_top1_sat_fig = plt.figure(figsize=(10, 4))
    plot_safa_attention_heads(wag_sat_attn, 1, wag_top1_sample.satellite)
    plt.suptitle('WAG SAFA — Top-1 Satellite')
    return wag_correct_sat_fig, wag_pano_fig, wag_top1_sat_fig


@app.cell
def _(
    correct_sample,
    plot_transformer_attention_heads,
    plt,
    swag_pano_attn,
    swag_sat_attn,
    swag_top1_sample,
):
    swag_pano_fig = plt.figure(figsize=(24, 12))
    plot_transformer_attention_heads(swag_pano_attn, 0, correct_sample.panorama)
    plt.suptitle('SWAG Transformer — Panorama')

    swag_correct_sat_fig = plt.figure(figsize=(24, 12))
    plot_transformer_attention_heads(swag_sat_attn, 0, correct_sample.satellite)
    plt.suptitle('SWAG Transformer — Correct Satellite')

    swag_top1_sat_fig = plt.figure(figsize=(24, 12))
    plot_transformer_attention_heads(swag_sat_attn, 1, swag_top1_sample.satellite)
    plt.suptitle('SWAG Transformer — Top-1 Satellite')
    return swag_correct_sat_fig, swag_pano_fig, swag_top1_sat_fig


@app.cell
def _(
    correct_sample,
    matplotlib,
    plot_gradient_importance,
    plt,
    swag_pano_importance,
    swag_sat_importance,
    swag_top1_sample,
):
    grad_fig = plt.figure(figsize=(18, 5))
    _gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1],
                                       left=0.02, right=0.98, bottom=0.05, top=0.9, wspace=0.05)

    plt.subplot(_gs[0])
    plot_gradient_importance(swag_sat_importance, 0, correct_sample.satellite)
    plt.title('Correct Satellite')

    plt.subplot(_gs[1])
    plot_gradient_importance(swag_pano_importance, 0, correct_sample.panorama)
    plt.title('Panorama')

    plt.subplot(_gs[2])
    plot_gradient_importance(swag_sat_importance, 1, swag_top1_sample.satellite)
    plt.title('Top-1 Satellite')

    plt.suptitle('SWAG Gradient Importance (token grad norms)', fontsize=14)
    return (grad_fig,)


@app.cell
def _(
    grad_fig,
    mo,
    swag_correct_sat_fig,
    swag_pano_fig,
    swag_top1_sat_fig,
    wag_correct_sat_fig,
    wag_pano_fig,
    wag_top1_sat_fig,
):
    mo.ui.tabs({
        "WAG SAFA Attention": mo.vstack([wag_pano_fig, wag_correct_sat_fig, wag_top1_sat_fig]),
        "SWAG Transformer Attention": mo.vstack([swag_pano_fig, swag_correct_sat_fig, swag_top1_sat_fig]),
        "SWAG Gradient Importance": grad_fig,
    })
    return


@app.cell
def _(
    mo,
    selected_correct_sat_idx,
    selected_pano_idx,
    selected_swag_rank,
    selected_swag_top1_idx,
    selected_wag_rank,
    selected_wag_top1_idx,
    swag_similarity,
    wag_similarity,
):
    _pi = selected_pano_idx
    _ci = selected_correct_sat_idx
    mo.md(f"""
    ### Similarity Score Comparison

    | Metric | WAG | SWAG |
    |--------|-----|------|
    | Correct Sim | {wag_similarity[_pi, _ci].item():.4f} | {swag_similarity[_pi, _ci].item():.4f} |
    | Max Sim | {wag_similarity[_pi].max().item():.4f} | {swag_similarity[_pi].max().item():.4f} |
    | Rank | {selected_wag_rank} | {selected_swag_rank} |
    | Top-1 Sat Idx | {selected_wag_top1_idx} | {selected_swag_top1_idx} |
    """)
    return


if __name__ == "__main__":
    app.run()
