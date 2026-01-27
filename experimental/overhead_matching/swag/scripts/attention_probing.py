import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    import json
    from pathlib import Path
    import pandas as pd

    from common.torch import (
        load_torch_deps,
        load_and_save_models as lsm
    )
    import torch
    import seaborn as sns
    from experimental.overhead_matching.swag.model import (
        patch_embedding as pe
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    from experimental.overhead_matching.swag.evaluation import (
        evaluate_swag as es
    )
    return Path, es, json, lsm, matplotlib, mo, pd, pe, plt, sns, torch, vd


@app.cell
def _(mo):
    mo.md("""## Find an interesting snippet""")
    return


@app.cell
def _(Path, json, lsm, pd, torch, vd):
    def extract_path_data(path_dir: Path, dataset: vd.VigorDataset):
        path_id = int(path_dir.name)
        records = []
        error = torch.load(path_dir / "error.pt")
        var = torch.load(path_dir / "var.pt")
        path = torch.load(path_dir / "path.pt")
        # Check for old format
        if path and isinstance(path[0], int):
            raise ValueError(
                f"path.pt in '{path_dir}' uses old index format (integers). "
                "Re-run evaluation with new path files to get pano_id format (strings)."
            )
        similarity = torch.load(path_dir / "similarity.pt").cpu()
        max_similarity = torch.max(similarity,dim=1)
        for i in range(len(path)):
            pano_metadata = dataset._panorama_metadata[
                dataset._panorama_metadata['pano_id'] == path[i]].iloc[0]
            sat_idx = pano_metadata.satellite_idx
            positive_similarity = similarity[i, sat_idx]
            records.append({
                'model': path_dir.parts[-2],
                'path_idx': path_id,
                'timestep_idx': i,
                'panorama_id': path[i],
                'positive_similarity': positive_similarity.item(),
                'max_similarity': max_similarity.values[i].item(),
                'max_similarity_idx': max_similarity.indices[i].item(),
                "error_m": error[i].item(),
                'var_sq_m': var[i].item()})
        return records


    def load_eval_results(p: Path):
        records = []
        args_json = json.loads((p / "args.json").read_text())
        dataset = vd.VigorDataset(args_json["dataset_path"],
                                 vd.VigorDatasetConfig(satellite_tensor_cache_info=None,
                                                      panorama_tensor_cache_info=None))
        for path_dir in sorted(p.glob("00[0-9]*")):
            records.extend(extract_path_data(path_dir, dataset))
        return pd.DataFrame.from_records(records)

    def load_dataset_and_model(p: Path):
        args_json = json.loads((p / "args.json").read_text())
        sat_model = lsm.load_model(args_json["sat_path"]).eval().cuda()
        pano_model = lsm.load_model(args_json["pano_path"]).eval().cuda()

        dataset = vd.VigorDataset(
            args_json["dataset_path"],
            vd.VigorDatasetConfig(
                panorama_tensor_cache_info=None,
                satellite_tensor_cache_info=None,
                satellite_patch_size=sat_model.patch_dims,
                panorama_size=pano_model.patch_dims))

        return dataset, sat_model, pano_model



    results_path = Path('/data/overhead_matching/evaluation/results/20250719_swag_model/NewYork/all_chicago_sat_dino_pano_dino_agg_small_attn_8')

    dataset, sat_model, pano_model = load_dataset_and_model(results_path)
    eval_results_df = load_eval_results(results_path)
    return dataset, eval_results_df, pano_model, sat_model


@app.cell
def _(Path, dataset, es, lsm, vd):
    baseline_model_path = Path('/data/overhead_matching/models/20250707_dino_features/all_chicago_dino_project_512')
    # baseline_model_path = Path('/data/overhead_matching/models/20250806_swag_model_fixed/all_chicago_sat_dino_pano_dino_agg_small_attn_8_layers_1/')
    baseline_sat_model = lsm.load_model(baseline_model_path / "0059_satellite").eval().cuda()
    baseline_pano_model = lsm.load_model(baseline_model_path / "0059_panorama").eval().cuda()
    baseline_dataset = vd.VigorDataset(
        Path('/data/overhead_matching/datasets/VIGOR/NewYork'),
        config=vd.VigorDatasetConfig(
            panorama_tensor_cache_info=None,
            satellite_tensor_cache_info=None,
            satellite_patch_size=baseline_sat_model.patch_dims,
            panorama_size=baseline_pano_model.patch_dims))

    baseline_similarity_matrix = es.compute_cached_similarity_matrix(baseline_sat_model, baseline_pano_model, dataset, device='cuda:0', use_cached_similarity=True)
    return (
        baseline_dataset,
        baseline_pano_model,
        baseline_sat_model,
        baseline_similarity_matrix,
    )


@app.cell
def _(eval_results_df):
    eval_results_df
    return


@app.cell
def _(eval_results_df, mo, plt, sns):
    # Find a high error path
    _final_error_df = eval_results_df.groupby(["model", "path_idx"], as_index=False).apply(func=lambda x: x.iloc[-1])
    _max_idx = _final_error_df["error_m"].argmax()
    max_error_path_df = eval_results_df[eval_results_df["path_idx"] == _max_idx].reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1, 1)
    ax = sns.lineplot(max_error_path_df, y='error_m', x='timestep_idx')
    plt.title(f'Path Idx: {_max_idx}')
    plt.subplot(2, 1, 2, sharex=ax)
    sns.lineplot(max_error_path_df, x='timestep_idx', y="positive_similarity")
    sns.lineplot(max_error_path_df, x='timestep_idx', y="max_similarity")

    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return (max_error_path_df,)


@app.cell
def _(mo):
    mo.md(r"""## Probe the Attention and SAFA Heads""")
    return


@app.cell
def _(max_error_path_df):
    max_error_path_df
    return


@app.cell
def _(baseline_dataset, dataset, max_error_path_df, vd):
    timestep_idx = 0

    _pano_idx = max_error_path_df.loc[timestep_idx, 'panorama_id']
    _max_similarity_idx = max_error_path_df.loc[timestep_idx, 'max_similarity_idx']
    _pano_metadata = dataset._panorama_metadata.loc[_pano_idx]
    correct_pair = vd.SamplePair(panorama_idx=_pano_idx, satellite_idx=_pano_metadata.satellite_idx)
    max_sim_pair = vd.SamplePair(panorama_idx=_pano_idx, satellite_idx=_max_similarity_idx)
    sample_pair = dataset[correct_pair]
    max_similarity_pair = dataset[max_sim_pair]
    baseline_sample_pair = baseline_dataset[correct_pair]
    baseline_max_similarity_pair = baseline_dataset[max_sim_pair]
    return (
        baseline_max_similarity_pair,
        baseline_sample_pair,
        correct_pair,
        max_similarity_pair,
        sample_pair,
    )


@app.cell
def _(matplotlib, max_similarity_pair, mo, plt, sample_pair):
    _fig = plt.figure(figsize=(15, 5))
    _gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1],
                                       left=0.04, right=0.98, bottom=0.05, top=0.95, wspace=0.0)
    _ax1 = _fig.add_subplot(_gs[0])
    _ax1.imshow(sample_pair.satellite.permute(1, 2, 0).numpy())
    _ax1.set_title(f"Correct Sat")

    _ax2 = _fig.add_subplot(_gs[1])
    _ax2.imshow(sample_pair.panorama.permute(1, 2, 0).numpy())
    _ax2.set_title("Panorama")
    _ax2.set_yticklabels([])

    _ax3 = _fig.add_subplot(_gs[2])
    _ax3.imshow(max_similarity_pair.satellite.permute(1, 2, 0).numpy())
    _ax3.set_title("Max Similarity")
    _ax3.set_yticklabels([])

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

    def get_attention_maps(model, input):
        attention_maps = {}
        input_tokens = {}
        # register forward hooks
        for idx in range(len(model._aggregator_model._encoder.layers)):
            model._aggregator_model._encoder.layers[idx].self_attn.register_forward_hook(
                create_attention_recording_hook(f"agg_layer_{idx}", attention_maps), with_kwargs=True)
        model._aggregator_model.register_forward_hook(
            create_input_token_recording_hook("input_tokens", input_tokens))

        try:
            output = model(input)
        except Exception as e:
            print(e)
        finally:
            torch.cuda.synchronize()
            for m in model.modules():
                m._forward_hooks.clear()
        return output, attention_maps, input_tokens["input_tokens"]

    def get_safa_attention_maps(model, input):
        attention_maps = {}
        model.register_forward_hook(
            create_safa_attention_recording_hook("safa", attention_maps), with_kwargs=True)

        try:
            output = model(input)
        except Exception as e:
            print(e)
        finally:
            torch.cuda.synchronize()
            for m in model.modules():
                m._forward_hooks.clear()
        return output, attention_maps
    return get_attention_maps, get_safa_attention_maps


@app.cell
def _(
    baseline_max_similarity_pair,
    baseline_pano_model,
    baseline_sample_pair,
    baseline_sat_model,
    get_attention_maps,
    get_safa_attention_maps,
    max_similarity_pair,
    pano_model,
    sample_pair,
    sat_model,
    torch,
    vd,
):
    _batch = vd.VigorDatasetItem(
        panorama_metadata=[sample_pair.panorama_metadata, max_similarity_pair.panorama_metadata],
        satellite_metadata=[sample_pair.satellite_metadata, max_similarity_pair.satellite_metadata],
        panorama=torch.stack([sample_pair.panorama, sample_pair.panorama]),
        satellite=torch.stack([sample_pair.satellite, max_similarity_pair.satellite]),
        cached_panorama_tensors={},
        cached_satellite_tensors={},
    )

    _baseline_batch = vd.VigorDatasetItem(
        panorama_metadata=[baseline_sample_pair.panorama_metadata, baseline_max_similarity_pair.panorama_metadata],
        satellite_metadata=[baseline_sample_pair.satellite_metadata, baseline_max_similarity_pair.satellite_metadata],
        panorama=torch.stack([baseline_sample_pair.panorama, baseline_sample_pair.panorama]),
        satellite=torch.stack([baseline_sample_pair.satellite, baseline_max_similarity_pair.satellite]),
        cached_panorama_tensors={},
        cached_satellite_tensors={},
    )

    # Run through the transformer model
    sat_model.zero_grad()
    pano_model.zero_grad()

    sat_embeddings, sat_attn, sat_tokens = get_attention_maps(
        sat_model, sat_model.model_input_from_batch(_batch).to('cuda'))
    pano_embeddings, pano_attn, pano_tokens = get_attention_maps(
        pano_model, pano_model.model_input_from_batch(_batch).to('cuda'))
    sat_tokens.retain_grad()
    pano_tokens.retain_grad()
    # sat_tokens.grad.zero_()
    # pano_tokens.grad.zero_()
    sim_correct = sat_embeddings[0] @ pano_embeddings[0]
    sim_incorrect = sat_embeddings[1] @ pano_embeddings[0]
    delta = (sim_correct - sim_incorrect)
    delta.backward()

    # Run through the baseline model
    baseline_sat_embeddings, baseline_sat_attn = get_safa_attention_maps(baseline_sat_model, baseline_sat_model.model_input_from_batch(_baseline_batch).to("cuda"))
    baseline_pano_embeddings, baseline_pano_attn = get_safa_attention_maps(baseline_pano_model, baseline_pano_model.model_input_from_batch(_baseline_batch).to("cuda"))

    return (
        baseline_pano_attn,
        baseline_pano_embeddings,
        baseline_sat_attn,
        baseline_sat_embeddings,
        pano_attn,
        pano_tokens,
        sat_attn,
        sat_tokens,
    )


@app.cell
def _(
    matplotlib,
    max_similarity_pair,
    mo,
    pano_tokens,
    plt,
    sample_pair,
    sat_tokens,
    torch,
):
    sat_importance = sat_tokens.grad.norm(dim=-1)
    pano_importance = pano_tokens.grad.norm(dim=-1)

    def resize_attn_map(attn_map, size):
        attn_map = attn_map.reshape(1, 1, *attn_map.shape)
        attn_map = torch.nn.functional.interpolate(
            attn_map, size, mode='bilinear')
        return attn_map.squeeze()

    _fig = plt.figure(figsize=(15, 5))
    _gs = matplotlib.gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1],
                                       left=0.04, right=0.98, bottom=0.05, top=0.95, wspace=0.0)
    _ax1 = _fig.add_subplot(_gs[0])
    _ax1.imshow(sample_pair.satellite.cpu().permute(1, 2, 0))
    _attn_to_plot = resize_attn_map(
        sat_importance[0, 1:].reshape(23, -1).cpu(), sample_pair.satellite.shape[1:])
    _ax1.imshow(_attn_to_plot, alpha=0.5)
    _ax1.set_title(f"Correct Sat")

    _ax2 = _fig.add_subplot(_gs[1])
    _ax2.imshow(sample_pair.panorama.permute(1, 2, 0).cpu())
    _attn_to_plot = resize_attn_map(
        pano_importance[0, 1:].reshape(23, -1).cpu(), sample_pair.panorama.shape[1:])
    _ax2.imshow(_attn_to_plot, alpha=0.5)
    _ax2.set_title("Panorama")
    _ax2.set_yticklabels([])

    _ax3 = _fig.add_subplot(_gs[2])
    _ax3.imshow(max_similarity_pair.satellite.permute(1, 2, 0).cpu())
    _attn_to_plot = resize_attn_map(
        sat_importance[1, 1:].reshape(23, -1).cpu(), max_similarity_pair.satellite.shape[1:])
    _ax3.imshow(_attn_to_plot, alpha=0.5)
    _ax3.set_title("Max Similarity")
    _ax3.set_yticklabels([])

    mo.mpl.interactive(_fig)
    return (resize_attn_map,)


@app.cell
def _():
    return


@app.cell
def _(plt, resize_attn_map):
    def plot_attention_heads(all_attentions, batch_idx, image):
        num_layers = len(all_attentions)
        for _layer_idx in range(len(all_attentions)):
            _attn = all_attentions[f"agg_layer_{_layer_idx}"].detach().cpu()
            num_attention_heads = _attn.shape[1]
            for _head_idx in range(num_attention_heads):
                _cls_attn = _attn[batch_idx, _head_idx, 0, 1:]
                _num_rows = 23
                _cls_attn = _cls_attn.reshape(23, -1)
                _plot_idx = _layer_idx * num_attention_heads + _head_idx + 1
                plt.subplot(num_layers, num_attention_heads, _plot_idx)

                plt.imshow(image.permute(1, 2, 0).cpu())
                _attn_to_plot = resize_attn_map(_cls_attn, image.shape[1:])
                plt.imshow(_attn_to_plot, alpha=0.5)

                if _plot_idx % num_attention_heads == 1:
                    plt.ylabel(f'agg_layer_{_layer_idx}')
                if _plot_idx <= num_attention_heads: 
                    plt.title(f'Attention Head: {_head_idx}')

    return (plot_attention_heads,)


@app.cell
def _(
    max_similarity_pair,
    mo,
    pano_attn,
    plot_attention_heads,
    plt,
    sample_pair,
    sat_attn,
):
    _pano_fig = plt.figure(figsize=(24, 4))
    plot_attention_heads(pano_attn, 0, sample_pair.panorama.cpu())
    plt.suptitle('Panorama')
    _correct_sat_fig = plt.figure(figsize=(24, 4))
    plot_attention_heads(sat_attn, 0, sample_pair.satellite.cpu())
    plt.suptitle('Correct Satellite Patch')
    _incorrect_sat_fig = plt.figure(figsize=(24, 4))
    plot_attention_heads(sat_attn, 1, max_similarity_pair.satellite.cpu())
    plt.suptitle('Incorrect Satellite Patch')

    mo.vstack([_pano_fig, _correct_sat_fig, _incorrect_sat_fig])
    return


@app.cell
def _(plt, resize_attn_map, torch):
    def plot_safa_attention_heads(all_attentions, batch_idx, image):
        print(all_attentions["safa"].shape)
        attn = all_attentions["safa"][batch_idx]
        num_attn_heads = attn.shape[0]
        attn = attn.reshape(num_attn_heads, 23, -1)

        for i in range(num_attn_heads):
            plt.subplot(1, num_attn_heads, i + 1)
            plt.imshow(image.permute(1, 2, 0).cpu())
            attn_to_plot = resize_attn_map(attn[i].detach().cpu(), image.shape[1:])
            attn_to_plot = torch.abs(attn_to_plot)
            attn_to_plot = attn_to_plot / attn_to_plot.max()
            plt.imshow(attn_to_plot, alpha=0.5)

            if i % num_attn_heads == 0:
                plt.ylabel(f'SAFA')
            if i <= num_attn_heads: 
                plt.title(f'Attention Head: {i}')
        plt.tight_layout()
    return (plot_safa_attention_heads,)


@app.cell
def _(
    baseline_max_similarity_pair,
    baseline_pano_attn,
    baseline_sample_pair,
    baseline_sat_attn,
    mo,
    plot_safa_attention_heads,
    plt,
):
    _pano_fig = plt.figure(figsize=(20, 4))
    plot_safa_attention_heads(baseline_pano_attn, 0, baseline_sample_pair.panorama)
    plt.suptitle("panorama")

    _correct_fig = plt.figure(figsize=(20, 4))
    plot_safa_attention_heads(baseline_sat_attn, 0, baseline_sample_pair.satellite)
    plt.suptitle('correct')

    _max_sim_fig = plt.figure(figsize=(20, 4))
    plot_safa_attention_heads(baseline_pano_attn, 1, baseline_max_similarity_pair.satellite)
    plt.suptitle("max similarity")

    # _pano_fig = plt.figure(figsize=(20, 4))
    # plot_attention_heads(baseline_pano_attn, 0, baseline_sample_pair.panorama)
    # plt.suptitle("panorama")

    # _correct_fig = plt.figure(figsize=(20, 4))
    # plot_attention_heads(baseline_sat_attn, 0, baseline_sample_pair.satellite)
    # plt.suptitle('correct')

    # _max_sim_fig = plt.figure(figsize=(20, 4))
    # plot_attention_heads(baseline_sat_attn, 1, baseline_max_similarity_pair.satellite)
    # plt.suptitle("max similarity")

    mo.vstack([_pano_fig, _correct_fig, _max_sim_fig])
    # mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(baseline_pano_embeddings, baseline_sat_embeddings):
    baseline_pano_embeddings @ baseline_sat_embeddings.T
    return


@app.cell
def _(baseline_dataset, baseline_similarity_matrix, correct_pair, plt, vd):
    _max = baseline_similarity_matrix[correct_pair.panorama_idx].max(dim=0)
    _max_baseline_similarity = baseline_dataset[vd.SamplePair(panorama_idx=correct_pair.panorama_idx, satellite_idx=_max.indices.item())]
    _fig = plt.figure()
    plt.imshow(_max_baseline_similarity.satellite.permute(1, 2, 0))
    _fig
    return


@app.cell
def _(baseline_pano_attn, torch):
    torch.abs(baseline_pano_attn["safa"])
    return


@app.cell
def _(sat_model):
    sat_model
    return


@app.cell
def _(baseline_sat_model):
    baseline_sat_model
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
