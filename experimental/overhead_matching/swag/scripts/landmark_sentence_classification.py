import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _(Path, psle):
    _config = psle.PanoramaSemanticLandmarkExtractorConfig(
        openai_embedding_size=1536,
        embedding_version='pano_v1',
        auxiliary_info_key=None,
        should_classify_against_grouping=True
    )

    model = psle.PanoramaSemanticLandmarkExtractor(
        _config, Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/'))
    model.load_files()
    return (model,)


@app.cell
def _(Path, vd):
    dataset = vd.VigorDataset(
        Path('/data/overhead_matching/datasets/VIGOR/Chicago'),
        config=vd.VigorDatasetConfig(
            satellite_tensor_cache_info=None,
            panorama_tensor_cache_info=None,
            should_load_images=False,
            should_load_landmarks=False,
            landmark_version='v3'
        )
    ).get_pano_view()
    return (dataset,)


@app.cell
def _(dataset, model, psle, vd):
    _dataloader = vd.get_dataloader(dataset, batch_size=128)
    _batch = next(iter(_dataloader))
    sample_idx = 0
    _sample = dataset[sample_idx]
    _model_input = psle.ModelInput(
        image=_batch.panorama,
        metadata=_batch.panorama_metadata,
        cached_tensors={}
    )

    output  = model(_model_input)
    return (output,)


@app.cell
def _(model, output, torch):
    _high_level_classes = list(model.semantic_groupings["semantic_groups"].keys())
    _low_level_classes = list(model.semantic_groupings["class_details"].keys())

    for _batch_idx, _lm_idx, _high_level_class_idx in zip(*torch.where(output.features)):
        if output.mask[_batch_idx, _lm_idx]:
            continue
        # print(_batch_idx, _lm_idx)
        _sentence = bytearray(output.debug["sentences"][_batch_idx, _lm_idx]).strip(b'\x00').decode('utf-8')
        _low_level_classification = torch.where(output.debug["low_level_classification"][_batch_idx, _lm_idx])[0].item()
        _sorted_low_level_sim = torch.sort(output.debug["low_level_similarity"][_batch_idx, _lm_idx])
        top_k_low_level = [f"({_low_level_classes[_sorted_low_level_sim.indices[_low_level_idx]]}, {_sorted_low_level_sim.values[_low_level_idx].item():0.3f})" for _low_level_idx in range(-1, -4, -1)]
        # print('sentence:', _sentence)
        print(f"high level: {_high_level_classes[_high_level_class_idx]:<25} {_sentence:135s} {', '.join(top_k_low_level)}")
        # for _idx, _sentence_buffer in enumerate(output.debug["sentences"][_batch_idx][~output.mask[_batch_idx]]):
        #     print(bytearray(_sentence_buffer).decode('utf-8'), )
    return


@app.cell
def _():
    from pathlib import Path
    return (Path,)


@app.cell
def _():
    import marimo as mo
    from experimental.overhead_matching.swag.model import (
        panorama_semantic_landmark_extractor as psle
    )
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )
    return psle, vd


@app.cell
def _():
    import matplotlib.pyplot as plt
    return


@app.cell
def _():
    import common.torch.load_torch_deps
    import torch
    return (torch,)


if __name__ == "__main__":
    app.run()
