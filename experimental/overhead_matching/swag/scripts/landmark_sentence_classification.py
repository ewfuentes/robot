import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _(Path, psle):
    _config = psle.PanoramaSemanticLandmarkExtractorConfig(
        openai_embedding_size=1536,
        embedding_version='pano_v1',
        auxiliary_info_key=None
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
def _(output, torch):
    # Print landmark sentences from debug output
    for _batch_idx in range(len(output.mask)):
        for _lm_idx in range(len(output.mask[_batch_idx])):
            if output.mask[_batch_idx, _lm_idx]:
                continue
            _sentence = bytearray(output.debug["sentences"][_batch_idx, _lm_idx]).strip(b'\x00').decode('utf-8')
            # Get the feature vector norm as a simple metric
            _feature_norm = torch.norm(output.features[_batch_idx, _lm_idx]).item()
            print(f"Batch {_batch_idx}, Landmark {_lm_idx}: {_sentence:135s} (norm: {_feature_norm:.3f})")
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
