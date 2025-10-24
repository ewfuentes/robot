import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from experimental.overhead_matching.swag.model.semantic_landmark_extractor import (
        SemanticLandmarkExtractor, SemanticLandmarkExtractorConfig)
    from experimental.overhead_matching.swag.data import (
        vigor_dataset as vd
    )

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    import numpy as np
    from common.ollama import pyollama
    import json
    import tqdm
    import tqdm.asyncio
    import pandas as pd
    import geopandas as gpd
    from pathlib import Path
    import math
    return (
        SemanticLandmarkExtractor,
        SemanticLandmarkExtractorConfig,
        json,
        mo,
        np,
        plt,
        pyollama,
        tqdm,
        vd,
    )


@app.cell
def _():
    from experimental.overhead_matching.swag.model import swag_model_input_output as smio
    return (smio,)


@app.cell
def _():
    from experimental.overhead_matching.swag.model.semantic_landmark_extractor import prune_landmark
    return (prune_landmark,)


@app.cell
def _():
    import openai
    return (openai,)


@app.cell
def _():
    import asyncio
    return


@app.cell
def _(SemanticLandmarkExtractorConfig):
    SemanticLandmarkExtractorConfig()
    return


@app.cell
def _(SemanticLandmarkExtractor, SemanticLandmarkExtractorConfig):
    model = SemanticLandmarkExtractor(SemanticLandmarkExtractorConfig(llm_str='gemma3:4b-it-qat'))
    return (model,)


@app.cell
def _(vd):
    dataset = vd.VigorDataset('/data/overhead_matching/datasets/VIGOR/Chicago',
                             vd.VigorDatasetConfig(
                                 satellite_tensor_cache_info=None,
                                 panorama_tensor_cache_info=None,
                                 landmark_version='v3'
                             ))
    return (dataset,)


@app.cell
def _(vd):
    seattle_dataset = vd.VigorDataset('/data/overhead_matching/datasets/VIGOR/Seattle',
                             vd.VigorDatasetConfig(
                                 satellite_tensor_cache_info=None,
                                 panorama_tensor_cache_info=None,
                                 landmark_version='v3'
                             ))
    return (seattle_dataset,)


@app.cell
def _(seattle_dataset):

    unique_landmarks = set()
    # for _, v  in dataset._landmark_metadata.iterrows():
    #     unique_landmarks.add(v['pruned_props'])

    for _, v  in seattle_dataset._landmark_metadata.iterrows():
        unique_landmarks.add(v['pruned_props'])

    len(unique_landmarks)
    return (unique_landmarks,)


@app.cell
def _(unique_landmarks):
    for _i, _v in enumerate(unique_landmarks):
        if _i > 100:
            break
        print(_i, _v)
    return


@app.cell
def _(json, openai, unique_landmarks):
    system_prompt = "Your job is to produce short natural language descriptions of openstreetmap landmarks that are helpful for visually identifying the landmark. For example, do not include information about building identifiers that are unlikely to be discernable by visual inspection. Don't include any details not derived from the provided landmark information. Don't include descriptions about the lack of information. Do not include instructions on how to identify the landmark. Do include an address if provided."
    prompt_text = "Produce a short natural language description for this landmark: "


    tasks = []
    client = openai.AsyncOpenAI()
    for idx, item in enumerate(unique_landmarks):
        if idx > 100:
            break
        props = dict(item)
        tasks.append(client.chat.completions.create(
            model="gpt-5-nano",
            reasoning_effort='low',
            max_tokens=3000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text + json.dumps(props)}]))
    return item, tasks


@app.cell
def _(tasks):
    tasks[0]
    return


@app.cell
async def _(tasks, tqdm):
    responses = await tqdm.asyncio.tqdm_asyncio.gather(*tasks, desc='generating descriptions')
    return (responses,)


@app.cell
def _(responses):
    responses
    return


@app.cell
def _(responses):
    max([r.usage.total_tokens for r in responses])
    return


@app.cell
def _(dataset, vd):
    dataloader = vd.get_dataloader(dataset, batch_size=5)
    return (dataloader,)


@app.cell
def _(dataloader):
    batch = next(iter(dataloader))
    return (batch,)


@app.cell
def _(batch):
    batch
    return


@app.cell
def _(batch, smio):
    model_input = smio.ModelInput(metadata=batch.satellite_metadata, image=batch.satellite, cached_tensors=None)
    return (model_input,)


@app.cell
def _(model, model_input):
    extractor_output = model(model_input)
    return (extractor_output,)


@app.cell
def _(extractor_output):
    bytes(extractor_output.debug["sentences"][0][0])
    return


@app.cell
def _(batch):
    batch.satellite_metadata
    return


@app.cell
def _(item):
    item.panorama_metadata
    return


@app.cell
def _(dataset, mo, np, plt):
    num_sat_landmark_idxs = []
    num_pano_landmark_idxs = []
    for _, row in dataset._satellite_metadata.iterrows():
        num_sat_landmark_idxs.append(len(row.landmark_idxs))

    for _, row in dataset._panorama_metadata.iterrows():
        num_pano_landmark_idxs.append(len(row.landmark_idxs))

    bins = np.arange(-0.5, 15.5, 1.0)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(121)
    plt.hist(num_sat_landmark_idxs, bins=bins)
    plt.title(f'Satellite (Count: {len(dataset._satellite_metadata)})')
    plt.yscale('log')
    plt.subplot(122, sharey=ax)
    plt.hist(num_pano_landmark_idxs, bins=bins)
    plt.ylim(100, 10000)
    plt.title(f'Panorama (Count: {len(dataset._panorama_metadata)})')
    plt.yscale('log')
    plt.suptitle(f'Chicago Landmarks (Count: {len(dataset._landmark_metadata)})')
    fig.supylabel('Count')
    fig.supxlabel('Number of landmarks')
    plt.tight_layout()
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(dataset):
    _mask = [len(x) == 0 for x in dataset._satellite_metadata.landmark_idxs]
    dataset._satellite_metadata[_mask]
    return


@app.cell
def _(dataset):
    dataset._panorama_metadata.iloc[1172]
    return


@app.cell
def _(dataset, df, json, pyollama, tqdm):

    _unique_landmarks = set() 
    check_date_cols = [x for x in dataset._landmark_metadata.columns if x.startswith('check_date')]
    for i, _df in dataset._landmark_metadata.iterrows():
        fields = df.drop([
            "web_mercator_y", "web_mercator_x", "panorama_idxs", "satellite_idxs",
            "landmark_type", 'element', 'id', 'geometry', 'opening_hours', 'website',
            'addr:city', 'addr:state'] + check_date_cols)
        fields = fields.dropna()
        fields = fields.to_dict()
        fields = frozenset(fields.items())
        _unique_landmarks.add(fields)

    descriptions = []
    with pyollama.Ollama('gemma3:4b-it-qat') as ollama:
        for _i, obj in tqdm.tqdm(list(enumerate(_unique_landmarks))):
            d = dict(obj)
            # print(dict(obj))
            try:
                prompt = f"Generate a natural language description of this openstreetmap landmark. Only include information relevant for visually identifying the object. For example, don't include payment methods accepted. Don't include any details not derived from the landmark information. Include no other details: {json.dumps(d)}"
                # prompt = f"Generate a brief description for this OpenStreetMap landmark dictionary. The brief description should describe the landmark, only including information in the dictionary relevant to visually identifying the landmark if you were walking by it on the street. Don't include addresses, but include the name if it has one.\n{d}"
                description = ollama(prompt)
                descriptions.append(description)
            except:
                print('failed to serialize', d)
                descriptions.append(None)

            # print(ollama(prompt))
            if _i > 200:
                break

    return


if __name__ == "__main__":
    app.run()
