import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import experimental.overhead_matching.swag.model.semantic_landmark_extractor as sle
    return Path, sle


@app.cell
def _():
    import tqdm
    return (tqdm,)


@app.cell
def _():
    import common.torch.load_torch_deps
    import torch
    return (torch,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    return


@app.cell
def _(Path):
    embedding_dirs = {
        'pano': Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v1/embeddings'),
        'osm': Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/v3_no_addresses/embeddings/')
    }

    return (embedding_dirs,)


@app.cell
def _(embedding_dirs, sle):
    osm_json = sle.load_all_jsonl_from_folder(embedding_dirs["osm"])

    return (osm_json,)


@app.cell
def _(embedding_dirs, sle):
    _osm_sentence_json = sle.load_all_jsonl_from_folder(embedding_dirs["osm"]/ '..' / 'sentences')
    osm_sentences, _ = sle.make_sentence_dict_from_json(_osm_sentence_json)
    return


@app.cell
def _(osm_json, sle):
    osm_embeddings = sle.make_embedding_dict_from_json(osm_json)
    return (osm_embeddings,)


@app.function
def create_embedding_response(custom_id, embedding):
  """Create an OpenAI batch API response in the expected format"""
  return {
      "id": f"batch_req_{custom_id[:16]}",
      "custom_id": custom_id,
      "response": {
          "status_code": 200,
          "request_id": "test_request",
          "body": {
              "object": "list",
              "data": [{
                  "object": "embedding",
                  "index": 0,
                  "embedding": embedding
              }],
              "model": "text-embedding-3-small",
              "usage": {"prompt_tokens": 1, "total_tokens": 1}
          }
      },
      "error": None
  }


@app.cell
def _(osm_embeddings):
    all_osm_keys = list(osm_embeddings.keys())
    return (all_osm_keys,)


@app.cell
def _(all_osm_keys, osm_embeddings):
    osm_embeddings[all_osm_keys[0]]
    return


@app.cell
def _(all_osm_keys):
    all_osm_keys[0]
    return


@app.cell
def _(osm_embeddings, torch):
    all_embeddings = torch.stack([torch.tensor(v) for v in osm_embeddings.values()]).to('cuda')

    return (all_embeddings,)


@app.cell
def _(all_embeddings):
    all_embeddings.shape
    return


@app.cell
def _(Path, all_embeddings, all_osm_keys, json, torch, tqdm):
    _batch_size=1024


    target_sim = [1.0, 0.99, 0.96, 0.92, 0.85, 0.8, 0.7, 0.6, 0.4, 0.2]

    for t in target_sim:

        _all_values = []
        _all_idxs = []
    
        _threshold = t
    
        for _i in range(0, all_embeddings.shape[0], _batch_size):
            _working_set = all_embeddings[_i:_i+_batch_size]
            _set_similarities = all_embeddings @ _working_set.T
    
            _set_similarities[_set_similarities > _threshold] = -1
            _max = torch.max(_set_similarities, 0)
    
            _all_values.append(_max.values)
            _all_idxs.append(_max.indices)
    
        all_values = torch.cat(_all_values)
        all_idxs = torch.cat(_all_idxs)

        target_embeddings = all_embeddings[all_idxs]
    
        sims = torch.einsum('ij,ij->i', all_embeddings, target_embeddings)

    
        lamda = ((t - sims) / (1 - sims)).unsqueeze(-1)

        new_embeddings = lamda * all_embeddings + (1 - lamda) * target_embeddings

        out_file = Path(f'/tmp/target_similarity_{t:0.2f}_v3_no_addresses/embeddings/file')
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with out_file.open('w') as out:
            for i in tqdm.tqdm(range(len(all_osm_keys)), desc=f'Target Sim: {t:02f}'):
                out.write(json.dumps(create_embedding_response(all_osm_keys[i], new_embeddings[i, :].cpu().tolist())))
                out.write("\n")
    return (new_embeddings,)


@app.cell
def _():
    return


@app.cell
def _():
    import json
    return (json,)


@app.cell
def _(new_embeddings):
    new_embeddings.dtype
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
