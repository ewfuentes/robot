import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import common.torch.load_torch_deps
    import torch
    from pathlib import Path
    import json
    import pickle

    from experimental.overhead_matching.swag.model.sentence_embedding_model import create_model_from_config
    from experimental.overhead_matching.swag.scripts.sentence_configs import load_config
    return Path, create_model_from_config, json, load_config, torch


@app.cell
def _(Path, json, load_config, torch):
    model_base = Path('/data/overhead_matching/models/sentence_embeddings/sentence_embeddings_addrname_debug_w_fix/')
    checkpoint = torch.load(model_base / 'best_model.pt')
    vocab = json.loads((model_base / 'tag_vocabs.json').read_text())
    train_config = load_config(model_base / 'config.json')
    return checkpoint, train_config, vocab


@app.cell
def _(checkpoint, create_model_from_config, train_config, vocab):

    model = create_model_from_config(
        train_config.model,
        tag_vocabs=vocab,
        classification_task_names=train_config.classification_tags,
        contrastive_task_names=train_config.contrastive_tags).cuda()
    model.load_state_dict(checkpoint)
    return (model,)


@app.cell
def _():
    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version="v4_202001",
    )
    dataset = vd.VigorDataset('/data/overhead_matching/datasets/VIGOR/Seattle', _config)

    return (dataset,)


@app.cell
def _(Path):
    import experimental.overhead_matching.swag.model.semantic_landmark_utils as slu
    sentence_path = Path('/data/overhead_matching/datasets/semantic_landmark_embeddings/v4_202001_no_addresses/sentences/')
    sentence_dict, _ = slu.make_sentence_dict_from_json(slu.load_all_jsonl_from_folder(sentence_path))
    return sentence_dict, slu


@app.cell
def _(dataset, model, sentence_dict, slu, torch, vocab):
    examined_ids = set()
    for _idx in range(10000):
        _pruned_props = dataset._landmark_metadata.pruned_props[_idx]
        _custom_id = slu.custom_id_from_props(_pruned_props)
        _sentence = sentence_dict[_custom_id]
        if _custom_id in examined_ids:
            continue
        examined_ids.add(_custom_id)
    
        _model_output = model([_sentence])
        _class_type = None
        _vocab = None
        for k, _ in _pruned_props:
            if k in _model_output["classification_logits"]:
                _class_type = k
                _vocab = {v: k for k, v in vocab[_class_type].items()}
                break
        if _class_type is None:
            print(f'{_pruned_props} No classification task found')
            continue
    
        _distribution = torch.nn.functional.softmax(_model_output["classification_logits"][_class_type]).detach().cpu()
        _values, _idxs = torch.sort(_distribution.squeeze(), descending=True)

        _preds = []
        for _i in range(3):
            _preds.append(f"{_vocab[_idxs[_i].item()]}: {_values[_i].item(): 0.3f}")
    
        print(f"{_pruned_props} {', '.join(_preds)} {_sentence=} ")

    return


@app.cell
def _(model_outpt):
    model_outpt
    return


@app.cell
def _():
    import experimental.overhead_matching.swag.data.osm_sentence_generator as osg
    return


@app.cell
def _(vocab):
    vocab.keys()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
