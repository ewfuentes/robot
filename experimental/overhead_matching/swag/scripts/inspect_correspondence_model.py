import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import common.torch.load_torch_deps  # noqa: F401

    import json
    import math
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader

    import experimental.overhead_matching.swag.data.vigor_dataset as vd
    from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
        LandmarkCorrespondenceDataset,
        collate_correspondence,
        compute_cross_features,
        encode_tag_bundle,
        load_pairs_from_directory,
        load_text_embeddings,
        parse_prompt_landmarks,
    )
    from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
        CorrespondenceClassifier,
        CorrespondenceClassifierConfig,
        TagBundleEncoderConfig,
    )

    return (
        CorrespondenceClassifier,
        CorrespondenceClassifierConfig,
        DataLoader,
        F,
        Image,
        LandmarkCorrespondenceDataset,
        Path,
        TagBundleEncoderConfig,
        collate_correspondence,
        compute_cross_features,
        encode_tag_bundle,
        go,
        json,
        load_pairs_from_directory,
        load_text_embeddings,
        np,
        parse_prompt_landmarks,
        pd,
        px,
        torch,
        vd,
    )


@app.cell
def _(Path):
    MODEL_PATH = Path(
        "/data/overhead_matching/training_outputs/landmark_correspondence/v1_all/best_model.pt"
    )
    TEXT_EMB_PATH = Path(
        "/data/overhead_matching/datasets/landmark_correspondence/"
        "chicago_seattle_neg_v3_full/text_embeddings.pkl"
    )
    DATA_DIR = Path(
        "/data/overhead_matching/datasets/landmark_correspondence/"
        "chicago_seattle_neg_v3_full/Seattle"
    )
    VIGOR_PATH = Path("/data/overhead_matching/datasets/VIGOR/Seattle")
    return DATA_DIR, MODEL_PATH, TEXT_EMB_PATH, VIGOR_PATH


@app.cell
def _(
    CorrespondenceClassifier,
    CorrespondenceClassifierConfig,
    MODEL_PATH,
    TEXT_EMB_PATH,
    TagBundleEncoderConfig,
    load_text_embeddings,
    mo,
    torch,
):
    text_embeddings = load_text_embeddings(TEXT_EMB_PATH)
    encoder_config = TagBundleEncoderConfig(text_input_dim=768, text_proj_dim=128)
    classifier_config = CorrespondenceClassifierConfig(encoder=encoder_config)
    _model = CorrespondenceClassifier(classifier_config)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model.to(device).eval()
    mo.md(
        f"**Model loaded** ({sum(p.numel() for p in model.parameters()):,} params) "
        f"on `{device}` | **{len(text_embeddings):,}** text embeddings"
    )
    return device, model, text_embeddings


@app.cell
def _(VIGOR_PATH, mo, np, pd, vd):
    vigor_config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version="v4_202001",
    )
    vigor_dataset = vd.VigorDataset(dataset_path=VIGOR_PATH, config=vigor_config)

    _lm = vigor_dataset._landmark_metadata
    osm_landmarks_df = pd.DataFrame(
        {
            "landmark_idx": range(len(_lm)),
            "lat": [
                g.centroid.y if g is not None else np.nan for g in _lm.geometry
            ],
            "lon": [
                g.centroid.x if g is not None else np.nan for g in _lm.geometry
            ],
            "tags": [dict(_lm.iloc[i]["pruned_props"]) for i in range(len(_lm))],
            "tags_str": [
                "; ".join(f"{k}={v}" for k, v in sorted(_lm.iloc[i]["pruned_props"]))
                for i in range(len(_lm))
            ],
        }
    )
    osm_landmarks_df = osm_landmarks_df[osm_landmarks_df["tags_str"] != ""].reset_index(
        drop=True
    )

    mo.md(
        f"**VIGOR Seattle**: {len(vigor_dataset._panorama_metadata):,} panoramas, "
        f"{len(osm_landmarks_df):,} OSM landmarks with tags"
    )
    return osm_landmarks_df, vigor_dataset


@app.cell
def _(DATA_DIR, json, load_pairs_from_directory, mo, parse_prompt_landmarks):
    val_pairs = load_pairs_from_directory(DATA_DIR)

    # Build pano_id -> pano landmarks lookup from JSONL
    pano_id_to_pano_landmarks = {}
    _jsonl_files = list(DATA_DIR.rglob("predictions.jsonl")) or list(
        DATA_DIR.rglob("*.jsonl")
    )
    _skipped = 0
    for _jf in _jsonl_files:
        with open(_jf) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _data = json.loads(_line)
                    _pid = _data["key"]
                    _prompt = _data["request"]["contents"][0]["parts"][0]["text"]
                    _set1, _set2 = parse_prompt_landmarks(_prompt)
                    pano_id_to_pano_landmarks[_pid] = _set1
                except (json.JSONDecodeError, KeyError, ValueError):
                    _skipped += 1
                    continue
    if _skipped:
        print(f"WARNING: Skipped {_skipped} unparseable JSONL lines")

    pano_ids_in_val = sorted(pano_id_to_pano_landmarks.keys())

    mo.md(
        f"**Validation**: {len(val_pairs):,} pairs from {len(pano_ids_in_val):,} panoramas"
    )
    return pano_id_to_pano_landmarks, pano_ids_in_val, val_pairs


@app.cell
def _(F, collate_correspondence, device, model, torch):
    import functools as _functools

    _collate = _functools.partial(collate_correspondence, text_input_dim=768)

    def run_model_on_batch(batch, _model=model, _device=device):
        """Run the correspondence model on a CorrespondenceBatch."""
        batch = batch.to(_device)
        with torch.no_grad():
            logits = _model(
                pano_key_indices=batch.pano_key_indices,
                pano_text_embeddings=batch.pano_text_embeddings,
                pano_tag_mask=batch.pano_tag_mask,
                osm_key_indices=batch.osm_key_indices,
                osm_text_embeddings=batch.osm_text_embeddings,
                osm_tag_mask=batch.osm_tag_mask,
                cross_features=batch.cross_features,
            ).squeeze(-1)
        return logits

    def score_pairs(samples_list):
        """Run model on a list of sample dicts. Returns (logits, probs, losses)."""
        all_logits, all_probs, all_losses = [], [], []
        _bs = 512
        for _start in range(0, len(samples_list), _bs):
            _batch = _collate(samples_list[_start : _start + _bs])
            _logits = run_model_on_batch(_batch)
            _labels = _batch.labels.to(_logits.device)
            _loss = F.binary_cross_entropy_with_logits(
                _logits, _labels, reduction="none"
            )
            all_logits.extend(_logits.cpu().tolist())
            all_probs.extend(torch.sigmoid(_logits).cpu().tolist())
            all_losses.extend(_loss.cpu().tolist())
        return all_logits, all_probs, all_losses

    return (run_model_on_batch,)


@app.cell
def _(
    DataLoader,
    F,
    LandmarkCorrespondenceDataset,
    collate_correspondence,
    mo,
    pd,
    run_model_on_batch,
    text_embeddings,
    torch,
    val_pairs,
):
    mo.output.replace(mo.md("Computing validation losses... (this takes ~1 min)"))

    _include = ("positive", "hard", "easy")
    _filtered = [p for p in val_pairs if p.difficulty in _include]

    # Build metadata
    _meta = []
    for _i, _p in enumerate(_filtered):
        _meta.append(
            {
                "pano_tags": "; ".join(
                    f"{k}={v}" for k, v in sorted(_p.pano_tags.items())
                ),
                "osm_tags": "; ".join(
                    f"{k}={v}" for k, v in sorted(_p.osm_tags.items())
                ),
                "label": _p.label,
                "difficulty": _p.difficulty,
                "uniqueness": _p.uniqueness_score,
                "pano_id": _p.pano_id,
            }
        )
    loss_df = pd.DataFrame(_meta)

    import functools as _functools_inner

    # Run model
    _dataset = LandmarkCorrespondenceDataset(
        _filtered, text_embeddings, 768, _include,
        allow_missing_text_embeddings=True,
    )
    _loader = DataLoader(
        _dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=_functools_inner.partial(collate_correspondence, text_input_dim=768),
        num_workers=4,
        pin_memory=True,
    )

    _all_probs = []
    _all_losses = []
    with torch.no_grad():
        for _batch in _loader:
            _logits = run_model_on_batch(_batch)
            _labels = _batch.labels.to(_logits.device)
            _loss = F.binary_cross_entropy_with_logits(
                _logits, _labels, reduction="none"
            )
            _all_probs.extend(torch.sigmoid(_logits).cpu().tolist())
            _all_losses.extend(_loss.cpu().tolist())

    loss_df["prob"] = _all_probs
    loss_df["loss"] = _all_losses
    loss_df = loss_df.sort_values("loss", ascending=False).reset_index(drop=True)

    mo.output.replace(
        mo.md(f"**Validation losses computed** for {len(loss_df):,} pairs")
    )
    return (loss_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Validation Loss Analysis
    """)
    return


@app.cell
def _(loss_df, mo, px):
    _fig = px.histogram(
        loss_df,
        x="loss",
        color="difficulty",
        nbins=100,
        title="Per-sample BCE loss distribution",
        barmode="overlay",
        opacity=0.6,
    )
    _fig.update_layout(height=350)

    _worst = loss_df.head(25)[
        ["pano_tags", "osm_tags", "label", "difficulty", "prob", "loss", "pano_id"]
    ]
    _best = loss_df.tail(25).iloc[::-1][
        ["pano_tags", "osm_tags", "label", "difficulty", "prob", "loss", "pano_id"]
    ]

    mo.vstack(
        [
            _fig,
            mo.md("### Worst predictions (highest loss)"),
            mo.ui.table(_worst, label="Worst 25"),
            mo.md("### Best predictions (lowest loss)"),
            mo.ui.table(_best, label="Best 25"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Panorama Explorer
    """)
    return


@app.cell
def _(mo, pano_ids_in_val):
    pano_selector = mo.ui.dropdown(
        options=pano_ids_in_val, label="Select panorama", full_width=False
    )
    pano_selector
    return (pano_selector,)


@app.cell
def _(Image, mo, pano_id_to_pano_landmarks, pano_selector, vigor_dataset):
    mo.stop(pano_selector.value is None, mo.md("*Select a panorama above*"))

    _pid = pano_selector.value
    _pano_row = vigor_dataset._panorama_metadata[
        vigor_dataset._panorama_metadata.pano_id == _pid
    ]
    mo.stop(len(_pano_row) == 0, mo.md(f"Panorama `{_pid}` not found in VIGOR dataset"))
    _pano_row = _pano_row.iloc[0]

    # Load panorama image
    _img = Image.open(_pano_row.path)
    _img = _img.resize((1024, 512))

    # Get pano landmarks
    _pano_lms = pano_id_to_pano_landmarks.get(_pid, [])
    _lm_strs = [
        "; ".join(f"{k}={v}" for k, v in sorted(lm.items())) for lm in _pano_lms
    ]
    _lm_options = {s: i for i, s in enumerate(_lm_strs)} if _lm_strs else {"(no landmarks)": -1}

    landmark_selector = mo.ui.dropdown(
        options=_lm_options, label="Select landmark", full_width=True
    )

    selected_pano_landmarks = _pano_lms

    mo.vstack(
        [
            mo.md(f"**Panorama**: `{_pid}` | lat={_pano_row.lat:.6f}, lon={_pano_row.lon:.6f}"),
            mo.image(_img),
            mo.md(f"**{len(_pano_lms)} pano landmarks** detected:"),
            landmark_selector,
        ]
    )
    return landmark_selector, selected_pano_landmarks


@app.cell
def _(
    collate_correspondence,
    compute_cross_features,
    encode_tag_bundle,
    landmark_selector,
    mo,
    np,
    osm_landmarks_df,
    run_model_on_batch,
    selected_pano_landmarks,
    text_embeddings,
    torch,
):
    mo.stop(
        landmark_selector.value is None or landmark_selector.value == -1,
        mo.md("*Select a landmark above*"),
    )

    _lm_idx = landmark_selector.value
    _pano_tags = selected_pano_landmarks[_lm_idx]
    _pano_tags_str = "; ".join(f"{k}={v}" for k, v in sorted(_pano_tags.items()))

    mo.output.replace(
        mo.md(f"Scoring `{_pano_tags_str}` against {len(osm_landmarks_df):,} OSM landmarks...")
    )

    import functools as _functools_inspect

    # Encode pano side once
    _pano_encoded = encode_tag_bundle(
        _pano_tags, text_embeddings, 768, allow_missing_text_embeddings=True,
    )

    # Build samples for all OSM landmarks
    _samples = []
    for _i, _row in osm_landmarks_df.iterrows():
        _osm_tags = _row["tags"]
        _osm_encoded = encode_tag_bundle(
            _osm_tags, text_embeddings, 768, allow_missing_text_embeddings=True,
        )
        _cross = compute_cross_features(_pano_tags, _osm_tags, text_embeddings)
        _samples.append(
            {
                "pano": _pano_encoded,
                "osm": _osm_encoded,
                "cross_features": _cross,
                "label": 0.0,
                "difficulty": "query",
            }
        )

    # Batch model inference
    _collate_inspect = _functools_inspect.partial(
        collate_correspondence, text_input_dim=768,
    )
    _all_logits = []
    _bs = 512
    with torch.no_grad():
        for _start in range(0, len(_samples), _bs):
            _batch = _collate_inspect(_samples[_start : _start + _bs])
            _logits = run_model_on_batch(_batch)
            _all_logits.extend(_logits.cpu().tolist())

    # Build results
    scores_df = osm_landmarks_df.copy()
    scores_df["logit"] = _all_logits
    scores_df["prob"] = 1.0 / (1.0 + np.exp(-scores_df["logit"]))
    scores_df = scores_df.sort_values("logit", ascending=False).reset_index(drop=True)

    selected_pano_tags_str = _pano_tags_str

    mo.output.replace(mo.md(f"Scored **{len(scores_df):,}** OSM landmarks"))
    return scores_df, selected_pano_tags_str


@app.cell
def _(mo, scores_df, selected_pano_tags_str):
    _top = scores_df[["tags_str", "prob", "logit", "lat", "lon"]].copy()
    # _top = scores_df.head(30)[["tags_str", "prob", "logit", "lat", "lon"]].copy()
    _top.columns = ["OSM tags", "P(match)", "logit", "lat", "lon"]

    mo.vstack(
        [
            mo.md(f"### Top 30 matches for: `{selected_pano_tags_str}`"),
            mo.ui.table(_top, label="Top matches"),
        ]
    )
    return


@app.cell
def _(go, mo, np, scores_df, selected_pano_tags_str):
    _df = scores_df[scores_df["lat"].notna()].copy()

    # Only plot top 2000 by logit to keep output size manageable
    _df = _df.head(2000)
    _df["logit_clamp"] = _df["logit"].clip(-5, 5)

    _fig = go.Figure()

    # Background: lower-scoring landmarks
    _bg = _df[_df["prob"] < 0.5]
    _fig.add_trace(
        go.Scattermap(
            lat=_bg["lat"],
            lon=_bg["lon"],
            mode="markers",
            marker=dict(
                size=4,
                color=_bg["logit_clamp"],
                colorscale="RdYlGn",
                cmin=-5,
                cmax=5,
                opacity=0.3,
            ),
            text=_bg["tags_str"],
            hovertemplate="%{text}<br>logit=%{marker.color:.2f}<extra></extra>",
            name="P < 0.5",
        )
    )

    # Foreground: high-probability matches
    _fg = _df[_df["prob"] >= 0.5]
    if len(_fg) > 0:
        _fig.add_trace(
            go.Scattermap(
                lat=_fg["lat"],
                lon=_fg["lon"],
                mode="markers",
                marker=dict(
                    size=np.clip(_fg["prob"] * 15, 6, 20).tolist(),
                    color=_fg["logit_clamp"],
                    colorscale="RdYlGn",
                    cmin=-5,
                    cmax=5,
                    opacity=0.9,
                    colorbar=dict(title="logit"),
                ),
                text=_fg["tags_str"],
                hovertemplate="%{text}<br>logit=%{marker.color:.2f}<extra></extra>",
                name="P >= 0.5",
            )
        )

    _center_lat = _df["lat"].mean()
    _center_lon = _df["lon"].mean()

    _fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=_center_lat, lon=_center_lon),
            zoom=10,
        ),
        height=700,
        title=f"Top 2000 OSM landmarks by logit for: {selected_pano_tags_str[:80]}",
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    mo.ui.plotly(_fig)
    return


if __name__ == "__main__":
    app.run()
