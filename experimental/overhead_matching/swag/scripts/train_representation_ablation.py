"""Fixed-embedding semantic representation ablations on simple_v1_v5's training loop.

Frozen-embedding alternatives to the structured-tag correspondence model. Each
side of a pair is a 768-dimensional text-embedding-005 vector of one text. The
panorama text is fixed per prepared root by `prepare --pano-text`: the VLM
landmark sentence (descriptions) or the landmark's sorted literal key=value
string (tag_strings). The OSM text is chosen per `train --variant`: generated
sentences (descriptions) or sorted literal key=value text (tag_strings). The
three paper variants are sentence/sentence, sentence/KV, and KV/KV; KV/KV needs
its own prepared root and its own panorama embedding run because the panorama
texts differ. FixedEmbeddingClassifier consumes [pano, osm, pano * osm] with no
learned tag encoder or engineered cross features; its head is the structured
model's Linear -> BatchNorm -> ReLU -> Dropout -> Linear, and the original
classifier.* checkpoint keys, including BatchNorm statistics, stay compatible.

Comparing any variant against the structured model also changes the encoder,
cross features, and parameter count; it is not a format-only comparison.
Comparing variants against each other changes only the named text.

Workflow (all locations are explicit arguments; there is no default experiment
directory or cloud project):

  bazel run //experimental/overhead_matching/swag/scripts:train_representation_ablation -- prepare \
    --root $EXPERIMENT_DIR --config $BASELINE_CONFIG \
    --osm-embeddings-dir $OSM_EMBEDDINGS_DIR \
    --osm-source-dir $OSM_SOURCE_DIR --pano-source-dir $PANO_SOURCE_DIR \
    --pano-text descriptions
  bazel run ... -- embed --root $EXPERIMENT_DIR
  bazel run ... -- check --root $EXPERIMENT_DIR --config $BASELINE_CONFIG \
    --osm-embeddings-dir $OSM_EMBEDDINGS_DIR [--checkpoint best_model.pt]
  bazel run ... -- train --root $EXPERIMENT_DIR --config $BASELINE_CONFIG \
    --osm-embeddings-dir $OSM_EMBEDDINGS_DIR --variant descriptions

Inputs. BASELINE_CONFIG is the baseline correspondence training YAML: label data
directory, train and validation cities, optimizer settings, classifier
width/dropout, and seed. OSM_EMBEDDINGS_DIR holds aligned keys.json,
descriptions.npy and tag_strings.npy. OSM_SOURCE_DIR holds the manifest.json
whose sample tags that population was generated from. PANO_SOURCE_DIR holds
<city>/embeddings/embeddings.pkl with the original panorama landmark tags and
descriptions.

prepare verifies labels, landmark ordering, and difficulty selection against
the existing correspondence parser, checks that both OSM arrays have one row per
keys.json entry, then writes pair indices, panorama texts,
input hashes, and a frozen copy of the baseline configuration. Training pairs
carry the OSM tags as the label parser reads them back from the prompt, which
strips whitespace the map data keeps, so a parsed bundle can hash differently
from its entity; prepare resolves such bundles by applying the same stripping
to the population's tags and requires exactly one match. A bundle that still
does not resolve stops preparation: that means the parser is not inverting the
prompt format (the pre-#744 parser truncated values at "; " and newlines) and
must be fixed rather than patched around. prepared.json counts the resolved
pairs per city.

embed calls Vertex AI through the repository embedding helper with your
configured credentials/project/location, preserving text-embedding-005,
SEMANTIC_SIMILARITY, 768 dimensions and auto_truncate=False, with bounded
requests and resumable output. train uses CUDA and the existing correspondence
training/evaluation functions, writes under EXPERIMENT_DIR/<variant>, refuses to
overwrite an existing run, and selects the best checkpoint by validation ROC
AUC. Repeat train with --variant tag_strings for the other OSM text. For KV/KV,
prepare a second root with --pano-text tag_strings, run embed there, and train
with --variant tag_strings. check --checkpoint FILE validates an archived
checkpoint without training.

Panorama tag strings use the label parser's dictionary form (repeated keys
collapse to the last value), which is also what the structured baseline encodes
at export time. Evaluation-side tag strings must be built the same way; the
archived KV/KV evaluation inputs instead kept repeated keys for 48 landmark rows.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import pickle
from msgspec.structs import replace

import common.torch.load_torch_deps
import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
    CorrespondenceBatch,
    parse_jsonl_line,
    parse_prompt_landmarks,
    parse_tag_string,
    load_pairs_from_directory,
)
from experimental.overhead_matching.swag.model.semantic_landmark_utils import (
    custom_id_from_props,
    _TAGS_TO_KEEP_SET,
    _TAGS_TO_KEEP_PREFIXES,
)
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    FixedEmbeddingClassifier,
)
from experimental.overhead_matching.swag.scripts.correspondence_configs import (
    load_config,
    save_config,
)
from experimental.overhead_matching.swag.scripts.train_landmark_correspondence import (
    setup_reproducibility,
    create_lr_scheduler,
    train_epoch,
    evaluate,
    compute_metrics,
)


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def tags_text(tags):
    """Serialize a tag bundle exactly as the OSM ``tag_strings`` embeddings were.

    Accepts a dict or (key, value) pairs; pairs collapse through dict() so repeated
    keys keep the last value, as the label parser and the baseline exporter do.
    Evaluation inputs built elsewhere must use this same form.
    """
    text = ", ".join(f"{k}={v}" for k, v in sorted(dict(tags).items()))
    assert text, "Empty tag bundle"
    return text


def parser_whitespace(tags):
    """Apply the label parser's whitespace handling to original tag values."""
    return {
        str(k).strip(): "\n".join(line.strip() for line in str(v).split("\n")).strip()
        for k, v in tags.items()
    }


def prepare(args):
    root, base_config = args.root, args.config
    osm, osm_base, pano = args.osm_embeddings_dir, args.osm_source_dir, args.pano_source_dir
    assert not (root / "prepared.json").exists(), "Already prepared"
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(base_config)
    osm_keys = json.loads((osm / "keys.json").read_text())
    for variant in ["descriptions", "tag_strings"]:
        rows = np.load(osm / f"{variant}.npy", mmap_mode="r").shape[0]
        assert rows == len(osm_keys), (variant, rows, len(osm_keys), "keys.json rows")
    osm_rows = {key: i for i, key in enumerate(osm_keys)}
    # The parser strips whitespace that the map data keeps, so a parsed bundle may
    # hash differently from its entity. Resolve those through the same stripping;
    # anything else (e.g. a truncating parser) must fail here, not be patched over.
    population = json.loads((osm_base / "manifest.json").read_text())["sample"]
    assert set(population) == set(osm_rows), "manifest.json does not back keys.json"
    normalized = {}
    for key, entry in population.items():
        normalized.setdefault(custom_id_from_props(parser_whitespace(entry["tags"])), []).append(key)
    del population
    texts, text_rows, records, stats = [], {}, [], {}
    save_config(config, root / "baseline_config.yaml")
    for city in [config.train_city, config.val_city]:
        with (pano / city / "embeddings/embeddings.pkl").open("rb") as f:
            pano_data = pickle.load(f)["panoramas"]
        panos = {}
        for raw_id, data in pano_data.items():
            clean = raw_id.split(",")[0]
            assert clean not in panos
            filtered = []
            for i, lm in enumerate(data["landmarks"]):
                raw = [lm.get("primary_tag", {})] + lm.get("additional_tags", [])
                tags = [
                    (t["key"], t["value"])
                    for t in raw
                    if t.get("key")
                    and t.get("value")
                    and (
                        t["key"] in _TAGS_TO_KEEP_SET
                        or any(t["key"].startswith(p) for p in _TAGS_TO_KEEP_PREFIXES)
                    )
                ]
                if tags:
                    # Exactly mirror the label generator's formatting and parser.
                    tags = parse_tag_string("; ".join(f"{k}={v}" for k, v in tags))
                    filtered.append((tags, lm.get("description"), i))
            panos[clean] = filtered
        del pano_data
        pano_idx, osm_idx, labels, difficulties, pano_ids = [], [], [], [], []
        seen_records, whitespace_resolved = set(), 0
        files = list((config.data_dir / city).rglob("predictions.jsonl"))
        assert files
        for file in files:
            with file.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    baseline_pairs = parse_jsonl_line(row)
                    if not baseline_pairs:
                        continue
                    pid = row["key"]
                    set1, set2 = parse_prompt_landmarks(
                        row["request"]["contents"][0]["parts"][0]["text"]
                    )
                    source = panos[pid]
                    assert [x[0] for x in source] == set1, (
                        city,
                        pid,
                        "Pano annotation order/tags changed",
                    )
                    response = json.loads(
                        row["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    )
                    events = []
                    for match in response.get("matches", []):
                        a = match.get("set_1_id")
                        if a is None or not 0 <= a < len(set1):
                            continue
                        for b in match.get("set_2_matches", []):
                            if 0 <= b < len(set2):
                                events.append((a, b, 1.0, "positive"))
                        for neg in match.get("negatives", []):
                            b, difficulty = neg.get("set_2_id"), neg.get("difficulty")
                            if (
                                b is not None
                                and 0 <= b < len(set2)
                                and difficulty is not None
                            ):
                                events.append((a, b, 0.0, difficulty))
                    assert len(events) == len(baseline_pairs)
                    for (a, b, label, difficulty), pair in zip(events, baseline_pairs):
                        assert (set1[a], set2[b], label, difficulty) == (
                            pair.pano_tags,
                            pair.osm_tags,
                            pair.label,
                            pair.difficulty,
                        )
                        if difficulty not in config.include_difficulties:
                            continue
                        if args.pano_text == "descriptions":
                            text = source[a][1]
                            assert isinstance(text, str) and text.strip()
                        else:
                            text = tags_text(set1[a])
                        if text not in text_rows:
                            text_rows[text] = len(texts)
                            texts.append(text)
                        pr = text_rows[text]
                        record_key = (pid, source[a][2])
                        if record_key not in seen_records:
                            records.append(
                                dict(
                                    city=city,
                                    pano_id=pid,
                                    landmark_idx=source[a][2],
                                    embedding_row=pr,
                                )
                            )
                            seen_records.add(record_key)
                        key = custom_id_from_props(set2[b])
                        if key not in osm_rows:
                            candidates = normalized.get(
                                custom_id_from_props(parser_whitespace(set2[b])), []
                            )
                            assert len(candidates) == 1, (
                                city,
                                pid,
                                set2[b],
                                f"{len(candidates)} OSM entities match this parsed bundle",
                            )
                            key = candidates[0]
                            whitespace_resolved += 1
                        pano_idx.append(pr)
                        osm_idx.append(osm_rows[key])
                        labels.append(label)
                        difficulties.append(difficulty)
                        pano_ids.append(pid)
        reference = [
            p
            for p in load_pairs_from_directory(config.data_dir / city)
            if p.difficulty in config.include_difficulties
        ]
        assert len(reference) == len(labels)
        assert [(p.label, p.difficulty, p.pano_id) for p in reference] == list(
            zip(labels, difficulties, pano_ids)
        )
        np.savez(
            root / f"{city}_pairs.npz",
            pano_idx=np.array(pano_idx, dtype=np.int64),
            osm_idx=np.array(osm_idx, dtype=np.int64),
            labels=np.array(labels, dtype=np.float32),
            difficulties=np.array(difficulties),
            pano_ids=np.array(pano_ids),
        )
        stats[city] = dict(
            pairs=len(labels),
            difficulties=dict(Counter(difficulties)),
            panorama_landmarks=len(seen_records),
            whitespace_resolved_osm_pairs=whitespace_resolved,
            label_files=[str(p) for p in files],
            label_file_sha256=[
                hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            ],
        )
        print(city, stats[city], flush=True)
    write_json(root / "pano_texts.json", texts)
    write_json(root / "pano_landmark_index.json", records)
    write_json(
        root / "prepared.json",
        dict(
            cities=stats,
            pano_text=args.pano_text,
            unique_pano_texts=len(texts),
            base_osm_rows=len(osm_keys),
            note="All baseline pairs retained in original order.",
        ),
    )
    print("Prepared", len(texts), args.pano_text, "pano texts", flush=True)


def batches(texts, start):
    """Bound requests by item count and UTF-8 bytes without truncating sentences."""
    while start < len(texts):
        end, size = start, 0
        while end < len(texts) and end - start < 250:
            length = len(texts[end].encode("utf-8"))
            if not 0 < length <= 18000:
                raise ValueError("Embedding text must contain 1..18000 UTF-8 bytes")
            if size + length > 18000:
                break
            size += length
            end += 1
        yield start, end
        start = end


def embed_inputs(args):
    from experimental.overhead_matching.swag.scripts.precompute_value_embeddings import (
        embed_texts_vertex,
    )

    root = args.root
    config = dict(
        model="text-embedding-005", output_dimensionality=768, auto_truncate=False
    )
    texts = json.loads((root / "pano_texts.json").read_text())
    path = root / "pano_embeddings.npy"
    matrix = np.lib.format.open_memmap(
        path,
        mode="r+" if path.exists() else "w+",
        dtype=np.float32,
        shape=(len(texts), 768),
    )
    checkpoint = root / "pano_embedding_progress.json"
    start = json.loads(checkpoint.read_text())["rows"] if checkpoint.exists() else 0
    iterator = iter(batches(texts, start))
    with ThreadPoolExecutor(max_workers=16) as pool:
        while True:
            bounds = [x for _ in range(16) if (x := next(iterator, None)) is not None]
            if not bounds:
                break
            results = pool.map(
                lambda ab: embed_texts_vertex(texts[ab[0] : ab[1]], **config), bounds
            )
            for (a, b), values in zip(bounds, results):
                assert values.shape == (b - a, 768) and np.isfinite(values).all()
                assert (np.linalg.norm(values, axis=1) > 0).all()
                matrix[a:b] = values
            matrix.flush()
            write_json(checkpoint, dict(rows=bounds[-1][1], total=len(texts)))
            print("Pano embeddings", bounds[-1][1], "/", len(texts), flush=True)
    write_json(
        root / "embedding_complete.json",
        dict(
            model=config["model"],
            dimensions=768,
            task_type="SEMANTIC_SIMILARITY",
            auto_truncate=False,
            pano_rows=len(texts),
        ),
    )


class FixedEmbeddingPairs(Dataset):
    def __init__(self, root, osm, city, variant):
        self.pairs = dict(np.load(root / f"{city}_pairs.npz"))
        self.pano = np.load(root / "pano_embeddings.npy", mmap_mode="r")
        self.osm = np.load(osm / f"{variant}.npy", mmap_mode="r")
        prepared = json.loads((root / "prepared.json").read_text())
        assert len(self.osm) == prepared["base_osm_rows"], (
            len(self.osm),
            prepared["base_osm_rows"],
            "OSM array is not the prepared population",
        )
        assert self.pairs["osm_idx"].max() < len(self.osm), (
            "pairs reference rows beyond the OSM population (root prepared with "
            "supplemental rows); re-run prepare"
        )

    def __len__(self):
        return len(self.pairs["labels"])

    def __getitem__(self, i):
        a, b = self.pairs["pano_idx"][i], self.pairs["osm_idx"][i]
        return self.pano[a], self.osm[b], self.pairs["labels"][i]


def collate_fixed(batch):
    pano, osm, labels = zip(*batch)
    empty = torch.empty(len(batch), 0)
    return CorrespondenceBatch(
        empty,
        torch.from_numpy(np.stack(pano)),
        empty,
        empty,
        torch.from_numpy(np.stack(osm)),
        empty,
        empty,
        torch.tensor(labels, dtype=torch.float32),
    )


def check(args):
    root, base_config, osm = args.root, args.config, args.osm_embeddings_dir
    original = load_config(base_config)
    relocated = replace(original, output_dir=root / "check_only")
    assert all(
        getattr(original, k) == getattr(relocated, k)
        for k in original.__struct_fields__
        if k != "output_dir"
    )
    model = (
        FixedEmbeddingClassifier.load_checkpoint(args.checkpoint)
        if args.checkpoint
        else FixedEmbeddingClassifier(
            original.classifier.mlp_hidden_dim, original.classifier.dropout
        ).eval()
    )
    a, b = torch.randn(4, 768), torch.randn(4, 768)
    batch = collate_fixed([(x.numpy(), y.numpy(), 1.0) for x, y in zip(a, b)])
    from experimental.overhead_matching.swag.scripts.train_landmark_correspondence import (
        _forward_batch,
    )

    assert _forward_batch(model, batch).shape == (4,)
    torch.testing.assert_close(
        _forward_batch(model, batch),
        model.classifier(torch.cat([a, b, a * b], -1)).squeeze(-1),
    )
    assert not any("encoder" in k for k in model.state_dict())
    if (root / "embedding_complete.json").exists():
        for variant in ["descriptions", "tag_strings"]:
            for city in [original.train_city, original.val_city]:
                ds = FixedEmbeddingPairs(root, osm, city, variant)
                for i in [0, len(ds) // 2, len(ds) - 1]:
                    p, o, label = ds[i]
                    assert (
                        p.shape == o.shape == (768,)
                        and np.isfinite(p).all()
                        and np.isfinite(o).all()
                    )
                    assert label in [0.0, 1.0]
    print("Classifier and input checks passed", flush=True)


def train(args):
    root, base_config, osm = args.root, args.config, args.osm_embeddings_dir
    variant = args.variant
    assert (root / "embedding_complete.json").exists()
    pano_text = json.loads((root / "prepared.json").read_text())["pano_text"]
    config = load_config(base_config)
    reference = load_config(root / "baseline_config.yaml")
    assert config == reference, "Baseline configuration changed after preparation"
    config = replace(config, output_dir=root / variant)
    output = config.output_dir
    output.mkdir(exist_ok=False)
    save_config(config, output / "config.yaml")
    setup_reproducibility(config.seed)
    assert torch.cuda.is_available(), "Use GPU as in baseline"
    device = torch.device("cuda")
    train_ds = FixedEmbeddingPairs(root, osm, config.train_city, variant)
    val_ds = FixedEmbeddingPairs(root, osm, config.val_city, variant)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fixed,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fixed,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    model = FixedEmbeddingClassifier(
        config.classifier.mlp_hidden_dim, config.classifier.dropout
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    write_json(
        output / "ablation.json",
        dict(
            variant=variant,
            pano_input=pano_text,
            osm_input=variant,
            dimensions=768,
            encoder="none",
            cross_features="none",
            model_class="FixedEmbeddingClassifier",
            input_features=2304,
            trainable_parameters=parameters,
            initial_state_sha256=hashlib.sha256(
                b"".join(
                    t.detach().cpu().numpy().tobytes()
                    for t in model.state_dict().values()
                )
            ).hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            base_config=str(base_config),
            pano_embeddings=str(root / "pano_embeddings.npy"),
            osm_embeddings=str(osm / f"{variant}.npy"),
            config_note="encoder and text_embeddings_path fields retained for baseline provenance but unused; ablation.json specifies actual inputs",
        ),
    )
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps = len(train_loader) * config.num_epochs
    warmup = int(steps * config.warmup_fraction)
    scheduler = create_lr_scheduler(
        optimizer, warmup, steps, cosine=config.cosine_schedule
    )
    scaler = GradScaler() if config.use_amp else None
    writer = SummaryWriter(output / "tensorboard")
    best_auc, history = -math.inf, []
    print(
        "Run",
        variant,
        "pairs",
        len(train_ds),
        len(val_ds),
        "parameters",
        parameters,
        "steps",
        steps,
        "warmup",
        warmup,
        flush=True,
    )
    for epoch in range(config.num_epochs):
        train_loss, labels, probs = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            config.gradient_clip_norm,
        )
        metrics = compute_metrics(labels, probs)
        val_loss, val_metrics = evaluate(model, val_loader, device)
        record = dict(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            train=metrics,
            val=val_metrics,
        )
        history.append(record)
        write_json(output / "metrics.json", history)
        print(json.dumps(record), flush=True)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        for split, values in [("train", metrics), ("val", val_metrics)]:
            for name, value in values.items():
                writer.add_scalar(f"{split}/{name}", value, epoch)
        if not math.isnan(val_metrics["auc_roc"]) and val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), output / "best_model.pt")
            write_json(output / "best_metrics.json", record)
        torch.save(
            dict(
                epoch=epoch + 1,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                val_metrics=val_metrics,
            ),
            output / f"checkpoint_epoch_{epoch + 1}.pt",
        )
    writer.close()
    assert best_auc > -math.inf
    write_json(
        output / "complete.json", dict(epochs=config.num_epochs, best_val_auc=best_auc)
    )
    print("COMPLETE", variant, "best val AUC", best_auc, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "embed", "check", "train"):
        command = commands.add_parser(action)
        command.add_argument(
            "--root",
            type=Path,
            required=True,
            help="Prepared inputs and variant training outputs",
        )
        if action != "embed":
            command.add_argument(
                "--config",
                type=Path,
                required=True,
                help="Baseline correspondence training YAML",
            )
            command.add_argument(
                "--osm-embeddings-dir",
                type=Path,
                required=True,
                help="keys.json, descriptions.npy and tag_strings.npy",
            )
        if action == "prepare":
            command.add_argument(
                "--osm-source-dir",
                type=Path,
                required=True,
                help="manifest.json whose sample tags back keys.json",
            )
            command.add_argument(
                "--pano-source-dir",
                type=Path,
                required=True,
                help="City directories containing embeddings/embeddings.pkl",
            )
            command.add_argument(
                "--pano-text",
                choices=["descriptions", "tag_strings"],
                required=True,
                help="Panorama landmark text to embed: VLM sentence or sorted key=value",
            )
        if action == "check":
            command.add_argument(
                "--checkpoint",
                type=Path,
                help="Also validate loading an archived best_model.pt",
            )
        if action == "train":
            command.add_argument(
                "--variant",
                choices=["descriptions", "tag_strings"],
                required=True,
                help="OSM text embeddings to pair with the prepared panorama text",
            )
    args = parser.parse_args(argv)
    {"prepare": prepare, "embed": embed_inputs, "check": check, "train": train}[
        args.action
    ](args)


if __name__ == "__main__":
    main()
