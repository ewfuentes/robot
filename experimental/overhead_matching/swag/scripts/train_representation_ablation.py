"""Two fixed-embedding correspondence ablations using simple_v1_v5's training loop."""

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


def prepare(args):
    root, base_config = args.root, args.config
    osm, osm_base = args.osm_embeddings_dir, args.osm_source_dir
    pano, supplement = args.pano_source_dir, args.supplement_dir
    assert not (root / "prepared.json").exists(), "Already prepared"
    root.mkdir(parents=True, exist_ok=True)
    config = load_config(base_config)
    osm_keys = json.loads((osm / "keys.json").read_text())
    osm_rows = {key: i for i, key in enumerate(osm_keys)}
    texts, text_rows, records, missing, stats = [], {}, [], {}, {}
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
                    filtered.append((tags, lm["description"], i))
            panos[clean] = filtered
        del pano_data
        pano_idx, osm_idx, labels, difficulties, pano_ids = [], [], [], [], []
        seen_records = set()
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
                        description = source[a][1]
                        assert isinstance(description, str) and description.strip()
                        if description not in text_rows:
                            text_rows[description] = len(texts)
                            texts.append(description)
                        pr = text_rows[description]
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
                            osm_rows[key] = len(osm_rows)
                            missing[key] = {"tags": set2[b], "datasets": [city]}
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
            label_files=[str(p) for p in files],
            label_file_sha256=[
                hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            ],
        )
        print(city, stats[city], flush=True)
    write_json(root / "pano_texts.json", texts)
    write_json(root / "pano_landmark_index.json", records)
    write_json(root / "osm_keys.json", list(osm_rows))
    if missing:
        supplement.mkdir(exist_ok=False)
        manifest = json.loads((osm_base / "manifest.json").read_text())
        manifest.update(sample=missing, unique_population=len(missing))
        write_json(supplement / "manifest.json", manifest)
        template = json.loads((osm_base / "requests.jsonl").open().readline())
        requests = []
        for key, entry in missing.items():
            request = json.loads(json.dumps(template))
            request["key"] = key
            request["request"]["contents"][0]["parts"][0]["text"] = (
                "Produce a short natural language description for this landmark: "
                + json.dumps(entry["tags"], sort_keys=True)
            )
            requests.append(request)
        (supplement / "requests.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in requests)
        )
    write_json(
        root / "prepared.json",
        dict(
            cities=stats,
            unique_pano_sentences=len(texts),
            base_osm_rows=len(osm_keys),
            supplemental_osm_rows=len(missing),
            note="All baseline pairs retained in original order. Supplement uses the exact parsed osm training tags.",
        ),
    )
    print(
        "Prepared",
        len(texts),
        "pano sentences;",
        len(missing),
        "supplemental osm bundles",
        flush=True,
    )


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

    root, supplement = args.root, args.supplement_dir
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
    missing_count = json.loads((root / "prepared.json").read_text())[
        "supplemental_osm_rows"
    ]
    if missing_count and not (supplement / "sentences.json").exists():
        print(
            "Pano embeddings complete; collect supplemental sentences then rerun embed.",
            flush=True,
        )
        return
    missing = (
        json.loads((supplement / "manifest.json").read_text())["sample"]
        if missing_count
        else {}
    )
    sentences = (
        json.loads((supplement / "sentences.json").read_text()) if missing_count else {}
    )
    assert set(missing) == set(sentences)
    for variant in ["descriptions", "tag_strings"]:
        inputs = [
            sentences[k]
            if variant == "descriptions"
            else ", ".join(f"{a}={b}" for a, b in sorted(v["tags"].items()))
            for k, v in missing.items()
        ]
        path = root / f"supplement_{variant}.npy"
        if not path.exists():
            values = (
                np.concatenate(
                    [
                        embed_texts_vertex(inputs[a:b], **config)
                        for a, b in batches(inputs, 0)
                    ]
                )
                if inputs
                else np.empty((0, 768), dtype=np.float32)
            )
            assert values.shape == (len(inputs), 768) and np.isfinite(values).all()
            np.save(path, values)
    write_json(
        root / "embedding_complete.json",
        dict(
            model=config["model"],
            dimensions=768,
            task_type="SEMANTIC_SIMILARITY",
            auto_truncate=False,
            pano_rows=len(texts),
            supplemental_osm_rows=len(missing),
        ),
    )


class FixedEmbeddingPairs(Dataset):
    def __init__(self, root, osm, city, variant):
        self.pairs = dict(np.load(root / f"{city}_pairs.npz"))
        self.pano = np.load(root / "pano_embeddings.npy", mmap_mode="r")
        self.osm = np.load(osm / f"{variant}.npy", mmap_mode="r")
        self.extra = np.load(root / f"supplement_{variant}.npy")

    def __len__(self):
        return len(self.pairs["labels"])

    def __getitem__(self, i):
        a, b = self.pairs["pano_idx"][i], self.pairs["osm_idx"][i]
        osm = self.osm[b] if b < len(self.osm) else self.extra[b - len(self.osm)]
        return self.pano[a], osm, self.pairs["labels"][i]


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
            pano_input="pano sentence embedding",
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
            supplemental_embeddings=str(root / f"supplement_{variant}.npy"),
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
    parser = argparse.ArgumentParser(description=__doc__)
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
        if action in ("prepare", "embed"):
            command.add_argument(
                "--supplement-dir",
                type=Path,
                required=True,
                help="Supplemental OSM requests and collected sentences",
            )
        if action == "prepare":
            command.add_argument(
                "--osm-source-dir",
                type=Path,
                required=True,
                help="Original OSM manifest.json and requests.jsonl",
            )
            command.add_argument(
                "--pano-source-dir",
                type=Path,
                required=True,
                help="City directories containing embeddings/embeddings.pkl",
            )
        if action == "check":
            command.add_argument(
                "--checkpoint",
                type=Path,
                help="Also validate loading an archived best_model.pt",
            )
        if action == "train":
            command.add_argument(
                "--variant", choices=["descriptions", "tag_strings"], required=True
            )
    args = parser.parse_args(argv)
    {"prepare": prepare, "embed": embed_inputs, "check": check, "train": train}[
        args.action
    ](args)


if __name__ == "__main__":
    main()
