"""Paper pipeline adapter: frozen 768-D inputs, unchanged correspondence matching."""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import common.torch.load_torch_deps
import numpy as np
import torch
from experimental.overhead_matching.swag.scripts.export_correspondence_similarity import (
    load_vigor_dataset,
)
from experimental.overhead_matching.swag.evaluation import correspondence_matching as cm
from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
    FixedEmbeddingClassifier,
)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def context(args):
    raw = torch.load(args.raw_metadata, weights_only=False)
    dataset = load_vigor_dataset(args.dataset_path, args.landmark_version, 1.0)
    columns = {idx: c for c, idx in enumerate(raw["osm_lm_indices"])}
    # Fail on changed OSM ordering/tags, rather than silently score different entities.
    for idx, tags in zip(raw["osm_lm_indices"], raw["osm_lm_tags"]):
        assert dict(dataset._landmark_metadata.iloc[idx]["pruned_props"]) == dict(
            tags
        ), (args.dataset_path, idx)
    sat_cols = [
        [columns[i] for i in indexes if i in columns]
        for indexes in dataset._satellite_metadata.landmark_idxs
    ]
    # Archived raw data may include entities outside the current satellite grid;
    # retain them because the paper's uniqueness counts used the full raw matrix.
    assert set(i for cols in sat_cols for i in cols).issubset(range(len(columns)))
    pano_ids = list(dataset._panorama_metadata.pano_id)
    assert set(raw["pano_id_to_lm_rows"]).issubset(pano_ids)
    identity = dict(
        pano_ids=pano_ids,
        satellite_paths=[str(p) for p in dataset._satellite_metadata.path],
    )
    return raw, dataset, sat_cols, identity


def audit(args):
    raw, dataset, cols, identity = context(args)
    cost_path = args.reference_cost_matrix or Path(raw["cost_matrix_path"])
    cost = np.load(cost_path, mmap_mode="r")
    reference = torch.load(args.reference_similarity, weights_only=False)
    assert isinstance(reference, torch.Tensor)
    assert reference.shape == (len(identity["pano_ids"]), len(cols))
    errors = {(u, d): [] for u in [False, True] for d in [False, True]}
    rng = np.random.default_rng(42)
    checked = 0
    for pi in rng.permutation(len(identity["pano_ids"])):
        rows = raw["pano_id_to_lm_rows"].get(identity["pano_ids"][pi])
        if not rows:
            continue
        values = cost[rows]
        weights = cm.compute_uniqueness_weights(values, 0.8)
        nonzero = torch.nonzero(reference[pi] > 0).flatten().numpy()
        si = np.unique(
            np.concatenate(
                [
                    rng.choice(len(cols), min(100, len(cols)), replace=False),
                    nonzero[:100],
                ]
            )
        )
        for s in si:
            for (unique, dustbin), err in errors.items():
                score = cm.match_and_aggregate(
                    values[:, cols[s]],
                    cm.MatchingMethod.HUNGARIAN,
                    cm.AggregationMode.SUM,
                    0.8,
                    weights if unique else None,
                    dustbin,
                ).similarity_score
                err.append(abs(score - reference[pi, s].item()))
        checked += 1
        if checked == 100:
            break
    stats = {
        f"uniqueness={u},dustbin={d}": dict(
            max_error=max(v), mismatches=sum(x > 2e-6 for x in v), samples=len(v)
        )
        for (u, d), v in errors.items()
    }
    print(json.dumps(stats, indent=2), flush=True)
    matches = [(u, d) for (u, d), v in errors.items() if max(v) < 2e-6]
    expected = (True, True)
    assert expected in matches, (
        "Paper matching settings do not reproduce baseline",
        matches,
    )
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    save(
        args.audit_dir / "audit.json",
        dict(
            city=args.dataset_path.name,
            settings=dict(uniqueness_weighted=expected[0], use_dustbin=expected[1]),
            comparison=stats,
        ),
    )
    save(args.audit_dir / "identity.json", identity)


def inference(args, target):
    destination = target / "cost_matrix.npy"
    if destination.exists():
        return destination
    indices = np.load(args.indices)
    pano = torch.from_numpy(
        np.load(args.pano_embeddings, mmap_mode="r")[indices["pano"]]
    ).to(args.device)
    osm = torch.from_numpy(
        np.load(args.osm_embeddings, mmap_mode="r")[indices["osm"]]
    ).to(args.device)
    model = FixedEmbeddingClassifier.load_checkpoint(args.checkpoint, args.device)
    partial = target / "cost_matrix.partial.npy"
    progress = target / "inference_progress.json"
    start = json.loads(progress.read_text())["rows"] if progress.exists() else 0
    cost = np.lib.format.open_memmap(
        partial,
        mode="r+" if partial.exists() else "w+",
        dtype=np.float32,
        shape=(len(pano), len(osm)),
    )
    with torch.inference_mode():
        # Keep float32 inference, as in the original correspondence exporter.
        for a in range(start, len(pano), 16):
            b = min(a + 16, len(pano))
            for c in range(0, len(osm), 3072):
                d = min(c + 3072, len(osm))
                p = pano[a:b, None, :].expand(-1, d - c, -1)
                o = osm[None, c:d, :].expand(b - a, -1, -1)
                probs = model.classify_from_reprs(p, o).sigmoid().reshape(b - a, d - c)
                cost[a:b, c:d] = probs.cpu().numpy()
            if a == start or b % 1024 == 0 or b == len(pano):
                cost.flush()
                save(progress, dict(rows=b, total=len(pano)))
                print(
                    args.dataset_path.name,
                    args.variant,
                    "inference",
                    b,
                    "/",
                    len(pano),
                    flush=True,
                )
        # Runnable equivalence check of block/cartesian indexing against direct pairs.
        for a, b in [
            (0, 0),
            (len(pano) - 1, len(osm) - 1),
            (len(pano) // 2, len(osm) // 2),
        ]:
            exact = (
                model.classify_from_reprs(pano[a : a + 1], osm[b : b + 1])
                .sigmoid()
                .item()
            )
            assert abs(exact - float(cost[a, b])) < 2e-6
    cost.flush()
    partial.replace(destination)
    del pano, osm, model
    torch.cuda.empty_cache()
    return destination


def score_rows(task):
    """Score one partition with explicit inputs; no inherited process state."""
    cost_path, pano_rows, columns, settings, indices = task
    cost = np.load(cost_path, mmap_mode="r")
    result = np.zeros((len(indices), len(columns)), dtype=np.float32)
    for offset, index in enumerate(indices):
        rows = pano_rows[index]
        if not rows:
            continue
        values = cost[rows]
        weights = (
            cm.compute_uniqueness_weights(values, 0.8)
            if settings["uniqueness_weighted"]
            else None
        )
        for satellite, cols in enumerate(columns):
            if cols:
                result[offset, satellite] = cm.match_and_aggregate(
                    values[:, cols],
                    cm.MatchingMethod.HUNGARIAN,
                    cm.AggregationMode.SUM,
                    0.8,
                    weights,
                    settings["use_dustbin"],
                ).similarity_score
    return indices, result


def score_matrix(cost_path, pano_rows, columns, settings, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    result = np.zeros((len(pano_rows), len(columns)), dtype=np.float32)
    if not pano_rows:
        return result
    workers = min(workers, len(pano_rows))
    tasks = [
        (cost_path, pano_rows, columns, settings, indices.tolist())
        for indices in np.array_split(np.arange(len(pano_rows)), workers)
    ]
    if workers == 1:
        indices, rows = score_rows(tasks[0])
        result[indices] = rows
    else:
        # Spawn avoids inheriting the CUDA context used for classifier inference.
        # Each task opens the large matrix read-only instead of pickling it.
        with mp.get_context("spawn").Pool(workers) as pool:
            for indices, rows in pool.imap_unordered(score_rows, tasks):
                result[indices] = rows
    return result


def score(args):
    target = args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    if (target / "similarity.pt").exists():
        return
    audit_path = args.audit_dir / "audit.json"
    assert audit_path.exists(), "Verify original paper aggregation first"
    settings = json.loads(audit_path.read_text())["settings"]
    raw, dataset, cols, identity = context(args)
    expected = json.loads((args.audit_dir / "identity.json").read_text())
    assert identity == expected, (
        "Dataset identity/order differs from audited paper inputs"
    )
    pano_ids = identity["pano_ids"]
    path = inference(args, target)
    cost = np.load(path, mmap_mode="r")
    assert cost.shape == (len(raw["pano_lm_tags"]), len(raw["osm_lm_indices"]))
    pano_rows = [raw["pano_id_to_lm_rows"].get(pid, []) for pid in pano_ids]
    similarity = score_matrix(path, pano_rows, cols, settings, args.workers)
    assert np.isfinite(similarity).all() and (similarity >= 0).all()
    temporary = target / "similarity.partial.pt"
    torch.save(torch.from_numpy(similarity), temporary)
    temporary.replace(target / "similarity.pt")
    save(
        target / "complete.json",
        dict(
            city=args.dataset_path.name,
            variant=args.variant,
            shape=list(similarity.shape),
            settings=settings,
            threshold=0.8,
            method="hungarian",
            aggregation="sum",
            dimensions=768,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "score"):
        command = commands.add_parser(name)
        command.add_argument("--dataset-path", type=Path, required=True)
        command.add_argument(
            "--landmark-version",
            default=None,
            help="Auto-detected when the dataset has one version",
        )
        command.add_argument(
            "--raw-metadata",
            type=Path,
            required=True,
            help="Archived baseline_raw_metadata.pt",
        )
        command.add_argument(
            "--audit-dir",
            type=Path,
            required=True,
            help="Directory for audit.json and identity.json",
        )
        if name == "audit":
            command.add_argument("--reference-similarity", type=Path, required=True)
            command.add_argument(
                "--reference-cost-matrix",
                type=Path,
                help="Override the path stored in raw metadata",
            )
        else:
            command.add_argument(
                "--variant", choices=["descriptions", "tag_strings"], required=True
            )
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument(
                "--indices",
                type=Path,
                required=True,
                help="NPZ with pano/osm row mappings into embedding arrays",
            )
            command.add_argument("--pano-embeddings", type=Path, required=True)
            command.add_argument("--osm-embeddings", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--workers", type=int, default=12)
            command.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    {"audit": audit, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
