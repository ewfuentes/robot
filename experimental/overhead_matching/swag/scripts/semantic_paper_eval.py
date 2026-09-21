"""Paper pipeline adapter: frozen 768-D inputs, unchanged correspondence matching."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import common.torch.load_torch_deps
import numpy as np
import torch
from torch import nn
from experimental.overhead_matching.swag.scripts.export_correspondence_similarity import load_vigor_dataset
from experimental.overhead_matching.swag.evaluation import correspondence_matching as cm

ROOT = Path('/data/overhead_matching/rebuttal/semantic_representation_20260921')
OUT = ROOT / 'paper_eval'
OSM = Path('/data/overhead_matching/rebuttal/osm_sentences_paper_full_20260920/text_embedding_005_768')
VIGOR = Path('/data/overhead_matching/datasets/VIGOR')


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def context(city):
    raw = torch.load(OUT / 'inputs' / city / 'baseline_raw_metadata.pt', weights_only=False)
    version = 'v4_202001' if city in ['Seattle', 'NewYork'] else ('boston' if city in ['Boston', 'nightdrive'] else None)
    dataset = load_vigor_dataset(VIGOR / city, version, 1.0)
    columns = {idx: c for c, idx in enumerate(raw['osm_lm_indices'])}
    # Fail on changed OSM ordering/tags, rather than silently score different entities.
    for idx, tags in zip(raw['osm_lm_indices'], raw['osm_lm_tags']):
        assert dict(dataset._landmark_metadata.iloc[idx]['pruned_props']) == dict(tags), (city, idx)
    sat_cols = [[columns[i] for i in indexes if i in columns]
                for indexes in dataset._satellite_metadata.landmark_idxs]
    # Archived raw data may include entities outside the current satellite grid;
    # retain them because the paper's uniqueness counts used the full raw matrix.
    assert set(i for cols in sat_cols for i in cols).issubset(range(len(columns)))
    pano_ids = list(dataset._panorama_metadata.pano_id)
    assert set(raw['pano_id_to_lm_rows']).issubset(pano_ids)
    identity = dict(pano_ids=pano_ids, satellite_paths=[str(p) for p in dataset._satellite_metadata.path])
    return raw, dataset, sat_cols, identity


def audit(city):
    raw, dataset, cols, identity = context(city)
    cost_path = Path(raw['cost_matrix_path'])
    if not cost_path.exists():
        cost_path = VIGOR / city / 'correspondence_scores/simple_v1_v6_raw_cost_matrix.npy'
    cost = np.load(cost_path, mmap_mode='r')
    reference = torch.load(VIGOR / city / 'similarity_matrices/simple_v1_v6_hungarian_0.8_similarity.pt', weights_only=False)
    assert isinstance(reference, torch.Tensor)
    assert reference.shape == (len(identity['pano_ids']), len(cols))
    errors = {(u, d): [] for u in [False, True] for d in [False, True]}
    rng = np.random.default_rng(42)
    checked = 0
    for pi in rng.permutation(len(identity['pano_ids'])):
        rows = raw['pano_id_to_lm_rows'].get(identity['pano_ids'][pi])
        if not rows:
            continue
        values = cost[rows]
        weights = cm.compute_uniqueness_weights(values, .8)
        nonzero = torch.nonzero(reference[pi] > 0).flatten().numpy()
        si = np.unique(np.concatenate([rng.choice(len(cols), min(100, len(cols)), replace=False), nonzero[:100]]))
        for s in si:
            for (unique, dustbin), err in errors.items():
                score = cm.match_and_aggregate(values[:, cols[s]], cm.MatchingMethod.HUNGARIAN,
                    cm.AggregationMode.SUM, .8, weights if unique else None, dustbin).similarity_score
                err.append(abs(score - reference[pi, s].item()))
        checked += 1
        if checked == 100:
            break
    stats = {f'uniqueness={u},dustbin={d}': dict(max_error=max(v), mismatches=sum(x > 2e-6 for x in v), samples=len(v)) for (u, d), v in errors.items()}
    print(json.dumps(stats, indent=2), flush=True)
    matches = [(u, d) for (u, d), v in errors.items() if max(v) < 2e-6]
    expected = (True, True)
    assert expected in matches, ('Paper matching settings do not reproduce baseline', matches)
    save(OUT / 'inputs' / city / 'audit.json', dict(city=city, settings=dict(uniqueness_weighted=expected[0], use_dustbin=expected[1]), comparison=stats))
    save(OUT / 'inputs' / city / 'identity.json', identity)


def classifier(variant):
    model = nn.Sequential(nn.Linear(2304, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 1))
    state = torch.load(ROOT / variant / 'best_model.pt', weights_only=True, map_location='cpu')
    model.load_state_dict({k.removeprefix('classifier.'): v for k, v in state.items()})
    return model.eval().cuda()


def inference(city, variant, target):
    destination = target / 'cost_matrix.npy'
    if destination.exists():
        return destination
    assert (OUT / 'embedding_complete.json').exists()
    indices = np.load(OUT / 'inputs' / city / 'indices.npz')
    pano = torch.from_numpy(np.load(OUT / 'pano_embeddings.npy', mmap_mode='r')[indices['pano']]).cuda()
    osm = torch.from_numpy(np.load(OSM / f'{variant}.npy', mmap_mode='r')[indices['osm']]).cuda()
    model = classifier(variant)
    partial = target / 'cost_matrix.partial.npy'
    progress = target / 'inference_progress.json'
    start = json.loads(progress.read_text())['rows'] if progress.exists() else 0
    cost = np.lib.format.open_memmap(partial, mode='r+' if partial.exists() else 'w+', dtype=np.float32, shape=(len(pano), len(osm)))
    with torch.inference_mode():
        # Keep float32 inference, as in the original correspondence exporter.
        for a in range(start, len(pano), 16):
            b = min(a + 16, len(pano))
            for c in range(0, len(osm), 3072):
                d = min(c + 3072, len(osm))
                p = pano[a:b, None, :].expand(-1, d-c, -1)
                o = osm[None, c:d, :].expand(b-a, -1, -1)
                probs = model(torch.cat([p, o, p*o], -1).reshape(-1, 2304)).sigmoid().reshape(b-a, d-c)
                cost[a:b, c:d] = probs.cpu().numpy()
            if a == start or b % 1024 == 0 or b == len(pano):
                cost.flush()
                save(progress, dict(rows=b, total=len(pano)))
                print(city, variant, 'inference', b, '/', len(pano), flush=True)
        # Runnable equivalence check of block/cartesian indexing against direct pairs.
        for a, b in [(0, 0), (len(pano)-1, len(osm)-1), (len(pano)//2, len(osm)//2)]:
            exact = model(torch.cat([pano[a], osm[b], pano[a]*osm[b]])[None]).sigmoid().item()
            assert abs(exact - float(cost[a, b])) < 2e-6
    cost.flush()
    partial.replace(destination)
    del pano, osm, model
    torch.cuda.empty_cache()
    return destination


def row_score(index):
    rows = RAW['pano_id_to_lm_rows'].get(PANO_IDS[index])
    result = np.zeros(len(COLS), dtype=np.float32)
    if not rows:
        return index, result
    values = COST[rows]
    weights = cm.compute_uniqueness_weights(values, .8) if SETTINGS['uniqueness_weighted'] else None
    for s, cols in enumerate(COLS):
        if cols:
            result[s] = cm.match_and_aggregate(values[:, cols], cm.MatchingMethod.HUNGARIAN,
                cm.AggregationMode.SUM, .8, weights, SETTINGS['use_dustbin']).similarity_score
    return index, result


def score(city, variant, workers):
    global RAW, COST, COLS, PANO_IDS, SETTINGS
    target = OUT / variant / city
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'similarity.pt').exists():
        return
    audit_path = OUT / 'inputs' / city / 'audit.json'
    assert audit_path.exists(), 'Verify original paper aggregation first'
    SETTINGS = json.loads(audit_path.read_text())['settings']
    RAW, dataset, COLS, identity = context(city)
    expected = json.loads((OUT / 'inputs' / city / 'identity.json').read_text())
    assert identity == expected, 'Dataset identity/order differs from audited paper inputs'
    PANO_IDS = identity['pano_ids']
    path = inference(city, variant, target)
    COST = np.load(path, mmap_mode='r')
    assert COST.shape == (len(RAW['pano_lm_tags']), len(RAW['osm_lm_indices']))
    similarity = np.zeros((len(PANO_IDS), len(COLS)), dtype=np.float32)
    torch.set_num_threads(1)
    with mp.get_context('fork').Pool(workers) as pool:
        for n, (i, row) in enumerate(pool.imap_unordered(row_score, range(len(PANO_IDS)), chunksize=8)):
            similarity[i] = row
            if n % 100 == 0:
                print(city, variant, 'matching', n, '/', len(PANO_IDS), flush=True)
    assert np.isfinite(similarity).all() and (similarity >= 0).all()
    temporary = target / 'similarity.partial.pt'
    torch.save(torch.from_numpy(similarity), temporary)
    temporary.replace(target / 'similarity.pt')
    save(target / 'complete.json', dict(city=city, variant=variant, shape=list(similarity.shape), settings=SETTINGS,
        threshold=.8, method='hungarian', aggregation='sum', dimensions=768,
        checkpoint_sha256=hashlib.sha256((ROOT / variant / 'best_model.pt').read_bytes()).hexdigest()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit', 'score'])
    parser.add_argument('--city', required=True)
    parser.add_argument('--variant', choices=['descriptions', 'tag_strings'], default='descriptions')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'audit':
        audit(args.city)
    else:
        score(args.city, args.variant, args.workers)
