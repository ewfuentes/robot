# Marimo notebook for learning per-tag-key weights via multiclass logistic
# regression (softmax cross-entropy). Trains on Chicago VIGOR data and evaluates
# MRR on both Chicago and Seattle.

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import common.torch.load_torch_deps
    import torch
    import torch.nn.functional as F
    import numpy as np
    import polars as pl
    import pickle
    from collections import Counter
    from pathlib import Path
    from tqdm import tqdm

    from experimental.overhead_matching.swag.data import vigor_dataset as vd

    return Path, np, pickle, pl, torch, tqdm, vd


@app.cell
def _(Path, mo):
    DATASET_BASE = Path("/data/overhead_matching/datasets/VIGOR")
    LANDMARK_VERSION = "v4_202001"
    PANORAMA_LANDMARK_RADIUS_PX = 640.0
    INFLATION_FACTOR = 1.0

    mo.md(f"""
    **Configuration**
    - Dataset base: `{DATASET_BASE}`
    - Landmark version: `{LANDMARK_VERSION}`
    - Panorama landmark radius: `{PANORAMA_LANDMARK_RADIUS_PX}` px
    - Inflation factor: `{INFLATION_FACTOR}`
    """)
    return (
        DATASET_BASE,
        INFLATION_FACTOR,
        LANDMARK_VERSION,
        PANORAMA_LANDMARK_RADIUS_PX,
    )


@app.cell
def _(
    DATASET_BASE,
    INFLATION_FACTOR,
    LANDMARK_VERSION,
    PANORAMA_LANDMARK_RADIUS_PX,
    vd,
):
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=LANDMARK_VERSION,
        panorama_landmark_radius_px=PANORAMA_LANDMARK_RADIUS_PX,
        landmark_correspondence_inflation_factor=INFLATION_FACTOR,
    )
    chicago_dataset = vd.VigorDataset(DATASET_BASE / "Chicago", _config)
    print(
        f"Chicago: {len(chicago_dataset._panorama_metadata)} panos, "
        f"{len(chicago_dataset._satellite_metadata)} sats, "
        f"{len(chicago_dataset._landmark_metadata)} landmarks"
    )
    return (chicago_dataset,)


@app.cell
def _(
    DATASET_BASE,
    INFLATION_FACTOR,
    LANDMARK_VERSION,
    PANORAMA_LANDMARK_RADIUS_PX,
    vd,
):
    _config = vd.VigorDatasetConfig(
        satellite_tensor_cache_info=None,
        panorama_tensor_cache_info=None,
        should_load_images=False,
        should_load_landmarks=True,
        landmark_version=LANDMARK_VERSION,
        panorama_landmark_radius_px=PANORAMA_LANDMARK_RADIUS_PX,
        landmark_correspondence_inflation_factor=INFLATION_FACTOR,
    )
    seattle_dataset = vd.VigorDataset(DATASET_BASE / "Seattle", _config)
    print(
        f"Seattle: {len(seattle_dataset._panorama_metadata)} panos, "
        f"{len(seattle_dataset._satellite_metadata)} sats, "
        f"{len(seattle_dataset._landmark_metadata)} landmarks"
    )
    return (seattle_dataset,)


@app.cell
def _():
    _base = "~/scratch/landmark_tables"
    CHI_MATCHES_PATH = f"{_base}/chicago_pano_osm_matches.parquet"
    CHI_SAT_OSM_PATH = f"{_base}/chicago_sat_osm_table.parquet"
    SEA_MATCHES_PATH = f"{_base}/seattle_pano_osm_matches.parquet"
    SEA_SAT_OSM_PATH = f"{_base}/seattle_sat_osm_table.parquet"
    return (
        CHI_MATCHES_PATH,
        CHI_SAT_OSM_PATH,
        SEA_MATCHES_PATH,
        SEA_SAT_OSM_PATH,
    )


@app.cell
def _(CHI_MATCHES_PATH, pl):
    pl.read_parquet(CHI_MATCHES_PATH)
    return


@app.cell
def _(CHI_SAT_OSM_PATH, pl):
    pl.read_parquet(CHI_SAT_OSM_PATH)
    return


@app.cell
def _(pickle):
    # with open("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/Chicago/embeddings/embeddings.pkl", "rb") as f:
    #     _chic_pano_lms = pickle.load(f)
    with open("/data/overhead_matching/datasets/semantic_landmark_embeddings/pano_v2/Seattle/embeddings/embeddings.pkl", "rb") as f:
        _sea_pano_lms = pickle.load(f)
    pano_landmarks = dict(
        # chicago=_chic_pano_lms,
        seattle=_sea_pano_lms
    )
    return (pano_landmarks,)


@app.cell
def _(pano_landmarks):
    next(iter(pano_landmarks["seattle"]["panoramas"].items()))
    return


@app.cell
def _(CHI_MATCHES_PATH, mo, pl):
    tag_key_df = (
        pl.scan_parquet(CHI_MATCHES_PATH)
        .select("tag_key")
        .group_by("tag_key")
        .len()
        .sort("len", descending=True)
        .collect()
    )
    tag_keys = sorted(
        tag_key_df.filter(pl.col("len") >= 5)["tag_key"].to_list()
    )
    num_tag_keys = len(tag_keys)
    _vocab_set = set(tag_keys)
    tag_key_df = tag_key_df.with_columns(
        pl.col("tag_key").is_in(_vocab_set).alias("in_vocab")
    )

    print(f"Total unique keys: {len(tag_key_df)}, in vocabulary (>=5 occurrences): {num_tag_keys}")
    mo.ui.table(tag_key_df, label="Tag Key Vocabulary")
    return num_tag_keys, tag_key_df, tag_keys


@app.cell
def _(mo, num_tag_keys, pl, tag_key_df):
    mo.vstack([
        mo.md(f"**Tag key vocabulary size: {num_tag_keys}**"),
        mo.ui.table(
            tag_key_df.filter(pl.col("in_vocab")),
            label="Vocabulary Keys (>=5 occurrences)",
        ),
    ])
    return


@app.cell
def _(seattle_dataset):
    seattle_dataset._panorama_metadata
    return


@app.cell
def _(np, pl, torch, tqdm, vd):
    from collections import defaultdict

    class MatchData:
        """Compact per-panorama match data for on-the-fly score computation.

        Stores condensed (osm_idx, key_idx, count) matches per panorama as flat
        arrays, plus an osm_idx → sat_idx expansion table. Scores are computed
        on-the-fly during optimization — no (pano × sat × key) matrix needed.
        """
        def __init__(self, num_panos, num_sats, num_keys,
                     match_osm_idxs, match_key_idxs, match_counts,
                     pano_boundaries, osm_to_sat_idxs, osm_to_sat_offsets,
                     positive_sat_idxs_per_pano):
            self.num_panos = num_panos
            self.num_sats = num_sats
            self.num_keys = num_keys
            # Flat arrays over all condensed matches, sorted by pano_idx
            self.match_osm_idxs = match_osm_idxs        # int32 (num_condensed,)
            self.match_key_idxs = match_key_idxs        # int16 (num_condensed,)
            self.match_counts = match_counts             # int16 (num_condensed,)
            # (start, end) into match arrays for each pano_idx
            self.pano_boundaries = pano_boundaries       # list of (int, int)
            # Flat CSR for osm_idx → list of sat_idxs
            self.osm_to_sat_idxs = osm_to_sat_idxs      # int32 (num_osm_sat_pairs,)
            self.osm_to_sat_offsets = osm_to_sat_offsets  # int32 (max_osm_idx+2,)
            # Ground truth
            self.positive_sat_idxs_per_pano = positive_sat_idxs_per_pano  # list of sets

        def compute_pano_scores(self, theta, pano_idx):
            """Compute score(pano_idx, sat_idx) for all sats, given weight vector theta.

            Returns a dense float64 tensor of shape (num_sats,).
            Fully vectorized via scatter_add (no Python loop).
            """
            start, end = self.pano_boundaries[pano_idx]
            scores = torch.zeros(self.num_sats, dtype=torch.float64)
            if start == end:
                return scores

            osm_idxs = self.match_osm_idxs[start:end].long()
            key_idxs = self.match_key_idxs[start:end].long()
            counts = self.match_counts[start:end].double()

            weights = theta[key_idxs] * counts  # (M,)

            # Vectorized osm→sat expansion
            o_starts = self.osm_to_sat_offsets[osm_idxs]
            o_ends = self.osm_to_sat_offsets[osm_idxs + 1]
            sizes = (o_ends - o_starts).long()
            total_expanded = sizes.sum().item()
            if total_expanded == 0:
                return scores

            expanded_weights = torch.repeat_interleave(weights, sizes)
            cumsize = sizes.cumsum(0)
            cumsize_prev = torch.cat([torch.zeros(1, dtype=torch.long), cumsize[:-1]])
            rep_o_starts = torch.repeat_interleave(o_starts.long(), sizes)
            rep_cumsize_prev = torch.repeat_interleave(cumsize_prev, sizes)
            flat_idx = rep_o_starts + torch.arange(total_expanded, dtype=torch.long) - rep_cumsize_prev
            expanded_sat_idxs = self.osm_to_sat_idxs[flat_idx].long()

            scores.scatter_add_(0, expanded_sat_idxs, expanded_weights)
            return scores

    def precompute_match_data(
        matches_path: str,
        sat_osm_path: str,
        vigor_dataset: vd.VigorDataset,
        vocab: list[str],
        chunk_rows: int = 10_000_000,
    ) -> MatchData:
        """Condense match parquet into compact per-panorama arrays.

        Groups by (pano_id, osm_idx, tag_key) in chunks, then builds flat
        arrays sorted by pano_idx. ~1-2 GB total.
        """
        import tempfile, os

        num_keys = len(vocab)
        key_to_idx = {k: i for i, k in enumerate(vocab)}
        pano_meta = vigor_dataset._panorama_metadata
        num_panos = len(pano_meta)
        num_sats = len(vigor_dataset._satellite_metadata)
        pano_id_to_idx = {row["pano_id"]: idx for idx, (_, row) in enumerate(pano_meta.iterrows())}

        # --- Build osm → sat CSR from small table ---
        sat_osm_df = pl.read_parquet(sat_osm_path).select("osm_idx", "sat_idx").unique()
        max_osm = sat_osm_df["osm_idx"].max()
        # Some osm_idxs in matches may not appear in sat table — size CSR to cover all
        max_osm_matches = (
            pl.scan_parquet(matches_path)
            .select(pl.col("osm_idx").max())
            .collect()
            .item()
        )
        max_osm = max(max_osm, max_osm_matches)
        osm_to_sat_lists = defaultdict(list)
        for osm_idx, sat_idx in zip(sat_osm_df["osm_idx"], sat_osm_df["sat_idx"]):
            osm_to_sat_lists[osm_idx].append(sat_idx)
        # Build CSR
        osm_to_sat_offsets = np.zeros(max_osm + 2, dtype=np.int32)
        all_sat_idxs = []
        for osm in range(max_osm + 1):
            sats = osm_to_sat_lists.get(osm, [])
            all_sat_idxs.extend(sats)
            osm_to_sat_offsets[osm + 1] = osm_to_sat_offsets[osm] + len(sats)
        osm_to_sat_flat = torch.tensor(all_sat_idxs, dtype=torch.int32)
        osm_to_sat_offsets = torch.tensor(osm_to_sat_offsets, dtype=torch.int32)
        del sat_osm_df, osm_to_sat_lists
        print(f"  osm→sat CSR: {len(osm_to_sat_flat)} entries, max_osm={max_osm}")

        # --- Condense matches: group by (pano_id, osm_idx, tag_key) in chunks ---
        total_rows = pl.scan_parquet(matches_path).select(pl.len()).collect().item()
        num_chunks = (total_rows + chunk_rows - 1) // chunk_rows
        print(f"  Total matches: {total_rows}, condensing in {num_chunks} chunks")

        tmp_dir = tempfile.mkdtemp(prefix="tag_weight_")
        chunk_paths = []
        for i in tqdm(range(num_chunks), desc="Condensing chunks"):
            offset = i * chunk_rows
            condensed = (
                pl.scan_parquet(matches_path)
                .slice(offset, chunk_rows)
                .filter(pl.col("tag_key").is_in(vocab))
                .select("pano_id", "osm_idx", "tag_key")
                .group_by("pano_id", "osm_idx", "tag_key")
                .len()
                .collect()
            )
            if len(condensed) > 0:
                chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.parquet")
                condensed.write_parquet(chunk_path)
                chunk_paths.append(chunk_path)

        # --- Merge chunks and build flat arrays ---
        # Stream chunks into per-pano lists
        pano_matches = defaultdict(list)  # pano_idx -> list of (osm_idx, key_idx, count)
        total_condensed = 0
        for ci, chunk_path in enumerate(tqdm(chunk_paths, desc="Merging chunks")):
            chunk = pl.read_parquet(chunk_path)
            total_condensed += len(chunk)
            for pano_id, osm_idx, tag_key, count in chunk.iter_rows():
                pano_idx = pano_id_to_idx.get(pano_id)
                if pano_idx is None:
                    continue
                kid = key_to_idx[tag_key]
                pano_matches[pano_idx].append((osm_idx, kid, count))
            del chunk
            os.remove(chunk_path)
        os.rmdir(tmp_dir)
        print(f"  Condensed rows: {total_condensed}, panos with matches: {len(pano_matches)}")

        # Note: same (pano, osm, key) can appear in multiple chunks — merge counts
        all_osm = []
        all_key = []
        all_count = []
        boundaries = []
        for p in range(num_panos):
            start = len(all_osm)
            if p in pano_matches:
                # Merge duplicates from different chunks
                merged = defaultdict(int)
                for osm_idx, kid, count in pano_matches[p]:
                    merged[(osm_idx, kid)] += count
                for (osm_idx, kid), count in merged.items():
                    all_osm.append(osm_idx)
                    all_key.append(kid)
                    all_count.append(count)
            boundaries.append((start, len(all_osm)))
        del pano_matches

        match_osm_idxs = torch.tensor(all_osm, dtype=torch.int32)
        match_key_idxs = torch.tensor(all_key, dtype=torch.int16)
        match_counts = torch.tensor(all_count, dtype=torch.int16)

        total_matches = len(match_osm_idxs)
        mem_mb = (match_osm_idxs.nbytes + match_key_idxs.nbytes + match_counts.nbytes) / 1e6
        print(f"  Final: {total_matches} condensed matches, {mem_mb:.0f} MB")

        # Ground truth positive sets
        positive_sat_idxs_per_pano = [
            set(pano_meta.iloc[i]["positive_satellite_idxs"]) for i in range(num_panos)
        ]

        return MatchData(
            num_panos=num_panos, num_sats=num_sats, num_keys=num_keys,
            match_osm_idxs=match_osm_idxs,
            match_key_idxs=match_key_idxs,
            match_counts=match_counts,
            pano_boundaries=boundaries,
            osm_to_sat_idxs=osm_to_sat_flat,
            osm_to_sat_offsets=osm_to_sat_offsets,
            positive_sat_idxs_per_pano=positive_sat_idxs_per_pano,
        )

    def compute_loss_and_mrr(match_data, theta):
        """Compute softmax cross-entropy loss and MRR (evaluation only, no grad)."""
        bd = prepare_batched_data(match_data)
        loss, mrr, _ = run_batched_epoch(match_data, bd, theta, l2=0.0)
        return loss, mrr

    def _expand_osm_to_sat(md, osm_idxs):
        """Expand osm indices to sat indices via CSR. Returns (flat_sat_idxs, sizes, match_idx)."""
        o_starts = md.osm_to_sat_offsets[osm_idxs]
        o_ends = md.osm_to_sat_offsets[osm_idxs + 1]
        sizes = (o_ends - o_starts).long()
        total_exp = sizes.sum().item()
        if total_exp == 0:
            return None, sizes, None
        cumsize = sizes.cumsum(0)
        cumsize_prev = torch.cat([torch.zeros(1, dtype=torch.long), cumsize[:-1]])
        rep_o_starts = torch.repeat_interleave(o_starts.long(), sizes)
        rep_cp = torch.repeat_interleave(cumsize_prev, sizes)
        flat_idx = rep_o_starts + torch.arange(total_exp, dtype=torch.long) - rep_cp
        exp_sat_idxs = md.osm_to_sat_idxs[flat_idx].long()
        match_idx = torch.repeat_interleave(torch.arange(len(osm_idxs)), sizes)
        return exp_sat_idxs, sizes, match_idx

    def prepare_batched_data(md, batch_size=1024):
        """Partition valid panoramas into batches. Lightweight — no expansion precomputed."""
        valid_panos = [p for p in range(md.num_panos) if md.positive_sat_idxs_per_pano[p]]
        max_pos = max(len(md.positive_sat_idxs_per_pano[p]) for p in valid_panos)

        batches = []
        for b_start in range(0, len(valid_panos), batch_size):
            batch_panos = valid_panos[b_start:b_start + batch_size]
            B = len(batch_panos)

            pos_idxs = torch.zeros(B, max_pos, dtype=torch.long)
            pos_mask = torch.zeros(B, max_pos, dtype=torch.bool)
            for local_idx, p in enumerate(batch_panos):
                pos = list(md.positive_sat_idxs_per_pano[p])
                pos_idxs[local_idx, :len(pos)] = torch.tensor(pos)
                pos_mask[local_idx, :len(pos)] = True

            batches.append(dict(
                B=B,
                batch_panos=batch_panos,
                pos_idxs=pos_idxs,
                pos_mask=pos_mask,
            ))

        return dict(batches=batches, num_valid=len(valid_panos))

    def _build_batch_tensors(md, batch, device):
        """Build expansion tensors for a single batch on-the-fly and move to device."""
        B = batch["B"]
        batch_panos = batch["batch_panos"]

        match_slices = []
        pano_local_list = []
        for local_idx, p in enumerate(batch_panos):
            start, end = md.pano_boundaries[p]
            n = end - start
            if n > 0:
                match_slices.append((start, end))
                pano_local_list.append(torch.full((n,), local_idx, dtype=torch.long))

        if not match_slices:
            return None

        all_ranges = [torch.arange(s, e) for s, e in match_slices]
        all_idx = torch.cat(all_ranges)
        all_osm = md.match_osm_idxs[all_idx].long()
        all_key = md.match_key_idxs[all_idx].long()
        all_count = md.match_counts[all_idx].float()
        all_pano_local = torch.cat(pano_local_list)

        exp_sat_idxs, sizes, match_idx = _expand_osm_to_sat(md, all_osm)
        if exp_sat_idxs is None:
            return None
        exp_pano_local = torch.repeat_interleave(all_pano_local, sizes)
        scatter_idx = exp_pano_local * md.num_sats + exp_sat_idxs

        return dict(
            all_key=all_key.to(device),
            all_count=all_count.to(device),
            sizes=sizes.to(device),
            scatter_idx=scatter_idx.to(device),
            exp_pano_local=exp_pano_local.to(device),
            exp_sat_idxs=exp_sat_idxs.to(device),
            match_idx=match_idx.to(device),
        )

    def run_batched_epoch(md, batched_data, theta, l2):
        """Run one full epoch. Builds expansions on-the-fly per batch to save memory."""
        device = theta.device
        num_keys = len(theta)
        grad_accum = torch.zeros(num_keys, dtype=torch.float32, device=device)
        total_loss = 0.0
        mrr_sum = 0.0
        N = batched_data["num_valid"]

        for batch in tqdm(batched_data["batches"], desc="Batches"):
            B = batch["B"]
            bt = _build_batch_tensors(md, batch, device)
            if bt is None:
                continue

            b_pos_idxs = batch["pos_idxs"].to(device)
            b_pos_mask = batch["pos_mask"].to(device)

            weights = theta[bt["all_key"]] * bt["all_count"]
            exp_weights = torch.repeat_interleave(weights, bt["sizes"])

            # Scatter into (B, num_sats) score matrix
            scores_flat = torch.zeros(B * md.num_sats, dtype=torch.float32, device=device)
            scores_flat.scatter_add_(0, bt["scatter_idx"], exp_weights)
            scores = scores_flat.view(B, md.num_sats)

            # Softmax
            log_Z = torch.logsumexp(scores, dim=1)
            probs = torch.exp(scores - log_Z.unsqueeze(1))

            # Positive scores and loss
            pos_scores = scores.gather(1, b_pos_idxs)
            pos_scores[~b_pos_mask] = float("-inf")
            best_pos_score, best_local = pos_scores.max(dim=1)
            best_pos_idx = b_pos_idxs.gather(1, best_local.unsqueeze(1)).squeeze(1)

            batch_loss = -best_pos_score + log_Z
            total_loss += batch_loss.sum().item()

            # MRR
            ranks = (scores > best_pos_score.unsqueeze(1)).sum(dim=1) + 1
            mrr_sum += (1.0 / ranks.float()).sum().item()

            # --- Gradient ---
            sat_probs = probs[bt["exp_pano_local"], bt["exp_sat_idxs"]]
            prob_mass = torch.zeros(len(bt["all_key"]), dtype=torch.float32, device=device)
            prob_mass.scatter_add_(0, bt["match_idx"], sat_probs)
            expected = torch.zeros(num_keys, dtype=torch.float32, device=device)
            expected.scatter_add_(0, bt["all_key"], bt["all_count"] * prob_mass)

            target_pos = best_pos_idx[bt["exp_pano_local"]]
            is_pos_exp = (bt["exp_sat_idxs"] == target_pos)
            match_has_pos = torch.zeros(len(bt["all_key"]), dtype=torch.float32, device=device)
            if is_pos_exp.any():
                match_has_pos.scatter_add_(0, bt["match_idx"][is_pos_exp],
                                           torch.ones(is_pos_exp.sum(), dtype=torch.float32, device=device))
            pos_match_mask = match_has_pos > 0
            observed = torch.zeros(num_keys, dtype=torch.float32, device=device)
            if pos_match_mask.any():
                observed.scatter_add_(0, bt["all_key"][pos_match_mask],
                                      bt["all_count"][pos_match_mask])

            grad_accum += expected - observed

        if N > 0:
            grad_accum.div_(N)
            grad_accum.add_(l2 * theta)

        loss = total_loss / N if N > 0 else 0.0
        mrr = mrr_sum / N if N > 0 else 0.0
        return loss, mrr, grad_accum.cpu()

    return (
        compute_loss_and_mrr,
        precompute_match_data,
        prepare_batched_data,
        run_batched_epoch,
    )


@app.cell
def _(
    CHI_MATCHES_PATH,
    CHI_SAT_OSM_PATH,
    chicago_dataset,
    mo,
    precompute_match_data,
    tag_keys,
):
    with mo.persistent_cache(name="chi_match_data"):
        print("Precomputing Chicago match data (training)...")
        chi_match_data = precompute_match_data(
            CHI_MATCHES_PATH, CHI_SAT_OSM_PATH, chicago_dataset, tag_keys,
        )
    return (chi_match_data,)


@app.cell
def _(
    SEA_MATCHES_PATH,
    SEA_SAT_OSM_PATH,
    mo,
    precompute_match_data,
    seattle_dataset,
    tag_keys,
):
    with mo.persistent_cache(name="sea_match_data"):
        print("Precomputing Seattle match data (validation)...")
        sea_match_data = precompute_match_data(
            SEA_MATCHES_PATH, SEA_SAT_OSM_PATH, seattle_dataset, tag_keys,
        )
    return (sea_match_data,)


@app.cell
def _(chi_match_data, mo, sea_match_data):
    def _stats(md, name):
        _total = len(md.match_osm_idxs)
        _with = sum(1 for s, e in md.pano_boundaries if s < e)
        _mem = (md.match_osm_idxs.nbytes + md.match_key_idxs.nbytes +
                md.match_counts.nbytes + md.osm_to_sat_idxs.nbytes +
                md.osm_to_sat_offsets.nbytes) / 1e6
        return (f"**{name}**: {_total:,} matches, {_with:,}/{md.num_panos:,} panos, "
                f"{md.num_sats:,} sats, {_mem:.0f} MB, {len(md.osm_to_sat_idxs):,} osm→sat")

    mo.md(f"""
    {_stats(chi_match_data, "Chicago (train)")}

    {_stats(sea_match_data, "Seattle (val)")}
    """)
    return


@app.cell
def _(mo):
    optimizer_dropdown = mo.ui.dropdown(["L-BFGS", "Adam"], value="L-BFGS", label="Optimizer")
    lr_slider = mo.ui.slider(0.001, 1.0, value=1.0, step=0.001, label="Learning rate")
    l2_slider = mo.ui.slider(0.0, 1.0, value=0.01, step=0.005, label="L2 regularization (λ)")
    epochs_number = mo.ui.number(value=20, start=1, stop=1000, step=1, label="Steps")
    nonneg_check = mo.ui.checkbox(label="Non-negative weights (θ≥0 via exp reparameterization)", value=False)
    train_btn = mo.ui.run_button(label="Train", kind="success")
    mo.vstack([
        mo.md("### Optimization Hyperparameters"),
        mo.hstack([optimizer_dropdown, lr_slider, l2_slider, epochs_number]),
        nonneg_check,
        train_btn,
    ])
    return (
        epochs_number,
        l2_slider,
        lr_slider,
        nonneg_check,
        optimizer_dropdown,
        train_btn,
    )


@app.cell
def _(
    chi_match_data,
    epochs_number,
    l2_slider,
    lr_slider,
    mo,
    nonneg_check,
    optimizer_dropdown,
    prepare_batched_data,
    run_batched_epoch,
    sea_match_data,
    tag_keys,
    torch,
    train_btn,
):
    mo.stop(not train_btn.value, mo.md("Click **Train** to start optimization"))

    _num_keys = len(tag_keys)
    _lr = lr_slider.value
    _l2 = l2_slider.value
    _steps = int(epochs_number.value)
    _nonneg = nonneg_check.value
    _opt_type = optimizer_dropdown.value
    _md = chi_match_data
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {_device}, optimizer: {_opt_type}")

    print("Preparing batched data (Chicago train)...")
    _batched = prepare_batched_data(_md, batch_size=1024)
    print(f"  {_batched['num_valid']} valid panoramas in {len(_batched['batches'])} batches")

    print("Preparing batched data (Seattle val)...")
    _val_batched = prepare_batched_data(sea_match_data, batch_size=1024)
    print(f"  {_val_batched['num_valid']} valid panoramas in {len(_val_batched['batches'])} batches")

    if _nonneg:
        _phi = torch.zeros(_num_keys, dtype=torch.float32, requires_grad=True, device=_device)
        _param = _phi
    else:
        _theta_param = torch.zeros(_num_keys, dtype=torch.float32, requires_grad=True, device=_device)
        _param = _theta_param

    if _opt_type == "L-BFGS":
        _optimizer = torch.optim.LBFGS([_param], lr=_lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    else:
        _optimizer = torch.optim.Adam([_param], lr=_lr)

    train_history = {"loss": [], "mrr": [], "val_loss": [], "val_mrr": [], "val_epochs": []}

    def _get_theta():
        return torch.exp(_phi).detach() if _nonneg else _theta_param.detach()

    def _set_grad(grad):
        if _nonneg:
            _phi.grad = grad.to(_device) * torch.exp(_phi).detach()
        else:
            _theta_param.grad = grad.to(_device)

    _last = {}
    def _closure():
        _optimizer.zero_grad()
        _loss, _mrr, _grad = run_batched_epoch(_md, _batched, _get_theta(), _l2)
        _set_grad(_grad)
        _last["loss"] = _loss
        _last["mrr"] = _mrr
        return torch.tensor(_loss)

    for _step in range(_steps):
        if _opt_type == "L-BFGS":
            _optimizer.step(_closure)
            _loss, _mrr = _last["loss"], _last["mrr"]
        else:
            _closure()
            _optimizer.step()
            _loss, _mrr = _last["loss"], _last["mrr"]

        train_history["loss"].append(_loss)
        train_history["mrr"].append(_mrr)

        if (_step + 1) % 5 == 0 or _step == _steps - 1:
            _val_loss, _val_mrr, _ = run_batched_epoch(sea_match_data, _val_batched, _get_theta(), 0.0)
            train_history["val_loss"].append(_val_loss)
            train_history["val_mrr"].append(_val_mrr)
            train_history["val_epochs"].append(_step + 1)
            print(f"  Step {_step+1}: loss={_loss:.4f}, MRR={_mrr:.4f} | val_loss={_val_loss:.4f}, val_MRR={_val_mrr:.4f}")
        else:
            print(f"  Step {_step+1}: loss={_loss:.4f}, MRR={_mrr:.4f}")

    learned_theta = _get_theta().cpu().clone()
    print(f"\nTraining complete. Final loss={train_history['loss'][-1]:.4f}, MRR={train_history['mrr'][-1]:.4f}")
    return learned_theta, train_history


@app.cell
def _(mo, np, train_history):
    import matplotlib
    matplotlib.style.use("ggplot")
    import matplotlib.pyplot as _plt

    _fig, (_ax1, _ax2) = _plt.subplots(1, 2, figsize=(14, 5))

    _epochs = np.arange(1, len(train_history["loss"]) + 1)

    _val_epochs = np.array(train_history["val_epochs"])

    _ax1.plot(_epochs, train_history["loss"], "b-", linewidth=1.5, label="Chicago (train)")
    _ax1.plot(_val_epochs, train_history["val_loss"], "r-o", linewidth=1.5, markersize=4, label="Seattle (val)")
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Cross-entropy Loss")
    _ax1.set_title("Loss")
    _ax1.legend()
    _ax1.grid(True, alpha=0.3)

    _ax2.plot(_epochs, train_history["mrr"], "b-", linewidth=1.5, label="Chicago (train)")
    _ax2.plot(_val_epochs, train_history["val_mrr"], "r-o", linewidth=1.5, markersize=4, label="Seattle (val)")
    _ax2.set_xlabel("Epoch")
    _ax2.set_ylabel("Mean Reciprocal Rank")
    _ax2.set_title("MRR")
    _ax2.legend()
    _ax2.grid(True, alpha=0.3)

    _plt.tight_layout()
    mo.mpl.interactive(_plt.gcf())
    return


@app.cell
def _(learned_theta, mo, np, tag_keys):
    import matplotlib.pyplot as _plt

    _weights = learned_theta.numpy()
    _sorted_idx = np.argsort(np.abs(_weights))[::-1]

    _fig, _ax = _plt.subplots(figsize=(10, max(6, len(tag_keys) * 0.3)))
    _y = np.arange(len(tag_keys))
    _colors = ["green" if _weights[i] >= 0 else "red" for i in _sorted_idx]
    _ax.barh(_y, _weights[_sorted_idx], color=_colors, alpha=0.8)
    _ax.set_yticks(_y)
    _ax.set_yticklabels([tag_keys[i] for i in _sorted_idx], fontsize=7)
    _ax.set_xlabel("Learned weight θ_k")
    _ax.set_title("Learned Tag Key Weights (sorted by |θ|)")
    _ax.axvline(0, color="black", linewidth=0.5)
    _ax.invert_yaxis()
    _plt.tight_layout()
    mo.mpl.interactive(_plt.gcf())
    return


@app.cell
def _(
    chi_match_data,
    compute_loss_and_mrr,
    learned_theta,
    mo,
    sea_match_data,
    tag_keys,
    torch,
):
    _theta = learned_theta
    _uniform = torch.ones(len(tag_keys), dtype=torch.float32)
    _zeros = torch.zeros(len(tag_keys), dtype=torch.float32)

    print("Evaluating on Chicago (train)...")
    _chi_loss, _chi_mrr = compute_loss_and_mrr(chi_match_data, _theta)
    _chi_uni_loss, _chi_uni_mrr = compute_loss_and_mrr(chi_match_data, _uniform)
    _chi_zero_loss, _chi_zero_mrr = compute_loss_and_mrr(chi_match_data, _zeros)

    print("Evaluating on Seattle (val)...")
    _sea_loss, _sea_mrr = compute_loss_and_mrr(sea_match_data, _theta)
    _sea_uni_loss, _sea_uni_mrr = compute_loss_and_mrr(sea_match_data, _uniform)
    _sea_zero_loss, _sea_zero_mrr = compute_loss_and_mrr(sea_match_data, _zeros)

    mo.md(f"""
    ### Evaluation Results

    **Chicago (train)**

    | Method | Loss | MRR |
    |--------|------|-----|
    | **Learned** | {_chi_loss:.4f} | {_chi_mrr:.4f} |
    | Uniform | {_chi_uni_loss:.4f} | {_chi_uni_mrr:.4f} |
    | Zero | {_chi_zero_loss:.4f} | {_chi_zero_mrr:.4f} |

    **Seattle (val)**

    | Method | Loss | MRR |
    |--------|------|-----|
    | **Learned** | {_sea_loss:.4f} | {_sea_mrr:.4f} |
    | Uniform | {_sea_uni_loss:.4f} | {_sea_uni_mrr:.4f} |
    | Zero | {_sea_zero_loss:.4f} | {_sea_zero_mrr:.4f} |
    """)
    return


@app.cell
def _(
    Path,
    chi_match_data,
    learned_theta,
    sea_match_data,
    tag_keys,
    torch,
    tqdm,
    train_history,
):
    # Build full (num_panos, num_sats) similarity matrix from learned weights
    _save_prefix = "osm_tag_lbfgs"
    for _city, _match_data in zip(["chicago", "seattle"], [chi_match_data, sea_match_data]):
        _theta = learned_theta
        _md = _match_data
        _sim = torch.zeros(_md.num_panos, _md.num_sats, dtype=torch.float32)
    
        for p in tqdm(range(_md.num_panos), desc="Building similarity matrix"):
          start, end = _md.pano_boundaries[p]
          if start == end:
              continue
          osm_idxs = _md.match_osm_idxs[start:end].long()
          key_idxs = _md.match_key_idxs[start:end].long()
          counts = _md.match_counts[start:end].float()
          weights = _theta[key_idxs] * counts
    
          o_starts = _md.osm_to_sat_offsets[osm_idxs]
          o_ends = _md.osm_to_sat_offsets[osm_idxs + 1]
          sizes = (o_ends - o_starts).long()
          if sizes.sum() == 0:
              continue
          exp_weights = torch.repeat_interleave(weights, sizes)
          cumsize = sizes.cumsum(0)
          cumsize_prev = torch.cat([torch.zeros(1, dtype=torch.long), cumsize[:-1]])
          rep_o_starts = torch.repeat_interleave(o_starts.long(), sizes)
          rep_cp = torch.repeat_interleave(cumsize_prev, sizes)
          flat_idx = rep_o_starts + torch.arange(sizes.sum(), dtype=torch.long) - rep_cp
          sat_idxs = _md.osm_to_sat_idxs[flat_idx].long()
          _sim[p].scatter_add_(0, sat_idxs, exp_weights)
    
        _city = "chicago"  # change for seattle
    
        _path = Path(f"~/scratch/similarities/{_save_prefix}/{_city}.pt").expanduser()
        _path.parent.mkdir(exist_ok=True)
        torch.save(_sim, _path)
        print(f"Saved {_sim.shape} to {_path}")
    
    _save = {
      "theta": learned_theta,
      "tag_keys": tag_keys,
      "train_history": train_history,
    }
    _path = Path(f"~/scratch/similarities/{_save_prefix}/weights.pt").expanduser()
    torch.save(_save, _path)
    print(f"Saved {len(tag_keys)} weights to {_path}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
