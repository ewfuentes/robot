# Fixed-embedding semantic representation ablations

`train_representation_ablation` prepares and trains frozen-embedding alternatives
to the structured-tag correspondence model. Each side of a pair is a 768-dimensional
`text-embedding-005` vector of one text. The panorama text is fixed per prepared
root by `prepare --pano-text`: the VLM landmark sentence (`descriptions`) or the
landmark's sorted literal `key=value` string (`tag_strings`). The OSM text is
chosen per `train --variant`: generated sentences (`descriptions`) or sorted
literal `key=value` text (`tag_strings`). The three paper variants are
sentence/sentence, sentence/KV, and KV/KV; KV/KV needs its own prepared root
(and its own panorama embedding run) because the panorama texts differ. The
shared `FixedEmbeddingClassifier` consumes `[pano, osm, pano * osm]`; it has no
learned tag encoder or engineered cross features. Its head is the same
Linear → BatchNorm → ReLU → Dropout → Linear head used by the structured model.
The original `classifier.*` checkpoint keys, including BatchNorm statistics,
remain compatible.

Comparing any variant against the structured model also changes the encoder,
cross features, and parameter count; it is not a format-only comparison.
Comparing variants against each other changes only the named text.

## Training

All locations are explicit arguments. In these examples, set the shell variables
to your own data locations; there is no default experiment directory or cloud
project. The baseline training YAML provides the label data directory, train and
validation cities, optimizer settings, classifier width/dropout, and seed.

```sh
bazel run //experimental/overhead_matching/swag/scripts:train_representation_ablation -- prepare \
  --root "$EXPERIMENT_DIR" --config "$BASELINE_CONFIG" \
  --osm-embeddings-dir "$OSM_EMBEDDINGS_DIR" \
  --osm-source-dir "$OSM_SOURCE_DIR" --pano-source-dir "$PANO_SOURCE_DIR" \
  --supplement-dir "$SUPPLEMENT_DIR" --pano-text descriptions
```

Inputs:

- `OSM_EMBEDDINGS_DIR`: aligned `keys.json`, `descriptions.npy`, and `tag_strings.npy`.
- `OSM_SOURCE_DIR`: the original `manifest.json` and `requests.jsonl` used to
  generate OSM descriptions. Preparation copies the request template for any
  exact parsed training tag bundles missing from the OSM embedding population.
- `PANO_SOURCE_DIR`: `<city>/embeddings/embeddings.pkl`, containing the original
  panorama landmark tags and descriptions.

Preparation verifies labels, landmark ordering, and difficulty selection against
the existing correspondence parser. It writes pair indices, panorama sentences,
input hashes, and a frozen copy of the baseline configuration. If supplemental
OSM bundles are needed, submit the generated requests using your existing batch
workflow and collect a key-to-sentence mapping as `SUPPLEMENT_DIR/sentences.json`.
This driver does not submit those generation requests.

Supplemental bundles exist because training pairs carry the OSM tags as the
baseline label parser read them back from the prompt, and that parser truncates
values at an unescaped `; ` or newline and strips trailing whitespace. Such a
bundle hashes to a key absent from the full-map `keys.json`, so a sentence and a
tag string are generated for the truncated bundle itself rather than for the
original entity. This keeps the ablation's view of those pairs identical to the
baseline's. On the paper training set this is 19 bundles and 73 of 194,936 pairs.

```sh
bazel run //experimental/overhead_matching/swag/scripts:train_representation_ablation -- embed \
  --root "$EXPERIMENT_DIR" --supplement-dir "$SUPPLEMENT_DIR"

bazel run //experimental/overhead_matching/swag/scripts:train_representation_ablation -- check \
  --root "$EXPERIMENT_DIR" --config "$BASELINE_CONFIG" \
  --osm-embeddings-dir "$OSM_EMBEDDINGS_DIR"

bazel run //experimental/overhead_matching/swag/scripts:train_representation_ablation -- train \
  --root "$EXPERIMENT_DIR" --config "$BASELINE_CONFIG" \
  --osm-embeddings-dir "$OSM_EMBEDDINGS_DIR" --variant descriptions
```

`embed` makes Vertex AI calls through the repository embedding helper using your
configured credentials/project/location. It preserves `text-embedding-005`,
`SEMANTIC_SIMILARITY`, 768 dimensions, and `auto_truncate=False`, with bounded
requests and resumable output. It also supports preparation with no supplemental
bundles. Repeat `train` with `--variant tag_strings` for the other model.
For KV/KV, prepare a second root with `--pano-text tag_strings`, run `embed`
there (its panorama texts differ, so they are embedded again), and train with
`--variant tag_strings`.
Training uses CUDA and the existing correspondence training/evaluation functions,
writes under `EXPERIMENT_DIR/<variant>`, and refuses to overwrite an existing run.
The best checkpoint is selected by validation ROC AUC. `check --checkpoint FILE`
can validate an archived checkpoint without starting training.

## Paper matrix export

`semantic_paper_eval` consumes separately prepared evaluation inputs:

- Baseline raw metadata with archived panorama landmark rows, OSM indices/tags,
  and a `pano_id_to_lm_rows` mapping.
- An `indices.npz` containing `pano` and `osm` indices into the supplied embedding
  arrays, ordered exactly like those archived rows/columns.
- The baseline similarity and raw cost matrices for the audit, plus the VIGOR
  dataset defining panorama and satellite-patch order.

Training preparation covers the configured train/validation pairs; it does not
create these full evaluation-city mappings. Retain all archived OSM columns,
including those outside the current satellite grid, because they affect
uniqueness weights.

```sh
bazel run //experimental/overhead_matching/swag/scripts:semantic_paper_eval -- audit \
  --dataset-path "$CITY_DATASET" --raw-metadata "$RAW_METADATA" \
  --audit-dir "$AUDIT_DIR" --reference-similarity "$BASELINE_SIMILARITY" \
  --reference-cost-matrix "$BASELINE_COST_MATRIX"

bazel run //experimental/overhead_matching/swag/scripts:semantic_paper_eval -- score \
  --dataset-path "$CITY_DATASET" --raw-metadata "$RAW_METADATA" \
  --audit-dir "$AUDIT_DIR" --variant descriptions --checkpoint "$CHECKPOINT" \
  --indices "$EVALUATION_INDICES" --pano-embeddings "$PANO_EMBEDDINGS" \
  --osm-embeddings "$OSM_EMBEDDINGS" --output-dir "$MATRIX_OUTPUT" --workers 12
```

Use `--landmark-version` if the dataset has multiple versions. The reference-cost
argument overrides the location recorded in archived metadata. Scoring defaults
to CUDA; `--device cpu` supports small checks. The audit verifies sampled baseline
scores with threshold 0.8, Hungarian matching, sum aggregation, uniqueness weights,
and dustbins. Scoring checks the audited dataset identity and writes
`cost_matrix.npy`, `similarity.pt`, and `complete.json`.

Parallel matching passes explicit metadata to spawned workers; each opens the
cost matrix as a read-only memory map. There is no mutable module-level scoring
state or dependency on a forked CUDA context. Use a separate output directory for
each checkpoint/input combination: completed matrices are reused and partial
inference is resumed. Existing external launch scripts must be updated to pass
these explicit arguments.

The resulting similarity matrix is input to the existing `calibrate_sigma` and
`evaluate_histogram_on_paths` tools. This adapter does not perform final path
evaluation or write paper tables.
