# Farfield data organization — `/data/farfield_matching/`

How to decide where a file goes on the data disk, so that (1) new datasets slot
in without inventing structure, (2) baselines can add inputs/models/outputs
without colliding with the pipeline's, and (3) release is a directory
predicate, not an archaeology project.

This is the source of truth; `/data/farfield_matching/ORGANIZATION.md` is a
pointer to it.

## The one rule

Organize by **lifecycle**, not by source or by project. Every file belongs to
exactly one lane, chosen by how it changes over time:

| Lane | Contents | Mutability |
|---|---|---|
| `datasets/` | frozen problem definitions (images, poses, map catalog) | append-only, checksummed |
| `raw_material/` | raw self-collected material (videos, sensor logs) + collection manifests | frozen at collect time |
| `artifacts/` | derived inputs a method consumes (landmark descriptions, pinhole faces, tracks) | versioned `v1, v2, …`, never mutated |
| `runs/` | experiment outputs (filter runs, evaluations, debug boards) | append-only campaigns |
| `models/` | checkpoints, one dir per model family, each with a `SOURCE.md` | append-only |
| `inbox/` | unprocessed drops awaiting ingest | transient |
| `archive/` | tarballs of frozen releases, retired/bad datasets | frozen |

Anything named `cache/` (at any depth) is disposable: never released, never
checksummed, safe to `rm -rf`. Hash-keyed content (e.g. `catalog_cache/`)
belongs in a cache — the hash is its manifest.

Release = `datasets/ + artifacts/ + selected runs/` (and `raw_material/` if we
choose to ship raw), minus every `cache/`. Third-party checkpoints ship as a
downloader script, not bytes.

**Release terms are per-dataset, not a property of the disk.** Much of what is
here is redistributable with attribution (self-collects are ours; Mapillary
imagery is CC-BY-SA; OSM is ODbL; NOAA ENC is public domain), but some of it is
not, and the restricted material sits in the normal lanes rather than in
quarantine. So a release is assembled by checking each dataset's terms, never by
taking a lane wholesale. Each dataset records its own terms in
`pipeline_metadata.json` — `upstream.license` where it comes from a third party,
`anonymization` where the constraint is about people in frame — and
`/data/farfield_matching/ORGANIZATION.md` lists the current restrictions.
**Anything derived from a restricted dataset inherits its restriction**,
including artifacts that carry pixels (pinhole faces, tracking crops, debug
boards) and any run output that embeds them.

Quarantine in a clearly-named top-level directory is still the right move for an
input that cannot be redistributed *at all* (e.g. satellite tiles for a
baseline). It is the wrong move for material that is redistributable under
conditions — that belongs in the lanes, labelled.

## Top-level target layout

```
/data/farfield_matching/
├── ORGANIZATION.md            pointer to this doc
├── STATUS.md                  live registry: one row per dataset (see below)
├── datasets/
│   ├── seattle/               ← mapillary_datasets/* move here unchanged
│   ├── folkestone_dover/
│   ├── boston_harbor_leg1/    ← boston legs become first-class datasets
│   ├── boston_harbor_leg2/
│   └── charles_river_20260727/
├── raw_material/
│   ├── boston_harbor_20260712/     videos/ logs/ sync.csv manifest.json
│   │                               process_legs.py blur_people.py checksums.sha256
│   ├── charles_river_20260727/
│   └── mapillary_manifests/        ← mapillary_datasets/_manifests/*
│                                     (incl. mount_offsets.json, vehicle_anchor.json)
├── artifacts/
│   └── <kind>/<dataset_id>/v<N>/   see the kind table below
├── runs/
│   └── <YYMMDD>_<slug>/            campaign (may span datasets)
│       ├── notes.md
│       └── <experiment>/manifest.json + outputs
├── models/
│   └── sam2/                       + SOURCE.md (url, sha256, license)
├── inbox/
└── archive/
    └── bad_trajectories/           ← mapillary_datasets/_bad_trajectories/
```

## `datasets/` — the frozen contract

One directory per **trajectory** (the unit the pipeline ingests). One flat
namespace regardless of source; the source lives in
`pipeline_metadata.json:source` and in `STATUS.md`, not in the path. This is
what lets an evaluation loop over `datasets/*` without knowing whether a
trajectory came from Mapillary, a ferry, or a sailboat.

The per-dataset contract is the one already documented in
[mapillary-dataset-creation.md](mapillary-dataset-creation.md) and enforced by
`//experimental/overhead_matching/swag/scripts:audit_dataset`. It is the same
for panoramic and perspective cameras — projection differences are carried by
`intrinsics.csv` (per-frame, because FOV varies within a trajectory) and
`pipeline_metadata.json:azimuth_convention`, never by directory structure:

```
datasets/<dataset_id>/
├── panorama/ -> frames        relative symlink; ingest requires this name
├── frames/                    images as captured (unrotated)
├── frames_gps.csv
├── intrinsics.csv             required for BOTH projections
├── extraction_log.csv         per-frame provenance (Mapillary creator ids — load-bearing
│                              for CC-BY-SA attribution at release; never drop it)
├── pano_id_mapping.csv
├── pipeline_metadata.json     projection, azimuth_convention, source, raw_material id,
│                              calibration (mount_offset_deg + how it was measured)
├── landmarks/
│   ├── v1.feather             merged catalog (canonical name — no dataset prefix;
│   ├── v1_trimmed.feather      the path already carries the id)
│   ├── sources/               per-extract feathers before merging
│   ├── PROVENANCE.json        bbox, PBFs, ENC cells
│   └── landmark_coverage.png
├── trajectory.png
├── gps_timelapse.mp4
└── checksums.sha256           written once, when the dataset is declared frozen
```

Rules:

* **A dataset dir is append-only after freeze.** New landmark catalog → new
  `v<N>` next to the old one, update `PROVENANCE.json`, regenerate
  `checksums.sha256`. Never overwrite `v<N>` in place — we have been burned by
  a regenerated-in-place matrix whose settings could no longer be inferred from
  its filename.
* **Nothing that a method computes goes in here.** Landmark descriptions,
  tracks, filter outputs all go to `artifacts/` or `runs/`. The line is NOT
  raw-vs-derived — everything in `datasets/` is derived from `raw_material/`.
  The test: **if this file changes, whose numbers change — everyone's, or one
  method's?** The OSM/ENC catalog is in here because it is the map the task is
  defined against: change it and every method's results change meaning, exactly
  like ground-truth GPS. An LLM's frame descriptions are an artifact because
  only the methods built on them care. (A baseline that brings its own map —
  satellite tiles, say — carries that map as *its* input in `artifacts/`, not
  as a reason to evict the shared one.)
* **Every dataset is self-contained.** If several datasets share a map catalog
  (the boston legs share one harbor extraction), duplicate the feather into
  each `landmarks/` rather than symlinking across directories — feathers are
  ~200 MB and a release must not chase links out of the dataset dir.
* **Dataset ids** are `[a-z0-9_]+`, globally unique, and permanent — run
  manifests reference them, so renaming one after runs exist is forbidden (add
  a symlink if truly forced). Self-collects carry a date
  (`charles_river_20260727`); a multi-leg collect yields one dataset per leg
  (`boston_harbor_leg1`). Frame/pano ids must be globally unique **across**
  datasets, not just within one.
* No dot-files, no AppleDouble `._*` junk — `vigor_dataset.iterdir()` ingests
  them as phantom panoramas; the audit flags this.

### Current state (2026-08-16)

The 14 kept Mapillary datasets already meet this contract uniformly. Known
drift to clean up:

* Four datasets (`seattle`, `mississippi_rural`, `nyc_east_river`,
  `nyc_inner_harbor`) keep dataset-prefixed feathers with `v1.feather` as a
  symlink to them; the rest have real files under the canonical names. Either
  is readable; converge on real files named `v1*.feather` when convenient.
* Four datasets carry a `trimmed_frames/` dir. This is `trim_dataset`'s
  reversible audit trail — the **dropped** frames plus the pre-trim CSVs;
  `frames/` is always the live set (fukuoka kept 165 of ~393 because the
  operator swung the camera, so its dropped pile is bigger than its live set —
  that's expected, not drift). Keep them; exclude from release.

The boston legs predate the contract (no `intrinsics.csv`,
`pipeline_metadata.json`, `extraction_log.csv`, `pano_id_mapping.csv`;
landmarks shared at the collect level) and get backfilled when they migrate.
`charles_river_20260727` is still an un-ingested zip with its own filename
convention.

## `raw_material/` — what datasets are made from

One directory per physical collection event, named `<place>_<YYYYMMDD>`.
Holds what `datasets/` deliberately excludes: original videos, raw sensor
logs, sync tables, and the **collect-local processing scripts** frozen beside
the data they processed (`process_legs.py`, `blur_people.py`). This is the
only place standalone scripts are allowed on the data disk — everything else
must be a bazel target in the repo.

Each collect's `manifest.json` lists the dataset ids it produced; each
produced dataset's `pipeline_metadata.json` names its `raw_material/<id>`.
Mapillary has no raw imagery lane here (staging lives in
`~/scratch/mappilary/_raw`), but its collection manifests plus the
cross-dataset calibration registries (`mount_offsets.json`,
`vehicle_anchor.json`) live in `raw_material/mapillary_manifests/` — they are
records of collection, not of any one dataset.

## Conventions are part of the contract

A dataset's `pipeline_metadata.json` states frames and formulas, and those
statements are consumed by code in a different package. **The project-wide
register is [`conventions.md`](conventions.md); read it before writing or reading
a frame, sign, zero point or id format here.**

The equirect `azimuth_convention.formula` references **column_0**, while
`pano_geometry` — which every bearing consumer uses — puts zero at the **centre**
column. Both are correct; they are different zeros, exactly 180° apart. A
`mount_offset_deg` reasoned in the wrong one is half a turn out and nothing will
tell you: that is what happened to `pohang_canal_04`, and a near-identical error
put north at column 0 on six earlier Mapillary datasets. Both ingest writers now
carry a `mount_offset_frame` note stating which frame their formula is in.

## `artifacts/` — versioned derived inputs

`artifacts/<kind>/<dataset_id>/v<N>/` — kind first, then dataset. A kind is
one producer with one consumer contract, so kind-first keeps a whole pipeline
stage's output under one directory: sweeping or regenerating "all
frame_landmarks" is a single glob, and a new baseline extends the lane by
adding one new kind directory.

A kind is named for **what it promises its consumer**, and only that promise
is contractual — the internal layout belongs to the producer and may change
between versions. Current kinds:

| Kind | Promise to the consumer | Producer → consumer |
|---|---|---|
| `frame_landmarks/` | per-frame LLM landmark descriptions (what `ingest.py` reads) | LLM extraction → ingest/matching |
| `pinhole_images/` | four 90° faces per frame (equirect datasets only) | renderer → LLM extraction |
| `object_tracks/` | the tracking pipeline's working tree: m-stage products (SAM2 state, track JSONs, m11 exports) with their debug boards; internal rNNN run ids | SAM2 tracking → matching/filter |
| `landmark_matching/` | staged run-artifact JSONs from `run_filter_pipeline` (self-provenanced: git_hash, config, yaw calibration, per-frame products) | matching pipeline → viewer/filter |
| *(future)* `<baseline_input>/` | whatever a baseline precomputes per dataset | baseline precompute → baseline eval |

Deliberately **not** kinds: description embeddings (`embeddings.pkl` feeds
only the cosine-similarity matcher in `semantic_similarity.py`; it becomes a
kind only if that matcher survives the LLM-chunked one), and batch-API
intermediates like `sentence_requests/` — no stage reads those back, but they
are **not** deletable scratch: every request line carries the system prompt
verbatim, so they are the artifact's only record of the prompt text it was
built with, and the only thing `request_sha256` can be recomputed against.
Keep them with the artifact; archive rather than drop. The one existing extraction
(`panorama_landmarks/boston_harbor_leg1`, in the old overhead_matching
batch-job layout that `ingest.py` globs through) migrates as
`frame_landmarks/v1/` with a manifest noting the legacy layout; when
extraction runs on the Mapillary datasets, write a clean format and call it
what it is — the version bump is the migration path.

Every `v<N>/` contains a `manifest.json`:

```json
{
  "kind": "frame_landmarks",
  "dataset": "boston_harbor_leg1",
  "version": "v1",
  "generator": "//experimental/overhead_matching/swag/landmark_filtering:...",
  "git_commit": "9849411",
  "config": { "prompt": "osm_tags_farfield", "model": "gemini-3.1-pro", "resolution": 2048, "detail": "ULTRA_HIGH" },
  "inputs": ["datasets/boston_harbor_leg1", "artifacts/pinhole_images/boston_harbor_leg1/v1"],
  "created": "2026-07-28",
  "notes": "legacy batch-job layout; ingest.py globs sentences/results/*/prediction-*/predictions.jsonl"
}
```

### Required: an output records everything needed to recreate it

**Every artifact and every run must carry, with its outputs, the complete set of
inputs and settings needed to reproduce it.** Not "most of", and not "whatever
the producer happened to find convenient to print". If a number changed the
output, it belongs in the record. This is a hard requirement, not a
best-practice aspiration: an artifact that cannot be traced back to how it was
made is not releasable, and it cannot be compared against a later version.

The record must pin all of:

| what | why it is not optional |
|---|---|
| **generator** — bazel target, not a prose description | so the producer can actually be re-run |
| **git commit** of the workspace | the code is a setting |
| **every input path**, dataset and artifact versions included | `v1` vs `v2` of an input silently changes the output |
| **every knob that affects the output** — thresholds, resolutions, batch sizes, gates | anything a flag can change |
| **model identity** for every model involved, including secondary ones | an extraction has both a VLM *and* an embedding model; naming only the first hides half the pipeline |
| **prompt content, by digest** — not just its name | prompt names are lookup keys whose text can be edited in place, so two artifacts can claim identical provenance from different prompts |
| **coverage / completeness** — how many inputs produced usable output | see below |

Two failure modes this exists to prevent, both observed in this project:

- **A name is not a version.** `"prompt": "osm_tags_farfield"` names a key in
  `SYSTEM_PROMPTS`. Editing that key's text changes every future extraction
  while the manifest keeps saying the same thing. Record a digest over what was
  actually sent (`request_sha256` over the request JSONL), which pins prompt,
  resolution and input set together.
- **Partial success is indistinguishable from a quiet environment.** A Vertex
  batch job reports success at the job level while individual requests fail;
  `boston_harbor_leg2` lost 23 of 236 frames to transient TPU errors. Nothing
  downstream objects — `ingest` skips a frame with no prediction with a bare
  `continue` — so tracking simply sees 10% of the leg as containing no objects.
  A manifest that records only settings would look identical for a complete and
  a 90%-complete artifact. So the count of inputs that produced usable output is
  part of the record, and an incomplete artifact says so in its own manifest.

Corollary for reproducibility: when a producer's provenance lives in an
intermediate, that intermediate is part of the artifact and the manifest must
say so. `request_sha256` is computed over `sentence_requests/`, and those
request lines also carry the prompt text verbatim — which is what makes the
extraction reproducible at all, since `"prompt": "osm_tags_farfield"` names a
key whose text is edited in place and `git_commit` pins the tree rather than
the working copy that ran. Delete the requests and the artifact can no longer
say what it asked the model, nor be checked against its own digest. A record
that silently becomes uncheckable is worse than one that admits its limits.

The manifest, not the filename, is the record of how something was generated —
kinds need it because nothing in their path says which prompt/model/config
produced them. Versions are immutable: to change settings, mint `v<N+1>` and
leave `v<N>` alone. For working-tree kinds whose producer manages its own run
ids (`object_tracks`' rNNN, `landmark_matching`'s per-config JSONs), the
immutability lives on those internal ids — never regenerate one in place with
different settings — and `v<N+1>` marks the producer itself changing shape. These artifacts are expensive (LLM extraction is real
money), so they are release candidates in their own right — precomputed
artifacts shipped next to the datasets so others can skip the token bill.

Contrast with caches: the map-side text embeddings
(`catalog_cache/catalog_<hash>.pkl`) stay a cache, not an artifact — the hash
key *is* the manifest, so it can never silently go stale and needs no version
ceremony. Worth including in a release to save others the tokens, but always
regenerable and deletable.

## `runs/` — experiment outputs

`runs/<YYMMDD>_<slug>/` per campaign, matching the existing
`260813_m4_bodyframe_leg1` convention. Campaigns may span datasets and
methods; each experiment subdir carries a `manifest.json` naming the method,
dataset ids, artifact versions consumed, git commit, and config. A campaign
gets a `notes.md` while it's live.

The dividing line vs `artifacts/`: if anything downstream consumes it as an
input, it's an artifact; if it's terminal (metrics, figures, debug HTML,
particle checkpoints), it's a run. Tracking and matching product trees are
artifacts even though they carry their own debug boards — a kind's internal
layout is the producer's, and the boards travel with the products they
explain. What stays in `runs/` is the filter/eval campaigns: truth, metrics,
checkpoints, notes — outputs nothing re-consumes as an input.

Localization evaluations — ours and baselines alike — share this one lane —
a baseline result and an ours result in the same campaign directory is exactly
what makes the comparison table easy to assemble later.

## `models/`

Flat: `models/<name>/` per model family (today just `sam2/`, which
`object_tracking` already references in place). Each dir gets a `SOURCE.md`:
URL, sha256, license, download date. Third-party checkpoints are never
redistributed — the release ships a downloader script instead (downloaders
live in `~/scratch`, per the COSMOS precedent). If we start training our own
models for baselines, revisit this section then (they'd be saved via
`common/torch/load_and_save_models.py` so the git commit rides along) — no
structure for hypothetical needs before that.

## `STATUS.md` — the registry

One table at the root, one row per dataset: id, source, projection, frames,
km, heading quality (`heading_reliable` / `heading_sources_disagree` per the
projection-specific rule), landmarks version, mount offset status, and a
free-text "usable for" column. This replaces scanning directories to know
what exists. Update it in the same session that creates or freezes a dataset.

## Where does this file go? (decision procedure)

1. Needed to *pose* the localization problem (image, pose, map)? → `datasets/`
2. Raw material from a physical collect, or a collection manifest? →
   `raw_material/`
3. Computed from a dataset and consumed downstream as an input? →
   `artifacts/<kind>/<dataset>/v<N>/` + manifest
4. Terminal output of an experiment? → `runs/<YYMMDD>_<slug>/`
5. A checkpoint? → `models/<name>/` + `SOURCE.md`
6. Regenerable and hash-keyed, or an intermediate nothing reads back? → a
   `cache/` dir near its consumer (or delete it)
7. Not yet processed? → `inbox/`. Dead? → `archive/`.
8. A venv, a repo checkout, an API token? → **not on this disk.**

## Migration from the current tree

Ordered cheapest-first; 1–4 are mechanical, 5 is the only invasive one.

1. **Hygiene:** move `sam3_env/` off the data disk (it's an environment, not
   data); move `charles_river_20260727_dataset.zip` → `inbox/`,
   `boston_harbor_dataset.tar.zst` → `archive/`; purge `._*`/`.DS_Store`
   (`find /data/farfield_matching -name '._*' -o -name '.DS_Store' | xargs rm`).
2. **Datasets flatten:** move `mapillary_datasets/_manifests` →
   `raw_material/mapillary_manifests/` and `_bad_trajectories` →
   `archive/bad_trajectories/`, then `mv mapillary_datasets datasets` and
   `ln -s datasets mapillary_datasets` so `~/scratch/mappilary` scripts and
   existing docs kept working until their output root was updated. (Both since
   done; that directory is retired.)
3. **Mapillary drift cleanup:** converge feather names on `v1*.feather` real
   files; resolve the four inconsistent `trimmed_frames/` dirs (see Current
   state above).
4. **New collects use the scheme natively:** charles river ingests as
   `raw_material/charles_river_20260727/` + `datasets/charles_river_20260727/`.
5. **Boston harbor split** (defer to when legs 2/3 are processed): raw
   (`videos/ logs/ sync.csv manifest.json process_legs.py blur_people.py`) →
   `raw_material/boston_harbor_20260712/`; each `processed/legN` becomes
   `datasets/boston_harbor_legN/` meeting the full contract (add
   `intrinsics.csv` + `pipeline_metadata.json`; copy the harbor catalog into
   each leg's `landmarks/`); `panorama_landmarks/boston_harbor_leg1` →
   `artifacts/frame_landmarks/boston_harbor_leg1/v1/` (+ manifest; keep
   `sentence_requests/`, 449 MB — no stage reads it, but it holds the prompt
   text and backs `request_sha256`); `object_track_runs/` →
   `artifacts/object_tracks/<dataset>/v1/` (the whole working tree — products
   and boards together); `filter_runs/` → `artifacts/landmark_matching/<dataset>/v1/`;
   `evaluations/` → `runs/`.
6. **Pinhole faces:** new ones go to `artifacts/pinhole_images/<id>/v1/`;
   existing ones under `/data/overhead_matching/datasets/pinhole_images/`
   stay until a convenient moment, then move + symlink.

**Status (2026-08-16): all six steps are executed** — 18 datasets live under
`datasets/` (audits clean, checksummed), the boston shell has been deleted,
and all repo path constants were repointed to the new lanes (m-scripts →
`artifacts/object_tracks/.../v1`, `artifacts/frame_landmarks/.../v1`,
`datasets/boston_harbor_leg1`, `raw_material/.../videos`).
`mapillary_datasets` was a compat symlink for `~/scratch/mappilary`; **that
directory is retired as of 2026-08-19** — the collection code moved in-repo
2026-08-17 and the last docs and helper script followed, so nothing in this repo
reads from it. The symlink is now only for old recorded artifact paths.
Pinhole faces for the five equirect datasets moved to
`artifacts/pinhole_images/<id>/v1/` (+manifests); the audit no longer checks
them — they are an artifact, not dataset contract — and old-pipeline scripts
keep the old `/data/overhead_matching` base for the old-project sets that
stayed there. Still pending: the boston leg videos
(`raw_material/boston_harbor_20260712/videos/`, read directly by m1/m2/m3)
and the orchestrator's stage-5 output root (**done**: the orchestrator moved
in-repo 2026-08-17 with lane-correct defaults, and `~/scratch/mappilary` is
retired — see below).
