"""One-time migration of the farfield data root into the REORG.md layout.

Prints a complete move manifest and exits. `--apply` executes exactly that
manifest and refreshes the index chain. Nothing is deleted unless it is
regenerable (caches, generated index pages) or a verified byte-identical
copy, and every move that could break a recorded path leaves a relative
symlink behind.

What it does, in order:

1. **runs/ becomes experiment directories.** Every top-level entry under
   `runs/` ends up as `<date>_<experiment>/` holding runs plus a required
   `experiment.md`. The six `260819_<leg>` dirs are folded into
   `260819_final/` as `<leg>_pre_selection_charge/`: they are NOT duplicates
   of their `260819_final` namesakes (only `truth.jsonl` matches — the
   checkpoints and `particle_history_sha256` differ and their manifests
   predate three ProposalConfig fields), they are the prior generation, and
   the newer runs beat them (mtw leg3: 78 m vs 190 m final). The loose logs
   and figures at the `260819_final` root move into `logs/` and `figures/`.

2. **The 28 localization runs hiding in the artifacts lane surface.**
   `artifacts/object_tracks/<ds>/v1/m3_tracks/runs/<rNNN>/localization_run_*`
   is the runs lane duplicated inside the artifacts lane. They move to
   experiment dirs grouped by the investigation their names describe, each
   leaving a `moved_to.txt` pointer behind. The `localization_export_*` dirs
   STAY: an export is a per-tracking-run artifact.

3. **Catalogs leave the frozen datasets lane** for
   `artifacts/catalogs/<dataset>/`, carrying their provenance sidecars, the
   collection-era `PROVENANCE.json`, the `sources/` inputs and the coverage
   plot. Ten recorded `matching/settings.json` files name the old absolute
   paths, so each moved feather leaves a relative symlink behind. Verified
   byte-identical siblings (six datasets have 2-3 identical trims; four
   carry a legacy `<ds>_osm_enc_v1*` name for the same bytes) collapse to
   one file plus relative symlinks — content the trim tool now refuses to
   create in the first place. `catalog_cache/` is deleted: it is
   regenerable and keyed on an anchor that may have moved.

4. **The dead `landmark_matching` lane is archived.** One dataset, untouched
   since it was written; matching for the other nine lives with its
   tracking run, which is where the contract puts it.

5. **Documentation of record is regenerated**: `ORGANIZATION.md` rewritten
   to the real layout, `STATUS.md` marked stale so the status tool
   regenerates it (7 of 27 datasets are undocumented today), `inbox/`
   drained into `raw_material/`, `models/SOURCE.md` written.

6. **The index chain is generated** so a static server at the root reaches
   everything by clicking.

    bazel run //experimental/overhead_matching/swag/farfield/dataset_tools:migrate_data_root
    bazel run //...:migrate_data_root -- --apply
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from experimental.overhead_matching.swag.farfield import paths as paths_lib
from experimental.overhead_matching.swag.farfield import provenance
from experimental.overhead_matching.swag.farfield.viewers import indexes

# The six prior-generation leg runs, and the 260819_final run they precede.
PRIOR_GENERATION = {
    "260819_boston_harbor_leg1": "boston_leg1_r004",
    "260819_boston_harbor_leg2": "boston_leg2",
    "260819_boston_harbor_leg3": "boston_leg3",
    "260819_mount_washington_20260815_leg1": "mtw_leg1",
    "260819_mount_washington_20260815_leg2": "mtw_leg2",
    "260819_mount_washington_20260815_leg3": "mtw_leg3",
}

# Where a hidden localization run belongs, by what its name says it was.
# The two pohang runs the 260819_final symlinks point at are part of THAT
# campaign, so they land there as real directories and the symlinks go.
CAMPAIGN_RUNS = {
    ("pohang_canal_04", "r002_v5_seamfix", "localization_run_m9_uniform"):
        ("260819_final", "pohang_canal_04"),
    ("pohang_canal_04", "r002_v5_seamfix", "localization_run_m9_truthinit"):
        ("260819_final", "pohang_canal_04_truthinit"),
}

EXPERIMENT_NOTES = {
    "260813_m4_bodyframe_leg1": None,   # already carries notes.md
    "260819_final": """# 260819 whole-map localization campaign

Uniform-prior whole-map runs across every leg that had tracks, matches and
a mount offset. This is the campaign the repo-side write-up
(`260819_localization_campaign.md`, now `experiment.md` here) describes.

## Status

Complete. Five of seven legs produced data that localizes; the two boston
leg1 exports disagree by ~80x and that was bisected to the tracks, not the
matcher.

## Contents

- `<leg>/` - the campaign runs, selection charge ON.
- `<leg>_pre_selection_charge/` - the PRIOR GENERATION of the same legs,
  kept because it is the before-half of the selection-charge evidence.
  Their manifests predate `evidence_gate_selection_charge`,
  `min_tracklets_for_injection` and `init_max_wait_keyframes`; their
  `particle_history_sha256` differs from the newer runs, so these are
  distinct executions, not copies. mount_washington leg3 is the clearest
  comparison: 190 m final before, 78 m after.
- `pohang_canal_04{,_truthinit}/` - moved here from the artifacts lane,
  where they were reachable only through absolute symlinks. The truth-init
  one is a DIAGNOSTIC (basin of attraction), never a result.
- `logs/`, `figures/` - the per-run console logs and two figures that used
  to sit loose in this directory.
""",
    "260820_extent_sigma": """# Extent-aware bearing sigma

Does giving an extended landmark a bearing sigma from its angular extent,
rather than a point-object sigma, fix the runs where confident matches sat
far off their measured bearing?

## Status

Shipped. This is the change that produced pohang's first uniform
convergence. These runs are the boston leg3 / charles_river arm: the
control (`extsigma_uniform`) plus the two proposal-sigma variants.

## Contents

Recovered from `artifacts/object_tracks/<dataset>/v1/m3_tracks/runs/r001_v4/`,
where they were written before `runs/` was the home for filter runs. Each
source directory keeps a `moved_to.txt` pointer.
""",
    "260820_pohang_taxonomy": """# Pohang match-failure taxonomy

Why did pohang's confident matches land on the wrong map rows, and which of
the four candidate root causes actually binds? Each root cause was tested in
isolation, so the run names are the treatment: `extsigma`, `neargate`,
`expansion` (category expansion), `stacked`, `v4final`, plus seed repeats
(`_s1`, `_s2`) and the proposal-sigma orderings.

## Status

Complete. Extent-sigma shipped; category expansion was a NO-GO (proposal
tie-truncation, 22-29 km); the near gate shipped. The oracle-matcher bound
of 5613 m is what proved the geometry was binding rather than the matcher.

## Contents

Recovered from `artifacts/object_tracks/pohang_canal_04/v1/m3_tracks/runs/`
(`r001_v5` and `r002_v5_seamfix`), where filter runs used to be written.
Each source directory keeps a `moved_to.txt` pointer. `*_truthinit` runs are
DIAGNOSTIC instruments (basin of attraction), never results - only the
uniform-prior runs are evaluations.
""",
}

ORGANIZATION_MD = """# /data/farfield_matching

The data root for far-field cross-view geolocalization. The authority on this
layout is `docs/farfield/datasets.md` in the robot repo; this file is the
on-disk short version, and `farfield/paths.py` is the code that resolves it.

```
datasets/<dataset>/          frozen problem definitions
artifacts/<kind>/<dataset>/<version>/    derived products, each + manifest.json
artifacts/catalogs/<dataset>/            map catalogs (feather + provenance)
runs/<experiment>/<run>/     localization experiments, each + experiment.md
models/<family>/             weights (+ SOURCE.md)
raw_material/                source material (video, GPX, collection scratch)
archive/                     retired datasets and tarballs
```

## The rules that matter

1. **`datasets/` is frozen.** No stage writes into it. A dataset's
   `pipeline_metadata.json` mount_offset block is written by exactly one
   tool, `dataset_tools:publish_mount_offset`, which regenerates
   `checksums.sha256`.
2. **Every artifact names its inputs.** Each `artifacts/<kind>/<ds>/<vN>/`
   carries a `manifest.json` with the generator, git commit, argv, resolved
   inputs and config. An artifact without one is a bug.
3. **Versions mean different content.** A `vN` whose bytes equal an existing
   sibling is versioning noise; the tools now refuse to create one. Where
   the history already contains such pairs, the duplicates are relative
   symlinks to the one real file.
4. **Every directory under `runs/` is an experiment** with an
   `experiment.md` saying what is being explored, its status, and what was
   concluded. Actual runs are its children.
5. **Everything is browsable.** Every stage refreshes the index chain, so a
   static server at this root reaches every page by clicking:

       cd /data/farfield_matching && python3 -m http.server 8935

## Compatibility pointers

Some `datasets/<ds>/landmarks/*.feather` paths are relative symlinks into
`artifacts/catalogs/`, because ten `matching/settings.json` files recorded
the old absolute paths and recorded provenance must keep resolving. They can
go once nothing references them.

## What is deletable

`catalog_cache/` anywhere (regenerable, keyed on the ENU anchor), generated
`index.html` pages, and `artifacts/*/index.html.bak`. Everything else is
either an input or a record.
"""

MODELS_SOURCE_MD = """# Model weights

| path | what | source |
|---|---|---|
| `sam2/sam2.1_hiera_large.pt` | SAM 2.1 Hiera-Large image/video segmentation checkpoint | Meta AI SAM 2 release (`facebook/sam2.1-hiera-large`), Apache-2.0 |

Weights are inputs, not artifacts: they are never regenerated here. The
tracking stage takes `--checkpoint` explicitly (there is no default) and
records the path it used in the run's `run_meta.json`.
"""


class Plan:
    """A list of operations, printable before anything happens."""

    def __init__(self):
        self.ops = []

    def add(self, kind, src, dst=None, note="", bytes_=0):
        self.ops.append({"kind": kind, "src": str(src),
                         "dst": str(dst) if dst else "",
                         "note": note, "bytes": bytes_})

    def show(self):
        """Grouped by kind for reading; operations EXECUTE in plan order,
        which is why blocking symlinks are planned before the moves that
        take their paths."""
        by_kind = defaultdict(list)
        for op in self.ops:
            by_kind[op["kind"]].append(op)
        total = sum(op["bytes"] for op in self.ops)
        for kind in ("mkdir", "write", "move", "symlink", "pointer",
                     "delete"):
            ops = by_kind.get(kind, [])
            if not ops:
                continue
            print(f"\n=== {kind.upper()} ({len(ops)}) "
                  + "=" * (50 - len(kind)))
            for op in ops:
                line = f"  {op['src']}"
                if op["dst"]:
                    line += f"\n      -> {op['dst']}"
                if op["note"]:
                    line += f"\n      ({op['note']})"
                print(line)
        print(f"\n{len(self.ops)} operations, {total / 1e9:.2f} GB moved or "
              f"deleted")
        deletes = by_kind.get("delete", [])
        if deletes:
            print("\nDELETIONS (each must be regenerable or a verified "
                  "byte-identical copy):")
            for op in deletes:
                print(f"  {op['src']}\n      reason: {op['note']}")


def du(path: Path) -> int:
    if not path.exists() or path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*")
               if p.is_file() and not p.is_symlink())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan_runs(root: Path, plan: Plan) -> None:
    runs = root / "runs"
    final = runs / "260819_final"

    # 1a. The prior-generation legs fold into the campaign they precede.
    for old, newer in PRIOR_GENERATION.items():
        src = runs / old
        if not src.is_dir():
            continue
        dst = final / f"{newer}_pre_selection_charge"
        plan.add("move", src, dst,
                 "prior generation: manifests predate 3 ProposalConfig "
                 "fields and particle_history_sha256 differs, so NOT a copy",
                 du(src))

    # 1b. Loose logs and figures at the campaign root.
    if final.is_dir():
        logs = [p for p in final.iterdir() if p.suffix == ".log"]
        figs = [p for p in final.iterdir() if p.suffix == ".png"]
        for p in sorted(logs):
            plan.add("move", p, final / "logs" / p.name, "", du(p))
        for p in sorted(figs):
            plan.add("move", p, final / "figures" / p.name, "", du(p))

    # 1c. The campaign write-up becomes the experiment's own experiment.md,
    # replacing pointers that named the prior-generation dirs.
    campaign = runs / "260819_localization_campaign.md"
    if campaign.exists():
        plan.add("move", campaign, final / "campaign_writeup.md",
                 "kept verbatim; its run-directory pointers named the "
                 "prior-generation dirs and are corrected by experiment.md",
                 du(campaign))

    # 1d. Every experiment dir gets an experiment.md.
    for name, body in EXPERIMENT_NOTES.items():
        if body is None:
            continue
        plan.add("write", runs / name / "experiment.md",
                 note="experiment description (required by the contract)")


def hidden_runs(root: Path):
    """(dataset, tracking_run, run_name, path) for every localization run
    written inside the artifacts lane."""
    out = []
    base = root / "artifacts" / "object_tracks"
    for path in sorted(base.glob("*/*/m3_tracks/runs/*/localization_run_*")):
        if not path.is_dir() or path.is_symlink():
            continue
        out.append((path.parents[4].name, path.parent.name, path.name, path))
    return out


def experiment_for(dataset: str, run_name: str) -> str:
    if dataset == "pohang_canal_04":
        return "260820_pohang_taxonomy"
    return "260820_extent_sigma"


def plan_hidden_runs(root: Path, plan: Plan) -> None:
    runs = root / "runs"

    # The campaign's two absolute symlinks into the artifacts lane come FIRST:
    # their targets become real directories at exactly those paths below, and
    # a hardened static server refuses to follow them anyway. Operations are
    # applied in plan order, so this must precede the moves.
    for link in sorted((root / "runs").glob("*/*")):
        if link.is_symlink():
            plan.add("delete", link,
                     note="absolute symlink into the artifacts lane; its "
                          "target becomes a real directory at this path")

    for dataset, tracking_run, name, path in hidden_runs(root):
        key = (dataset, tracking_run, name)
        if key in CAMPAIGN_RUNS:
            experiment, run_dir_name = CAMPAIGN_RUNS[key]
        else:
            experiment = experiment_for(dataset, name)
            suffix = name[len("localization_run_"):]
            run_dir_name = f"{dataset}_{tracking_run}_{suffix}"
        dst = runs / experiment / run_dir_name
        plan.add("move", path, dst, "", du(path))
        plan.add("pointer", path / "moved_to.txt", dst,
                 "left behind so the tracking run still points at its "
                 "filter runs")


def plan_catalogs(root: Path, plan: Plan) -> None:
    for landmarks in sorted((root / "datasets").glob("*/landmarks")):
        dataset = landmarks.parent.name
        dst_dir = root / "artifacts" / "catalogs" / dataset

        # Group REAL FILES by content: one stays real, the rest become
        # relative symlinks to it. Pre-existing symlinks are re-pointed, never
        # treated as dedup candidates -- four datasets already had
        # `v1.feather -> <ds>_osm_enc_v1.feather`, and taking the symlink as
        # the keeper deletes the file it points at and leaves a cycle. A
        # symlink is a name for content, not a copy of it.
        existing_links = {}
        by_digest = defaultdict(list)
        for feather in sorted(landmarks.glob("*.feather")):
            if feather.is_symlink():
                existing_links[feather] = os.readlink(feather)
                continue
            by_digest[sha256(feather)].append(feather)

        for digest, group in sorted(by_digest.items()):
            # Prefer the canonical vN name as the real file.
            group.sort(key=lambda p: (("_osm_enc_" in p.name), len(p.name)))
            keeper, rest = group[0], group[1:]
            plan.add("move", keeper, dst_dir / keeper.name, "", du(keeper))
            plan.add("symlink", landmarks / keeper.name,
                     f"../../../artifacts/catalogs/{dataset}/{keeper.name}",
                     "recorded matching/settings.json paths must keep "
                     "resolving")
            for extra in rest:
                plan.add("delete", extra,
                         note=f"byte-identical to {keeper.name} "
                              f"(sha256 {digest[:12]}); replaced by a "
                              f"symlink, so every recorded path resolves")
                plan.add("symlink", dst_dir / extra.name, keeper.name,
                         "identical content, kept as a name")
                plan.add("symlink", landmarks / extra.name,
                         f"../../../artifacts/catalogs/{dataset}/"
                         f"{extra.name}", "compatibility pointer")

        # Pre-existing symlinks: recreate them in the new lane pointing at
        # the same name, and leave the compat pointer behind.
        for link, target in sorted(existing_links.items()):
            plan.add("symlink", dst_dir / link.name, Path(target).name,
                     "was already a symlink in the datasets lane; the name "
                     "is preserved, the target it named is the real file")
            plan.add("delete", link,
                     note="pre-existing symlink, recreated in the catalogs "
                          "lane and replaced by a compat pointer here")
            plan.add("symlink", landmarks / link.name,
                     f"../../../artifacts/catalogs/{dataset}/{link.name}",
                     "compatibility pointer")

        # Sidecars, the collection-era record, sources, coverage plot.
        for pattern in ("*.provenance.json", "PROVENANCE.json",
                        "landmark_coverage.png", "positive_set_*.json"):
            for extra in sorted(landmarks.glob(pattern)):
                plan.add("move", extra, dst_dir / extra.name, "", du(extra))
        sources = landmarks / "sources"
        if sources.is_dir():
            plan.add("move", sources, dst_dir / "sources",
                     "catalog inputs travel with the catalog", du(sources))
        cache = landmarks / "catalog_cache"
        if cache.is_dir():
            plan.add("delete", cache,
                     note="regenerable; its key includes the ENU anchor, "
                          "which the dataset's frames determine",
                     bytes_=du(cache))
        plan.add("write", dst_dir / "manifest.json",
                 note="catalogs lane provenance")


def plan_housekeeping(root: Path, plan: Plan) -> None:
    # 4. The dead matching lane.
    dead = root / "artifacts" / "landmark_matching"
    if dead.is_dir():
        plan.add("move", dead, root / "archive" / "landmark_matching_lane",
                 "one dataset, superseded by per-tracking-run matching/",
                 du(dead))

    # 5. Documentation of record and the inbox.
    plan.add("write", root / "ORGANIZATION.md", note="rewritten to the real layout")
    plan.add("write", root / "models" / "SOURCE.md", note="required by the standard")
    plan.add("write", root / "STATUS.md.stale", note="marker: regenerate via dataset_status_table")
    inbox = root / "inbox"
    if inbox.is_dir():
        for entry in sorted(p for p in inbox.iterdir() if p.name != ".keep"):
            plan.add("move", entry, root / "raw_material" / entry.name,
                     "inbox drains; its datasets are built", du(entry))

    # Generated-page cruft.
    for bak in sorted(root.rglob("index.html.bak")):
        plan.add("delete", bak, note="stale copy of a generated index page")


def build_plan(root: Path) -> Plan:
    plan = Plan()
    plan_runs(root, plan)
    plan_hidden_runs(root, plan)
    plan_catalogs(root, plan)
    plan_housekeeping(root, plan)
    return plan


def apply_plan(root: Path, plan: Plan) -> None:
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    for op in plan.ops:
        src, dst = Path(op["src"]), Path(op["dst"]) if op["dst"] else None
        kind = op["kind"]
        if kind == "move":
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        elif kind == "symlink":
            src.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink() or src.exists():
                continue
            src.symlink_to(op["dst"])
        elif kind == "pointer":
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(
                f"This filter run moved to the runs lane on {stamp}:\n"
                f"  {op['dst']}\n\n"
                f"Filter runs live in runs/<experiment>/; exports and "
                f"matching stay with their tracking run.\n")
        elif kind == "delete":
            if src.is_symlink() or src.is_file():
                src.unlink(missing_ok=True)
            elif src.is_dir():
                shutil.rmtree(src)
        elif kind == "write":
            src.parent.mkdir(parents=True, exist_ok=True)
            write_generated(root, src, stamp)


def write_generated(root: Path, target: Path, stamp: str) -> None:
    name = target.name
    if name == "experiment.md":
        body = EXPERIMENT_NOTES.get(target.parent.name)
        if body:
            target.write_text(body)
    elif name == "ORGANIZATION.md":
        target.write_text(ORGANIZATION_MD)
    elif name == "SOURCE.md":
        target.write_text(MODELS_SOURCE_MD)
    elif name == "STATUS.md.stale":
        target.write_text(
            f"STATUS.md was last hand-edited before the {stamp} migration "
            f"and documents 20 of 27 datasets.\nRegenerate it:\n\n"
            f"  bazel run //experimental/overhead_matching/swag/farfield/"
            f"dataset_tools:dataset_status_table -- \\\n"
            f"      --dataset_path {root}/datasets/*/ --output "
            f"{root}/STATUS.md\n\n"
            f"Delete this marker once that has been done.\n")
    elif name == "manifest.json":
        dataset = target.parent.name
        provenance.write(
            target.parent,
            generator="farfield/dataset_tools/migrate_data_root.py",
            inputs={"moved_from": f"datasets/{dataset}/landmarks"},
            config={"lane": "catalogs"},
            extra={"kind": "catalogs", "dataset": dataset},
            notes="Catalogs are derived products and moved out of the frozen "
                  "datasets lane by the one-time data migration. Identical "
                  "sibling versions are relative symlinks to the one real "
                  "file.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=None,
                        help=f"default: ${paths_lib.ROOT_ENV_VAR} or "
                             f"{paths_lib.DEFAULT_ROOT}")
    parser.add_argument("--apply", action="store_true",
                        help="execute the manifest (default: print it)")
    parser.add_argument("--skip_indexes", action="store_true",
                        help="do not refresh the index chain afterwards")
    args = parser.parse_args()

    root = args.data_root or paths_lib.default_root()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    plan = build_plan(root)
    print(f"data root: {root}")
    plan.show()

    if not args.apply:
        print("\n(manifest only; re-run with --apply to execute it)")
        return 0

    print("\napplying...")
    apply_plan(root, plan)
    manifest_path = root / "migration_manifest.json"
    manifest_path.write_text(json.dumps(
        {"applied": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
         "generator": "farfield/dataset_tools/migrate_data_root.py",
         "git_commit": provenance.git_commit(),
         "operations": plan.ops}, indent=1))
    print(f"manifest written to {manifest_path}")
    if not args.skip_indexes:
        result = indexes.refresh(root)
        print(f"index chain: {len(result['written'])} pages written")
        for skipped in result["skipped"]:
            print(f"  left alone (not ours): {skipped}")
    print(f"\nserve it: cd {root} && python3 -m http.server 8935")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
