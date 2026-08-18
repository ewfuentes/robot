"""One source of truth for `/data/farfield_matching` paths.

Every stage of the farfield pipeline needs the same handful of paths for a
dataset -- panoramas, the `frame_landmarks` artifact, the source video, the map
catalog, the SAM2 checkpoint -- and before this module each of the thirteen
tracking modules carried its own hardcoded copy pointing at
`boston_harbor_leg1`. That is a *silent wrong answer* rather than a crash: `m3`
defaulted to leg1's **video** while being handed leg2's panoramas via
`--dataset_base`, which builds tracks from the wrong imagery and reports no
error at all. Resolving from one place makes the whole set move together.

The layout is documented in `docs/farfield-data-organization.md` and mirrored at
`/data/farfield_matching/ORGANIZATION.md`. Encoded once, here:

    datasets/<dataset>/                        frozen problem definition
    artifacts/<kind>/<dataset>/v<N>/           derived, immutable, + manifest.json
    raw_material/<collect>/videos/<name>.mp4   source material
    models/<family>/<checkpoint>               weights

Resolution is keyed by **dataset name**. Anything the layout cannot supply comes
from the dataset's own `pipeline_metadata.json` -- notably the source video,
whose filename is arbitrary per leg (`long_wharf_to_hull_wharf.mp4` for leg1,
`hull_wharf_to_hingham_wharf.mp4` for leg2), so it can only be looked up, never
derived.

Typical use in a stage script:

    farfield_paths.add_arguments(parser, video=True, checkpoint=True)
    args = parser.parse_args()
    paths = farfield_paths.from_args(args)
    ingest.run_ingest(paths.dataset_base, paths.frame_landmarks, ...)

Explicit path flags still win over resolution, so an ad-hoc directory can always
be substituted for one input without disturbing the rest.
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOT = Path("/data/farfield_matching")
ROOT_ENV_VAR = "FARFIELD_ROOT"

# Artifact kinds are consumer contracts, not layouts: the name promises what a
# downstream stage can rely on finding, while the internal shape belongs to
# whichever producer wrote it.
FRAME_LANDMARKS = "frame_landmarks"
PINHOLE_IMAGES = "pinhole_images"
OBJECT_TRACKS = "object_tracks"
LANDMARK_MATCHING = "landmark_matching"

ARTIFACT_KINDS = (FRAME_LANDMARKS, PINHOLE_IMAGES, OBJECT_TRACKS,
                  LANDMARK_MATCHING)

DEFAULT_VERSION = "v1"

# The trimmed catalog is the matching default: m9 does no spatial gating, so it
# compares against every row, and the trim is what keeps that tractable.
DEFAULT_CATALOG = "v1_trimmed"

DEFAULT_SAM2_CHECKPOINT = Path("models/sam2/sam2.1_hiera_large.pt")


def default_root() -> Path:
    """Disk root, overridable by `FARFIELD_ROOT` for a mirror or a test tree."""
    return Path(os.environ.get(ROOT_ENV_VAR) or DEFAULT_ROOT)


def git_commit() -> str:
    """HEAD of the source workspace, for manifests. `unknown` if unavailable."""
    # bazel run sets BUILD_WORKSPACE_DIRECTORY to the source workspace; without
    # it the runfiles tree is not a git checkout.
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    try:
        return subprocess.check_output(
            ["git", "-C", workspace or ".", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, TypeError):
        return "unknown"


class MissingInput(Exception):
    """A required path could not be resolved, with the reason attached.

    Raised instead of returning a best-guess path so a misconfigured run stops
    at the point of resolution rather than deep inside a stage.
    """


@dataclass
class FarfieldPaths:
    """Resolved paths for one dataset.

    `versions` maps artifact kind -> version dir (default `v1`). `overrides`
    maps a property name -> an explicit path that wins over resolution, which is
    how the `--dataset_base` / `--video` / `--feather` flags keep working.
    """

    dataset: str
    root: Path = field(default_factory=default_root)
    versions: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    catalog: str = DEFAULT_CATALOG
    _metadata: dict | None = field(default=None, repr=False, compare=False)

    # --- lanes ---------------------------------------------------------------

    @property
    def datasets_root(self) -> Path:
        return self.root / "datasets"

    @property
    def artifacts_root(self) -> Path:
        return self.root / "artifacts"

    @property
    def raw_material_root(self) -> Path:
        return self.root / "raw_material"

    @property
    def models_root(self) -> Path:
        return self.root / "models"

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    # --- dataset lane --------------------------------------------------------

    @property
    def dataset_base(self) -> Path:
        return self.overrides.get("dataset_base") or (
            self.datasets_root / self.dataset)

    @property
    def panorama_dir(self) -> Path:
        """Equirectangular frames named `f####,<lat>,<lon>,.jpg`.

        The GPS fix lives in the filename because `ingest.py` parses it from
        there; the stems are also the keys of `embeddings.pkl` and the directory
        names under `pinhole_images`, so they must match byte for byte across
        all three.
        """
        return self.overrides.get("panorama_dir") or (
            self.dataset_base / "panorama")

    @property
    def frames_gps(self) -> Path:
        return self.dataset_base / "frames_gps.csv"

    @property
    def intrinsics(self) -> Path:
        return self.dataset_base / "intrinsics.csv"

    @property
    def metadata_path(self) -> Path:
        return self.dataset_base / "pipeline_metadata.json"

    @property
    def feather(self) -> Path:
        """Map catalog (OSM + ENC) that ships with the dataset.

        The catalog is part of the *problem definition*, not a method's
        artifact: changing it changes everyone's numbers, the same way changing
        the ground-truth GPS would.
        """
        return self.overrides.get("feather") or (
            self.dataset_base / "landmarks" / f"{self.catalog}.feather")

    # --- artifact lane -------------------------------------------------------

    def version(self, kind: str) -> str:
        return self.versions.get(kind, DEFAULT_VERSION)

    def artifact(self, kind: str, version: str | None = None) -> Path:
        if kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"unknown artifact kind {kind!r}; known: {ARTIFACT_KINDS}")
        return (self.artifacts_root / kind / self.dataset /
                (version or self.version(kind)))

    @property
    def frame_landmarks(self) -> Path:
        """VLM detections: `sentences/results/**/predictions.jsonl` + embeddings."""
        return self.overrides.get("frame_landmarks") or self.artifact(
            FRAME_LANDMARKS)

    @property
    def pinhole_images(self) -> Path:
        """One directory per pano stem, four face JPEGs inside."""
        return self.overrides.get("pinhole_images") or self.artifact(
            PINHOLE_IMAGES)

    @property
    def object_tracks(self) -> Path:
        return self.overrides.get("object_tracks") or self.artifact(
            OBJECT_TRACKS)

    @property
    def landmark_matching(self) -> Path:
        return self.overrides.get("landmark_matching") or self.artifact(
            LANDMARK_MATCHING)

    def tracks_stage(self, stage: str) -> Path:
        """Per-stage working dir inside the `object_tracks` artifact.

        Stage dirs (`m0_boxes`, `m1_heading`, `m2_sam2`, `m3_tracks`) are the
        producer's internal layout; immutability lives on the run ids beneath
        `m3_tracks/runs/`, not on these names.
        """
        return self.object_tracks / stage

    @property
    def tracks_runs_root(self) -> Path:
        return self.tracks_stage("m3_tracks") / "runs"

    # --- material and models -------------------------------------------------

    @property
    def sam2_checkpoint(self) -> Path:
        return self.overrides.get("sam2_checkpoint") or (
            self.root / DEFAULT_SAM2_CHECKPOINT)

    def metadata(self) -> dict:
        """Parsed `pipeline_metadata.json`, cached."""
        if self._metadata is None:
            if not self.metadata_path.exists():
                raise MissingInput(
                    f"{self.metadata_path} not found -- is {self.dataset!r} a "
                    f"dataset under {self.datasets_root}?")
            self._metadata = json.loads(self.metadata_path.read_text())
        return self._metadata

    @property
    def video(self) -> Path:
        """Source video, from the dataset's own metadata.

        Only the dataset knows this: the filename encodes the leg's route
        (`hull_wharf_to_hingham_wharf.mp4`) and cannot be derived from the
        dataset name. `video.source_video` is a root-relative path, sometimes
        carrying a trailing parenthetical note about retention, which is
        stripped here so the note stays human-readable in the metadata.
        """
        override = self.overrides.get("video")
        if override:
            return override
        meta = self.metadata()
        raw = (meta.get("video") or {}).get("source_video")
        if not raw:
            raise MissingInput(
                f"{self.metadata_path} has no video.source_video; pass --video "
                f"explicitly. Tracking stages need the dense video to "
                f"propagate masks between keyframes.")
        # "raw_material/.../x.mp4 (not retained; ~38 GB originals)" -> the path.
        path = Path(raw.split(" (")[0].strip())
        return path if path.is_absolute() else self.root / path

    # --- validation ----------------------------------------------------------

    def require(self, *names: str) -> None:
        """Fail early, listing every missing input at once.

        Stages call this before spending GPU hours or API tokens; a run that is
        going to die on a missing video should die in the first second.
        """
        missing = []
        for name in names:
            try:
                path = getattr(self, name)
            except MissingInput as exc:
                missing.append(f"  {name}: {exc}")
                continue
            if not path.exists():
                missing.append(f"  {name}: {path} does not exist")
        if missing:
            raise MissingInput(
                f"missing inputs for dataset {self.dataset!r}:\n" +
                "\n".join(missing))

    def describe(self) -> str:
        """Human-readable resolution table, for a stage's opening banner."""
        rows = [("dataset", self.dataset), ("root", self.root)]
        for name in ("dataset_base", "panorama_dir", "frame_landmarks",
                     "pinhole_images", "object_tracks", "feather"):
            rows.append((name, getattr(self, name)))
        try:
            rows.append(("video", self.video))
        except MissingInput:
            rows.append(("video", "<unresolved>"))
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)

    # --- manifests -----------------------------------------------------------

    def write_manifest(self, kind: str, *, generator: str, config: dict,
                       inputs: list, notes: str = "",
                       created: str | None = None,
                       version: str | None = None) -> Path:
        """Write `manifest.json` into an artifact version dir.

        Per the organization guide it is the manifest -- not the filename or the
        directory name -- that records how an artifact was made, so every
        version dir carries one. `created` is a parameter rather than
        `date.today()` so a backfilled manifest can state the date the artifact
        actually came from.
        """
        target = self.artifact(kind, version)
        target.mkdir(parents=True, exist_ok=True)
        manifest = {
            "kind": kind,
            "dataset": self.dataset,
            "version": version or self.version(kind),
            "generator": generator,
            "git_commit": git_commit(),
            "config": config,
            "inputs": inputs,
            "created": created or _today(),
            "notes": notes,
        }
        path = target / "manifest.json"
        path.write_text(json.dumps(manifest, indent=1) + "\n")
        return path


def _today() -> str:
    # Imported lazily so the module stays usable where date is stubbed out.
    from datetime import date
    return date.today().isoformat()


def relative_to_root(path: Path, root: Path | None = None) -> str:
    """Render `path` root-relative for a manifest's `inputs`, if it is inside."""
    root = root or default_root()
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def resolve(parser: argparse.ArgumentParser, args: argparse.Namespace, *,
            infer_from: Path | None = None,
            require: tuple = ()) -> FarfieldPaths:
    """Resolve paths, turning any failure into a clean CLI error.

    Stages call this instead of `from_args` so an unresolvable dataset or a
    missing input prints one usage message and exits, rather than raising out of
    the middle of a stage. `require` names the inputs that stage genuinely needs,
    checked before it does any work.
    """
    try:
        paths = from_args(args, infer_from=infer_from)
        if require:
            paths.require(*require)
    except MissingInput as exc:
        parser.error(str(exc))
    return paths


RUN_META = "run_meta.json"


def recorded_run_inputs(path: Path) -> dict:
    """Inputs a tracking run recorded for itself, as `{name: Path}`.

    A run states which dataset, `frame_landmarks` version and video it was built
    from, and that record outranks re-resolution: the artifact lane's *default*
    version is v1, so a run built against `frame_landmarks/v2` would otherwise
    have its audit, merge and matching stages silently read v1's detections back
    -- different objects, same tracklet ids, no error anywhere. Inferring the
    dataset from the run's path already fixed the coarse version of this; this
    fixes the version-level one.

    Empty when the run predates provenance recording, in which case the caller
    falls back to normal resolution.
    """
    meta_path = Path(path) / RUN_META
    if not meta_path.exists():
        return {}
    try:
        recorded = json.loads(meta_path.read_text()).get("inputs") or {}
    except (json.JSONDecodeError, OSError):
        return {}
    known = ("dataset_base", "frame_landmarks", "pinhole_images", "video",
             "feather")
    return {name: Path(recorded[name]) for name in known
            if recorded.get(name)}


def infer_from_artifact_path(path: Path) -> FarfieldPaths | None:
    """Recover the dataset from a path inside an artifact version dir.

    Layout is `<root>/artifacts/<kind>/<dataset>/<version>/...`, so a run
    directory such as
    `.../artifacts/object_tracks/boston_harbor_leg3/v1/m3_tracks/runs/r001`
    already states which dataset it belongs to. Later stages take `--run_dir`
    and previously *also* took a `--dataset_base` defaulting to leg1, so
    pointing at one leg's run while reading another leg's frames was a one-flag
    mistake with no error message. Inferring removes the chance to disagree.

    Returns None when `path` is not inside a recognizable artifact lane (a
    scratch directory, say), in which case the caller should fall back to an
    explicit `--dataset`.
    """
    path = Path(path).resolve()
    parts = path.parts
    for i in range(len(parts) - 3, -1, -1):
        if parts[i] != "artifacts":
            continue
        kind = parts[i + 1]
        if kind not in ARTIFACT_KINDS:
            continue
        dataset, version = parts[i + 2], parts[i + 3]
        return FarfieldPaths(
            dataset=dataset,
            root=Path(*parts[:i]) if i else Path("/"),
            versions={kind: version},
        )
    return None


def add_arguments(parser: argparse.ArgumentParser, *, video: bool = False,
                  feather: bool = False, checkpoint: bool = False,
                  pinhole: bool = False, dataset_required: bool = False,
                  default_dataset: str | None = None) -> None:
    """Add the standard resolution flags to a stage's parser.

    `--dataset` drives everything; the explicit path flags are escape hatches
    that override one input each. Per-stage switches keep a parser from
    advertising flags its stage ignores (`m0` has no use for a video).
    """
    group = parser.add_argument_group(
        "dataset resolution",
        "Paths resolve from --dataset via the farfield disk layout; explicit "
        "path flags below override individual inputs.")
    group.add_argument("--dataset", default=default_dataset,
                       required=dataset_required,
                       help="Dataset name under <root>/datasets/, e.g. "
                            "boston_harbor_leg2")
    group.add_argument("--farfield_root", type=Path, default=None,
                       help=f"Disk root (default ${ROOT_ENV_VAR} or "
                            f"{DEFAULT_ROOT})")
    group.add_argument("--dataset_base", type=Path, default=None,
                       help="Override the resolved dataset directory")
    group.add_argument("--landmark_base", type=Path, default=None,
                       help="Override the resolved frame_landmarks artifact")
    group.add_argument("--frame_landmarks_version", default=None,
                       help=f"frame_landmarks version (default "
                            f"{DEFAULT_VERSION})")
    group.add_argument("--object_tracks_version", default=None,
                       help=f"object_tracks version (default "
                            f"{DEFAULT_VERSION})")
    if pinhole:
        group.add_argument("--pinhole_dir", type=Path, default=None,
                           help="Override the resolved pinhole_images artifact")
        group.add_argument("--pinhole_version", default=None,
                           help=f"pinhole_images version (default "
                                f"{DEFAULT_VERSION})")
    if video:
        group.add_argument("--video", type=Path, default=None,
                           help="Override the source video (default: "
                                "video.source_video in the dataset metadata)")
    if feather:
        group.add_argument("--feather", type=Path, default=None,
                           help="Override the resolved map catalog")
        group.add_argument("--catalog", default=DEFAULT_CATALOG,
                           help=f"Catalog stem under <dataset>/landmarks/ "
                                f"(default {DEFAULT_CATALOG})")
    if checkpoint:
        group.add_argument("--checkpoint", type=Path, default=None,
                           help="Override the SAM2 checkpoint")


def from_args(args: argparse.Namespace, *,
              infer_from: Path | None = None) -> FarfieldPaths:
    """Build `FarfieldPaths` from a parser built with `add_arguments`.

    `infer_from` is a path inside an artifact lane -- typically the stage's
    `--run_dir` -- used to recover the dataset when `--dataset` was not passed.
    When both are available and they disagree, that is an error rather than a
    precedence rule: one of the two is wrong, and guessing which would mean
    reading one dataset's frames against another's tracks.
    """
    dataset = getattr(args, "dataset", None)
    dataset_base = getattr(args, "dataset_base", None)
    inferred = infer_from_artifact_path(infer_from) if infer_from else None

    if inferred and dataset and inferred.dataset != dataset:
        raise MissingInput(
            f"--dataset {dataset!r} disagrees with {infer_from}, which belongs "
            f"to {inferred.dataset!r}. Drop --dataset to use the path's "
            f"dataset, or point at that dataset's directory.")

    if not dataset:
        if inferred:
            dataset = inferred.dataset
        elif dataset_base:
            # An explicit --dataset_base with no name still identifies the
            # dataset by its directory name, which is how artifact lanes key it.
            dataset = Path(dataset_base).name
        else:
            raise MissingInput(
                "pass --dataset (a name under <root>/datasets/), or "
                "--dataset_base for an out-of-tree directory")

    versions = {}
    if inferred:
        # A run inside object_tracks/<ds>/v2 belongs to v2 even when the flag
        # was not passed.
        versions.update(inferred.versions)
    for kind, attr in ((FRAME_LANDMARKS, "frame_landmarks_version"),
                       (PINHOLE_IMAGES, "pinhole_version"),
                       (OBJECT_TRACKS, "object_tracks_version")):
        value = getattr(args, attr, None)
        if value:
            versions[kind] = value

    overrides = {}
    # A run's recorded inputs come first so later stages read exactly what the
    # run was built from; an explicit flag below still wins over the record.
    if infer_from:
        overrides.update(recorded_run_inputs(infer_from))

    for name, attr in (("dataset_base", "dataset_base"),
                       ("frame_landmarks", "landmark_base"),
                       ("pinhole_images", "pinhole_dir"),
                       ("video", "video"),
                       ("feather", "feather"),
                       ("sam2_checkpoint", "checkpoint")):
        value = getattr(args, attr, None)
        if value:
            overrides[name] = Path(value)

    # An inferred root keeps a mirrored or test tree self-consistent: a run dir
    # under /mnt/copy/artifacts/... should read its dataset from /mnt/copy too.
    root = (getattr(args, "farfield_root", None) or
            (inferred.root if inferred else None) or default_root())

    return FarfieldPaths(
        dataset=dataset,
        root=root,
        versions=versions,
        overrides=overrides,
        catalog=getattr(args, "catalog", None) or DEFAULT_CATALOG,
    )
