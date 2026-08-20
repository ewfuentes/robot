"""One source of truth for the farfield data-root layout.

Every stage needs the same handful of paths for a dataset -- panoramas, the
`frame_landmarks` artifact, the source video, the map catalog -- and resolving
them from one place makes the whole set move together. Before this module each
tracking stage carried its own hardcoded copy pointing at one dataset, which is
a *silent wrong answer* rather than a crash: a stage defaulting to leg1's video
while being handed leg2's panoramas builds tracks from the wrong imagery and
reports no error at all.

The layout, encoded once, here:

    datasets/<dataset>/                        frozen problem definition
    artifacts/<kind>/<dataset>/v<N>/           derived, immutable, + manifest.json
    raw_material/<collect>/...                 source material
    models/<family>/<checkpoint>               weights (+ SOURCE.md)
    runs/<experiment>/<run>/                   localization experiments

Resolution is keyed by **dataset name**. Anything the layout cannot supply
comes from the dataset's own `pipeline_metadata.json` -- notably the source
video, whose filename is arbitrary per leg, so it can only be looked up, never
derived.

There are deliberately NO default artifact versions and NO default catalog:
which `frame_landmarks` version and which catalog trim a run uses are modeling
choices, so they come from the run's recorded config (see `run_config.py`) or
an explicit flag, and resolution fails loudly when neither supplies them. A
default here is how a stage silently reads v1's detections against v4's
tracks.

Typical use in a stage script:

    paths_lib.add_arguments(parser, video=True)
    args = parser.parse_args()
    paths = paths_lib.resolve(parser, args, require=("dataset_base",))

Explicit path flags always win over resolution, so an ad-hoc directory can be
substituted for one input without disturbing the rest.
"""

import argparse
import json
import os
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
CATALOGS = "catalogs"

ARTIFACT_KINDS = (FRAME_LANDMARKS, PINHOLE_IMAGES, OBJECT_TRACKS, CATALOGS)


def default_root() -> Path:
    """Disk root, overridable by `FARFIELD_ROOT` for a mirror or a test tree."""
    return Path(os.environ.get(ROOT_ENV_VAR) or DEFAULT_ROOT)


class MissingInput(Exception):
    """A required path or value could not be resolved, with the reason attached.

    Raised instead of returning a best-guess so a misconfigured run stops at
    the point of resolution rather than deep inside a stage.
    """


@dataclass
class FarfieldPaths:
    """Resolved paths for one dataset.

    `versions` maps artifact kind -> version dir; there is no default version,
    so asking for an artifact whose version nothing supplied raises
    `MissingInput`. `overrides` maps a property name -> an explicit path that
    wins over resolution, which is how the `--dataset_base` / `--video` /
    `--feather` flags keep working. `catalog` is the catalog stem (e.g.
    `v3_trimmed`); it has no default for the same reason versions don't.
    """

    dataset: str
    root: Path = field(default_factory=default_root)
    versions: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    catalog: str | None = None
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

    def experiment_dir(self, experiment: str) -> Path:
        """`runs/<experiment>/` -- the home of localization runs.

        Every directory under `runs/` is an experiment dir carrying an
        `experiment.md`; actual runs are its children.
        """
        return self.runs_root / experiment

    # --- dataset lane --------------------------------------------------------

    @property
    def dataset_base(self) -> Path:
        return self.overrides.get("dataset_base") or (
            self.datasets_root / self.dataset)

    @property
    def panorama_dir(self) -> Path:
        """Equirectangular frames named `f####,<lat>,<lon>,.jpg`.

        The GPS fix lives in the filename because ingest parses it from there;
        the stems are also the keys of `embeddings.pkl` and the directory
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
        """Map catalog (OSM + ENC) for this dataset.

        The catalog is part of the *problem definition*: changing it changes
        everyone's numbers, the same way changing the ground-truth GPS would.
        Which trim to use is therefore a recorded choice, never a default.
        """
        override = self.overrides.get("feather")
        if override:
            return override
        if not self.catalog:
            raise MissingInput(
                "no catalog selected: pass --catalog / --feather, or read it "
                "from the run's recorded config. There is no default on "
                "purpose -- the catalog trim is a modeling choice.")
        return self.artifact(CATALOGS) / f"{self.catalog}.feather"

    # --- artifact lane -------------------------------------------------------

    def version(self, kind: str) -> str:
        if kind not in self.versions:
            raise MissingInput(
                f"no {kind} version resolved for dataset {self.dataset!r}: "
                f"pass --{kind}_version, or run from a directory whose "
                f"recorded config names one. There is no default version on "
                f"purpose -- a default is how a stage silently reads one "
                f"version's data against another's.")
        return self.versions[kind]

    def artifact(self, kind: str, version: str | None = None) -> Path:
        if kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"unknown artifact kind {kind!r}; known: {ARTIFACT_KINDS}")
        if kind == CATALOGS:
            # Catalogs are versioned by their stem (the trim name), not a vN
            # directory: artifacts/catalogs/<dataset>/<stem>.feather.
            return self.artifacts_root / kind / self.dataset
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
    def tracks_runs_root(self) -> Path:
        return self.object_tracks / "runs"

    # --- material and models -------------------------------------------------

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

        Only the dataset knows this: the filename encodes the leg's route and
        cannot be derived from the dataset name. `video.source_video` is a
        root-relative path, sometimes carrying a trailing parenthetical note
        about retention, which is stripped here so the note stays
        human-readable in the metadata.
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

    @property
    def sam2_checkpoint(self) -> Path:
        """SAM2 weights. Which checkpoint is a modeling choice: no default."""
        override = self.overrides.get("sam2_checkpoint")
        if override:
            return override
        raise MissingInput(
            "no SAM2 checkpoint selected: pass --checkpoint, or read it from "
            "the run's recorded config (models live under "
            f"{self.models_root}).")

    # --- validation ----------------------------------------------------------

    def require(self, *names: str) -> None:
        """Fail early, listing every missing input at once.

        Stages call this before spending GPU hours or API tokens; a run that
        is going to die on a missing video should die in the first second.
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
        rows = [("dataset", self.dataset), ("root", self.root),
                ("dataset_base", self.dataset_base),
                ("panorama_dir", self.panorama_dir)]
        for name in ("frame_landmarks", "pinhole_images", "object_tracks",
                     "feather", "video"):
            try:
                rows.append((name, getattr(self, name)))
            except MissingInput:
                rows.append((name, "<unresolved>"))
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)


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
    missing input prints one usage message and exits, rather than raising out
    of the middle of a stage. `require` names the inputs that stage genuinely
    needs, checked before it does any work.
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

    A run states which dataset, `frame_landmarks` version, catalog and video
    it was built from, and that record outranks re-resolution: a run built
    against `frame_landmarks/v2` must never have its audit or matching stages
    silently read another version's detections back -- different objects, same
    tracklet ids, no error anywhere.

    Empty when the run recorded nothing, in which case resolution proceeds
    from explicit flags alone (and fails loudly on anything version-shaped).
    """
    meta_path = Path(path) / RUN_META
    if not meta_path.exists():
        return {}
    try:
        recorded = json.loads(meta_path.read_text()).get("inputs") or {}
    except (json.JSONDecodeError, OSError):
        return {}
    known = ("dataset_base", "frame_landmarks", "pinhole_images", "video",
             "feather", "sam2_checkpoint")
    return {name: Path(recorded[name]) for name in known
            if recorded.get(name)}


def infer_from_artifact_path(path: Path) -> FarfieldPaths | None:
    """Recover the dataset from a path inside an artifact version dir.

    Layout is `<root>/artifacts/<kind>/<dataset>/<version>/...`, so a run
    directory already states which dataset (and which artifact version) it
    belongs to. Later stages take `--run_dir` only; inferring removes the
    chance for a second dataset flag to disagree.

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
                  pinhole: bool = False, dataset_required: bool = False
                  ) -> None:
    """Add the standard resolution flags to a stage's parser.

    `--dataset` drives everything; the explicit path flags are escape hatches
    that override one input each. Per-stage switches keep a parser from
    advertising flags its stage ignores. Version and catalog flags carry no
    defaults: a stage that needs them gets them from its run's recorded
    config or from the operator, explicitly.
    """
    group = parser.add_argument_group(
        "dataset resolution",
        "Paths resolve from --dataset via the farfield disk layout; explicit "
        "path flags below override individual inputs.")
    group.add_argument("--dataset", default=None, required=dataset_required,
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
                       help="frame_landmarks version (no default: from the "
                            "run's recorded config, or explicit)")
    group.add_argument("--object_tracks_version", default=None,
                       help="object_tracks version (no default: from the "
                            "run's recorded config, or explicit)")
    if pinhole:
        group.add_argument("--pinhole_dir", type=Path, default=None,
                           help="Override the resolved pinhole_images artifact")
        group.add_argument("--pinhole_version", default=None,
                           help="pinhole_images version (no default)")
    if video:
        group.add_argument("--video", type=Path, default=None,
                           help="Override the source video (default: "
                                "video.source_video in the dataset metadata)")
    if feather:
        group.add_argument("--feather", type=Path, default=None,
                           help="Override the resolved map catalog")
        group.add_argument("--catalog", default=None,
                           help="Catalog stem under artifacts/catalogs/ (no "
                                "default: the trim is a modeling choice)")
    if checkpoint:
        group.add_argument("--checkpoint", type=Path, default=None,
                           help="SAM2 checkpoint (no default: a modeling "
                                "choice)")


def from_args(args: argparse.Namespace, *,
              infer_from: Path | None = None) -> FarfieldPaths:
    """Build `FarfieldPaths` from a parser built with `add_arguments`.

    `infer_from` is a path inside an artifact lane -- typically the stage's
    `--run_dir` -- used to recover the dataset when `--dataset` was not
    passed. When both are available and they disagree, that is an error rather
    than a precedence rule: one of the two is wrong, and guessing which would
    mean reading one dataset's frames against another's tracks.
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
        catalog=getattr(args, "catalog", None),
    )
