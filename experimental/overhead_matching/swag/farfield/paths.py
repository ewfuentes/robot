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
    builds/<dataset>/<build>/                  mutable orchestration state
    runs/<experiment>/<run>/                   completed localization runs

Resolution is keyed by **dataset name**. Anything the layout cannot supply
comes from the dataset's own `pipeline_metadata.json` -- notably the source
video, whose filename is arbitrary per leg, so it can only be looked up, never
derived.

There are deliberately NO default artifact versions and NO default catalog:
which `frame_landmarks` version and which catalog trim a run uses are modeling
choices, so they come from the build's recorded config (see `build_config.py`) or
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

from experimental.overhead_matching.swag.farfield import artifact

DEFAULT_ROOT = Path("/data/farfield_matching")
ROOT_ENV_VAR = "FARFIELD_ROOT"

# Artifact kinds are consumer contracts, not layouts: the name promises what a
# downstream stage can rely on finding, while the internal shape belongs to
# whichever producer wrote it.
FRAME_LANDMARKS = "frame_landmarks"
PINHOLE_IMAGES = "pinhole_images"
OBJECT_TRACKS = "object_tracks"
SEMANTIC_AUDITS = "semantic_audits"
BEARING_OBSERVATIONS = "bearing_observations"
LANDMARK_MATCHES = "landmark_matches"
ALIGNMENT_DIAGNOSTICS = "alignment_diagnostics"
LOCALIZATION_INPUTS = "localization_inputs"
CATALOGS = "catalogs"
# Review-only collection diagnostic. It is a typed artifact so publication is
# transactional and provenance-bound, but it is deliberately not a pipeline
# input and therefore is not part of ARTIFACT_KINDS/build configuration.
CATALOG_COVERAGE = "catalog_coverage"

ARTIFACT_KINDS = (
    FRAME_LANDMARKS,
    PINHOLE_IMAGES,
    OBJECT_TRACKS,
    SEMANTIC_AUDITS,
    BEARING_OBSERVATIONS,
    LANDMARK_MATCHES,
    ALIGNMENT_DIAGNOSTICS,
    LOCALIZATION_INPUTS,
    CATALOGS,
)

DATASET_PIPELINE_METADATA_SHA256 = "dataset_pipeline_metadata_sha256"
DATASET_FRAMES_GPS_SHA256 = "dataset_frames_gps_sha256"
DATASET_PANORAMA_SHA256 = "dataset_panorama_sha256"
DATASET_SOURCE_DIGEST_KEYS = (
    DATASET_PIPELINE_METADATA_SHA256,
    DATASET_FRAMES_GPS_SHA256,
    DATASET_PANORAMA_SHA256,
)


def default_root() -> Path:
    """Disk root, overridable by `FARFIELD_ROOT` for a mirror or a test tree."""
    return Path(os.environ.get(ROOT_ENV_VAR) or DEFAULT_ROOT)


class MissingInput(Exception):
    """A required path or value could not be resolved, with the reason attached.

    Raised instead of returning a best-guess so a misconfigured run stops at
    the point of resolution rather than deep inside a stage.
    """


class PathContractError(ValueError):
    """A lane component could escape or ambiguously name its directory."""


def require_identifier(value: str, what: str) -> str:
    """Validate a directory-name component against artifact identities."""
    try:
        return artifact.require_identifier(value, what)
    except artifact.ArtifactValidationError as exc:
        raise PathContractError(str(exc)) from exc


def dataset_source_digests(dataset_base: Path) -> dict[str, str]:
    """Hash every dataset byte that shapes extraction or tracking.

    Panorama entries may be symlink views of frozen source frames, so their
    resolved regular-file bytes are hashed under their logical filenames.
    Metadata and GPS tables themselves must be regular non-symlink files.
    """
    dataset_base = Path(dataset_base)
    metadata = dataset_base / "pipeline_metadata.json"
    frames_gps = dataset_base / "frames_gps.csv"
    panorama_dir = dataset_base / "panorama"
    if panorama_dir.is_symlink():
        panorama_root = panorama_dir.resolve()
    else:
        panorama_root = panorama_dir
    if not panorama_root.is_dir():
        raise MissingInput(f"dataset panorama directory does not exist: "
                           f"{panorama_dir}")
    panoramas = sorted(panorama_dir.glob("*.jpg"))
    if not panoramas:
        raise MissingInput(f"dataset has no panorama JPEGs: {panorama_dir}")
    records = []
    for path in panoramas:
        target = path.resolve()
        if not target.is_file() or target.is_symlink():
            raise MissingInput(f"panorama is not a regular file: {path}")
        records.append({
            "path": path.name,
            "size": target.stat().st_size,
            "sha256": artifact.sha256_file(target),
        })
    try:
        metadata_digest = artifact.sha256_file(metadata)
        gps_digest = artifact.sha256_file(frames_gps)
    except artifact.ArtifactValidationError as exc:
        raise MissingInput(f"invalid dataset source: {exc}") from exc
    return {
        DATASET_PIPELINE_METADATA_SHA256: metadata_digest,
        DATASET_FRAMES_GPS_SHA256: gps_digest,
        DATASET_PANORAMA_SHA256: artifact.sha256_json(records),
    }


@dataclass
class FarfieldPaths:
    """Resolved paths for one dataset.

    `versions` maps every artifact kind to its immutable version directory.
    There are no special layouts and no default versions. `overrides` is for
    explicit diagnostic inputs; production orchestration passes artifact
    directories directly and validates their manifests.
    """

    dataset: str
    root: Path = field(default_factory=default_root)
    versions: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    _metadata: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.dataset = require_identifier(self.dataset, "dataset")
        self.root = Path(self.root)
        if not isinstance(self.versions, dict):
            raise PathContractError("versions must be a mapping")
        for kind, version in self.versions.items():
            if kind not in ARTIFACT_KINDS:
                raise PathContractError(f"unknown artifact kind {kind!r}")
            require_identifier(version, f"{kind} version")
        self.versions = dict(self.versions)
        if not isinstance(self.overrides, dict):
            raise PathContractError("overrides must be a mapping")
        self.overrides = {
            key: Path(value) for key, value in self.overrides.items()
        }

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

    @property
    def builds_root(self) -> Path:
        return self.root / "builds"

    def build_dir(self, build_name: str) -> Path:
        """Mutable orchestration state, never a scientific artifact."""
        return (self.builds_root / self.dataset /
                require_identifier(build_name, "build name"))

    def experiment_dir(self, experiment: str) -> Path:
        """`runs/<experiment>/` -- the home of localization runs.

        Every directory under `runs/` is an experiment dir carrying an
        `experiment.md`; actual runs are its children.
        """
        return self.runs_root / require_identifier(experiment, "experiment")

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
        """Validated catalog payload inside the selected catalog artifact."""
        override = self.overrides.get("feather")
        if override:
            return override
        return self.artifact(CATALOGS) / "catalog.feather"

    # --- artifact lane -------------------------------------------------------

    def version(self, kind: str) -> str:
        if kind not in self.versions:
            raise MissingInput(
                f"no {kind} version resolved for dataset {self.dataset!r}: "
                f"pass --{kind}_version, or resolve it from build_config.json. "
                f"There is no default version on "
                f"purpose -- a default is how a stage silently reads one "
                f"version's data against another's.")
        return require_identifier(self.versions[kind], f"{kind} version")

    def artifact(self, kind: str, version: str | None = None) -> Path:
        if kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"unknown artifact kind {kind!r}; known: {ARTIFACT_KINDS}")
        selected = (require_identifier(version, f"{kind} version")
                    if version is not None else self.version(kind))
        return self.artifacts_root / kind / self.dataset / selected

    @property
    def frame_landmarks(self) -> Path:
        """Validated VLM detections: one canonical `predictions.jsonl`."""
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
    def semantic_audits(self) -> Path:
        return self.overrides.get("semantic_audits") or self.artifact(
            SEMANTIC_AUDITS)

    @property
    def bearing_observations(self) -> Path:
        return self.overrides.get("bearing_observations") or self.artifact(
            BEARING_OBSERVATIONS)

    @property
    def landmark_matches(self) -> Path:
        return self.overrides.get("landmark_matches") or self.artifact(
            LANDMARK_MATCHES)

    @property
    def alignment_diagnostics(self) -> Path:
        return self.overrides.get("alignment_diagnostics") or self.artifact(
            ALIGNMENT_DIAGNOSTICS)

    @property
    def localization_inputs(self) -> Path:
        return self.overrides.get("localization_inputs") or self.artifact(
            LOCALIZATION_INPUTS)

    @property
    def catalogs(self) -> Path:
        return self.overrides.get("catalogs") or self.artifact(CATALOGS)

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
        if not isinstance(raw, str):
            raise MissingInput(
                f"{self.metadata_path} video.source_video must be a string")
        path = Path(raw.split(" (")[0].strip())
        if (not path.parts or path.is_absolute()
                or any(part in ("", ".", "..") for part in path.parts)):
            raise MissingInput(
                f"{self.metadata_path} video.source_video must be a normalized "
                "root-relative path")
        candidate = self.root / path
        try:
            candidate.resolve(strict=False).relative_to(self.root.resolve())
        except ValueError as exc:
            raise MissingInput(
                f"{self.metadata_path} video.source_video escapes farfield root") \
                from exc
        return candidate

    @property
    def sam2_checkpoint(self) -> Path:
        """SAM2 weights. Which checkpoint is a modeling choice: no default."""
        override = self.overrides.get("sam2_checkpoint")
        if override:
            return override
        raise MissingInput(
            "no SAM2 checkpoint selected: pass --checkpoint, or read it from "
            "the build's recorded config (models live under "
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
                     "semantic_audits", "bearing_observations",
                     "landmark_matches", "alignment_diagnostics",
                     "localization_inputs", "catalogs", "video"):
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
    except (MissingInput, PathContractError) as exc:
        parser.error(str(exc))
    return paths


def infer_from_artifact_path(path: Path) -> FarfieldPaths | None:
    """Recover the dataset from a path inside an artifact version dir.

    Layout is `<root>/artifacts/<kind>/<dataset>/<version>/...`; an explicit
    artifact path therefore identifies its dataset and version without a
    redundant, potentially conflicting flag.

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
    defaults: a stage that needs them gets them from its build's recorded
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
                            "build's recorded config, or explicit)")
    for kind in (OBJECT_TRACKS, SEMANTIC_AUDITS, BEARING_OBSERVATIONS,
                 LANDMARK_MATCHES, ALIGNMENT_DIAGNOSTICS,
                 LOCALIZATION_INPUTS):
        group.add_argument(
            f"--{kind}_version", default=None,
            help=f"{kind} artifact version (no default)")
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
        group.add_argument("--catalogs_version", default=None,
                           help="catalog artifact version (no default)")
    if checkpoint:
        group.add_argument("--checkpoint", type=Path, default=None,
                           help="SAM2 checkpoint (no default: a modeling "
                                "choice)")


def from_args(args: argparse.Namespace, *,
              infer_from: Path | None = None) -> FarfieldPaths:
    """Build `FarfieldPaths` from a parser built with `add_arguments`.

    `infer_from` is a path inside an artifact lane used to recover the dataset
    when `--dataset` was not
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
    for kind, attr in (
            (FRAME_LANDMARKS, "frame_landmarks_version"),
            (PINHOLE_IMAGES, "pinhole_version"),
            (OBJECT_TRACKS, "object_tracks_version"),
            (SEMANTIC_AUDITS, "semantic_audits_version"),
            (BEARING_OBSERVATIONS, "bearing_observations_version"),
            (LANDMARK_MATCHES, "landmark_matches_version"),
            (ALIGNMENT_DIAGNOSTICS, "alignment_diagnostics_version"),
            (LOCALIZATION_INPUTS, "localization_inputs_version"),
            (CATALOGS, "catalogs_version")):
        value = getattr(args, attr, None)
        if value:
            versions[kind] = value

    overrides = {}
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
    )
