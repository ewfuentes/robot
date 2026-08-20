"""Run configs: result-shaping parameters are set once and recorded.

The rule (REORG.md #2/#3): any value that encodes a modeling or dataset
assumption -- model names, catalog/artifact versions, thresholds, offsets,
resolutions, fusion windows -- is set at run creation, validated, and written
to `<run_dir>/run_config.json`. Every subsequent stage takes `--run_dir` and
reads the record; it refuses to run if a value it needs is absent. Changing a
parameter means a new run. Readers and viewers use the *recorded* config,
never a freshly-constructed default object.

This kills the two failure modes the checkpoint branch kept hitting: stale
argparse defaults over-fit to whichever dataset the stage was written on, and
viewers silently re-rendering old runs with today's thresholds.

The file is immutable once written. `create` refuses to overwrite; a stage
that wants a different value starts a new run (cheap by design) so the old
run's record keeps describing the old run.
"""

import json
from pathlib import Path

from experimental.overhead_matching.swag.farfield import provenance

RUN_CONFIG_NAME = "run_config.json"


class MissingConfigValue(Exception):
    """A stage asked for a config value the run never recorded."""


def create(run_dir: Path, config: dict, *, required: tuple,
           generator: str, inputs: dict, notes: str = "") -> Path:
    """Validate and record a new run's config; returns the file path.

    `required` lists the dotted keys that must be present and non-None --
    every assumption-carrying parameter the run's stages will need. All
    missing keys are reported at once, before anything is written. Refuses to
    overwrite an existing config: runs are immutable.
    """
    run_dir = Path(run_dir)
    path = run_dir / RUN_CONFIG_NAME
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; runs are immutable. Start a new run to "
            f"change a parameter.")
    missing = [key for key in required if _get(config, key) is None]
    if missing:
        raise MissingConfigValue(
            "run config is missing required values (no defaults are supplied "
            "on purpose):\n" + "\n".join(f"  {k}" for k in missing))
    run_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "farfield_run_config/v1",
        "generator": generator,
        "git_commit": provenance.git_commit(),
        "created": _now(),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "config": config,
        "notes": notes,
    }
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def load(run_dir: Path) -> dict:
    """The run's recorded document (schema/inputs/config/...).

    Raises with a pointed message when absent: a directory without a
    run_config.json is not a run.
    """
    path = Path(run_dir) / RUN_CONFIG_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- {Path(run_dir)} is not a run "
            f"directory (runs are created via run_config.create, which "
            f"records every result-shaping parameter up front).")
    return json.loads(path.read_text())


def value(run_dir_or_doc, key: str):
    """Config value by dotted key, e.g. `value(run_dir, "tracking.epoch_keyframes")`.

    Raises MissingConfigValue naming the run and the key when absent -- the
    caller must not substitute a default.
    """
    if isinstance(run_dir_or_doc, dict):
        doc, where = run_dir_or_doc, "<in-memory config>"
    else:
        doc, where = load(run_dir_or_doc), str(
            Path(run_dir_or_doc) / RUN_CONFIG_NAME)
    result = _get(doc.get("config", {}), key)
    if result is None:
        raise MissingConfigValue(
            f"{where} does not record {key!r}. Stages read every "
            f"result-shaping value from the run config; add it at run "
            f"creation (there is no default).")
    return result


def _get(config: dict, dotted: str):
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
