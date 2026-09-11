"""Embed the standard provenance manifest inside single-file JSON artifacts.

Several collection outputs are one JSON file in a lane of many (stage-1 stitch
manifests, discovery track lists, QC reports), so they cannot each carry a
sibling `manifest.json` without colliding. Instead the manifest is embedded in
the payload under a "provenance" key — produced by the ONE writer
(`farfield.provenance.write`) via a temp directory and read back, so the
embedded record is byte-for-byte the same schema as every directory manifest
and cannot drift from it.
"""

import tempfile
from pathlib import Path

from experimental.overhead_matching.swag.farfield import provenance


def provenance_record(*, generator: str, inputs: dict, config: dict,
                      notes: str = "") -> dict:
    """The dict `provenance.write` would put in manifest.json, for embedding."""
    with tempfile.TemporaryDirectory() as tmp:
        provenance.write(Path(tmp), generator=generator, inputs=inputs,
                         config=config, notes=notes)
        return provenance.read(Path(tmp))
