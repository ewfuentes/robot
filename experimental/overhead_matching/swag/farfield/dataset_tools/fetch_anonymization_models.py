"""Fetch and verify the pinned models used by ``anonymize_video``.

The weights are data, not source code, and belong under the farfield data
root's ``models/`` lane.  Every download is content-addressed here so an
upstream replacement cannot silently change a released dataset.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


MODELS = {
    "license_plate_yolov9": {
        "filename": "yolo-v9-t-640-license-plates-end2end.onnx",
        "url": (
            "https://github.com/ankandrew/open-image-models/releases/"
            "download/assets/yolo-v9-t-640-license-plates-end2end.onnx"
        ),
        "sha256": (
            "c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076"
        ),
        "project": "Open Image Models",
        "project_url": "https://github.com/ankandrew/open-image-models",
        "source_revision": "f22000e02b30642f317cdba7755c0631638b109e",
        "license": "MIT",
        "license_url": (
            "https://github.com/ankandrew/open-image-models/blob/"
            "f22000e02b30642f317cdba7755c0631638b109e/LICENSE"
        ),
        "use": "license-plate detection",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(model: dict, output_dir: Path) -> tuple[Path, bool]:
    destination = output_dir / model["filename"]
    if destination.is_file():
        actual = sha256_file(destination)
        if actual != model["sha256"]:
            raise ValueError(
                f"refusing mismatched existing model {destination}: "
                f"expected {model['sha256']}, got {actual}")
        return destination, False
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"model destination is not a regular file: {destination}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".incomplete", dir=output_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with (urllib.request.urlopen(model["url"]) as response,
              temporary.open("wb") as output):
            shutil.copyfileobj(response, output)
        actual = sha256_file(temporary)
        if actual != model["sha256"]:
            raise ValueError(
                f"downloaded model digest mismatch for {model['filename']}: "
                f"expected {model['sha256']}, got {actual}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    records = []
    for name, model in MODELS.items():
        path, downloaded = fetch(model, args.output)
        records.append({
            "id": name,
            **model,
            "path": path.name,
            "size_bytes": path.stat().st_size,
        })
        verb = "downloaded" if downloaded else "verified"
        print(f"{verb} {name}: {path}")

    source_path = args.output / "SOURCE.json"
    source_path.write_text(json.dumps({
        "schema_version": 1,
        "models": records,
        "note": (
            "Weights are used only to locate regions for irreversible blur; "
            "no face recognition or license-plate OCR is performed."
        ),
    }, indent=2) + "\n")
    print(f"wrote {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
