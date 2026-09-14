"""Re-run one archived causal grid result, optionally with a new IMU seed."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from experimental.overhead_matching.swag.farfield.localization import grid_filter


def configuration(reference, output, seed=None, path_maps=()):
    if reference.get("schema") != "farfield_causal_grid/v1":
        raise ValueError("expected a causal grid result, not a plan or summary")
    config = dict(reference["config"])
    if (config.get("smoother") != "none" or config.get("smooth_lag", 0)
            or config.get("smooth_lags", "").strip() or "smoothing" in reference):
        raise ValueError("this reproducer requires a causal-only reference")
    for old, new in path_maps:
        if not Path(old).is_absolute() or not Path(new).is_absolute():
            raise ValueError("path mappings must use absolute directory prefixes")
    for key, value in config.items():
        if isinstance(value, str) and Path(value).is_absolute():
            for old, new in path_maps:
                if Path(value).is_relative_to(old):
                    config[key] = str(Path(new) / Path(value).relative_to(old))
                    break
    config["out"] = str(output)
    if seed is not None:
        config["odometry_seed"] = seed
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--odometry-seed", type=int)
    parser.add_argument("--path-map", nargs=2, action="append", default=[],
                        metavar=("OLD_DIRECTORY", "NEW_DIRECTORY"))
    args = parser.parse_args()
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    config = configuration(reference, partial, args.odometry_seed, args.path_map)
    if args.out.exists() or partial.exists():
        raise ValueError("choose a new output path; completed/partial runs are preserved")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial.touch(exist_ok=False)
    sys.argv = ["grid_filter"]
    for key, value in config.items():
        if value is not None:
            sys.argv.extend(["--" + key, str(value)])
    grid_filter.main()
    result = json.loads(partial.read_text())
    if ("smoothing" in result or result["grid"] != reference["grid"]
            or result.get("episode") != reference.get("episode")
            or any(result["config"][key] != value for key, value in config.items())):
        raise ValueError("reproduction changed the grid, episode or requested settings")
    difference = 0.0
    for radius, expected in reference["mass_by_keyframe"].items():
        actual = result["mass_by_keyframe"][radius]
        if (len(actual) != len(expected)
                or any(not math.isfinite(v) or not -1e-6 <= v <= 1 + 1e-6 for v in actual)):
            raise ValueError("invalid or incomplete causal mass series")
        difference = max(difference, max(abs(x-y) for x, y in zip(actual, expected)))
    same_seed = config["odometry_seed"] == reference["config"]["odometry_seed"]
    if same_seed and difference > 2e-4:
        raise ValueError(f"same-seed mass differs by {difference}; inspect inputs/code")
    result["reproduction"] = {
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "reference_seed": reference["config"]["odometry_seed"],
        "same_seed_max_mass_difference": difference if same_seed else None,
        "source_sha256": {
            name: hashlib.sha256(Path(grid_filter.__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("grid_filter.py", "detection_audit.py")},
    }
    partial.write_text(json.dumps(result, indent=1))
    partial.rename(args.out)
    print("reproduced", args.out,
          {r: result["summary"]["dn_mass_" + r] for r in ("500", "100")})


if __name__ == "__main__":
    main()
