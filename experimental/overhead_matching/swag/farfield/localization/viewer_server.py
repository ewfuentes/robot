"""Local viewer server: the same page, plus what a file cannot do.

`viewer.py` writes a self-contained page — the frozen record, shareable and
immune to code drift. This serves the *same* payload from
`viewer_payload.build`, with the same HTML and the same JavaScript, and adds the
two things a static file structurally cannot have:

  GET /checkpoint/<kf>       every particle at that keyframe, not a 900-point
                             sample. Reading whether a mode has a real tail or
                             three stragglers needs all 20,000 of them, and
                             77 checkpoints x 20k particles is 83 MB — fine to
                             stream one at a time, impossible to inline.
  POST /replay               a live counterfactual: apply `Edits`, run the
                             production filter, return the ghost trajectory.
                             Ghosts are written to the deterministic sibling
                             <run>.counterfactuals/ directory
                             (replay.default_counterfactual_dir), leaving the
                             completed source artifact immutable.

The page feature-detects this server (`GET /api/health`) and lights up the
extra affordances when it answers, so one HTML/JS implementation covers both
deployments and neither can drift from the other.

Deliberately localhost-only and single-run. This is a workbench, not a service:
it executes the filter on request, so exposing it beyond the loopback interface
would be handing out arbitrary compute.

Usage:
  bazel run //experimental/overhead_matching/swag/farfield/localization:viewer_server -- \\
    --run_dir RUN --feather F --port 8765
"""

import argparse
import json
import threading
from pathlib import Path

import msgspec
import numpy as np
from flask import Flask, Response, jsonify, request

from experimental.overhead_matching.swag.farfield.localization import (
    replay as replay_mod,
    run_io,
    viewer,
    viewer_payload,
)


def create_app(run_dir: Path, feather: Path | None = None,
               ghost_dirs=()) -> Flask:
    app = Flask(__name__)
    state = {
        "payload": None,
        "checkpoints": None,
        # One replay at a time: each one saturates the GPU (or several CPU
        # cores), and two concurrent replays would just make both slow while
        # making the progress reporting meaningless.
        "lock": threading.Lock(),
        "ghosts": list(ghost_dirs),
    }

    def payload(rebuild: bool = False) -> dict:
        if state["payload"] is None or rebuild:
            state["payload"] = viewer_payload.build(
                run_dir, feather=feather,
                ghost_dirs=state["ghosts"])
            state["payload"]["server"] = True
        return state["payload"]

    def checkpoints() -> dict:
        if state["checkpoints"] is None:
            state["checkpoints"] = run_io.read_run(run_dir).checkpoints
        return state["checkpoints"]

    @app.get("/")
    def index():
        return Response(viewer.render_html(payload()), mimetype="text/html")

    @app.get("/api/health")
    def health():
        """What the page probes to decide whether to offer live features."""
        return jsonify({
            "ok": True, "run_dir": str(run_dir),
            "features": ["checkpoint", "replay"],
            "n_checkpoints": len(checkpoints()),
            "busy": state["lock"].locked(),
        })

    @app.get("/api/payload")
    def api_payload():
        return Response(json.dumps(payload(), separators=(",", ":")),
                        mimetype="application/json")

    @app.get("/api/checkpoint/<int:keyframe_idx>")
    def api_checkpoint(keyframe_idx: int):
        """Every particle at a checkpoint, with weights.

        Weights are sent rather than pre-sampled: with the full set in hand the
        page can draw opacity by weight, which shows the difference between a
        wide posterior and a wide cloud of dead particles — a distinction the
        subsampled static page cannot make.
        """
        arrays = checkpoints().get(keyframe_idx)
        if arrays is None:
            available = sorted(checkpoints())
            return jsonify({"error": f"no checkpoint at keyframe "
                                     f"{keyframe_idx}",
                            "available": available}), 404
        log_weight = arrays["log_weight"]
        weights = np.exp(log_weight - log_weight.max())
        weights = weights / weights.sum()
        return jsonify({
            "kf": keyframe_idx,
            "n": int(arrays["east_m"].shape[0]),
            "e": [round(float(v), 1) for v in arrays["east_m"]],
            "n_m": [round(float(v), 1) for v in arrays["north_m"]],
            "h": [round(float(np.degrees(v)), 1)
                  for v in arrays["heading_rad"]],
            "w": [float(f"{v:.4g}") for v in weights],
            "mode": [int(v) for v in arrays["mode_id"]],
            "event": [int(v) for v in arrays["proposal_event_id"]],
        })

    @app.post("/api/replay")
    def api_replay():
        """Run a counterfactual and return its ghost trajectory.

        Synchronous on purpose. A replay is ~30 s on the GPU backend, which is
        short enough that a progress protocol would cost more complexity than
        it saves, and a caller that has to wait cannot forget it asked.
        """
        body = request.get_json(silent=True) or {}
        try:
            edits = msgspec.convert(body.get("edits") or {}, replay_mod.Edits,
                                    strict=False)
        except msgspec.ValidationError as exc:
            return jsonify({"error": f"bad edits: {exc}"}), 400
        if edits.is_empty:
            return jsonify({"error": "no edits: that is just the baseline"}), 400

        if not state["lock"].acquire(blocking=False):
            return jsonify({"error": "a replay is already running"}), 409
        try:
            result = replay_mod.replay(run_dir, edits=edits)
            output_dir = replay_mod.default_counterfactual_dir(run_dir, edits)
            replay_mod.write_counterfactual(output_dir, run_dir, result)
        except Exception as exc:  # noqa: BLE001 - report, do not 500 silently
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
        finally:
            state["lock"].release()

        if str(output_dir) not in [str(g) for g in state["ghosts"]]:
            state["ghosts"].append(output_dir)
        ghosts, notes = viewer_payload._ghost_payload([output_dir])  # noqa: SLF001
        state["payload"] = None  # rebuild on next request, with the new ghost
        return jsonify({
            "describe": edits.describe(),
            "output_dir": str(output_dir),
            "elapsed_s": round(result.elapsed_s, 2),
            "ghost": ghosts[0] if ghosts else None,
            "notes": notes,
        })

    @app.post("/api/rebuild")
    def api_rebuild():
        """Rebuild the payload from disk — picks up a new attribution cache."""
        payload(rebuild=True)
        return jsonify({"ok": True, "notes": state["payload"]["notes"]})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--feather", type=Path, default=None)
    parser.add_argument("--ghost", type=Path, action="append", default=[])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback by default: this endpoint runs the "
                             "filter on request")
    args = parser.parse_args()

    app = create_app(args.run_dir, args.feather, args.ghost)
    print(f"serving {args.run_dir} at http://{args.host}:{args.port}/")
    print("  /api/checkpoint/<kf>  every particle at that keyframe")
    print("  /api/replay           live counterfactual (POST {\"edits\": ...})")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
