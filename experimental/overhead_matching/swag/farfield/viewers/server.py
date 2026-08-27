"""Serve the farfield data tree and centralized human matcher notes.

This is the writable counterpart to ``python -m http.server``: existing
indexes and generated viewers are served unchanged, while same-origin viewer
pages may read and update one fixed notes file.  It binds only to IPv4
loopback; use SSH local forwarding when the browser is on another machine.

Run with the standard data root:

  bazel run //experimental/overhead_matching/swag/farfield/viewers:server
"""

from __future__ import annotations

import argparse
import html
import json
import stat
import urllib.parse
from pathlib import Path

from flask import Flask, Response, abort, jsonify, redirect, request, send_file

from experimental.overhead_matching.swag.farfield.viewers import match_notes


DEFAULT_ROOT = Path("/data/farfield_matching")
DEFAULT_PORT = 8765
HOST = "127.0.0.1"
WRITE_HEADER = "X-Farfield-Viewer"
WRITE_HEADER_VALUE = "match-notes-v1"
MAX_REQUEST_BYTES = 64 * 1024


def _require_root(data_root: Path) -> Path:
    data_root = Path(data_root)
    try:
        metadata = data_root.lstat()
    except FileNotFoundError as error:
        raise match_notes.MatchNotesError(
            f"data root does not exist: {data_root}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise match_notes.MatchNotesError(
            f"data root must be a real directory: {data_root}")
    return data_root.resolve(strict=True)


def _safe_target(data_root: Path, relative: str) -> Path | None:
    if relative.split("/", 1)[0] in {
            "api", match_notes.ANNOTATIONS_DIR_NAME}:
        return None
    try:
        target = (data_root / relative).resolve(strict=True)
        target.relative_to(data_root)
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    return target


def _directory_page(data_root: Path, directory: Path, request_path: str) -> str:
    rows = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if (entry.name.startswith(".")
                or (directory == data_root
                    and entry.name == match_notes.ANNOTATIONS_DIR_NAME)):
            continue
        try:
            resolved = entry.resolve(strict=True)
            resolved.relative_to(data_root)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        is_directory = resolved.is_dir()
        suffix = "/" if is_directory else ""
        href = urllib.parse.quote(entry.name) + suffix
        label = html.escape(entry.name) + suffix
        rows.append(f"<li><a href=\"{href}\">{label}</a></li>")
    parent = "<li><a href=\"../\">../</a></li>" if request_path != "/" else ""
    title = "/" + str(directory.relative_to(data_root))
    if title == "/.":
        title = "/"
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:24px;"
            "background:#14171c;color:#d8dce2}a{color:#7fb4e8}"
            "li{margin:5px 0}</style></head><body>"
            f"<h1>{html.escape(title)}</h1><ul>{parent}{''.join(rows)}</ul>"
            "</body></html>")


def _require_same_origin_write():
    if request.headers.get(WRITE_HEADER) != WRITE_HEADER_VALUE:
        abort(403)
    if request.headers.get("Sec-Fetch-Site") not in (None, "same-origin"):
        abort(403)
    expected_origin = request.host_url.rstrip("/")
    if request.headers.get("Origin") != expected_origin:
        abort(403)


def create_app(data_root: Path = DEFAULT_ROOT) -> Flask:
    root = _require_root(data_root)
    store = match_notes.MatchNotesStore(root)
    store.initialize()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(match_notes.MatchNotesError)
    def notes_error(error):
        return jsonify({"error": str(error)}), 400

    @app.get("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "root": str(root),
            "features": ["match_notes"],
            "notes_file": str(store.notes_path),
        })

    @app.get("/api/match-notes")
    def get_match_notes():
        result = store.get(request.args.get("matching_digest", ""))
        return jsonify({"schema": match_notes.SCHEMA, **result})

    @app.put("/api/match-notes")
    def put_match_note():
        _require_same_origin_write()
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        expected = {"matching", "tracklet_id", "text"}
        if set(value) != expected:
            return jsonify({
                "error": "request body must have exact keys "
                         f"{sorted(expected)}",
            }), 400
        note = store.put(
            matching=value["matching"], tracklet_id=value["tracklet_id"],
            text=value["text"])
        return jsonify({"ok": True, "note": note})

    @app.get("/")
    @app.get("/<path:relative>")
    def static_tree(relative=""):
        target = _safe_target(root, relative)
        if target is None:
            abort(404)
        if target.is_dir():
            if request.path != "/" and not request.path.endswith("/"):
                return redirect(request.path + "/")
            index = target / "index.html"
            if index.is_file() and not index.is_symlink():
                return send_file(index, conditional=True)
            return Response(
                _directory_page(root, target, request.path),
                mimetype="text/html")
        if not target.is_file():
            abort(404)
        return send_file(target, conditional=True)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    try:
        app = create_app(args.root)
    except (match_notes.MatchNotesError, OSError) as error:
        parser.error(str(error))
    print(f"serving {args.root} at http://{HOST}:{args.port}/")
    print(f"matcher notes: {args.root / match_notes.ANNOTATIONS_DIR_NAME / match_notes.NOTES_NAME}")
    print("loopback only; use SSH -L forwarding from another machine")
    app.run(host=HOST, port=args.port, threaded=True, debug=False,
            use_reloader=False)


if __name__ == "__main__":
    main()
