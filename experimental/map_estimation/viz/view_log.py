"""View an Argoverse 2 log in rerun: the vehicle and the path it drove.

    V="bazel run //experimental/map_estimation/viz:view_log --"

    $V sensor/val                              # list what is on disk and exit
    $V sensor/val --log_id 02678d04-cc9f-3148-9f95-1ba66347dff9
    $V tbv --log_id 07YOTz... --serve          # stream to a browser instead
    $V sensor/val --log_id 02678d04... --save /tmp/log.rrd

The dataset spec, ``--log_id`` and ``--root`` mean what they mean in the download CLI
(``//experimental/map_estimation/data:argoverse``), and paths are resolved through the same
layout module, so whatever that tool downloaded is what this one opens.
"""

import argparse
import logging
import sys
from pathlib import Path

import rerun as rr

from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.viz import av2_scene, av2_source

APPLICATION_ID = "argoverse_log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="view_log",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spec", type=str,
                        help="dataset[/split], e.g. sensor/val, tbv, lidar/train")
    parser.add_argument("--log_id", type=str, default=None,
                        help="log to view; omit to list the logs present on disk")
    parser.add_argument("--root", type=Path, default=al.DEFAULT_ROOT,
                        help=f"local dataset root (default: {al.DEFAULT_ROOT})")
    parser.add_argument("--verbose", "-v", action="count", default=0)

    sink = parser.add_mutually_exclusive_group()
    sink.add_argument("--spawn", action="store_true",
                      help="open the native viewer (default)")
    sink.add_argument("--serve", action="store_true",
                      help="serve to a browser; use this when the data is on a remote box")
    sink.add_argument("--save", type=Path, default=None,
                      help="write an .rrd file instead of opening a viewer")
    return parser


def _bundled_viewer_path() -> Path:
    """Path to the viewer executable shipped inside the rerun-sdk wheel.

    Resolved the way the wheel resolves it for itself -- ``rerun_cli/__main__.py`` runs
    ``os.path.dirname(__file__) + "/rerun"`` -- so this follows the binary if a future version
    moves it, and fails loudly if one stops shipping it.
    """
    import rerun_cli

    executable = Path(rerun_cli.__file__).parent / "rerun"
    if not executable.is_file():
        raise FileNotFoundError(f"rerun-sdk shipped no viewer executable at {executable}")
    return executable


def _spawn_bundled_viewer(blueprint, *, port: int = 9876) -> None:
    """Spawn the wheel's own viewer and stream this recording to it.

    Not ``rr.spawn()``. That calls the Rust ``spawn`` without an ``executable_path``, which
    falls through to handing the bare name "rerun" to the OS -- so the viewer has to be on
    ``$PATH``. pip normally puts it there via a console script generated at install time, and
    Bazel unpacks wheels without running that step, leaving the 100 MB viewer sitting in the
    runfiles where nothing will look for it.

    Passing ``executable_path`` takes the first branch of the Rust lookup, which uses the path
    verbatim and never consults ``$PATH`` at all.
    """
    import rerun_bindings

    rerun_bindings.spawn(port=port, executable_path=str(_bundled_viewer_path()))
    rr.connect_grpc(f"rerun+http://127.0.0.1:{port}/proxy", default_blueprint=blueprint)


def _list_logs(request: al.Request, root: Path) -> int:
    log_ids = av2_source.discover_log_ids(request, root)
    if not log_ids:
        print(f"no logs on disk under {request.local_dir(root)}", file=sys.stderr)
        return 1
    print(f"{len(log_ids)} log(s) under {request.local_dir(root)}:")
    for log_id in log_ids:
        source = av2_source.LogSource(request, log_id, root)
        items = ", ".join(item.token for item in source.present_items())
        print(f"  {log_id}  [{items}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # A bad spec, a missing log, or a log without poses are all ordinary user errors here, so
    # they get a one-line message rather than a traceback.
    try:
        request = al.make_request(args.spec)
        if args.log_id is None:
            return _list_logs(request, args.root)
        source = av2_source.LogSource(request, args.log_id, args.root)
    except (al.UnknownSplitError, al.UnknownItemError, av2_source.MissingStreamError,
            av2_source.UnsupportedDatasetError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # init before logging, and pick the sink before anything is sent: rerun buffers what is
    # logged, so a sink chosen afterwards still receives it, but --save wants the file opened up
    # front rather than flushed at exit.
    rr.init(APPLICATION_ID, recording_id=args.log_id)
    blueprint = av2_scene.default_blueprint()
    if args.save is not None:
        rr.save(args.save, default_blueprint=blueprint)
    elif args.serve:
        rr.serve_web(default_blueprint=blueprint)
    else:
        _spawn_bundled_viewer(blueprint)

    try:
        summary = av2_scene.log_ego_motion(source)
    except av2_source.MissingStreamError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"{summary.log_id}: {summary.poses} poses, "
        f"{summary.path_length_m:.1f} m over {summary.duration_s:.1f} s"
    )

    if args.save is not None:
        print(f"wrote {args.save}")
    elif args.serve:
        print("serving; ctrl-c to stop")
        try:
            while True:
                input()
        except (EOFError, KeyboardInterrupt):
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
