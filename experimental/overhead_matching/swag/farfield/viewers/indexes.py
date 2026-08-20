"""Index-page generators: make the whole data root clickable.

`refresh(data_root)` regenerates the navigation chain by scanning the disk —
no hand-maintained HTML, no hardcoded dataset lists, so a page can never go
stale relative to what exists:

  <root>/index.html                     lanes with counts
  <root>/datasets/index.html            dataset table (+ per-dataset assets)
  <root>/artifacts/index.html           artifact kinds
  <root>/artifacts/<kind>/index.html    dataset -> versions (manifest info)
  <root>/runs/index.html                experiment table
  <root>/runs/<experiment>/index.html   experiment.md rendered + run table

Stage-produced pages deeper in the tree (run viewers, boards, match review)
are linked, never regenerated, from here. Every page this module writes
starts with page.GENERATED_MARK, and refresh() REFUSES to overwrite an
index.html that lacks the mark — a stage-owned or hand-made page is never
clobbered; it is reported instead.

Every stage that writes under the data root finishes by calling refresh()
(REORG.md rule 5), so pointing `python -m http.server` at the root is always
a complete, current view.
"""

import json
from pathlib import Path

from experimental.overhead_matching.swag.farfield.viewers import page as pg

GENERATOR = "farfield.viewers.indexes"

# Files worth surfacing on a dataset row when present.
DATASET_ASSETS = ("trajectory.png", "vehicle_anchor.png",
                  "gps_timelapse.mp4")

# Entry points a run directory may carry, in display order.
RUN_PAGES = (
    ("index.html", "run"),
    ("board.html", "board"),
    ("keyframes/index.html", "keyframes"),
    ("semantic_audit/review/index.html", "audit"),
    ("matching/review/index.html", "matching"),
    ("viewer.html", "filter viewer"),
    ("plots/map.png", "map"),
)


def _write_index(directory: Path, html: str, skipped: list) -> None:
    target = Path(directory) / "index.html"
    if target.exists():
        head = target.read_text(errors="replace")[:200]
        if pg.GENERATED_MARK not in head:
            skipped.append(str(target))
            return
    target.write_text(html)


def _manifest_summary(version_dir: Path) -> tuple:
    manifest = version_dir / "manifest.json"
    if not manifest.exists():
        return ("<span class='warn'>no manifest</span>", "")
    try:
        doc = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return ("<span class='bad'>unreadable manifest</span>", "")
    return (pg.esc(doc.get("generator", "?")),
            pg.esc(doc.get("created", "")))


def _dirs(path: Path) -> list:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def refresh(data_root: Path) -> dict:
    """Regenerate the index chain. Returns {"written": [...], "skipped":
    [...]} where skipped lists index.html files owned by something else."""
    data_root = Path(data_root)
    written, skipped = [], []

    def emit(directory, title, body, crumbs):
        html = pg.page(title, body, generator=GENERATOR, crumbs=crumbs)
        before = len(skipped)
        _write_index(directory, html, skipped)
        if len(skipped) == before:
            written.append(str(Path(directory) / "index.html"))

    # --- datasets lane -----------------------------------------------------
    datasets = _dirs(data_root / "datasets")
    rows = []
    for ds in datasets:
        assets = " ".join(
            f'<a href="{ds.name}/{name}">{name.split(".")[0]}</a>'
            for name in DATASET_ASSETS if (ds / name).exists())
        n_panos = len(list((ds / "panorama").glob("*.jpg"))) \
            if (ds / "panorama").exists() else 0
        rows.append([f'<a href="{ds.name}/">{pg.esc(ds.name)}</a>',
                     str(n_panos), assets or "<span class='muted'>—</span>"])
    if (data_root / "datasets").exists():
        emit(data_root / "datasets", "datasets",
             pg.table(["dataset", "panoramas", "assets"], rows),
             [("farfield", "../index.html"), ("datasets", None)])

    # --- artifacts lane ----------------------------------------------------
    kinds = _dirs(data_root / "artifacts")
    for kind in kinds:
        krows = []
        for ds in _dirs(kind):
            versions = _dirs(ds)
            cells = []
            for version in versions:
                generator, created = _manifest_summary(version)
                inner = (f'<a href="{ds.name}/{version.name}/">'
                         f'{pg.esc(version.name)}</a>')
                # Link a version's own index page when a stage produced one.
                if (version / "index.html").exists():
                    inner = (f'<a href="{ds.name}/{version.name}/'
                             f'index.html">{pg.esc(version.name)}</a>')
                cells.append(f"{inner} <span class='muted'>{generator} "
                             f"{created}</span>")
            loose = [p.name for p in ds.iterdir()
                     if p.is_file() and p.suffix in (".feather", ".json")]
            for name in sorted(loose):
                cells.append(f'<a href="{ds.name}/{name}">{pg.esc(name)}</a>')
            krows.append([f"{pg.esc(ds.name)}", "<br>".join(cells)])
        emit(kind, f"artifacts / {kind.name}",
             pg.table(["dataset", "versions"], krows),
             [("farfield", "../../index.html"),
              ("artifacts", "../index.html"), (kind.name, None)])
    if (data_root / "artifacts").exists():
        emit(data_root / "artifacts", "artifacts",
             pg.table(["kind", "datasets"], [
                 [f'<a href="{k.name}/index.html">{pg.esc(k.name)}</a>',
                  str(len(_dirs(k)))] for k in kinds]),
             [("farfield", "../index.html"), ("artifacts", None)])

    # --- runs lane (experiments) --------------------------------------------
    experiments = _dirs(data_root / "runs")
    for experiment in experiments:
        body_parts = []
        notes = experiment / "experiment.md"
        if notes.exists():
            body_parts.append(pg.render_markdown_lite(notes.read_text()))
        else:
            body_parts.append(
                "<p class='warn'>no experiment.md — every experiment "
                "directory carries one (what is being explored, status, "
                "conclusions).</p>")
        run_rows = []
        for run in _dirs(experiment):
            links = " · ".join(
                f'<a href="{run.name}/{rel}">{label}</a>'
                for rel, label in RUN_PAGES if (run / rel).exists())
            run_rows.append([pg.esc(run.name),
                             links or "<span class='muted'>no pages</span>"])
        body_parts.append(pg.table(["run", "pages"], run_rows))
        emit(experiment, experiment.name, "\n".join(body_parts),
             [("farfield", "../../index.html"), ("runs", "../index.html"),
              (experiment.name, None)])
    if (data_root / "runs").exists():
        rows = []
        for experiment in experiments:
            first_line = ""
            notes = experiment / "experiment.md"
            if notes.exists():
                for line in notes.read_text().splitlines():
                    if line.strip() and not line.startswith("#"):
                        first_line = line.strip()
                        break
            rows.append([
                f'<a href="{experiment.name}/index.html">'
                f'{pg.esc(experiment.name)}</a>',
                str(len(_dirs(experiment))), pg.esc(first_line)])
        emit(data_root / "runs", "runs (experiments)",
             pg.table(["experiment", "runs", "summary"], rows),
             [("farfield", "../index.html"), ("runs", None)])

    # --- root ----------------------------------------------------------------
    lane_rows = []
    for lane, blurb in (("datasets", "frozen problem definitions"),
                        ("artifacts", "derived per-dataset products"),
                        ("runs", "experiments (localization runs)"),
                        ("models", "weights"),
                        ("raw_material", "source material")):
        path = data_root / lane
        if not path.exists():
            continue
        href = (f"{lane}/index.html"
                if (path / "index.html").exists()
                or lane in ("datasets", "artifacts", "runs") else f"{lane}/")
        lane_rows.append([f'<a href="{href}">{pg.esc(lane)}</a>',
                          str(len(_dirs(path))), pg.esc(blurb)])
    emit(data_root, "farfield data root",
         pg.table(["lane", "entries", ""], lane_rows),
         None)

    return {"written": written, "skipped": skipped}
