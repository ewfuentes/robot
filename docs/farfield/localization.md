# Localization: exports, filter runs, forensics

## The seam

`localization/structs.py` is the data contract between the pipeline and the
filter: `TrackletMeasurement` (one fused body-frame bearing per tracklet per
information epoch, serialized in [0, 360)), `OdometryDelta` (body-frame SE(2)
increments), `CompatibilityTable` (the matcher seam — landmarks absent from a
table score its default, and the filter clips every log-LR).

`localization:build_export` produces an export from a tracking run:
tracklets come straight from tracks + audit (audit membership is the gate;
there is no merge stage), the mount offset is baked in here with its source
recorded, odometry derives from GPS course, and `--tables` selects the
matcher's compatibility tables or the uninformative floor. The export is
read back through its own loader before it is declared done;
`export_ingest.load` refuses an export whose mount-offset provenance is
missing or in the wrong frame.

## Running the filter

`localization:run_export` runs the filter on an export and writes a
self-describing run directory (manifest with the full config echo, git
commit, argv, the export path, and the catalog visibility radius — the
recorded value is the replay contract). `localization:run_localization` does
the same for synthetic scenarios.

**Evaluation rule: only uniform-prior whole-map runs are evaluations.**
`--init truth` is a diagnostic instrument (basin-of-attraction control) and
is labelled as such in its output; never report it as a result. An
odometry-only control (`--no_bearings`) writes a run directory containing
exactly what the filter consumed — empty measurement files — so the record
always describes the run that happened.

What a GPS-supervised export can honestly support is printed at the end of
every run: **bearing residuals** against the filter's own pose and the
**association posteriors**. Final position error is a sanity check, not the
figure of merit — GPS odometry nearly solves a leg by itself.

## Reading a run

- `localization:viewer` — the self-contained single-file run viewer
  (offline basemap, checkpoints, associations, proposal events).
  `localization:viewer_server` adds live replay/crop endpoints on top of the
  same payload.
- `localization:plot_run` — map/strip/animation images into the run dir.
- `localization:satellite_underlay` — optional imagery underlay (licensed
  tiles; not redistributed).
- `localization:forensics` — the CLI that answers "why did this leg fail":
  `check` (invariants), `attribute` (per-measurement error attribution),
  `replay` (bit-exact re-run + counterfactual edits, into the run's own
  `counterfactuals/`), `tracklet` (one tracklet's whole story), `events`,
  `triage` (the ranked what-went-wrong table).

All of it reads only the run directory. Every page and plot lands inside the
run (never `/tmp`), so serving the data root reaches everything.
