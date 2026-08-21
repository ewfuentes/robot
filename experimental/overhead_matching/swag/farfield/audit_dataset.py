"""Audit a farfield dataset against the contracts that consume it.

Checks the things that fail silently rather than loudly: filename parsing,
frame ordering, table agreement, convention metadata (north_aligned, azimuth
formula, mount-offset qualifiers), image integrity, GPS plausibility, and
video addressing. Exits non-zero if any FAIL is found; a path that is not a
dataset directory is itself an error, never silently skipped.

    bazel run //experimental/overhead_matching/swag/farfield:audit_dataset -- \
        /data/farfield_matching/datasets/folkestone_dover [more dirs...]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from experimental.overhead_matching.swag.farfield import dataset as ds_lib
from experimental.overhead_matching.swag.farfield import geometry as geo

# Above this implied speed a consecutive-frame jump is not a vehicle.
MAX_SPEED_MPS = 30.0

# Correctly-addressed frames cross-correlate 0.98-0.9999 against their
# panorama; charles_river_20260727's 510 s-offset frames scored 0.31-0.56.
MIN_VIDEO_NCC = 0.90


class Audit:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def ok(self, msg):
        self.rows.append(("ok", msg))

    def warn(self, msg):
        self.rows.append(("warn", msg))

    def fail(self, msg):
        self.rows.append(("FAIL", msg))

    @property
    def failed(self):
        return any(k == "FAIL" for k, _ in self.rows)

    def report(self):
        n_fail = sum(1 for k, _ in self.rows if k == "FAIL")
        n_warn = sum(1 for k, _ in self.rows if k == "warn")
        print(f"\n=== {self.name}  ({n_fail} fail, {n_warn} warn)")
        for kind, msg in self.rows:
            mark = {"ok": "  ok  ", "warn": "  WARN", "FAIL": "  FAIL"}[kind]
            print(f"{mark}  {msg}")


def _gray(array, size=(640, 320)):
    from PIL import Image
    import numpy as np
    im = Image.fromarray(array) if not isinstance(array, Image.Image) else array
    return np.asarray(im.convert("L").resize(size), dtype=float)


def _ncc(a, b):
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float((((a - a.mean()) / a.std()) * ((b - b.mean()) / b.std())).mean())


def check_video_addressing(a, ds, meta, gps, imgs):
    """Does `video_t_s` actually address the frame it claims?

    `video_t_s` is the address the tracking stages seek to in the source
    video, and nothing downstream can tell a wrong address from a right one: a
    seek lands on a real frame either way, SAM2 happily tracks whatever is
    there, and the run completes. charles_river's trim rebased the column to
    zero, so every window came from 510 s earlier in the sail; the only
    visible symptom was tracks that made no sense. Every other check in this
    file passed on that dataset, which is why this one exists: decode what the
    column points at and compare it to the panorama it is supposed to be.
    """
    raw = ((meta.get("video") or {}).get("source_video") or "").split(" (")[0].strip()
    if not raw:
        a.ok("no source video declared (nothing to address)")
        return
    path = Path(raw)
    if not path.is_absolute():
        # <root>/datasets/<name> -> <root>; resolve() so a symlink shell does
        # not send the search off to the wrong root.
        path = ds.resolve().parent.parent / path
    if not path.exists():
        a.warn(f"declared source video is absent on disk ({path}); tracking "
               f"stages will fail until it is restored")
        return
    try:
        import cv2
        import numpy as np  # noqa: F401
        from PIL import Image
    except ImportError as exc:
        a.warn(f"{exc.name} unavailable; skipped video addressing check")
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        a.fail(f"could not open declared source video {path}")
        return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            a.warn(f"source video reports fps={fps}; skipped addressing check")
            return
        picks = [i for i in (len(gps) // 4, len(gps) // 2, 3 * len(gps) // 4)
                 if 0 <= i < len(gps)] or [0]
        scores, offset_scores = [], []
        # If the address is wrong, the likeliest cause is a trim that rebased
        # the column, and the amount cut is recorded -- so report the fix too.
        shift = float((meta.get("video") or {}).get("export_start_video_t_s") or 0.0)
        for i in picks:
            t_s = float(gps[i]["video_t_s"])
            pano = _gray(Image.open(imgs[i]))
            for delta, sink in ((0.0, scores), (shift, offset_scores)):
                if delta == 0.0 and sink is offset_scores:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((t_s + delta) * fps)))
                ok, bgr = cap.read()
                if not ok:
                    sink.append(0.0)
                    continue
                sink.append(_ncc(_gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), pano))
    finally:
        cap.release()
    if not scores:
        a.warn("could not decode any sampled video frame; addressing unchecked")
        return
    worst = min(scores)
    if worst >= MIN_VIDEO_NCC:
        a.ok(f"video_t_s addresses {path.name} correctly "
             f"(frame match {worst:.3f}-{max(scores):.3f} over {len(scores)})")
        return
    hint = ""
    if offset_scores and min(offset_scores) >= MIN_VIDEO_NCC:
        hint = (f" Adding export_start_video_t_s ({shift} s) fixes it "
                f"(match {min(offset_scores):.3f}), so the column was rebased "
                f"by a trim: restore it with video_t_s += {shift}.")
    a.fail(f"video_t_s does NOT address {path.name}: sampled frames match "
           f"their panoramas at only {worst:.3f}-{max(scores):.3f} "
           f"(want >= {MIN_VIDEO_NCC}). Tracking would crop its windows from "
           f"the wrong part of the video.{hint}")


def read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def audit(ds: Path) -> Audit:
    a = Audit(ds.name)

    # -- required files -------------------------------------------------------
    required = ["pipeline_metadata.json", "pano_id_mapping.csv",
                "frames_gps.csv", "extraction_log.csv", "intrinsics.csv"]
    missing = [f for f in required if not (ds / f).exists()]
    if missing:
        a.fail(f"missing required files: {missing}")
        return a
    a.ok("all required tables present")

    meta = json.loads((ds / "pipeline_metadata.json").read_text())
    is_equirect = meta.get("is_equirectangular")

    # -- images must not have been rotated ------------------------------------
    conv = meta.get("azimuth_convention") or {}
    if conv.get("images_rotated") is False and meta.get("north_aligned") is False:
        a.ok("stored unrotated (images_rotated=false, north_aligned=false)")
    else:
        a.fail(f"expected unrotated storage, got images_rotated="
               f"{conv.get('images_rotated')} north_aligned="
               f"{meta.get('north_aligned')} — the tracking pipeline refuses "
               f"datasets whose orientation is rotated or unrecorded")
    if not conv.get("formula"):
        a.fail("pipeline_metadata.azimuth_convention has no column->azimuth formula")
    if conv.get("heading_deg_is_bearing_of") not in ("column_0", "optical_axis"):
        a.fail(f"azimuth_convention.heading_deg_is_bearing_of is "
               f"{conv.get('heading_deg_is_bearing_of')!r}")
    if "mount_offset_frame" not in conv:
        a.warn("azimuth_convention carries no mount_offset_frame note; a "
               "reader deriving an offset from this formula lands in the "
               "column-0 frame, 180 deg out (the pohang incident)")

    # -- mount offset qualifiers (the 180-degree contract) ---------------------
    try:
        record = ds_lib.mount_offset_record(meta, ds)
    except ds_lib.ContractViolation as exc:
        a.fail(str(exc).replace("\n", " "))
    else:
        if record is None:
            a.warn("no mount_offset recorded (fine for an uncalibrated "
                   "dataset; localization export will require one)")
        else:
            flag = ("accuracy-validated" if record.accuracy_validated
                    else f"status={record.status or 'unvalidated'}")
            a.ok(f"mount_offset {record.offset_deg:.1f} deg, frame OK, "
                 f"applied_to_heading_deg={record.applied_to_heading_deg}, "
                 f"{flag}")

    # -- panorama/ symlink -----------------------------------------------------
    pano = ds / "panorama"
    if not pano.exists():
        a.fail("no panorama/ (ingest requires it)")
        return a
    if pano.is_symlink():
        target = pano.readlink()
        if target.is_absolute():
            a.fail(f"panorama/ symlink is absolute ({target}); dataset is "
                   f"not relocatable")
        else:
            a.ok(f"panorama/ -> {target} (relative)")
    else:
        a.warn("panorama/ is a real directory, not a symlink")

    imgs = sorted(pano.glob("*.jpg"))
    hidden = [p.name for p in pano.iterdir() if p.name.startswith(".")]
    if hidden:
        a.fail(f"{len(hidden)} dot-file(s) in panorama/ (dataset loaders use "
               f"iterdir and would ingest them as phantom panos): {hidden[:3]}")
    else:
        a.ok("panorama/ free of dot-files")

    # -- filename contract -----------------------------------------------------
    bad_fields = [p.name for p in imgs if len(p.stem.split(",")) != 4]
    if bad_fields:
        a.fail(f"{len(bad_fields)} filename(s) do not split into 4 comma "
               f"fields (ingest requires pano_id,lat,lon,): {bad_fields[:2]}")
    else:
        a.ok(f"{len(imgs)} filenames parse as pano_id,lat,lon, (trailing comma)")

    ids = [p.stem.split(",")[0] for p in imgs]
    bad_ids = [i for i in ids if not (i[:1].isalpha() and i[1:].isdigit())]
    if bad_ids:
        a.fail(f"pano_ids not <letter><digits>, so int(pano_id[1:]) breaks: "
               f"{bad_ids[:3]}")
    elif ids != sorted(ids) or [int(i[1:]) for i in ids] != sorted(
            int(i[1:]) for i in ids):
        a.fail("pano_id string sort != numeric order (ingest sorts by string)")
    else:
        a.ok("pano_ids zero-padded; string sort == numeric order")

    dupe_ids = [i for i, c in Counter(ids).items() if c > 1]
    if dupe_ids:
        a.fail(f"duplicate pano_ids: {dupe_ids[:3]}")

    numeric = [int(i[1:]) for i in ids]
    if numeric and numeric != list(range(numeric[0], numeric[0] + len(numeric))):
        a.warn(f"pano numbers are not contiguous ({len(numeric)} panos, "
               f"range {numeric[0]}-{numeric[-1]}): pano_id[1:] != frame_idx "
               f"after the first gap, so anything parsing ids for indices "
               f"breaks (use dataset.frame_index_by_pano_id)")

    # -- table agreement --------------------------------------------------------
    gps = read_csv(ds / "frames_gps.csv")
    mapping = read_csv(ds / "pano_id_mapping.csv")
    log = read_csv(ds / "extraction_log.csv")
    intr = read_csv(ds / "intrinsics.csv")
    counts = {"panorama/": len(imgs), "frames_gps": len(gps),
              "pano_id_mapping": len(mapping), "extraction_log": len(log),
              "intrinsics": len(intr)}
    if len(set(counts.values())) == 1:
        a.ok(f"row counts agree across all tables and images ({len(imgs)})")
    else:
        a.fail(f"row counts disagree: {counts}")

    # -- frames_gps contract (idx / dist_m / video_t_s) --------------------------
    idxs = [int(r["idx"]) for r in gps]
    if idxs != list(range(len(gps))):
        a.fail("frames_gps idx is not 0..N-1 contiguous")
    else:
        a.ok("frames_gps idx contiguous from 0")

    if any(int(i[1:]) != int(r["idx"]) for i, r in zip(ids, gps)):
        a.fail("pano_id numeric part does not match frames_gps idx (ingest "
               "join key)")
    else:
        a.ok("pano_id[1:] == frames_gps idx (join key sound)")

    dists = [float(r["dist_m"]) for r in gps]
    if any(b < a_ for a_, b in zip(dists, dists[1:])):
        a.fail("dist_m is not monotonically non-decreasing")
    else:
        a.ok(f"dist_m monotonic, total {dists[-1]/1000:.2f} km")

    times = [float(r["video_t_s"]) for r in gps]
    if any(b < a_ for a_, b in zip(times, times[1:])):
        a.fail("video_t_s decreases somewhere (frames out of temporal order)")
    else:
        a.ok(f"video_t_s monotonic, span {times[-1]/60:.1f} min")

    # -- coordinates agree between filename and tables ---------------------------
    worst = 0.0
    for p, r in zip(imgs, gps):
        _, slat, slon, _ = p.stem.split(",")
        d = geo.haversine_m(float(slat), float(slon),
                            float(r["latitude"]), float(r["longitude"]))
        worst = max(worst, d)
    if worst > 1.0:
        a.fail(f"filename coords disagree with frames_gps by up to {worst:.1f} m")
    else:
        a.ok(f"filename coords match frames_gps (max {worst:.2f} m rounding)")

    # -- GPS plausibility ---------------------------------------------------------
    jumps = []
    for i in range(len(gps) - 1):
        dt = times[i + 1] - times[i]
        d = geo.haversine_m(
            float(gps[i]["latitude"]), float(gps[i]["longitude"]),
            float(gps[i + 1]["latitude"]), float(gps[i + 1]["longitude"]))
        if dt > 0 and d / dt > MAX_SPEED_MPS:
            jumps.append((gps[i]["idx"], round(d), round(dt, 1),
                          round(d / dt, 1)))
    if jumps:
        steps = [geo.haversine_m(
            float(gps[i]["latitude"]), float(gps[i]["longitude"]),
            float(gps[i + 1]["latitude"]), float(gps[i + 1]["longitude"]))
            for i in range(len(gps) - 1)]
        med_step = sorted(steps)[len(steps) // 2] if steps else 0.0
        med_dt = sorted(j[2] for j in jumps)[len(jumps) // 2]
        if med_step < 2.0:
            cause = (f" — expected here: GPS is quantized (median step "
                     f"{med_step:.1f} m), so position updates arrive in "
                     f"bursts and instantaneous speed is not meaningful")
        elif med_dt < 0.5:
            cause = (f" — median dt is only {med_dt}s, so these are "
                     f"metre-scale GPS jitter amplified by a tiny time base, "
                     f"not real motion")
        else:
            cause = ""
        a.warn(f"{len(jumps)} consecutive jump(s) over {MAX_SPEED_MPS} m/s "
               f"(idx, m, s, m/s): {jumps[:3]}{cause}")

        # When neither benign explanation applies and outliers are common,
        # the positions themselves are unreliable -- and because dist_m is a
        # cumulative sum over them, every teleport inflates the track length.
        if not cause:
            speeds = sorted(steps[i] / (times[i + 1] - times[i])
                            for i in range(len(steps))
                            if times[i + 1] > times[i])
            outlier_mps = MAX_SPEED_MPS / 2
            share = 100.0 * sum(v > outlier_mps for v in speeds) / max(
                1, len(speeds))
            if share > 10.0:
                med_speed = speeds[len(speeds) // 2]
                span_s = times[-1] - times[0]
                a.warn(f"{share:.1f}% of steps exceed {outlier_mps:.0f} m/s "
                       f"against a median of {med_speed:.1f} m/s — the "
                       f"positions contain frequent outliers, so dist_m "
                       f"({float(gps[-1]['dist_m']) / 1000:.1f} km) "
                       f"overstates the real track: the median speed over "
                       f"{span_s / 60:.0f} min implies "
                       f"~{med_speed * span_s / 1000:.0f} km. Treat this "
                       f"dataset's GPS as noisy ground truth")
    else:
        a.ok(f"no consecutive-frame jump over {MAX_SPEED_MPS} m/s")

    # -- provenance: no duplicate source images ------------------------------------
    mids = [r["mapillary_id"] for r in log if r.get("mapillary_id")]
    dupe_m = [m for m, c in Counter(mids).items() if c > 1]
    if dupe_m:
        a.fail(f"{len(dupe_m)} Mapillary id(s) appear more than once: "
               f"{dupe_m[:3]}")
    elif mids:
        a.ok(f"{len(set(mids))} distinct Mapillary source images, no repeats")

    seqs = {r.get("sequence_id") for r in log if r.get("sequence_id")}
    if seqs:
        a.ok(f"stitched from {len(seqs)} source sequence(s)")
        if not any(r.get("sequence_position") for r in log):
            a.warn("no sequence_position recorded; ordering fell back to "
                   "captured_at")

    # -- intrinsics ------------------------------------------------------------------
    want_ref = "column_0" if is_equirect else "optical_axis"
    refs = {r.get("heading_reference") for r in intr}
    if refs != {want_ref}:
        a.fail(f"intrinsics heading_reference is {refs}, expected "
               f"{{{want_ref}}}")
    else:
        a.ok(f"intrinsics heading_reference = {want_ref}")
    if any(r["heading_deg"] in ("", None) for r in intr):
        a.fail("some intrinsics rows have no heading_deg")

    # Heading quality. Equirectangular frames are scored against GPS course
    # (`heading_reliable`); a perspective camera need not point along the
    # direction of travel, so those are cross-checked between Mapillary's two
    # heading fields instead -- the ONLY heading-quality signal they have.
    if is_equirect:
        if meta.get("heading_reliable") is False:
            a.warn("heading_reliable=false: heading disagrees with GPS "
                   "course by more than the tolerance, so bearings need "
                   "external calibration before use")
    else:
        spread = meta.get("heading_sources_median_disagreement_deg")
        if meta.get("heading_sources_disagree"):
            a.warn(f"the two heading sources disagree by {spread} deg "
                   f"(median) — for a perspective capture this is the only "
                   f"heading check available, so treat heading_deg as "
                   f"uncalibrated: a bearing built on it can be wrong by "
                   f"about that much")
        elif spread is not None:
            a.ok(f"heading sources agree to {spread} deg (median)")
    hfovs = {float(r["hfov_deg"]) for r in intr if r["hfov_deg"]}
    if is_equirect:
        if hfovs != {360.0}:
            a.fail(f"equirect hfov should be 360, got {sorted(hfovs)[:4]}")
        else:
            a.ok("equirect hfov = 360")
    else:
        api = [r for r in intr if r.get("focal_source", "api") == "api"]
        subbed = [r for r in intr
                  if r.get("focal_source") == "substituted_implausible"]
        hfovs = {float(r["hfov_deg"]) for r in api if r["hfov_deg"]}
        if not hfovs or min(hfovs) < 20 or max(hfovs) > 180:
            a.fail(f"implausible perspective hfov range: {sorted(hfovs)[:4]}")
        else:
            a.ok(f"perspective hfov in [{min(hfovs):.1f}, {max(hfovs):.1f}] "
                 f"deg ({len(hfovs)} distinct — must be applied per frame)")
        if subbed:
            # Judge the *basis* for the substitution, not its share: these are
            # fixed single-camera captures, so a trajectory median is fine --
            # unless the plausible set it rests on is thin.
            share = 100.0 * len(subbed) / len(intr)
            note = (f"{len(subbed)} frame(s) ({share:.1f}%) had an unphysical "
                    f"API focal; intrinsics carry the trajectory median from "
                    f"{len(api)} plausible frame(s), labelled "
                    f"focal_source=substituted_implausible")
            a.fail(note) if len(api) < 30 else a.warn(note)

    # -- image integrity ------------------------------------------------------------
    try:
        from PIL import Image
    except ImportError:
        a.warn("PIL unavailable; skipped image integrity")
    else:
        sample = imgs[:: max(1, len(imgs) // 25)]
        sizes, broken = set(), []
        for p in sample:
            try:
                im = Image.open(p)
                im.verify()
                sizes.add(Image.open(p).size)
            except Exception as e:
                broken.append((p.name, str(e)[:40]))
        if broken:
            a.fail(f"{len(broken)} unreadable image(s) in sample: {broken[:2]}")
        else:
            a.ok(f"{len(sample)} sampled images decode cleanly; sizes "
                 f"{sorted(sizes)}")
        if is_equirect:
            bad_ar = [s for s in sizes if abs(s[0] / s[1] - 2.0) > 0.01]
            if bad_ar:
                a.fail(f"equirect images not 2:1: {bad_ar}")
            elif sizes:
                a.ok("equirect images are 2:1")
        cap = meta.get("resize_max_width")
        if cap and any(s[0] > cap for s in sizes):
            a.fail(f"image wider than resize cap {cap}: {sorted(sizes)}")

    check_video_addressing(a, ds, meta, gps, imgs)

    # -- legacy in-dataset catalogs ---------------------------------------------------
    if (ds / "landmarks").exists():
        a.warn("legacy landmarks/ directory inside the dataset: catalogs are "
               "derived products and live in artifacts/catalogs/<dataset>/ "
               "(REORG.md data migration moves them)")

    return a


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_paths", nargs="+", type=Path,
                        help="Dataset directories to audit")
    args = parser.parse_args(argv)

    bad_paths = [p for p in args.dataset_paths if not p.is_dir()]
    if bad_paths:
        parser.error(
            "not a directory: " + ", ".join(str(p) for p in bad_paths))

    audits = [audit(p.resolve()) for p in args.dataset_paths]
    for a in audits:
        a.report()
    bad = [a.name for a in audits if a.failed]
    print(f"\n{len(audits) - len(bad)}/{len(audits)} datasets clean")
    if bad:
        print(f"failures in: {', '.join(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
