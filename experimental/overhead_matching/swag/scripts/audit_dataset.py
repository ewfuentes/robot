#!/usr/bin/env python3
"""Audit a collected far-field dataset against the contracts that consume it.

Checks the things that fail silently rather than loudly: filename parsing,
frame ordering, table agreement, image integrity, and the landmark/pinhole
sidecars. Exits non-zero if any FAIL is found.

    python audit_dataset.py /data/farfield_matching/mapillary_datasets/folkestone_dover
    python audit_dataset.py /data/farfield_matching/mapillary_datasets/*/
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Above this implied speed a consecutive-frame jump is not a boat.
MAX_SPEED_MPS = 30.0


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


def haversine_m(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def audit(ds: Path) -> Audit:
    a = Audit(ds.name)

    # ── required files ────────────────────────────────────────────────────────
    required = ["pipeline_metadata.json", "pano_id_mapping.csv", "frames_gps.csv",
                "extraction_log.csv", "intrinsics.csv"]
    missing = [f for f in required if not (ds / f).exists()]
    if missing:
        a.fail(f"missing required files: {missing}")
        return a
    a.ok("all required tables present")

    meta = json.loads((ds / "pipeline_metadata.json").read_text())
    is_equirect = meta.get("is_equirectangular")

    # ── images must not have been rotated ────────────────────────────────────
    conv = meta.get("azimuth_convention") or {}
    if conv.get("images_rotated") is False and meta.get("north_aligned") is False:
        a.ok("stored unrotated (images_rotated=false, north_aligned=false)")
    else:
        a.fail(f"expected unrotated storage, got images_rotated="
               f"{conv.get('images_rotated')} north_aligned={meta.get('north_aligned')}")
    if not conv.get("formula"):
        a.fail("pipeline_metadata.azimuth_convention has no column->azimuth formula")
    if conv.get("heading_deg_is_bearing_of") not in ("column_0", "optical_axis"):
        a.fail(f"azimuth_convention.heading_deg_is_bearing_of is "
               f"{conv.get('heading_deg_is_bearing_of')!r}")

    # ── panorama/ symlink ────────────────────────────────────────────────────
    pano = ds / "panorama"
    if not pano.exists():
        a.fail("no panorama/ (landmark_filtering ingest requires it)")
        return a
    if pano.is_symlink():
        target = pano.readlink()
        if target.is_absolute():
            a.fail(f"panorama/ symlink is absolute ({target}); dataset is not relocatable")
        else:
            a.ok(f"panorama/ -> {target} (relative)")
    else:
        a.warn("panorama/ is a real directory, not a symlink")

    imgs = sorted(pano.glob("*.jpg"))
    hidden = [p.name for p in pano.iterdir() if p.name.startswith(".")]
    if hidden:
        a.fail(f"{len(hidden)} dot-file(s) in panorama/ (vigor_dataset uses iterdir "
               f"and would ingest them as phantom panos): {hidden[:3]}")
    else:
        a.ok("panorama/ free of dot-files")

    # ── filename contract ────────────────────────────────────────────────────
    bad_fields = [p.name for p in imgs if len(p.stem.split(",")) != 4]
    if bad_fields:
        a.fail(f"{len(bad_fields)} filename(s) do not split into 4 comma fields "
               f"(ingest requires pano_id,lat,lon,): {bad_fields[:2]}")
    else:
        a.ok(f"{len(imgs)} filenames parse as pano_id,lat,lon, (trailing comma)")

    ids = [p.stem.split(",")[0] for p in imgs]
    bad_ids = [i for i in ids if not (i[:1].isalpha() and i[1:].isdigit())]
    if bad_ids:
        a.fail(f"pano_ids not <letter><digits>, so int(pano_id[1:]) breaks: {bad_ids[:3]}")
    elif ids != sorted(ids) or [int(i[1:]) for i in ids] != sorted(int(i[1:]) for i in ids):
        a.fail("pano_id string sort != numeric order (ingest sorts by string)")
    else:
        a.ok("pano_ids zero-padded; string sort == numeric order")

    dupe_ids = [i for i, c in Counter(ids).items() if c > 1]
    if dupe_ids:
        a.fail(f"duplicate pano_ids: {dupe_ids[:3]}")

    # ── table agreement ──────────────────────────────────────────────────────
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

    # ── frames_gps contract (idx / dist_m / video_t_s) ───────────────────────
    idxs = [int(r["idx"]) for r in gps]
    if idxs != list(range(len(gps))):
        a.fail("frames_gps idx is not 0..N-1 contiguous")
    else:
        a.ok("frames_gps idx contiguous from 0")

    if any(int(i[1:]) != int(r["idx"]) for i, r in zip(ids, gps)):
        a.fail("pano_id numeric part does not match frames_gps idx (ingest join key)")
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

    # ── coordinates agree between filename and tables ────────────────────────
    worst = 0.0
    for p, r in zip(imgs, gps):
        _, slat, slon, _ = p.stem.split(",")
        d = haversine_m((float(slat), float(slon)),
                        (float(r["latitude"]), float(r["longitude"])))
        worst = max(worst, d)
    if worst > 1.0:
        a.fail(f"filename coords disagree with frames_gps by up to {worst:.1f} m")
    else:
        a.ok(f"filename coords match frames_gps (max {worst:.2f} m rounding)")

    # ── GPS plausibility ─────────────────────────────────────────────────────
    jumps = []
    for i in range(len(gps) - 1):
        dt = times[i + 1] - times[i]
        d = haversine_m((float(gps[i]["latitude"]), float(gps[i]["longitude"])),
                        (float(gps[i + 1]["latitude"]), float(gps[i + 1]["longitude"])))
        if dt > 0 and d / dt > MAX_SPEED_MPS:
            jumps.append((gps[i]["idx"], round(d), round(dt, 1), round(d / dt, 1)))
    if jumps:
        steps = [haversine_m((float(gps[i]["latitude"]), float(gps[i]["longitude"])),
                             (float(gps[i + 1]["latitude"]), float(gps[i + 1]["longitude"])))
                 for i in range(len(gps) - 1)]
        med_step = sorted(steps)[len(steps) // 2] if steps else 0.0
        med_dt = sorted(j[2] for j in jumps)[len(jumps) // 2]
        if med_step < 2.0:
            cause = (f" — expected here: GPS is quantized (median step "
                     f"{med_step:.1f} m), so position updates arrive in bursts "
                     f"and instantaneous speed is not meaningful")
        elif med_dt < 0.5:
            cause = (f" — median dt is only {med_dt}s, so these are metre-scale "
                     f"GPS jitter amplified by a tiny time base, not real motion")
        else:
            cause = ""
        a.warn(f"{len(jumps)} consecutive jump(s) over {MAX_SPEED_MPS} m/s "
               f"(idx, m, s, m/s): {jumps[:3]}{cause}")

        # When neither benign explanation applies and outliers are common, the
        # positions themselves are unreliable -- and because dist_m is a
        # cumulative sum over them, every teleport is added to the track length.
        # harima_b_pano reports 93.1 km this way while its median speed over the
        # same 92 minutes implies ~37 km. Say so, because dist_m reads as
        # authoritative and is what downstream along-track metrics use.
        if not cause:
            speeds = sorted(steps[i] / (times[i + 1] - times[i])
                            for i in range(len(steps))
                            if times[i + 1] > times[i])
            outlier_mps = MAX_SPEED_MPS / 2
            share = 100.0 * sum(v > outlier_mps for v in speeds) / max(1, len(speeds))
            if share > 10.0:
                med_speed = speeds[len(speeds) // 2]
                span_s = times[-1] - times[0]
                a.warn(f"{share:.1f}% of steps exceed {outlier_mps:.0f} m/s "
                       f"against a median of {med_speed:.1f} m/s — the positions "
                       f"contain frequent outliers, so dist_m "
                       f"({float(gps[-1]['dist_m']) / 1000:.1f} km) overstates the "
                       f"real track: the median speed over {span_s / 60:.0f} min "
                       f"implies ~{med_speed * span_s / 1000:.0f} km. Treat this "
                       f"dataset's GPS as noisy ground truth")
    else:
        a.ok(f"no consecutive-frame jump over {MAX_SPEED_MPS} m/s")

    # ── provenance: no duplicate source images ───────────────────────────────
    mids = [r["mapillary_id"] for r in log if r.get("mapillary_id")]
    dupe_m = [m for m, c in Counter(mids).items() if c > 1]
    if dupe_m:
        a.fail(f"{len(dupe_m)} Mapillary id(s) appear more than once: {dupe_m[:3]}")
    else:
        a.ok(f"{len(set(mids))} distinct Mapillary source images, no repeats")

    seqs = {r.get("sequence_id") for r in log if r.get("sequence_id")}
    a.ok(f"stitched from {len(seqs)} source sequence(s)")
    if not any(r.get("sequence_position") for r in log):
        a.warn("no sequence_position recorded; ordering fell back to captured_at")

    # ── intrinsics ───────────────────────────────────────────────────────────
    want_ref = "column_0" if is_equirect else "optical_axis"
    refs = {r.get("heading_reference") for r in intr}
    if refs != {want_ref}:
        a.fail(f"intrinsics heading_reference is {refs}, expected {{{want_ref}}}")
    else:
        a.ok(f"intrinsics heading_reference = {want_ref}")
    if any(r["heading_deg"] in ("", None) for r in intr):
        a.fail("some intrinsics rows have no heading_deg")

    # Heading quality. The usable signal differs by projection: equirectangular
    # frames are scored against GPS course (`heading_reliable`), but a perspective
    # camera need not point along the direction of travel, so that test would
    # reject a legitimately side-facing rig. Those are cross-checked between
    # Mapillary's two heading fields instead -- which means for perspective
    # captures this is the ONLY heading-quality signal there is, and it must not
    # be silent when it fails.
    if is_equirect:
        if meta.get("heading_reliable") is False:
            a.warn("heading_reliable=false: heading disagrees with GPS course by "
                   "more than the tolerance, so bearings need external "
                   "calibration before use")
    else:
        spread = meta.get("heading_sources_median_disagreement_deg")
        if meta.get("heading_sources_disagree"):
            a.warn(f"the two heading sources disagree by {spread}° (median) — for "
                   f"a perspective capture this is the only heading check "
                   f"available, so treat heading_deg as uncalibrated: a bearing "
                   f"built on it can be wrong by about that much")
        elif spread is not None:
            a.ok(f"heading sources agree to {spread}° (median)")
    hfovs = {float(r["hfov_deg"]) for r in intr if r["hfov_deg"]}
    if is_equirect:
        if hfovs != {360.0}:
            a.fail(f"equirect hfov should be 360, got {sorted(hfovs)[:4]}")
        else:
            a.ok("equirect hfov = 360")
    else:
        # Mapillary's SfM/EXIF focal is occasionally unphysical for a run of
        # frames, and the converter substitutes a trajectory median for those.
        # Judge the API-sourced rows on plausibility, and judge the substituted
        # ones on how many there are -- a handful is a metadata glitch, a large
        # share means the trajectory's geometry is guesswork.
        api = [r for r in intr if r.get("focal_source", "api") == "api"]
        subbed = [r for r in intr if r.get("focal_source") == "substituted_implausible"]
        hfovs = {float(r["hfov_deg"]) for r in api if r["hfov_deg"]}
        if not hfovs or min(hfovs) < 20 or max(hfovs) > 180:
            a.fail(f"implausible perspective hfov range: {sorted(hfovs)[:4]}")
        else:
            a.ok(f"perspective hfov in [{min(hfovs):.1f}, {max(hfovs):.1f}]° "
                 f"({len(hfovs)} distinct — must be applied per frame)")
        if subbed:
            # Judge the *basis* for the substitution, not its share. These are
            # fixed single-camera captures with no zoom, so the true FOV is very
            # nearly constant and a trajectory median is arguably better than
            # Mapillary's noisy per-frame SfM focals -- fukuyama_yasunari has 45
            # distinct values for one 4096x3072 camera. A large substituted share
            # is therefore not itself a defect; a thin plausible set is, because
            # then the median rests on almost nothing.
            share = 100.0 * len(subbed) / len(intr)
            note = (f"{len(subbed)} frame(s) ({share:.1f}%) had an unphysical "
                    f"API focal; intrinsics carry the trajectory median from "
                    f"{len(api)} plausible frame(s), labelled "
                    f"focal_source=substituted_implausible")
            a.fail(note) if len(api) < 30 else a.warn(note)

    # ── image integrity ──────────────────────────────────────────────────────
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
            a.ok(f"{len(sample)} sampled images decode cleanly; sizes {sorted(sizes)}")
        if is_equirect:
            bad_ar = [s for s in sizes if abs(s[0] / s[1] - 2.0) > 0.01]
            if bad_ar:
                a.fail(f"equirect images not 2:1: {bad_ar}")
            else:
                a.ok("equirect images are 2:1")
        cap = meta.get("resize_max_width")
        if cap and any(s[0] > cap for s in sizes):
            a.fail(f"image wider than resize cap {cap}: {sorted(sizes)}")

    # ── landmarks ────────────────────────────────────────────────────────────
    lm_dir = ds / "landmarks"
    version = f"{lm_dir}/v1.feather"
    if not lm_dir.exists():
        a.fail("no landmarks/ (stage 4 not run)")
    else:
        link = lm_dir / "v1.feather"
        if not link.exists():
            a.fail("landmarks/v1.feather missing (vigor_dataset opens it by name)")
        else:
            try:
                import pandas as pd
                df = pd.read_feather(link)
                need = {"id", "geometry", "landmark_type"}
                if not need.issubset(df.columns):
                    a.fail(f"landmark feather lacks {need - set(df.columns)}")
                elif "tags" not in df.columns:
                    a.warn(f"landmark feather uses the legacy wide layout "
                           f"({len(df.columns)} columns); readers handle it, but "
                           f"regenerating gives the dict schema")
                elif len(df) == 0:
                    a.fail("landmark feather has 0 rows")
                else:
                    per = len(df) / max(1, len(imgs))
                    extra = ""
                    if "tags" in df.columns:
                        import json as _json
                        n_tags = sum(len(_json.loads(t)) if isinstance(t, str)
                                     else len(t or {}) for t in df["tags"])
                        extra = f", {n_tags} tag values (dict schema)"
                    a.ok(f"landmarks: {len(df)} rows, {df['id'].nunique()} unique ids, "
                         f"{per:.1f} per frame, "
                         f"types={sorted(set(df['landmark_type']))[:3]}{extra}")
                    if df["id"].duplicated().any():
                        a.fail("duplicate landmark ids (merge requires global uniqueness)")
            except ImportError:
                a.warn("pandas unavailable; landmark feather not inspected")
            except Exception as e:
                a.fail(f"landmark feather unreadable: {str(e)[:60]}")
        prov = lm_dir / "PROVENANCE.json"
        if prov.exists():
            pj = json.loads(prov.read_text())
            a.ok(f"provenance: pbf={pj.get('osm_pbf')} enc={pj.get('enc_state')} "
                 f"enc_available={pj.get('enc_available')}")
        else:
            a.warn("landmarks/PROVENANCE.json missing")

    # Pinhole faces are audited no longer: they are an artifact
    # (artifacts/pinhole_images/<dataset>/v<N>/ with a manifest), not part of
    # the dataset contract. Their stem-match and staleness checks belong to
    # the extraction workflow that consumes them.

    return a


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    audits = [audit(Path(p).resolve()) for p in args if Path(p).is_dir()]
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
