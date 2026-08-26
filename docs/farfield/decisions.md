# Far-field decision journal

Append-only, newest last. One entry per decision that a later reader could
otherwise reasonably undo.

**What belongs here.** A rule in this codebase is usually the scar of a
measurement — a filter that drifted 13 km, a prompt clause that tripled
misdirection, a vocabulary choice worth 6,360 rows. The rule survives in the
code; the measurement that justifies it does not, unless it is written down.
Without it the rule looks arbitrary, and the next person to read that function
— human or agent — simplifies it back out and re-pays the cost of learning.

**What does not belong here.** Current values (they go stale — the code and
the build configs are authoritative), API documentation, or anything derivable
from `git log`. This is the record of *why*, and specifically of why-with-a-
number.

**How to write an entry.** Date it, say what changed, and give the evidence in
the units it was measured in, naming the run or dataset it came from. If
something was tried and rejected, say so explicitly and say what it cost —
those are the entries that actually prevent repeat work. Leave a pointer
comment at the code site (`# see docs/farfield/decisions.md, <date>`) so the
rule and its reason can find each other.

---

## 2026-05 — 2026-08 · Bearing-only filter: five fixes from the whole-map campaign

Recovered from comments on the pre-reorg filter; each number below cost a
whole-map run to measure. The code implementing all five is current in
`localization/filter.py`.

**Association persistence — pay the identity prior once, not per epoch.**
Marginalising identity independently per epoch re-pays the `1/|catalog|`
candidate prior on *every* epoch of a tracklet, which couples bearing evidence
to catalog size. That dilution is what kept the whole-map harbour run pinned to
the null. Persistence pays the prior once per tracklet; subsequent epochs are
pure geometry.

**Normalise the identity posterior — `w_j * LR_j` directly leaves the mixture
improper.** The landmark branch then integrates to
`(1-pi0) * sum_j w_j LR_j`, a table-dependent constant measured at ~0.02 on the
whole-map harbour tables, so the effective clutter share was ~92% rather than
the configured `pi0`. Worse, the tables' `default_log_lr` summed over 13,208
unlisted rows claimed 87% odds that the true landmark was one the matcher
*rejected*, against a measured miss rate of ~40%. Under that arithmetic a
perfect top-endorsed explanation scored *below* "call it clutter" (0.07 against
`1/2pi` = 0.159 per radian), so the filter null-committed good tracklets
whenever its pose was a few hundred metres off — muting exactly the bearings
that would have corrected it. This was the exp5 drift.

**Never commit to an unendorsed candidate.** Letting a particle commit to the
best-aligned of hundreds of in-cone default candidates and then ride full von
Mises concentration is data-association overfitting. Measured on the whole-map
run: the dominant mode anchored itself on cherry-picked `place=square`-grade
nodes — a residual of ~0.2° beats the true landmark's honest 2° by `e^2` per
epoch — and drifted 13 km while holding 80% of the posterior. A background
commitment scores `1/(2 pi)`, and uniform cannot compound.

**Regularised resampling.** Plain resampling replaces the posterior with Dirac
atoms drawn only from locations already present, so repeated resampling
collapses diversity while the filter grows confident about a spread it no
longer represents. Measured on `harbor_loop`: final-state NEES ran 67 at 4k
particles and 12 at 20k, against an ideal of 2.0 — the signature of
impoverishment, not of a wrong model.

**Stratified allocation, per-group bandwidth.** Resampling is a
representational step: mass may move between hypotheses only through evidence,
never through sampling noise. Globally-drawn resampling let exactly symmetric
twin modes drift to 87/13 within 32 resamples (measured, T-F3 world);
stratified allocation holds them balanced until evidence separates them. One
global bandwidth is only right for a unimodal belief — during global
localization the belief is a set of tight hypotheses inside a region-scale
cloud, and the region-scale bandwidth (4.6 km median on the whole-map harbour
run) re-diffused every cluster at every resample. That run repeatedly found the
true pose and was un-finding it. Known approximation: an arc-shaped proposal
cluster gets an isotropic bandwidth from its own elongated spread, acceptable
because injected clusters are re-scored by the very next measurement.

**Related, same campaign.** The evidence gate scores an event by the poses that
actually contribute, never by its luckiest single pose: a tie-storm event
scores ~10^4 poses, and max-of-all-poses hands junk a multiple-comparisons
bonus of several nats — measured, a 597-hypothesis event cleared the margin
against a converged incumbent at kf 300 and displaced half of a 50 m-accurate
belief by 10 km. And the incumbent must be scored through its own mode, not
the plain mixture: the mixture-referenced gate passed junk injections against a
mode holding half the posterior within 1.5 km of truth, and the resulting churn
destroyed that mode during a junk-measurement stretch (kf 280–330). An absolute
null-share trigger cannot substitute — in a dense catalog the null share stays
high even while tracking the true pose, and 25 of 27 injections on the
whole-map harbour run carried no truth-consistent hypothesis, each displacing
half the belief. The filter repeatedly found the pose and was reset off it by
its own recovery mechanism.

## 2026-08 · Extraction prompt v2: three clauses in, one clause permanently out

v1's `<naming_rules>` licensed recognition without constraining it to the
structure's own visible features, which produced whole-scene misrecognition (a
Boston Harbor panorama named with the Chicago lakefront) and bare buoy board
numbers as names.

Three clauses were added, all measured on a 22-frame control set, 3 passes
each: name the STRUCTURE not the SCENE; lookalikes need a differentiator you
can see; a painted number or letter is `ref=`, never `name=`. Out-of-region
names and designator-as-name both went to zero, and precision rose 66% → 74%.

**A fourth clause was tried and rejected. Do not add it back.** "All the names
you give for one panorama must belong to ONE locality" reads as protective but
*tripled* in-region misdirection (real wrong-bearing names 0.7 → 2.3 per pass).
It appears to make the model commit to a locality and then name more things in
it.

**Known cost of v2, unmeasured end-to-end:** fewer names overall (30 → 20 per
pass) and fewer mountain-peak names (6.3 → 3.7). A carve-out exempting summits
from the lookalike clause was tried and made everything worse (58% precision,
2.7 peak names), so the peak loss is *not* caused by that clause's ridge
example, and its mechanism is still unknown.

Prompt text is a versioned artifact: every extraction run records
`prompt_sha256`. Add a new key rather than editing a shipped one.

## 2026-08 · Far-field tag vocabulary is not the street-level keep-list

`FAR_FIELD_KEEP_KEYS` is named for what it selects — tags a distant observer
could judge — not for the environment it was first written in. At street level
the nearest landmark is metres away and a shopfront's opening hours are
legible; here most landmarks are kilometres off. So it adds the maritime
vocabulary harbour tables actually carry, above all `seamark:*` and ENC's
`object_class` (surveyed navigation aids are the single most matchable class we
have), and drops operational street furniture — `addr:*`, lanes, surface,
opening hours, payment — that cannot be observed at range and only dilutes the
bundle.

**`religion` / `denomination` are kept deliberately.**
`amenity=place_of_worship` alone is nearly free of information in an Anglophone
harbour (~90% christian) and the opposite elsewhere. A torii, tiered tiled
roofs, a minaret and a steeple are about as far-field-legible as buildings get,
and the split is real: `tokyo_bay` carries shinto 3,464 against buddhist 3,328.

**`name:en` and `*-Latn` survive the `name:` drop.** "Language variants are not
identity" holds only when the bare `name` is already in the observer's
language. Elsewhere the bare `name` is in the local script, and these are the
only strings sharing an alphabet with what a VLM reports: on
`pohang_canal_04`, 6,360 of 7,207 named rows are Hangul-only, and dropping
these left just 387 of 12,766 rows with any Latin identity. The rule is
asymmetric because the *observer* emits English. Kept narrow on purpose —
every other variant duplicates the bare name or adds unrelated script
(pohang's bbox carries 2,520 `name:el`).

## 2026-08 · OSM extraction memory: what the bbox-bounded node index was for

The common libosmium extractor builds a node-location index for the whole PBF,
so peak memory tracks file size rather than the requested bbox. Whole-France
(4.7 GB) reached 28 GB RSS growing at 1.2 GB/min and took the machine down.

Two mitigations existed and both have since been removed: a bounded node
margin (`node_margin_deg`, which held only nodes within bbox+margin, bringing a
4.7 GB national extract to a 3.0 GB peak), and a `MAX_PBF_MB` refusal. A
subsequent osmium-based smart pre-extract was also tried and removed for adding
an unpinned external CLI dependency.

**Current state:** direct full-PBF extraction, with the cgroup ceiling
(`EXTRACT_MEM_CAP_GB`) as the only guard. The host is protected — a runaway is
killed in its own scope — but a country-sized PBF is no longer runnable, and
the working practice is to bound the source region instead (see the Boston
catalog's 625 km² scope). If a national extract is ever needed again, the
bounded node index is the mechanism to restore, and the caveats it carried
were: a way is still selected iff one of its own vertices is in a bbox, so
*which* ways match is unchanged, but a selected way's stored geometry is
truncated to retained vertices, and a way with a single interior vertex needs
its neighbours retained to clear the two-coordinate minimum. Use a margin
comfortably larger than the longest segment that matters — coastlines and
submarine cables have vertices kilometres apart.

## 2026-08 · Viewer settings live in the build config

`pipeline.viewer_completed` compared the published viewer's manifest against a
literal `{"max_particles": 900, "basemap_detail": 1.0, "body_only": False,
"embed_source_chips": True}`, while `localization/viewer.py` carried its own
argparse defaults for the same four values. Two owners of one setting: changing
either side declares every viewer already on disk stale, from a change nobody
made deliberately. The tuning values moved into the `viewer` config section
with `pipeline.viewer_config()` as their single reader.

`viewer.*` is deliberately outside every stage's `config_prefixes` — retuning
the page must not invalidate the localization run it displays.

## 2026-08 · A lossy ingest says so

`dataset.run_ingest` drops a malformed bounding box rather than raising, and
counts it. Nothing read those counters — not the four call sites, not a log
line, not a manifest — so a run that quietly ingested 90% of its predicted
geometry and one that ingested all of it were indistinguishable from the
outside, all the way to a localization result. `IngestStats.summary()` is now
printed by `run_ingest` itself on every call, prefixed `WARNING` when anything
was discarded.

Reporting lives in `run_ingest` and not in its callers on purpose: with four
call sites, a caller that forgets looks exactly like a clean ingest.

The counters should read zero for any artifact this pipeline produced — the
extraction stage validates predictions harder than ingest does and rejects a
whole response that fails. They can only be non-zero for a hand-built or
foreign `--frame_landmarks_dir`, which is exactly when a silent drop is most
misleading.

## 2026-08 · One document skeleton, two page styles

`viewers/page.py` was "the one HTML page helper", but the two largest viewers —
the localization run viewer and the match viewer — each built their own
document, because `page()` imposes the index pages' stylesheet and title
heading and those viewers are full designs with their own type scale and
header. The cost was real: the match viewer's page had no doctype, no viewport,
the title "matches", no provenance footer, and no `GENERATED_MARK` — which is
what `indexes.refresh` checks before it is willing to overwrite a page, so an
unmarked viewer page is indistinguishable from something hand-written that must
be preserved.

`page.document()` is now the skeleton alone — generated mark, doctype, head,
the caller's own stylesheet — and `page()` is written in terms of it. Shared
skeleton, separate designs.

## 2026-08-26 · Identity is data lineage; the path is not the artifact

Three separate rules were removed here, and the reason is the same in each
case: they gated on something that was not evidence about the data.

**Code out of identity.** Identity was `H(kind, dataset, stage config,
upstream manifest digests, build inputs)` plus a fingerprint of the producing
code. Measured against the real tree: three commits touching only viewer HTML
invalidated all eight producing stages, and one ordinary day's work
invalidated two of eight even with presentation files excluded. Code changes
constantly in a research tree and data does not, so gating on it means
near-permanent invalidation of exactly the artifacts that cost money — and
these are not byte-reproducible anyway, since extraction and matching are
provider calls with real variance. Gating and recording are separable; only
recording was ever needed. `code_provenance` records the commit and the
verbatim diff; nothing refuses an artifact for it.

**What the gate was worth.** `build_identity` — a digest over the *entire*
config and *all* inputs — meant changing `localization.pi0`, a knob only the
last stage reads, republished the paid `frame_landmarks` and the hours of
`object_tracks` upstream of it. `stage_reuse.py` (1,034 lines) existed solely
to undo that, and could only do so by asking a human to attest that the
prefix-computing code had not changed — one unverified claim holding up the
whole reuse story. Per-stage identity dissolves it: 4,013 lines deleted across
`stage_reuse` and `legacy_extraction_adoption`, and no attestation left to
make.

**The configured-lane path rule was a false guarantee.** `configured_lane`
used to reject an input read from anywhere but `<root>/artifacts/<kind>/
<dataset>/<version>`, justified as stopping "a stage handed leg2's tracks
against leg1's audits". It does not stop that — the copy carries its own
manifest, and the kind/dataset/version checks read it and reject the mismatch
wherever it sits. Since a differing lane is already caught by comparing
digests, the path rule rejected exactly one case: a copy identical in kind,
dataset, version and both digests. That copy *is* the artifact. What survives
is the narrow real check — the lane's current contents must still be the
artifact that was read, which fails when someone rebuilds a version in place
instead of choosing a new one.

**A record with no reader is not a record.** The only thing that read
`code_provenance` was a one-time survey, and it read the wrong field —
`manifest.config["farfield_code_provenance"]` rather than the top-level
`code_provenance` — so its lineage line reported "no artifacts in this
lineage" for every corpus, which is also what a corpus with nothing stamped
reports. The bug was therefore invisible against real data. The readout now
lives in `pipeline status`, which is where the question recurs.

**Migration tools are deleted once applied.** The backfill computed
identities for the 56 artifacts that predate identity and could be derived
exactly from a surviving build recipe -- seven kinds across eight datasets,
which is precisely the set the gate consults. The remaining 67 unattributed
artifacts are all outside the gate: catalogs and pinhole images enter as
upstream manifest digests, viewer sidecars re-stamp on their next refresh, and
the rescued v4/v5 extraction material is consumed by nothing. The planners
that computed all this are gone -- a spent one-time migration left in the tree
reads as something you might still need to run.

**One lookup path, and what it cost to get there.** Those 56 identities first
lived in a derived index beside the data, because a manifest could not gain a
field: `manifest_digest` was the sha256 of `manifest.json` and every
downstream ref records it. Re-signing the DAG looked like the way out, and it
is not. Measured on the real root: 32 of the 56 have their manifest digest
baked into a DOWNSTREAM artifact's immutable content -- the frozen
`request_set.json`, `matching_snapshot.json` and `settings.json` files, which
are covered by `content_digest` and therefore cannot be rewritten at all --
and `audit_requests` compares a stored ref to a live one
(`source_ref not in request_set.upstreams`, and `ArtifactRef` equality
includes the manifest digest). Re-signing would have broken that check on 4 of
the 7 kinds. Only 24 of 56 were reachable, which would have left both
mechanisms in place anyway.

The fix was to stop hashing the annotation: `manifest_digest` is now computed
over the manifest MINUS `artifact_identity`. Every manifest is written by
`atomic_write_json`, so its bytes already equal canonical JSON plus a newline,
and with no such key present the digest is bit-for-bit what it was --
confirmed against all 115 artifacts on the real root before anything was
written, and every upstream ref still agreed afterwards. Hashing the identity
into the digest that names the manifest was circular to begin with. The 56
were then signed in place, no digest moved, no immutable byte was touched, and
the index and its reader are gone.

**The gate was never exercised, and so it never worked.** Identity was read
but never written: no producer recorded one, and the backfill's 56 entries
were the only identities in existence. A freshly built artifact was therefore
rejected as "predating per-artifact identity". The reason no test caught it is
the shape of the check — `if build_inputs is not None` guards the identity
branch, every test called `stage_done` without `build_inputs`, and `cmd_run`
and `cmd_status` always pass it. A conditional gate plus a fixture that omits
the condition equals a gate nobody has ever run. Producers now receive
`--artifact_identity` from the orchestrator (computed at run time, since the
digest reads upstream manifests that may be built earlier in the same run) and
`ArtifactDirectoryBuilder` stamps it as a top-level manifest field. It is a
field rather than a `config` entry for a reason worth keeping: `config` is the
stage's resolved recipe and feeds the stage config digest, so an identity
stored there would have been an input to its own computation.

**`path` was never part of an artifact's identity, and the code already said
so.** `ArtifactRef` declares `path: str = field(compare=False)` — "moving an
immutable artifact does not change its identity; the two digests do" — and
then fourteen call sites compared `a.to_dict() != b.to_dict()`, which
reinstates exactly the field the type excludes. Those are the upstream-binding
checks ("the audits I was handed must be the audits this artifact was built
from"), so relocating a data root would have broken every one of them while
every byte still agreed — and this project does move data roots. They now use
dataclass equality, with `artifact.records_same_artifact` for the two that
compare against a ref stored as JSON.
