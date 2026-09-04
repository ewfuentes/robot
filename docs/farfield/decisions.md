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

## 2026-08-26 · An artifact answers for itself; the checked-in recipes did not

Two questions have to be answerable about anything on disk: how do I
reproduce it, and are the things it was made from out of date.

**The frozen YAML recipes answered neither, and drifted two generations
because nothing could tell.** `configs/active_regeneration/*.yaml` was a third
copy of a recipe that already existed in `builds/` and, partly, in every
manifest -- and the only copy with no feedback loop. Measured against the real
root before deleting them: for `boston_harbor_leg1`, **all nine** artifact
versions the recipe named differed from what the live builds use. It named
`stage3_7b88e81_adopted_v1` / `_regen_v1` / `_trim_v1`; the live builds use
`_adopted_v2`, `_regen_v2`, `stage3_17c8031_regen_v8_machine` and
`stage3_a6e45b9_trim625_v1`. The `_v1` strings appear in no `build_config.json`
and no artifact directory anywhere, and the experiment name the recipes assert
(`260824_stage3_active_regeneration`) never ran -- what ran was
`260824_stage3_active_regeneration_v2`.

Its test could not notice, by construction: it built `expected_artifacts` from
three constants copied out of the YAMLs and asserted equality, never touching
the data root. A tautology.

What made that a merge blocker rather than a wart is the third ingredient:
every recipe also carried `approve_cost: true` with `limit_usd: 50.0`. Since
`object_tracks_version` named a version that does not exist, running one would
not have reused the existing tracks -- it would have re-run tracking and then
the paid audit and matching stages, with spend pre-approved, and published
under version names nothing uses.

**Reproducing an artifact used to require a join to mutable state.** manifest
-> `build_identity` -> `builds/<dataset>/<build>/build_config.json`. That join
holds today -- 24 surviving recipes cover all 56 signed artifacts, zero
orphans -- but `builds/` is documented as "only orchestration state ... not a
scientific artifact lane" and nothing protects it. Lose one recipe and the
artifact is both irreproducible and its identity uncomputable, which is the
hole the identity backfill was written to paper over.

`artifact_recipe` closes it by recording the identity terms a manifest could
not otherwise recover: the resolved stage config, the build inputs the stage
read, and the upstream digests that entered the identity. Three ad-hoc
conventions were replaced -- `resolved_stage_config` (flat dotted keys, in
`semantic_audits` and `landmark_matches`), `resolved` (nested blocks, in
`object_tracks`) and per-domain blocks (in `localization_inputs`) --  and
`semantic_audits` had recorded no `source_digests` at all, so for that kind
you could not even verify you had found the right build recipe.

**The check that makes the record trustworthy: an identity recomputed from the
manifest alone must equal the identity the manifest records.** A manifest
missing a term its identity depends on cannot satisfy that, so it is total
rather than illustrative -- unlike a round-trip test, which would only have
covered the config half. Writing it immediately found something: identity is
computed over the stage's CONFIGURED upstreams, which is not
`manifest.upstreams`. A `frame_landmarks` manifest records its pinhole
artifact and the canonical LLM result artifact, while `extract` declares no
artifact upstreams -- and the orchestrator could not name the result artifact
anyway, since it does not exist until the stage has run. So the recipe records
which upstreams entered the identity, and the verification separately requires
that set to be a subset of the manifest's lineage, so a recipe cannot invent
an ancestor the manifest does not show.

## 2026-08-26 · The identity a migration computes must be the identity the gate computes

The first identity backfill computed the value itself, with `manifest.upstreams`
as the upstream term. The gate computes it with the stage's CONFIGURED
upstreams. Those differ for exactly one kind -- a `frame_landmarks` manifest
records its pinhole artifact and the canonical LLM result artifact, while
`extract` declares no artifact upstreams, and the orchestrator cannot name the
result artifact anyway because it does not exist until the stage has run.

Eight artifacts were therefore signed with an identity nothing would ever
compute, and because every downstream stage validates its upstreams, **every
stage of every build read INVALID**: `frame_landmarks` identity 1040d55cb665
where the gate wanted 3ad7ddf1c69c. One wrong value, 112 symptoms, on data
that was entirely fine.

The migration that fixed it computes nothing of its own -- it calls
`pipeline.expected_artifact_identity` and `pipeline.stage_recipe`, so agreement
with the gate is by construction rather than by a second implementation kept in
step by hand. That is the rule worth keeping: a migration that writes a value
the gate will later check must obtain it from the gate's own code path.

It also has to converge. Computing a child's identity validates its upstreams
through the gate, so a child is uncomputable until its parent carries the right
identity -- one pass per level of the DAG, two here. The values never change
between passes: both `artifact_identity` and `recipe` sit outside
`manifest_digest`, so correcting a parent cannot move a child's identity. Only
the set of computable artifacts grows.

**What let this ship wrong the first time:** the migration was verified against
its own invariants -- every ref agrees with its lane, no digest moved -- and
those all passed. Nobody ran `pipeline status`, the one command that consumes
the data the way the pipeline does. Verifying a migration's internal
consistency is not verifying that the result is usable.

## 2026-08-26 · Presentation settings are not under the no-defaults rule

`CONFIG_SCHEMA` supplies no defaults on purpose: "an ignored typo in a
result-shaping key is indistinguishable from silently taking a default". Adding
`viewer.max_particles`, `viewer.basemap_detail` and `viewer.embed_source_chips`
to it made every build recipe recorded before those keys existed unreadable --
all 24 on the real root, so `pipeline status` could not even parse a build.

They moved to `PRESENTATION_SCHEMA`: validated when present, rejected when
misspelled, but not required. The rule's rationale does not reach them, because
they shape a rendered page and cannot change a scientific result -- retuning
them does not invalidate the run the page displays. `viewer_config` names its
fallbacks explicitly so the orchestrator and the viewer still cannot hold two
different sets, which is what putting them in the config was for.

The fallback fires on ABSENCE only. Catching an invalid value there too would
have let a typo in a presentation key silently become the fallback, which is
the failure the no-defaults rule exists to prevent.

The alternative -- adding the block to the 24 recipes on disk -- was rejected:
`source_digests.build_config` records each recipe's sha256 in the artifacts it
produced, and although nothing compares it today, editing the files would
leave 24 recorded digests quietly wrong.

## 2026-08-26 · Identity is read from a manifest; integrity is a separate question

Obtaining an upstream's `manifest_digest` went through `open_artifact`, which
re-hashes the artifact's entire contents. The digest itself is a hash of one
small `manifest.json`, and the manifest already records `content_digest` -- so
a `pipeline status` re-read gigabytes to compute values it could have read.
Measured on the real root: `boston_harbor_leg1`'s artifacts total 4.5 GB,
`object_tracks` alone 400 MB, re-hashed once per stage that names it.
`digest_cache` did not help: it caches file digests for dataset source inputs
and `artifact.py` never imported it, so `sha256_directory` was entirely
uncached. The earlier claim that #685 "stopped rehashing frozen inputs on every
stage" was true only of dataset sources, not of the far larger artifact path.

Two questions were being answered by one call. INTEGRITY -- do the bytes still
match? -- must read every byte. IDENTITY -- what are this artifact's kind,
dataset, version and digests? -- needs only the manifest.
`artifact.reference_from_manifest` answers the second and the pipeline's three
hot call sites use it. Full `status` on a real build went from minutes to
**7.0 s including bazel startup**, same output.

It still validates everything that is cheap: the manifest parses and is
complete, kind/dataset/version match, and every declared output exists. Only
the byte comparison is skipped, and `open_artifact` is unchanged, so every
other caller keeps full verification.

**What this gives up, deliberately:** a file that changes under a published
artifact is no longer noticed by `run` or `status`. That used to be caught
incidentally on every check, paid for in gigabytes. It is now
`pipeline verify --build_dir`, which re-hashes each artifact against its
manifest and reports per kind. A routine status catches strictly less than
before; the trade is that it can actually be run.

## 2026-08-27 · Resampling survival floor: a hypothesis may die of evidence, not of rounding

**What changed.** `systematic_resample` accepts a survival floor
(`localization.resample_survival_floor`, with
`localization.resample_survival_min_mass` as the genuinely-refuted cutoff):
every mode and proposal-cluster group holding at least the minimum mass keeps
at least `floor` offspring, and offspring then carry their group's true mass
(`log(mass/count)`) instead of the uniform post-resample reset — so the
posterior is unchanged; only representation is guaranteed. At 0 the historical
uniform-weight behavior reproduces bit-for-bit.

**Why — the seed study that motivated it.** mount_washington leg3 whole-map
runs are bimodal: 4 seeds at identical code/config/inputs gave 500 m
time-normalized mass 0.233/0.011/0.258/0.018 (23x spread). The filter is
bit-deterministic on GPU (a re-run reproduces the archived run byte-for-byte
at both e6c4825 and 7e898528; an apparent cross-commit "17x improvement" was
the #683 candidate-chunking change remapping the RNG stream — i.e. a hidden
reseed). Per-checkpoint particle provenance showed the coin flip is lineage
survival: every seed carries hundreds of init-injection descendants near truth
through kf ~190; misleading measurements in kf ~200–235 cull them while every
proposal event in kf 150–260 fails the evidence gate; winning seeds keep a
remnant of 61–278 particles of 50k that explodes to 26–36k when decisive
evidence arrives at kf ~240–260, and the winning mass is 100% event-0
descendants. Losing seeds go extinct. 10x particles did NOT raise the
convergence rate (1/4 at 500k vs 2/4 at 50k) — remnant SIZE was never the
binding constraint.

**Measured, 8 datasets x 4 seeds (runs/260826_washington_iteration; leg3 arms
under the same experiment):** the floor at 64 never harmed a converging
dataset and fixed tails. boston_harbor_leg3: 3/4 -> 4/4 converged, all seeds
175–187 m (baseline seed2 failed at 493 m). charles: 8–9 m final vs 10–11 m
(best recorded). mount_washington leg3 converged runs: 435/476 m vs 515/2223 m.
Unconverged tails land km closer (pohang 1.5–2.2 km vs 5–13 km on 3 of 4
seeds). The one measured cost: boston_harbor_leg1, which never truly
localizes, dilutes partial mass 0.27 -> 0.21 because more wrong hypotheses
stay represented when nothing can win. Dead datasets (boston leg2, mtw leg1,
pohang) are matcher-limited and unchanged — a resampling fix cannot repair
wrong evidence.

**Tried and rejected in the same study:** (1) gate-side revival of dead
hypotheses — bit-for-bit a no-op at exact landmark-set matching AND at 250 m
pose matching, because the proposal generator never re-proposes the truth
region after the early window (1 of ~1,100 later hypotheses shared any
landmark with the dead near-truth mode); a workable version must inject from
the remembered dead pose. (2) a whole-belief per-measurement damage cap —
3 nats freezes global discrimination entirely (0/4, means 26–193 km out);
6 nats alone matches the floor (2/4, 479/554 m) but stacked with the floor is
anti-synergistic (0/4): both protect hypothesis survival and together the
belief stays permanently fragmented. If damage bounding returns, it must be
asymmetric (clamp tracked-mode decay, not the diffuse cloud).

## 2026-09-02 · Reuse is checked from the artifact's manifest, not recomputed from the consumer's recipe

**What changed.** A build that names an existing artifact version no longer
has that artifact's stage config digest and identity recomputed from the
consuming build's config and compared for equality. The artifact is checked
from its own manifest: self-describing (its identity recomputes from its
recorded recipe), config-compatible (shared settings must match; settings the
artifact predates are reported as unverified; `execution`/`cost` never gate),
and lineage-consistent (it binds the sibling versions the build names).
Producers still refuse to publish over a version made with different settings.

**Why — two measurements on the real root, 2026-09-02.**

1. #702 added `tracking.fragment_min_dominant_cc` and `fragment_patience` as
   required keys. At HEAD, `pipeline run --only localize` on the untouched
   260828_imu_baseline Pohang lineage failed with "required object_tracks
   artifact has a different resolved configuration". The track digest is a
   hash over the key set, so no HEAD config could equal what
   `stage3_7b88e81_regen_v2` recorded (`5f6c116c…` vs `04cb0650…`), and the
   check recursed from localize down to tracks. Every pre-#702 chain on all
   eight datasets was orphaned for every HEAD build, filter-only reruns
   included. Nothing about the tracks was stale.
2. `execution.*` and `cost.*` sit in the extract, audit and match prefixes as
   one shared block. Switching matching to on-demand transport moved the
   digest expected of the reused extraction (`0bafa822…` → `179c00ab…`); a
   new GCS staging prefix alone did too (`af1a13b3…`). Transport was pinned by
   the oldest reused LLM artifact.

Both are the consumer re-deriving settings for stages it does not run. The
"stale artifact used in silence" hazard that recomputation guarded against is
a producer-side hazard, and the producer-side guard ("choose a new output
version") already covers it.

The producers carried the same check in miniature: match, bearings,
diagnostics and tracking each refused an input whose `config.build_identity`
differed from their own build's ("belongs to a different immutable build"),
so a new build could not run match over an audit another build had published
even after the orchestrator had accepted it. Those four equalities are gone;
the lineage checks beside them (an audit binds its tracks exactly once, and
so on) stay, because those are about the data.

**Rule (ekf).** Builds may name prior parts. When the config for a plugged-in
stage is given, a differing shared value fails; a key the artifact predates
warns. Schema growth must not orphan artifacts.

## 2026-09-03 · A second source enters the catalog as a derived artifact, under the OSM vocabulary

Overture Places (Meta, Microsoft, Foursquare, AllThePlaces; no OSM content)
names things OSM lacks. On the Pohang review set it holds the marina T199, the
seafood hall behind T164, the pier T6/T14/T17/T18 point at, and the exact
name of the canal pavilion T0/T13 missed 14 times. Two ways to bring it in:

1. Rebuild the stage-5 full catalog with Overture as a third input beside OSM
   and ENC. That reopens the reviewed coverage plan for a source the plan
   cannot attest (Overture has no clip polygons), and every existing lineage
   would point at a catalog that no longer exists.
2. Publish a **derived** catalog: the full catalog plus the source's rows,
   with the full catalog as its single CATALOGS upstream. Lineage still
   terminates at the OSM coverage attestation; `trim_catalog` runs on top
   unchanged.

We do 2 (`dataset_tools:add_catalog_source`). Three rules travel with it:

- **Overture rows speak OSM.** `extract_landmarks_from_overture` maps each
  place's taxonomy hierarchy onto OSM-style tags (root default, leaf
  override). The trim then judges an Overture café by the same
  name-rescue/unobservable rules as an OSM `amenity=cafe` node. There is no
  Overture-specific keep list; a "far_field=retain" flag computed outside the
  repo is exactly the kind of untracked selection the artifact contract
  exists to prevent.
- **Same name within radius = same thing.** A source row is dropped when any
  name-like tag equals a catalog row's (NFKC, casefold, letters and digits
  only) within `--dedupe_name_radius_m` (150 m here, the collision-report
  radius). Different-name near neighbours are kept: the matcher, not the
  merge, decides which of two records of one building is right. Every
  dropped pair is in the manifest config.
- **Licences stay on the row.** Places is licensed per source record, so the
  `overture:sources` tag carries each record's dataset, licence and id. It is
  outside the keep vocabulary and never reaches the matcher.

`landmark_type` gains `overture`; the matcher's landmark ids are namespaced by
that column (`overture:<gers-id>`) instead of folding every non-ENC source
into `osm:`.

Fetch box: the catalog manifest's `bbox_wsen` (trajectory + 25 km), not the
trajectory extent, so range landmarks (POSCO, Yeongildae, the breakwater
lights) are covered the same way the OSM extract covers them.

## 2026-09-03 · The extractor's distance bucket is a one-sided range cap, carried per keyframe

The extraction prompt asks for a `distance_estimate` bucket per detection
(<100 m, 100-500 m, 500 m-2 km, 2-10 km, >10 km). Until now it reached only
the audit dossier text. Measured on Pohang against 44 labelled tracks whose
bearing rays hit the labelled geometry (1,628 detections):

- rank-correlated with true range (Spearman 0.87 per detection, 0.91 per
  track), but **biased one bucket far**: exact agreement 13 %, within one
  bucket 88 %, only 13.5 % of detections beyond their bucket's lower edge;
- **99.8 % of detections nearer than the predicted bucket's upper edge**;
- `under_100m` right 124/124 times (median 22 m), and it is what the bridges
  we pass under carry.

So the bucket enters the filter as a cap, never as a two-sided estimate or a
lower bound. Three touch points, no new stage:

1. `bearings`: each `CameraBearingObservation` gains `range_max_m`, the
   tightest finite upper edge among the detections associated at that
   keyframe (birth plus evidence-class supports, the audit's own set). The
   stage now re-ingests the frame_landmarks the tracks bind, under the
   recorded `ingest.*` settings, which therefore join its config prefixes;
   `over_10km` is no cap. Readers of `observations.jsonl` treat a missing key
   as no cap, so pre-cap artifacts still load.
2. `localization_inputs`: the epoch reducer keeps the minimum cap in the
   epoch (`under_100m` was never wrong here); `TrackletMeasurement.range_max_m`
   rides into the export.
3. `localize`: every landmark component is multiplied by
   g(r) = 1 for r <= cap, exp(-½((r-cap)/(s·cap))²) beyond, in both the numpy
   and torch kernels and in the committed-landmark density. The null
   component is untouched, so clutter stays a competitive explanation.
   `localization.range_cap.enabled` / `.softness_frac` are required keys;
   disabled strips caps before any kernel, so the likelihood is bit-identical
   to before whatever the export recorded.

Why not per-keyframe measurements first: measured on the same tracks, the
5-keyframe epoch reducer gives near landmarks (<100 m) 15 % of the total
bearing precision and a 22-36° sigma under bridges, because the mask-width/4
term and the circular mean erase the parallax. That is a separate change
(ekf: stay with epochs for now); the cap is the cheap test of whether range
information moves Pohang at all.

**Measured 2026-09-04 (Pohang, 4 seeds, distance-normalised mass @500 m).**
OSM-trim baseline 0.200 → 0.594 with the cap; Overture catalog 0.031 →
0.612. Mass @100 m 0.024 → 0.18; final MAP 2–4 km → 0.5–0.8 km. Same
matches, same tracks, no LLM call: one likelihood term. The Overture arm's
collapse had been a handful of confident instance matches to landmarks 2–23
km away; under a 100–500 m cap those components vanish, so the cap does not fix
the association but makes wrong far associations harmless while near caps add
the range information a bearing lacks. Cross-dataset check:
`runs/260904_range_cap`. Two reader shims came with it: `run_io` fills
`range_cap_*` / `range_max_m` with the legacy value for runs and exports
recorded before the field existed (`_PREDATING_*`), and alignment diagnostics
ignore the bearings manifest's `range_cap` block, since they verify every other
key as an exact set.

**Cross-dataset check (2026-09-04, `runs/260904_range_cap`, successors of the
260828 IMU baselines, 4 seeds, dn500 baseline → cap).** Boston leg1
0.342 → 0.779, leg2 0.001 → 0.003, leg3 0.263 → 0.277; Mount Washington leg1
0.000 → 0.004, leg2 0.940 → 0.947, leg3 0.431 → 0.623 with the seed
bimodality gone (0.21–0.52 → 0.622–0.624) and a 39 m final MAP. Never worse;
helps wherever the filter had near landmarks to bind, neutral where every
landmark is beyond the 2–10 km buckets (Boston leg3), nothing where the data
was already dead upstream (Boston leg2, mtw leg1 with 136 observations).
Second batch the same day: Charles River (IMU-baseline successors) stays at
its saturated 0.997 with dn100 0.649 → 0.655 and a 21 m final MAP; Boston
snowy (successors of the 260903 `osmv2_v1` build, a 25 × 25 km OSM box)
0.387 → 0.854 at 500 m, 0.171 → 0.604 at 100 m, final MAP 580–690 m → 26 m
on every seed. Each dataset kept the catalog its baseline used, so the
comparison is only the cap.
