"""Window-joint proposal: pose hypotheses from many concurrent tracklets over
a window of keyframes, one landmark identity per tracklet (§5.5 extension).

Why (runs/260906_proposal_audit). In a field of interchangeable landmarks
(Flevoland: 336 turbines ~400 m apart) a *snapshot* resection stays
lattice-aliased however many tracklets it uses: eight concurrent turbine
tracklets, exhaustively resected and cross-checked, still leave ~1,000
distinct sites, and the truth holds 0.01 % of them. What breaks the lattice
is what the filter's association persistence also relies on: ONE identity
per tracklet held across a window of keyframes whose poses are related by
odometry. Over 20 keyframes the same eight tracklets leave four sites with
the truth ranked first (with odometry of realistic quality).

Per event:
  1. tracklets with epochs anchored in [kf - W, kf], most epochs then tightest
     range cap first, at most `window_joint_max_tracklets`; the first
     `window_joint_resection_tracklets` generate hypotheses, all of them score.
  2. exhaustive identity triples of the generating tracklets, pruned by the
     PER-TRACKLET range caps (pairwise baseline <= cap_a + cap_b), closed-form
     three-point resection of one epoch per tracklet chosen nearest a common
     reference keyframe and treated as simultaneous there (coarse tolerance;
     the fix is moved to the current keyframe by odometry and step 3 removes
     the approximation).
  3. Gauss-Newton refinement of every fix on its three tracklets' window
     epochs with the proper per-epoch ray origins from back-integrated
     odometry.
  4. window scoring against every window tracklet: per tracklet the best
     identity by window rms (bearing residuals plus the soft range-cap excess
     of every epoch, as one chi-square); consistent below
     `window_joint_rms_tolerance_deg` widened by the declared heading drift. Up to `window_joint_max_outlier_tracklets`
     inconsistent tracklets are tolerated (a third of Flevoland's tracklets
     align no endorsed row at the true pose).
  5. rank by the tempered window log-likelihood plus the decayed score of the
     same site at the previous event (moved here by odometry): a lattice
     alias fits one window, the truth fits them in a row. Dedupe on the
     solution-cluster grid, emit PointHypothesis(kind=TRIPLE) with
     compatibility_mass = exp(score) so the existing allocation and injection
     apply unchanged.

The generator is a proposal, not a likelihood: every injected particle is
re-scored under the exact measurement model, so what matters here is recall
(the truth among the sites) and density (enough particles per site), not the
calibration of the score.
"""
import dataclasses
import itertools
import math

import msgspec
import numpy as np

from experimental.overhead_matching.swag.farfield import geometry as geo
from experimental.overhead_matching.swag.farfield.localization import (
    proposal,
    resection,
    structs,
)

# Slack on the extractor's one-sided range cap: the cap is a 99.8 % bound
# (range_cap_log_term), and the proposal only needs recall.
CAP_SLACK = 1.25
# Coarse verification tolerance for the simultaneous-snapshot fix, before
# the window refinement (staggered epochs a few keyframes apart at 20 m
# spacing misplace a 500 m landmark's bearing by several degrees).
SNAPSHOT_TOLERANCE_RAD = math.radians(5.0)
REFINE_ITERATIONS = 3
REFINE_MAX_STEP_M = 300.0
REFINE_MAX_STEP_RAD = math.radians(15.0)
SCORE_CHUNK = 4000


@dataclasses.dataclass
class WindowTrack:
    tracklet_id: str
    # (anchor_keyframe_idx, bearing_rad, kappa, cap_m) for every epoch in the window
    epochs: list
    cand_idx: np.ndarray  # catalog indices of endorsed candidates
    cand_w: np.ndarray
    cap_m: float  # the latest epoch's cap (catalog maximum when none)
    landmark_ids: tuple
    # log identity prior over the endorsed candidates (softmax of the clipped
    # matcher log-odds, as the filter's _identity_log_weights), max 0.
    cand_logp: np.ndarray = None


@dataclasses.dataclass
class WindowMemory:
    """Hypotheses kept at the last event with their accumulated scores, so a
    site that stays consistent from one window to the next earns credit. A
    lattice alias fits one window; only the truth fits the windows in a row."""
    keyframe_idx: int
    poses: np.ndarray  # (m, 3) at keyframe_idx
    scores: np.ndarray  # accumulated tempered log-scores


MEMORY_MATCH_M = 100.0
MEMORY_MATCH_RAD = math.radians(5.0)
MEMORY_DECAY = 0.9
# In tempered log-score units (temperature 4): one inconsistent tracklet.
MEMORY_NEW_SITE_PENALTY = 1.0


@dataclasses.dataclass
class WindowScore:
    n_consistent: np.ndarray  # (n_hyp,)
    mean_rms_rad: np.ndarray  # over consistent tracklets
    identity: np.ndarray  # (n_hyp, n_tracks) catalog index, -1 = inconsistent
    score: np.ndarray
    best_identity: np.ndarray = None  # (n_hyp, n_tracks) argmin identity, consistent or not


def relative_poses(odometry, kf: int, window_keyframes: int) -> dict:
    """Body-frame pose (at kf) of the platform at each earlier keyframe.

    Returns {keyframe: (u, v, dtheta, yaw_var)} with v along the kf heading, u
    to the right, dtheta the heading change relative to kf (CW positive) and
    yaw_var the declared variance of that relative heading. Uses the
    filter's rotate-then-move semantics in reverse: odometry[j - 1] is the
    increment INTO keyframe j.
    """
    u = v = theta = 0.0
    yaw_var = 0.0
    out = {kf: (0.0, 0.0, 0.0, 0.0)}
    for j in range(kf, max(kf - window_keyframes, 0), -1):
        if j - 1 >= len(odometry):
            break
        delta = odometry[j - 1]
        # pos_j = pos_{j-1} + forward * f(theta_j) + left * l(theta_j) with
        # f = (sin, cos), l = (-cos, sin) in the (u, v) frame where heading 0
        # points along +v.
        u -= delta.forward_m * math.sin(theta) - delta.left_m * math.cos(theta)
        v -= delta.forward_m * math.cos(theta) + delta.left_m * math.sin(theta)
        theta -= delta.delta_yaw_cw_rad
        # Declared heading uncertainty of the relative pose k -> kf: the
        # per-step sigmas compose as independent increments (§5.2).
        yaw_var += delta.sigma_yaw_rad ** 2
        out[j - 1] = (u, v, theta, yaw_var)
    return out


def collect_tracks(measurements, tables, catalog, config: structs.ProposalConfig,
                   kf: int) -> list:
    """Window tracklets, most epochs then tightest cap first, at most max_tracklets."""
    window_start = kf - config.window_joint_keyframes
    by_tid = {}
    for meas in measurements:
        if not window_start <= meas.anchor_keyframe_idx <= kf:
            continue
        if meas.tracklet_id not in tables:
            continue
        by_tid.setdefault(meas.tracklet_id, []).append(meas)
    tracks = []
    for tid, epochs in by_tid.items():
        candidates = proposal._candidate_landmarks(tables[tid], catalog)  # noqa: SLF001
        if not candidates:
            continue
        # A cap is a bound on the range AT ITS OWN EPOCH; ranges change as
        # the platform moves, so the tightest cap in the window must not be
        # applied to the latest epoch. The latest epoch's cap prunes tuples;
        # every epoch's cap is checked at its own platform position in scoring.
        catalog_max = float(np.max(catalog.max_visible_range_m))
        latest = max(epochs, key=lambda m: m.anchor_keyframe_idx)
        cap = latest.range_max_m if latest.range_max_m is not None else catalog_max
        ids = tuple(c.landmark_id for c in candidates)
        log_w = np.array([math.log(max(c.weight, 1e-300)) for c in candidates])
        tracks.append(WindowTrack(
            tracklet_id=tid,
            epochs=sorted((m.anchor_keyframe_idx,
                           math.radians(m.bearing_forward_cw_deg), m.kappa,
                           m.range_max_m if m.range_max_m is not None else catalog_max)
                          for m in epochs),
            cand_idx=np.array([catalog.index_of(i) for i in ids], dtype=int),
            cand_w=np.array([c.weight for c in candidates]),
            cand_logp=log_w - log_w.max(),
            cap_m=float(cap), landmark_ids=ids))
    # Most epochs first (more constraint, and a long tracklet is more often a
    # real object), tightest cap second. The tightest cap alone is a bad
    # generator key: a short near tracklet with a wrong cap heads the list and
    # then sits in every generating triple.
    tracks.sort(key=lambda t: (-len(t.epochs), t.cap_m, t.tracklet_id))
    # Two tracklets whose bearings agree within DUPLICATE_BEARING_RAD at
    # nearby epochs are almost surely one object tracked twice; a triple that
    # contains both is degenerate (same landmark, no baseline), and in a
    # sparse window they would crowd out the independent tracklets.
    kept = []
    for track in tracks:
        if not any(_looks_duplicate(track, other) for other in kept):
            kept.append(track)
    return kept[:config.window_joint_max_tracklets]


DUPLICATE_BEARING_RAD = math.radians(2.0)
DUPLICATE_KEYFRAMES = 2


def _looks_duplicate(track, other) -> bool:
    for k, b, _, _ in track.epochs:
        for k2, b2, _, _ in other.epochs:
            if (abs(k - k2) <= DUPLICATE_KEYFRAMES
                    and abs(geo.wrap_rad(b - b2)) < DUPLICATE_BEARING_RAD):
                return True
    return False


def _circles(a, b, gamma):
    """Inscribed-angle circles through chord (a, b): both centres, radius."""
    d = np.hypot(*(b - a).T)
    mid = (a + b) / 2
    tangent = (b - a) / d[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
    offset = d / (2 * math.tan(gamma))
    radius = d / (2 * math.sin(gamma))
    return (np.stack([mid + offset[:, None] * normal,
                      mid - offset[:, None] * normal], 1), radius)


def _intersect(c1, r1, c2, r2):
    dist = np.hypot(*(c2 - c1).T)
    ok = (dist > 1e-6) & (dist <= r1 + r2 + 1e-6) & (dist >= np.abs(r1 - r2) - 1e-6)
    safe = np.where(ok, dist, 1.0)
    along = (r1 ** 2 - r2 ** 2 + safe ** 2) / (2 * safe)
    height = np.sqrt(np.maximum(r1 ** 2 - along ** 2, 0.0))
    ex = (c2 - c1) / safe[:, None]
    ey = np.stack([-ex[:, 1], ex[:, 0]], 1)
    base = c1 + along[:, None] * ex
    points = np.stack([base + height[:, None] * ey,
                       base - height[:, None] * ey], 1)
    points[~ok] = np.nan
    return points


def _enumerate_tuples(tracks3, east, north, max_tuples, rng):
    """Identity triples surviving the pairwise per-tracklet cap prune."""
    (t1, t2, t3) = tracks3
    i1, i2, i3 = t1.cand_idx, t2.cand_idx, t3.cand_idx
    d12 = np.hypot(east[i1][:, None] - east[i2][None],
                   north[i1][:, None] - north[i2][None])
    ok12 = (d12 >= resection.MIN_BASELINE_M) & (d12 <= (t1.cap_m + t2.cap_m) * CAP_SLACK)
    d13 = np.hypot(east[i1][:, None] - east[i3][None],
                   north[i1][:, None] - north[i3][None])
    ok13 = (d13 >= resection.MIN_BASELINE_M) & (d13 <= (t1.cap_m + t3.cap_m) * CAP_SLACK)
    total = len(i1) * len(i2) * len(i3)
    parts = []
    for k in range(len(i1)):
        j2 = np.nonzero(ok12[k])[0]
        j3 = np.nonzero(ok13[k])[0]
        if not len(j2) or not len(j3):
            continue
        g2, g3 = np.meshgrid(j2, j3, indexing="ij")
        g2 = g2.ravel()
        g3 = g3.ravel()
        d23 = np.hypot(east[i2[g2]] - east[i3[g3]], north[i2[g2]] - north[i3[g3]])
        keep = ((d23 >= resection.MIN_BASELINE_M)
                & (d23 <= (t2.cap_m + t3.cap_m) * CAP_SLACK)
                & (i2[g2] != i3[g3]) & (i1[k] != i2[g2]) & (i1[k] != i3[g3]))
        if keep.any():
            parts.append(np.stack([np.full(int(keep.sum()), k), g2[keep], g3[keep]], 1))
    if not parts:
        return np.zeros((0, 3), dtype=int), total, 0
    tuples = np.concatenate(parts)
    pruned_to = len(tuples)
    if len(tuples) > max_tuples:
        # Systematic thinning keeps coverage of every candidate of the first
        # tracklet rather than dropping whole blocks.
        stride = len(tuples) / max_tuples
        pick = (np.arange(max_tuples) * stride + rng.uniform(0, stride)).astype(int)
        tuples = tuples[np.minimum(pick, len(tuples) - 1)]
    return tuples, total, pruned_to


def reference_epochs(tracks3):
    """Pick a reference keyframe and one epoch per tracklet for the snapshot
    fix: the keyframe minimising the largest anchor offset among the three
    tracklets' nearest epochs (staggered epochs are only approximately
    simultaneous; the offset is what the refinement then removes).
    Returns (k_ref, [epoch, epoch, epoch])."""
    best = None
    for k_ref in sorted({e[0] for t in tracks3 for e in t.epochs}, reverse=True):
        chosen = [min(t.epochs, key=lambda e: (abs(e[0] - k_ref), -e[0])) for t in tracks3]
        cost = max(abs(e[0] - k_ref) for e in chosen)
        if best is None or cost < best[0]:
            best = (cost, k_ref, chosen)
    return best[1], best[2]


def resect_snapshot(tracks3, east, north, bbox, max_tuples, rng, chunk=150000,
                    epochs=None, offsets_m=(0.0, 0.0, 0.0)):
    """Closed-form fixes of one epoch per tracklet (the latest by default),
    treated as simultaneous. `offsets_m` is how far the platform had moved
    between each epoch and the reference keyframe; the verification
    tolerance widens by the bearing error that motion can induce at the
    tracklet's range cap (a near tracklet three keyframes off is several
    degrees out, and the refinement, not this stage, removes it).

    Returns (poses (m, 3): east, north, heading; identities (m, 3): catalog
    indices; n_tuples_total; n_tuples_after_cap_prune).
    """
    if epochs is None:
        epochs = [t.epochs[-1] for t in tracks3]
    bearings = [e[1] for e in epochs]
    tolerances = [SNAPSHOT_TOLERANCE_RAD + math.atan2(off, max(t.cap_m, 1.0))
                  for t, off in zip(tracks3, offsets_m)]
    order = None
    for perm in itertools.permutations(range(3)):
        g12 = abs(geo.wrap_rad(bearings[perm[1]] - bearings[perm[0]]))
        g13 = abs(geo.wrap_rad(bearings[perm[2]] - bearings[perm[0]]))
        if (resection.MIN_SUBTENDED_RAD < g12 < math.pi - resection.MIN_SUBTENDED_RAD
                and resection.MIN_SUBTENDED_RAD < g13 < math.pi - resection.MIN_SUBTENDED_RAD):
            order = perm
            break
    if order is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), 0, 0
    tracks3 = [tracks3[p] for p in order]
    b1, b2, b3 = [bearings[p] for p in order]
    t1, t2, t3 = [tolerances[p] for p in order]
    # The heading comes from tracklet 1, so its error adds to both residuals.
    tol2, tol3 = t1 + t2, t1 + t3
    g12 = abs(geo.wrap_rad(b2 - b1))
    g13 = abs(geo.wrap_rad(b3 - b1))
    tuples, total, pruned_to = _enumerate_tuples(tracks3, east, north, max_tuples, rng)
    if not len(tuples):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), total, pruned_to
    i1, i2, i3 = [t.cand_idx for t in tracks3]
    caps = [t.cap_m * CAP_SLACK for t in tracks3]
    poses, idents = [], []
    for start in range(0, len(tuples), chunk):
        tt = tuples[start:start + chunk]
        l1 = np.stack([east[i1[tt[:, 0]]], north[i1[tt[:, 0]]]], 1)
        l2 = np.stack([east[i2[tt[:, 1]]], north[i2[tt[:, 1]]]], 1)
        l3 = np.stack([east[i3[tt[:, 2]]], north[i3[tt[:, 2]]]], 1)
        c12, r12 = _circles(l1, l2, g12)
        c13, r13 = _circles(l1, l3, g13)
        points = np.concatenate(
            [_intersect(c12[:, a], r12, c13[:, b], r13)
             for a in range(2) for b in range(2)], 1)  # (m, 8, 2)

        def bearing_to(landmark):
            return np.arctan2(landmark[:, None, 0] - points[..., 0],
                              landmark[:, None, 1] - points[..., 1])

        def range_to(landmark):
            return np.hypot(landmark[:, None, 0] - points[..., 0],
                            landmark[:, None, 1] - points[..., 1])

        heading = geo.wrap_rad(bearing_to(l1) - b1)
        e2 = geo.wrap_rad(bearing_to(l2) - heading - b2)
        e3 = geo.wrap_rad(bearing_to(l3) - heading - b3)
        good = ((np.abs(e2) < tol2)
                & (np.abs(e3) < tol3)
                & (range_to(l1) > resection.MIN_BASELINE_M)
                & (range_to(l1) <= caps[0]) & (range_to(l2) <= caps[1])
                & (range_to(l3) <= caps[2])
                & np.isfinite(points[..., 0])
                & (points[..., 0] > bbox[0]) & (points[..., 0] < bbox[1])
                & (points[..., 1] > bbox[2]) & (points[..., 1] < bbox[3]))
        mi, mj = np.nonzero(good)
        if len(mi):
            poses.append(np.stack([points[mi, mj, 0], points[mi, mj, 1],
                                   heading[mi, mj]], 1))
            idents.append(np.stack([i1[tt[mi, 0]], i2[tt[mi, 1]], i3[tt[mi, 2]]], 1))
    if not poses:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int), total, pruned_to
    poses = np.concatenate(poses)
    idents = np.concatenate(idents)
    # The same triple yields the same point through several circle pairs.
    key = np.concatenate([np.round(poses[:, :2] / 5.0).astype(int), idents], 1)
    _, unique = np.unique(key, axis=0, return_index=True)
    unique = np.sort(unique)
    # Restore the caller's tracklet order for the identity columns.
    inverse = np.argsort(order)
    return poses[unique], idents[unique][:, inverse], total, pruned_to


def to_current_frame(poses, rel_ref):
    """Fixes computed for the platform at the reference keyframe, expressed
    as the pose at the current keyframe: p_ref = p_kf + R(h_kf)(u, v) and
    h_ref = h_kf + dtheta, inverted."""
    u, v, dtheta = rel_ref[0], rel_ref[1], rel_ref[2]
    out = poses.copy()
    out[:, 2] = geo.wrap_rad(poses[:, 2] - dtheta)
    out[:, 0] = poses[:, 0] - (u * np.cos(out[:, 2]) + v * np.sin(out[:, 2]))
    out[:, 1] = poses[:, 1] - (-u * np.sin(out[:, 2]) + v * np.cos(out[:, 2]))
    return out


def _epoch_arrays(track, rel):
    """(u, v, dtheta, bearing, kappa, cap, yaw_var) arrays over the tracklet's
    window epochs."""
    rows = [(rel[k][0], rel[k][1], rel[k][2], b, kappa, cap, rel[k][3])
            for k, b, kappa, cap in track.epochs if k in rel]
    if not rows:
        return None
    return tuple(np.array(col) for col in zip(*rows))


def _platform_at(poses, u, v):
    """World position of the platform at body offset (u, v) for each pose."""
    sin_h = np.sin(poses[:, 2:3])
    cos_h = np.cos(poses[:, 2:3])
    px = poses[:, 0:1] + u[None] * cos_h + v[None] * sin_h
    py = poses[:, 1:2] - u[None] * sin_h + v[None] * cos_h
    return px, py


def refine(poses, idents, tracks, rel, east, north):
    """Gauss-Newton on the tracklets' window epochs, identities fixed per
    (hypothesis, tracklet); an identity of -1 leaves that tracklet out."""
    arrays = [_epoch_arrays(t, rel) for t in tracks]
    poses = poses.copy()
    for _ in range(REFINE_ITERATIONS):
        jtj = np.zeros((len(poses), 3, 3))
        jtr = np.zeros((len(poses), 3))
        sin_h = np.sin(poses[:, 2:3])
        cos_h = np.cos(poses[:, 2:3])
        for col, arr in enumerate(arrays):
            if arr is None:
                continue
            u, v, dtheta, bearing, kappa, _, _ = arr
            used = idents[:, col] >= 0
            if not used.any():
                continue
            safe = np.where(used, idents[:, col], 0)
            lx = east[safe][:, None]
            ly = north[safe][:, None]
            px, py = _platform_at(poses, u, v)
            dx = lx - px
            dy = ly - py
            rho2 = np.maximum(dx ** 2 + dy ** 2, 1.0)
            res = geo.wrap_rad(np.arctan2(dx, dy) - poses[:, 2:3] - dtheta[None] - bearing[None])
            dpx_dh = -u[None] * sin_h + v[None] * cos_h
            dpy_dh = -u[None] * cos_h - v[None] * sin_h
            jx = -dy / rho2
            jy = dx / rho2
            jh = (dy / rho2) * (-dpx_dh) + (-dx / rho2) * (-dpy_dh) - 1.0
            jac = np.stack([jx, jy, jh], 2)  # (m, e, 3)
            w = kappa[None, :, None] * used[:, None, None]
            jtj += np.einsum("mei,mej->mij", jac * w, jac)
            jtr += np.einsum("mei,me->mi", jac * w, res)
        jtj += np.eye(3)[None] * 1e-9
        step = -np.linalg.solve(jtj, jtr[..., None])[..., 0]
        norm = np.hypot(step[:, 0], step[:, 1])
        scale = np.minimum(1.0, REFINE_MAX_STEP_M / np.maximum(norm, 1e-9))
        step[:, :2] *= scale[:, None]
        step[:, 2] = np.clip(step[:, 2], -REFINE_MAX_STEP_RAD, REFINE_MAX_STEP_RAD)
        poses += step
        poses[:, 2] = geo.wrap_rad(poses[:, 2])
    return poses


def track_tolerance(arr, tol_rad):
    """Tolerance for one tracklet's window rms: the configured bearing
    tolerance plus the declared heading drift of the relative poses its epochs
    were observed from (the odometry noise the filter itself assumes)."""
    yaw_var = arr[6]
    return math.sqrt(tol_rad ** 2 + float(np.mean(yaw_var)))


# The identity prior chooses the candidate at full strength but enters the
# hypothesis score at this weight: matcher confidence tiers among genuinely
# interchangeable landmarks are noisy (a 0.9 vs 0.98 turbine is a 1.8-nat gap
# that would cost the true site an outlier-equivalent over a few tracklets),
# while in a named harbour the choice itself is what matters.
PRIOR_SCORE_WEIGHT = 0.25

# One-sided Gaussian tail beyond the cap, as the filter's range_cap_log_term
# (softness 0.25): a 500 m cap on a 700 m object costs 1.3 nats, not a veto.
CAP_SOFTNESS = 0.25


def _cap_chi2(rng_m, cap_m):
    """Squared standardised excess beyond the cap: -2 log g(range)."""
    return np.square(np.maximum(rng_m - cap_m, 0.0) / (CAP_SOFTNESS * cap_m))


def own_track_rms(poses, idents, tracks3, rel, east, north, tol_rad):
    """Window rms of each generating tracklet under its fixed identity,
    divided by that tracklet's tolerance: (m, 3), < 1 means consistent."""
    out = np.full((len(poses), 3), np.inf)
    for col, track in enumerate(tracks3):
        arr = _epoch_arrays(track, rel)
        if arr is None:
            continue
        u, v, dtheta, bearing, kappa, cap, _ = arr
        tol = track_tolerance(arr, tol_rad)
        lx = east[idents[:, col]][:, None]
        ly = north[idents[:, col]][:, None]
        px, py = _platform_at(poses, u, v)
        res = geo.wrap_rad(np.arctan2(lx - px, ly - py)
                           - poses[:, 2:3] - dtheta[None] - bearing[None])
        sigma = 1.0 / np.sqrt(np.maximum(kappa, 1e-9))[None]
        # Bearing residuals plus the soft range-cap excess, as one chi-square
        # per epoch, expressed as an rms-equivalent bearing error.
        chi2 = np.square(res / sigma) + _cap_chi2(np.hypot(lx - px, ly - py), cap[None])
        rms = np.sqrt(np.mean(chi2, 1)) * float(np.mean(sigma))
        out[:, col] = rms / tol
    return out


def score_window(poses, tracks, rel, east, north, tol_rad, max_outliers,
                 sigma_rad, temperature) -> WindowScore:
    """Best in-cap identity per tracklet by window rms; rank hypotheses."""
    n_hyp = len(poses)
    n_tracks = len(tracks)
    best = np.full((n_hyp, n_tracks), np.inf)
    ident = np.full((n_hyp, n_tracks), -1, dtype=int)
    prior_cost = np.zeros((n_hyp, n_tracks))
    n_epochs = np.zeros(n_tracks)
    tol_col = np.full(n_tracks, tol_rad)
    for col, track in enumerate(tracks):
        arr = _epoch_arrays(track, rel)
        if arr is None:
            continue
        u, v, dtheta, bearing, kappa, cap, _ = arr
        n_epochs[col] = len(u)
        tol_col[col] = track_tolerance(arr, tol_rad)
        cand = track.cand_idx
        lx = east[cand][None]
        ly = north[cand][None]
        sigma_e = 1.0 / np.sqrt(np.maximum(kappa, 1e-9))
        sigma_mean = float(np.mean(sigma_e))
        for start in range(0, n_hyp, SCORE_CHUNK):
            block = poses[start:start + SCORE_CHUNK]
            acc = np.zeros((len(block), len(cand)))
            for e in range(len(u)):
                px, py = _platform_at(block, u[e:e + 1], v[e:e + 1])
                res = geo.wrap_rad(np.arctan2(lx - px, ly - py)
                                   - block[:, 2:3] - dtheta[e] - bearing[e])
                # Each epoch's one-sided cap holds at its own platform
                # position, softly: the excess joins the chi-square.
                acc += np.square(res / sigma_e[e]) + _cap_chi2(np.hypot(lx - px, ly - py), cap[e])
            # Candidate choice weighs geometry against the matcher's identity
            # prior, as the filter does: in a harbour with ~9 named candidates
            # per tracklet a wrong pose can fit every tracklet on some
            # low-confidence row; in a turbine field the prior is flat.
            logp = track.cand_logp if track.cand_logp is not None else np.zeros(len(cand))
            objective = 0.5 * acc - logp[None]
            j = np.argmin(objective, 1)
            rows = np.arange(len(block))
            rms = np.sqrt(acc[rows, j] / len(u)) * sigma_mean
            best[start:start + SCORE_CHUNK, col] = rms
            ident[start:start + SCORE_CHUNK, col] = cand[j]
            prior_cost[start:start + SCORE_CHUNK, col] = -PRIOR_SCORE_WEIGHT * logp[j]
    consistent = best < tol_col[None]
    n_consistent = consistent.sum(1)
    mean_rms = np.where(consistent, best, 0.0).sum(1) / np.maximum(n_consistent, 1)
    # Gaussian log-likelihood of the consistent tracklets' epochs under each
    # tracklet's own tolerance (bearing noise + declared heading drift), plus
    # the tolerance-boundary penalty for each inconsistent one, tempered.
    scale = np.maximum(tol_col[None] / math.radians(1.5), 1.0) * sigma_rad
    penalty = 0.5 * (tol_col[None] / scale) ** 2 * n_epochs[None]
    ll = np.where(consistent, -0.5 * (best / scale) ** 2 * n_epochs[None] - prior_cost, -penalty)
    ll = np.where(np.isfinite(ll), ll, -penalty)
    score = ll.sum(1) / max(temperature, 1e-9)
    return WindowScore(n_consistent=n_consistent, mean_rms_rad=mean_rms,
                       identity=np.where(consistent, ident, -1), score=score,
                       best_identity=ident)


def _dedupe(poses, score, config):
    """Best hypothesis per solution-cluster cell (position + heading)."""
    order = np.argsort(-score, kind="stable")
    cell = config.solution_cluster_position_m
    hcell = math.radians(config.solution_cluster_heading_deg)
    key = np.concatenate([np.round(poses[order][:, :2] / cell).astype(int),
                          np.round(poses[order][:, 2:3] / hcell).astype(int)], 1)
    _, first = np.unique(key, axis=0, return_index=True)
    return order[np.sort(first)]


def accumulate(poses, score, memory, odometry, keyframe_idx):
    """Add the decayed accumulated score of the nearest remembered site
    (moved to this keyframe by odometry) to each hypothesis."""
    if memory is None or len(memory.poses) == 0 or memory.keyframe_idx >= keyframe_idx:
        return score
    rel = relative_poses(odometry, keyframe_idx, keyframe_idx - memory.keyframe_idx)
    if memory.keyframe_idx not in rel:
        return score
    prev = to_current_frame(memory.poses, rel[memory.keyframe_idx])
    dist = np.hypot(poses[:, None, 0] - prev[None, :, 0], poses[:, None, 1] - prev[None, :, 1])
    dhead = np.abs(geo.wrap_rad(poses[:, None, 2] - prev[None, :, 2]))
    ok = (dist <= MEMORY_MATCH_M) & (dhead <= MEMORY_MATCH_RAD)
    matched = np.where(ok, memory.scores[None, :], -np.inf).max(1)
    # A site with no remembered counterpart is treated as if it had scored
    # one outlier tracklet worse than the worst remembered site: memory is
    # credit for persisting, never a penalty for having been proposed.
    floor = float(memory.scores.min()) - MEMORY_NEW_SITE_PENALTY
    credit = np.where(np.isfinite(matched), matched, floor) - floor
    return score + MEMORY_DECAY * credit


def _generate_and_score(measurements, odometry, tables, catalog, config, keyframe_idx,
                        window_keyframes, rng, east, north, bbox):
    """One window length: tracklets, exhaustive triples, refinement, scoring.
    Returns (poses, WindowScore, n_tracks, counts) or None."""
    cfg = msgspec.structs.replace(config, window_joint_keyframes=window_keyframes)
    tracks = collect_tracks(measurements, tables, catalog, cfg, keyframe_idx)
    n_gen = min(cfg.window_joint_resection_tracklets, len(tracks))
    if n_gen < 3:
        return None
    rel = relative_poses(odometry, keyframe_idx, window_keyframes)
    tol_rad = math.radians(cfg.window_joint_rms_tolerance_deg)
    sigma_rad = 1.0 / math.sqrt(max(
        min(kappa for t in tracks for _, _, kappa, _ in t.epochs), 1e-9))
    all_poses, total, examined, pruned = [], 0, 0, 0
    for trip in itertools.combinations(range(n_gen), 3):
        tracks3 = [tracks[i] for i in trip]
        k_ref, chosen = reference_epochs(tracks3)
        if k_ref not in rel:
            continue
        offsets = [math.hypot(rel[e[0]][0] - rel[k_ref][0], rel[e[0]][1] - rel[k_ref][1])
                   if e[0] in rel else 0.0 for e in chosen]
        poses, idents, n_total, n_pruned = resect_snapshot(
            tracks3, east, north, bbox, cfg.window_joint_max_tuples, rng,
            epochs=chosen, offsets_m=offsets)
        total += n_total
        examined += n_pruned
        pruned += n_total - n_pruned
        if len(poses):
            poses = to_current_frame(poses, rel[k_ref])
            refined = refine(poses, idents, tracks3, rel, east, north)
            own = own_track_rms(refined, idents, tracks3, rel, east, north, tol_rad)
            keep = (own < 1.0).all(1)
            if keep.any():
                all_poses.append(refined[keep])
    counts = (total, examined, pruned)
    if not all_poses:
        return None, None, len(tracks), counts, tracks
    poses = np.concatenate(all_poses)
    scored = score_window(poses, tracks, rel, east, north, tol_rad,
                          cfg.window_joint_max_outlier_tracklets,
                          sigma_rad, cfg.window_joint_temperature)
    return poses, scored, len(tracks), counts, tracks


def propose(measurements, odometry, tables, catalog,
            config: structs.ProposalConfig, event_id: int, keyframe_idx: int,
            trigger: str, *, particle_budget: int,
            rng: np.random.Generator, memory: WindowMemory | None = None
            ) -> tuple[proposal.ProposalResult, WindowMemory | None]:
    """Window-joint hypothesis set for an injection budget, and the memory
    to hand to the next event (unchanged when this event produced nothing).
    With `window_joint_keyframes_long` > 0 the hypothesis sets of both window
    lengths are pooled before ranking."""
    particle_budget = int(particle_budget)
    east = np.asarray(catalog.east_m, dtype=float)
    north = np.asarray(catalog.north_m, dtype=float)
    margin = 2000.0
    bbox = (east.min() - margin, east.max() + margin,
            north.min() - margin, north.max() + margin)
    windows = [config.window_joint_keyframes]
    if config.window_joint_keyframes_long > config.window_joint_keyframes:
        windows.append(config.window_joint_keyframes_long)
    pooled_poses, pooled_score, pooled_ncons, pooled_rms, pooled_ident = [], [], [], [], []
    total = examined = pruned = 0
    n_tracks_seen = 0
    union_tids = []  # tracklets across windows, for identity provenance
    for w in windows:
        out = _generate_and_score(measurements, odometry, tables, catalog, config,
                                  keyframe_idx, w, rng, east, north, bbox)
        if out is None:
            continue
        poses, scored, n_tracks, counts, tracks = out
        total += counts[0]
        examined += counts[1]
        pruned += counts[2]
        n_tracks_seen = max(n_tracks_seen, n_tracks)
        if poses is None:
            continue
        need = max(3, n_tracks - config.window_joint_max_outlier_tracklets)
        keep = scored.n_consistent >= need
        if not keep.any():
            best_tier = int(scored.n_consistent.max())
            if best_tier < 3:
                continue
            keep = scored.n_consistent >= best_tier
        # Identities are reported against this window's tracklet list; remap
        # the columns onto the union of tracklets seen across windows.
        for t in tracks:
            if t.tracklet_id not in union_tids:
                union_tids.append(t.tracklet_id)
        pooled_ident.append((scored.identity[keep], [t.tracklet_id for t in tracks]))
        pooled_poses.append(poses[keep])
        pooled_score.append(scored.score[keep])
        pooled_ncons.append(scored.n_consistent[keep])
        pooled_rms.append(scored.mean_rms_rad[keep])
    empty = proposal.ProposalResult(
        event_id=event_id, keyframe_idx=keyframe_idx, trigger=trigger,
        hypotheses=[], particle_budget=particle_budget,
        n_tracklets_considered=n_tracks_seen, n_combinations_total=total,
        n_combinations_enumerated=examined, n_combinations_sampled=0,
        n_combinations_geometry_pruned=pruned, n_partially_represented_ties=0,
        n_solution_clusters_merged=0, represented_compatibility_mass=0.0)
    if not pooled_poses:
        return empty, memory
    remapped = []
    for ident, tids_w in pooled_ident:
        full = -np.ones((len(ident), len(union_tids)), dtype=int)
        for col, tid in enumerate(tids_w):
            full[:, union_tids.index(tid)] = ident[:, col]
        remapped.append(full)
    poses = np.concatenate(pooled_poses)
    score = accumulate(poses, np.concatenate(pooled_score), memory, odometry, keyframe_idx)
    n_cons = np.concatenate(pooled_ncons)
    mean_rms = np.concatenate(pooled_rms)
    identity = np.concatenate(remapped)
    tids = tuple(union_tids)
    order = _dedupe(poses, score, config)
    merged = len(poses) - len(order)
    limit = min(config.window_joint_max_hypotheses,
                proposal._max_active_solutions(  # noqa: SLF001
                    proposal.TRIPLE, particle_budget, config))
    order = order[:limit]
    top = float(score[order].max())
    hypotheses = []
    for i in order:
        used = [c for c in range(len(tids)) if identity[i, c] >= 0]
        hypotheses.append(proposal.PointHypothesis(
            kind=proposal.TRIPLE,
            tracklet_ids=tuple(tids[c] for c in used),
            landmark_ids=tuple(str(catalog.landmark_ids[identity[i, c]]) for c in used),
            residual_rad=float(mean_rms[i]),
            compatibility_mass=float(math.exp(score[i] - top)),
            east_m=float(poses[i, 0]), north_m=float(poses[i, 1]),
            heading_rad=float(poses[i, 2]),
            position_sigma_m=config.window_joint_injection_sigma_m,
            heading_sigma_deg=config.window_joint_injection_heading_sigma_deg))
    mass = sum(h.compatibility_mass for h in hypotheses)
    for i, h in enumerate(hypotheses):
        hypotheses[i] = dataclasses.replace(h, compatibility_mass=h.compatibility_mass / mass)
    # Represented mass here reports how concentrated the proposal is: the
    # share of the kept hypotheses' mass on the best-consistency tier.
    best_tier = int(n_cons[order].max())
    tier_mass = sum(h.compatibility_mass for h, i in zip(hypotheses, order)
                    if n_cons[i] == best_tier)
    new_memory = WindowMemory(keyframe_idx=keyframe_idx, poses=poses[order].copy(),
                              scores=score[order].copy())
    return proposal.ProposalResult(
        event_id=event_id, keyframe_idx=keyframe_idx, trigger=trigger,
        hypotheses=hypotheses, particle_budget=particle_budget,
        n_tracklets_considered=n_tracks_seen, n_combinations_total=total,
        n_combinations_enumerated=examined, n_combinations_sampled=0,
        n_combinations_geometry_pruned=pruned, n_partially_represented_ties=0,
        n_solution_clusters_merged=merged,
        represented_compatibility_mass=float(tier_mass),
        best_score=float(top)), new_memory


def incumbent_score(east_m, north_m, heading_rad, measurements, odometry, tables,
                    catalog, config: structs.ProposalConfig, keyframe_idx: int,
                    refine_poses: bool = False):
    """Window score of given poses (the belief's) on the same footing as the
    proposal's hypotheses; None when the window has too few tracklets.

    With `refine_poses`, each pose is first Gauss-Newton-refined on the
    identities the window assigns it, as every generated hypothesis was: a
    particle 30 m off its site must not lose to a refined copy of the same
    site. Returns (score, refined poses)."""
    tracks = collect_tracks(measurements, tables, catalog, config, keyframe_idx)
    if len(tracks) < 3:
        return None
    rel = relative_poses(odometry, keyframe_idx, config.window_joint_keyframes)
    east = np.asarray(catalog.east_m, dtype=float)
    north = np.asarray(catalog.north_m, dtype=float)
    sigma_rad = 1.0 / math.sqrt(max(
        min(kappa for t in tracks for _, _, kappa, _ in t.epochs), 1e-9))
    poses = np.stack([np.asarray(east_m, float), np.asarray(north_m, float),
                      np.asarray(heading_rad, float)], 1)
    tol_rad = math.radians(config.window_joint_rms_tolerance_deg)
    scored = score_window(poses, tracks, rel, east, north, tol_rad,
                          config.window_joint_max_outlier_tracklets, sigma_rad,
                          config.window_joint_temperature)
    if not refine_poses:
        return scored, poses
    # Refine on the best identity of every tracklet, consistent or not: a
    # pose a few tens of metres off may have no consistent tracklet yet.
    poses = refine(poses, scored.best_identity, tracks, rel, east, north)
    scored = score_window(poses, tracks, rel, east, north, tol_rad,
                          config.window_joint_max_outlier_tracklets, sigma_rad,
                          config.window_joint_temperature)
    return scored, poses
