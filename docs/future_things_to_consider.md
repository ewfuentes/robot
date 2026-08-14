# Future things to consider

Ideas parked deliberately: each is a real improvement that is not worth doing
yet, with the reason it is not worth doing yet. Delete an entry when it lands
or when the reasoning that deferred it stops holding.

---

## Derive bearing κ from observed scatter, not from box width

**Where it lives now.** `landmark_filtering/object_tracking/track_merge.py:fuse_bearings`
sets a fused bearing's concentration purely geometrically:

```python
width  = (box[2] - box[0]) / pano_w * 360.0   # object's angular extent
sigma  = hypot(bearing_sigma_deg, mean_width / 4.0)   # base 1.0 deg
kappa  = 1 / radians(sigma) ** 2
```

Two terms in quadrature: a fixed 1° floor standing in for tracker/mask/
calibration error, plus a quarter of the object's angular width as centroid
ambiguity. On Boston Harbor leg 1 that yields σ from 1.0° (point-like object)
to 25.5° (an object spanning 102° of azimuth — a coastline).

**What it deliberately ignores.** The docstring is explicit that κ does *not*
grow with the number of fused keyframes, "the conservative choice while the
correlation is unmodelled". That is right, and it matches design-doc §3.1:
consecutive bearings on one object share a mask and a tracker, so treating
them as independent would overcount evidence — the frame-vs-tracklet error.
A tracklet seen across 20 keyframes gets exactly the same κ as one seen twice.

**What it costs.** κ is a property of the bounding box, not a statement about
how well the track actually behaved. Nothing about the observed consistency of
the bearings feeds in. So:

- A long, geometrically self-consistent track is genuinely better evidence
  than a brief one, and that is thrown away.
- A track that is wide *and wobbling* is indistinguishable from one that is
  wide and rock-steady, though the first deserves far less trust.
- Conversely a narrow box on a jittering mask reports 1.0° and is believed.

**The upgrade.** Estimate κ empirically from the bearing scatter *within* the
epoch, after compensating vehicle rotation between keyframes (odometry is
world-frame, so Δheading is available): the residual spread about the fused
mean is a direct measurement of how concentrated the bearing really is. Blend
it with the width-based prior — the geometric term is a sensible floor, since
an extended object's centroid cannot be sharper than its extent — and cap the
gain so a short epoch cannot claim high confidence from two agreeing samples.

**Why not yet.** It is only sound once the intra-tracklet correlation is
modelled: scatter across correlated samples underestimates the true
uncertainty, so an empirical κ derived from a shared-mask tracker would be
optimistic in exactly the direction the audit's overconfidence findings warn
about. Do this together with a correlation model (or an effective-sample-size
discount), not before. Until then the conservative geometric κ is the right
default, and the filter is separately protected by the null hypothesis and by
per-hypothesis injection spread.

**How we would know it worked.** The consistency gate (T-F2 NEES) on a real
leg, plus the bearing-residual distribution: a well-estimated κ should make
the normalized residual (residual / σ) approximately unit-variance across the
whole range of object widths. That is directly checkable on the leg 1 export
today — the current geometric κ almost certainly fails it at the wide end.
