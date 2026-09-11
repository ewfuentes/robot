# Franconia A1 proposal experiment

This branch is an archived research snapshot, not proposed production
behavior. It preserves the localization-proposal code used for the Franconia
A1 study after the project selected the simpler causal camera-heading path.
All added proposal controls default to disabled.

## What is preserved

The experiment extended the legacy snapshot proposer for sparse observations
spread across multiple keyframes. It added:

- odometry-yaw transport into the proposal trigger frame;
- refinement against each observation's moving anchor pose;
- compatible pair-to-triple joining and moving-anchor SE(2) resection;
- optional bearing-precision weighting and Fisher proposal covariance;
- an observable-recovery guard and rejected-gate RNG neutrality; and
- replay, CLI, old-manifest compatibility, and focused tests for those paths.

The snapshot is the ten-file change under `localization/` committed with this
note. Camera-heading dataset support, epoch alignment during export, and
tier-1 odometry derivation are intentionally excluded.

## Recorded result

On Franconia leg 1 (808 keyframes), the complete A1 proposal stack with O5
heading correction produced the following three-seed results:

| Seed | mass@500 | MAP within 500 m | Median MAP error | Final MAP error |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.143176 | 177/808 | 1170.5 m | 2198.5 m |
| 1 | 0.117835 | 171/808 | 1996.8 m | 7349.3 m |
| 2 | 0.129712 | 175/808 | 1411.5 m | 2851.0 m |

This establishes that the compound A1/O5 experiment had localization signal;
it does not isolate the proposal stack's contribution. O5 also had a reviewed
visual failure and was not selected for dataset publication.

## Relationship to other work

This is not the `window-proposal` implementation. That branch owns the ongoing
proposal research direction and uses a separate window-joint architecture with
multi-track identity persistence, site memory, soft range caps, and short/long
windows. The exact A1 controls archived here do not exist there.

The production-oriented work merged separately in PR #718 loads the optional
per-frame camera heading and uses it to derive tier-1 odometry before applying
the existing #714 IMU noise profile. Within-epoch observation alignment was
measured during this experiment but deliberately excluded from that PR; the
legacy measurement reducer remains unchanged.
