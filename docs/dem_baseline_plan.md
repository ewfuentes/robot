# CrossLocate-Depth Baseline: Design and Data Plan

Status: working design for implementation and experiment planning  
Primary baseline: CrossLocate-Depth  
Optional, independent task: HORAYZON skyline matching  
Last revised: 2026-08-26

## 1. Decision summary

The main elevation-map baseline will adapt the RGB-to-rendered-depth retrieval
formulation introduced by Tomešek et al. in CrossLocate. The baseline will be
evaluated under the same high-level localization conditions as the proposed
method:

- The initial pose belief is uniform over the declared evaluation region.
- Global heading is not supplied by GPS or compass.
- The primary prior-map product is a bare-earth DEM. In urban and coastal
  domains, also report a static LiDAR-derived DSM containing ground, buildings,
  and bridges. Vegetation is excluded from the primary DSM and may be added only
  as a separately named sensitivity condition.
- The baseline receives the same calibrated image stream and relative odometry
  as the proposed method.
- Test-region imagery is not used to train, fine-tune, select, or calibrate the
  retrieval model. The test-region DEM/DSM is permitted because it is the prior
  map against which localization is performed.
- The candidate support is identical across CrossLocate-Depth, LOCI, and the
  proposed method. If a physically justified navigable or traversable polygon
  restricts a river, harbor, or trail experiment, every method receives the
  same restriction.

CrossLocate-Depth is the named, learned geometric baseline. A separate
HORAYZON task may be implemented as a transparent, training-free skyline
control and as a diagnostic for whether a given elevation map contains usable
localization information. HORAYZON is not required for completion of
CrossLocate-Depth.

Recommended paper-facing names are:

- **CrossLocate-Depth (DEM)**: retrieval against a bare-earth terrain model.
- **CrossLocate-Depth (DSM)**: the same algorithm against a declared digital
  surface model.
- **CrossLocate-Depth + Bayes**: CrossLocate-Depth scores fused with odometry
  over time.
- **HORAYZON Skyline (oracle)** and **HORAYZON Skyline (automatic)**: optional
  analytic skyline variants, if completed.

## 2. Purpose and experimental role

CrossLocate-Depth should answer the following question:

> How much localization information can be recovered by matching the holistic
> geometry in ground imagery to a prior elevation or surface model, without
> using semantic landmark identities?

Here, “Depth” denotes the reference modality: an RGB query is retrieved against
synthetic depth views rendered from the elevation map, producing ranked observer
location/yaw hypotheses. The baseline does not estimate monocular query depth,
match landmark identities, or expose a bearing-only observation.

This is complementary to LOCI and to the proposed method:

- LOCI tests local semantic map matching.
- CrossLocate-Depth tests holistic geometric map matching.
- The proposed method tests persistent, open-world semantic association and
  nonlocal bearing constraints from far-field landmark instances.

The baseline must be competitive enough to be informative, but it should not be
given an easier localization problem. In particular, a local position window,
ground-truth heading, or target-region training imagery would invalidate the
main comparison.

## 3. Scope and non-goals

### 3.1 In scope

- Global or region-wide RGB-to-rendered-depth retrieval.
- Perspective query images and calibrated crops from cylindrical panoramas.
- Joint scoring of candidate location and heading.
- Framewise retrieval and temporal fusion with relative odometry.
- Public DEMs and publicly obtainable LiDAR-derived DSMs.
- Zero-shot use of released CrossLocate weights and geographically disjoint
  retraining if needed.
- Explicit measurement of map applicability and map/image mismatch.

### 3.2 Out of scope for the first implementation

- Reconstructing a textured photogrammetric city model.
- Using target-trajectory GPS or compass as an online input.
- Per-dataset manual tuning based on test results.
- Claiming an exact reproduction before the released CrossLocate preprocessing,
  depth encoding, and checkpoint behavior have been verified.
- Treating a top-1 retrieval score as a calibrated observation probability
  without validation.
- Making HORAYZON a dependency of the primary baseline.

## 4. Inputs, outputs, and allowed information

### 4.1 Online inputs

At time (t), the baseline receives:

- A calibrated RGB observation (o_t), either a perspective image or a
  cylindrical panorama.
- Camera intrinsics and panorama projection parameters.
- Measured camera-to-rig extrinsics and rig height above the local support
  surface or waterline.
- Gravity alignment or measured roll and pitch, if these are also available to
  the proposed method. No global yaw is provided.
- Relative SE(2) odometry (u_{t-1}) and its covariance from time (t-1) to (t).
- A declared candidate region shared with the other global localization
  methods.

Ground-truth position, ground-truth yaw, Mapillary coordinates, and compass
metadata are evaluation or training labels only. They are not online baseline
inputs.

### 4.2 Prior-map inputs

Each evaluation region provides:

- A georeferenced DEM or DSM.
- Horizontal and vertical coordinate reference systems.
- Map resolution, acquisition date, and source metadata.
- A water mask and declared water-surface treatment where relevant.
- Observer height above the local surface or waterline, determined from sensor
  calibration rather than test-image fitting.
- The candidate lattice and the declared rendering range.

### 4.3 Outputs

For each frame, CrossLocate-Depth produces:

1. A ranked list of candidate location-heading cells.
2. A sparse or dense score field over candidate (SE(2)) states.
3. A calibrated observation likelihood suitable for a Bayes or particle-filter
   update.
4. Diagnostics including retrieval entropy, score margin, top-(k) candidates,
   abstention/outlier probability, and map-coverage status.

The sequence version additionally produces the posterior belief, MAP pose,
posterior mass near ground truth, and convergence/failure indicators over time.

## 5. System design

### 5.1 Reference-map construction

For each region:

1. Acquire the raw DEM and, when needed, classified LiDAR or building-height
   data.
2. Preserve and hash every raw tile, then reproject every source into one local
   metric horizontal CRS and one declared vertical datum. NAD83(2011) / UTM
   zone 19N and NAVD88 / GEOID18 are the preferred project coordinates when the
   source metadata supports that transformation.
3. Construct the evaluation surface according to the fixed DEM or DSM recipe in
   Section 7.
4. Add a buffer around the candidate region large enough to render all visible
   far-field geometry. The buffer is determined from a declared physical render
   range, not from the test images. Mt. Washington may require terrain coverage
   approaching 100 km beyond the query route; the exact range is fixed before
   test evaluation.
5. Generate a regular candidate-location lattice. A hierarchical lattice is
   preferred so that the entire region can be searched coarsely and promising
   areas can be evaluated at the resolution needed for the paper's metrics.
6. At every candidate location, render a ring of perspective depth views with
   known pose and intrinsics.
7. Extract and store one descriptor for every depth view, together with all
   rendering metadata.
8. Build an exact or approximate nearest-neighbor index for framewise retrieval,
   while retaining the location and heading organization needed for joint
   panorama scoring.

The search polygon and all database bounds come from the declared prior before
evaluation; they must not be cropped around the ground-truth route. Candidate
locations that are physically impossible may be removed only by a support mask
shared by every compared method.

The published CrossLocate configuration provides the starting protocol:

- 12 headings per location.
- 30-degree yaw spacing.
- 60-degree horizontal field of view.
- Square perspective views.
- A 512-dimensional global descriptor.

These values should remain unchanged for the first New England experiments
unless the release-reproduction task shows that a different public checkpoint
expects different preprocessing.

As a first storage/compute point, use a 50--100 m global grid and optionally
rerender the retrieved top neighborhoods at 10--20 m. These are engineering
starting values, not settled experimental hyperparameters; they must be frozen
on validation regions without consulting test poses.

The original Uniform CrossLocate setup used a much coarser 500 m location grid.
That spacing cannot substantiate 20--100 m localization claims. The adaptation
therefore needs a denser database, or it must report the 500 m quantization
floor and use correspondingly coarse success radii.

### 5.2 Depth rendering

The preferred order of implementation is:

1. Reproduce the appearance and numeric encoding of the released CrossLocate
   depth images.
2. Use the released LandscapeAR `itr` renderer if it can generate compatible
   depth products reliably.
3. Otherwise implement a documented height-field or mesh ray caster that emits
   the same camera model and depth encoding.

Before rendering new regions, inspect the released CrossLocate data loader and
sample depth files to resolve:

- Whether pixels encode metric range, inverse depth, normalized depth, or a
  visualization transform.
- How sky and no-return pixels are represented.
- Near and far clipping behavior.
- Image resizing, channel replication, and intensity normalization.
- Camera height above the terrain and the treatment of slopes.
- Whether Earth curvature is represented by the original renderer.

Preserve raw metric floating-point depth and a valid/sky mask as the canonical
render product. Derive the versioned network tensor from that product. When
using released weights, reproduce their expected encoding exactly; otherwise
declare the clipping, depth transform, sky value, and channel conversion.
Never feed an undocumented visualization colormap to the network.

No new depth encoding should be selected solely because it performs well on a
test trajectory. If several reasonable encodings remain, choose on a separate
validation region and freeze the decision.

### 5.3 Query preprocessing

#### Perspective images

For an ordinary perspective query, rectify lens distortion, apply the released
CrossLocate resize/normalization, and extract one RGB descriptor. Candidate yaw
is represented by the heading of each reference depth view.

#### Panoramic images

The paper's principal observations are panoramic. A panorama is converted into
a ring of calibrated perspective crops using the same field of view and yaw
spacing as the reference ring. The relative yaw between query crops is known,
but the ring's global yaw is not.

This produces query descriptors

\[
    Q_t = \{q_{t,m}\}_{m=0}^{M-1},
\]

where (m) indexes relative query yaw. At candidate location (r_i), the
reference database stores

\[
    D_i = \{d_{i,n}\}_{n=0}^{N_\theta-1}.
\]

For discrete global-yaw shift (k), the default score is

\[
    S_t(i,k) = \operatorname{Agg}_{m}
    \operatorname{cos}\!\left(q_{t,m},
    d_{i,(m+k)\bmod N_\theta}\right).
\]

The initial aggregation should be the mean over valid crops. A robust mean or
trimmed mean may be selected on validation data if a small number of occluded or
water-dominated crops otherwise controls the result. Invalid crops and the
aggregation rule must be logged.

This circular alignment searches location and global heading jointly and does
not require compass initialization.

As a convention test, if reference view (j) has map yaw (psi_j) and query crop
(c) has body-relative azimuth (alpha_c), the implied robot heading is
`wrap(psi_j - alpha_c)`. Verify the sign and camera-to-body rotation with a
synthetic yaw round trip before any evaluation. All overlapping crops from one
panorama are consolidated into one pose factor; they are not treated as
independent measurements.

### 5.4 Embedding network

The first milestone should use the released CrossLocate checkpoint and
preprocessing without architectural changes. This establishes whether the
published RGB-to-depth representation can be executed and whether it transfers
at all to New England.

The faithful configuration to verify is a shared VGG-16 encoder for RGB and
depth, no ReLU after the final convolution, per-pixel L2 normalization, MAC
pooling, final descriptor L2 normalization, a 512-dimensional descriptor, and
Euclidean retrieval. Depth is duplicated to three channels and 500-by-500
renders are resized to 224-by-224 network inputs. Treat these as release-audit
targets: the code and fixture data take precedence if they reveal a documented
variant.

If a maintained implementation is required, port the released model to modern
PyTorch or TensorFlow while preserving:

- Network topology and descriptor dimensionality.
- Weight sharing or branch behavior present in the release.
- Pre- and post-feature normalization.
- Global pooling.
- Input preprocessing.
- Descriptor-distance convention.

Build an exact-nearest-neighbor index first and use it as a ranking fixture
before introducing an approximate index. Save raw descriptor distances and
stable database-view IDs. Any modern backbone or two-tower architecture is a
separately named extension, not the CrossLocate-Depth baseline.

The port is accepted only after descriptor and ranking agreement is measured on
a fixed sample from the released CrossLocate data. If the legacy and ported
implementations do not agree, report the new system as a reimplementation rather
than the original baseline.

### 5.5 From retrieval score to observation likelihood

CrossLocate similarity is not a probability by definition. For a candidate
state (x=(r_i,\theta_k)) in the discrete search space (X), first define

\[
    K_t(x) = \exp\!\left(\frac{S_t(x)-S_t^{\max}}{\tau}\right),
\]

and then mix the normalized score kernel with an explicit uniform outlier model:

\[
    L_t(x) = (1-\epsilon)
    \frac{K_t(x)}{\sum_{x'\in X} K_t(x')} +
    \frac{\epsilon}{|X|}.
\]

Temperature (tau) and outlier mass (epsilon) are selected once on geographically
disjoint validation trajectories after the encoder is frozen. Alternative
monotone calibration may be used if it improves held-out NLL, Brier score, or
expected calibration error, but test accuracy must not select the calibration.

For particle states between reference cells:

- Interpolate the score or descriptor field spatially when supported by the
  reference representation.
- Interpolate circularly between adjacent heading bins.
- Otherwise use a declared nearest-cell lookup and include the resulting
  quantization in the reported error floor.

Top-(k)-only operation should retain a uniform likelihood floor outside the
retrieved set so that one failed retrieval cannot irreversibly delete the true
hypothesis.

### 5.6 Temporal integration

The principal temporal baseline uses the same motion model, odometry stream,
initial candidate region, particle count, resampling policy, and stochastic
evaluation protocol as the proposed method wherever possible. The only changed
component is the image observation likelihood.

Report both:

- **CrossLocate-Depth framewise:** independent per-frame retrieval.
- **CrossLocate-Depth + Bayes:** one incremental CrossLocate-Depth likelihood
  update per selected frame, fused with odometry.

Do not repeatedly apply a cumulative likelihood. Do not initialize the filter
from ground-truth GPS, a local window, or Mapillary coordinates. If observations
are subsampled to match compute budgets, use the same fixed frame schedule or
declare the schedules separately.

Do not multiply overlapping panorama crops or adjacent video frames as if they
were conditionally independent. Crop evidence is pooled into one calibrated
panorama likelihood. Apply that factor once after the odometry prediction at a
fixed time/displacement cadence, or calibrate the aggregate observation at the
chosen cadence. Entropy or quality gating may abstain through the outlier model,
but its thresholds are fixed on validation data. Report odometry-only,
framewise retrieval, and CrossLocate-Depth + Bayes as separate conditions.

## 6. Training design

### 6.1 Three model conditions

Evaluate training in the following order:

1. **Released CrossLocate:** use the public Alps checkpoint with no target-domain
   adaptation.
2. **Port verification:** verify a maintained inference implementation against
   the released checkpoint.
3. **Disjoint retraining:** train or fine-tune on geographically disjoint real
   RGB images and corresponding rendered-depth candidates only if zero-shot
   transfer is inadequate.

The released model is valuable even if it fails because it distinguishes
geographic/domain transfer from map quality. Disjoint retraining tests whether
the formulation, rather than the Alps checkpoint, can transfer.

### 6.2 Geographic separation

Splits are made by geographic region, not by random frame or sequence:

- No Mt. Washington query imagery may appear in training or validation for the
  Mt. Washington test.
- No Charles River or Boston Harbor imagery may appear in training or validation
  for those tests.
- Nearby frames, reverse traversals, or another capture day in the same test
  corridor remain part of the held-out geographic region.
- The exclusion region extends beyond the test route by the maximum render
  range so that a training camera cannot observe the same long-range skyline.
  This may be tens of kilometers, and approximately 100 km for the mountain
  experiment if that is the fixed render range.
- If Mapillary supplies training images, training city/region IDs are recorded
  and disjoint from all evaluation regions. Split complete `sequence_id` groups
  and keep all crops from one panorama in one partition.
- Hyperparameters are selected on one or more separate validation regions and
  frozen before opening test results.
- Evaluation poses live in a separate sealed manifest that inference and map
  construction code cannot access.

The DEM/DSM for a test region is allowed at inference because the method is
defined as localization against a prior geometric map. Test imagery must not be
used to repair that map manually. Surface construction rules must be fixed from
source metadata and validation regions.

### 6.3 Training tuples

Start from CrossLocate's weakly supervised triplet formulation:

- A geotagged and heading-labeled real RGB query.
- A rendered-depth positive selected by metric proximity and viewing direction.
- Hard or semi-hard rendered negatives outside a declared exclusion radius.
- Photometric augmentation on RGB queries and only geometry-preserving
  augmentation across paired modalities.

First reproduce the published thresholds and mining behavior on the released
dataset. For a new, finer reference lattice, positive and negative radii must be
declared relative to cell spacing and the target accuracy before test
evaluation. A reasonable proposal is to treat the nearest compatible reference
cell as positive and to exclude a broad spatial buffer from negatives; the
exact radii are a validation decision, not a per-dataset test tuning parameter.

Record:

- Training regions and image counts.
- Reference locations and render counts.
- Positive/negative thresholds.
- Heading tolerance.
- Hard-negative mining schedule.
- Batch construction and augmentation.
- Initialization checkpoint.
- Optimizer, learning-rate schedule, epoch count, and random seeds.
- Validation-selection criterion.

Balance batches by route and region so that long videos do not dominate. Any
horizontal shift, flip, or crop augmentation that changes yaw must update the
orientation label or be excluded from orientation-supervised training. Audit
the checked-in training defaults against the paper and supplementary settings;
if they differ, publish the exact chosen configuration and do not call an
ambiguous default run a reproduction.

### 6.4 Query-data candidates for disjoint retraining

Potential sources, in priority order, are:

1. Released CrossLocate and LandscapeAR real/synthetic data.
2. Existing project imagery from geographic regions that will never be test
   regions.
3. Mapillary sequences from non-test regions with usable camera metadata.

Do not assume Mapillary compass or camera metadata are exact. They may create
training labels after quality control, but they may not become hidden online
priors at test time.

## 7. Geometric map and data design

### 7.1 DEM and DSM definitions

For this project:

- A **DEM** is a bare-earth elevation surface with vegetation and structures
  removed.
- A **DSM** is a visible-surface approximation that may include buildings,
  bridges, and vegetation derived from a declared public source.

The distinction must appear in the method name and every result table. A
bare-earth DEM is expected to be informative at Mt. Washington but may remove
most of the discriminative skyline in Boston and along the Charles River.

### 7.2 Proposed surface recipes

#### DEM recipe

1. Use the provider's highest-quality standard bare-earth raster available for
   the full region.
2. Preserve provider elevations until reprojection.
3. Reproject once into the experiment CRS with a documented resampler.
4. Fill only provider-declared no-data seams using a fixed, logged rule.
5. Do not add structures inferred from test imagery.

#### DSM recipe

1. Begin with the published bare-earth DEM.
2. Remove LiDAR points carrying noise or invalid classifications.
3. For the primary static DSM, retain ground, building, and bridge classes but
   exclude vegetation. Rasterize a robust upper-surface statistic at a fixed
   resolution. Compare maximum and high-quantile rasterization on a validation
   tile to avoid single-return spikes; freeze one rule.
4. Fill small holes using a fixed neighborhood rule and fall back to the DEM for
   cells with no valid surface returns.
5. Where a public building layer provides roof or top elevations, use it only
   through a deterministic region-wide rule, not manual selection based on test
   views.
6. Record whether bridges are retained or incomplete. If vegetation is useful,
   publish it only as a separately named `DSM+vegetation` sensitivity condition.

Height-field rasterization can create artificial sloped walls across roof and
shoreline discontinuities. Preserve discontinuities in the render mesh where
possible, or quantify the artifact on representative validation tiles. Also
record the source acquisition date relative to the query capture date; a
post-query map is a temporal mismatch that must be replaced or disclosed.

Report DEM and DSM variants separately where practical. This distinguishes
algorithm failure from absence of the visible geometry in the prior map.

### 7.3 Water and vertical datum

Water requires explicit treatment because bathymetry is not a visible surface:

- Rasterize a water mask from a declared public source.
- Set water cells to a declared water-surface elevation rather than the seafloor.
- Record the relationship among map datum, camera height, river level, and tide.
- For Boston Harbor, associate image timestamps with a tide estimate only if the
  same deterministic policy is available for all relevant frames. Otherwise use
  a declared nominal surface and quantify sensitivity.
- Convert ellipsoidal and orthometric heights explicitly; do not silently mix
  them.
- For Boston Harbor, use NOAA VDatum for tidal/orthometric conversion and record
  the station, observation, and uncertainty used for each water-level estimate.

Long-range coastal renders should account for Earth curvature. Atmospheric
refraction may be included as a fixed physical correction or tested as a map
model sensitivity, not fit per image.

### 7.4 Dataset/source matrix

| Evaluation domain | Query imagery | Primary geometric source | Main surface | Expected fit | Principal risks |
|---|---|---|---|---|---|
| Mt. Washington | Project hiking/panoramic sequences | Pinned USGS 3DEP project: latest complete 1 m bare-earth DEM; source QL2 LiDAR if needed | DEM; optional first-return DSM | High | Canopy, snow, haze, terrain beyond tile buffer, camera elevation |
| Charles River | Project boat/panoramic sequences | MassGIS 2021 Central--Eastern Massachusetts QL1 LiDAR and 0.5 m DEM | Static DSM, with DEM ablation | Low to medium | Low relief, canopy, buildings and bridges, repetitive banks, water elevation |
| Boston Harbor | Project maritime/panoramic sequences | Same MassGIS LiDAR/DEM; one pinned NOAA coastal project only for verified gaps | Static DSM, with DEM ablation | Medium if skyline geometry is represented | Tide/datum, ships and cranes, map age, incomplete island/building geometry, water dominance |
| Optional Mapillary regions | Perspective or spherical Mapillary sequences | USGS 3DEP or a documented local public DSM | Depends on region | High only where terrain/surface geometry is visible | Camera/FOV metadata, urban map completeness, domain shift, imagery terms and redistribution |

Primary source links:

- CrossLocate project and code:
  <https://github.com/JanTomesek/CrossLocate>
- CrossLocate project page:
  <https://cphoto.fit.vutbr.cz/crosslocate/>
- CrossLocate paper:
  <https://openaccess.thecvf.com/content/WACV2022/html/Tomesek_CrossLocate_Cross-Modal_Large-Scale_Visual_Geo-Localization_in_Natural_Environments_Using_Rendered_WACV_2022_paper.html>
- LandscapeAR and the `itr` renderer:
  <https://github.com/brejchajan/LandscapeAR>
- USGS LidarExplorer:
  <https://www.usgs.gov/tools/lidarexplorer>
- USGS 3DEP products:
  <https://www.usgs.gov/3d-elevation-program/about-3dep-products-services>
- USGS New Hampshire coverage:
  <https://www.usgs.gov/ngp-user-engagement-office/news/new-3d-elevation-program-fact-sheet-new-hampshire>
- MassGIS LiDAR terrain data:
  <https://www.mass.gov/info-details/massgis-data-lidar-terrain-data>
- MassGIS building structures:
  <https://www.mass.gov/info-details/massgis-data-building-structures-2-d>
- City of Boston building geometry and elevations:
  <https://gis.boston.gov/arcgis/rest/services/Assessing/DOIT_buildings/MapServer/2>
- NOAA coastal LiDAR:
  <https://www.coast.noaa.gov/digitalcoast/data/coastallidar.html>
- NOAA VDatum:
  <https://vdatum.noaa.gov/>
- Mapillary image metadata:
  <https://mapillary.github.io/mapillary-python-sdk/docs/mapillary.config.api/mapillary.config.api.entities/>

### 7.5 Data manifest

The minimum logical schema has three linked records:

- `trajectory`: domain, sequence, split, frame timestamp, panorama URI and
  SHA-256, projection/resolution, intrinsics, camera extrinsics, relative
  odometry and covariance, evaluation-only WGS84/local pose, and environmental
  flags.
- `db_view`: stable view and location IDs, WGS84 and local metric coordinates,
  ellipsoidal and orthometric altitude, camera height, yaw/pitch/roll, field of
  view, render dimensions, and DEM/render version.
- `render_manifest`: elevation product and tile IDs, source URL/date/terms,
  resolution, CRS and vertical-datum transform, no-data/water policy, curvature
  and refraction settings, search range, depth encoding/clipping, valid/sky
  mask, renderer commit, and checksums.

Every raw or derived dataset should have a machine-readable manifest containing:

- Stable dataset and region ID.
- Intended split: train, validation, or test.
- Bounding geometry and excluded buffer.
- Query source, capture dates, platform, and camera model.
- Intrinsics, projection type, image dimensions, and calibration version.
- Pose/heading source and whether each field is training-only, evaluation-only,
  or available online.
- Odometry source and coordinate convention.
- Map provider, product/work-unit IDs, acquisition dates, and download date.
- Native CRS, vertical datum, units, resolution, and no-data value.
- Raw-file checksums and licenses or terms.
- Derivation code version, parameters, and output checksums.
- Known coverage gaps and anomalies.

For a derived DSM, additionally record point classifications used, raster
statistic, cell size, gap-filling rule, building overlay rule, water treatment,
and all datum transformations.

Each query manifest also records image provenance and license, timestamp,
camera-to-rig extrinsics, distortion model, weather/visibility, and tide or
water level where relevant. Ground-truth pose and its uncertainty are stored
separately from inference-visible metadata.

### 7.6 Storage and release policy

- Keep raw elevation tiles, derived surfaces, depth renders, descriptors, and
  run outputs outside the paper repository.
- Version control code, small manifests, region polygons, configurations, and
  checksums.
- Retain a small set of representative renders for visual regression tests.
- Government elevation products are generally straightforward to reference and
  redistribute subject to their metadata; preserve the source notices.
- Confirm the CrossLocate repository and dataset licenses before copying code or
  republishing assets. The public repository does not currently display a
  software license; citation instructions are not a license. Obtain permission
  before distributing copied code or a weight-bearing port, or publish a clean
  implementation from the paper.
- For Mapillary, release image IDs, region queries, metadata-processing code,
  attribution records, and split manifests unless the applicable CC BY-SA terms
  and per-image attribution requirements are satisfied for redistributed
  pixels. Never store access tokens in the repository.
- Do not make the City of Boston building layer a required reproducibility
  dependency: its license restricts sublicensing. It may be used for internal QA
  or after written permission; the MassGIS point cloud remains the reproducible
  DSM source.

## 8. Evaluation protocol

### 8.1 Required baseline conditions

At minimum, report:

1. Released CrossLocate checkpoint, framewise.
2. Released CrossLocate checkpoint with temporal Bayes filtering.
3. Disjointly retrained model, framewise, if retraining is completed.
4. Disjointly retrained model with temporal Bayes filtering.
5. DEM versus DSM where both maps are defensible.

If a condition cannot run because the necessary surface model or camera
metadata is unavailable, report it as not applicable and state why. Do not
silently omit difficult frames or substitute a local pose prior.

### 8.2 Framewise metrics

- Recall@(k) within 25, 50, 100, 250, 500, and 1000 m, where compatible with
  region size.
- Joint position-and-yaw recall at declared angular thresholds.
- Position error of top-1 and best top-(k) candidate.
- Heading error at the retrieved location.
- Correct-cell rank and top-(k) coverage.
- Median and percentile error, not only mean error.
- Retrieval entropy, top-1/top-2 margin, NLL, Brier score, expected calibration
  error, and abstention coverage-risk where defined.
- Runtime split into query descriptor extraction, search, and scoring.
- Reference-database storage and offline rendering cost.

### 8.3 Sequence metrics

Use the paper's main localization metrics:

- Success within fixed radii.
- Time and distance to first convergence.
- Final MAP position and heading error.
- Posterior credible-region area.
- Posterior probability mass near ground truth.
- NLL of the true pose where the representation permits it.
- False-convergence rate and recovery rate.
- Performance over repeated filter seeds.

### 8.4 Applicability and diagnostic metrics

Report the fraction of frames with:

- Valid map coverage through the full render range.
- Valid camera calibration and projection metadata.
- Sufficient non-sky/non-water visual content.
- A usable rendered depth view rather than all sky/no-return.
- A top-(k) candidate inside the evaluation region.

Stratify results by domain, surface type, visible-terrain fraction, map age,
weather/visibility, and distance to dominant geometry. This prevents results
from hiding selective applicability.

### 8.5 Fairness controls

- Use the same global candidate region as the proposed method and LOCI.
- Do not use compass heading for CrossLocate-Depth unless every compared method
  receives the same heading prior in a separately labeled experiment.
- Gravity alignment and calibrated intrinsics are allowed when shared.
- Freeze observation cadence, likelihood calibration, and surface-processing
  rules before test evaluation.
- Give the framewise and temporally filtered variants separate names.
- Report the reference grid's quantization floor.
- Include a random or map-prior retrieval result to expose region-size effects.
- Report total candidate locations/views, render/index preprocessing time,
  stored bytes, query latency, and peak CPU/GPU memory.

## 9. Implementation work packages

### CLD-0: Reproduction and license audit

Deliverables:

- Downloaded public code, checkpoint, metadata, and a small released data subset.
- Recorded software and dataset licenses or unresolved restrictions.
- Legacy environment capable of extracting query and depth descriptors.
- Reproduced descriptor rankings on a fixed released-data sample.
- Documented depth encoding and preprocessing.

Exit criterion: the released model can be evaluated deterministically, or the
specific blocker is documented and a clean-room reimplementation is approved.

### CLD-1: Renderer validation

Deliverables:

- One georeferenced DEM tile rendered with the CrossLocate camera ring.
- Pixel-level comparison against released render conventions.
- Automated render metadata and visual regression tests.
- Surveyed-pose QA covering image/render horizon overlays, known peak
  azimuth/elevation, yaw round trips, no-data seams, and depth-range checks.
- Measured rendering throughput and storage.

Exit criterion: new depth renders pass geometric sanity checks at known camera
poses and the embedding network produces nondegenerate descriptors.

### CLD-2: Mt. Washington minimum viable baseline

Deliverables:

- Complete Mt. Washington DEM manifest and derived surface.
- Candidate reference database and descriptor index.
- Perspective and panorama query pipelines.
- Framewise top-(k) evaluation and qualitative retrieval panels.

Exit criterion: the full candidate region is searched without a local prior and
results can be reproduced from manifests and configuration.

### CLD-3: Pose likelihood and Bayes integration

Deliverables:

- Circular panorama/location-heading score field.
- Calibrated observation likelihood.
- Integration with the common odometry filter.
- Framewise versus temporal results under identical initial beliefs.

Exit criterion: the filter never consumes ground-truth position or heading and
can retain multiple hypotheses after an ambiguous frame.

### CLD-4: Massachusetts DSM pipeline

Deliverables:

- Deterministic LiDAR-to-DSM processing for Charles River and Boston Harbor.
- Building and water-surface integration rules.
- DEM/DSM render comparisons at known poses.
- Map coverage, age, and datum audit.

Exit criterion: visible skyline geometry is represented to the extent allowed
by the public sources, and remaining mismatches are documented rather than
manually repaired.

### CLD-5: Geographically disjoint retraining

Deliverables:

- Frozen region-level train/validation/test registry.
- Training query and render manifests.
- Reproduced training configuration and seeds.
- Zero-shot versus retrained transfer study.

Exit criterion: no evaluation-region image or nearby traversal appears in
training/validation, and model selection uses only declared validation regions.

### CLD-6: Main experimental package

Deliverables:

- All required framewise, temporal, DEM, and DSM conditions.
- Runtime/storage and applicability results.
- Qualitative success and failure cases.
- Paper-ready baseline description with third-person narrative citations.

## 10. Key risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Legacy TensorFlow/Python stack | Reproduction or GPU failure | Freeze a minimal legacy inference environment, then port with descriptor-level regression tests |
| Unknown depth encoding or renderer details | New maps are out of distribution | Inspect released data/code first; validate render histograms and descriptors before scaling |
| Bare-earth DEM omits visible urban/coastal structure | Artificially weak baseline | Build and report a deterministic DSM variant; retain DEM as an ablation |
| Alps-to-New-England domain shift | Released checkpoint fails for reasons unrelated to formulation | Report zero-shot result and train on geographically disjoint regions |
| Target-region leakage | Invalid unseen-environment claim | Split by buffered geographic region and audit every image ID |
| Heading or FOV metadata becomes a hidden prior | Unfair comparison | Search yaw jointly; use only shared calibration/gravity inputs |
| Water, datum, or camera-height error | Large angular error in coastal scenes | Declare surface elevation and datum conversion; run physical sensitivity tests |
| Reference database becomes too large | Excessive rendering, storage, or query time | Hierarchical grid, descriptor-only storage, batched extraction, FAISS/exact-search comparison |
| Descriptor scores are overconfident | Catastrophic filter collapse | Validate temperature and likelihood floor; retain a uniform outlier component |
| Dynamic ships, foliage, cranes, and weather dominate | Map/image mismatch | Report applicability strata and failures; do not edit the prior map from test imagery |
| Public code/data lack clear reuse terms | Release risk | Complete license audit before copying or redistributing; reimplement if necessary |

## 11. Optional independent task: HORAYZON skyline matching

### 11.1 Purpose

HORAYZON provides an analytically interpretable test of the elevation map's
skyline information. It is useful even if it never becomes a full paper
baseline because it can answer:

- Does the DEM/DSM predict the visible skyline at ground-truth poses?
- Is the correct region distinguishable using geometry alone?
- Is a CrossLocate-Depth failure caused by the embedding/domain gap or by the
  map lacking the visible surface?

This task should be scheduled and tracked separately from CrossLocate-Depth.
CrossLocate-Depth completion must not depend on it.

HORAYZON source:
<https://github.com/ChristianSteger/HORAYZON>

HORAYZON is MIT-licensed and supports arbitrary observer locations, curved-Earth
terrain, and multiscale high-resolution terrain workflows. Verify and pin the
exact release used in the experiment manifest.

A recent long-standoff localization example using HORAYZON-style synthetic
horizons and correlation is available at:
<https://www.mdpi.com/2076-3417/16/15/7397>

### 11.2 Minimal pipeline

1. Ingest the same declared DEM or rasterized DSM used by CrossLocate-Depth.
2. At each candidate location, compute a 360-degree horizon elevation profile
   (h_i(\alpha)), using measured observer height, Earth curvature, and a fixed
   search range. HORAYZON's location-based horizon routine is the intended API;
   also request horizon distance for geometry and coverage diagnostics.
3. For a query image, extract the sky/non-sky boundary.
4. Convert query boundary pixels through the calibrated camera model into
   elevation angle versus relative azimuth, masking occlusions and unreliable
   columns.
5. Search circular yaw shift and candidate location using normalized
   correlation, MASS, robust squared error, or a predeclared combination.
   Low-pass or Fourier-domain comparison should be tested on validation data to
   reduce sensitivity to pixel-scale skyline errors without erasing distinctive
   terrain structure.
6. Convert the score surface into a likelihood using validation-only
   calibration.
7. Optionally fuse the likelihood with the common odometry filter.

Use the same candidate grid, prior support, altitude convention, outlier model,
and temporal-update policy as CrossLocate-Depth wherever the representation
allows. Report azimuth resolution, DEM resolution, search radius, camera-level
sensitivity, valid terrain-horizon coverage, and runtime. HORAYZON generates a
one-dimensional terrain horizon; it is not a dense perspective-depth renderer
and is not a backend for CrossLocate-Depth.

The synthetic profile and query profile should remain one-dimensional. This is
the important distinction from CrossLocate-Depth, which learns a descriptor
between an RGB perspective image and a two-dimensional depth render.

### 11.3 Two query-boundary conditions

- **Oracle skyline:** manually corrected boundary on a representative evaluation
  subset. This estimates the information ceiling of the prior surface.
- **Automatic skyline:** one fixed sky/ground segmenter or documented classical
  extractor, trained only on disjoint data if training is required.

Run the oracle condition first. If the oracle boundary cannot retrieve the
correct region, additional skyline-network training is unlikely to be useful.

### 11.4 HORAYZON work packages

#### HZ-0: Geometry smoke test

- Install HORAYZON in an isolated maintained environment.
- Render profiles from a small Mt. Washington DEM tile.
- Verify azimuth convention, observer height, curvature, units, and datum.
- Overlay the predicted profile on several calibrated ground-truth images.

#### HZ-1: Oracle retrieval

- Create a small, versioned set of manually corrected query skylines.
- Build a candidate profile database over the full Mt. Washington region.
- Search location and yaw without GPS/compass initialization.
- Compare ground-truth rank and score maps against random/map-prior baselines.

#### HZ-2: Automatic skyline extraction

- Select one fixed extractor on geographically disjoint validation imagery.
- Produce per-column confidence and obstruction masks.
- Measure boundary angular error and localization degradation relative to the
  oracle.

#### HZ-3: Optional temporal baseline

- Calibrate an outlier-robust horizon likelihood.
- Integrate it into the common Bayes filter.
- Report framewise and temporal metrics under the same uniform initial belief.

### 11.5 HORAYZON decision gates

Proceed from HZ-0 to HZ-1 only if predicted ground-truth profiles align with the
dominant visible skyline after coordinate and datum checks. Proceed to the
automatic extractor only if oracle retrieval ranks the correct region
materially above chance on held-out frames. Promote HORAYZON to a main-table
baseline only if automatic extraction has adequate coverage and the result is
not based on selectively chosen clear-sky frames.

If a gate fails, retain the result as a documented map-information diagnostic
and stop the optional task.

### 11.6 HORAYZON deliverables

- Profile-generation configuration and surface manifest.
- Candidate profile database and checksums.
- Oracle skyline annotations for the diagnostic subset.
- Automatic skyline predictions and confidence masks, if attempted.
- Ground-truth overlays, retrieval heatmaps, and applicability statistics.
- A clear statement that the method is a project implementation inspired by
  prior profile-matching work unless an exact published pipeline is reproduced.

## 12. Open decisions to resolve during CLD-0 and CLD-1

1. Can the released CrossLocate checkpoint and dataset be used under sufficiently
   clear terms, or is a clean reimplementation required?
2. What exact depth transform and sky/no-return convention does the released
   checkpoint expect?
3. Can LandscapeAR `itr` produce compatible renders on the target compute
   environment?
4. What coarse and fine lattice spacings meet the accuracy target without
   making the reference database impractical?
5. What fixed physical far range and curvature model should all regions use?
6. What deterministic DSM rasterization and building-overlay rule will be used?
7. Which disjoint geographic regions can support retraining without weakening
   the previously untraversed test claim?
8. Should panoramic crop aggregation use a mean, robust mean, or learned
   aggregation selected strictly on validation regions?
9. Will the main paper report both released and retrained models, or only retain
   retraining if it materially changes the scientific conclusion?
10. Is the optional HORAYZON task sufficiently informative after the Mt.
    Washington oracle smoke test to justify automatic skyline work?

## 13. Paper-writing conventions

- Refer to publications narratively when they are the grammatical subject, for
  example: “Tomešek et al. introduce CrossLocate ...”.
- Do not describe LOCI possessively in the double-blind submission.
- State clearly that CrossLocate-Depth is an adaptation when panorama
  aggregation, a new renderer, a new surface source, or a modern network port is
  used.
- Do not compare HorizonNet's local-prior meter-level result directly with the
  uniform-prior global results.
- Give DEM and DSM rows distinct names and disclose map source, resolution,
  acquisition date, and preprocessing.
- Report not-applicable conditions and applicability coverage rather than
  evaluating only favorable frames.

