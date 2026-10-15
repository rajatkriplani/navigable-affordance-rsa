# Pipeline Details

This document records parameter provenance, statistical design, and
verification results. It is the reference for the Methods section.

## 1. Overview

Input: 50 indoor scene photographs from Bonner & Epstein (2017) Exp. 2.
Output: an (50, 50) RDM representing pairwise dissimilarity in navigable
affordance space, and its RSA correlation against OPA, PPA, and EVC.

Five stages: segment → extract features → build RDM → RSA → ablation.

## 2. Affordance extraction

Each floor mask is reduced to an 8-dimensional vector. The full
specification — taxonomy, per-feature formulas, computational lineage, and
empirical results across OPA/PPA/EVC — is in `docs/features.md`.

In brief, the features fall into three mathematical invariance classes:
one topological invariant (SC), three geometric (BPD, GTI, NNE), and four
metric (PFF, MAC, BCR, SCE). The pipeline's "Topology" / "Clearance" labels
are a coarser presentational grouping of these classes.

Skeleton is computed with `skimage.morphology.medial_axis`. The skeleton
tie-breaking RNG is a dedicated `default_rng(0)` — deterministic and
isolated from the caller's RNG, which is reserved for GTI endpoint sampling.

## 3. RDM construction

Features are z-score standardised (StandardScaler), then pairwise Euclidean
distances produce the (50, 50) RDM via `scipy.spatial.distance.squareform`.
The lower-triangle extraction used by RSA and ablation uses `np.tril_indices`,
which is the same ordering produced by `pairwise_distances_to_tril` in
`src.stats`. This is enforced by regression tests.

NaN rows (SAM failures) are excluded from the RDM. The RDM has NaN in every
position touching an excluded row, and a boolean `valid_mask` is returned.

## 4. Parameter choices and sensitivity

### 4.1 `r_base_px` — body clearance radius

- **Value:** 60.0 pixels (default).
- **Units:** pixels, not cm. This parameter is defined at the image bottom.
- **Provenance:** this parameter does not exist in Bonner & Epstein (2017),
  whose model is angular and direction-only. It is required by our metric
  clearance model, which operationalises the width/depth of navigable
  space — a dimension the L/C/R model omits.
- **Calibration:** 60 pixels corresponds to the approximate body clearance
  subtended at the image bottom for the 768×1024 Bonner stimuli.
- **Validation:** cross-checked against the horizontal spread of human-drawn
  trajectories in the lower portion of each image (source: `Trajs.mat`,
  obtained by direct correspondence with M. Bonner).
- **Sensitivity:** `affordance-sensitivity` sweeps
  `r_base_px ∈ {30, 45, 60, 80, 120}` and reports whether the ablation
  verdict (clearance vs. topology) is robust across the range.

### 4.2 `horizon_ratio` — perspective origin

- **Value:** 0.5 (default).
- **Interpretation:** fraction of image height at which the visual horizon sits.
- **Provenance:** derived, not chosen. Bonner & Epstein (2017) photographed
  their Exp. 2 stimuli at eye level; for a level camera at eye height,
  pinhole projection places the visual horizon at the image midline.
- **Validation:** cross-checked against the convergence point of the
  deepest-reaching segment of human-drawn trajectories. Mean convergence y
  across scenes and raters is reported in
  `outputs/stats/trajectory_validation.json`.
- **Sensitivity:** `affordance-sensitivity` sweeps
  `horizon_ratio ∈ {0.4, 0.45, 0.5, 0.55, 0.6}`.

### 4.3 Sensitivity results

Full results in `outputs/stats/sensitivity.json`. The headline finding
(clearance, not topology, drives the OPA signal) is robust across the
default sweep range.

## 5. Relationship to Bonner & Epstein (2017)

The 2017 paper modelled navigational affordances using subjective human
path drawings, reduced to three angular channels (left / centre / right).
This repo replaces that with an objective, reproducible geometric
representation. The two models are complementary, not competing:

- Their L/C/R channels encode path *direction*.
- Our clearance features encode path *metric width and depth*.
- Their model is not reproducible without human raters; ours is.
- Their model has no body-scale parameter; ours does (see §4.1).

The paper's Discussion explicitly leaves the egocentric reference frame as
an open question ("Our data suggest that the OPA may encode affordances in
an egocentric reference frame but this remains to be tested"). The clearance
family is a first operationalisation of that question.

**What we do not claim:** we do not refute Bonner & Epstein (2017), and we
do not claim OPA encodes topology. Our ablation shows clearance drives the
signal and topology does not.

## 6. Statistical design

### 6.1 Permutation testing

Every reported p-value is a label-permutation p-value with 10,000 iterations.
Permuting the model RDM's stimulus labels breaks any true association while
preserving the RDM's internal structure. The brain and covariate RDMs stay
fixed. The p-value is the fraction of permuted draws with a partial
correlation at least as extreme as the observed.

### 6.2 FDR correction

Benjamini-Hochberg FDR correction is applied within each family of tests:
- RSA family: 10 tests (6 standard + 4 partial).
- Ablation family: 8 tests (one per feature).

Adjusted q-values are reported alongside raw p-values.

### 6.3 Noise ceiling

Nili et al. (2014) leave-one-subject-out noise ceiling is computed when
per-subject per-ROI RDM stacks are available. The current dataset provides
per-voxel RDMs at `data/rdms/all_sub_slrdms.npy`, but per-subject ROI masks
are not available, so ROI-level noise ceilings are not yet computable. This
is noted in the paper's limitations.

### 6.4 Reproducibility

All random state is local. Each test receives its own `np.random.Generator`
derived from a shared `SeedSequence` spawned from the master seed. Test
execution order does not affect results.

## 7. Verification against the Bonner data

Three verifications confirm the pipeline's data handling:

1. **Per-rater trajectory maps match the aggregated maps.**
   `allPaths[:, :, k] == binary_sum(Trajs[winner_k])` at r = 1.0000 for all
   50 fMRI scenes. Confirms the pathway ↔ RDM mapping.

2. **binCounts rows match Trajs field order.**
   Mass correlation r = 0.9984 between `binCounts` row sums and `Trajs`
   field masses. Confirms the 173 ↔ 173 alignment.

3. **Angular histogram reconstruction.**
   Reconstructing 180-bin forward-hemisphere histograms from `allPaths`
   recovers the published `binCounts` at mean r = 0.9941 (min 0.9842) across
   the 50 fMRI scenes. Confirms the angular convention.