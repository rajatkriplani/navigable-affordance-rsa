# Changelog

All notable changes are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- `src/` package with 5 CLI entry points:
  `affordance-build-masks`, `affordance-extract-features`,
  `affordance-build-rdm`, `affordance-rsa`, `affordance-ablation`.
- Loaders for the Bonner & Epstein (2017) dataset: `src/bonner.py`.
- Trajectory-derived features: `src/features.py`
  (`reconstruct_angular_histogram`, `reimplement_nam`).
- `data/affordances/allpaths_mapping.csv` — verified scene_k ↔ pathways####
  mapping (r = 1.0000 against `binary_sum(Trajs)`).
- `docs/pipeline.md` — parameter provenance and statistical design.
- `tests/` — 180+ unit tests, CPU-only, runnable without GPU or SAM.

### Changed
- All random state is local (`np.random.default_rng`). Global
  `np.random.seed` is never touched.
- Scene ordering is enforced by `build_rdm` and `ablation`: features CSV
  rows are reordered via the mapping CSV before RDM construction.
- SAM segmentation now defaults to device `auto`; supports cuda, mps, cpu.
- SAM mask cache with resume-by-default. Re-running `affordance-build-masks`
  skips already-processed images unless `--force` is passed.
- Per-image feature extraction seed is derived from the image filename
  (SHA-256 prefix), so GTI is deterministic across runs and machines.

### Fixed
- **RDM ordering mismatch (Fix 3, critical).** `scipy.spatial.distance.pdist`
  returns the upper triangle in row-major order; `np.tril_indices` returns
  the lower triangle. Original code compared these misaligned vectors,
  producing garbage correlations. Now uses `squareform` → tril extraction.
- **EVC not controlled (Fix 1).** SAM predicts EVC as strongly as OPA.
  The primary novelty test now partials out both NAM and EVC.
- **Parametric p-values (Fix 2).** 1,225 non-independent RDM pairs inflate
  t-statistics ~5×. Replaced with 10,000-iteration permutation tests.
- **MATLAB v7.3 incompatibility (Fix 6).** `scipy.io.loadmat` silently
  fails on HDF5. Now uses `h5py` for v7.3 files.
- **Global RNG contamination (Fix 7).** Module-level `np.random.seed(42)`
  reset global state on every call, making results order-dependent. Now
  uses local `Generator` instances.
- **O(n²) endpoint distance loop (Fix 8).** Replaced with vectorised
  `scipy.spatial.distance.cdist`.
- **SAM mask quality not validated (Fix 9).** Pathologically small floor
  masks produced all-zero feature vectors and corrupted the RDM. Now uses
  `MIN_FLOOR_FRACTION = 0.02` plus a fallback prompt grid; failed images
  are NaN'd and excluded.
- **Hardcoded SUCCESS verdicts (Fix 10/11).** Conclusions are now
  evidence-gated on the actual statistical results, not on a single
  p-value threshold.
- **`cdist` scalar conversion on NumPy ≥ 1.25.** Replaced with
  `np.linalg.norm` for single-pair distances.

### Data
- `data/README_DATA.md` — provenance of the Bonner & Epstein dataset
  (obtained by direct correspondence; not redistributed).
- `Trajs.mat`, `allPaths.mat`, `binCounts.mat`, and the four RDM `.mat`
  files are not committed (gitignored under `data/`).
- `allpaths_mapping.csv` is committed (3 KB, deterministic, unblocks the
  pipeline for new users).