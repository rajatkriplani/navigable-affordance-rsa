# navigable-affordance-rsa

An RSA pipeline testing whether the human occipital place area (OPA)
encodes the *topological structure* of navigable space, above and beyond
the 2017 navigational-affordance model of Bonner & Epstein and low-level
visual cortex (EVC).

## Headline result

After controlling for both the published NAM model **and** EVC, the SAM-derived RDM's unique predictive signal for OPA is carried by
metric clearance features (PFF, MAC, BCR) and by geometric extent
(NNE). The only feature in the set that is a genuine topological
invariant — the number of connected skeleton components (SC) — shows no
unique OPA variance. PPA shows a narrower profile, driven by geometric
extent alone. EVC shows no unique affordance signal once NAM is
partialled out. The topological-graph hypothesis is not supported.
This refines rather than refutes Bonner & Epstein (2017): the OPA's
sensitivity to navigable affordances is metric, not graph-theoretic.

Full numbers: `outputs/stats/rsa.json`, `outputs/stats/ablation_opa.json`
after running the pipeline.

## What this repo does

1. **Segment** the navigable floor in each of the 50 Bonner & Epstein
   Exp. 2 scene photographs using SAM (Segment Anything).
2. **Extract** an 8-dimensional Gibsonian affordance vector from each
   floor mask: 4 topological features (BPD, SC, GTI, NNE) and 4 clearance
   features (PFF, MAC, BCR, SCE).
3. **Build** a scene-ordered 50×50 RDM from the feature vectors.
4. **RSA** against OPA, PPA, and EVC, with permutation testing (10,000
   iterations) and FDR correction.
5. **Ablate** each of the 8 features to determine which one carries the
   unique OPA signal.

## Pipeline diagram

```text
data/stimuli/*.jpg
    │
    ▼  affordance-build-masks    (SAM, cacheable)
outputs/masks/*.png
    │
    ▼  affordance-extract-features
outputs/features/extracted_features.csv
    │
    ▼  affordance-build-rdm      (uses data/affordances/allpaths_mapping.csv
    │                              for scene-order alignment)
outputs/rdms/sam_topological_rdm.npy
    │
    ├──▼  affordance-rsa        (partial RSA vs OPA/PPA/EVC)
    │    outputs/stats/rsa.json
    │    outputs/figures/rsa_summary.png
    │
    └──▼  affordance-ablation   (per-feature partial RSA)
         outputs/stats/ablation_opa.json
         outputs/figures/ablation_opa.png
```

## Installation

Requires Python ≥ 3.10. Two paths:

### Full pipeline (includes SAM segmentation)

```bash
git clone <this-repo>
cd navigable-affordance-rsa
pip install -e ".[sam,dev]"
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download a SAM checkpoint (choose one)
# vit_b: 375 MB, fastest on CPU
wget -P checkpoints/ \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Analysis-only (assumes pre-computed masks are available)

If you have `outputs/masks/*.png` from a prior run, you don't need SAM:

```bash
pip install -e ".[dev]"
make analysis   # features → rdm → rsa → ablation
```

This runs in ~5 minutes on a laptop.

## Hardware

SAM segmentation is the only heavy step. Everything else runs on CPU.

**Recommendation:** `--sam-model vit_b` on CPU. For floor segmentation, mask
quality is nearly identical between checkpoints. Use `vit_h` for paper-parity
reproduction if you have a GPU.

RSA and ablation with 10,000 permutations each take ~2–3 minutes on CPU
regardless of hardware, since permutation testing is single-threaded numpy.

## Usage

```bash
# Full pipeline (assumes SAM installed and checkpoint downloaded)
make all DEVICE=cpu SAM_MODEL=vit_b

# Analysis-only (assumes masks already cached)
make analysis

# Individual stages
affordance-build-masks      --config configs/default.yaml --device cpu --sam-model vit_b
affordance-extract-features --config configs/default.yaml
affordance-build-rdm        --config configs/default.yaml
affordance-rsa              --config configs/default.yaml
affordance-ablation         --config configs/default.yaml

# Quick test with fewer permutations
affordance-rsa      --config configs/default.yaml --n-perm 1000
affordance-ablation --config configs/default.yaml --n-perm 1000
```

## Data layout

```text
data/
├── stimuli/           # 50 Bonner & Epstein Exp. 2 scene photos (not committed)
├── affordances/       # Bonner norming data (not committed)
│   ├── Trajs.mat       # per-rater trajectory heat maps, 173 scenes
│   ├── allPaths.mat    # aggregated maps for the 50 fMRI scenes
│   ├── binCounts.mat   # 180-bin angular histograms
│   └── allpaths_mapping.csv  # scene_k ↔ pathways#### (committed)
└── rdms/              # brain RDMs (not committed)
    ├── OPA.mat, PPA.mat, EVC.mat   # group-level ROI RDMs, 50×50
    ├── NAM.mat                     # Bonner 2017 affordance model RDM
    └── all_sub_slrdms.npy          # per-subject per-voxel RDMs (4.6 GB)

outputs/               # all regenerable, gitignored
├── masks/             # SAM floor masks + metadata
├── features/          # 8-dim affordance vectors
├── rdms/              # SAM-derived topological RDM + order sidecar
├── stats/             # JSON results
└── figures/           # PNG plots
```

See `data/README_DATA.md` for provenance of the Bonner dataset.

## Methodology notes

### Why EVC is controlled in the partial RSA

SAM predicts low-level visual texture in EVC as strongly as it predicts OPA.
Without controlling for EVC, the SAM→OPA correlation could be entirely
explained by visual confounds. The primary novelty test partials out both
NAM **and** EVC simultaneously.

### Why permutation testing

The 1,225 RDM pairs are not independent. A Spearman correlation across them
inflates the effective sample size and produces anti-conservative p-values
(typically ~5× too small). Every reported p-value comes from a 10,000-iteration
label-permutation test that respects the dependence structure.

### Why FDR correction

The RSA runs 10 tests; the ablation runs 8. Benjamini-Hochberg FDR at q < 0.05
is applied across each family of tests.

### Why scene ordering matters

The features CSV is sorted alphabetically by image name. The brain RDMs are
indexed by scene_k (the position of each stimulus in the original experiment).
These orderings differ. `data/affordances/allpaths_mapping.csv` resolves the
mapping. Without it, the RDM would compare the wrong pairs — silently. The
pipeline refuses to proceed without the mapping.

## Relationship to Bonner & Epstein (2017)

| Dimension | Bonner & Epstein | This repo |
|---|---|---|
| Representation | 3 angular channels (L/C/R) from human raters | 8 geometric features (4 topology + 4 clearance) |
| Source of signal | Subjective human path drawings | Objective geometry (SAM + skeleton + clearance) |
| Direction encoding | Yes | Implicit (via skeleton + clearance) |
| Metric width/depth | No | **Yes (clearance family)** |
| Topological graph | No | **Yes (topology family), but ablation says it doesn't matter** |
| Requires body scale | No | Yes (`r_base_px`) |
| Reproducible without raters | No | Yes |

Our finding: OPA's unique affordance signal above NAM + EVC is carried by
the clearance family. This is a refinement of the 2017 result, not a refutation.

## Parameter choices

See `docs/pipeline.md` §4 for provenance and sensitivity of `r_base_px` and
`horizon_ratio`.

## Citation

```bibtex
@software{navigable_affordance_rsa,
  author = {Rajat Kriplani},
  title  = {Navigable Affordance RSA},
  year   = {2026},
  url    = {https://github.com/rajatkriplani/navigable-affordance-rsa}
}
```

If you use the Bonner dataset, also cite Bonner & Epstein (2017):

> Bonner, M. F., & Epstein, R. A. (2017). Coding of navigational
> affordances in the human visual system. *PNAS*, 114(18), 4793–4798.

## License

MIT. See `LICENSE`.