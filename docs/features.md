# Affordance Features

This document specifies the 8-dimensional affordance representation used
throughout the pipeline: what each feature measures, where its underlying
concept comes from, how it is computed, and how the features are classified.
It is the reference for the paper's Methods §X and Discussion §Y.

## 1. Motivation

Gibsonian affordance theory holds that perception is organised around the
action possibilities an environment offers. Bonner & Epstein (2017) tested
this for navigation by showing that OPA encodes the *direction* of
navigable paths — where one can walk in a scene — above and beyond low-level
visual properties. Their model reduces each scene to three angular channels
(left / centre / right), derived from human trajectory drawings.

That model captures direction but not two other properties that matter for
navigation: the **metric width and depth** of traversable space, and the
**topological and geometric structure** of the navigable path network. The
8-feature representation here fills both gaps. Every feature is derived
objectively from a binary floor mask, so the representation requires no
human raters and is exactly reproducible.

## 2. Taxonomy

Features are classified along three orthogonal axes.

### 2.1 Invariance class (primary)

The mathematical property that determines what a feature measures and
what theoretical claims it can support.

| Class | Definition | Features |
|---|---|---|
| **Topological invariant** | Unchanged under any continuous deformation of the mask | SC |
| **Geometric** | Changes under deformation, but requires no body scale | BPD, GTI, NNE |
| **Metric** | Requires an explicit body-scale map $r_{\text{body}}(y)$ | PFF, MAC, BCR, SCE |

Only SC is a genuine topological invariant in the mathematical sense. The
"Topology" / "Clearance" labels used in `src/ablation.py` and in the JSON
output are a coarser, presentational grouping — see §2.2.

### 2.2 Domain (presentational)

Used for paper vocabulary and column headers. A derived grouping of the
invariance classes.

| Domain | Invariance classes it contains | Features |
|---|---|---|
| **Configural** | Topological invariant + Geometric | SC, BPD, GTI, NNE |
| **Clearance** | Metric | PFF, MAC, BCR, SCE |

### 2.3 Aspect

The specific property within the domain.

| Aspect | Question answered | Features |
|---|---|---|
| **Connectivity** | Is the space one piece, and where does it fork? | SC, BPD |
| **Extent** | How far does the navigable region stretch, and how directly? | GTI, NNE |
| **Magnitude** | How much room is there, typically? | MAC, PFF |
| **Constraint** | What is the worst-case tightness, and how heterogeneous? | BCR, SCE |

## 3. Feature specifications

Each feature is defined for a binary floor mask $F \in \{0,1\}^{H \times W}$.
Let $S$ be the skeleton of $F$ (via `skimage.morphology.medial_axis`), $D$
the distance transform of $F$ (also returned by `medial_axis`), and $r_{\text{body}}(y)$
the body-scale map:

$$
r_{\text{body}}(y) = \max\!\left(1,\ r_{\text{base}} \cdot \frac{y - y_{\text{horiz}}}{H - y_{\text{horiz}}}\right)
$$

with $y_{\text{horiz}} = \lfloor H \cdot h_{\text{ratio}} \rfloor$. The two
parameters $r_{\text{base}}$ and $h_{\text{ratio}}$ are documented in
`docs/pipeline.md §4`.

### 3.1 SC — Skeleton Components (Topological invariant)

**Primitive.** Count of 8-connected components in the skeleton.

**Origin.** Standard mathematical morphology; see Serra (1982), *Image
Analysis and Mathematical Morphology*, and Rosenfeld & Kak (1976),
*Digital Picture Processing*.

**Formula.** $\text{SC} = |\{\text{components of } S\}|$.

**Interpretation.** How many topologically distinct navigable regions does
this scene contain? $\text{SC} = 1$ means a single connected corridor;
$\text{SC} = 2$ means two separated areas.

### 3.2 BPD — Branch Point Density (Geometric)

**Primitive.** Density of skeleton pixels with three or more neighbours.

**Origin.** Skeleton graph branch-point analysis, standard in vascular
morphometry for retinal vessel analysis (Cheung et al., 2010,
*Ophthalmology*).

**Formula.** $\text{BPD} = \frac{|\{p \in S : |\text{neighbours}(p)| \geq 3\}|}{|S|}$.

**Interpretation.** How often does the navigable path network fork,
normalised by skeleton length?

### 3.3 GTI — Graph Tortuosity Index (Geometric)

**Primitive.** Mean ratio of graph geodesic length to Euclidean chord
length across random endpoint pairs.

**Origin.** Vascular tortuosity, formalised for retinal vessel analysis
(Grisan et al., 2008, *IEEE TMI*). The same measurement is standard in
pore network analysis and road network efficiency.

**Formula.** $\text{GTI} = \frac{1}{|\mathcal{P}|} \sum_{(u,v) \in \mathcal{P}} \frac{d_G(u,v)}{\|u - v\|}$
where $\mathcal{P}$ is a sample of endpoint pairs and $d_G$ is the graph
shortest-path distance.

**Interpretation.** On average, how much longer is the actual walkable
path than the straight-line distance? $\text{GTI} = 1$ means perfectly
straight; $\text{GTI} > 1$ means paths wind.

### 3.4 NNE — Nearest-Neighbour Endpoint Span (Geometric)

**Primitive.** Maximum pairwise distance between skeleton endpoints,
normalised by image diagonal.

**Origin.** Endpoint span is standard in skeleton pruning and shape
description (Bai et al., 2007, *Pattern Recognition*). The normalisation
by image diagonal is a scale-invariance device used in scene statistics
(Oliva & Torralba, 2001, *IJCV*).

**Formula.** $\text{NNE} = \frac{\max_{i,j} \|e_i - e_j\|}{\sqrt{H^2 + W^2}}$
where $\{e_i\}$ are skeleton endpoints.

**Interpretation.** How far does the navigable region extend, relative to
the image size? This is an "extent" measure: how deep into the scene does
the walkable floor reach?

### 3.5 MAC — Mean Axial Clearance (Metric)

**Primitive.** Mean distance-to-boundary along the skeleton, divided by
the body-scale map.

**Origin.** Distance transform (Rosenfeld & Pfaltz, 1966) combined with
body-scaled affordance theory (Warren, 1984; Warren & Whang, 1987),
which established that behavioural affordances are ratios of environmental
geometry to body metrics.

**Formula.** $\text{MAC} = \operatorname{mean}\left(\frac{D(s)}{r_{\text{body}}(y_s)}\right),\ s \in S$.

**Interpretation.** On average, how many body-widths wide is the navigable
space?

### 3.6 PFF — Perspective Floor Fraction (Metric)

**Primitive.** Fraction of skeleton points where normalised clearance
exceeds one body radius.

**Origin.** "Floor fraction" is common in scene statistics. The
perspective correction via $r_{\text{body}}(y)$ adapts ground-plane
estimation from visual navigation (Lee, 1980) and pinhole-camera
perspective geometry (Criminisi et al., 2000).

**Formula.** $\text{PFF} = \frac{|\{s \in S : D(s)/r_{\text{body}}(y_s) > 1\}|}{|S|}$.

**Interpretation.** What fraction of the navigable skeleton is wide
enough for a body to pass?

### 3.7 BCR — Body Clearance Ratio (Metric)

**Primitive.** Fifth percentile of the normalised clearance distribution.

**Origin.** Robust percentile statistics (Tukey, 1977, *Exploratory Data
Analysis*), applied as a bottleneck measure in the sense of motion
planning (Latombe, 1991; Choset et al., 2005).

**Formula.** $\text{BCR} = Q_{0.05}\left(\{D(s)/r_{\text{body}}(y_s) : s \in S\}\right)$.

**Interpretation.** What is the tightest body-scaled passage along the
navigable skeleton? BCR is a "worst-case" measure — it captures whether
there is any bottleneck in the scene.

### 3.8 SCE — Spatial Clearance Entropy (Metric)

**Primitive.** Shannon entropy of the normalised clearance histogram.

**Origin.** Shannon (1948) information theory, applied to spatial histogram
diversity. Analogous to isovist entropy in architectural analysis
(Benedikt, 1979) and vessel density entropy in retinal imaging.

**Formula.** $\text{SCE} = -\sum_i p_i \log p_i$, where $p_i$ is the
density of skeleton points in histogram bin $i$ of the normalised
clearance distribution (20 bins by default).

**Interpretation.** Is the clearance uniformly distributed (low SCE)
or heterogeneous (high SCE)?

## 4. Empirical results

The ablation (`affordance-ablation`, 10,000 permutations per feature,
controlling for NAM + EVC simultaneously) ran against three ROIs. Partial
Spearman correlations:

| Feature | Class | OPA | PPA | EVC |
|---|---|---|---|---|
| PFF | Metric | **+0.167** ✓ | −0.030 | −0.004 |
| MAC | Metric | **+0.134** ✓ | −0.021 | −0.001 |
| NNE | Geometric | **+0.109** ✓ | **+0.138** ✓ | +0.002 |
| BCR | Metric | **+0.100** ✓ | −0.001 | +0.001 |
| BPD | Geometric | +0.026 | +0.007 | −0.002 |
| GTI | Geometric | +0.021 | +0.008 | −0.004 |
| SC | Topological invariant | +0.010 | −0.003 | −0.006 |
| SCE | Metric | −0.013 | −0.009 | −0.006 |

✓ = FDR q < 0.05.

Three observations:

1. **No topological-invariant feature is significant at any ROI.** The
   only genuinely topological feature (SC) is null everywhere. The
   topological-graph hypothesis is not supported by these data.

2. **OPA and PPA show a double dissociation.** OPA is driven by metric
   clearance features (PFF, MAC, BCR) plus geometric extent (NNE). PPA
   is driven by geometric extent alone. The difference is metric clearance.

3. **EVC is a clean null across all 8 features.** Once NAM is partialled
   out, no affordance feature predicts EVC. This confirms that the
   standard-RSA EVC effect observed for the SAM RDM was entirely
   NAM-mediated, not a genuine affordance signal.

## 5. Sensitivity to parameter choice

The sweep in `affordance-sensitivity` re-runs the OPA ablation at 25 grid
points over `r_base_px ∈ {30, 45, 60, 80, 120}` and
`horizon_ratio ∈ {0.40, 0.45, 0.50, 0.55, 0.60}`. Results in
`outputs/stats/sensitivity_opa.json` and
`outputs/figures/sensitivity_opa.png`.

## 6. Design rationale

Why these 8 and not others:

- **Two axes of navigable structure.** The features cover both the *path
  network* (topology and geometry of the skeleton) and the *traversable
  floor* (metric clearance). This is the minimum set that captures both
  the graph-theoretic and metric aspects of navigation.

- **Every feature is objectively computable.** No human raters, no learned
  parameters, no dependence on the specific SAM checkpoint. The extraction
  is deterministic given the mask.

- **One feature per invariance × aspect cell where possible.** The
  resulting 2×4 grid (domain × aspect) is the smallest taxonomy that
  partitions the empirical behaviour cleanly — see §4.

- **No redundant measurements.** Each feature is invariant to the
  transformations the others are not: SC to continuous deformation, BPD
  to skeleton path length, GTI to overall path extent, NNE to internal
  branching, MAC to clearance heterogeneity, PFF to clearance magnitude,
  BCR to the clearance distribution shape, SCE to clearance magnitude.

## 7. References

- Bai, X., Latecki, L. J., & Liu, W. Y. (2007). Skeleton pruning by
  contour partitioning with discrete curve evolution. *IEEE TPAMI*, 29(3),
  449–462.
- Benedikt, M. L. (1979). To take hold of space: isovists and isovist
  fields. *Environment and Planning B*, 6(1), 47–65.
- Bonner, M. F., & Epstein, R. A. (2017). Coding of navigational
  affordances in the human visual system. *PNAS*, 114(18), 4793–4798.
- Cheung, C. Y., et al. (2010). Retinal vascular tortuosity, blood
  pressure, and cardiovascular risk factors. *Ophthalmology*, 117(8),
  1567–1574.
- Choset, H., et al. (2005). *Principles of Robot Motion*. MIT Press.
- Criminisi, A., Reid, I., & Zisserman, A. (2000). Single view metrology.
  *IJCV*, 40(2), 123–148.
- Grisan, E., Foracchia, M., & Ruggeri, A. (2008). A novel method for the
  automatic evaluation of retinal vessel tortuosity. *IEEE TBME*, 55(5),
  1404–1411.
- Latombe, J. C. (1991). *Robot Motion Planning*. Springer.
- Lee, D. N. (1980). The optic flow field: The foundation of vision.
  *Philosophical Transactions of the Royal Society B*, 290(1038), 169–179.
- Oliva, A., & Torralba, A. (2001). Modeling the shape of the scene: A
  holistic representation of the spatial envelope. *IJCV*, 42(3), 145–175.
- Rosenfeld, A., & Kak, A. C. (1976). *Digital Picture Processing*.
  Academic Press.
- Rosenfeld, A., & Pfaltz, J. L. (1966). Sequential operations in digital
  picture processing. *Journal of the ACM*, 13(4), 471–494.
- Serra, J. (1982). *Image Analysis and Mathematical Morphology*.
  Academic Press.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell
  System Technical Journal*, 27(3), 379–423.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Warren, W. H. (1984). Perceiving affordances: Visual guidance of stair
  climbing. *Journal of Experimental Psychology: HPP*, 10(5), 683–703.
- Warren, W. H., & Whang, S. (1987). Visual guidance of walking through
  apertures: Body-scaled information for affordances. *Journal of
  Experimental Psychology: HPP*, 13(3), 371–383.