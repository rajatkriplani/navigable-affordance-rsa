"""
Feature ablation: per-feature partial RSA against OPA controlling for NAM + EVC.

The scientific question
-----------------------
After establishing that the SAM-derived RDM predicts OPA above NAM + EVC
The question: which of the 8 individual features carries that signal?

Each feature k is:
  1. Extracted as a single-column vector
  2. Standardised
  3. Converted to an (N, N) single-feature RDM via Euclidean distance
  4. Tested with a partial Spearman correlation against OPA, controlling
     for NAM and EVC
  5. Permutation-tested (label-permutation of the feature RDM)
  6. FDR-corrected across the 8 features

Design notes
------------
- Features are z-scored before RDM construction, same as in src.rdm, so
  that the distance scale is comparable across features.

- The pairwise_distances_to_tril helper from src.stats is used for the
  lower-triangle extraction. This enforces the Fix-3 ordering: brain RDMs
  are extracted with the same np.tril_indices convention.

- Interpretation is evidence-gated (Fix 10/11). The output includes an
  explicit classification of which family (topological, clearance, or
  mixed) carries the significant signal.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata

from .logging_utils import get_logger
from .stats import (
    exact_partial_spearman,
    fdr_correct,
    pairwise_distances_to_tril,
    rdm_pair_indices,
)

logger = get_logger(__name__)


# Domain labels — used by the CLI for column headers and plot legends.
# Note: "Topology" here is a legacy label; the mathematically correct
# invariance class of these features is tracked separately (see below).
TOPOLOGICAL_FEATURES = frozenset({"GTI", "BPD", "NNE", "SC"})
CLEARANCE_FEATURES   = frozenset({"MAC", "PFF", "BCR", "SCE"})

# Invariance classes (mathematical). Only SC is a genuine topological
# invariant under continuous deformation; the others are geometric or
# metric. This is the taxonomy used in docs/features.md and in the
# paper's discussion.
TOPOLOGICAL_INVARIANTS = frozenset({"SC"})
GEOMETRIC_FEATURES     = frozenset({"BPD", "GTI", "NNE"})
METRIC_FEATURES        = frozenset({"PFF", "MAC", "BCR", "SCE"})


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class AblationFeatureResult:
    feature: str
    family: str
    domain: str
    std_r: float
    std_p: float
    partial_r: float
    partial_p: float
    partial_q: float = float("nan")

    @property
    def significant(self):
        return np.isfinite(self.partial_q) and self.partial_q < 0.05

    def as_dict(self):
        return {
            "feature": self.feature,
            "family": self.family,
            "domain": self.domain,
            "std_r": float(self.std_r),
            "std_p": float(self.std_p),
            "partial_r": float(self.partial_r),
            "partial_p": float(self.partial_p),
            "partial_q": float(self.partial_q),
            "significant_q05": bool(self.significant),
        }


@dataclass
class AblationResult:
    features: list = field(default_factory=list)
    interpretation: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "features": [f.as_dict() for f in self.features],
            "interpretation": self.interpretation,
        }

    def sort_by_partial_r(self):
        self.features = sorted(self.features,
                               key=lambda f: f.partial_r, reverse=True)


# ── Single-feature RDM ────────────────────────────────────────────────────────

def single_feature_rdm(feature_values):
    """
    Build an (N, N) single-feature RDM from a 1D vector.

    The vector is z-scored (so that a single feature's absolute scale
    does not affect the distances), then converted to Euclidean distances
    in lower-triangle ordering.
    """
    x = np.asarray(feature_values, dtype=float).ravel()
    if x.size < 2:
        raise ValueError(f"need at least 2 values, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("feature_values contains NaN or inf")
    std = x.std()
    if std < 1e-12:
        # Constant feature. Distances are all zero; the RDM is all zeros.
        # Return an all-zero matrix so the correlation is 0 or NaN without
        # crashing. Caller should flag this feature.
        n = x.size
        return np.zeros((n, n), dtype=np.float64)
    x_scaled = (x - x.mean()) / std
    # Convert to a 2D column and use the shared helper for lower-tri extraction
    from scipy.spatial.distance import pdist, squareform
    full = squareform(pdist(x_scaled[:, None], metric="euclidean"))
    return full.astype(np.float64)


# ── Permutation test for a single feature ─────────────────────────────────────

def _permute_feature_rdm(feature_rdm, rng):
    """Relabel rows/cols of a symmetric RDM."""
    n = feature_rdm.shape[0]
    perm = rng.permutation(n)
    return feature_rdm[np.ix_(perm, perm)]


def permutation_test_single_feature(
    feature_rdm,
    brain_rdm,
    covar_rdms,
    n_perm=10_000,
    rng=None,
):
    """
    Two permutation tests for one feature:

    1. Standard Spearman r(feature RDM, brain RDM). No controls.
    2. Partial Spearman r(feature RDM, brain RDM | covariates).

    Both use the same label-permutation scheme: the feature RDM's labels
    are permuted, brain and covariates stay fixed.

    Returns
    -------
    (std_r, std_p, partial_r, partial_p)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = feature_rdm.shape[0]
    i_idx, j_idx = rdm_pair_indices(n)

    b_vec = brain_rdm[i_idx, j_idx]
    f_vec = feature_rdm[i_idx, j_idx]
    c_vecs = [c[i_idx, j_idx] for c in covar_rdms]

    std_r = float(pearsonr(rankdata(f_vec), rankdata(b_vec))[0])
    partial_r = exact_partial_spearman(f_vec, b_vec, c_vecs)

    null_std = np.empty(n_perm)
    null_partial = np.empty(n_perm)
    for p in range(n_perm):
        f_perm = _permute_feature_rdm(feature_rdm, rng)[i_idx, j_idx]
        null_std[p] = float(pearsonr(rankdata(f_perm), rankdata(b_vec))[0])
        null_partial[p] = exact_partial_spearman(f_perm, b_vec, c_vecs)

    std_p = float((null_std >= std_r).mean())
    partial_p = float((null_partial >= partial_r).mean())
    return std_r, std_p, partial_r, partial_p


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_ablation(
    features_df,
    brain_rdm,
    covar_rdms,
    feature_names=None,
    n_perm=10_000,
    master_seed=42,
    fdr_method="fdr_bh",
):
    """
    Run per-feature partial RSA.

    Parameters
    ----------
    features_df : pandas.DataFrame, scene-ordered
        One row per stimulus, one column per feature. Must be in the same
        scene order as brain_rdm and covar_rdms.
    brain_rdm : 2D array
        The target brain RDM (typically OPA).
    covar_rdms : list of 2D arrays
        Covariate RDMs (e.g. [NAM, EVC]).
    feature_names : list of str or None
        Defaults to features_df.columns.
    n_perm : int
    master_seed : int
    fdr_method : str

    Returns
    -------
    AblationResult
    """
    if feature_names is None:
        feature_names = list(features_df.columns)

    n_features = len(feature_names)
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(n_features)

    results = []
    for feature, seed in zip(feature_names, child_seeds):
        values = features_df[feature].to_numpy()
        if not np.all(np.isfinite(values)):
            logger.warning(
                "Feature %s contains NaN/inf; skipping.", feature,
            )
            continue

        feature_rdm = single_feature_rdm(values)
        std_r, std_p, partial_r, partial_p = permutation_test_single_feature(
            feature_rdm, brain_rdm, covar_rdms,
            n_perm=n_perm,
            rng=np.random.default_rng(seed),
        )

        domain = (
           "Topology" if feature in TOPOLOGICAL_FEATURES else
           "Clearance" if feature in CLEARANCE_FEATURES else
           "Unknown"
        )
        family = (
           "Topological invariant" if feature in TOPOLOGICAL_INVARIANTS else
           "Geometric" if feature in GEOMETRIC_FEATURES else
           "Metric" if feature in METRIC_FEATURES else
           "Unknown"
        )

        results.append(AblationFeatureResult(
            feature=feature,
            family=family,
            domain=domain,
            std_r=std_r, std_p=std_p,
            partial_r=partial_r, partial_p=partial_p,
        ))

    # FDR across features
    if results:
        q_values = fdr_correct([r.partial_p for r in results],
                               method=fdr_method)
        for r, q in zip(results, q_values):
            r.partial_q = float(q)

    result = AblationResult(features=results)
    result.interpretation = derive_interpretation(result)
    result.sort_by_partial_r()
    return result


# ── Interpretation (Fix 10/11) ─────────────────────────────────────────────────

def derive_interpretation(result):
    """
    Evidence-gated interpretation of the ablation.

    Classification is by mathematical invariance class (topological /
    geometric / metric), not by the domain labels (configural / clearance).
    Domain labels are still reported for compatibility with the CLI's
    column headers.
    """
    sig_features = [f.feature for f in result.features if f.significant]
    sig_inv = [f for f in sig_features if f in TOPOLOGICAL_INVARIANTS]
    sig_geo = [f for f in sig_features if f in GEOMETRIC_FEATURES]
    sig_met = [f for f in sig_features if f in METRIC_FEATURES]

    # Keep legacy keys for downstream compatibility
    sig_topo_dom = [f for f in sig_features if f in TOPOLOGICAL_FEATURES]
    sig_clr_dom  = [f for f in sig_features if f in CLEARANCE_FEATURES]

    if not sig_features:
        verdict = "no_significant_features"
        description = (
            "No feature survives FDR correction at any invariance class. "
            "The model RDM's predictive power emerges from redundancy "
            "across features, not from any single geometrically "
            "interpretable dimension."
        )
    elif not sig_inv and (sig_geo or sig_met):
        verdict = "no_topological_invariant"
        description = (
            "No feature in the topological-invariance class is significant. "
            f"The unique signal is carried by geometric ({sig_geo}) "
            f"and/or metric ({sig_met}) features. The topological-graph "
            "hypothesis is not supported."
        )
    elif sig_inv and not (sig_geo or sig_met):
        verdict = "topological_only"
        description = (
            "Only topological-invariant features are significant. This is "
            "unusual and warrants scrutiny — the geometric and metric "
            "measures are null while the topological invariant is not."
        )
    elif sig_met and not sig_geo and not sig_inv:
        verdict = "metric_only"
        description = (
            "Only metric features are significant. This is the purest "
            "form of the clearance-dominance result: the model's signal "
            "reduces entirely to body-scaled width/depth."
        )
    else:
        verdict = "mixed"
        description = (
            f"Multiple invariance classes contribute: "
            f"topological_invariant={sig_inv}, "
            f"geometric={sig_geo}, metric={sig_met}. The result does not "
            f"reduce to a single class."
        )

    return {
        "verdict": verdict,
        "significant_features": sig_features,
        # Legacy domain-based keys (kept for CLI/plot compatibility)
        "significant_topological": sig_topo_dom,
        "significant_clearance": sig_clr_dom,
        # Invariance-class keys (primary)
        "significant_invariant": sig_inv,
        "significant_geometric": sig_geo,
        "significant_metric": sig_met,
        "description": description,
    }


# ── Scene ordering helper (shared with build_rdm CLI) ─────────────────────────

def reorder_features_to_scene_order(features_df, mapping_df):
    """
    Reorder features so that row k corresponds to scene_k in the brain RDMs.

    Mirrors the logic in src.cli.build_rdm. Kept in src.ablation so the
    ablation CLI can reorder without importing the build_rdm CLI.

    Returns (reordered_df with image_name as index, list of missing scene_k).
    """
    scene_to_image = dict(zip(
        mapping_df["scene_k"].astype(int),
        mapping_df["trajs_field"].astype(str),
    ))
    indexed = features_df.set_index("image_name")
    ordered_names = []
    missing = []
    for k in sorted(scene_to_image):
        name = scene_to_image[k]
        if name in indexed.index:
            ordered_names.append(name)
        else:
            missing.append(k)
    return indexed.loc[ordered_names], missing