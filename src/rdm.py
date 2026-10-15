"""
RDM construction for the navigable-affordance RSA pipeline.

Given an (N, F) feature matrix — one row per stimulus, one column per
feature — build an (N, N) representational dissimilarity matrix.

Design notes
------------
- Features are z-score standardised before computing distances. This is
  essential: the 8 affordance features have wildly different scales
  (SC counts components, BPD is a fraction, MAC is a ratio, etc.) and
  un-standardised Euclidean distance would be dominated by whichever
  feature happens to have the largest range.

- Distances are computed with scipy.spatial.distance.pdist, then reshaped
  via squareform. This produces a properly symmetric matrix with zero
  diagonal. Downstream code that extracts pairs via np.tril_indices will
  get the same values as pairwise_distances_to_tril() from src.stats —
  this is enforced by a regression test.

- NaN rows (e.g. from SAM segmentation failures) are excluded from RDM
  computation. The returned full RDM has NaN in every position touching
  an excluded row, and a boolean valid_mask is returned alongside so that
  downstream code can decide whether to drop or impute.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from .logging_utils import get_logger

logger = get_logger(__name__)


# Non-feature columns that commonly appear in the features CSV
DEFAULT_METADATA_COLUMNS = frozenset({
    "image_name",
    "floor_fraction",
    "sam_failed",
})


# ── Standardisation ────────────────────────────────────────────────────────────

def standardise_features(feature_matrix):
    """
    Z-score standardise each column of an (N, F) matrix.

    Parameters
    ----------
    feature_matrix : array-like, shape (N, F)

    Returns
    -------
    X_scaled : ndarray, shape (N, F)
    scaler : sklearn StandardScaler
        Fitted scaler, useful for reporting the per-feature means and
        standard deviations applied.
    """
    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2D, got shape {X.shape}"
        )
    if X.shape[0] < 2:
        raise ValueError(
            f"need at least 2 rows to standardise, got {X.shape[0]}"
        )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ── RDM construction ───────────────────────────────────────────────────────────

def build_rdm(feature_matrix, metric="euclidean"):
    """
    Build an (N, N) RDM from an (N, F) feature matrix.

    Features are z-score standardised before distances are computed.

    Parameters
    ----------
    feature_matrix : array-like, shape (N, F)
        Must be finite (no NaN, no inf). Use build_rdm_from_dataframe if
        you need NaN-row handling.
    metric : str
        Any metric accepted by scipy.spatial.distance.pdist.

    Returns
    -------
    rdm : ndarray, shape (N, N), float64
        Symmetric, zero diagonal.
    """
    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"need at least 2 stimuli, got {X.shape[0]}")
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "feature_matrix contains NaN or inf; use "
            "build_rdm_from_dataframe for NaN-aware construction"
        )

    X_scaled, _ = standardise_features(X)
    full = squareform(pdist(X_scaled, metric=metric))
    return full.astype(np.float64)


# ── DataFrame-aware construction ───────────────────────────────────────────────

@dataclass
class RdmBuildResult:
    rdm: np.ndarray           # (N, N) with NaN in positions touching invalid rows
    valid_mask: np.ndarray    # (N,) bool
    feature_names: list       # names used for distance computation
    n_valid: int
    n_total: int


def infer_feature_columns(df, metadata_columns=DEFAULT_METADATA_COLUMNS):
    """
    Return column names of df that look like feature columns (not metadata).
    """
    return [c for c in df.columns if c not in metadata_columns]


def build_rdm_from_dataframe(df, feature_names=None, metric="euclidean"):
    """
    Build an (N, N) RDM from a feature DataFrame, excluding NaN rows from
    the RDM computation and marking them as NaN in the returned matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per stimulus. Feature columns plus optional metadata
        (image_name, floor_fraction, sam_failed).
    feature_names : list of str or None
        Columns to use as features. If None, infer from all columns not
        in DEFAULT_METADATA_COLUMNS.
    metric : str
        Distance metric passed to scipy.spatial.distance.pdist.

    Returns
    -------
    RdmBuildResult
        .rdm : (N, N) full RDM, NaN at positions involving invalid rows
        .valid_mask : (N,) bool, True where a row was used
        .feature_names : list of the feature columns used
        .n_valid, .n_total : ints
    """
    if feature_names is None:
        feature_names = infer_feature_columns(df)
    if len(feature_names) == 0:
        raise ValueError(
            "no feature columns found in DataFrame; "
            f"all columns are metadata: {list(df.columns)}"
        )

    subset = df[feature_names]
    valid_mask = subset.notna().all(axis=1).to_numpy()
    n_valid = int(valid_mask.sum())
    n_total = len(df)

    if n_valid < 2:
        raise ValueError(
            f"need at least 2 valid rows to build RDM, got {n_valid} "
            f"out of {n_total}"
        )

    valid_features = subset.to_numpy()[valid_mask].astype(float)
    valid_rdm = build_rdm(valid_features, metric=metric)

    full_rdm = np.full((n_total, n_total), np.nan, dtype=np.float64)
    valid_idx = np.where(valid_mask)[0]
    full_rdm[np.ix_(valid_idx, valid_idx)] = valid_rdm

    logger.info(
        "Built %dx%d RDM from %d/%d valid rows (%d features, metric=%s)",
        n_total, n_total, n_valid, n_total, len(feature_names), metric,
    )

    return RdmBuildResult(
        rdm=full_rdm,
        valid_mask=valid_mask,
        feature_names=list(feature_names),
        n_valid=n_valid,
        n_total=n_total,
    )