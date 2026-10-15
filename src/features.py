"""
Feature construction for the navigable-affordance RSA pipeline.

Two families of construction live here:

1. Trajectory-derived features from the Bonner & Epstein (2017) data:
   - reconstruct_angular_histogram: forward-hemisphere histogram, verified
     against the published binCounts at r = 0.994 (50 fMRI scenes).
   - reimplement_nam: L/C/R channel model, using Gaussian tuning curves.
     Approximation — exact reproduction requires the paper's SI parameters.

2. (Placeholder for the SAM-derived affordance features, which land in
   Commit 2 as src/affordance.py.)
"""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .logging_utils import get_logger

logger = get_logger(__name__)


# ── Trajectory aggregation ─────────────────────────────────────────────────────

def aggregate_trajs_binary_sum(trajs_array):
    """
    Sum a (H, W, n_raters) per-rater map into a single (H, W) map.

    Thresholds at > 0 before summing. This matches the algebraically exact
    identity that recovers allPaths.mat from Trajs.mat (r = 1.0000).
    """
    return (trajs_array > 0).sum(axis=2).astype(np.float32)


# ── Angular histogram reconstruction ──────────────────────────────────────────

def reconstruct_angular_histogram(map_2d, n_bins=180):
    """
    Reconstruct a forward-hemisphere angular histogram from a 2D trajectory map.

    Convention (Bonner & Epstein, 2017, Fig. 3B):
        0 deg   = path along bottom-left
        90 deg  = path forward, into the scene
        180 deg = path along bottom-right
    Bin width = 180 / n_bins degrees (1 degree for n_bins=180).

    Parameters
    ----------
    map_2d : 2D array, non-negative trajectory density.
    n_bins : int, default 180.

    Returns
    -------
    ndarray, shape (n_bins,), float32
    """
    map_2d = np.asarray(map_2d)
    if map_2d.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {map_2d.shape}")

    H, W = map_2d.shape
    cx = W / 2.0
    cy = H - 1.0

    ys, xs = np.nonzero(map_2d)
    hist = np.zeros(n_bins, dtype=np.float32)
    if ys.size == 0:
        return hist

    vals = map_2d[ys, xs].astype(np.float32)
    dx = xs.astype(np.float64) - cx
    dy = cy - ys.astype(np.float64)          # positive = upward / forward
    angles_deg = np.degrees(np.arctan2(dx, dy)) + 90.0
    # Clip to [0, 180) to avoid a rightward bin boundary wrap
    angles_deg = np.clip(angles_deg, 0.0, 180.0 - 1e-6)

    bin_width = 180.0 / n_bins
    bin_idx = (angles_deg / bin_width).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    np.add.at(hist, bin_idx, vals)
    return hist


def build_angular_histograms(maps_3d, n_bins=180):
    """
    Reconstruct angular histograms for a stack of 2D maps.

    Parameters
    ----------
    maps_3d : ndarray, shape (H, W, N)
    n_bins : int

    Returns
    -------
    ndarray, shape (N, n_bins), float32
    """
    maps_3d = np.asarray(maps_3d)
    if maps_3d.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {maps_3d.shape}")

    n = maps_3d.shape[2]
    out = np.zeros((n, n_bins), dtype=np.float32)
    for k in range(n):
        out[k] = reconstruct_angular_histogram(maps_3d[:, :, k], n_bins=n_bins)
    return out


# ── NAM reimplementation (approximate) ────────────────────────────────────────

def _gaussian_tuning(n_bins, center_deg, sigma_deg):
    """Gaussian tuning curve over the 0-180 angular range."""
    edges = (np.arange(n_bins) + 0.5) * (180.0 / n_bins)
    return np.exp(-0.5 * ((edges - center_deg) / sigma_deg) ** 2)


def reimplement_nam(
    bin_counts,
    n_bins=180,
    centers_deg=(45.0, 90.0, 135.0),
    sigma_deg=30.0,
):
    """
    Reimplement the Bonner & Epstein (2017) left/centre/right channel model.

    Each scene is represented by the response of three Gaussian tuning curves
    to its angular histogram. The scene-by-scene representational dissimilarity
    is then the Euclidean distance between 3-vectors.

    NOTE: The exact published NAM RDM uses the paper's SI tuning parameters,
    which are not publicly documented in the main text. This is an
    approximation. It is used for sanity-checking the pipeline, not as a
    replacement for the published NAM.mat RDM.

    Parameters
    ----------
    bin_counts : ndarray, shape (N, n_bins)
    n_bins : int
    centers_deg : tuple of float
        Tuning curve centers. Default (45, 90, 135) for L, C, R.
    sigma_deg : float
        Shared standard deviation of the tuning curves.

    Returns
    -------
    features : ndarray, shape (N, 3), float32
    rdm : ndarray, shape (N, N), float32
    """
    bin_counts = np.asarray(bin_counts, dtype=np.float32)
    if bin_counts.ndim != 2 or bin_counts.shape[1] != n_bins:
        raise ValueError(
            f"bin_counts must be (N, {n_bins}); got {bin_counts.shape}"
        )

    T = np.stack(
        [_gaussian_tuning(n_bins, c, sigma_deg) for c in centers_deg],
        axis=0,
    )  # (3, n_bins)
    features = bin_counts @ T.T  # (N, 3)
    rdm = squareform(pdist(features)).astype(np.float32)
    return features, rdm