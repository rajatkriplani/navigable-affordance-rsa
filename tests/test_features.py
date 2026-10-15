"""Tests for src.features."""

import numpy as np
import pytest

from src.features import (
    aggregate_trajs_binary_sum,
    build_angular_histograms,
    reconstruct_angular_histogram,
    reimplement_nam,
)


# ── aggregate_trajs_binary_sum ─────────────────────────────────────────────────

def test_aggregate_binary_sum(rng):
    """Sum across raters after threshold at > 0."""
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[1, 1, 0] = 1.0
    arr[1, 1, 1] = 2.0      # still just one rater's contribution
    arr[2, 2, 2] = 1.0
    out = aggregate_trajs_binary_sum(arr)
    assert out.shape == (4, 4)
    assert out[1, 1] == 2.0   # two raters passed through (0 and 1)
    assert out[2, 2] == 1.0
    assert out.sum() == 3.0


def test_aggregate_binary_sum_on_empty(rng):
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    out = aggregate_trajs_binary_sum(arr)
    assert out.sum() == 0.0


# ── reconstruct_angular_histogram ─────────────────────────────────────────────

def test_angular_reconstruction_empty_map():
    hist = reconstruct_angular_histogram(np.zeros((16, 16)))
    assert hist.shape == (180,)
    assert hist.sum() == 0.0


def test_angular_reconstruction_bottom_center_is_90():
    """Pixel at bottom-centre should bin to 90 deg (forward)."""
    m = np.zeros((64, 64), dtype=np.float32)
    m[63, 32] = 1.0
    hist = reconstruct_angular_histogram(m)
    assert hist[90] == 1.0
    assert hist.sum() == 1.0


def test_angular_reconstruction_bottom_left_is_0():
    m = np.zeros((64, 64), dtype=np.float32)
    m[63, 0] = 1.0
    hist = reconstruct_angular_histogram(m)
    assert hist[0] == 1.0


def test_angular_reconstruction_bottom_right_is_179():
    m = np.zeros((64, 64), dtype=np.float32)
    m[63, 63] = 1.0
    hist = reconstruct_angular_histogram(m)
    assert hist[179] == 1.0


def test_angular_reconstruction_directly_above_is_90():
    m = np.zeros((64, 64), dtype=np.float32)
    m[0, 32] = 1.0
    hist = reconstruct_angular_histogram(m)
    assert hist[90] == 1.0


def test_angular_reconstruction_diagonal_is_about_45():
    """Pixel at bottom-left, one step up: angle should be near 45 deg."""
    m = np.zeros((64, 64), dtype=np.float32)
    # dx = 12 - 32 = -20, dy = 63 - 43 = 20 -> atan2(-20, 20) = -45 deg -> +90 = 45
    m[43, 12] = 1.0
    hist = reconstruct_angular_histogram(m)
    assert hist[45] == 1.0


def test_angular_reconstruction_weighted_pixels():
    """Multiple pixels with different values accumulate correctly."""
    m = np.zeros((64, 64), dtype=np.float32)
    m[63, 32] = 3.0
    m[0, 32] = 2.0
    hist = reconstruct_angular_histogram(m)
    assert hist[90] == 5.0
    assert hist.sum() == 5.0


def test_angular_reconstruction_custom_n_bins():
    m = np.zeros((64, 64), dtype=np.float32)
    m[63, 32] = 1.0
    hist = reconstruct_angular_histogram(m, n_bins=90)
    assert hist.shape == (90,)
    # 90 deg forward maps to bin 45 in the 90-bin scheme
    assert hist[45] == 1.0


# ── build_angular_histograms ──────────────────────────────────────────────────

def test_build_angular_histograms_shape(rng):
    maps = rng.random((32, 32, 7)).astype(np.float32)
    hists = build_angular_histograms(maps, n_bins=180)
    assert hists.shape == (7, 180)


def test_build_angular_histograms_empty_slices(rng):
    maps = rng.random((32, 32, 5)).astype(np.float32)
    maps[:, :, 2] = 0.0
    hists = build_angular_histograms(maps)
    assert hists[2].sum() == 0.0


# ── reimplement_nam ────────────────────────────────────────────────────────────

def test_nam_shapes():
    counts = np.zeros((10, 180), dtype=np.float32)
    features, rdm = reimplement_nam(counts)
    assert features.shape == (10, 3)
    assert rdm.shape == (10, 10)


def test_nam_rdm_symmetric_zero_diagonal():
    rng = np.random.default_rng(0)
    counts = rng.random((10, 180)).astype(np.float32)
    _, rdm = reimplement_nam(counts)
    assert np.allclose(rdm, rdm.T)
    assert np.allclose(np.diag(rdm), 0.0)


def test_nam_left_biased_histogram_peaks_left_channel():
    """A histogram concentrated near 45 deg should drive the L channel most."""
    counts = np.zeros((1, 180), dtype=np.float32)
    counts[0, 40:50] = 1.0     # around 45 deg
    features, _ = reimplement_nam(counts)
    L, C, R = features[0]
    assert L > C
    assert L > R


def test_nam_centre_biased_histogram_peaks_centre_channel():
    counts = np.zeros((1, 180), dtype=np.float32)
    counts[0, 85:95] = 1.0     # around 90 deg
    features, _ = reimplement_nam(counts)
    L, C, R = features[0]
    assert C > L
    assert C > R


def test_nam_right_biased_histogram_peaks_right_channel():
    counts = np.zeros((1, 180), dtype=np.float32)
    counts[0, 130:140] = 1.0   # around 135 deg
    features, _ = reimplement_nam(counts)
    L, C, R = features[0]
    assert R > L
    assert R > C


def test_nam_rejects_wrong_bin_count():
    counts = np.zeros((5, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="bin_counts"):
        reimplement_nam(counts, n_bins=180)


# ── Integration: algebraic identity (allPaths = binary_sum of Trajs) ──────────

def test_binary_sum_recovers_aggregate(rng):
    """The verified mapping implies allPaths == binary_sum(Trajs)."""
    H, W, R = 16, 16, 3
    per_rater = rng.integers(0, 2, size=(H, W, R)).astype(np.float32)
    aggregate = aggregate_trajs_binary_sum(per_rater)
    manual = (per_rater > 0).sum(axis=2).astype(np.float32)
    assert np.array_equal(aggregate, manual)