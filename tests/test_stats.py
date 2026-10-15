"""Tests for src.stats."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.stats import (
    compute_noise_ceiling,
    distance_matrix_to_tril,
    exact_partial_spearman,
    fdr_correct,
    pairwise_distances_to_tril,
    permutation_test_partial,
    rdm_pair_indices,
)


# ── Partial Spearman ───────────────────────────────────────────────────────────

def test_partial_spearman_no_covariates_equals_spearman(rng):
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    r_partial = exact_partial_spearman(x, y)
    r_direct = float(spearmanr(x, y)[0])
    assert abs(r_partial - r_direct) < 1e-10


def test_partial_spearman_with_single_array_covariate(rng):
    """Regression test: single 1D numpy array must not crash."""
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    z = rng.normal(size=100)
    r = exact_partial_spearman(x, y, z)
    assert np.isfinite(r)


def test_partial_spearman_with_list_of_covariates(rng):
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    z1 = rng.normal(size=100)
    z2 = rng.normal(size=100)
    r = exact_partial_spearman(x, y, [z1, z2])
    assert np.isfinite(r)


def test_partial_spearman_removes_confound(rng):
    """Classic confound: x and z correlate, y is driven by z (not x).
    Partialling z should reduce |r(x, y)| toward zero."""
    n = 500
    z = rng.normal(size=n)
    x = 0.7 * z + 0.3 * rng.normal(size=n)   # x shares variance with z
    y = z + 0.1 * rng.normal(size=n)         # y driven by z, not by x

    r_raw = float(spearmanr(x, y)[0])
    r_partial = exact_partial_spearman(x, y, [z])

    # The apparent raw correlation should drop after controlling z
    assert abs(r_partial) < abs(r_raw)
    # And it should be close to zero: x has no genuine effect on y
    assert abs(r_partial) < 0.15


# ── Permutation test ───────────────────────────────────────────────────────────

def test_permutation_identity_rdm(synthetic_rdm):
    r, _, p = permutation_test_partial(
        synthetic_rdm, synthetic_rdm, n_perm=200,
        rng=np.random.default_rng(0),
    )
    assert abs(r - 1.0) < 1e-10
    assert p == 0.0


def test_permutation_reproducible(synthetic_rdm):
    r1, null1, p1 = permutation_test_partial(
        synthetic_rdm, synthetic_rdm, n_perm=50,
        rng=np.random.default_rng(123),
    )
    r2, null2, p2 = permutation_test_partial(
        synthetic_rdm, synthetic_rdm, n_perm=50,
        rng=np.random.default_rng(123),
    )
    assert np.array_equal(null1, null2)
    assert p1 == p2


def test_permutation_does_not_touch_global_state(synthetic_rdm):
    """Calling with a local rng must not affect np.random global."""
    np.random.seed(7)
    before = np.random.random()
    np.random.seed(7)
    _ = permutation_test_partial(
        synthetic_rdm, synthetic_rdm, n_perm=20,
        rng=np.random.default_rng(999),
    )
    after = np.random.random()
    assert before == after


# ── Noise ceiling ──────────────────────────────────────────────────────────────

def test_noise_ceiling_upper_geq_lower(synthetic_subject_rdms):
    lo, hi = compute_noise_ceiling(synthetic_subject_rdms)
    assert lo is not None and hi is not None
    assert hi >= lo


def test_noise_ceiling_empty_returns_none():
    lo, hi = compute_noise_ceiling(np.array([]))
    assert lo is None and hi is None


def test_noise_ceiling_single_subject_returns_none(synthetic_rdm):
    stack = synthetic_rdm[:, :, None]  # (50, 50, 1)
    lo, hi = compute_noise_ceiling(stack)
    assert lo is None and hi is None


# ── FDR ────────────────────────────────────────────────────────────────────────

def test_fdr_monotone_in_p():
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    q = fdr_correct(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_fdr_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.001, 0.01, 0.02, 0.5, 0.9])
    expected = multipletests(p, method="fdr_bh")[1]
    got = fdr_correct(p)
    assert np.allclose(got, expected)


# ── RDM alignment (Fix 3 regression test) ──────────────────────────────────────

def test_pdist_tril_alignment(synthetic_feature_matrix):
    """pdist returns upper-tri row-major; squareform+tril must realign."""
    from scipy.spatial.distance import pdist, squareform
    n = synthetic_feature_matrix.shape[0]
    i_idx, j_idx = rdm_pair_indices(n)

    pdist_vec = pdist(synthetic_feature_matrix)
    tril_vec = pairwise_distances_to_tril(synthetic_feature_matrix)

    assert pdist_vec.shape == tril_vec.shape
    # Same multiset of distances, but potentially different orderings
    assert np.allclose(np.sort(pdist_vec), np.sort(tril_vec))

    # Correct extraction: squareform then tril
    full = squareform(pdist_vec)
    manual = full[i_idx, j_idx]
    assert np.allclose(tril_vec, manual)


def test_distance_matrix_to_tril_matches_tril_indices(synthetic_rdm):
    i_idx, j_idx = rdm_pair_indices(50)
    vec = distance_matrix_to_tril(synthetic_rdm)
    assert np.allclose(vec, synthetic_rdm[i_idx, j_idx])