"""Tests for src.rdm."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.rdm import (
    DEFAULT_METADATA_COLUMNS,
    RdmBuildResult,
    build_rdm,
    build_rdm_from_dataframe,
    infer_feature_columns,
    standardise_features,
)
from src.stats import pairwise_distances_to_tril, rdm_pair_indices


# ── Standardisation ────────────────────────────────────────────────────────────

def test_standardise_zero_mean_unit_std(rng):
    X = rng.normal(size=(30, 5)) * np.array([1, 10, 100, 1000, 0.1])
    X_scaled, scaler = standardise_features(X)
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)
    # scaler exposes the applied means and scales
    assert scaler.mean_.shape == (5,)
    assert scaler.scale_.shape == (5,)


def test_standardise_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        standardise_features(np.arange(10.0))


def test_standardise_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 2"):
        standardise_features(np.array([[1.0, 2.0]]))


# ── build_rdm basics ──────────────────────────────────────────────────────────

def test_rdm_shape(synthetic_features_df):
    X = synthetic_features_df[
        ["BPD", "SC", "GTI", "NNE", "PFF", "MAC", "BCR", "SCE"]
    ].to_numpy()
    rdm = build_rdm(X)
    assert rdm.shape == (50, 50)


def test_rdm_symmetric(synthetic_features_df):
    X = synthetic_features_df[
        ["BPD", "SC", "GTI", "NNE", "PFF", "MAC", "BCR", "SCE"]
    ].to_numpy()
    rdm = build_rdm(X)
    assert np.allclose(rdm, rdm.T)


def test_rdm_zero_diagonal(synthetic_features_df):
    X = synthetic_features_df[
        ["BPD", "SC", "GTI", "NNE", "PFF", "MAC", "BCR", "SCE"]
    ].to_numpy()
    rdm = build_rdm(X)
    assert np.allclose(np.diag(rdm), 0.0)


def test_rdm_nonnegative(synthetic_features_df):
    X = synthetic_features_df[
        ["BPD", "SC", "GTI", "NNE", "PFF", "MAC", "BCR", "SCE"]
    ].to_numpy()
    rdm = build_rdm(X)
    # off-diagonal entries are distances -> non-negative
    off = rdm[~np.eye(rdm.shape[0], dtype=bool)]
    assert np.all(off >= 0.0)


def test_rdm_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        build_rdm(np.arange(10.0))


def test_rdm_rejects_too_few_stimuli():
    with pytest.raises(ValueError, match="at least 2"):
        build_rdm(np.array([[1.0, 2.0, 3.0]]))


def test_rdm_rejects_nan(rng):
    X = rng.normal(size=(10, 3))
    X[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        build_rdm(X)


# ── Standardisation invariance ────────────────────────────────────────────────

def test_standardisation_makes_scaling_irrelevant(rng):
    """
    Two feature matrices that differ only by per-column scaling must
    produce identical RDMs after standardisation.
    """
    X = rng.normal(size=(20, 4))
    scale = np.array([1.0, 100.0, 0.01, 5.0])
    X_scaled_up = X * scale
    rdm_a = build_rdm(X)
    rdm_b = build_rdm(X_scaled_up)
    assert np.allclose(rdm_a, rdm_b, atol=1e-10)


# ── Ordering consistency (Fix 3 regression) ───────────────────────────────────

def test_rdm_tril_matches_standardized_pairwise_distances(rng):
    """
    build_rdm(X)[i_idx, j_idx] must equal
    pairwise_distances_to_tril(StandardScaler().fit_transform(X)).

    Both compute the same quantity: standardized Euclidean distances in
    lower-triangle ordering. If either the standardisation or the ordering
    diverges, this test fails.
    """
    from sklearn.preprocessing import StandardScaler

    X = rng.normal(size=(30, 6))
    X_std = StandardScaler().fit_transform(X)

    rdm = build_rdm(X)
    n = rdm.shape[0]
    i_idx, j_idx = rdm_pair_indices(n)

    from_rdm = rdm[i_idx, j_idx]
    from_helper = pairwise_distances_to_tril(X_std)

    assert np.allclose(from_rdm, from_helper)


def test_rdm_tril_ordering_is_lower_triangle(rng):
    """
    Fix-3 regression: the tril vector from build_rdm must correspond to
    scene pairs (i, j) with i > j, in row-major order of the lower triangle.

    This checks the ordering convention directly, independent of any
    standardisation choice.
    """
    X = rng.normal(size=(6, 3))
    rdm = build_rdm(X)
    i_idx, j_idx = rdm_pair_indices(6)
    vec = rdm[i_idx, j_idx]

    # Manually compute the same vector without scipy/pdist
    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X)
    manual = np.array([
        np.linalg.norm(X_std[i] - X_std[j])
        for i in range(6) for j in range(i)
    ])

    # np.tril_indices returns pairs in exactly the order (1,0),(2,0),(2,1),...
    # so `manual` and `vec` must match element-by-element.
    assert vec.shape == manual.shape
    assert np.allclose(vec, manual)


def test_rdm_tril_is_not_upper_triangle(rng):
    """
    Explicitly check that we are not accidentally using upper-triangle
    ordering (the Fix-3 bug). For a symmetric RDM the two orderings contain
    the same multiset of values but in different positions.
    """
    X = rng.normal(size=(8, 4))
    rdm = build_rdm(X)

    i_tril, j_tril = np.tril_indices(8, k=-1)   # strict lower, 28 pairs
    i_triu, j_triu = np.triu_indices(8, k=1)    # strict upper, 28 pairs

    tril_vec = rdm[i_tril, j_tril]
    triu_vec = rdm[i_triu, j_triu]

    assert tril_vec.shape == triu_vec.shape == (28,)

    # Same multiset of values — RDM is symmetric, so this must hold
    assert np.allclose(np.sort(tril_vec), np.sort(triu_vec))

    # But different orderings — the two index schemes do not enumerate
    # pairs in the same sequence, so the unsorted vectors must differ
    assert not np.allclose(tril_vec, triu_vec)


def test_rdm_matches_sklearn_pairwise_distances(rng):
    """Cross-check against an independent implementation."""
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler

    X = rng.normal(size=(25, 4))
    X_std = StandardScaler().fit_transform(X)
    rdm = build_rdm(X)
    expected = pairwise_distances(X_std, metric="euclidean")
    assert np.allclose(rdm, expected)


# ── DataFrame-aware construction ──────────────────────────────────────────────

def test_infer_feature_columns(synthetic_features_df):
    cols = infer_feature_columns(synthetic_features_df)
    assert "image_name" not in cols
    assert "floor_fraction" not in cols
    assert "sam_failed" not in cols
    assert set(cols) == {"BPD", "SC", "GTI", "NNE", "PFF", "MAC", "BCR", "SCE"}


def test_rdm_from_dataframe_all_valid(synthetic_features_df):
    result = build_rdm_from_dataframe(synthetic_features_df)
    assert isinstance(result, RdmBuildResult)
    assert result.n_total == 50
    assert result.n_valid == 50
    assert result.valid_mask.all()
    assert result.rdm.shape == (50, 50)
    assert np.all(np.isfinite(result.rdm))


def test_rdm_from_dataframe_with_failures(synthetic_features_df_with_failures):
    df = synthetic_features_df_with_failures
    result = build_rdm_from_dataframe(df)
    assert result.n_total == 50
    assert result.n_valid == 47
    assert result.valid_mask.sum() == 47

    invalid_idx = np.where(~result.valid_mask)[0]
    assert set(invalid_idx.tolist()) == {7, 22, 41}

    # NaN in every position touching an invalid row
    for i in invalid_idx:
        assert np.all(np.isnan(result.rdm[i, :]))
        assert np.all(np.isnan(result.rdm[:, i]))

    # Valid submatrix is finite
    valid_idx = np.where(result.valid_mask)[0]
    sub = result.rdm[np.ix_(valid_idx, valid_idx)]
    assert np.all(np.isfinite(sub))


def test_rdm_from_dataframe_valid_submatrix_matches_manual(
    synthetic_features_df_with_failures,
):
    """The valid submatrix must equal the RDM built from only valid rows."""
    df = synthetic_features_df_with_failures
    result = build_rdm_from_dataframe(df)

    feature_cols = result.feature_names
    valid_rows = df.loc[result.valid_mask, feature_cols].to_numpy()
    manual = build_rdm(valid_rows)

    valid_idx = np.where(result.valid_mask)[0]
    sub = result.rdm[np.ix_(valid_idx, valid_idx)]
    assert np.allclose(sub, manual)


def test_rdm_from_dataframe_explicit_feature_names(synthetic_features_df):
    result = build_rdm_from_dataframe(
        synthetic_features_df,
        feature_names=["MAC", "PFF", "BCR", "SCE"],
    )
    assert result.feature_names == ["MAC", "PFF", "BCR", "SCE"]


def test_rdm_from_dataframe_rejects_all_metadata(synthetic_features_df):
    with pytest.raises(ValueError, match="no feature columns"):
        build_rdm_from_dataframe(
            synthetic_features_df,
            feature_names=[],
        )


def test_rdm_from_dataframe_rejects_too_many_failures(synthetic_features_df):
    df = synthetic_features_df.copy()
    feature_cols = infer_feature_columns(df)
    # NaN out 49 of 50 rows
    df.loc[df.index[:49], feature_cols] = np.nan
    with pytest.raises(ValueError, match="at least 2 valid rows"):
        build_rdm_from_dataframe(df)


# ── Metric pass-through ───────────────────────────────────────────────────────

def test_rdm_metric_pass_through(rng):
    X = rng.normal(size=(20, 3))
    euclid = build_rdm(X, metric="euclidean")
    cityblock = build_rdm(X, metric="cityblock")
    # Different metrics must give different RDMs
    assert not np.allclose(euclid, cityblock)


def test_rdm_cosine_matches_manual(rng):
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler

    X = rng.normal(size=(15, 4))
    X_std = StandardScaler().fit_transform(X)
    rdm = build_rdm(X, metric="cosine")
    expected = pairwise_distances(X_std, metric="cosine")
    assert np.allclose(rdm, expected)