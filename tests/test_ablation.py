"""Tests for src.ablation."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from src.ablation import (
    CLEARANCE_FEATURES,
    TOPOLOGICAL_FEATURES,
    TOPOLOGICAL_INVARIANTS,
    GEOMETRIC_FEATURES,
    METRIC_FEATURES,
    AblationResult,
    derive_interpretation,
    permutation_test_single_feature,
    reorder_features_to_scene_order,
    run_ablation,
    single_feature_rdm,
)


# ── single_feature_rdm ────────────────────────────────────────────────────────

def test_single_feature_rdm_shape(rng):
    x = rng.normal(size=10)
    rdm = single_feature_rdm(x)
    assert rdm.shape == (10, 10)


def test_single_feature_rdm_symmetric_zero_diagonal(rng):
    x = rng.normal(size=10)
    rdm = single_feature_rdm(x)
    assert np.allclose(rdm, rdm.T)
    assert np.allclose(np.diag(rdm), 0.0)


def test_single_feature_rdm_zero_for_constant():
    x = np.ones(10)
    rdm = single_feature_rdm(x)
    assert np.allclose(rdm, 0.0)


def test_single_feature_rdm_scale_invariant(rng):
    """Scaling the input must not change the RDM."""
    x = rng.normal(size=10)
    a = single_feature_rdm(x)
    b = single_feature_rdm(x * 100.0 + 5.0)
    assert np.allclose(a, b, atol=1e-10)


def test_single_feature_rdm_rejects_too_few_values():
    with pytest.raises(ValueError, match="at least 2"):
        single_feature_rdm(np.array([1.0]))


def test_single_feature_rdm_rejects_nan(rng):
    x = rng.normal(size=10)
    x[3] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        single_feature_rdm(x)


def test_single_feature_rdm_matches_manual(rng):
    """Cross-check against a manual Euclidean computation."""
    x = rng.normal(size=8)
    rdm = single_feature_rdm(x)
    x_std = (x - x.mean()) / x.std()
    manual = np.abs(x_std[:, None] - x_std[None, :])
    assert np.allclose(rdm, manual)


# ── permutation_test_single_feature ───────────────────────────────────────────

def test_permutation_identity_rdm(rng):
    """A feature RDM identical to the brain RDM should yield r=1 and p=0."""
    x = rng.normal(size=20)
    rdm = single_feature_rdm(x)
    std_r, std_p, partial_r, partial_p = permutation_test_single_feature(
        rdm, rdm, covar_rdms=[], n_perm=100,
        rng=np.random.default_rng(0),
    )
    assert abs(std_r - 1.0) < 1e-10
    assert std_p == 0.0
    assert abs(partial_r - 1.0) < 1e-10


def test_permutation_reproducible(rng):
    x = rng.normal(size=20)
    f_rdm = single_feature_rdm(x)
    b_rdm = single_feature_rdm(rng.normal(size=20))
    c_rdm = single_feature_rdm(rng.normal(size=20))
    r1 = permutation_test_single_feature(
        f_rdm, b_rdm, [c_rdm], n_perm=50, rng=np.random.default_rng(7),
    )
    r2 = permutation_test_single_feature(
        f_rdm, b_rdm, [c_rdm], n_perm=50, rng=np.random.default_rng(7),
    )
    assert r1 == r2


# ── run_ablation ──────────────────────────────────────────────────────────────

def test_ablation_returns_all_features(synthetic_ablation_data):
    data = synthetic_ablation_data
    result = run_ablation(
        features_df=data["features_df"][data["feature_names"]],
        brain_rdm=data["brain_rdm"],
        covar_rdms=data["covar_rdms"],
        n_perm=100,
    )
    assert isinstance(result, AblationResult)
    assert len(result.features) == 8
    assert {f.feature for f in result.features} == set(data["feature_names"])


def test_ablation_feature_families():
    """Each feature must be tagged with its invariance class."""
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        "BPD": rng.normal(size=n), "SC": rng.normal(size=n),
        "GTI": rng.normal(size=n), "NNE": rng.normal(size=n),
        "PFF": rng.normal(size=n), "MAC": rng.normal(size=n),
        "BCR": rng.normal(size=n), "SCE": rng.normal(size=n),
    })
    b_rdm = squareform(pdist(rng.normal(size=(n, 3))))
    c_rdm = squareform(pdist(rng.normal(size=(n, 3))))
    result = run_ablation(df, b_rdm, [c_rdm], n_perm=20)
    families = {f.feature: f.family for f in result.features}
    domains  = {f.feature: f.domain for f in result.features}

    for f in TOPOLOGICAL_INVARIANTS:
        assert families[f] == "Topological invariant"
    for f in GEOMETRIC_FEATURES:
        assert families[f] == "Geometric"
    for f in METRIC_FEATURES:
        assert families[f] == "Metric"

    # Domain labels remain available for column headers
    for f in TOPOLOGICAL_FEATURES:
        assert domains[f] == "Topology"
    for f in CLEARANCE_FEATURES:
        assert domains[f] == "Clearance"


def test_ablation_fdr_q_values_are_finite(synthetic_ablation_data):
    data = synthetic_ablation_data
    result = run_ablation(
        data["features_df"][data["feature_names"]],
        data["brain_rdm"], data["covar_rdms"], n_perm=100,
    )
    for f in result.features:
        assert np.isfinite(f.partial_q)
        assert 0.0 <= f.partial_q <= 1.0


def test_ablation_clearance_dominant(synthetic_ablation_data):
    """
    With constructed data where clearance features carry the OPA signal
    and topology features are noise, the clearance features should
    dominate the partial-r ranking.
    """
    data = synthetic_ablation_data
    result = run_ablation(
        data["features_df"][data["feature_names"]],
        data["brain_rdm"], data["covar_rdms"],
        n_perm=500, master_seed=42,
    )
    top4 = {f.feature for f in result.features[:4]}
    clearance_in_top4 = top4 & CLEARANCE_FEATURES
    assert len(clearance_in_top4) >= 3


def test_ablation_reproducible(synthetic_ablation_data):
    data = synthetic_ablation_data
    r1 = run_ablation(
        data["features_df"][data["feature_names"]],
        data["brain_rdm"], data["covar_rdms"],
        n_perm=100, master_seed=42,
    )
    r2 = run_ablation(
        data["features_df"][data["feature_names"]],
        data["brain_rdm"], data["covar_rdms"],
        n_perm=100, master_seed=42,
    )
    for a, b in zip(r1.features, r2.features):
        assert a.feature == b.feature
        assert a.partial_r == b.partial_r
        assert a.partial_q == b.partial_q


def test_ablation_handles_nan_feature(rng):
    """A feature with NaN should be skipped, not crash the run."""
    n = 20
    df = pd.DataFrame({
        "BPD": rng.normal(size=n),
        "SC": rng.normal(size=n),
        "MAC": rng.normal(size=n),
    })
    df.loc[3, "BPD"] = np.nan
    b_rdm = squareform(pdist(rng.normal(size=(n, 3))))
    c_rdm = squareform(pdist(rng.normal(size=(n, 3))))
    result = run_ablation(df, b_rdm, [c_rdm], n_perm=20)
    assert len(result.features) == 2
    assert "BPD" not in {f.feature for f in result.features}


# ── Interpretation ────────────────────────────────────────────────────────────

def test_interpretation_no_significant():
    from src.ablation import AblationFeatureResult
    result = AblationResult(features=[
        AblationFeatureResult("BPD", "Topology", 0.01, 0.5, 0.01, 0.5, 0.9),
        AblationFeatureResult("MAC", "Clearance", 0.02, 0.4, 0.02, 0.4, 0.9),
    ])
    interp = derive_interpretation(result)
    assert interp["verdict"] == "no_significant_features"


def test_interpretation_clearance_only():
    from src.ablation import AblationFeatureResult
    result = AblationResult(features=[
        AblationFeatureResult("BPD", "Topology", 0.02, 0.4, 0.02, 0.4, 0.5),
        AblationFeatureResult("MAC", "Clearance", 0.20, 0.001, 0.20, 0.001, 0.01),
        AblationFeatureResult("PFF", "Clearance", 0.15, 0.005, 0.15, 0.005, 0.02),
    ])
    interp = derive_interpretation(result)
    assert interp["verdict"] == "clearance_only"
    assert set(interp["significant_clearance"]) == {"MAC", "PFF"}


def test_interpretation_topology_only():
    from src.ablation import AblationFeatureResult
    result = AblationResult(features=[
        AblationFeatureResult("BPD", "Topology", 0.20, 0.001, 0.20, 0.001, 0.01),
        AblationFeatureResult("MAC", "Clearance", 0.02, 0.4, 0.02, 0.4, 0.5),
    ])
    interp = derive_interpretation(result)
    assert interp["verdict"] == "topology_only"


def test_interpretation_mixed():
    from src.ablation import AblationFeatureResult
    result = AblationResult(features=[
        AblationFeatureResult("BPD", "Topology", 0.15, 0.005, 0.15, 0.005, 0.02),
        AblationFeatureResult("MAC", "Clearance", 0.20, 0.001, 0.20, 0.001, 0.01),
    ])
    interp = derive_interpretation(result)
    assert interp["verdict"] == "mixed"


# ── Reordering helper ─────────────────────────────────────────────────────────

def test_reorder_to_scene_order():
    df = pd.DataFrame({
        "image_name": ["pathways0010", "pathways0011", "pathways0014"],
        "MAC": [1.0, 2.0, 3.0],
    })
    mapping = pd.DataFrame({
        "scene_k": [0, 1, 2],
        "trajs_field": ["pathways0014", "pathways0010", "pathways0011"],
    })
    reordered, missing = reorder_features_to_scene_order(df, mapping)
    assert missing == []
    assert reordered.index.tolist() == [
        "pathways0014", "pathways0010", "pathways0011",
    ]
    assert reordered["MAC"].tolist() == [3.0, 1.0, 2.0]


def test_reorder_missing_scene():
    df = pd.DataFrame({
        "image_name": ["pathways0010"],
        "MAC": [1.0],
    })
    mapping = pd.DataFrame({
        "scene_k": [0, 1],
        "trajs_field": ["pathways0010", "pathways9999"],
    })
    _, missing = reorder_features_to_scene_order(df, mapping)
    assert missing == [1]