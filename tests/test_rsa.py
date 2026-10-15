"""Tests for src.rsa."""

import numpy as np
import pytest

from src.rsa import (
    PARTIAL_TESTS,
    STANDARD_TESTS,
    RsaResult,
    RsaTestResult,
    apply_fdr,
    compute_all_noise_ceilings,
    derive_conclusions,
    run_full_analysis,
    run_partial_rsa,
    run_standard_rsa,
)


# ── Standard RSA ──────────────────────────────────────────────────────────────

def test_standard_returns_all_tests(synthetic_brain_model_rdms):
    data = synthetic_brain_model_rdms
    rngs = [np.random.default_rng(i) for i in range(len(STANDARD_TESTS))]
    results = run_standard_rsa(
        data["model_rdms"], data["brain_rdms"],
        n_perm=50, seed_sequences=rngs,
    )
    assert len(results) == len(STANDARD_TESTS)
    assert all(isinstance(r, RsaTestResult) for r in results)


def test_standard_result_fields(synthetic_brain_model_rdms):
    data = synthetic_brain_model_rdms
    rngs = [np.random.default_rng(0)] * len(STANDARD_TESTS)
    results = run_standard_rsa(
        data["model_rdms"], data["brain_rdms"],
        n_perm=50, seed_sequences=rngs,
    )
    for r in results:
        assert r.model in ("NAM", "SAM")
        assert r.brain in ("OPA", "PPA", "EVC")
        assert np.isfinite(r.r)
        assert 0.0 <= r.p_perm <= 1.0
        assert r.controls == []


def test_standard_nam_opa_positive(synthetic_brain_model_rdms):
    """NAM shares signal with OPA by construction, so r should be positive."""
    data = synthetic_brain_model_rdms
    rngs = [np.random.default_rng(0)] * len(STANDARD_TESTS)
    results = run_standard_rsa(
        data["model_rdms"], data["brain_rdms"],
        n_perm=100, seed_sequences=rngs,
    )
    nam_opa = next(r for r in results if r.model == "NAM" and r.brain == "OPA")
    assert nam_opa.r > 0.2


def test_standard_reproducible(synthetic_brain_model_rdms):
    data = synthetic_brain_model_rdms
    rngs1 = [np.random.default_rng(i) for i in range(len(STANDARD_TESTS))]
    rngs2 = [np.random.default_rng(i) for i in range(len(STANDARD_TESTS))]
    r1 = run_standard_rsa(data["model_rdms"], data["brain_rdms"],
                          n_perm=50, seed_sequences=rngs1)
    r2 = run_standard_rsa(data["model_rdms"], data["brain_rdms"],
                          n_perm=50, seed_sequences=rngs2)
    for a, b in zip(r1, r2):
        assert a.r == b.r
        assert a.p_perm == b.p_perm


# ── Partial RSA ───────────────────────────────────────────────────────────────

def test_partial_returns_all_tests(synthetic_brain_model_rdms):
    data = synthetic_brain_model_rdms
    rngs = [np.random.default_rng(i) for i in range(len(PARTIAL_TESTS))]
    results = run_partial_rsa(
        data["model_rdms"], data["brain_rdms"],
        n_perm=50, seed_sequences=rngs,
    )
    assert len(results) == len(PARTIAL_TESTS)


def test_partial_result_has_controls(synthetic_brain_model_rdms):
    data = synthetic_brain_model_rdms
    rngs = [np.random.default_rng(0)] * len(PARTIAL_TESTS)
    results = run_partial_rsa(
        data["model_rdms"], data["brain_rdms"],
        n_perm=50, seed_sequences=rngs,
    )
    for r in results:
        assert r.controls  # never empty in partial tests


def test_partial_removes_shared_signal(synthetic_brain_model_rdms):
    """
    NAM→OPA partial (controlling SAM, EVC) should be lower than the standard
    NAM→OPA, because SAM shares a component with OPA by construction.
    """
    data = synthetic_brain_model_rdms
    std_rngs = [np.random.default_rng(0)] * len(STANDARD_TESTS)
    par_rngs = [np.random.default_rng(0)] * len(PARTIAL_TESTS)
    std = run_standard_rsa(data["model_rdms"], data["brain_rdms"],
                           n_perm=50, seed_sequences=std_rngs)
    par = run_partial_rsa(data["model_rdms"], data["brain_rdms"],
                          n_perm=50, seed_sequences=par_rngs)
    std_nam_opa = next(r for r in std if r.model == "NAM" and r.brain == "OPA")
    par_nam_opa = next(r for r in par if r.model == "NAM" and r.brain == "OPA")
    assert abs(par_nam_opa.r) < abs(std_nam_opa.r)


# ── FDR ───────────────────────────────────────────────────────────────────────

def test_apply_fdr_assigns_q():
    results = [
        RsaTestResult("A", "X", r=0.3, p_perm=0.001),
        RsaTestResult("B", "X", r=0.2, p_perm=0.01),
        RsaTestResult("C", "X", r=0.1, p_perm=0.5),
    ]
    apply_fdr(results)
    assert all(np.isfinite(r.q_fdr) for r in results)
    # q is monotone in p
    qs = [r.q_fdr for r in results]
    assert qs[0] <= qs[1] <= qs[2]


# ── Noise ceiling ─────────────────────────────────────────────────────────────

def test_noise_ceiling_from_stacks(synthetic_subject_rdms_by_roi):
    out = compute_all_noise_ceilings(synthetic_subject_rdms_by_roi)
    assert set(out.keys()) == {"opa", "ppa"}
    for roi in ("opa", "ppa"):
        assert out[roi] is not None
        assert out[roi]["upper"] >= out[roi]["lower"]


def test_noise_ceiling_missing_stack():
    out = compute_all_noise_ceilings({"opa": np.array([])})
    assert out["opa"] is None


# ── Conclusions ───────────────────────────────────────────────────────────────

def test_conclusions_sam_unique_opa():
    partial = [
        RsaTestResult("SAM", "OPA", r=0.15, p_perm=0.001, q_fdr=0.005,
                      controls=["NAM", "EVC"]),
        RsaTestResult("SAM", "PPA", r=0.02, p_perm=0.4, q_fdr=0.5,
                      controls=["NAM", "EVC"]),
    ]
    c = derive_conclusions(partial)
    assert c["verdict"] == "sam_unique_opa"


def test_conclusions_sam_no_unique_signal():
    partial = [
        RsaTestResult("SAM", "OPA", r=0.01, p_perm=0.5, q_fdr=0.7,
                      controls=["NAM", "EVC"]),
        RsaTestResult("SAM", "PPA", r=0.02, p_perm=0.4, q_fdr=0.6,
                      controls=["NAM", "EVC"]),
    ]
    c = derive_conclusions(partial)
    assert c["verdict"] == "sam_no_unique_signal"


def test_conclusions_sam_both():
    partial = [
        RsaTestResult("SAM", "OPA", r=0.15, p_perm=0.001, q_fdr=0.005,
                      controls=["NAM", "EVC"]),
        RsaTestResult("SAM", "PPA", r=0.12, p_perm=0.002, q_fdr=0.006,
                      controls=["NAM", "EVC"]),
    ]
    c = derive_conclusions(partial)
    assert c["verdict"] == "sam_both_regions"


# ── Full analysis ─────────────────────────────────────────────────────────────

def test_full_analysis_produces_all_parts(synthetic_brain_model_rdms):
    data = synthetic_brain_model_rdms
    result = run_full_analysis(
        data["model_rdms"], data["brain_rdms"],
        subject_rdms_by_roi=None,
        n_perm=50, master_seed=42,
    )
    assert isinstance(result, RsaResult)
    assert len(result.standard) == len(STANDARD_TESTS)
    assert len(result.partial) == len(PARTIAL_TESTS)
    assert all(np.isfinite(r.q_fdr) for r in result.standard + result.partial)
    assert "verdict" in result.conclusions


def test_full_analysis_reproducible(synthetic_brain_model_rdms):
    data = synthetic_brain_model_rdms
    r1 = run_full_analysis(data["model_rdms"], data["brain_rdms"],
                           n_perm=50, master_seed=42)
    r2 = run_full_analysis(data["model_rdms"], data["brain_rdms"],
                           n_perm=50, master_seed=42)
    for a, b in zip(r1.standard, r2.standard):
        assert a.r == b.r
        assert a.p_perm == b.p_perm
        assert a.q_fdr == b.q_fdr


def test_full_analysis_with_noise_ceiling(synthetic_brain_model_rdms,
                                          synthetic_subject_rdms_by_roi):
    data = synthetic_brain_model_rdms
    result = run_full_analysis(
        data["model_rdms"], data["brain_rdms"],
        subject_rdms_by_roi=synthetic_subject_rdms_by_roi,
        n_perm=50, master_seed=42,
    )
    assert result.noise_ceiling
    assert result.noise_ceiling["opa"]["upper"] >= \
           result.noise_ceiling["opa"]["lower"]


def test_as_dict_is_json_serialisable(synthetic_brain_model_rdms):
    import json
    data = synthetic_brain_model_rdms
    result = run_full_analysis(data["model_rdms"], data["brain_rdms"],
                               n_perm=50, master_seed=42)
    d = result.as_dict()
    # Should not raise
    json.dumps(d)
    assert "standard" in d and "partial" in d and "conclusions" in d