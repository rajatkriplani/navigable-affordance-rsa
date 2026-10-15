"""
Core RSA analysis: standard, partial, and noise-ceiling-corrected tests
between model RDMs and brain RDMs.

Design notes
------------
- All random state is local. permutation_test_partial from src.stats
  accepts an explicit rng, and every rng is constructed from a caller-
  supplied master seed via SeedSequence.spawn, so test order does not
  affect results.

- The RSA produces three classes of result:
    1. Standard tests: model RDM vs brain RDM, no controls.
    2. Partial tests: model RDM vs brain RDM controlling for other model
       + EVC (novelty test).
    3. Noise ceiling: Nili et al. (2014) LOO bound from per-subject RDMs.

- The full structured output is designed for JSON serialisation, and the
  CLI writes both a JSON file and a summary figure.
"""

from dataclasses import dataclass, field

import numpy as np

from .logging_utils import get_logger
from .stats import (
    compute_noise_ceiling,
    fdr_correct,
    permutation_test_partial,
)

logger = get_logger(__name__)


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class RsaTestResult:
    model: str
    brain: str
    r: float
    p_perm: float
    q_fdr: float = float("nan")
    controls: list = field(default_factory=list)

    def as_dict(self):
        return {
            "model": self.model,
            "brain": self.brain,
            "controls": list(self.controls),
            "r": float(self.r),
            "p_perm": float(self.p_perm),
            "q_fdr": float(self.q_fdr),
        }


@dataclass
class RsaResult:
    standard: list = field(default_factory=list)
    partial: list = field(default_factory=list)
    noise_ceiling: dict = field(default_factory=dict)
    conclusions: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "standard": [r.as_dict() for r in self.standard],
            "partial": [r.as_dict() for r in self.partial],
            "noise_ceiling": self.noise_ceiling,
            "conclusions": self.conclusions,
        }


# ── Test suite definitions ────────────────────────────────────────────────────

STANDARD_TESTS = [
    # (model_name, brain_name, model_key, brain_key)
    ("NAM", "OPA", "nam", "opa"),
    ("NAM", "PPA", "nam", "ppa"),
    ("NAM", "EVC", "nam", "evc"),
    ("SAM", "OPA", "sam", "opa"),
    ("SAM", "PPA", "sam", "ppa"),
    ("SAM", "EVC", "sam", "evc"),
]

PARTIAL_TESTS = [
    # (target, brain, target_key, brain_key, control_names, control_keys)
    ("SAM", "OPA", "sam", "opa", ["NAM", "EVC"], ["nam", "evc"]),
    ("SAM", "PPA", "sam", "ppa", ["NAM", "EVC"], ["nam", "evc"]),
    ("NAM", "OPA", "nam", "opa", ["SAM", "EVC"], ["sam", "evc"]),
    ("NAM", "PPA", "nam", "ppa", ["SAM", "EVC"], ["sam", "evc"]),
]


# ── Analysis functions ────────────────────────────────────────────────────────

def run_standard_rsa(
    model_rdms,
    brain_rdms,
    n_perm=10_000,
    seed_sequences=None,
):
    """
    Compute standard Spearman RSA correlations for all pairs in STANDARD_TESTS.

    Parameters
    ----------
    model_rdms : dict {key: 2D array}
    brain_rdms : dict {key: 2D array}
    n_perm : int
    seed_sequences : list of Generator or None
        One per test. If None, a fresh Generator(42) is used for all.

    Returns
    -------
    list of RsaTestResult
    """
    if seed_sequences is None:
        seed_sequences = [np.random.default_rng(42)] * len(STANDARD_TESTS)

    results = []
    for (mn, bn, mk, bk), rng in zip(STANDARD_TESTS, seed_sequences):
        r, _, p = permutation_test_partial(
            model_rdms[mk], brain_rdms[bk],
            covar_rdms=None,
            n_perm=n_perm, rng=rng,
        )
        results.append(RsaTestResult(
            model=mn, brain=bn, r=r, p_perm=p, controls=[],
        ))
    return results


def run_partial_rsa(
    model_rdms,
    brain_rdms,
    n_perm=10_000,
    seed_sequences=None,
):
    """
    Compute partial Spearman RSA correlations controlling for named
    covariates (Fix 1: EVC + other model as joint controls).

    Controls may live in either the model or brain dict. We merge them
    for lookup so that e.g. NAM (a model RDM) and EVC (a brain RDM) can
    both be controls for the SAM→OPA test.
    """
    if seed_sequences is None:
        seed_sequences = [np.random.default_rng(42)] * len(PARTIAL_TESTS)

    # Merge for control lookup only. Model/brain identity of the target
    # is preserved below.
    lookup = {**model_rdms, **brain_rdms}

    results = []
    for (tn, bn, tk, bk, cn, ck), rng in zip(PARTIAL_TESTS, seed_sequences):
        covars = [lookup[k] for k in ck]
        r, _, p = permutation_test_partial(
            model_rdms[tk], brain_rdms[bk],
            covar_rdms=covars,
            n_perm=n_perm, rng=rng,
        )
        results.append(RsaTestResult(
            model=tn, brain=bn, r=r, p_perm=p, controls=list(cn),
        ))
    return results


def apply_fdr(results, method="fdr_bh"):
    """
    Apply FDR correction across all results in a list. Modifies each
    result's q_fdr in place.
    """
    p_values = [r.p_perm for r in results]
    q_values = fdr_correct(p_values, method=method)
    for r, q in zip(results, q_values):
        r.q_fdr = float(q)


def compute_all_noise_ceilings(subject_rdms_by_roi, brain_names=None):
    """
    Compute LOO noise ceilings per ROI from per-subject RDM stacks.

    Parameters
    ----------
    subject_rdms_by_roi : dict {brain_name: (50, 50, n_sub) array}
    brain_names : list or None
        Order of keys to produce. If None, sorted dict keys.

    Returns
    -------
    dict {brain_name: {"lower": float, "upper": float}} or
        {"brain_name": None} if not computable.
    """
    if brain_names is None:
        brain_names = sorted(subject_rdms_by_roi.keys())

    out = {}
    for name in brain_names:
        stack = subject_rdms_by_roi.get(name)
        if stack is None or stack.size == 0:
            out[name] = None
            continue
        lo, hi = compute_noise_ceiling(stack)
        if lo is None:
            out[name] = None
        else:
            out[name] = {"lower": float(lo), "upper": float(hi)}
    return out


# ── Conclusion logic (Fix 10/11: evidence-gated, no hardcoded SUCCESS) ────────

def derive_conclusions(partial_results, alpha=0.05):
    """
    Extract the paper's headline conclusion from the partial RSA results.

    Returns a dict with the SAM→OPA and SAM→PPA partial correlations and
    a verdict string that Phase 6 ablation can refine.
    """
    def _find(model, brain):
        for r in partial_results:
            if r.model == model and r.brain == brain:
                return r
        return None

    sam_opa = _find("SAM", "OPA")
    sam_ppa = _find("SAM", "PPA")

    verdict = "insufficient_data"
    if sam_opa is not None and sam_ppa is not None:
        opa_sig = sam_opa.q_fdr < alpha
        ppa_sig = sam_ppa.q_fdr < alpha

        if opa_sig and not ppa_sig:
            verdict = "sam_unique_opa"
        elif opa_sig and ppa_sig:
            verdict = "sam_both_regions"
        elif not opa_sig:
            verdict = "sam_no_unique_signal"

    return {
        "sam_opa_partial_r": float(sam_opa.r) if sam_opa else None,
        "sam_opa_partial_q": float(sam_opa.q_fdr) if sam_opa else None,
        "sam_ppa_partial_r": float(sam_ppa.r) if sam_ppa else None,
        "sam_ppa_partial_q": float(sam_ppa.q_fdr) if sam_ppa else None,
        "verdict": verdict,
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_full_analysis(
    model_rdms,
    brain_rdms,
    subject_rdms_by_roi=None,
    n_perm=10_000,
    master_seed=42,
    fdr_method="fdr_bh",
):
    """
    Run standard RSA, partial RSA, FDR correction, noise ceiling, and
    conclusion derivation. Returns an RsaResult.

    Seed derivation: each test gets its own Generator from a shared
    SeedSequence, so tests are reproducible regardless of execution order.
    """
    n_standard = len(STANDARD_TESTS)
    n_partial = len(PARTIAL_TESTS)
    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(n_standard + n_partial)

    logger.info("Running %d standard + %d partial RSA tests (n_perm=%d) …",
                n_standard, n_partial, n_perm)

    standard_rngs = [np.random.default_rng(s) for s in child_seeds[:n_standard]]
    partial_rngs = [np.random.default_rng(s) for s in child_seeds[n_standard:]]

    standard_results = run_standard_rsa(
        model_rdms, brain_rdms, n_perm=n_perm, seed_sequences=standard_rngs,
    )
    partial_results = run_partial_rsa(
        model_rdms, brain_rdms, n_perm=n_perm, seed_sequences=partial_rngs,
    )

    # FDR across all tests
    all_results = standard_results + partial_results
    apply_fdr(all_results, method=fdr_method)

    # Noise ceiling
    ceilings = {}
    if subject_rdms_by_roi:
        ceilings = compute_all_noise_ceilings(subject_rdms_by_roi)

    # Conclusions
    conclusions = derive_conclusions(partial_results)

    return RsaResult(
        standard=standard_results,
        partial=partial_results,
        noise_ceiling=ceilings,
        conclusions=conclusions,
    )