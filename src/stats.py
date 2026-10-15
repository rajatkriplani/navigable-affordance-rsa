"""
Statistical core for the navigable-affordance RSA pipeline.

All functions use local numpy Generator instances — no module-level
global random state — so results are reproducible regardless of call order.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests


# ── RDM index helpers ──────────────────────────────────────────────────────────

def rdm_pair_indices(n, k=-1):
    """Return (i_idx, j_idx) for the lower triangle of an n x n matrix."""
    return np.tril_indices(n, k=k)


def distance_matrix_to_tril(distance_matrix, k=-1):
    """
    Extract the lower triangle of a symmetric matrix as a 1D vector,
    using the same ordering as np.tril_indices(n, k=-1).
    """
    n = distance_matrix.shape[0]
    i_idx, j_idx = rdm_pair_indices(n, k=k)
    return distance_matrix[i_idx, j_idx]


def pairwise_distances_to_tril(feature_matrix, metric="euclidean"):
    """
    Compute pairwise distances and return them in lower-triangle ordering.

    This is the Fix-3-correct pattern: squareform then tril extraction,
    NOT direct pdist output (which uses upper-triangle row-major order).
    """
    n = feature_matrix.shape[0]
    pdist_vec = pdist(feature_matrix, metric=metric)
    full = squareform(pdist_vec)
    return distance_matrix_to_tril(full)


# ── Partial correlation ────────────────────────────────────────────────────────

def _normalise_covariates(covariates):
    """Return a list of 1D arrays; accept None, single 1D array, or iterable."""
    if covariates is None:
        return []
    if isinstance(covariates, np.ndarray):
        if covariates.ndim == 1:
            return [covariates]
        # assume 2D: rows are observations, columns are covariates
        return [covariates[:, c] for c in range(covariates.shape[1])]
    return list(covariates)


def exact_partial_spearman(x, y, covariates=None):
    """
    Partial Spearman correlation r(x, y | covariates) via OLS rank-residualisation.

    Parameters
    ----------
    x, y : 1D arrays, equal length.
    covariates : None, single 1D array, list of 1D arrays, or 2D array
        (columns = covariates).

    Returns
    -------
    float
        Partial Spearman correlation. Equals Spearman r when no covariates
        are supplied.
    """
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)

    covars = _normalise_covariates(covariates)
    if len(covars) == 0:
        return float(pearsonr(rx, ry)[0])

    rz = np.column_stack([rankdata(c).astype(float) for c in covars])
    reg = LinearRegression()
    res_x = rx - reg.fit(rz, rx).predict(rz)
    res_y = ry - reg.fit(rz, ry).predict(rz)
    return float(pearsonr(res_x, res_y)[0])


# ── Permutation test ───────────────────────────────────────────────────────────

def permutation_test_partial(
    model_rdm,
    brain_rdm,
    covar_rdms=None,
    n_perm=10_000,
    rng=None,
):
    """
    Label-permutation test for a partial RSA correlation.

    Permutes the model RDM's stimulus labels; brain and covariate RDMs stay
    fixed. Returns the observed partial r, the null distribution, and a
    one-tailed p-value.

    Parameters
    ----------
    model_rdm, brain_rdm : 2D symmetric arrays, same shape.
    covar_rdms : list of 2D arrays or None.
    n_perm : int
    rng : np.random.Generator or None
        If None, a fresh default_rng(42) is used.

    Returns
    -------
    (r_obs, null_distribution, p_value)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = model_rdm.shape[0]
    i_idx, j_idx = rdm_pair_indices(n)

    b_vec = brain_rdm[i_idx, j_idx]
    m_vec = model_rdm[i_idx, j_idx]
    c_vecs = [c[i_idx, j_idx] for c in (covar_rdms or [])]

    r_obs = exact_partial_spearman(m_vec, b_vec, c_vecs)

    null = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        perm = rng.permutation(n)
        m_perm = model_rdm[np.ix_(perm, perm)][i_idx, j_idx]
        null[p] = exact_partial_spearman(m_perm, b_vec, c_vecs)

    p_value = float((null >= r_obs).mean())
    return r_obs, null, p_value


# ── Noise ceiling ──────────────────────────────────────────────────────────────

def compute_noise_ceiling(subject_rdms_3d):
    """
    Nili et al. (2014) leave-one-subject-out noise ceiling.

    Parameters
    ----------
    subject_rdms_3d : ndarray, shape (n_stim, n_stim, n_subject)

    Returns
    -------
    (lower, upper) : tuple of floats, or (None, None) if not computable.
    """
    if subject_rdms_3d is None or subject_rdms_3d.size == 0:
        return None, None

    n_stim = subject_rdms_3d.shape[0]
    n_sub = subject_rdms_3d.shape[2]
    if n_sub < 2:
        return None, None

    i_idx, j_idx = rdm_pair_indices(n_stim)
    mean_all = subject_rdms_3d.mean(axis=2)
    mean_all_vec = mean_all[i_idx, j_idx]

    lows, highs = [], []
    for s in range(n_sub):
        sub_vec = subject_rdms_3d[:, :, s][i_idx, j_idx]
        mean_loo = (mean_all * n_sub - subject_rdms_3d[:, :, s]) / (n_sub - 1)
        lows.append(float(spearmanr(sub_vec, mean_loo[i_idx, j_idx])[0]))
        highs.append(float(spearmanr(sub_vec, mean_all_vec)[0]))

    return float(np.mean(lows)), float(np.mean(highs))


# ── Multiple-comparison correction ─────────────────────────────────────────────

def fdr_correct(p_values, method="fdr_bh"):
    """
    FDR-correct a 1D array of p-values.

    Returns
    -------
    ndarray of q-values, same length and order as input.
    """
    p_values = np.asarray(p_values, dtype=float)
    _, q_values, _, _ = multipletests(p_values, method=method)
    return q_values