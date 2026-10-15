"""
Loaders for the Bonner & Epstein (2017) dataset.

All loaders preserve MATLAB struct field order, which is required for the
173-row alignment between Trajs.mat and binCounts.mat (verified at mass
correlation r = 0.9984 and angular-histogram reconstruction r = 0.994).

Memory policy:
  - Trajs.mat (6.5 MB) is loaded fully.
  - allPaths.mat (394 KB) is loaded fully.
  - binCounts.mat (48 KB) is loaded fully.
  - all_sub_slrdms.npy (4.6 GB) is memory-mapped, never loaded into RAM.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)


# ── Trajs.mat ──────────────────────────────────────────────────────────────────

def load_trajs(path):
    """
    Load Trajs.mat.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    field_names : list of str
        Field names in MATLAB struct order (matches binCounts row order).
    arrays : dict
        {field_name: (H, W, n_raters) float32 array}. H=384, W=512,
        n_raters=10 for the Bonner data.
    """
    path = Path(path)
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    t = mat["Trajs"]

    if hasattr(t, "_fieldnames"):
        field_names = list(t._fieldnames)
        arrays = {f: np.asarray(getattr(t, f), dtype=np.float32)
                  for f in field_names}
    elif hasattr(t, "dtype") and t.dtype.names:
        # numpy record array fallback
        field_names = list(t.dtype.names)
        flat = np.atleast_1d(t).ravel()
        arrays = {f: np.asarray(flat[0][f], dtype=np.float32)
                  for f in field_names}
    else:
        raise ValueError(
            f"Unexpected Trajs.mat structure: {type(t).__name__}"
        )

    logger.info(
        "Loaded %d Trajs fields from %s (first=%s, last=%s)",
        len(field_names), path.name,
        field_names[0] if field_names else None,
        field_names[-1] if field_names else None,
    )
    return field_names, arrays


# ── allPaths.mat ───────────────────────────────────────────────────────────────

def load_all_paths(path):
    """
    Load allPaths.mat.

    Returns
    -------
    ndarray, shape (H, W, 50), float32
        Rater-aggregated binary trajectory maps for the 50 fMRI scenes,
        in RDM row/column order.
    """
    path = Path(path)
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    arr = np.asarray(mat["allPaths"], dtype=np.float32)
    logger.info("Loaded allPaths %s from %s", arr.shape, path.name)
    return arr


# ── binCounts.mat ──────────────────────────────────────────────────────────────

def load_bincounts(path):
    """
    Load binCounts.mat.

    Returns
    -------
    ndarray, shape (173, 180), float32
        Row order matches Trajs.mat field order (verified: mass r = 0.998).
    """
    path = Path(path)
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    arr = np.asarray(mat["binCounts"], dtype=np.float32)
    logger.info("Loaded binCounts %s from %s", arr.shape, path.name)
    return arr


# ── all_sub_slrdms.npy ─────────────────────────────────────────────────────────

def load_subject_rdms(path):
    """
    Memory-map the per-subject per-voxel RDM stack.

    Returns
    -------
    ndarray, memory-mapped, shape (n_sub, n_vox, 50, 50), float32
        For Bonner data: (16, 30679, 50, 50). Do not call np.asarray() on
        the full array — it is 4.6 GB. Index by subject and reduce.
    """
    path = Path(path)
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    logger.info(
        "Memory-mapped %s with shape %s (%.1f GB)",
        path.name, arr.shape, arr.nbytes / 1e9,
    )
    return arr


# ── Pathway-to-RDM mapping ─────────────────────────────────────────────────────

def load_pathway_mapping(csv_path):
    """
    Load the pathway-to-RDM mapping produced by the mapping verification.

    Parameters
    ----------
    csv_path : str or Path
        CSV with columns: scene_k, trajs_field, best_r, ...

    Returns
    -------
    dict {scene_k: pathway_id}
        scene_k is int (0..49); pathway_id is str (e.g. "pathways0010").
    """
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df["scene_k"].astype(int), df["trajs_field"].astype(str)))
    logger.info("Loaded pathway mapping for %d scenes from %s",
                len(mapping), Path(csv_path).name)
    return mapping