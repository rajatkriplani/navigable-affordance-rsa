"""
File IO for the navigable-affordance RSA pipeline.

Handles .mat (v5/v7 and v7.3 via h5py fallback), .npy, .json.
Read-only for input data, write for outputs.
"""

import json
from pathlib import Path

import numpy as np


# ── RDM loaders ────────────────────────────────────────────────────────────────

def _is_hdf5(path):
    """MATLAB v7.3 files start with the HDF5 magic bytes."""
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def load_brain_rdm(path):
    """
    Load a 50x50 RDM from a .mat file, symmetrise, zero the NaN diagonal.

    Handles both MATLAB v5/v7 (scipy.io) and v7.3 (h5py).

    Returns
    -------
    ndarray, shape (n_stim, n_stim), float
    """
    path = Path(path)
    if _is_hdf5(path):
        import h5py
        with h5py.File(path, "r") as f:
            rdm = np.asarray(f["rdm"]).T
    else:
        import scipy.io as sio
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        rdm = np.asarray(mat["rdm"])

    rdm = np.nan_to_num(rdm, nan=0.0)
    lower = np.tril(rdm, k=-1)
    return lower + lower.T


def load_npy_rdm(path):
    """
    Load an .npy RDM (or RDM stack) in memory-mapped mode.

    Returns
    -------
    ndarray, memory-mapped. Call np.asarray() to force load.
    """
    return np.load(path, mmap_mode="r", allow_pickle=False)


# ── JSON IO ────────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj, path):
    """Write obj to JSON, converting numpy scalars/arrays/Paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── CSV IO (thin pandas wrappers, kept for symmetry) ──────────────────────────

def save_features_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_features_csv(path):
    import pandas as pd
    return pd.read_csv(path, index_col=0)