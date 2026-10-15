"""Tests for src.io."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.io import load_brain_rdm, save_json, load_json


def test_load_brain_rdm_symmetrises(tmp_mat_file):
    rdm = load_brain_rdm(tmp_mat_file)
    assert rdm.shape == (50, 50)
    assert np.allclose(rdm, rdm.T)
    assert np.allclose(np.diag(rdm), 0.0)


def test_load_brain_rdm_handles_nan_diagonal(tmp_path):
    import scipy.io as sio
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 5))
    d = squareform(pdist(x))
    np.fill_diagonal(d, np.nan)
    path = tmp_path / "nan_rdm.mat"
    sio.savemat(path, {"rdm": d})

    loaded = load_brain_rdm(path)
    assert np.all(np.isfinite(loaded))


def test_load_brain_rdm_preserves_off_diagonal(tmp_path):
    """Symmetrisation must not corrupt valid off-diagonal entries."""
    import scipy.io as sio
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 5))
    d_original = squareform(pdist(x))
    path = tmp_path / "test.mat"
    sio.savemat(path, {"rdm": d_original})

    loaded = load_brain_rdm(path)
    assert np.allclose(loaded, d_original, atol=1e-6)


def test_save_load_json_roundtrip(tmp_path):
    obj = {"a": 1, "b": [1.5, 2.5], "c": {"nested": True}}
    path = tmp_path / "out.json"
    save_json(obj, path)
    back = load_json(path)
    assert back == obj


def test_save_json_handles_numpy(tmp_path):
    obj = {
        "arr": np.array([1.0, 2.0, 3.0]),
        "scalar": np.float64(3.14),
        "int": np.int32(7),
    }
    path = tmp_path / "numpy.json"
    save_json(obj, path)
    back = load_json(path)
    assert back["arr"] == [1.0, 2.0, 3.0]
    assert abs(back["scalar"] - 3.14) < 1e-10
    assert back["int"] == 7


def test_save_json_creates_parent_dirs(tmp_path):
    obj = {"x": 1}
    nested = tmp_path / "a" / "b" / "out.json"
    save_json(obj, nested)
    assert nested.exists()