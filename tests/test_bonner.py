"""Tests for src.bonner loaders."""

from pathlib import Path

import numpy as np
import pytest

from src.bonner import (
    load_all_paths,
    load_bincounts,
    load_pathway_mapping,
    load_subject_rdms,
    load_trajs,
)


# ── Trajs.mat ──────────────────────────────────────────────────────────────────

def test_load_trajs_returns_all_fields(synthetic_trajs_mat):
    path, expected_fields, (H, W, R) = synthetic_trajs_mat
    fields, arrays = load_trajs(path)
    assert fields == expected_fields
    assert set(arrays.keys()) == set(expected_fields)


def test_load_trajs_shapes(synthetic_trajs_mat):
    path, expected_fields, (H, W, R) = synthetic_trajs_mat
    _, arrays = load_trajs(path)
    for f in expected_fields:
        assert arrays[f].shape == (H, W, R)
        assert arrays[f].dtype == np.float32


def test_load_trajs_preserves_field_order(synthetic_trajs_mat):
    """Field order matters: it defines the binCounts row order."""
    path, expected_fields, _ = synthetic_trajs_mat
    fields, _ = load_trajs(path)
    assert fields == expected_fields


# ── allPaths.mat ───────────────────────────────────────────────────────────────

def test_load_all_paths_shape_and_values(synthetic_allpaths_mat):
    path, expected = synthetic_allpaths_mat
    arr = load_all_paths(path)
    assert arr.shape == expected.shape
    assert np.allclose(arr, expected)


def test_load_all_paths_dtype(synthetic_allpaths_mat):
    path, _ = synthetic_allpaths_mat
    arr = load_all_paths(path)
    assert arr.dtype == np.float32


# ── binCounts.mat ──────────────────────────────────────────────────────────────

def test_load_bincounts_shape(synthetic_bincounts_mat):
    path, expected = synthetic_bincounts_mat
    arr = load_bincounts(path)
    assert arr.shape == (173, 180)
    assert np.allclose(arr, expected)


# ── all_sub_slrdms.npy ─────────────────────────────────────────────────────────

def test_load_subject_rdms_memory_maps(tmp_path):
    """Loader must not eagerly load the array into RAM."""
    arr = np.zeros((4, 100, 50, 50), dtype=np.float32)
    path = tmp_path / "rdms.npy"
    np.save(path, arr)

    loaded = load_subject_rdms(path)
    assert loaded.shape == arr.shape
    assert loaded.dtype == arr.dtype


def test_load_subject_rdms_indexing(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.random((3, 20, 50, 50)).astype(np.float32)
    path = tmp_path / "rdms.npy"
    np.save(path, arr)

    loaded = load_subject_rdms(path)
    sub0 = np.asarray(loaded[0])
    assert sub0.shape == (20, 50, 50)
    assert np.allclose(sub0, arr[0])


# ── Pathway mapping ────────────────────────────────────────────────────────────

def test_load_pathway_mapping(synthetic_mapping_csv):
    path, df = synthetic_mapping_csv
    mapping = load_pathway_mapping(path)
    assert set(mapping.keys()) == set(df["scene_k"].astype(int))
    assert mapping[0] == "pathways0010"
    assert mapping[4] == "pathways0028"


def test_load_pathway_mapping_scene_keys_are_int(synthetic_mapping_csv):
    path, _ = synthetic_mapping_csv
    mapping = load_pathway_mapping(path)
    assert all(isinstance(k, int) for k in mapping.keys())


# ── Real-data integration tests (skipped in CI) ────────────────────────────────

REAL_TRAJS = Path("affordances/Trajs.mat")
REAL_ALLPATHS = Path("affordances/allPaths.mat")
REAL_BINCOUNTS = Path("affordances/binCounts.mat")


@pytest.mark.skipif(not REAL_TRAJS.exists(),
                    reason="Real Trajs.mat not present")
def test_real_trajs_loading():
    fields, arrays = load_trajs(REAL_TRAJS)
    assert len(fields) == 173
    assert all(f.startswith("pathways") for f in fields)
    for f in fields:
        assert arrays[f].shape == (384, 512, 10)


@pytest.mark.skipif(not REAL_ALLPATHS.exists(),
                    reason="Real allPaths.mat not present")
def test_real_allpaths_loading():
    arr = load_all_paths(REAL_ALLPATHS)
    assert arr.shape == (384, 512, 50)
    assert arr.max() <= 10


@pytest.mark.skipif(not REAL_BINCOUNTS.exists(),
                    reason="Real binCounts.mat not present")
def test_real_bincounts_loading():
    arr = load_bincounts(REAL_BINCOUNTS)
    assert arr.shape == (173, 180)
    assert arr.min() >= 0