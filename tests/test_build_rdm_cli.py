"""Tests for src.cli.build_rdm."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.affordance import FEATURE_NAMES
from src.cli.build_rdm import main as build_rdm_main


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def build_rdm_workspace(tmp_path, rng):
    """A features CSV (alphabetically sorted), a mapping CSV, and a config."""
    n_scenes = 6
    # Scene order (0..5) maps to these image names, deliberately NOT alphabetical
    scene_order = [
        "pathways0035", "pathways0010", "pathways0024",
        "pathways0011", "pathways0028", "pathways0014",
    ]
    # Features CSV sorted alphabetically
    all_images = sorted(scene_order)
    df_features = pd.DataFrame({
        "image_name": all_images,
        "floor_fraction": rng.uniform(0.1, 0.9, len(all_images)),
        "sam_failed": [False] * len(all_images),
    })
    for k in FEATURE_NAMES:
        df_features[k] = rng.normal(size=len(all_images))

    features_csv = tmp_path / "outputs" / "features" / "features.csv"
    features_csv.parent.mkdir(parents=True)
    df_features.to_csv(features_csv, index=False)

    mapping_csv = tmp_path / "allpaths_mapping.csv"
    pd.DataFrame({
        "scene_k": list(range(n_scenes)),
        "trajs_field": scene_order,
        "best_r": [1.0] * n_scenes,
    }).to_csv(mapping_csv, index=False)

    cfg = {
        "paths": {
            "stimuli_dir":     str(tmp_path / "stimuli"),
            "affordances_dir": str(tmp_path / "affordances"),
            "brain_rdms_dir":  str(tmp_path / "rdms"),
            "mask_cache_dir":  str(tmp_path / "outputs" / "masks"),
            "features_csv":    str(features_csv),
            "sam_rdm":         str(tmp_path / "outputs" / "rdms" / "sam.rdm.npy"),
            "stats_dir":       str(tmp_path / "outputs" / "stats"),
            "figures_dir":     str(tmp_path / "outputs" / "figures"),
        },
        "sam": {
            "checkpoint": str(tmp_path / "fake.pth"),
            "model_type": "vit_b", "device": "cpu",
            "primary_prompt": [0.5, 0.9], "fallback_prompts": [],
            "min_floor_fraction": 0.02,
        },
        "features": {
            "r_base_px": 60.0, "horizon_ratio": 0.5,
            "names": list(FEATURE_NAMES),
            "topological": ["GTI","BPD","NNE","SC"],
            "clearance": ["MAC","PFF","BCR","SCE"],
            "per_image_seed_policy": "hash",
        },
        "stats": {
            "n_stimuli": n_scenes, "n_permutations": 100,
            "n_permutations_quick": 50,
            "fdr_method": "fdr_bh", "master_seed": 42,
        },
    }
    for d in ("stimuli", "affordances", "rdms", "configs"):
        (tmp_path / d).mkdir(exist_ok=True)
    cfg_path = tmp_path / "configs" / "test.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return cfg_path, tmp_path, features_csv, mapping_csv, scene_order


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_build_rdm_writes_npy(build_rdm_workspace):
    cfg_path, tmp_path, _, mapping_csv, _ = build_rdm_workspace
    rc = build_rdm_main([
        "--config", str(cfg_path),
        "--mapping", str(mapping_csv),
    ])
    assert rc == 0
    out = tmp_path / "outputs" / "rdms" / "sam.rdm.npy"
    assert out.exists()


def test_build_rdm_shape(build_rdm_workspace):
    cfg_path, tmp_path, _, mapping_csv, scene_order = build_rdm_workspace
    build_rdm_main(["--config", str(cfg_path), "--mapping", str(mapping_csv)])
    rdm = np.load(tmp_path / "outputs" / "rdms" / "sam.rdm.npy")
    n = len(scene_order)
    assert rdm.shape == (n, n)
    assert np.allclose(rdm, rdm.T)
    assert np.allclose(np.diag(rdm), 0.0)


def test_build_rdm_scene_order_matches_mapping(build_rdm_workspace):
    """
    The order sidecar must list scene_k in increasing order with the
    pathways id from the mapping CSV, not alphabetical order.
    """
    cfg_path, tmp_path, _, mapping_csv, scene_order = build_rdm_workspace
    build_rdm_main(["--config", str(cfg_path), "--mapping", str(mapping_csv)])

    order_csv = tmp_path / "outputs" / "rdms" / "sam.rdm.order.csv"
    order_df = pd.read_csv(order_csv)
    assert order_df["scene_k"].tolist() == list(range(len(scene_order)))
    assert order_df["image_name"].tolist() == scene_order
    # Sanity: alphabetical sort would give a different order
    assert scene_order != sorted(scene_order)


def test_build_rdm_writes_order_sidecar(build_rdm_workspace):
    cfg_path, tmp_path, _, mapping_csv, _ = build_rdm_workspace
    build_rdm_main(["--config", str(cfg_path), "--mapping", str(mapping_csv)])
    order_csv = tmp_path / "outputs" / "rdms" / "sam.rdm.order.csv"
    assert order_csv.exists()


def test_build_rdm_missing_features_returns_error(build_rdm_workspace):
    cfg_path, tmp_path, features_csv, mapping_csv, _ = build_rdm_workspace
    features_csv.unlink()
    rc = build_rdm_main(["--config", str(cfg_path), "--mapping", str(mapping_csv)])
    assert rc == 1


def test_build_rdm_missing_mapping_returns_error(build_rdm_workspace, monkeypatch):
    cfg_path, tmp_path, _, _, _ = build_rdm_workspace
    # Move into a directory with no mapping candidates anywhere
    monkeypatch.chdir(tmp_path)
    for candidate in ("allpaths_mapping.csv", "mapping_binary_sum.csv"):
        p = tmp_path / candidate
        if p.exists():
            p.unlink()
    rc = build_rdm_main(["--config", str(cfg_path)])
    assert rc == 1


def test_build_rdm_no_mapping_flag_uses_csv_order(build_rdm_workspace):
    """--no-mapping skips reordering and uses features CSV order verbatim."""
    cfg_path, tmp_path, _, _, _ = build_rdm_workspace
    rc = build_rdm_main(["--config", str(cfg_path), "--no-mapping"])
    assert rc == 0
    out = tmp_path / "outputs" / "rdms" / "sam.rdm.npy"
    assert out.exists()


def test_build_rdm_rejects_missing_scene_in_mapping(build_rdm_workspace):
    """If the mapping references a scene not in the features CSV, error."""
    cfg_path, tmp_path, _, mapping_csv, _ = build_rdm_workspace
    # Add a scene_k that has no matching feature row
    df = pd.read_csv(mapping_csv)
    df = pd.concat([
        df,
        pd.DataFrame({"scene_k": [99], "trajs_field": ["pathways9999"],
                      "best_r": [1.0]}),
    ], ignore_index=True)
    df.to_csv(mapping_csv, index=False)

    rc = build_rdm_main(["--config", str(cfg_path), "--mapping", str(mapping_csv)])
    assert rc == 1