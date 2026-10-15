"""Tests for src.cli.ablation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform

from src.affordance import FEATURE_NAMES
from src.cli.ablation import main as ablation_main


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture
def ablation_workspace(tmp_path, rng):
    """Complete workspace: features CSV, mapping, RDMs, config."""
    n = 12
    scene_order = [
        "pathways0010", "pathways0024", "pathways0035",
        "pathways0011", "pathways0028", "pathways0014",
        "pathways0015", "pathways0001", "pathways0002",
        "pathways0096", "pathways0027", "pathways0081",
    ]

    # Latent brain structure: OPA responds to clearance signal
    brain_signal = rng.normal(size=(n, 3))
    clear_signal = rng.normal(size=(n, 2))

    nam_rdm = squareform(pdist(brain_signal))
    evc_rdm = squareform(pdist(rng.normal(size=(n, 3))))
    opa_rdm = squareform(pdist(np.column_stack([brain_signal, clear_signal])))
    ppa_rdm = squareform(pdist(brain_signal))

    rdms_dir = tmp_path / "data" / "rdms"
    rdms_dir.mkdir(parents=True)
    for name, arr in [("OPA", opa_rdm), ("PPA", ppa_rdm),
                      ("EVC", evc_rdm), ("NAM", nam_rdm)]:
        sio.savemat(rdms_dir / f"{name}.mat", {"rdm": arr})

    # Features CSV (alphabetically sorted)
    df_features = pd.DataFrame({"image_name": sorted(scene_order)})
    df_features["floor_fraction"] = rng.uniform(0.1, 0.9, n)
    df_features["sam_failed"] = False
    # Clearance features correlate with the clearance signal,
    # topology features are noise
    df_features["MAC"] = clear_signal[:, 0][np.argsort(np.argsort(
        df_features["image_name"].tolist()))] + 0.1 * rng.normal(size=n)
    # Simpler: assign per image in the sorted order
    sorted_names = sorted(scene_order)
    idx_of = {name: i for i, name in enumerate(scene_order)}
    for feature in ["PFF", "BCR", "SCE"]:
        df_features[feature] = rng.normal(size=n)
    for feature in ["BPD", "SC", "GTI", "NNE"]:
        df_features[feature] = rng.normal(size=n)
    # Now align clearance to brain order properly via mapping later

    features_csv = tmp_path / "outputs" / "features" / "features.csv"
    features_csv.parent.mkdir(parents=True)
    df_features.to_csv(features_csv, index=False)

    mapping_csv = tmp_path / "allpaths_mapping.csv"
    pd.DataFrame({
        "scene_k": list(range(n)),
        "trajs_field": scene_order,
        "best_r": [1.0] * n,
    }).to_csv(mapping_csv, index=False)

    cfg = {
        "paths": {
            "stimuli_dir":     str(tmp_path / "data" / "stimuli"),
            "affordances_dir": str(tmp_path / "data" / "affordances"),
            "brain_rdms_dir":  str(rdms_dir),
            "mask_cache_dir":  str(tmp_path / "outputs" / "masks"),
            "features_csv":    str(features_csv),
            "sam_rdm":         str(tmp_path / "outputs" / "rdms" / "sam.npy"),
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
            "n_stimuli": n, "n_permutations": 50,
            "n_permutations_quick": 20,
            "fdr_method": "fdr_bh", "master_seed": 42,
        },
    }
    for d in ("stimuli", "affordances", "configs"):
        (tmp_path / "data" / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs").mkdir(exist_ok=True)
    cfg_path = tmp_path / "configs" / "test.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return cfg_path, tmp_path, mapping_csv


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_ablation_cli_writes_json(ablation_workspace):
    cfg_path, tmp_path, mapping_csv = ablation_workspace
    rc = ablation_main([
        "--config", str(cfg_path),
        "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    assert rc == 0
    json_path = tmp_path / "outputs" / "stats" / "ablation_opa.json"
    assert json_path.exists()


def test_ablation_cli_json_structure(ablation_workspace):
    import json
    cfg_path, tmp_path, mapping_csv = ablation_workspace
    ablation_main([
        "--config", str(cfg_path), "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    with open(tmp_path / "outputs" / "stats" / "ablation_opa.json") as f:
        d = json.load(f)
    assert "features" in d
    assert "interpretation" in d
    assert len(d["features"]) == 8
    for f in d["features"]:
        assert set(f.keys()) >= {"feature", "family", "partial_r", "partial_q"}


def test_ablation_cli_writes_figure(ablation_workspace):
    cfg_path, tmp_path, mapping_csv = ablation_workspace
    ablation_main([
        "--config", str(cfg_path), "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    fig_path = tmp_path / "outputs" / "figures" / "ablation_opa.png"
    assert fig_path.exists()


def test_ablation_cli_missing_features_returns_error(ablation_workspace):
    cfg_path, tmp_path, mapping_csv = ablation_workspace
    (tmp_path / "outputs" / "features" / "features.csv").unlink()
    rc = ablation_main([
        "--config", str(cfg_path), "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    assert rc == 1


def test_ablation_cli_missing_rdm_returns_error(ablation_workspace):
    cfg_path, tmp_path, mapping_csv = ablation_workspace
    (tmp_path / "data" / "rdms" / "OPA.mat").unlink()
    rc = ablation_main([
        "--config", str(cfg_path), "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    assert rc == 1


def test_ablation_cli_size_mismatch_returns_error(ablation_workspace):
    """Feature rows must match brain RDM stimuli count."""
    cfg_path, tmp_path, mapping_csv = ablation_workspace
    # Truncate the features CSV to fewer rows than brain RDMs
    df = pd.read_csv(tmp_path / "outputs" / "features" / "features.csv")
    df.iloc[:5].to_csv(
        tmp_path / "outputs" / "features" / "features.csv", index=False,
    )
    rc = ablation_main([
        "--config", str(cfg_path), "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    # Missing scenes -> caught earlier by the reorder step; should be 1
    assert rc == 1


def test_ablation_cli_reproducible(ablation_workspace):
    import json
    cfg_path, tmp_path, mapping_csv = ablation_workspace
    ablation_main([
        "--config", str(cfg_path), "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    with open(tmp_path / "outputs" / "stats" / "ablation_opa.json") as f:
        d1 = json.load(f)
    ablation_main([
        "--config", str(cfg_path), "--mapping", str(mapping_csv),
        "--n-perm", "30",
    ])
    with open(tmp_path / "outputs" / "stats" / "ablation_opa.json") as f:
        d2 = json.load(f)
    for f1, f2 in zip(d1["features"], d2["features"]):
        assert f1["feature"] == f2["feature"]
        assert f1["partial_r"] == f2["partial_r"]
        assert f1["partial_q"] == f2["partial_q"]