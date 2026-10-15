"""Tests for src.cli.rsa."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.distance import pdist, squareform
import scipy.io as sio

from src.cli.rsa import main as rsa_main


# ── Fixture: build a fake data directory ───────────────────────────────────────

@pytest.fixture
def synthetic_rsa_workspace(tmp_path, rng):
    """
    Create a minimal workspace for the RSA CLI:
      - rdms/{OPA,PPA,EVC,NAM}.mat each as a 50x50 symmetric RDM
      - outputs/rdms/sam.rdm.npy as a 50x50 SAM RDM
      - a config pointing at all of the above
    """
    n = 50
    shared = rng.normal(size=(n, 3))
    sam_signal = rng.normal(size=(n, 2))

    nam_rdm = squareform(pdist(shared))
    opa_rdm = squareform(pdist(np.column_stack([shared, sam_signal])))
    ppa_rdm = squareform(pdist(shared + 0.1 * rng.normal(size=(n, 3))))
    evc_rdm = squareform(pdist(rng.normal(size=(n, 3))))
    sam_rdm = squareform(pdist(sam_signal))

    rdms_dir = tmp_path / "rdms"
    rdms_dir.mkdir()
    for name, arr in [("OPA", opa_rdm), ("PPA", ppa_rdm),
                      ("EVC", evc_rdm), ("NAM", nam_rdm)]:
        sio.savemat(rdms_dir / f"{name}.mat", {"rdm": arr})

    outputs_rdms = tmp_path / "outputs" / "rdms"
    outputs_rdms.mkdir(parents=True)
    np.save(outputs_rdms / "sam.rdm.npy", sam_rdm.astype(np.float32))

    cfg = {
        "paths": {
            "stimuli_dir":     str(tmp_path / "stimuli"),
            "affordances_dir": str(tmp_path / "affordances"),
            "brain_rdms_dir":  str(rdms_dir),
            "mask_cache_dir":  str(tmp_path / "outputs" / "masks"),
            "features_csv":    str(tmp_path / "outputs" / "features.csv"),
            "sam_rdm":         str(outputs_rdms / "sam.rdm.npy"),
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
            "names": ["BPD","SC","GTI","NNE","PFF","MAC","BCR","SCE"],
            "topological": ["GTI","BPD","NNE","SC"],
            "clearance": ["MAC","PFF","BCR","SCE"],
        },
        "stats": {
            "n_stimuli": 50, "n_permutations": 100,
            "n_permutations_quick": 50,
            "fdr_method": "fdr_bh", "master_seed": 42,
        },
    }
    (tmp_path / "stimuli").mkdir()
    (tmp_path / "affordances").mkdir()
    (tmp_path / "configs").mkdir()
    cfg_path = tmp_path / "configs" / "test.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return cfg_path, tmp_path


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_cli_writes_json(synthetic_rsa_workspace):
    cfg_path, tmp_path = synthetic_rsa_workspace
    rc = rsa_main(["--config", str(cfg_path), "--n-perm", "50"])
    assert rc == 0
    json_path = tmp_path / "outputs" / "stats" / "rsa.json"
    assert json_path.exists()


def test_cli_json_has_expected_structure(synthetic_rsa_workspace):
    import json
    cfg_path, tmp_path = synthetic_rsa_workspace
    rsa_main(["--config", str(cfg_path), "--n-perm", "50"])
    with open(tmp_path / "outputs" / "stats" / "rsa.json") as f:
        d = json.load(f)
    assert set(d.keys()) >= {"standard", "partial", "conclusions"}
    assert len(d["standard"]) == 6
    assert len(d["partial"]) == 4


def test_cli_writes_figure(synthetic_rsa_workspace):
    cfg_path, tmp_path = synthetic_rsa_workspace
    rsa_main(["--config", str(cfg_path), "--n-perm", "50"])
    fig_path = tmp_path / "outputs" / "figures" / "rsa_summary.png"
    assert fig_path.exists()


def test_cli_missing_sam_rdm_returns_error(synthetic_rsa_workspace):
    cfg_path, tmp_path = synthetic_rsa_workspace
    (tmp_path / "outputs" / "rdms" / "sam.rdm.npy").unlink()
    rc = rsa_main(["--config", str(cfg_path), "--n-perm", "50"])
    assert rc == 1


def test_cli_shape_mismatch_returns_error(synthetic_rsa_workspace):
    cfg_path, tmp_path = synthetic_rsa_workspace
    # Save a 30x30 SAM RDM where a 50x50 is expected
    np.save(tmp_path / "outputs" / "rdms" / "sam.rdm.npy",
            np.zeros((30, 30), dtype=np.float32))
    rc = rsa_main(["--config", str(cfg_path), "--n-perm", "50"])
    assert rc == 1


def test_cli_reproducible(synthetic_rsa_workspace):
    import json
    cfg_path, tmp_path = synthetic_rsa_workspace
    rsa_main(["--config", str(cfg_path), "--n-perm", "50"])
    with open(tmp_path / "outputs" / "stats" / "rsa.json") as f:
        d1 = json.load(f)
    rsa_main(["--config", str(cfg_path), "--n-perm", "50"])
    with open(tmp_path / "outputs" / "stats" / "rsa.json") as f:
        d2 = json.load(f)
    for r1, r2 in zip(d1["standard"], d2["standard"]):
        assert r1["r"] == r2["r"]
        assert r1["p_perm"] == r2["p_perm"]