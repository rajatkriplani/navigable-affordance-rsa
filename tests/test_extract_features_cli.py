"""Tests for src.cli.extract_features."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.affordance import FEATURE_NAMES
from src.cli.extract_features import _stable_seed, main as cli_main


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_config(tmp_path, mask_cache_dir, features_csv):
    """Write a minimal config that points to the synthetic fixture paths."""
    cfg = {
        "paths": {
            "stimuli_dir":    str(tmp_path / "stimuli"),
            "affordances_dir": str(tmp_path / "affordances"),
            "brain_rdms_dir": str(tmp_path / "rdms"),
            "mask_cache_dir": str(mask_cache_dir),
            "features_csv":   str(features_csv),
            "sam_rdm":        str(tmp_path / "out.rdm.npy"),
            "stats_dir":      str(tmp_path / "stats"),
            "figures_dir":    str(tmp_path / "figs"),
        },
        "sam": {
            "checkpoint":       str(tmp_path / "fake.pth"),
            "model_type":       "vit_b",
            "device":           "cpu",
            "primary_prompt":   [0.5, 0.9],
            "fallback_prompts": [[0.5, 0.85]],
            "min_floor_fraction": 0.02,
        },
        "features": {
            "r_base_px":      60.0,
            "horizon_ratio":  0.5,
            "names":          list(FEATURE_NAMES),
            "topological":    ["GTI", "BPD", "NNE", "SC"],
            "clearance":      ["MAC", "PFF", "BCR", "SCE"],
        },
        "stats": {
            "n_stimuli": 50,
            "n_permutations": 100,
            "n_permutations_quick": 50,
            "fdr_method": "fdr_bh",
            "master_seed": 42,
        },
    }
    # Create parent dirs so load_config's Path resolution succeeds
    (tmp_path / "stimuli").mkdir(exist_ok=True)
    (tmp_path / "affordances").mkdir(exist_ok=True)
    (tmp_path / "rdms").mkdir(exist_ok=True)
    (tmp_path / "configs").mkdir(exist_ok=True)
    cfg_path = tmp_path / "configs" / "test.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg_path


# ── _stable_seed ──────────────────────────────────────────────────────────────

def test_stable_seed_is_deterministic():
    a = _stable_seed("pathways0001")
    b = _stable_seed("pathways0001")
    assert a == b


def test_stable_seed_differs_across_names():
    a = _stable_seed("pathways0001")
    b = _stable_seed("pathways0002")
    assert a != b


def test_stable_seed_fits_in_32_bits():
    for name in ("pathways0001", "pathways0173", "pathways9999"):
        s = _stable_seed(name)
        assert 0 <= s < 2 ** 32


# ── CLI end-to-end on synthetic cache ────────────────────────────────────────

def test_cli_produces_features_csv(tmp_path, synthetic_mask_cache):
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    rc = cli_main(["--config", str(cfg_path)])
    assert rc == 0
    assert features_csv.exists()

    df = pd.read_csv(features_csv)
    assert len(df) == 4
    assert set(FEATURE_NAMES).issubset(df.columns)
    assert "image_name" in df.columns
    assert "floor_fraction" in df.columns
    assert "sam_failed" in df.columns


def test_cli_sorts_by_image_name(tmp_path, synthetic_mask_cache):
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    cli_main(["--config", str(cfg_path)])
    df = pd.read_csv(features_csv)
    assert df["image_name"].tolist() == sorted(df["image_name"].tolist())


def test_cli_sam_failed_row_is_zero_features(tmp_path, synthetic_mask_cache):
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    cli_main(["--config", str(cfg_path)])
    df = pd.read_csv(features_csv).set_index("image_name")

    # pathways0004 had an empty mask -> all features zero
    row = df.loc["pathways0004"]
    for k in FEATURE_NAMES:
        assert row[k] == 0.0, f"{k} should be 0 on empty mask"
    assert bool(row["sam_failed"]) is True


def test_cli_metadata_flows_through(tmp_path, synthetic_mask_cache):
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    cli_main(["--config", str(cfg_path)])
    df = pd.read_csv(features_csv).set_index("image_name")

    # floor_fraction should come from masks_metadata.csv
    assert df.loc["pathways0001", "floor_fraction"] == pytest.approx(0.075)
    assert df.loc["pathways0004", "floor_fraction"] == pytest.approx(0.0)


def test_cli_reproducible_across_runs(tmp_path, synthetic_mask_cache):
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    cli_main(["--config", str(cfg_path)])
    df1 = pd.read_csv(features_csv).set_index("image_name")

    cli_main(["--config", str(cfg_path), "--force"])
    df2 = pd.read_csv(features_csv).set_index("image_name")

    for k in FEATURE_NAMES:
        assert np.allclose(df1[k].values, df2[k].values), (
            f"{k} not reproducible between runs"
        )


def test_cli_resume_skips_existing(tmp_path, synthetic_mask_cache):
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    # First run
    cli_main(["--config", str(cfg_path)])
    df1 = pd.read_csv(features_csv)

    # Second run without --force should skip all 4
    cli_main(["--config", str(cfg_path)])
    df2 = pd.read_csv(features_csv)
    assert df1.equals(df2)


def test_cli_max_images(tmp_path, synthetic_mask_cache):
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    cli_main(["--config", str(cfg_path), "--max-images", "2"])
    df = pd.read_csv(features_csv)
    assert len(df) == 2


def test_cli_force_overwrites(tmp_path, synthetic_mask_cache):
    """If CSV exists with stale values, --force should recompute."""
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, synthetic_mask_cache, features_csv)

    # Write a CSV with deliberately wrong values
    stale = pd.DataFrame({
        "image_name": ["pathways0001", "pathways0002"],
        **{k: [999.0, 999.0] for k in FEATURE_NAMES},
        "floor_fraction": [0.0, 0.0],
        "sam_failed": [False, False],
    })
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    stale.to_csv(features_csv, index=False)

    # Without --force: run should skip both and preserve stale values
    cli_main(["--config", str(cfg_path)])
    df_skip = pd.read_csv(features_csv).set_index("image_name")
    assert df_skip.loc["pathways0001", "BPD"] == 999.0

    # With --force: recompute and overwrite
    cli_main(["--config", str(cfg_path), "--force"])
    df_forced = pd.read_csv(features_csv).set_index("image_name")
    assert df_forced.loc["pathways0001", "BPD"] != 999.0


def test_cli_no_masks_returns_error(tmp_path, monkeypatch):
    empty_cache = tmp_path / "empty_masks"
    empty_cache.mkdir()
    features_csv = tmp_path / "features.csv"
    cfg_path = _write_config(tmp_path, empty_cache, features_csv)

    rc = cli_main(["--config", str(cfg_path)])
    assert rc == 1