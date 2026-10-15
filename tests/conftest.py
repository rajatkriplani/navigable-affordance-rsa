"""Shared pytest fixtures. All data is synthetic — no external files."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
import pandas as pd


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_feature_matrix(rng):
    """(50, 8) feature matrix."""
    return rng.normal(size=(50, 8))


@pytest.fixture
def synthetic_rdm(rng):
    """(50, 50) symmetric distance RDM with zero diagonal."""
    x = rng.normal(size=(50, 5))
    return squareform(pdist(x))


@pytest.fixture
def synthetic_subject_rdms(rng):
    """(50, 50, 4) stack of per-subject RDMs around a shared base."""
    base = rng.normal(size=(50, 5))
    stack = np.empty((50, 50, 4))
    for s in range(4):
        x = base + rng.normal(scale=0.3, size=base.shape)
        stack[:, :, s] = squareform(pdist(x))
    return stack


@pytest.fixture
def tmp_mat_file(tmp_path, synthetic_rdm):
    """Synthetic .mat file with an 'rdm' variable."""
    import scipy.io as sio
    path = tmp_path / "test_rdm.mat"
    sio.savemat(path, {"rdm": synthetic_rdm})
    return path


# ── Synthetic Bonner-style .mat fixtures ───────────────────────────────────────

@pytest.fixture
def synthetic_trajs_mat(tmp_path, rng):
    """
    Mimic Trajs.mat: a struct under 'Trajs' with N fields of (H, W, R).
    Field names use the same pathways#### convention.
    """
    import scipy.io as sio

    H, W, R = 16, 16, 3
    n_scenes = 4
    fields = [f"pathways{i:04d}" for i in range(n_scenes)]
    struct = {
        f: rng.integers(0, 2, size=(H, W, R)).astype(np.float32)
        for f in fields
    }
    path = tmp_path / "Trajs.mat"
    sio.savemat(path, {"Trajs": struct})
    return path, fields, (H, W, R)


@pytest.fixture
def synthetic_allpaths_mat(tmp_path, rng):
    """Mimic allPaths.mat: (H, W, 50)."""
    import scipy.io as sio

    H, W = 16, 16
    arr = rng.integers(0, 10, size=(H, W, 50)).astype(np.float32)
    path = tmp_path / "allPaths.mat"
    sio.savemat(path, {"allPaths": arr})
    return path, arr


@pytest.fixture
def synthetic_bincounts_mat(tmp_path, rng):
    """Mimic binCounts.mat: (173, 180)."""
    import scipy.io as sio

    arr = rng.random((173, 180)).astype(np.float32) * 100
    path = tmp_path / "binCounts.mat"
    sio.savemat(path, {"binCounts": arr})
    return path, arr


@pytest.fixture
def synthetic_mapping_csv(tmp_path):
    """Mimic allpaths_mapping.csv."""
    import pandas as pd

    df = pd.DataFrame({
        "scene_k": [0, 1, 2, 3, 4],
        "trajs_field": [
            "pathways0010", "pathways0011", "pathways0014",
            "pathways0024", "pathways0028",
        ],
        "best_r": [0.99, 0.99, 0.99, 1.0, 0.99],
    })
    path = tmp_path / "mapping.csv"
    df.to_csv(path, index=False)
    return path, df


# ── Affordance-extractor fixtures ──────────────────────────────────────────────

@pytest.fixture
def dummy_floor_mask():
    """
    Reproduces the smoke-test mask from the original Phase 3 script.
    A corridor-shaped navigable region with one hole (a wall or obstacle).
    """
    mask = np.zeros((768, 1024), dtype=bool)
    mask[300:768, 400:600] = True   # main corridor
    mask[500:600, 500:700] = True   # side passage
    mask[600:650, 450:550] = False  # obstacle / wall segment
    return mask


@pytest.fixture
def empty_floor_mask():
    return np.zeros((128, 128), dtype=bool)


@pytest.fixture
def simple_corridor_mask():
    """A single straight vertical corridor. Skeleton is one line segment."""
    mask = np.zeros((200, 200), dtype=bool)
    mask[:, 95:105] = True
    return mask

# ── RDM fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_features_df(rng):
    """
    Mimic an extracted_features.csv: 50 rows, 8 features, 2 metadata cols.
    Feature scales are deliberately heterogeneous to exercise standardisation.
    """
    import pandas as pd

    n = 50
    df = pd.DataFrame({
        "image_name": [f"scene_{i:03d}.jpg" for i in range(n)],
        "floor_fraction": rng.uniform(0.1, 0.9, size=n),
        "BPD": rng.uniform(0.0, 0.02, size=n),
        "SC": rng.integers(1, 5, size=n).astype(float),
        "GTI": rng.uniform(1.0, 2.0, size=n),
        "NNE": rng.uniform(0.3, 0.9, size=n),
        "PFF": rng.uniform(0.5, 1.0, size=n),
        "MAC": rng.uniform(1.0, 5.0, size=n),
        "BCR": rng.uniform(0.5, 1.5, size=n),
        "SCE": rng.uniform(1.0, 3.0, size=n),
        "sam_failed": [False] * n,
    })
    return df


@pytest.fixture
def synthetic_features_df_with_failures(synthetic_features_df):
    """Same as above but with 3 rows NaN'd out, mimicking SAM failures."""
    df = synthetic_features_df.copy()
    feature_cols = [c for c in df.columns
                    if c not in ("image_name", "floor_fraction", "sam_failed")]
    for idx in (7, 22, 41):
        df.loc[idx, feature_cols] = np.nan
        df.loc[idx, "sam_failed"] = True
    return df


# ── SAM segmentation fixtures ──────────────────────────────────────────────────

@pytest.fixture
def tiny_rgb_image():
    """A 64x64 RGB image with a bright square on a dark background."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:54, 10:54] = 255
    return img


@pytest.fixture
def mock_predictor_cls():
    """
    A mock SamPredictor that returns a preset mask for every prompt.
    Usage:
        predictor = MockPredictor(masks_by_prompt={(0.5, 0.9): preset_mask})
        predictor.set_image(img)   # no-op
        masks, scores, _ = predictor.predict(...)
    """
    class _MockPredictor:
        def __init__(self, masks_by_prompt=None, default_mask=None):
            self.masks_by_prompt = masks_by_prompt or {}
            self.default_mask = default_mask
            self.image_set = False
            self.calls = []

        def set_image(self, image):
            self.image_set = True
            self.image_shape = image.shape

        def predict(self, point_coords, point_labels, multimask_output=True):
            # Recover the fractional prompt from the pixel coordinates
            H, W = self.image_shape[:2]
            x_frac = float(point_coords[0, 0]) / W
            y_frac = float(point_coords[0, 1]) / H
            self.calls.append((round(x_frac, 3), round(y_frac, 3)))

            # Nearest key match (rounded)
            key = min(
                self.masks_by_prompt.keys(),
                key=lambda k: abs(k[0] - x_frac) + abs(k[1] - y_frac),
            ) if self.masks_by_prompt else None
            mask = self.masks_by_prompt.get(key, self.default_mask)
            if mask is None:
                mask = np.zeros((H, W), dtype=bool)
            masks = np.stack([mask, mask, mask], axis=0)
            scores = np.array([0.9, 0.8, 0.7])
            return masks, scores, None

    return _MockPredictor


# ── Feature extraction CLI fixtures ───────────────────────────────────────────

@pytest.fixture
def synthetic_mask_cache(tmp_path):
    """
    Create a mask cache dir mimicking outputs/masks/ with 4 masks
    (some empty) and a masks_metadata.csv.
    """
    cache_dir = tmp_path / "masks"
    cache_dir.mkdir()

    # 3 valid masks with different geometry
    def write_mask(name, mask):
        import cv2
        cv2.imwrite(str(cache_dir / f"{name}.png"),
                    (mask.astype(np.uint8) * 255))

    corridor = np.zeros((128, 128), dtype=bool)
    corridor[20:110, 60:70] = True

    room = np.zeros((128, 128), dtype=bool)
    room[40:110, 20:110] = True

    two_rooms = np.zeros((128, 128), dtype=bool)
    two_rooms[40:110, 10:55] = True
    two_rooms[40:110, 75:120] = True

    write_mask("pathways0001", corridor)
    write_mask("pathways0002", room)
    write_mask("pathways0003", two_rooms)

    # 1 empty mask (SAM failure)
    write_mask("pathways0004", np.zeros((128, 128), dtype=bool))

    # Metadata
    import pandas as pd
    df = pd.DataFrame({
        "image_name": [
            "pathways0001.jpg", "pathways0002.jpg",
            "pathways0003.jpg", "pathways0004.jpg",
        ],
        "floor_fraction": [0.075, 0.48, 0.35, 0.0],
        "prompt_used": ["(0.5, 0.9)"] * 3 + ["(0.5, 0.85)"],
        "n_prompts_tried": [1, 1, 1, 6],
        "sam_failed": [False, False, False, True],
    })
    df.to_csv(cache_dir / "masks_metadata.csv", index=False)

    return cache_dir


# ── RSA fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_brain_model_rdms(rng):
    """
    A coherent set of RDMs: brain RDMs contain a signal from NAM (shared
    component) plus region-specific noise; SAM shares a small signal with
    OPA but not with PPA or EVC.
    """
    from scipy.spatial.distance import pdist, squareform

    n = 30
    # Latent structures
    shared = rng.normal(size=(n, 3))       # drives NAM and brain RDMs
    sam_specific = rng.normal(size=(n, 2)) # drives SAM and OPA-only

    nam_rdm = squareform(pdist(shared))

    opa_feat = np.column_stack([shared, sam_specific])
    ppa_feat = shared
    evc_feat = rng.normal(size=(n, 3))

    sam_rdm = squareform(pdist(sam_specific))
    opa_rdm = squareform(pdist(opa_feat))
    ppa_rdm = squareform(pdist(ppa_feat))
    evc_rdm = squareform(pdist(evc_feat))

    return {
        "model_rdms": {"nam": nam_rdm, "sam": sam_rdm},
        "brain_rdms": {"opa": opa_rdm, "ppa": ppa_rdm, "evc": evc_rdm},
    }


@pytest.fixture
def synthetic_subject_rdms_by_roi(rng):
    """Per-subject RDM stacks for OPA and PPA with 6 subjects."""
    from scipy.spatial.distance import pdist, squareform

    n, n_sub = 30, 6
    out = {}
    for roi in ("opa", "ppa"):
        base = rng.normal(size=(n, 4))
        stack = np.empty((n, n, n_sub))
        for s in range(n_sub):
            x = base + rng.normal(scale=0.3, size=base.shape)
            stack[:, :, s] = squareform(pdist(x))
        out[roi] = stack
    return out


# ── Ablation fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ablation_data(rng):
    """
    Constructed so that clearance features carry the OPA signal and
    topological features are noise. The ablation should find clearance
    significant and topology not.
    """
    from scipy.spatial.distance import pdist, squareform

    n = 30
    latent_brain = rng.normal(size=(n, 3))     # drives NAM and OPA
    latent_clear = rng.normal(size=(n, 2))     # drives clearance features
    latent_evc = rng.normal(size=(n, 2))       # drives EVC

    # Brain RDMs
    opa_feat = np.column_stack([latent_brain, latent_clear])
    brain_rdm = squareform(pdist(opa_feat))

    # Covariates: NAM and EVC
    nam_rdm = squareform(pdist(latent_brain))
    evc_rdm = squareform(pdist(latent_evc))

    # Features DataFrame
    # Clearance features correlate with the clearance signal
    # Topology features are pure noise
    df = pd.DataFrame({
        "image_name": [f"img_{i:03d}" for i in range(n)],
        "MAC": latent_clear[:, 0] + 0.2 * rng.normal(size=n),
        "PFF": latent_clear[:, 1] + 0.2 * rng.normal(size=n),
        "BCR": latent_clear[:, 0] + 0.3 * rng.normal(size=n),
        "SCE": latent_clear[:, 1] + 0.3 * rng.normal(size=n),
        "BPD": rng.normal(size=n),
        "SC":  rng.normal(size=n),
        "GTI": rng.normal(size=n),
        "NNE": rng.normal(size=n),
        "floor_fraction": rng.uniform(0.1, 0.9, size=n),
        "sam_failed": [False] * n,
    })

    return {
        "brain_rdm": brain_rdm,
        "covar_rdms": [nam_rdm, evc_rdm],
        "features_df": df,
        "feature_names": ["BPD", "SC", "GTI", "NNE",
                          "PFF", "MAC", "BCR", "SCE"],
    }