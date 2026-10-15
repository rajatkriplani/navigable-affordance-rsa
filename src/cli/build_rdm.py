"""
CLI: build the SAM-derived RDM from the extracted features CSV.

Reads:
  - outputs/features/extracted_features.csv
  - allpaths_mapping.csv (scene_k -> pathways####)

Writes:
  - outputs/rdms/sam_topological_rdm.npy  (or cfg.paths.sam_rdm)

Ordering is critical
--------------------
The brain RDMs (OPA/PPA/EVC/NAM) are indexed by scene_k (0..49), the
position of each stimulus in Bonner & Epstein (2017) Exp. 2. The features
CSV is sorted alphabetically by image_name (pathways0010, pathways0011,
...), which is NOT the same order. The mapping CSV from the earlier
verification step (allpaths_mapping.csv, columns scene_k, trajs_field)
resolves this: it tells you which image_name corresponds to each scene_k.

If --mapping is omitted, the CLI checks for allpaths_mapping.csv in a
few standard locations. If none is found, it uses the features CSV order
verbatim and logs a loud warning.

Usage:
    affordance-build-rdm --config configs/default.yaml
    affordance-build-rdm --config configs/default.yaml \\
        --mapping allpaths_mapping.csv
    affordance-build-rdm --config configs/default.yaml --no-mapping
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..logging_utils import get_logger
from ..rdm import build_rdm_from_dataframe

logger = get_logger(__name__)


# Where to look for the mapping CSV if --mapping is not given
MAPPING_CANDIDATES = [
    "allpaths_mapping.csv",
    "mapping_binary_sum.csv",
    "data/affordances/allpaths_mapping.csv",
    "outputs/allpaths_mapping.csv",
]


# ── Ordering helper ───────────────────────────────────────────────────────────

def _reorder_to_scene_order(features_df, mapping_df):
    """
    Reorder features so that row k corresponds to scene_k in the brain RDMs.

    Parameters
    ----------
    features_df : pandas.DataFrame with 'image_name' column
    mapping_df : pandas.DataFrame with 'scene_k' and 'trajs_field' columns

    Returns
    -------
    reordered_df : pandas.DataFrame with image_name as the index, in scene_k order
    missing : list of scene_k values with no matching feature row
    """
    scene_to_image = dict(zip(
        mapping_df["scene_k"].astype(int),
        mapping_df["trajs_field"].astype(str),
    ))

    indexed = features_df.set_index("image_name")
    ordered_names = []
    missing = []
    for k in sorted(scene_to_image):
        name = scene_to_image[k]
        if name in indexed.index:
            ordered_names.append(name)
        else:
            missing.append(k)

    return indexed.loc[ordered_names], missing


def _find_mapping_file(explicit_path):
    """Return a Path to the mapping CSV, or None."""
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"--mapping not found: {p}")
        return p
    for candidate in MAPPING_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--mapping", type=Path, default=None,
                    help="Path to allpaths_mapping.csv. If omitted, searches "
                         "a few standard locations.")
    ap.add_argument("--no-mapping", action="store_true",
                    help="Skip reordering; use the features CSV order verbatim.")
    ap.add_argument("--metric", default="euclidean",
                    help="Distance metric (default: euclidean).")
    ap.add_argument("--override", action="append", default=[],
                    metavar="key=value")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.override:
        if "=" not in kv:
            ap.error(f"--override must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        overrides[k] = v
    cfg = load_config(args.config, overrides=overrides)

    features_csv = Path(cfg.paths.features_csv)
    if not features_csv.exists():
        logger.error(
            "Features CSV not found at %s. "
            "Run affordance-extract-features first.",
            features_csv,
        )
        return 1

    df = pd.read_csv(features_csv)
    logger.info("Loaded %d feature rows from %s", len(df), features_csv.name)

    # ── Ordering ──────────────────────────────────────────────────────────
    expected_n = cfg.stats.n_stimuli
    if args.no_mapping:
        logger.warning(
            "--no-mapping set: features will be used in their CSV order. "
            "This is only correct if the CSV was already scene-ordered."
        )
        ordered_df = df.set_index("image_name")
    else:
        mapping_path = _find_mapping_file(args.mapping)
        if mapping_path is None:
            logger.error(
                "No mapping CSV found in any of: %s. "
                "Run test_allpaths_mapping.py to produce one, or pass "
                "--mapping explicitly, or pass --no-mapping to skip "
                "(dangerous).",
                MAPPING_CANDIDATES,
            )
            return 1
        mapping_df = pd.read_csv(mapping_path)
        logger.info(
            "Using mapping from %s (%d scenes).",
            mapping_path, len(mapping_df),
        )
        ordered_df, missing = _reorder_to_scene_order(df, mapping_df)
        if missing:
            logger.error(
                "Features are missing for scene_k values: %s. "
                "Cannot build a scene-ordered RDM.",
                missing,
            )
            return 1
        if len(ordered_df) != expected_n:
            logger.warning(
                "Expected %d stimuli, reordered features have %d. "
                "RDM will be %dx%d.",
                expected_n, len(ordered_df),
                len(ordered_df), len(ordered_df),
            )

    # ── Build ─────────────────────────────────────────────────────────────
    try:
        result = build_rdm_from_dataframe(ordered_df, metric=args.metric)
    except ValueError as e:
        logger.error("RDM construction failed: %s", e)
        return 1

    logger.info(
        "Built %dx%d RDM (%d valid / %d total rows).",
        result.rdm.shape[0], result.rdm.shape[1],
        result.n_valid, result.n_total,
    )

    if np.any(np.isnan(result.rdm)):
        n_nan_pairs = int(np.isnan(result.rdm).sum())
        logger.warning(
            "%d RDM pairs are NaN (involving %d invalid rows). "
            "These will need to be excluded before RSA.",
            n_nan_pairs, result.n_total - result.n_valid,
        )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(cfg.paths.sam_rdm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, result.rdm)
    logger.info("Saved SAM RDM to %s", out_path)

    # ── Save sidecar: the ordering used, for audit ────────────────────────
    order_path = out_path.with_suffix(".order.csv")
    pd.DataFrame({
        "scene_k": list(range(len(ordered_df))),
        "image_name": ordered_df.index.tolist(),
    }).to_csv(order_path, index=False)
    logger.info("Saved stimulus ordering to %s", order_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())