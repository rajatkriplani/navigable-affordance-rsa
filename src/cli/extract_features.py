"""
CLI: extract the 8-feature Gibsonian affordance vector from cached masks.

Reads one mask PNG per stimulus from the mask cache directory, runs
AffordanceExtractor, and writes a features CSV that Commit 6 (RSA) and
Commit 7 (ablation) consume.

Skips re-computation if the output CSV already lists an image, unless
--force is passed. This mirrors the resume behaviour of build_masks.

Usage:
    affordance-extract-features --config configs/default.yaml
    affordance-extract-features --force                      # recompute all
    affordance-extract-features --max-images 5               # smoke test
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..affordance import FEATURE_NAMES, AffordanceExtractor
from ..config import load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)


METADATA_COLUMNS = ["image_name", "floor_fraction", "sam_failed"]


# ── Mask IO ────────────────────────────────────────────────────────────────────

def _mask_path(cache_dir, image_name):
    return Path(cache_dir) / f"{Path(image_name).stem}.png"


def _load_mask(path):
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return arr > 127


def _stable_seed(image_name):
    """
    Deterministic per-image seed derived from the filename stem, so the
    GTI endpoint sampling is reproducible without depending on processing
    order. Uses a truncated SHA-256 rather than Python's hash() so it is
    stable across interpreter runs.
    """
    import hashlib
    return int(hashlib.sha256(image_name.encode()).hexdigest()[:8], 16)


# ── Metadata loader ────────────────────────────────────────────────────────────

def _load_masks_metadata(cache_dir):
    """
    Load masks_metadata.csv if present. Returns a dict {stem: {floor_fraction, sam_failed}}.
    Missing file is not an error — masks may exist without metadata.
    """
    meta_path = Path(cache_dir) / "masks_metadata.csv"
    if not meta_path.exists():
        return {}
    df = pd.read_csv(meta_path)
    out = {}
    for _, row in df.iterrows():
        stem = Path(row["image_name"]).stem
        out[stem] = {
            "floor_fraction": float(row.get("floor_fraction", np.nan)),
            "sam_failed": bool(row.get("sam_failed", False)),
        }
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--force", action="store_true",
                    help="Recompute features even if already present in CSV.")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Process at most N masks (for smoke testing).")
    ap.add_argument("--override", action="append", default=[],
                    metavar="key=value",
                    help="Dotted-key config override (repeatable).")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.override:
        if "=" not in kv:
            ap.error(f"--override must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        overrides[k] = v

    cfg = load_config(args.config, overrides=overrides)

    cache_dir = Path(cfg.paths.mask_cache_dir)
    features_csv = Path(cfg.paths.features_csv)

    mask_files = sorted(cache_dir.glob("*.png"))
    if args.max_images is not None:
        mask_files = mask_files[: args.max_images]

    if not mask_files:
        logger.error(
            "No masks found in %s. Run affordance-build-masks first.",
            cache_dir,
        )
        return 1

    metadata = _load_masks_metadata(cache_dir)

    # ── Resume: which masks still need feature extraction? ─────────────────
    existing = {}
    if features_csv.exists() and not args.force:
        try:
            existing_df = pd.read_csv(features_csv)
            for _, row in existing_df.iterrows():
                name = row["image_name"]
                existing[name] = row.to_dict()
        except Exception as e:
            logger.warning(
                "Could not read existing %s: %s. Recomputing.",
                features_csv, e,
            )
            existing = {}

    to_process = [p for p in mask_files
                  if Path(p).stem not in existing]

    logger.info(
        "Processing %d of %d masks (%d already in %s).",
        len(to_process), len(mask_files), len(mask_files) - len(to_process),
        features_csv.name,
    )

    extractor = AffordanceExtractor(
        r_base_px=cfg.features.r_base_px,
        horizon_ratio=cfg.features.horizon_ratio,
    )

    # ── Main loop ──────────────────────────────────────────────────────────
    new_rows = []
    pbar = tqdm(to_process, desc="Extracting", unit="mask")
    try:
        for mask_path in pbar:
            stem = mask_path.stem
            pbar.set_postfix_str(stem[:22])

            try:
                mask = _load_mask(mask_path)
            except FileNotFoundError as e:
                pbar.write(f"  ! skipping {mask_path.name}: {e}")
                continue

            seed = _stable_seed(stem)
            features = extractor.extract_features(mask, seed=seed)

            meta = metadata.get(stem, {})
            row = {"image_name": stem}
            row.update({k: features[k] for k in FEATURE_NAMES})
            row["floor_fraction"] = meta.get("floor_fraction", np.nan)
            row["sam_failed"] = meta.get("sam_failed", False)
            new_rows.append(row)

            if meta.get("sam_failed", False):
                pbar.write(
                    f"  ! {stem}: mask marked as SAM failure in metadata; "
                    f"features computed on empty mask (all zeros)."
                )
    except KeyboardInterrupt:
        pbar.write("Interrupted. Writing features for completed masks.")
    finally:
        pbar.close()

    # ── Merge with prior CSV and write ─────────────────────────────────────
    all_rows = {name: row for name, row in existing.items()}
    for row in new_rows:
        all_rows[row["image_name"]] = row

    df = pd.DataFrame(
        list(all_rows.values()),
        columns=["image_name"] + list(FEATURE_NAMES)
                     + ["floor_fraction", "sam_failed"],
    )
    df = df.sort_values("image_name").reset_index(drop=True)

    features_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_csv, index=False)

    n_sam_failed = int(df["sam_failed"].sum()) if "sam_failed" in df else 0
    logger.info(
        "Done. %d feature rows written to %s. %d SAM failures.",
        len(df), features_csv, n_sam_failed,
    )
    if n_sam_failed:
        logger.warning(
            "%d images had SAM-failed masks and now have all-zero feature "
            "vectors. These will be excluded from the RDM in build_rdm.",
            n_sam_failed,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())