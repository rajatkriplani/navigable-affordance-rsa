"""
CLI: generate interpretive figures for the 8 affordance features.

Produces per-feature interpretation grids, a pipeline schematic, feature
histograms, and a feature-correlation matrix. Writes everything under
outputs/figures/features/.

Runs on CPU only — no SAM, no torch. Reads cached masks and features CSV.

Usage:
    affordance-visualize-features --config configs/default.yaml
    affordance-visualize-features --config configs/default.yaml --no-photos
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..affordance import FEATURE_NAMES
from ..config import load_config
from ..logging_utils import get_logger
from ..viz.features import (
    load_example_pool,
    make_synthetic_examples,
    plot_feature_correlations,
    plot_feature_interpretation,
    plot_histograms,
    select_examples_for_feature,
)
from ..viz.pipeline import plot_pipeline_schematic

logger = get_logger(__name__)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--no-photos", action="store_true",
                    help="Skip scene photographs; show masks alone. "
                         "Useful if data/stimuli/ is not populated.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    features_csv = Path(cfg.paths.features_csv)
    if not features_csv.exists():
        logger.error(
            "Features CSV not found: %s. Run affordance-extract-features first.",
            features_csv,
        )
        return 1

    features_df = pd.read_csv(features_csv).set_index("image_name")
    logger.info("Loaded features for %d scenes.", len(features_df))

    mask_dir = Path(cfg.paths.mask_cache_dir)
    if not mask_dir.exists():
        logger.error("Masks directory not found: %s.", mask_dir)
        return 1

    stimuli_dir = Path(cfg.paths.stimuli_dir)
    if args.no_photos or not stimuli_dir.exists():
        stimuli_dir = None
        logger.info("Scene photographs disabled or not found; showing masks only.")

    out_dir = Path(cfg.paths.figures_dir) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all masks (+ photos) once ────────────────────────────────
    logger.info("Loading example pool …")
    pool = load_example_pool(mask_dir, stimuli_dir, features_df)
    logger.info("Loaded %d masks%s.",
                len(pool),
                f" with photos for {sum(1 for v in pool.values() if v['photo'] is not None)}"
                if stimuli_dir is not None else "")

    if len(pool) == 0:
        logger.error("No masks found in %s.", mask_dir)
        return 1

    # ── 2. Per-feature interpretation grids ──────────────────────────────
    logger.info("Building per-feature interpretation grids …")
    synthetic = make_synthetic_examples()

    for feature in FEATURE_NAMES:
        if feature not in features_df.columns:
            logger.warning("Feature %s not in CSV; skipping.", feature)
            continue
        low, high = select_examples_for_feature(features_df, pool, feature, n=3)
        if not low or not high:
            logger.warning("Not enough examples for %s; skipping.", feature)
            continue

        out_path = out_dir / f"feature_interpretation_{feature.lower()}.png"
        plot_feature_interpretation(
            feature=feature,
            synthetic=synthetic[feature],
            real_low=low,
            real_high=high,
            out_path=out_path,
        )
        logger.info("  wrote %s", out_path.name)

    # ── 3. Pipeline schematic (uses a middle-value scene as example) ─────
    logger.info("Building pipeline schematic …")
    example_names = list(pool.keys())
    example_name = example_names[len(example_names) // 2]
    example = pool[example_name]
    plot_pipeline_schematic(
        photo=example["photo"],
        mask=example["mask"],
        features_dict=features_df.loc[example_name].to_dict(),
        out_path=out_dir / "pipeline_schematic.png",
    )
    logger.info("  wrote pipeline_schematic.png  (example: %s)", example_name)

    # ── 4. Histograms ────────────────────────────────────────────────────
    logger.info("Building feature histograms …")
    plot_histograms(features_df, out_dir / "feature_histograms.png")
    logger.info("  wrote feature_histograms.png")

    # ── 5. Feature correlations ──────────────────────────────────────────
    logger.info("Building feature correlation matrix …")
    plot_feature_correlations(features_df, out_dir / "feature_correlations.png")
    logger.info("  wrote feature_correlations.png")

    logger.info("All figures written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())