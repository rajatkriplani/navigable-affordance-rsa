"""
CLI: feature ablation via partial RSA against OPA, controlling for NAM + EVC.

Reads:
  - outputs/features/extracted_features.csv
  - allpaths_mapping.csv (scene_k -> pathways####)
  - data/rdms/OPA.mat, NAM.mat, EVC.mat

Writes:
  - outputs/stats/ablation.json    — per-feature results + interpretation
  - outputs/figures/ablation.png   — bar chart, features ordered by partial r

The features CSV is reordered via the mapping CSV so that row k of the
feature matrix corresponds to scene k of the brain RDMs. This is the same
scene-order discipline enforced by build_rdm.

Usage:
    affordance-ablation --config configs/default.yaml
    affordance-ablation --config configs/default.yaml --n-perm 1000
    affordance-ablation --config configs/default.yaml --brain PPA
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..ablation import (
    TOPOLOGICAL_FEATURES,
    CLEARANCE_FEATURES,
    TOPOLOGICAL_INVARIANTS,
    GEOMETRIC_FEATURES,
    METRIC_FEATURES,
    reorder_features_to_scene_order,
    run_ablation,
)
from ..config import load_config
from ..io import load_brain_rdm, save_json
from ..logging_utils import get_logger

logger = get_logger(__name__)


# Same candidate paths as the build_rdm CLI
MAPPING_CANDIDATES = [
    "allpaths_mapping.csv",
    "mapping_binary_sum.csv",
    "data/affordances/allpaths_mapping.csv",
    "outputs/allpaths_mapping.csv",
]


COVARIATES_BY_TARGET = {
    "OPA": ["NAM", "EVC"],   # NAM + EVC both controls
    "PPA": ["NAM", "EVC"],   # same
    "EVC": ["NAM"],           # only NAM — EVC is the target
}


def _find_mapping_file(explicit_path):
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


def _load_scene_ordered_features(cfg, mapping_path, no_mapping=False):
    """Load the features CSV and reorder to scene_k order."""
    features_csv = Path(cfg.paths.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {features_csv}. "
            f"Run affordance-extract-features first."
        )
    df = pd.read_csv(features_csv)

    if no_mapping:
        logger.warning(
            "--no-mapping set: using features CSV order verbatim. "
            "This is only correct if the CSV is already scene-ordered."
        )
        return df.set_index("image_name")

    mapping_df = pd.read_csv(mapping_path)
    logger.info("Using mapping from %s (%d scenes).",
                mapping_path, len(mapping_df))

    ordered_df, missing = reorder_features_to_scene_order(df, mapping_df)
    if missing:
        raise ValueError(
            f"Features missing for scene_k values: {missing}. "
            f"Cannot build scene-ordered feature matrix."
        )
    return ordered_df


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--brain", default="OPA",
                    choices=["OPA", "PPA", "EVC"],
                    help="Target brain ROI (default: OPA).")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="Path to allpaths_mapping.csv. If omitted, "
                         "searches standard locations.")
    ap.add_argument("--no-mapping", action="store_true",
                    help="Skip reordering; use features CSV order verbatim.")
    ap.add_argument("--n-perm", type=int, default=None,
                    help="Override stats.n_permutations.")
    ap.add_argument("--master-seed", type=int, default=None)
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

    n_perm = args.n_perm or cfg.stats.n_permutations
    master_seed = args.master_seed or cfg.stats.master_seed
    fdr_method = cfg.stats.fdr_method

    # ── Load features, scene-ordered ───────────────────────────────────────
    mapping_path = None
    if not args.no_mapping:
        mapping_path = _find_mapping_file(args.mapping)
        if mapping_path is None:
            logger.error(
                "No mapping CSV found in any of: %s. "
                "Run test_allpaths_mapping.py to produce one, or pass "
                "--mapping explicitly, or pass --no-mapping.",
                MAPPING_CANDIDATES,
            )
            return 1

    try:
        ordered_df = _load_scene_ordered_features(
            cfg, mapping_path, no_mapping=args.no_mapping,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    # ── Load brain and covariates ──────────────────────────────────────────
    rdms_dir = Path(cfg.paths.brain_rdms_dir)
    try:
        brain_rdm = load_brain_rdm(rdms_dir / f"{args.brain}.mat")
        nam_rdm = load_brain_rdm(rdms_dir / "NAM.mat")
        evc_rdm = load_brain_rdm(rdms_dir / "EVC.mat")
    except FileNotFoundError as e:
        logger.error("Missing RDM file: %s", e)
        return 1

    n_stim = brain_rdm.shape[0]
    if len(ordered_df) != n_stim:
        logger.error(
            "Feature rows (%d) do not match brain RDM stimuli (%d). "
            "The mapping CSV and the brain RDMs are misaligned.",
            len(ordered_df), n_stim,
        )
        return 1

    # ── Select only feature columns (drop metadata) ────────────────────────
    metadata_cols = {"floor_fraction", "sam_failed"}
    feature_cols = [c for c in ordered_df.columns
                    if c not in metadata_cols]
    if not feature_cols:
        logger.error("No feature columns found in the reordered CSV.")
        return 1
    features_matrix = ordered_df[feature_cols]

    logger.info(
        "Running ablation: %d features, target=%s, controls=[NAM, EVC], "
        "n_perm=%d",
        len(feature_cols), args.brain, n_perm,
    )

    # ── Run ────────────────────────────────────────────────────────────────
    result = run_ablation(
        features_df=features_matrix,
        brain_rdm=brain_rdm,
        covar_rdms=[nam_rdm, evc_rdm],
        feature_names=feature_cols,
        n_perm=n_perm,
        master_seed=master_seed,
        fdr_method=fdr_method,
    )

    # ── Save JSON ──────────────────────────────────────────────────────────
    stats_dir = Path(cfg.paths.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    json_path = stats_dir / f"ablation_{args.brain.lower()}.json"
    save_json(result.as_dict(), json_path)
    logger.info("Saved ablation results to %s", json_path)

    # ── Console summary ────────────────────────────────────────────────────
    _print_summary(result, args.brain)

    # ── Figure ─────────────────────────────────────────────────────────────
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / f"ablation_{args.brain.lower()}.png"
    try:
        _plot_summary(result, fig_path, args.brain)
        logger.info("Saved ablation figure to %s", fig_path)
    except Exception as e:
        logger.warning("Figure generation failed: %s", e)

    return 0


def _print_summary(result, brain_name):
    print()
    print("=" * 70)
    print(f"FEATURE ABLATION  —  target: {brain_name}  —  "
          f"controls: NAM + EVC")
    print("=" * 70)
    print(f"  {'Feature':<6}  {'Class':<22}  "
          f"{'Std r':>8}  {'Std p':>8}  "
          f"{'Partial r':>10}  {'q_FDR':>8}  sig")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  "
          f"{'-'*10}  {'-'*8}  ---")
    for f in result.features:
        sig = "✓" if f.significant else " "
        print(f"  {f.feature:<6}  {f.family:<10}  "
              f"{f.std_r:>+8.4f}  {f.std_p:>8.4f}  "
              f"{f.partial_r:>+10.4f}  {f.partial_q:>8.4f}  {sig}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    interp = result.interpretation
    print(f"  Verdict: {interp['verdict']}")
    print(f"  {interp['description']}")
    print()


def _plot_summary(result, out_path, brain_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    features = result.features
    labels = [f.feature for f in features]
    partial_rs = [f.partial_r for f in features]
    std_rs = [f.std_r for f in features]

    FAMILY_COLORS = {
       "Topological invariant": "#8e44ad",   # purple
       "Geometric":             "#e67e22",   # orange
       "Metric":                "#2980b9",   # blue
       "Unknown":               "#888888",
    }

    colors = [FAMILY_COLORS.get(f.family, "#888888") for f in features]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.barh(labels, partial_rs, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.6)
    for bar, f in zip(bars, features):
        if f.significant:
            ax.text(max(f.partial_r, 0) + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    "✓", va="center", fontsize=12)
    ax.set_xlabel("Partial Spearman r  (NAM + EVC controlled)")
    ax.set_title(f"Unique {brain_name} variance per feature")
    legend_handles = [
       Patch(color=FAMILY_COLORS["Metric"], alpha=0.85,
             label="Metric (PFF, MAC, BCR, SCE)"),
       Patch(color=FAMILY_COLORS["Geometric"], alpha=0.85,
             label="Geometric (BPD, GTI, NNE)"),
       Patch(color=FAMILY_COLORS["Topological invariant"], alpha=0.85,
             label="Topological invariant (SC)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    ax2 = axes[1]
    bars2 = ax2.barh(labels, std_rs, color=colors, alpha=0.6)
    ax2.axvline(0, color="black", linewidth=0.6)
    for bar, f in zip(bars2, features):
        if f.std_p < 0.05:
            ax2.text(max(f.std_r, 0) + 0.002,
                     bar.get_y() + bar.get_height() / 2,
                     "✓", va="center", fontsize=12)
    ax2.set_xlabel("Standard Spearman r  (no controls)")
    ax2.set_title("Standard RSA per feature")

    fig.suptitle(
        f"Feature Ablation — {brain_name}\n"
        f"Permutation-tested ({10000} perms), FDR-corrected",
        fontsize=11, weight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())