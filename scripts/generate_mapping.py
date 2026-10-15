"""
Test the hypothesis that allPaths.mat is the rater-aggregated trajectory map
for the 50 fMRI scenes, derived from a subset of the 148 fields in Trajs.mat.

Method:
  1. For each of the 148 Trajs fields, aggregate across the 10 rater slices.
  2. For each of the 50 allPaths slices, find the best-matching Trajs aggregate
     by Pearson correlation of the flattened 2D arrays.
  3. Report the mapping, its uniqueness, and match quality.

Success criterion:
  - 50 distinct winners (each allPaths slice matched to a different Trajs field)
  - High correlations (mean > 0.9)
  - Meaningful margin over runner-up

Usage:
    python test_allpaths_mapping.py affordances/Trajs.mat affordances/allPaths.mat
    python test_allpaths_mapping.py affordances/Trajs.mat affordances/allPaths.mat --mode binary_sum
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAPPING_DIR = PROJECT_ROOT / "data" / "affordances"


# ── Loading ────────────────────────────────────────────────────────────────────

def load_trajs(path):
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    t = mat["Trajs"]
    fields = list(t._fieldnames)
    arrays = {f: np.asarray(getattr(t, f), dtype=np.float32) for f in fields}
    return fields, arrays


def load_allpaths(path):
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return np.asarray(mat["allPaths"], dtype=np.float32)


# ── Aggregation options ────────────────────────────────────────────────────────

def aggregate_trajs(arr, mode):
    """Aggregate a (H, W, N_raters) array across the rater dimension."""
    if mode == "sum":
        return arr.sum(axis=2)
    if mode == "mean":
        return arr.mean(axis=2)
    if mode == "max":
        return arr.max(axis=2)
    if mode == "binary_sum":
        return (arr > 0).sum(axis=2).astype(np.float32)
    raise ValueError(f"Unknown mode: {mode}")


# ── Correlation ────────────────────────────────────────────────────────────────

def correlation_matrix(A, B):
    """
    A: (n_a, P), B: (n_b, P). Returns (n_a, n_b) matrix of Pearson correlations.
    """
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    A -= A.mean(axis=1, keepdims=True)
    B -= B.mean(axis=1, keepdims=True)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    A_norm[A_norm == 0] = 1.0
    B_norm[B_norm == 0] = 1.0
    return (A @ B.T) / (A_norm * B_norm.T)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_test(trajs_path, allpaths_path, mode, out_csv, out_png):
    print(f"Loading {trajs_path} …")
    fields, arrays = load_trajs(trajs_path)
    print(f"  {len(fields)} Trajs fields, first: {fields[0]}, last: {fields[-1]}")

    print(f"Loading {allpaths_path} …")
    allpaths = load_allpaths(allpaths_path)
    print(f"  allPaths shape: {allpaths.shape}")

    H, W, n_scenes = allpaths.shape
    n_fields = len(fields)

    print(f"\nAggregating Trajs with mode='{mode}' …")
    trajs_aggs = np.stack(
        [aggregate_trajs(arrays[f], mode=mode).ravel() for f in fields],
        axis=0,
    )
    allpaths_flat = np.stack(
        [allpaths[:, :, k].ravel() for k in range(n_scenes)], axis=0
    )
    print(f"  trajs_aggs:   {trajs_aggs.shape}")
    print(f"  allpaths_flat: {allpaths_flat.shape}")

    print("\nComputing correlation matrix …")
    corr = correlation_matrix(trajs_aggs, allpaths_flat)  # (n_fields, n_scenes)

    # Best and second-best per scene
    best_idx = corr.argmax(axis=0)
    best_r = corr.max(axis=0)
    sorted_corr = np.sort(corr, axis=0)
    margin = sorted_corr[-1] - sorted_corr[-2]
    winners, counts = np.unique(best_idx, return_counts=True)
    n_unique = len(winners)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"MAPPING RESULTS  (aggregation mode = '{mode}')")
    print("=" * 70)
    print(f"\nUnique winners:       {n_unique} / {n_scenes}")
    print(f"Mean best r:          {best_r.mean():.4f}")
    print(f"Median best r:        {np.median(best_r):.4f}")
    print(f"Min / max best r:     {best_r.min():.4f} / {best_r.max():.4f}")
    print(f"Mean margin over #2:  {margin.mean():.4f}")
    print(f"Median margin:        {np.median(margin):.4f}")

    print(f"\nPer-scene mapping:")
    print(f"  {'scene_k':>7}  {'field':>15}  {'best_r':>8}  {'2nd_r':>8}  "
          f"{'margin':>8}")
    for k in range(n_scenes):
        order = np.argsort(corr[:, k])[::-1]
        top, second = order[0], order[1]
        print(f"  {k:>7}  {fields[top]:>15}  {corr[top, k]:>8.4f}  "
              f"{corr[second, k]:>8.4f}  {corr[top, k] - corr[second, k]:>8.4f}")

    # Save mapping to CSV
    with open(out_csv, "w") as f:
        f.write("scene_k,trajs_field,best_r,second_best_field,second_best_r,margin\n")
        for k in range(n_scenes):
            order = np.argsort(corr[:, k])[::-1]
            top, second = order[0], order[1]
            f.write(f"{k},{fields[top]},{corr[top, k]:.6f},"
                    f"{fields[second]},{corr[second, k]:.6f},"
                    f"{corr[top, k] - corr[second, k]:.6f}\n")
    print(f"\nMapping saved: {out_csv}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(best_r, bins=20, color="#4878cf", alpha=0.85)
    axes[0].set_xlabel("Best correlation per allPaths slice")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Best-match r  (mode='{mode}', "
                      f"unique winners = {n_unique}/{n_scenes})")

    axes[1].hist(margin, bins=20, color="#d65f5f", alpha=0.85)
    axes[1].set_xlabel("Best minus second-best correlation")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Margin over runner-up")

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Figure saved: {out_png}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if n_unique == n_scenes and best_r.mean() > 0.90:
        print("  ✓ HYPOTHESIS CONFIRMED")
        print("    50 distinct Trajs fields map 1-to-1 to the 50 allPaths slices.")
        print("    allPaths is the rater-aggregated trajectory map of the fMRI scenes.")
        print("    → Mapping saved. The pathway-to-RDM ordering is now known.")
    elif n_unique == n_scenes and best_r.mean() > 0.70:
        print("  ~ HYPOTHESIS LIKELY, moderate correlations")
        print("    All 50 scenes found a unique best match, but correlations")
        print("    are lower than ideal. Try other aggregation modes:")
        print("      --mode binary_sum   (if Trajs slices are binarisable)")
        print("      --mode mean         (if slices are already normalised)")
        print("    Then pick the mode with the highest mean best r.")
    elif n_unique < n_scenes:
        print("  ✗ HYPOTHESIS FAILS on uniqueness")
        print(f"    Only {n_unique} unique winners for {n_scenes} scenes.")
        print("    Some allPaths slices share the same best-matching Trajs field,")
        print("    which is inconsistent with a 1-to-1 mapping.")
        print("    Try alternative aggregation modes before concluding.")
    else:
        print("  ✗ HYPOTHESIS FAILS on correlation quality")
        print("    Correlations too low to support a derived mapping.")
    print("")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "trajs", type=Path,
        nargs="?",
        default=DEFAULT_MAPPING_DIR / "Trajs.mat",
        help="Path to Trajs.mat (default: data/affordances/Trajs.mat)",
    )
    ap.add_argument(
        "allpaths", type=Path,
        nargs="?",
        default=DEFAULT_MAPPING_DIR / "allPaths.mat",
        help="Path to allPaths.mat (default: data/affordances/allPaths.mat)",
    )
    ap.add_argument(
        "--mode", default="binary_sum",
        choices=["sum", "mean", "max", "binary_sum"],
        help="Aggregation mode (default: binary_sum, the algebraically exact one)",
    )
    ap.add_argument(
        "--out-csv", type=Path,
        default=DEFAULT_MAPPING_DIR / "allpaths_mapping.csv",
        help="Output CSV path (default: data/affordances/allpaths_mapping.csv)",
    )
    ap.add_argument(
        "--out-png", type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "allpaths_mapping.png",
        help="Output PNG path (default: outputs/figures/allpaths_mapping.png)",
    )
    args = ap.parse_args()

    # Ensure output directories exist
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)

    run_test(args.trajs, args.allpaths, args.mode, args.out_csv, args.out_png)


if __name__ == "__main__":
    main()