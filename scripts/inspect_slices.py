"""
Inspect the 3rd dimension of a Trajs.mat pathway array.

Purpose: determine whether the 10-slice dimension corresponds to
independent raters (thin distinct curves from a common start) or
something else (temporal frames, aggregated maps, etc.).

Produces:
  - 3-panel figure: raw slices, log-scaled slices, RGB overlay
  - Printed diagnostics per slice: pixel count, centroid, max-y,
    start-point location
  - Printed pairwise overlap (Jaccard) across slices

Usage:
    python inspect_slices.py Trajs.mat
    python inspect_slices.py Trajs.mat --pathway pathways0161
    python inspect_slices.py Trajs.mat --pathway pathways0161 --out slices.png
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")   # headless-safe; change to "TkAgg" if you want a window
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Loading ────────────────────────────────────────────────────────────────────

def load_pathway(matfile, pathway_name):
    print(f"Loading {matfile} …")
    mat = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)
    Trajs = mat["Trajs"]

    if pathway_name not in Trajs._fieldnames:
        raise KeyError(
            f"'{pathway_name}' not found. Available fields: "
            f"{Trajs._fieldnames[:5]} … ({len(Trajs._fieldnames)} total)"
        )
    arr = np.asarray(getattr(Trajs, pathway_name))
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array (H, W, N); got shape {arr.shape}"
        )
    return arr


# ── Diagnostics ────────────────────────────────────────────────────────────────

def per_slice_stats(arr, threshold=0.5):
    """
    Return a list of dicts, one per slice, with:
      - nonzero pixel count (above threshold)
      - centroid (y, x) of mass
      - max y reached (deepest into the scene)
      - start point = mean (x) of pixels in the bottom 5 rows with mass
      - mass
    """
    H, W, N = arr.shape
    stats = []
    for n in range(N):
        sl = arr[:, :, n]
        mask = sl > threshold
        n_pix = int(mask.sum())
        mass = float(sl.sum())
        if n_pix == 0:
            stats.append(dict(idx=n, n_pix=0, mass=0.0,
                              centroid=None, max_y=None,
                              start_x=None, start_y=None))
            continue
        ys, xs = np.nonzero(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        max_y = int(ys.max())            # larger y = deeper into scene
        # Start point: mass in the bottom 5 rows (image coords)
        bottom_rows = slice(max(0, H - 5), H)
        bm = sl[bottom_rows, :] > threshold
        if bm.any():
            _, bxs = np.nonzero(bm)
            start_x = float(bxs.mean())
            start_y = H - 1
        else:
            # Fallback: use the highest-y pixels
            deepest = ys.max()
            band = slice(max(0, deepest - 4), deepest + 1)
            bm = sl[band, :] > threshold
            if bm.any():
                _, bxs = np.nonzero(bm)
                start_x = float(bxs.mean())
            else:
                start_x = None
            start_y = int(deepest)
        stats.append(dict(idx=n, n_pix=n_pix, mass=mass,
                          centroid=(cy, cx), max_y=max_y,
                          start_x=start_x, start_y=start_y))
    return stats


def pairwise_jaccard(arr, threshold=0.5):
    """Jaccard overlap of binary masks between every pair of slices."""
    _, _, N = arr.shape
    masks = [arr[:, :, n] > threshold for n in range(N)]
    rows = []
    for i, j in combinations(range(N), 2):
        mi, mj = masks[i], masks[j]
        inter = np.logical_and(mi, mj).sum()
        union = np.logical_or(mi, mj).sum()
        jac = float(inter) / float(union) if union > 0 else 0.0
        rows.append((i, j, jac))
    return rows


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_slices(arr, pathway_name, out_path):
    H, W, N = arr.shape
    fig, axes = plt.subplots(3, 10, figsize=(22, 8))

    # Row 1: raw per-slice
    for n in range(N):
        ax = axes[0, n]
        ax.imshow(arr[:, :, n], cmap="magma", aspect="auto",
                  vmin=0, vmax=max(1.0, arr[:, :, n].max()))
        ax.set_title(f"slice {n}", fontsize=9)
        ax.axis("off")

    # Row 2: log1p-scaled per-slice (reveals thin low-count features)
    for n in range(N):
        ax = axes[1, n]
        ax.imshow(np.log1p(arr[:, :, n]), cmap="magma", aspect="auto")
        ax.axis("off")

    # Row 3: overlay in RGB — first 3 non-empty slices get R, G, B
    axes[2, 0].imshow(arr[:, :, 0], cmap="magma", aspect="auto")
    axes[2, 0].set_title("overlay →", fontsize=9, loc="left")
    axes[2, 0].axis("off")
    # Build RGB from slices 0..2 (or whatever is available)
    rgb = np.zeros((H, W, 3), dtype=float)
    for c, n in enumerate(range(min(3, N))):
        sl = arr[:, :, n]
        rgb[:, :, c] = sl / max(1.0, sl.max())
    axes[2, 1].imshow(rgb, aspect="auto")
    axes[2, 1].set_title("RGB = slices 0,1,2", fontsize=9)
    axes[2, 1].axis("off")
    # Mean across slices as a separate panel
    axes[2, 2].imshow(arr.mean(axis=2), cmap="magma", aspect="auto")
    axes[2, 2].set_title("mean of all slices", fontsize=9)
    axes[2, 2].axis("off")
    # Empty the rest of the bottom row
    for n in range(3, N):
        axes[2, n].axis("off")

    fig.suptitle(f"{pathway_name}  —  array shape {arr.shape}  "
                 f"(H={H}, W={W}, N={N})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Figure saved: {out_path}")


# ── Interpretation helper ──────────────────────────────────────────────────────

def interpret(arr, stats, jac_rows):
    N = arr.shape[2]
    counts = [s["n_pix"] for s in stats]
    nonzero = [c for c in counts if c > 0]
    jacs = [j for _, _, j in jac_rows]

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if not nonzero:
        print("All slices are empty above threshold — nothing to interpret.")
        return

    mean_pix = float(np.mean(nonzero))
    max_pix = float(np.max(nonzero))
    mean_jac = float(np.mean(jacs)) if jacs else 0.0
    max_jac = float(np.max(jacs)) if jacs else 0.0

    print(f"\n  Pixels per slice: mean={mean_pix:.1f}  max={max_pix:.1f}")
    print(f"  Pairwise Jaccard: mean={mean_jac:.3f}  max={max_jac:.3f}")

    # Start-point coherence: how similar are the start x-coordinates?
    start_xs = [s["start_x"] for s in stats if s["start_x"] is not None]
    if len(start_xs) >= 2:
        sx_std = float(np.std(start_xs))
        W = arr.shape[1]
        sx_std_frac = sx_std / W
        print(f"  Start-x spread: std={sx_std:.1f}px "
              f"({sx_std_frac*100:.1f}% of width)")

    # Decision logic
    print("\n  → Reading:")
    if mean_pix < 1000 and max_jac < 0.4:
        print("     Slices look like INDEPENDENT THIN CURVES with low overlap.")
        print("     Strongly consistent with 10 raters drawing separate paths.")
    elif mean_pix < 1000 and max_jac >= 0.4:
        print("     Slices are thin but show HIGH overlap.")
        print("     Could be 10 raters who mostly agreed, OR 10 trials of")
        print("     the same rater tracing the same path. Ambiguous — look")
        print("     at the RGB overlay panel to see whether curves are")
        print("     coincident (single colour) or offset (mixed colours).")
    elif mean_pix >= 1000 and mean_jac < 0.3:
        print("     Slices are diffuse but low-overlap.")
        print("     Could be 10 separate heat maps (e.g. 10 groups of raters,")
        print("     or 10 sessions). Not per-rater drawings.")
    else:
        print("     Slices are diffuse and highly overlapping.")
        print("     Consistent with aggregated heat maps or temporal frames,")
        print("     NOT independent raters. Per-rater variance is unavailable.")

    if len(start_xs) >= 2 and sx_std_frac < 0.05:
        print("     Start points are tightly clustered near bottom-center —")
        print("     consistent with the paper's 'starting from the bottom")
        print("     center of the image' instruction.")
    elif len(start_xs) >= 2:
        print("     Start points are spread out — inconsistent with a common")
        print("     bottom-center starting instruction. Worth investigating.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("matfile", type=Path)
    ap.add_argument("--pathway", default="pathways0173",
                    help="Field name to inspect (default: pathways0173)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binarisation threshold for pixel counting")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "figures",
                    help="Output PNG path (default: slices_<pathway>.png)")
    args = ap.parse_args()

    if not args.matfile.exists():
        print(f"ERROR: file not found: {args.matfile}", file=sys.stderr)
        sys.exit(1)

    arr = load_pathway(args.matfile, args.pathway)
    H, W, N = arr.shape
    print(f"\nArray shape: ({H}, {W}, {N})")
    print(f"Value range: min={arr.min()}  max={arr.max()}")
    print(f"Sparsity: {(arr <= 0).mean()*100:.2f}% zero voxels\n")

    stats = per_slice_stats(arr, threshold=args.threshold)
    print("Per-slice diagnostics:")
    print(f"  {'idx':>3}  {'n_pix':>6}  {'mass':>7}  "
          f"{'centroid (y,x)':>18}  {'max_y':>6}  {'start_x':>8}")
    for s in stats:
        c = s["centroid"]
        cstr = f"({c[0]:.1f}, {c[1]:.1f})" if c else "—"
        print(f"  {s['idx']:>3}  {s['n_pix']:>6}  {s['mass']:>7.2f}  "
              f"{cstr:>18}  "
              f"{(s['max_y'] if s['max_y'] is not None else -1):>6}  "
              f"{(s['start_x'] if s['start_x'] is not None else float('nan')):>8.1f}")

    jac_rows = pairwise_jaccard(arr, threshold=args.threshold)
    print("\nPairwise Jaccard overlap (i, j, J):")
    for i, j, v in jac_rows:
        flag = "  ← high" if v > 0.4 else ""
        print(f"  ({i:>2}, {j:>2})  {v:.3f}{flag}")

    out_path = args.out or Path(f"slices_{args.pathway}.png")
    plot_slices(arr, args.pathway, out_path)

    interpret(arr, stats, jac_rows)


if __name__ == "__main__":
    main()