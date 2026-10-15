"""
Verify that binCounts.mat rows correspond 1-to-1 with Trajs.mat fields,
in the same order.

Two independent tests:

  Test 1 (mass): binCounts row sums should correlate with Trajs field
                 masses (total non-zero pixel counts). Same underlying data
                 → same ordering → high correlation.

  Test 2 (reconstruction): Recompute angular histograms from allPaths
                 (the aggregated rater maps) using the paper's method
                 (180 angular bins from bottom-center), and correlate each
                 with binCounts rows. Expect 50 clean matches for the fMRI
                 scenes if ordering matches.

Usage:
    python verify_bincounts_ordering.py affordances/Trajs.mat \\
        affordances/binCounts.mat affordances/allPaths.mat
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_trajs_fields(path):
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    t = mat["Trajs"]
    return list(t._fieldnames), t


def aggregate_binary_sum(arr):
    return (arr > 0).sum(axis=2).astype(np.float32)


# ── Test 1: mass correlation ───────────────────────────────────────────────────

def test1_mass(trajs_fields, trajs, bin_counts):
    print("=" * 70)
    print("TEST 1 — Mass correlation (row sum vs. field mass)")
    print("=" * 70)
    field_masses = np.array([
        aggregate_binary_sum(np.asarray(getattr(trajs, f))).sum()
        for f in trajs_fields
    ])
    row_masses = bin_counts.sum(axis=1)
    n = min(len(field_masses), len(row_masses))
    r = np.corrcoef(field_masses[:n], row_masses[:n])[0, 1]
    print(f"\n  Trajs fields:      {len(trajs_fields)}")
    print(f"  binCounts rows:    {bin_counts.shape[0]}")
    print(f"  Common dimension:  {n}")
    print(f"  Pearson r (masses): {r:+.4f}")
    if r > 0.7:
        print("  → Row ordering is consistent with field ordering.")
    elif r > 0.3:
        print("  → Weak consistency. Ordering may differ or fields unrelated.")
    else:
        print("  → No consistency. binCounts is not derived from Trajs in this order.")
    return r


# ── Test 2: angular histogram reconstruction ──────────────────────────────────

def reconstruct_angular_histogram(map_2d, n_bins=180):
    """
    Reconstruct a 180-bin angular histogram from a 2D trajectory map.

    Convention (Bonner & Epstein, 2017, Fig. 3B):
        0°   = path points left
        90°  = path points forward (into the scene)
        180° = path points right
    Bin width = 180 / n_bins degrees (1° for n_bins=180).
    """
    H, W = map_2d.shape
    cx, cy = W / 2.0, H - 1  # bottom-center starting point

    ys, xs = np.nonzero(map_2d)
    if ys.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    vals = map_2d[ys, xs].astype(np.float32)

    dx = xs - cx
    dy = cy - ys                        # positive = upward (into scene)

    # arctan2(dy, dx): 0 rad = +x (right), π/2 = +y (forward)
    # We want: 0° = left, 90° = forward, 180° = right
    # → atan2(dx, dy) gives 0 rad at forward, ±π/2 at right/left
    angles_rad = np.arctan2(dx, dy)     # -π/2 (left) .. +π/2 (right)
    angles_deg = np.degrees(angles_rad) + 90.0   # 0 (left) .. 180 (right)

    # Clip safely in case any path went slightly behind the start
    angles_deg = np.clip(angles_deg, 0.0, 180.0 - 1e-6)

    bin_width = 180.0 / n_bins
    bin_idx = (angles_deg / bin_width).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    hist = np.zeros(n_bins, dtype=np.float32)
    np.add.at(hist, bin_idx, vals)
    return hist


def test2_reconstruction(allpaths, bin_counts):
    print("\n" + "=" * 70)
    print("TEST 2 — Angular histogram reconstruction from allPaths")
    print("=" * 70)

    n_scenes = allpaths.shape[2]
    hist_50 = np.stack(
        [reconstruct_angular_histogram(allpaths[:, :, k])
         for k in range(n_scenes)],
        axis=0,
    )

    # Correlate each reconstructed histogram with each binCounts row
    A = hist_50 - hist_50.mean(axis=1, keepdims=True)
    B = bin_counts - bin_counts.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    corr = A @ B.T  # (50, 173)

    best_idx = corr.argmax(axis=1)
    best_r = corr.max(axis=1)
    unique_winners = len(np.unique(best_idx))

    print(f"\n  Reconstructed histograms: {hist_50.shape}")
    print(f"  binCounts rows:            {bin_counts.shape}")
    print(f"  Unique best matches:       {unique_winners} / {n_scenes}")
    print(f"  Mean best r:               {best_r.mean():.4f}")
    print(f"  Min best r:                {best_r.min():.4f}")

    if unique_winners >= 40 and best_r.mean() > 0.7:
        print("  → binCounts rows identified for the 50 fMRI scenes.")
        print("    Row ordering matches the paper's L/C/R pipeline.")
    elif unique_winners >= 25:
        print("  → Partial match. Some scenes cleanly identified; others may")
        print("    use a different angle convention. Try flipping the sign of")
        print("    dx or dy in reconstruct_angular_histogram.")
    else:
        print("  → Reconstruction does not match binCounts.")
        print("    Either the bin convention differs (try other angle schemes)")
        print("    or binCounts comes from a different source.")

    return best_idx, best_r, corr


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trajs", type=Path)
    ap.add_argument("bincounts", type=Path)
    ap.add_argument("allpaths", type=Path)
    ap.add_argument("--out-csv", type=Path,
                    default=PROJECT_ROOT / "outputs" / "stats" / "bincounts_ordering.csv")
    args = ap.parse_args()

    fields, trajs = load_trajs_fields(args.trajs)
    bin_counts = np.asarray(sio.loadmat(args.bincounts, squeeze_me=True,
                                        struct_as_record=False)["binCounts"])
    allpaths = np.asarray(sio.loadmat(args.allpaths, squeeze_me=True,
                                      struct_as_record=False)["allPaths"])

    r_mass = test1_mass(fields, trajs, bin_counts)
    best_idx, best_r, corr = test2_reconstruction(allpaths, bin_counts)

    # Save
    with open(args.out_csv, "w") as f:
        f.write("scene_k,reconstructed_hist_index,best_r\n")
        for k in range(len(best_idx)):
            f.write(f"{k},{best_idx[k]},{best_r[k]:.6f}\n")
    print(f"\nReconstruction matches saved: {args.out_csv}")


if __name__ == "__main__":
    main()