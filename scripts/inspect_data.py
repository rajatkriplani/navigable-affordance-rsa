"""
Inspect all research data files for the navigable-affordance RSA project.

Handles:
  - .mat v5/v7 (scipy.io)
  - .mat v7.3 (h5py)
  - .npy (numpy, memory-mapped)

Reports per file: format, top-level variables, shape, dtype, range, sample.
Recurses into structs and cell arrays. Read-only.

Usage:
    python inspect_data.py affordances rdms
    python inspect_data.py affordances/allPaths.mat
    python inspect_data.py rdms/all_sub_slrdms.npy --max-depth 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Format helpers ─────────────────────────────────────────────────────────────

def is_hdf5(path):
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def human_size(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# ── .mat inspectors ────────────────────────────────────────────────────────────

def inspect_mat_v7(path, max_depth, max_items):
    import scipy.io as sio

    print(f"\n  whosmat:")
    try:
        for name, shape, cls in sio.whosmat(path):
            print(f"    {name:<30s}  shape={str(shape):<20s}  class={cls}")
    except Exception as e:
        print(f"    whosmat failed: {e}")

    print(f"\n  Loading …")
    try:
        mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        print("    scipy.io cannot load (likely v7.3). Trying HDF5 …")
        return inspect_mat_v73(path, max_depth, max_items)

    for key, val in mat.items():
        if key.startswith("__"):
            continue
        print(f"\n  ── {key} ──")
        _walk_v7(val, indent=4, depth=1, max_depth=max_depth,
                 max_items=max_items)


def _walk_v7(obj, indent, depth, max_depth, max_items):
    pad = " " * indent
    if depth > max_depth:
        print(f"{pad}… (max depth {max_depth} reached)")
        return

    if hasattr(obj, "_fieldnames"):
        print(f"{pad}struct, fields: {obj._fieldnames}")
        for fname in obj._fieldnames:
            print(f"{pad}  .{fname}:")
            _walk_v7(getattr(obj, fname), indent + 4, depth + 1,
                     max_depth, max_items)
        return

    if isinstance(obj, np.ndarray):
        _print_array(obj, pad, max_items)
        if obj.dtype == object:
            flat = obj.ravel()
            for i, item in enumerate(flat[:max_items]):
                print(f"{pad}  [{i}] →")
                _walk_v7(item, indent + 6, depth + 1, max_depth, max_items)
            if flat.size > max_items:
                print(f"{pad}  … and {flat.size - max_items} more")
        return

    print(f"{pad}{type(obj).__name__}: {repr(obj)[:200]}")


def inspect_mat_v73(path, max_depth, max_items):
    import h5py

    with h5py.File(path, "r") as f:
        print(f"\n  Root attrs: {dict(f.attrs)}")

        def visit(name, obj):
            depth = name.count("/")
            if depth > max_depth:
                return
            pad = "  " * (depth + 2)
            if isinstance(obj, h5py.Dataset):
                print(f"{pad}{name}  DATASET  shape={obj.shape}  "
                      f"dtype={obj.dtype}")
                if obj.size and obj.size <= 1_000_000:
                    try:
                        data = obj[()]
                        _print_array(data, pad + "  ", max_items)
                    except Exception as e:
                        print(f"{pad}  (read failed: {e})")
            elif isinstance(obj, h5py.Group):
                print(f"{pad}{name}  GROUP")

        f.visititems(visit)


def _print_array(arr, pad, max_items):
    if arr.size == 0:
        print(f"{pad}(empty)")
        return
    print(f"{pad}shape={arr.shape}  dtype={arr.dtype}  size={arr.size}")
    if np.issubdtype(arr.dtype, np.number):
        try:
            finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
            if finite.size:
                print(f"{pad}min={finite.min():.4g}  "
                      f"max={finite.max():.4g}  "
                      f"mean={finite.mean():.4g}")
            if arr.dtype.kind == "f":
                n_nan = int(np.isnan(arr).sum())
                if n_nan:
                    print(f"{pad}NaN count: {n_nan}")
        except Exception:
            pass
        flat = arr.ravel()
        n = min(max_items, flat.size)
        print(f"{pad}first {n}: {flat[:n].tolist()}")


# ── .npy inspector ─────────────────────────────────────────────────────────────

def inspect_npy(path, max_depth, max_items):
    """Memory-mapped inspection — does not load the full array into RAM."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    print(f"  shape={arr.shape}  dtype={arr.dtype}  "
          f"size={arr.size}  ({human_size(arr.nbytes)})")

    if np.issubdtype(arr.dtype, np.number):
        # Sample statistics on a subset to avoid loading huge arrays
        sample_size = min(arr.size, 100_000)
        flat = np.asarray(arr.ravel()[:sample_size])
        finite = flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat
        if finite.size:
            print(f"  sample min={finite.min():.4g}  "
                  f"max={finite.max():.4g}  "
                  f"mean={finite.mean():.4g}")
        if flat.dtype.kind == "f":
            n_nan = int(np.isnan(flat).sum())
            if n_nan:
                print(f"  NaN count (in sample): {n_nan}")
        print(f"  first {min(max_items, flat.size)}: "
              f"{flat[:max_items].tolist()}")

    # Provide guidance on what shape patterns mean
    if arr.ndim == 3 and arr.shape[0] == arr.shape[1]:
        print(f"  → looks like a per-subject RDM stack: "
              f"({arr.shape[0]}×{arr.shape[1]} stimuli, "
              f"{arr.shape[2]} subjects)")
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        print(f"  → looks like a group-mean RDM: "
              f"({arr.shape[0]}×{arr.shape[1]})")
    elif arr.ndim == 2 and arr.shape[0] == 50:
        print(f"  → looks like a 50-stimulus feature table: "
              f"({arr.shape[0]} images × {arr.shape[1]} features)")


# ── Dispatch ───────────────────────────────────────────────────────────────────

def inspect_file(path, max_depth, max_items):
    print("=" * 78)
    print(f"FILE:  {path}")
    print(f"SIZE:  {human_size(path.stat().st_size)}")
    suffix = path.suffix.lower()

    if suffix == ".mat":
        if is_hdf5(path):
            print("FORMAT: MATLAB v7.3 (HDF5)")
            inspect_mat_v73(path, max_depth, max_items)
        else:
            print("FORMAT: MATLAB v5/v7")
            inspect_mat_v7(path, max_depth, max_items)
    elif suffix == ".npy":
        print("FORMAT: numpy .npy")
        inspect_npy(path, max_depth, max_items)
    else:
        print(f"SKIP: unsupported extension '{suffix}'")


def discover_files(root):
    root = Path(root)
    if root.is_file():
        return [root]
    patterns = ("*.mat", "*.npy")
    found = []
    for pat in patterns:
        found.extend(sorted(root.rglob(pat)))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                    help="Files or directories to inspect")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-items", type=int, default=8)
    args = ap.parse_args()

    targets = []
    for p in args.paths:
        if not p.exists():
            print(f"WARNING: path does not exist: {p}", file=sys.stderr)
            continue
        targets.extend(discover_files(p))

    if not targets:
        print("No .mat or .npy files found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(targets)} file(s) to inspect.\n")
    for t in targets:
        try:
            inspect_file(t, args.max_depth, args.max_items)
        except Exception as e:
            print(f"\nERROR inspecting {t}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()