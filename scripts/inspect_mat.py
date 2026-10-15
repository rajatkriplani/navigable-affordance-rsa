"""
Inspect a MATLAB .mat file containing human-drawn trajectory data.

Handles:
  - MATLAB v5/v7  (via scipy.io.loadmat)
  - MATLAB v7.3   (via h5py, HDF5)

Recursively walks nested structs and cell arrays, printing for each leaf:
  - field path, type, shape, dtype
  - numeric: min / max / mean / NaN count / sample values
  - string/char: first few characters
  - object/ref: dereferences HDF5 references

No modifications are made to the file. Read-only inspection.

Usage:
    python inspect_trajs.py Trajs.mat
    python inspect_trajs.py Trajs.mat --max-depth 5 --max-items 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ── Format detection ───────────────────────────────────────────────────────────

def detect_format(path):
    """Return 'v7.3' (HDF5) or 'v7' (MATLAB classic)."""
    with open(path, "rb") as f:
        header = f.read(128)
    # MATLAB v7.3 files start with the HDF5 magic bytes at offset 0
    if header.startswith(b"\x89HDF\r\n\x1a\n"):
        return "v7.3"
    return "v7"


# ── v7 (scipy.io) inspector ────────────────────────────────────────────────────

def inspect_v7(path, max_depth, max_items):
    import scipy.io as sio

    print("=" * 78)
    print("FORMAT: MATLAB v5/v7 (scipy.io.loadmat)")
    print("=" * 78)

    # whosmat gives us top-level variable names without loading everything
    print("\n--- scipy.io.whosmat ---")
    try:
        info = sio.whosmat(path)
        if not info:
            print("  (no variables reported)")
        for name, shape, cls in info:
            print(f"  {name:<40s}  shape={str(shape):<20s}  class={cls}")
    except Exception as e:
        print(f"  whosmat failed: {e}")

    print("\n--- Full load ---")
    try:
        mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError as e:
        print(f"  scipy.io cannot load this file: {e}")
        print("  Falling back to HDF5 reader …")
        return inspect_v73(path, max_depth, max_items)

    for key, val in mat.items():
        if key.startswith("__"):
            continue
        print(f"\n  Top-level variable: {key}")
        _walk_scipy(val, indent=4, depth=1, max_depth=max_depth,
                    max_items=max_items)


def _walk_scipy(obj, indent, depth, max_depth, max_items):
    pad = " " * indent
    if depth > max_depth:
        print(f"{pad}… (max depth reached)")
        return

    # Struct-like: mat_struct instance
    if hasattr(obj, "_fieldnames"):
        print(f"{pad}struct with fields: {obj._fieldnames}")
        for fname in obj._fieldnames:
            print(f"{pad}  .{fname}:")
            _walk_scipy(getattr(obj, fname), indent + 4, depth + 1,
                        max_depth, max_items)
        return

    # Numeric / bool arrays
    if isinstance(obj, np.ndarray):
        _print_array_summary(obj, pad, max_items)
        # Recurse into object arrays (cell arrays materialised as ndarray of objects)
        if obj.dtype == object:
            flat = obj.ravel()
            for i, item in enumerate(flat[:max_items]):
                print(f"{pad}  [{i}] ->")
                _walk_scipy(item, indent + 6, depth + 1, max_depth, max_items)
            if flat.size > max_items:
                print(f"{pad}  … and {flat.size - max_items} more items")
        return

    # Scalar / string
    print(f"{pad}{type(obj).__name__}: {repr(obj)[:200]}")


def _print_array_summary(arr, pad, max_items):
    shape = arr.shape
    dtype = arr.dtype
    print(f"{pad}ndarray  shape={shape}  dtype={dtype}  size={arr.size}")

    if arr.size == 0:
        print(f"{pad}  (empty)")
        return

    if np.issubdtype(dtype, np.number):
        try:
            finite = arr[np.isfinite(arr)] if np.issubdtype(dtype, np.floating) else arr
            if finite.size > 0:
                print(f"{pad}  min={finite.min():.6g}  max={finite.max():.6g}  "
                      f"mean={finite.mean():.6g}")
            if np.issubdtype(dtype, np.floating):
                n_nan = int(np.isnan(arr).sum())
                if n_nan:
                    print(f"{pad}  NaN count: {n_nan}")
        except Exception as e:
            print(f"{pad}  (stat computation failed: {e})")

        # Show a small sample, respecting shape
        try:
            flat = arr.ravel()
            n = min(max_items, flat.size)
            print(f"{pad}  first {n} values: {flat[:n].tolist()}")
        except Exception:
            pass

    elif dtype.kind in ("U", "S"):
        flat = arr.ravel()
        n = min(max_items, flat.size)
        print(f"{pad}  first {n} strings: {[str(x) for x in flat[:n]]}")

    elif dtype == object:
        print(f"{pad}  (object array — recursing)")


# ── v7.3 (h5py) inspector ──────────────────────────────────────────────────────

def inspect_v73(path, max_depth, max_items):
    import h5py

    print("=" * 78)
    print("FORMAT: MATLAB v7.3 (HDF5 via h5py)")
    print("=" * 78)

    def visit(name, obj):
        depth = name.count("/")
        if depth > max_depth:
            return
        pad = "  " * (depth + 1)

        if isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{pad}{name}  DATASET  shape={shape}  dtype={dtype}")

            # Try to read small enough arrays
            if obj.size is not None and obj.size <= 1_000_000:
                try:
                    data = obj[()]
                    if isinstance(data, np.ndarray) and data.size > 0:
                        if np.issubdtype(data.dtype, np.number):
                            print(f"{pad}  min={np.nanmin(data):.6g}  "
                                  f"max={np.nanmax(data):.6g}  "
                                  f"mean={np.nanmean(data):.6g}")
                            flat = data.ravel()
                            n = min(max_items, flat.size)
                            print(f"{pad}  first {n} values: "
                                  f"{flat[:n].tolist()}")
                        elif data.dtype.kind in ("U", "S", "O"):
                            flat = data.ravel()
                            n = min(max_items, flat.size)
                            print(f"{pad}  first {n} values: "
                                  f"{[repr(x)[:80] for x in flat[:n]]}")
                except Exception as e:
                    print(f"{pad}  (read failed: {e})")

        elif isinstance(obj, h5py.Group):
            print(f"{pad}{name}  GROUP  attrs={dict(obj.attrs)}")

    with h5py.File(path, "r") as f:
        print(f"\nRoot attributes: {dict(f.attrs)}")
        print("\nStructure:")
        f.visititems(visit)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matfile", type=Path, help="Path to Trajs.mat")
    ap.add_argument("--max-depth", type=int, default=4,
                    help="Recursion depth limit (default: 4)")
    ap.add_argument("--max-items", type=int, default=10,
                    help="Max array elements to print per leaf (default: 10)")
    args = ap.parse_args()

    if not args.matfile.exists():
        print(f"ERROR: file not found: {args.matfile}", file=sys.stderr)
        sys.exit(1)

    size_mb = args.matfile.stat().st_size / (1024 * 1024)
    print(f"File:  {args.matfile}")
    print(f"Size:  {size_mb:.2f} MB")

    fmt = detect_format(args.matfile)
    print(f"Format detected: {fmt}\n")

    if fmt == "v7.3":
        inspect_v73(args.matfile, args.max_depth, args.max_items)
    else:
        inspect_v7(args.matfile, args.max_depth, args.max_items)


if __name__ == "__main__":
    main()