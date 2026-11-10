"""
Interpretive figures for the 8 affordance features.

Produces per-feature interpretation grids (synthetic + real examples),
distribution histograms, and a feature-feature correlation matrix.

All figures assume the pipeline has already run: masks exist in
outputs/masks/, features CSV exists in outputs/features/. Scene
photographs are used as background when available in data/stimuli/
and fall back to mask-only display otherwise.
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize

from ..affordance import (
    CLEARANCE_FEATURES,
    FEATURE_NAMES,
    TOPOLOGICAL_FEATURES,
)
from ..logging_utils import get_logger

logger = get_logger(__name__)


# ── Feature descriptions for titles ──────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    "BPD": ("Branch point density",
            "How often the navigable path network forks, normalised by skeleton length"),
    "SC":  ("Skeleton components",
            "How many disconnected navigable regions the scene contains"),
    "GTI": ("Graph tortuosity",
            "Mean ratio of graph path length to straight-line distance (1 = straight)"),
    "NNE": ("Endpoint span",
            "Max distance between skeleton endpoints, normalised by image diagonal"),
    "PFF": ("Passable floor fraction",
            "Fraction of the skeleton wide enough for a body to pass"),
    "MAC": ("Mean axial clearance",
            "Mean body-scaled room-to-body ratio along the skeleton"),
    "BCR": ("Worst-case clearance",
            "5th percentile of the body-scaled room-to-body ratio — the tightest passage"),
    "SCE": ("Clearance entropy",
            "Shannon entropy of the clearance histogram — heterogeneity of widths"),
}


def _invariance_class(feature):
    if feature == "SC":
        return "Topological invariant"
    if feature in {"BPD", "GTI", "NNE"}:
        return "Geometric"
    return "Metric"


# ── Synthetic examples ────────────────────────────────────────────────────────

def make_synthetic_examples():
    """
    Build 3 synthetic masks per feature, corresponding to low / mid / high
    values of that feature. Sized 200x200 to match typical mask aspect ratio
    while remaining cheap to display.

    Returns
    -------
    dict {feature: [(label, mask), (label, mask), (label, mask)]}
    """
    H = W = 200
    ex = {}

    # BPD: straight → T-junction → crossroads
    m = np.zeros((H, W), dtype=bool); m[20:180, 95:105] = True
    ex.setdefault("BPD", []).append(("straight (BPD = 0)", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 95:105] = True; m[100:110, 20:180] = True
    ex["BPD"].append(("one fork", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 95:105] = True; m[95:105, 20:180] = True; m[50:60, 20:180] = True
    ex["BPD"].append(("many forks", m))

    # SC: one region → two → three
    m = np.zeros((H, W), dtype=bool); m[40:160, 80:120] = True
    ex.setdefault("SC", []).append(("one region (SC = 1)", m))
    m = np.zeros((H, W), dtype=bool); m[40:160, 30:60] = True; m[40:160, 140:170] = True
    ex["SC"].append(("two regions", m))
    m = np.zeros((H, W), dtype=bool); m[40:160, 20:45] = True; m[40:160, 85:110] = True; m[40:160, 150:175] = True
    ex["SC"].append(("three regions", m))

    # GTI: straight → L → S
    m = np.zeros((H, W), dtype=bool); m[20:180, 95:105] = True
    ex.setdefault("GTI", []).append(("straight (GTI = 1)", m))
    m = np.zeros((H, W), dtype=bool); m[20:105, 20:30] = True; m[95:105, 20:180] = True
    ex["GTI"].append(("L-shaped", m))
    m = np.zeros((H, W), dtype=bool)
    m[20:60, 20:30] = True; m[50:60, 20:180] = True
    m[50:180, 170:180] = True; m[170:180, 20:180] = True; m[140:180, 20:30] = True
    ex["GTI"].append(("S-shaped", m))

    # NNE: compact → medium → long
    m = np.zeros((H, W), dtype=bool); m[80:120, 80:120] = True
    ex.setdefault("NNE", []).append(("compact", m))
    m = np.zeros((H, W), dtype=bool); m[40:160, 90:110] = True
    ex["NNE"].append(("medium", m))
    m = np.zeros((H, W), dtype=bool); m[10:190, 95:105] = True
    ex["NNE"].append(("long extent", m))

    # PFF: too narrow → borderline → wide
    m = np.zeros((H, W), dtype=bool); m[20:180, 95:100] = True
    ex.setdefault("PFF", []).append(("too narrow", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 90:110] = True
    ex["PFF"].append(("borderline", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 40:160] = True
    ex["PFF"].append(("wide open", m))

    # MAC: narrow → medium → wide
    m = np.zeros((H, W), dtype=bool); m[20:180, 93:107] = True
    ex.setdefault("MAC", []).append(("narrow", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 80:120] = True
    ex["MAC"].append(("medium", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 40:160] = True
    ex["MAC"].append(("wide", m))

    # BCR: severe pinch → mild pinch → no pinch
    m = np.zeros((H, W), dtype=bool); m[20:180, 80:120] = True; m[95:105, 80:97] = False; m[95:105, 103:120] = False
    ex.setdefault("BCR", []).append(("severe pinch", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 80:120] = True; m[95:105, 80:95] = False; m[95:105, 105:120] = False
    ex["BCR"].append(("mild pinch", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 80:120] = True
    ex["BCR"].append(("no pinch", m))

    # SCE: uniform → some variation → highly varied (trapezoid)
    m = np.zeros((H, W), dtype=bool); m[20:180, 40:160] = True
    ex.setdefault("SCE", []).append(("uniform", m))
    m = np.zeros((H, W), dtype=bool); m[20:180, 40:160] = True; m[95:105, 40:60] = False; m[95:105, 140:160] = False
    ex["SCE"].append(("some variation", m))
    m = np.zeros((H, W), dtype=bool)
    for y in range(20, 180):
        w = int(8 + 55 * (y - 20) / 160)
        m[y, 100 - w:100 + w] = True
    ex["SCE"].append(("highly varied", m))

    return ex


# ── Real examples ─────────────────────────────────────────────────────────────

def load_example_pool(mask_dir, stimuli_dir, features_df):
    """
    Load all masks (and photos where available) indexed by image name.
    Returns {name: {"mask": bool array, "photo": RGB array or None}}.
    """
    pool = {}
    for name in features_df.index:
        mask_path = Path(mask_dir) / f"{name}.png"
        if not mask_path.exists():
            continue
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        photo = None
        if stimuli_dir is not None:
            for ext in (".jpg", ".jpeg", ".png"):
                p = Path(stimuli_dir) / f"{name}{ext}"
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if img is not None:
                        photo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    break
        pool[name] = {"mask": m > 127, "photo": photo}
    return pool


def select_examples_for_feature(features_df, pool, feature, n=3):
    """
    Return (low_examples, high_examples) where each is a list of
    (name, mask, photo, value) sorted by feature value.
    """
    if feature not in features_df.columns:
        return [], []
    values = features_df[feature].dropna()
    # Restrict to images we actually have masks for
    available = values[values.index.isin(pool.keys())]
    if len(available) < 2 * n:
        return [], []
    low_names = available.nsmallest(n).index.tolist()
    high_names = available.nlargest(n).index.tolist()
    low = [(nm, pool[nm]["mask"], pool[nm]["photo"], features_df.loc[nm, feature]) for nm in low_names]
    high = [(nm, pool[nm]["mask"], pool[nm]["photo"], features_df.loc[nm, feature]) for nm in high_names]
    return low, high


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _scene_panel(ax, mask, photo, title):
    """Display a mask, overlaid on the photograph when available."""
    if photo is not None:
        h, w = photo.shape[:2]
        mask_resized = resize(mask.astype(float), (h, w), order=0, anti_aliasing=False)
        ax.imshow(photo)
        ax.imshow(mask_resized, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    else:
        ax.imshow(mask, cmap="Greys", aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _synthetic_panel(ax, mask, title):
    ax.imshow(mask, cmap="Greys", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ── Per-feature interpretation grid ───────────────────────────────────────────

def plot_feature_interpretation(feature, synthetic, real_low, real_high, out_path):
    """
    3x3 grid: row 1 = synthetic low/mid/high, row 2 = real lowest 3,
    row 3 = real highest 3. Title shows feature description + class.
    """
    short, long = FEATURE_DESCRIPTIONS[feature]
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.10)

    fig.suptitle(f"{feature} — {short}", fontsize=14, weight="bold", y=0.985)
    fig.text(0.5, 0.955, long, ha="center", fontsize=10, style="italic")
    fig.text(0.5, 0.935,
             f"Invariance class: {_invariance_class(feature)}",
             ha="center", fontsize=9, color="#555555")

    for i, (label, mask) in enumerate(synthetic[:3]):
        ax = fig.add_subplot(gs[0, i])
        _synthetic_panel(ax, mask, f"synthetic: {label}")

    row_labels = ["lowest 3 (real)", "highest 3 (real)"]
    for row, examples in enumerate([real_low, real_high], start=1):
        for i, (name, mask, photo, value) in enumerate(examples):
            ax = fig.add_subplot(gs[row, i])
            _scene_panel(ax, mask, photo, f"{name}\n{feature} = {value:.3f}")
        # Row label on the left
        fig.text(0.02, 0.62 - 0.31 * (row - 1), row_labels[row - 1],
                 rotation=90, va="center", fontsize=10, weight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Histograms ────────────────────────────────────────────────────────────────

def plot_histograms(features_df, out_path):
    """2x4 grid of feature histograms with mean/std/min/max annotations."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, feature in enumerate(FEATURE_NAMES):
        ax = axes[i]
        if feature not in features_df.columns:
            ax.text(0.5, 0.5, "not computed", ha="center", va="center")
            ax.axis("off")
            continue
        values = features_df[feature].dropna().values
        if values.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.axis("off")
            continue
        ax.hist(values, bins=15, color="#4878cf", alpha=0.85,
                edgecolor="black", linewidth=0.5)
        ax.axvline(values.mean(), color="#d65f5f", linestyle="--",
                   linewidth=1.2, label=f"mean = {values.mean():.3f}")
        ax.set_title(f"{feature}  ({_invariance_class(feature)})",
                     fontsize=11, weight="bold")
        ax.set_xlabel("value", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        # Annotation of summary stats
        stats = (f"n = {values.size}\n"
                 f"min  = {values.min():.3f}\n"
                 f"max  = {values.max():.3f}\n"
                 f"std  = {values.std():.3f}")
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                fontsize=7, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))

    fig.suptitle("Distribution of feature values across scenes",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Correlation matrix ────────────────────────────────────────────────────────

def plot_feature_correlations(features_df, out_path):
    """8x8 Spearman correlation matrix between features."""
    cols = [f for f in FEATURE_NAMES if f in features_df.columns]
    if len(cols) < 2:
        return
    corr = features_df[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    plt.colorbar(im, label="Spearman r")
    ax.set_title("Feature-feature correlations across 50 scenes\n"
                 "High off-diagonal values indicate redundant measurements",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)