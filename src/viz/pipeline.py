"""
Schematic of the full pipeline, using real example thumbnails at each
stage so that readers can see what actually flows through.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from skimage.morphology import medial_axis
from skimage.transform import resize


def plot_pipeline_schematic(photo, mask, features_dict, out_path,
                            feature_names=("BPD", "SC", "GTI", "NNE",
                                           "PFF", "MAC", "BCR", "SCE")):
    """
    Draw a horizontal pipeline with six stages and real thumbnails.

    Parameters
    ----------
    photo : RGB array or None
    mask : 2D bool array
    features_dict : {feature: value}
    out_path : Path
    """
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # ── Stage definitions ────────────────────────────────────────────────
    # x position of each stage centre, stage label, thumbnail array or None
    S, D = medial_axis(mask, return_distance=True)

    stages = [
        (1.5,  "Input\nscene\nphotograph",    photo),
        (4.5,  "SAM\nsegmentation",            None),
        (7.5,  "Binary\nfloor mask",           mask),
        (10.5, "Medial axis\n(skeleton + dist)", S.astype(float)),
        (13.5, "8-dimensional\nfeature vector", None),
        (16.5, "RDM\n(50 × 50)",               None),
    ]

    # ── Draw boxes with thumbnails ───────────────────────────────────────
    for x, label, thumb in stages:
        box = FancyBboxPatch(
            (x - 0.95, 2.5), 1.9, 2.5,
            boxstyle="round,pad=0.08",
            edgecolor="#333333", facecolor="#f7f7f7", linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x, 2.15, label, ha="center", va="top",
                fontsize=10, weight="bold")

        if thumb is not None:
            inset = ax.inset_axes([x - 0.75, 3.0, 1.5, 1.7])
            if thumb.ndim == 3:
                inset.imshow(thumb)
            else:
                inset.imshow(thumb, cmap="Greys" if thumb.max() <= 1 else "magma")
            inset.axis("off")

    # ── Arrows between stages ────────────────────────────────────────────
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + 0.95
        x2 = stages[i + 1][0] - 0.95
        arrow = FancyArrowPatch(
            (x1, 3.75), (x2, 3.75),
            arrowstyle="-|>", mutation_scale=20,
            linewidth=1.4, color="#333333",
        )
        ax.add_patch(arrow)

    # ── Feature vector breakdown under stage 5 ───────────────────────────
    x_feat = 13.5
    ax.text(x_feat, 1.6,
            "Topological invariant: SC",
            ha="center", fontsize=9, family="monospace", color="#8e44ad",
            weight="bold")
    ax.text(x_feat, 1.25,
            "Geometric: BPD, GTI, NNE",
            ha="center", fontsize=9, family="monospace", color="#e67e22",
            weight="bold")
    ax.text(x_feat, 0.9,
            "Metric: PFF, MAC, BCR, SCE",
            ha="center", fontsize=9, family="monospace", color="#2980b9",
            weight="bold")

    # ── Example feature values under stage 5 ─────────────────────────────
    if features_dict:
        vals = "   ".join(f"{f}={features_dict.get(f, float('nan')):.2f}"
                         for f in ("MAC", "PFF", "NNE"))
        ax.text(x_feat, 0.55, f"example:  {vals}",
                ha="center", fontsize=8, family="monospace",
                color="#555555", style="italic")

    # ── Final caption ────────────────────────────────────────────────────
    ax.text(9.0, 6.6,
            "Pipeline: from scene photograph to navigable-affordance RDM",
            ha="center", fontsize=13, weight="bold")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)