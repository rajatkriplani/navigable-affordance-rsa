"""
Gibsonian affordance feature extractor.

Given a binary floor mask (True = navigable), computes an 8-dimensional
feature vector describing the navigable geometry of a scene. Features fall
into two families:

  Topological (4):   BPD, SC, GTI, NNE
      Describe the skeleton graph of navigable space: branching, connected
      components, path tortuosity, endpoint span.

  Clearance (4):     PFF, MAC, BCR, SCE
      Describe the metric width and depth of navigable space, normalised
      by a body-scaled clearance radius and a perspective horizon.

Parameter provenance for r_base_px and horizon_ratio is documented in
configs/default.yaml and docs/pipeline.md (Sections 4-5).

Reproducibility:
    extract_features accepts an optional rng parameter. When omitted, a
    fresh numpy Generator is constructed from the given seed. Global
    numpy random state is never touched.
"""

import numpy as np
import networkx as nx
from scipy.ndimage import convolve, label
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from skimage import morphology

from .logging_utils import get_logger

logger = get_logger(__name__)


FEATURE_NAMES = ("BPD", "SC", "GTI", "NNE", "PFF", "MAC", "BCR", "SCE")

TOPOLOGICAL_FEATURES = frozenset({"BPD", "SC", "GTI", "NNE"})
CLEARANCE_FEATURES = frozenset({"PFF", "MAC", "BCR", "SCE"})


class AffordanceExtractor:
    """
    Extract Gibsonian affordance features from navigable-floor masks.

    Parameters
    ----------
    r_base_px : float, default 60.0
        Body clearance radius at the bottom of the image, in PIXELS (not cm).
        Calibrated to the Bonner & Epstein (2017) stimulus resolution
        (768 x 1024). See configs/default.yaml for provenance.
    horizon_ratio : float, default 0.5
        Fraction of image height at which the visual horizon sits. 0.5 is
        derived from eye-level acquisition geometry + pinhole projection.
    """

    def __init__(self, r_base_px=60.0, horizon_ratio=0.5):
        if r_base_px <= 0:
            raise ValueError(f"r_base_px must be positive, got {r_base_px}")
        if not (0.0 < horizon_ratio < 1.0):
            raise ValueError(
                f"horizon_ratio must be in (0, 1), got {horizon_ratio}"
            )
        self.r_base_px = float(r_base_px)
        self.horizon_ratio = float(horizon_ratio)

    # ── Perspective clearance ──────────────────────────────────────────────

    def _get_perspective_clearance(self, D, H):
        """
        Body-scaled clearance: distance-to-wall divided by a body radius
        that grows with vertical distance from the horizon (rows below the
        horizon get a larger body footprint).
        """
        y_horiz = H * self.horizon_ratio
        y_coords = np.arange(H).reshape(H, 1)
        r_body_y = self.r_base_px * (y_coords - y_horiz) / (H - y_horiz)
        r_body_y = np.clip(r_body_y, 1.0, None)
        r_body_map = np.broadcast_to(r_body_y, D.shape)
        return D / r_body_map

    # ── Skeleton graph ─────────────────────────────────────────────────────

    def _build_skeleton_graph(self, S):
        """8-connected graph over skeleton pixels, weight 1 or sqrt(2)."""
        G = nx.Graph()
        coords = np.argwhere(S)
        coord_set = set(map(tuple, coords))

        for y, x in coords:
            G.add_node((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if (ny, nx_) in coord_set:
                        w = 1.414 if (dy != 0 and dx != 0) else 1.0
                        G.add_edge((y, x), (ny, nx_), weight=w)
        return G

    # ── Main entry point ───────────────────────────────────────────────────


    def _medial_axis_with_rng(self, mask, rng):
        """
        Version-agnostic wrapper around skimage.medial_axis.

        The keyword controlling the internal tie-breaking RNG was renamed from
        'random_state' to 'rng' in skimage >= 0.22. Try both, then fall back
        to a call without the keyword.
        """
        try:
            return morphology.medial_axis(mask, return_distance=True, rng=rng)
        except TypeError:
            pass
        try:
            return morphology.medial_axis(mask, return_distance=True, random_state=rng)
        except TypeError:
            pass
        return morphology.medial_axis(mask, return_distance=True)


    def extract_features(self, floor_mask, seed=42, rng=None):
        """
        Compute the 8-feature affordance vector for one floor mask.

        Parameters
        ----------
        floor_mask : 2D bool array
            True = navigable floor.
        seed : int
            Used only if rng is None; a fresh Generator is constructed
            from this seed. Does not affect global numpy random state.
        rng : numpy Generator or None
            Overrides seed when provided.

        Returns
        -------
        dict
            Keys: BPD, SC, GTI, NNE, PFF, MAC, BCR, SCE. All floats.
            A mask with no floor returns all zeros.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        floor_mask = np.asarray(floor_mask, dtype=bool)
        if floor_mask.ndim != 2:
            raise ValueError(
                f"floor_mask must be 2D, got shape {floor_mask.shape}"
            )

        H, W = floor_mask.shape
        features = {k: 0.0 for k in FEATURE_NAMES}

        if not np.any(floor_mask):
            return features

        # Skeleton is a deterministic function of the mask. medial_axis uses
        # random tie-breaking internally; we give it a fixed, dedicated
        # Generator so that the skeleton never varies between calls, and so
        # that the caller's rng remains reserved for GTI endpoint sampling.
        skeleton_rng = np.random.default_rng(0)

        S, D = self._medial_axis_with_rng(floor_mask, skeleton_rng)
        if not np.any(S):
            return features

        C = self._get_perspective_clearance(D, H)
        C_S = C[S]

        # ── Family 1: skeletal topology ────────────────────────────────────
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbor_map = convolve(S.astype(int), kernel, mode="constant", cval=0)

        branch_points = (neighbor_map >= 13) & S
        features["BPD"] = float(np.sum(branch_points) / np.sum(S))

        _, num_components = label(S, structure=np.ones((3, 3)))
        features["SC"] = float(num_components)

        G = self._build_skeleton_graph(S)
        endpoints = [n for n, deg in G.degree() if deg == 1]

        img_diagonal = float(np.sqrt(H ** 2 + W ** 2))

        if len(endpoints) >= 2:
            ep_arr = np.array(endpoints, dtype=float)
            dist_mat = cdist(ep_arr, ep_arr)
            features["NNE"] = float(dist_mat.max()) / img_diagonal
        else:
            coords = np.argwhere(S)
            features["NNE"] = (
                float(np.max(np.ptp(coords, axis=0))) / img_diagonal
            )

        features["GTI"] = self._compute_gti(G, endpoints, rng)

        # ── Family 2: clearance ────────────────────────────────────────────
        features["PFF"] = float(np.mean(C_S > 1.0))
        features["MAC"] = float(np.mean(C_S))
        features["BCR"] = float(np.percentile(C_S, 5))

        hist, _ = np.histogram(C_S, bins=20, density=True)
        hist = hist[hist > 0]
        features["SCE"] = float(entropy(hist)) if hist.size else 0.0

        return features

    # ── GTI helper ─────────────────────────────────────────────────────────

    def _compute_gti(self, G, endpoints, rng, max_pairs_per_component=10):
        """
        Graph tortuosity index: mean(arc length / chord length) across
        randomly sampled endpoint pairs within each connected component.
        """
        if len(endpoints) < 2:
            return 1.0

        tortuosities = []
        for comp in nx.connected_components(G):
            comp_eps = [ep for ep in endpoints if ep in comp]
            if len(comp_eps) < 2:
                continue

            num_pairs = min(max_pairs_per_component, len(comp_eps) // 2)
            sampled = [
                rng.choice(len(comp_eps), 2, replace=False)
                for _ in range(num_pairs)
            ]
            for idx_pair in sampled:
                u = comp_eps[idx_pair[0]]
                v = comp_eps[idx_pair[1]]
                two_pts = np.array([u, v], dtype=float)
                chord = float(np.linalg.norm(two_pts[0] - two_pts[1]))
                if chord > 5.0:
                    arc = nx.shortest_path_length(G, u, v, weight="weight")
                    tortuosities.append(arc / chord)

        return float(np.mean(tortuosities)) if tortuosities else 1.0