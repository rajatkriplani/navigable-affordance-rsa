"""
SAM-based segmentation of navigable floor from indoor scene images.

Design notes
------------
- Device-aware: accepts cuda, mps, or cpu. Torch and segment_anything are
  imported lazily so this module can be inspected without a heavy install.

- Checkpoint-agnostic: vit_h (paper parity), vit_l (balanced), vit_b
  (fastest on CPU). Smaller checkpoints are marginally noisier but
  adequate for coarse floor segmentation. See docs/pipeline.md for the
  timing table that motivates the default.

- Quality-aware: the primary bottom-center prompt is checked against a
  minimum floor-fraction threshold. If it fails, fallback prompts are
  tried in order; the best result is kept. If all prompts fail, the
  returned mask is empty and sam_failed is True.

- Deterministic: given the same image, checkpoint, and prompts, results
  are byte-identical. SAM has no stochastic sampling at inference.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    mask: np.ndarray                # (H, W) bool
    floor_fraction: float           # fraction of pixels in mask
    prompt_used: tuple              # (x_frac, y_frac) that produced the kept mask
    n_prompts_tried: int            # includes the primary
    sam_failed: bool                # True if best floor_fraction < threshold


# ── Segmenter ──────────────────────────────────────────────────────────────────

class SamSegmenter:
    """
    Thin wrapper around SAM's SamPredictor.

    Parameters
    ----------
    checkpoint : str or Path
        Path to a SAM checkpoint (.pth).
    model_type : str
        "vit_h" | "vit_l" | "vit_b".
    device : str
        "cuda" | "mps" | "cpu". Caller is responsible for resolving "auto"
        via src.device_utils.resolve_device.
    """

    def __init__(self, checkpoint, model_type="vit_h", device="cpu"):
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}\n"
                f"Download from "
                f"https://dl.fbaipublicfiles.com/segment_anything/"
                f"sam_{model_type}_*.pth"
            )
        if model_type not in {"vit_h", "vit_l", "vit_b"}:
            raise ValueError(
                f"model_type must be one of vit_h/vit_l/vit_b, got {model_type!r}"
            )

        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as e:
            raise ImportError(
                "segment_anything is required for SAM segmentation. "
                "Install with: "
                "pip install git+https://github.com/facebookresearch/"
                "segment-anything.git"
            ) from e

        logger.info(
            "Loading SAM %s from %s onto %s …",
            self.model_type, self.checkpoint.name, self.device,
        )
        sam = sam_model_registry[self.model_type](
            checkpoint=str(self.checkpoint),
        )
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)
        logger.info("SAM loaded.")
        return self._predictor

    # ── Single prompt ──────────────────────────────────────────────────────

    def _run_prompt(self, image_rgb, x_frac, y_frac):
        """Run a single positive-point prompt; return (mask, floor_fraction)."""
        predictor = self._get_predictor()
        predictor.set_image(image_rgb)

        H, W = image_rgb.shape[:2]
        pt = np.array([[int(W * x_frac), int(H * y_frac)]])
        label = np.array([1])

        masks, scores, _ = predictor.predict(
            point_coords=pt,
            point_labels=label,
            multimask_output=True,
        )
        best = masks[scores.argmax()]
        return best, float(best.mean())

    # ── Full segmentation with fallback ────────────────────────────────────

    def segment(
        self,
        image_rgb,
        primary_prompt=(0.50, 0.90),
        fallback_prompts=None,
        min_floor_fraction=0.02,
    ):
        """
        Segment the navigable floor using the primary prompt, falling back
        to alternatives if the mask is too small.

        Parameters
        ----------
        image_rgb : (H, W, 3) uint8 array, RGB order.
        primary_prompt : tuple (x_frac, y_frac)
        fallback_prompts : list of tuples or None
        min_floor_fraction : float
            Minimum acceptable fraction of image pixels in the mask.

        Returns
        -------
        SegmentationResult
        """
        if fallback_prompts is None:
            fallback_prompts = []

        image_rgb = np.asarray(image_rgb)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image_rgb must be (H, W, 3), got shape {image_rgb.shape}"
            )

        mask, frac = self._run_prompt(image_rgb, *primary_prompt)
        best_mask, best_frac = mask, frac
        best_prompt = primary_prompt
        n_tried = 1

        if best_frac < min_floor_fraction:
            for (x_f, y_f) in fallback_prompts:
                n_tried += 1
                m, f = self._run_prompt(image_rgb, x_f, y_f)
                if f > best_frac:
                    best_mask, best_frac = m, f
                    best_prompt = (x_f, y_f)
                if best_frac >= min_floor_fraction:
                    break

        sam_failed = best_frac < min_floor_fraction
        if sam_failed:
            # Return an empty mask — the caller will flag this scene.
            best_mask = np.zeros_like(best_mask)

        return SegmentationResult(
            mask=best_mask,
            floor_fraction=best_frac,
            prompt_used=best_prompt,
            n_prompts_tried=n_tried,
            sam_failed=sam_failed,
        )