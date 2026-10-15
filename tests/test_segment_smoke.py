"""
Real SAM segmentation smoke test.

Runs SAM on a 64x64 synthetic image with a bright central square. Verifies
that the returned mask is non-empty and roughly captures the square.

Skipped unless:
  - segment_anything is importable, AND
  - a SAM checkpoint exists at $SAM_CHECKPOINT or the path below.

This test is marked 'slow' — deselect with: pytest -m "not slow"
"""

import os
from pathlib import Path

import numpy as np
import pytest

# Skip whole module if SAM isn't installed
pytest.importorskip("segment_anything",
                    reason="segment_anything not installed")
torch = pytest.importorskip("torch",
                            reason="torch not installed")


DEFAULT_CHECKPOINT = Path("checkpoints/sam_vit_b_01ec64.pth")
CHECKPOINT = Path(os.environ.get("SAM_CHECKPOINT", DEFAULT_CHECKPOINT))


@pytest.mark.slow
@pytest.mark.skipif(not CHECKPOINT.exists(),
                    reason=f"SAM checkpoint not found at {CHECKPOINT}. "
                           f"Set $SAM_CHECKPOINT or place the .pth there.")
def test_sam_produces_nonempty_mask_on_white_square():
    from src.segment import SamSegmenter

    # Bright white square on black background
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = 255

    seg = SamSegmenter(CHECKPOINT, model_type="vit_b", device="cpu")

    result = seg.segment(
        img,
        primary_prompt=(0.5, 0.5),     # centre of the square
        fallback_prompts=[],
        min_floor_fraction=0.05,       # 32x32 = 1024/4096 ≈ 0.25 expected
    )

    assert not result.sam_failed
    assert result.mask.any()
    # Mask should cover the central region roughly
    assert 0.10 <= result.floor_fraction <= 0.60