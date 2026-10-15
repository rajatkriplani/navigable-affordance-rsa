"""
CLI: segment the navigable floor from every stimulus image.

Caches masks to disk so re-runs skip SAM entirely. Resumes cleanly after
interruption. Works on cuda, mps, or cpu.

Usage:
    affordance-build-masks --config configs/default.yaml
    affordance-build-masks --device cpu --sam-model vit_b   # fast CPU path
    affordance-build-masks --max-images 3                   # smoke test
    affordance-build-masks --force                          # recompute all
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import load_config
from ..device_utils import device_description, resolve_device
from ..logging_utils import get_logger
from ..segment import SamSegmenter

logger = get_logger(__name__)


METADATA_COLUMNS = [
    "image_name", "floor_fraction", "prompt_used",
    "n_prompts_tried", "sam_failed",
]


# ── Mask IO ────────────────────────────────────────────────────────────────────

def _mask_path(cache_dir, image_name):
    return Path(cache_dir) / f"{Path(image_name).stem}.png"


def _save_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def _load_mask(path):
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return arr > 127


def _load_image_rgb(path):
    """Load image as RGB uint8. cv2 reads BGR by default."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--device", default=None,
                    choices=["auto", "cuda", "mps", "cpu"],
                    help="Override config.sam.device.")
    ap.add_argument("--sam-model", default=None,
                    choices=["vit_h", "vit_l", "vit_b"],
                    help="Override config.sam.model_type.")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Override config.sam.checkpoint.")
    ap.add_argument("--force", action="store_true",
                    help="Recompute masks even if cached.")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Process at most N images (for smoke testing).")
    ap.add_argument("--override", action="append", default=[],
                    metavar="key=value",
                    help="Dotted-key config override (repeatable).")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.override:
        if "=" not in kv:
            ap.error(f"--override must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        overrides[k] = v

    cfg = load_config(args.config, overrides=overrides)

    device_requested = args.device or cfg.sam.device
    device = resolve_device(device_requested)
    logger.info("Device: %s", device_description(device))

    model_type = args.sam_model or cfg.sam.model_type
    checkpoint = args.checkpoint or cfg.sam.checkpoint

    stim_dir = Path(cfg.paths.stimuli_dir)
    cache_dir = Path(cfg.paths.mask_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in stim_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.max_images is not None:
        image_files = image_files[: args.max_images]

    if not image_files:
        logger.error("No images found in %s", stim_dir)
        return 1

    # ── Resume: decide which images still need processing ──────────────────
    metadata_path = cache_dir / "masks_metadata.csv"
    prior_rows = []
    if metadata_path.exists():
        prior_df = pd.read_csv(metadata_path)
        prior_rows = prior_df.to_dict("records")
    prior_by_name = {row["image_name"]: row for row in prior_rows}

    to_process = []
    for img_path in image_files:
        mask_p = _mask_path(cache_dir, img_path.name)
        if mask_p.exists() and not args.force:
            continue
        to_process.append(img_path)

    if not to_process:
        logger.info(
            "All %d masks already cached in %s. Use --force to recompute.",
            len(image_files), cache_dir,
        )
        return 0

    logger.info(
        "Processing %d of %d images (%d already cached). Model=%s, device=%s",
        len(to_process), len(image_files),
        len(image_files) - len(to_process), model_type, device,
    )

    # ── Load SAM (slow on CPU; single load, many inferences) ───────────────
    t0 = time.time()
    segmenter = SamSegmenter(checkpoint, model_type=model_type, device=device)
    segmenter._get_predictor()  # force eager load, so the progress bar is honest
    logger.info("SAM ready in %.1fs.", time.time() - t0)

    # ── Main loop with progress and resume-aware metadata ──────────────────
    new_rows = []
    pbar = tqdm(to_process, desc="Segmenting", unit="img")
    try:
        for img_path in pbar:
            pbar.set_postfix_str(img_path.stem[:22])
            img_rgb = _load_image_rgb(img_path)

            result = segmenter.segment(
                img_rgb,
                primary_prompt=tuple(cfg.sam.primary_prompt),
                fallback_prompts=[tuple(p) for p in cfg.sam.fallback_prompts],
                min_floor_fraction=cfg.sam.min_floor_fraction,
            )

            mask_p = _mask_path(cache_dir, img_path.name)
            _save_mask(result.mask, mask_p)

            row = {
                "image_name": img_path.name,
                "floor_fraction": result.floor_fraction,
                "prompt_used": str(tuple(result.prompt_used)),
                "n_prompts_tried": result.n_prompts_tried,
                "sam_failed": result.sam_failed,
            }
            new_rows.append(row)

            if result.sam_failed:
                pbar.write(
                    f"  ! {img_path.name}: all {result.n_prompts_tried} prompts "
                    f"failed (best floor_fraction = {result.floor_fraction:.4f})"
                )
    except KeyboardInterrupt:
        pbar.write("Interrupted. Saving metadata for completed images.")
    finally:
        pbar.close()

    # ── Merge with prior metadata and save ─────────────────────────────────
    merged = list(prior_by_name.values())
    for row in new_rows:
        prior_by_name[row["image_name"]] = row
    merged = list(prior_by_name.values())
    df = pd.DataFrame(merged, columns=METADATA_COLUMNS)
    df = df.sort_values("image_name").reset_index(drop=True)
    df.to_csv(metadata_path, index=False)

    n_failed = int(df["sam_failed"].sum()) if "sam_failed" in df else 0
    logger.info(
        "Done. %d masks cached. %d SAM failures. Metadata: %s",
        len(df), n_failed, metadata_path,
    )
    if n_failed:
        logger.warning(
            "%d images could not be segmented. Inspect %s "
            "and consider manual masks for these scenes.",
            n_failed, metadata_path,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())