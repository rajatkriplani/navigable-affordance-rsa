"""Unit tests for src.segment (mock-based, no torch required)."""

import numpy as np
import pytest

from src.segment import SamSegmenter, SegmentationResult


# ── Constructor ────────────────────────────────────────────────────────────────

def test_constructor_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        SamSegmenter(tmp_path / "does_not_exist.pth")


def test_constructor_rejects_bad_model_type(tmp_path):
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"")
    with pytest.raises(ValueError, match="vit_h/vit_l/vit_b"):
        SamSegmenter(fake, model_type="vit_xl")


def test_constructor_accepts_all_valid_model_types(tmp_path):
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"")
    for mt in ("vit_h", "vit_l", "vit_b"):
        seg = SamSegmenter(fake, model_type=mt)
        assert seg.model_type == mt


# ── Segment logic (via monkeypatched _run_prompt) ─────────────────────────────

def _make_segmenter(tmp_path):
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"")
    return SamSegmenter(fake, model_type="vit_b", device="cpu")


def test_segment_uses_primary_when_quality_ok(tmp_path, tiny_rgb_image, monkeypatch):
    seg = _make_segmenter(tmp_path)

    good_mask = np.zeros((64, 64), dtype=bool)
    good_mask[20:40, 20:40] = True   # 400/4096 ≈ 0.098 > 0.02

    calls = []
    def fake_run(image, x, y):
        calls.append((x, y))
        return good_mask, float(good_mask.mean())

    monkeypatch.setattr(seg, "_run_prompt", fake_run)

    result = seg.segment(
        tiny_rgb_image,
        primary_prompt=(0.5, 0.9),
        fallback_prompts=[(0.5, 0.85), (0.5, 0.95)],
        min_floor_fraction=0.02,
    )

    assert isinstance(result, SegmentationResult)
    assert not result.sam_failed
    assert result.prompt_used == (0.5, 0.9)
    assert result.n_prompts_tried == 1
    assert calls == [(0.5, 0.9)]


def test_segment_falls_back_when_primary_fails(tmp_path, tiny_rgb_image, monkeypatch):
    seg = _make_segmenter(tmp_path)

    bad_mask = np.zeros((64, 64), dtype=bool)
    bad_mask[0, 0] = True   # 1/4096 ≈ 0.0002 < 0.02

    good_mask = np.zeros((64, 64), dtype=bool)
    good_mask[20:40, 20:40] = True

    def fake_run(image, x, y):
        if (x, y) == (0.5, 0.9):
            return bad_mask, float(bad_mask.mean())
        return good_mask, float(good_mask.mean())

    monkeypatch.setattr(seg, "_run_prompt", fake_run)

    result = seg.segment(
        tiny_rgb_image,
        primary_prompt=(0.5, 0.9),
        fallback_prompts=[(0.5, 0.85), (0.5, 0.95)],
        min_floor_fraction=0.02,
    )

    assert not result.sam_failed
    assert result.prompt_used == (0.5, 0.85)
    assert result.n_prompts_tried == 2


def test_segment_picks_best_fallback(tmp_path, tiny_rgb_image, monkeypatch):
    seg = _make_segmenter(tmp_path)

    def make_mask(n):
        m = np.zeros((64, 64), dtype=bool)
        m[: n, :] = True
        return m

    masks = {
        (0.5, 0.9):  make_mask(1),    # tiny
        (0.5, 0.85): make_mask(2),    # small
        (0.5, 0.95): make_mask(10),   # largest among fallbacks
        (0.33, 0.9): make_mask(20),   # would win if reached
    }

    def fake_run(image, x, y):
        m = masks[(round(x, 2), round(y, 2))]
        return m, float(m.mean())

    monkeypatch.setattr(seg, "_run_prompt", fake_run)

    result = seg.segment(
        tiny_rgb_image,
        primary_prompt=(0.5, 0.9),
        fallback_prompts=[(0.5, 0.85), (0.5, 0.95), (0.33, 0.9)],
        min_floor_fraction=0.02,
    )

    # Fallback #2 crosses the threshold and stops the loop before #3
    assert not result.sam_failed
    assert result.prompt_used == (0.5, 0.95)
    assert result.n_prompts_tried == 3


def test_segment_fails_when_all_prompts_fail(tmp_path, tiny_rgb_image, monkeypatch):
    seg = _make_segmenter(tmp_path)

    tiny = np.zeros((64, 64), dtype=bool)
    tiny[0, 0] = True

    def fake_run(image, x, y):
        return tiny, float(tiny.mean())

    monkeypatch.setattr(seg, "_run_prompt", fake_run)

    result = seg.segment(
        tiny_rgb_image,
        primary_prompt=(0.5, 0.9),
        fallback_prompts=[(0.5, 0.85), (0.5, 0.95)],
        min_floor_fraction=0.02,
    )

    assert result.sam_failed
    assert not result.mask.any()   # empty mask on failure
    assert result.n_prompts_tried == 3


def test_segment_with_no_fallbacks(tmp_path, tiny_rgb_image, monkeypatch):
    seg = _make_segmenter(tmp_path)

    good = np.ones((64, 64), dtype=bool)  # 1.0, way above threshold

    def fake_run(image, x, y):
        return good, float(good.mean())

    monkeypatch.setattr(seg, "_run_prompt", fake_run)

    result = seg.segment(
        tiny_rgb_image,
        primary_prompt=(0.5, 0.9),
        fallback_prompts=None,
    )

    assert not result.sam_failed
    assert result.n_prompts_tried == 1


# ── Input validation ──────────────────────────────────────────────────────────

def test_segment_rejects_grayscale(tmp_path, monkeypatch):
    seg = _make_segmenter(tmp_path)
    monkeypatch.setattr(seg, "_run_prompt",
                        lambda img, x, y: (np.ones((64, 64), bool), 1.0))
    with pytest.raises(ValueError, match="H, W, 3"):
        seg.segment(np.zeros((64, 64), dtype=np.uint8))


def test_segment_rejects_wrong_channel_count(tmp_path, monkeypatch):
    seg = _make_segmenter(tmp_path)
    monkeypatch.setattr(seg, "_run_prompt",
                        lambda img, x, y: (np.ones((64, 64), bool), 1.0))
    with pytest.raises(ValueError, match="H, W, 3"):
        seg.segment(np.zeros((64, 64, 4), dtype=np.uint8))