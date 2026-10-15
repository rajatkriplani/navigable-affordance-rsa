"""
Typed configuration loader for the navigable-affordance RSA pipeline.

Reads a YAML file, validates required fields, and returns a dataclass tree.
Supports dotted-key overrides, e.g. overrides={"stats.n_permutations": 5000}.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    stimuli_dir: Path
    affordances_dir: Path
    brain_rdms_dir: Path
    mask_cache_dir: Path
    features_csv: Path
    sam_rdm: Path
    stats_dir: Path
    figures_dir: Path


@dataclass
class SamConfig:
    checkpoint: Path
    model_type: str = "vit_h"
    device: str = "auto"
    primary_prompt: tuple = (0.50, 0.90)
    fallback_prompts: list = field(default_factory=list)
    min_floor_fraction: float = 0.02


@dataclass
class FeaturesConfig:
    r_base_px: float = 60.0
    horizon_ratio: float = 0.5
    per_image_seed_policy: str = "hash"
    names: list = field(default_factory=lambda: [
        "BPD", "SC", "GTI", "NNE", "PFF", "MAC", "BCR", "SCE",
    ])
    # Domain labels (presentational grouping)
    topological: list = field(default_factory=lambda: [
        "GTI", "BPD", "NNE", "SC",
    ])
    clearance: list = field(default_factory=lambda: [
        "MAC", "PFF", "BCR", "SCE",
    ])
    # Invariance classes (mathematical classification)
    topological_invariant: list = field(default_factory=lambda: ["SC"])
    geometric: list = field(default_factory=lambda: ["BPD", "GTI", "NNE"])
    metric: list = field(default_factory=lambda: ["PFF", "MAC", "BCR", "SCE"])


@dataclass
class StatsConfig:
    n_stimuli: int = 50
    n_permutations: int = 10_000
    n_permutations_quick: int = 2_000
    fdr_method: str = "fdr_bh"
    master_seed: int = 42


@dataclass
class Config:
    paths: PathsConfig
    sam: SamConfig
    features: FeaturesConfig
    stats: StatsConfig


# ── Loading ────────────────────────────────────────────────────────────────────

def _set_nested(d, dotted_key, value):
    """Set d[a][b][c] = value for dotted_key='a.b.c'."""
    parts = dotted_key.split(".")
    target = d
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def _resolve_paths_section(raw, root):
    """Convert path strings to absolute Path objects anchored at root."""
    out = {}
    for key, val in raw.items():
        p = Path(val)
        out[key] = p if p.is_absolute() else (root / p)
    return out


def load_config(path, overrides=None):
    """
    Load a YAML config file into a typed Config object.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.
    overrides : dict or None
        Dotted-key overrides applied before dataclass construction,
        e.g. {"stats.n_permutations": 5000}.

    Returns
    -------
    Config
    """
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            _set_nested(raw, key, value)

    # Anchor relative paths at the repo root (parent of the configs/ directory)
    root = path.parent.parent

    paths = PathsConfig(**_resolve_paths_section(raw["paths"], root))

    sam_raw = dict(raw["sam"])
    sam_ckpt = Path(sam_raw["checkpoint"])
    sam_raw["checkpoint"] = (
        sam_ckpt if sam_ckpt.is_absolute() else (root / sam_ckpt)
    )
    sam_raw["primary_prompt"] = tuple(sam_raw["primary_prompt"])
    sam_raw["fallback_prompts"] = [
        tuple(p) for p in sam_raw.get("fallback_prompts", [])
    ]
    sam = SamConfig(**sam_raw)

    features = FeaturesConfig(**raw["features"])
    stats = StatsConfig(**raw["stats"])

    return Config(paths=paths, sam=sam, features=features, stats=stats)