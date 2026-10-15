"""
CLI: run RSA analysis on the SAM and NAM model RDMs against OPA/PPA/EVC.

Reads:
  - The brain RDM .mat files (OPA, PPA, EVC, NAM)
  - The SAM-derived topological RDM (from build_rdm CLI)
  - Optional: per-subject RDM stack for noise ceiling

Writes:
  - outputs/stats/rsa.json   — structured results
  - outputs/figures/rsa_summary.png — bar chart + partial RSA panel

Usage:
    affordance-rsa --config configs/default.yaml
    affordance-rsa --config configs/default.yaml --n-perm 1000  # quick test
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from ..config import load_config
from ..io import load_brain_rdm, save_json
from ..logging_utils import get_logger
from ..rsa import run_full_analysis

logger = get_logger(__name__)


# ── Loading ────────────────────────────────────────────────────────────────────

def _load_brain_data(brain_rdms_dir):
    """
    Load OPA, PPA, EVC, NAM from brain_rdms_dir.

    Returns
    -------
    dict of {key: 2D array}. Keys: opa, ppa, evc, nam.
    """
    out = {}
    for key in ("OPA", "PPA", "EVC", "NAM"):
        path = Path(brain_rdms_dir) / f"{key}.mat"
        if not path.exists():
            raise FileNotFoundError(
                f"Required brain RDM not found: {path}"
            )
        out[key.lower()] = load_brain_rdm(path)
        logger.info("Loaded %s RDM shape %s from %s",
                    key, out[key.lower()].shape, path.name)
    return out


def _load_subject_rdms(brain_rdms_dir):
    """
    Try to load per-subject RDM stacks for the noise ceiling. Returns an
    empty dict if not available — noise ceiling is optional.
    """
    try:
        from ..bonner import load_subject_rdms
    except ImportError:
        return {}

    candidates = {
        "opa": Path(brain_rdms_dir) / "all_sub_slrdms_opa.npy",
        "ppa": Path(brain_rdms_dir) / "all_sub_slrdms_ppa.npy",
        "evc": Path(brain_rdms_dir) / "all_sub_slrdms_evc.npy",
    }
    found = {k: load_subject_rdms(p) for k, p in candidates.items() if p.exists()}
    if found:
        logger.info("Loaded per-subject RDMs for noise ceiling: %s",
                    ", ".join(found))
    else:
        logger.info(
            "Per-subject ROI RDMs not found. Noise ceiling will be skipped. "
            "(See docs/pipeline.md §6 for how to supply them.)"
        )
    return found


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--n-perm", type=int, default=None,
                    help="Override stats.n_permutations from config.")
    ap.add_argument("--master-seed", type=int, default=None,
                    help="Override stats.master_seed.")
    ap.add_argument("--override", action="append", default=[],
                    metavar="key=value")
    args = ap.parse_args(argv)

    overrides = {}
    for kv in args.override:
        if "=" not in kv:
            ap.error(f"--override must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        overrides[k] = v
    cfg = load_config(args.config, overrides=overrides)

    n_perm = args.n_perm or cfg.stats.n_permutations
    master_seed = args.master_seed or cfg.stats.master_seed
    fdr_method = cfg.stats.fdr_method

    # ── Load RDMs ──────────────────────────────────────────────────────────
    brain_data = _load_brain_data(cfg.paths.brain_rdms_dir)
    brain_rdms = {
        "opa": brain_data["opa"],
        "ppa": brain_data["ppa"],
        "evc": brain_data["evc"],
    }

    sam_rdm_path = Path(cfg.paths.sam_rdm)
    if not sam_rdm_path.exists():
        logger.error(
            "SAM RDM not found at %s. Run affordance-build-rdm first.",
            sam_rdm_path,
        )
        return 1
    sam_rdm = np.load(sam_rdm_path)
    logger.info("Loaded SAM RDM shape %s from %s",
                sam_rdm.shape, sam_rdm_path.name)

    # ── Shape check ────────────────────────────────────────────────────────
    expected_shape = brain_rdms["opa"].shape
    if sam_rdm.shape != expected_shape:
        logger.error(
            "Shape mismatch: SAM RDM is %s, brain RDMs are %s. "
            "Both must cover the same stimulus set.",
            sam_rdm.shape, expected_shape,
        )
        return 1
    if np.any(np.isnan(sam_rdm)):
        n_nan = int(np.isnan(sam_rdm).sum())
        logger.error(
            "SAM RDM contains %d NaN values. Exclude failed rows via "
            "build_rdm_from_dataframe before running RSA.",
            n_nan,
        )
        return 1

    model_rdms = {
        "sam": sam_rdm,
        "nam": brain_data["nam"],
    }

    # ── Optional noise ceiling ─────────────────────────────────────────────
    subject_rdms_by_roi = _load_subject_rdms(cfg.paths.brain_rdms_dir)

    # ── Run ────────────────────────────────────────────────────────────────
    result = run_full_analysis(
        model_rdms=model_rdms,
        brain_rdms=brain_rdms,
        subject_rdms_by_roi=subject_rdms_by_roi or None,
        n_perm=n_perm,
        master_seed=master_seed,
        fdr_method=fdr_method,
    )

    # ── Save JSON ──────────────────────────────────────────────────────────
    stats_dir = Path(cfg.paths.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    json_path = stats_dir / "rsa.json"
    save_json(result.as_dict(), json_path)
    logger.info("Saved RSA results to %s", json_path)

    # ── Console summary ────────────────────────────────────────────────────
    _print_summary(result)

    # ── Figure ─────────────────────────────────────────────────────────────
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "rsa_summary.png"
    try:
        _plot_summary(result, fig_path)
        logger.info("Saved summary figure to %s", fig_path)
    except Exception as e:
        logger.warning("Figure generation failed: %s", e)

    return 0


def _print_summary(result):
    print()
    print("=" * 66)
    print("STANDARD RSA  (no controls, permutation-tested, FDR-corrected)")
    print("=" * 66)
    print(f"  {'Test':<18}  {'r':>8}  {'p_perm':>9}  {'q_FDR':>9}  sig")
    print(f"  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*9}  ---")
    for r in result.standard:
        sig = "✓" if r.q_fdr < 0.05 else " "
        label = f"{r.model}→{r.brain}"
        print(f"  {label:<18}  {r.r:>+8.4f}  {r.p_perm:>9.4f}  "
              f"{r.q_fdr:>9.4f}  {sig}")

    print()
    print("=" * 66)
    print("PARTIAL RSA  (novelty tests, controls in parentheses)")
    print("=" * 66)
    for r in result.partial:
        sig = "✓" if r.q_fdr < 0.05 else " "
        ctrls = "+".join(r.controls) if r.controls else "—"
        label = f"{r.model}→{r.brain} ({ctrls})"
        print(f"  {label:<32}  r = {r.r:+.4f}  q = {r.q_fdr:.4f}  {sig}")

    if result.noise_ceiling:
        print()
        print("=" * 66)
        print("NOISE CEILING  (Nili et al. 2014, LOO)")
        print("=" * 66)
        for roi, nc in result.noise_ceiling.items():
            if nc is None:
                print(f"  {roi}: not available")
            else:
                print(f"  {roi}: lower = {nc['lower']:.3f}  "
                      f"upper = {nc['upper']:.3f}")

    print()
    print("=" * 66)
    print("CONCLUSION")
    print("=" * 66)
    c = result.conclusions
    print(f"  SAM→OPA partial r = {c['sam_opa_partial_r']}, "
          f"q = {c['sam_opa_partial_q']}")
    print(f"  SAM→PPA partial r = {c['sam_ppa_partial_r']}, "
          f"q = {c['sam_ppa_partial_q']}")
    print(f"  Verdict: {c['verdict']}")
    print()


def _plot_summary(result, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: standard RSA
    ax = axes[0]
    labels = [f"{r.model}\n{r.brain}" for r in result.standard]
    values = [r.r for r in result.standard]
    colors = ["#4878cf" if r.model == "NAM" else "#6acc65"
              for r in result.standard]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.6)
    for bar, r in zip(bars, result.standard):
        if r.q_fdr < 0.05:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + 0.002 * np.sign(y or 1),
                    "✓", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=12)
    ax.set_ylabel("Spearman r")
    ax.set_title("Standard RSA")
    ax.tick_params(axis="x", labelsize=8)

    # Panel B: partial RSA
    ax2 = axes[1]
    labels2 = [f"{r.model}\n→{r.brain}" for r in result.partial]
    values2 = [r.r for r in result.partial]
    colors2 = ["#6acc65" if r.model == "SAM" else "#4878cf"
               for r in result.partial]
    bars2 = ax2.bar(labels2, values2, color=colors2, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.6)
    for bar, r in zip(bars2, result.partial):
        if r.q_fdr < 0.05:
            y = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     y + 0.002 * np.sign(y or 1),
                     "✓", ha="center",
                     va="bottom" if y >= 0 else "top", fontsize=12)
    ax2.set_ylabel("Partial Spearman r")
    ax2.set_title("Novelty Test (partial)")
    ax2.tick_params(axis="x", labelsize=8)

    fig.suptitle("RSA Summary — SAM topological vs NAM baseline",
                 fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())