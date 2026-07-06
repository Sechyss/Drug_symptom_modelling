#!/usr/bin/env python3
"""Parameter sweep for SEIRS v9: peak infections vs (restoration_efficiency, mr, phi_transmission).

This script tests model v9 where:
  - Contact penalty is KEPT (high virulence reduces contacts)
  - Recovery rate SCALES with transmission (maintains R0 balance at baseline)
  - Result: R0_high = R0_low at baseline (no drug)
           Drug creates selection pressure for high virulence

Outputs:
  Results/drug_restoration_sweep_v9.csv
    Figures/Model_v9_exploration/peak_infection_heatmaps.svg

Run from repo root:
    python Scripts/Drug_restoration_sweep_3D_v9.py \
        --phi-min 1.0 --phi-max 2.0 --n-phi 5 \
        --restore-min 0.0 --restore-max 1.0 --n-restore 20 \
        --mr-min 0.5 --mr-max 1.0 --n-mr 20 \
        --days 365 --steps 365

Notes:
  - Uses SEIRS_model_v9: contact penalty + scaled recovery
  - φ_t must satisfy: 1.0 <= φ_t < 2.0 (see constraint below)
  - R0 baseline is EXACTLY equal across strains (no drug)
"""
from typing import Dict, Tuple, List
import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib

from scipy.integrate import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import FuncFormatter

# allow imports from project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "../.."))
sys.path.insert(0, ROOT_DIR)
from Models.SEIRS_Models import SEIRS_model_v9
from Models import params as P

COLS = ["S", "Eh", "Indh", "Idh", "Rh", "El", "Indl", "Idl", "Rl"]
REL_DIFF_EPS = 1e-12


def initial_conditions(normalize: bool = True) -> np.ndarray:
    """Initial conditions normalized to fractions."""
    y0 = np.array(
        [
            getattr(P, "S", 10000),
            getattr(P, "Eh", 0),
            getattr(P, "Indh", 5),
            getattr(P, "Idh", 0),
            getattr(P, "Rh", 0),
            getattr(P, "El", 0),
            getattr(P, "Indl", 5),
            getattr(P, "Idl", 0),
            getattr(P, "Rl", 0),
        ],
        dtype=float,
    )
    y0 = np.maximum(y0, 0.0)
    if normalize:
        s = y0.sum()
        if s <= 0 or not np.isfinite(s):
            raise ValueError("Initial conditions must have positive finite sum.")
        y0 = y0 / s
    return y0


def params_tuple_v9(phi_t: float, restore: float, mr: float) -> Tuple[float, ...]:
    """Build parameter vector for SEIRS_model_v9 (10 params)."""
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
    )
    kappa_base = float(getattr(P, "kappa_base", 1.0))
    kappa_scale = float(getattr(P, "kappa_scale", 1.0))
    sigma = float(getattr(P, "sigma", 1.0 / 5.0))
    tau = float(getattr(P, "tau", 1.0 / 3.0))
    theta = float(getattr(P, "theta", 0.3))

    return (
        c_low, r_low, float(phi_t),
        float(restore), float(mr),
        kappa_base, kappa_scale,
        sigma, tau, theta
    )


def run_sim(phi_t: float, restore: float, mr: float, days: int, steps: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run v9 simulation."""
    y0 = initial_conditions(normalize=True)
    t = np.linspace(0, days, steps)
    params = params_tuple_v9(phi_t, restore, mr)

    try:
        sol = odeint(SEIRS_model_v9, y0, t, args=(params,), full_output=False)
    except Exception as e:
        print(f"  ODE solver failed for phi={phi_t}, restore={restore}, mr={mr}: {e}")
        return t, {col: np.full(len(t), np.nan) for col in COLS}

    sim = {col: sol[:, i] for i, col in enumerate(COLS)}
    return t, sim


def metrics(t: np.ndarray, sim: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract peak infection metrics."""
    peak_Ih_total = float(np.nanmax(sim["Indh"] + sim["Idh"])) if len(sim["Indh"]) > 0 else 0.0
    peak_Il_total = float(np.nanmax(sim["Indl"] + sim["Idl"])) if len(sim["Indl"]) > 0 else 0.0

    return {
        "peak_I_high": peak_Ih_total,
        "peak_I_low": peak_Il_total,
        "peak_I_high_total": peak_Ih_total,
        "peak_I_low_total": peak_Il_total,
    }


def fold_change(current: float, baseline: float) -> float:
    """Return fold change relative to baseline, or NaN if baseline is zero."""
    baseline = float(baseline)
    if not np.isfinite(baseline) or abs(baseline) <= REL_DIFF_EPS:
        return float(np.nan)
    return float(float(current) / baseline)


def peak_infection_heatmaps(df: pd.DataFrame, out_path: str, max_phi_panels: int = 8) -> None:
    """2D heatmaps faceted by phi, showing fold change from baseline.

    Layout:
    - Left column: Low-strain heatmaps (one per phi value)
    - Right column: High-strain heatmaps (one per phi value)
    - Single shared colorbar and shared color scale across all panels
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    phi_all = sorted(df["phi_transmission"].unique())
    if len(phi_all) > max_phi_panels:
        phi_vals = [phi_all[int(i * len(phi_all) / max_phi_panels)] for i in range(max_phi_panels)]
    else:
        phi_vals = phi_all
    n_phi = len(phi_vals)
    fig, axes = plt.subplots(n_phi, 2, figsize=(13.5, 4.8 * n_phi), squeeze=False)
    restoration_min = df["restoration_efficiency"].min()
    restoration_max = df["restoration_efficiency"].max()
    mr_min = df["mr"].min()
    mr_max = df["mr"].max()

    def make_local_norm(values: np.ndarray):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return Normalize(vmin=0.0, vmax=1.0), 0.0, 1.0

        vmin = float(finite.min())
        vmax = float(finite.max())
        if np.isclose(vmin, vmax):
            norm = Normalize(vmin=vmin - 1e-12, vmax=vmax + 1e-12)
        elif vmin < 1.0 < vmax:
            # Fold-change neutrality is 1.0, so center diverging colors at 1.0.
            norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        return norm, vmin, vmax

    def build_ticks(vmin: float, vmax: float) -> np.ndarray:
        if np.isclose(vmin, vmax):
            span = max(abs(vmin), 1e-9)
            return np.array([vmin - 0.05 * span, vmin, vmin + 0.05 * span])
        if vmin < 1.0 < vmax:
            lower_mid = 0.5 * (vmin + 1.0)
            upper_mid = 0.5 * (1.0 + vmax)
            ticks = np.array([vmin, lower_mid, 1.0, upper_mid, vmax])
            return np.sort(np.unique(ticks))
        return np.linspace(vmin, vmax, 5)

    sci_fmt = FuncFormatter(lambda x, _: f"{x:.2f}")

    for phi_idx, phi_t in enumerate(phi_vals):
        df_phi = df[df["phi_transmission"] == phi_t]
        pivot_high = df_phi.pivot_table(index="mr", columns="restoration_efficiency", values="fold_peak_I_high", sort=True)
        pivot_low = df_phi.pivot_table(index="mr", columns="restoration_efficiency", values="fold_peak_I_low", sort=True)

        low_vals = pivot_low.to_numpy(dtype=float)
        low_norm, low_min, low_max = make_local_norm(low_vals)
        im_l = axes[phi_idx, 0].imshow(
            pivot_low.values,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            norm=low_norm,
            extent=[restoration_min, restoration_max, mr_min, mr_max],
        )
        axes[phi_idx, 0].set_title(f"Low-strain (phi={phi_t:.2f})", fontsize=11, fontweight="bold", pad=15)
        axes[phi_idx, 0].set_xlabel("Restoration efficiency rho", fontsize=10)
        axes[phi_idx, 0].set_ylabel("Transmission multiplier m_r", fontsize=10)
        cbar_l = fig.colorbar(im_l, ax=axes[phi_idx, 0], fraction=0.046, pad=0.03)
        cbar_l.set_ticks(build_ticks(low_min, low_max))
        cbar_l.ax.yaxis.set_major_formatter(sci_fmt)
        cbar_l.ax.tick_params(labelsize=8, length=3, width=0.8)

        high_vals = pivot_high.to_numpy(dtype=float)
        high_norm, high_min, high_max = make_local_norm(high_vals)
        im_h = axes[phi_idx, 1].imshow(
            pivot_high.values,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            norm=high_norm,
            extent=[restoration_min, restoration_max, mr_min, mr_max],
        )
        axes[phi_idx, 1].set_title(f"High-strain (phi={phi_t:.2f})", fontsize=11, fontweight="bold", pad=15)
        axes[phi_idx, 1].set_xlabel("Restoration efficiency rho", fontsize=10)
        axes[phi_idx, 1].set_ylabel("Transmission multiplier m_r", fontsize=10)
        cbar_h = fig.colorbar(im_h, ax=axes[phi_idx, 1], fraction=0.046, pad=0.03)
        cbar_h.set_ticks(build_ticks(high_min, high_max))
        cbar_h.ax.yaxis.set_major_formatter(sci_fmt)
        cbar_h.ax.tick_params(labelsize=8, length=3, width=0.8)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.1, top=0.94, hspace=0.4, wspace=0.26)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.12)
    print(f"✓ Saved heatmap: {out_path}")
    plt.close(fig)


def compute_R0_and_beta_v9(
    phi_t: float, restore: float, mr: float, S_prop: float = 0.99
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute R0 and β (transmission rate) for v9 model.
    
    Returns:
        (R0_low, R0_high, beta_low, beta_high, sigma_low, sigma_high)
    """
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
    )
    sigma = float(getattr(P, "sigma", 1.0 / 5.0))
    
    alpha = 0.5
    vir_excess_pos = max(0.0, phi_t - 1.0)
    
    # Contact rates
    c_high_untreated = c_low * np.exp(-alpha * vir_excess_pos)
    c_low_untreated = c_low
    
    # Baseline transmission (no drug: restore=0, mr=1)
    beta_l = c_low * r_low * S_prop
    beta_h = c_high_untreated * r_low * phi_t * S_prop
    
    # Recovery rates scale to maintain R0 balance
    beta_ratio = beta_h / max(beta_l, 1e-10)
    sigma_h = sigma * beta_ratio
    sigma_l = sigma
    sigma_h = max(sigma_h, 1e-8)
    sigma_l = max(sigma_l, 1e-8)
    
    # R0 values (baseline)
    R0_low = beta_l / sigma_l
    R0_high = beta_h / sigma_h
    
    return (
        float(R0_low),
        float(R0_high),
        float(beta_l),
        float(beta_h),
        float(sigma_l),
        float(sigma_h),
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parameter sweep for SEIRS v9: recovery scales with transmission"
    )

    # Range mode (preferred for v9)
    parser.add_argument("--phi-min", type=float, default=1.0)
    parser.add_argument("--phi-max", type=float, default=2.0)
    parser.add_argument("--n-phi", type=int, default=5)

    parser.add_argument("--restore-min", type=float, default=0.0)
    parser.add_argument("--restore-max", type=float, default=1.0)
    parser.add_argument("--n-restore", type=int, default=20)

    parser.add_argument("--mr-min", type=float, default=0.5)
    parser.add_argument("--mr-max", type=float, default=1.0)
    parser.add_argument("--n-mr", type=int, default=20)

    parser.add_argument("--max-phi-panels", type=int, default=8)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--steps", type=int, default=365)

    args = parser.parse_args(argv)

    print("=" * 70)
    print("SEIRS v9: Contact Penalty + Scaled Recovery (R0 Balanced Baseline)")
    print("=" * 70)

    # Validate phi range
    if args.phi_max >= 2.0:
        print(f"⚠ WARNING: phi_max >= 2.0 may cause numerical instability")
        print(f"  Recommended: phi_t < 2.0 (to keep recovery rate positive)")
        print(f"  Current: phi_t ∈ [{args.phi_min}, {args.phi_max})")
        print()

    phi_vals = np.linspace(args.phi_min, args.phi_max, args.n_phi).tolist()
    restore_vals = np.linspace(args.restore_min, args.restore_max, args.n_restore).tolist()
    mr_vals = np.linspace(args.mr_min, args.mr_max, args.n_mr).tolist()

    print(f"φ_transmission: {[f'{x:.2f}' for x in phi_vals]}")
    print(f"restoration_efficiency (ρ): {[f'{x:.2f}' for x in restore_vals]}")
    print(f"m_r (transmission reduction): {[f'{x:.2f}' for x in mr_vals]}")
    print(f"Days: {args.days}, Steps: {args.steps}")
    print("=" * 70)

    # Baseline simulation per phi for consistent within-phi relative changes
    print("\nRunning baseline simulations by phi (ρ=0.0, m_r=1.0)...")
    baselines_by_phi: Dict[float, Dict[str, float]] = {}
    for phi_t in phi_vals:
        t_baseline, sim_baseline = run_sim(phi_t, 0.0, 1.0, args.days, args.steps)
        m_baseline = metrics(t_baseline, sim_baseline)
        baselines_by_phi[float(phi_t)] = {
            "peak_I_high": m_baseline["peak_I_high"],
            "peak_I_low": m_baseline["peak_I_low"],
        }
        print(
            f"  phi={phi_t:.3f} -> baseline peak_I_high={m_baseline['peak_I_high']:.6f}, "
            f"peak_I_low={m_baseline['peak_I_low']:.6f}"
        )
    print("=" * 70)

    # Parameter sweep
    n_total = len(phi_vals) * len(restore_vals) * len(mr_vals)
    results = []
    count = 0

    for phi_t in phi_vals:
        for restore in restore_vals:
            for mr in mr_vals:
                count += 1
                print(f"[{count}/{n_total}] φ={phi_t:.3f}, ρ={restore:.3f}, m_r={mr:.3f}...", end=" ")
                t, sim = run_sim(phi_t, restore, mr, args.days, args.steps)
                m = metrics(t, sim)
                baseline = baselines_by_phi[float(phi_t)]
                results.append({
                    "phi_transmission": float(phi_t),
                    "restoration_efficiency": float(restore),
                    "mr": float(mr),
                    "peak_I_high": m["peak_I_high"],
                    "peak_I_low": m["peak_I_low"],
                    "fold_peak_I_high": fold_change(m["peak_I_high"], baseline["peak_I_high"]),
                    "fold_peak_I_low": fold_change(m["peak_I_low"], baseline["peak_I_low"]),
                })
                print("✓")

    df = pd.DataFrame(results)
    csv_path = os.path.join(ROOT_DIR, "Results", "drug_restoration_sweep_v9.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")

    peak_infection_heatmaps(
        df,
        os.path.join(ROOT_DIR, "Figures", "Model_v9_exploration", "peak_infection_heatmaps.svg"),
        max_phi_panels=args.max_phi_panels,
    )

    # Print R0 and β values
    print("\n" + "=" * 90)
    print("R0 AND TRANSMISSION RATES BY VIRULENCE (baseline: ρ=0.0, m_r=1.0)")
    print("=" * 90)
    print(
        f"{'φ_trans':<12} {'β_low':<15} {'β_high':<15} "
        f"{'σ_low':<12} {'σ_high':<12} {'R0_low':<12} {'R0_high':<12}"
    )
    print("-" * 90)
    
    for phi_t in sorted(set(phi_vals)):
        (
            R0_low,
            R0_high,
            beta_low,
            beta_high,
            sigma_low,
            sigma_high,
        ) = compute_R0_and_beta_v9(phi_t, restore=0.0, mr=1.0, S_prop=0.99)
        equal_str = "✓ Yes" if abs(R0_high - R0_low) < 0.01 else "✗ No"
        print(
            f"{phi_t:<12.3f} {beta_low:<15.6f} {beta_high:<15.6f} "
            f"{sigma_low:<12.6f} {sigma_high:<12.6f} {R0_low:<12.4f} {R0_high:<12.4f} {equal_str:<10}"
        )
    
    print("=" * 90 + "\n")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())