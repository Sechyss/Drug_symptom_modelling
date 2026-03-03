#!/usr/bin/env python3
"""Parameter sweep: peak infections vs (restoration_efficiency, mr, phi_transmission).

Outputs heatmaps showing (restoration_efficiency, mr) → peak_I, faceted by phi.

Outputs
- Figures/drug_restoration_sweep/peak_infection_heatmaps.png

Run (from repo root)
    python Scripts/Drug_restoration_sweep_3D.py \
        --phi 1.0 1.2 1.5 1.8 2.2 \
        --restore 0.0 0.3 0.6 0.9 \
        --mr 0.5 0.7 1.0 \
        --days 365 --steps 365

Notes
- Uses SEIRS_model_v8: unified contact restoration mechanism.
- Produces 2D heatmaps: (restoration_efficiency, mr) → peak_I
- One row per phi: high-strain peak (top), low-strain peak (bottom)
- FIXED: All results now in consistent units (fraction of population)
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
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.SEIRS_Models import SEIRS_model_v8
from Models import params as P

COLS = ["S", "Eh", "Indh", "Idh", "Rh", "El", "Indl", "Idl", "Rl"]


def initial_conditions(normalize: bool = True) -> np.ndarray:
    """Initial conditions for SEIRS model.

    Args:
        normalize: If True, return fractions (sum=1). If False, return absolute counts.

    Returns:
        Initial state vector [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
    """
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


def c_high_from_phi_v8(c_low: float, phi_t: float, alpha: float = 0.5) -> float:
    """High-strain contact rate with virulence penalty (untreated)."""
    vir_excess_pos = max(0.0, float(phi_t) - 1.0)
    c_high = float(c_low) * np.exp(-alpha * vir_excess_pos)
    return c_high


def params_tuple_v8(phi_t: float, restore: float, mr: float) -> Tuple[float, ...]:
    """SEIRS_model_v8 expects 13 params."""
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
    )

    birth_rate = float(getattr(P, "birth_rate", 0.0))
    death_rate = float(getattr(P, "death_rate", 0.0))
    delta = float(getattr(P, "delta", 0.0))
    kappa_base = float(getattr(P, "kappa_base", 1.0))
    kappa_scale = float(getattr(P, "kappa_scale", 1.0))
    sigma = float(getattr(P, "sigma", 1.0 / 5.0))
    tau = float(getattr(P, "tau", 1.0 / 3.0))
    theta = float(getattr(P, "theta", 0.3))

    return (
        c_low, r_low, float(phi_t),
        float(restore), float(mr),
        birth_rate, death_rate, delta,
        kappa_base, kappa_scale,
        sigma, tau, theta
    )


def run_sim(phi_t: float, restore: float, mr: float, days: int, steps: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run simulation and return time grid and state trajectories.

    FIXED: Now consistently uses normalized initial conditions (fractions).
    """
    y0 = initial_conditions(normalize=True)  # ← FIXED: Always normalize
    t = np.linspace(0, days, steps)
    params = params_tuple_v8(phi_t, restore, mr)

    try:
        sol = odeint(SEIRS_model_v8, y0, t, args=(params,), full_output=False)
    except Exception as e:
        print(f"  ODE solver failed for phi={phi_t}, restore={restore}, mr={mr}: {e}")
        return t, {col: np.full(len(t), np.nan) for col in COLS}

    sim = {col: sol[:, i] for i, col in enumerate(COLS)}
    return t, sim


def metrics(t: np.ndarray, sim: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Extract key metrics from trajectory.

    Returns metrics as FRACTIONS of population (since initial conditions normalized).
    """
    peak_Ih = float(np.nanmax(sim["Idh"])) if len(sim["Idh"]) > 0 else 0.0
    peak_Il = float(np.nanmax(sim["Idl"])) if len(sim["Idl"]) > 0 else 0.0
    peak_Ih_total = float(np.nanmax(sim["Indh"] + sim["Idh"])) if len(sim["Indh"]) > 0 else 0.0
    peak_Il_total = float(np.nanmax(sim["Indl"] + sim["Idl"])) if len(sim["Indl"]) > 0 else 0.0

    return {
        "peak_I_high": peak_Ih,
        "peak_I_low": peak_Il,
        "peak_I_high_total": peak_Ih_total,
        "peak_I_low_total": peak_Il_total,
    }


def peak_infection_heatmaps(df: pd.DataFrame, out_path: str, max_phi_panels: int = 8) -> None:
    """2D heatmap slices: (restoration, mr) -> delta_peak_I, faceted by phi.

    If phi has many values, only evenly spaced slices are plotted.
    Shows difference from baseline (no drug).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    phi_all = sorted(df["phi_transmission"].unique())
    if len(phi_all) > max_phi_panels:
        phi_vals = [phi_all[int(i * len(phi_all) / max_phi_panels)] for i in range(max_phi_panels)]
    else:
        phi_vals = phi_all

    n_phi = len(phi_vals)

    fig, axes = plt.subplots(
        2, n_phi, figsize=(4.2 * n_phi, 7.5), squeeze=False, constrained_layout=True
    )

    # Symmetric normalization around zero for deltas
    max_abs_high = df["delta_peak_I_high"].abs().max()
    max_abs_low = df["delta_peak_I_low"].abs().max()
    norm_high = TwoSlopeNorm(vmin=-max_abs_high, vcenter=0.0, vmax=max_abs_high)
    norm_low = TwoSlopeNorm(vmin=-max_abs_low, vcenter=0.0, vmax=max_abs_low)

    im_h_ref = None
    im_l_ref = None

    # Get parameter ranges for drug trajectory
    restoration_min = df["restoration_efficiency"].min()
    restoration_max = df["restoration_efficiency"].max()
    restoration_baseline = df[df["restoration_efficiency"] == restoration_min].iloc[0]["restoration_efficiency"] \
        if len(df[df["restoration_efficiency"] == restoration_min]) > 0 else restoration_min
    mr_min = df["mr"].min()
    mr_max = df["mr"].max()

    for phi_idx, phi_t in enumerate(phi_vals):
        df_phi = df[df["phi_transmission"] == phi_t]

        # Pivot for heatmap
        pivot_high = df_phi.pivot_table(
            index="mr", columns="restoration_efficiency", values="delta_peak_I_high", sort=True
        )
        pivot_low = df_phi.pivot_table(
            index="mr", columns="restoration_efficiency", values="delta_peak_I_low", sort=True
        )

        # High-strain heatmap
        im_h = axes[0, phi_idx].imshow(
            pivot_high.values, origin="lower", aspect="auto", cmap="coolwarm", norm=norm_high,
            extent=[restoration_min, restoration_max, mr_min, mr_max]
        )
        axes[0, phi_idx].set_title(f"High-strain (φ={phi_t})")
        axes[0, phi_idx].set_xlabel("Restoration efficiency ρ")
        axes[0, phi_idx].set_ylabel("Drug transmission multiplier m_r")

        # Low-strain heatmap
        im_l = axes[1, phi_idx].imshow(
            pivot_low.values, origin="lower", aspect="auto", cmap="coolwarm", norm=norm_low,
            extent=[restoration_min, restoration_max, mr_min, mr_max]
        )
        axes[1, phi_idx].set_title(f"Low-strain (φ={phi_t})")
        axes[1, phi_idx].set_xlabel("Restoration efficiency ρ")
        axes[1, phi_idx].set_ylabel("Drug transmission multiplier m_r")

        if im_h_ref is None:
            im_h_ref = im_h
        if im_l_ref is None:
            im_l_ref = im_l

    cbar_h = fig.colorbar(im_h_ref, ax=axes[0, :], orientation="vertical", fraction=0.02, pad=0.02)
    cbar_h.set_label("Δ Peak I_high (vs baseline)", fontsize=9)

    cbar_l = fig.colorbar(im_l_ref, ax=axes[1, :], orientation="vertical", fraction=0.02, pad=0.02)
    cbar_l.set_label("Δ Peak I_low (vs baseline)", fontsize=9)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved heatmap figure with drug trajectories: {out_path}")
    plt.close(fig)


def bubble_chart(df: pd.DataFrame, out_path: str) -> None:
    """Bubble chart with jitter showing delta from baseline."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Add jitter to separate overlapping points
    np.random.seed(42)
    jitter_strength = 0.035
    df_jittered = df.copy()
    df_jittered["restoration_efficiency_j"] = df["restoration_efficiency"] + np.random.normal(0, jitter_strength, len(df))
    df_jittered["mr_j"] = df["mr"] + np.random.normal(0, jitter_strength, len(df))

    # Amplify bubble size differences
    phi_min = df["phi_transmission"].min()
    phi_max = df["phi_transmission"].max()
    phi_normalized = (df["phi_transmission"] - phi_min) / (phi_max - phi_min)
    size_scale = 500
    sizes = (phi_normalized ** 1.5) * size_scale + 100

    # Symmetric normalization for deltas
    max_abs_high = df["delta_peak_I_high"].abs().max()
    max_abs_low = df["delta_peak_I_low"].abs().max()

    # ─────────────────────────────────────────────────────────────────
    # HIGH-STRAIN BUBBLE CHART
    # ─────────────────────────────────────────────────────────────────
    sc1 = axes[0].scatter(
        df_jittered["restoration_efficiency_j"],
        df_jittered["mr_j"],
        s=sizes,
        c=df["delta_peak_I_high"],
        cmap="coolwarm",
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5,
        vmin=-max_abs_high,
        vmax=max_abs_high
    )
    axes[0].set_xlabel("Restoration Efficiency (ρ)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Transmission Multiplier (mr)", fontsize=12, fontweight="bold")
    axes[0].set_title("High-Strain Δ Peak Infections (vs baseline)\n(Bubble size = Virulence φ)", 
                      fontsize=13, fontweight="bold")
    axes[0].grid(alpha=0.3, linestyle='--')
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label("Δ Peak I_high (vs baseline)", fontsize=11, fontweight="bold")

    # ─────────────────────────────────────────────────────────────────
    # LOW-STRAIN BUBBLE CHART
    # ─────────────────────────────────────────────────────────────────
    sc2 = axes[1].scatter(
        df_jittered["restoration_efficiency_j"],
        df_jittered["mr_j"],
        s=sizes,
        c=df["delta_peak_I_low"],
        cmap="coolwarm",
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5,
        vmin=-max_abs_low,
        vmax=max_abs_low
    )
    axes[1].set_xlabel("Restoration Efficiency (ρ)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Transmission Multiplier (mr)", fontsize=12, fontweight="bold")
    axes[1].set_title("Low-Strain Δ Peak Infections (vs baseline)\n(Bubble size = Virulence φ)", 
                      fontsize=13, fontweight="bold")
    axes[1].grid(alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label("Δ Peak I_low (vs baseline)", fontsize=11, fontweight="bold")

    # Add legend for bubble sizes
    phi_vals = sorted(df["phi_transmission"].unique())
    legend_bubbles = []
    legend_labels = []
    for phi in phi_vals:
        phi_norm = (phi - phi_min) / (phi_max - phi_min)
        size = (phi_norm ** 1.5) * size_scale + 100
        legend_bubbles.append(axes[1].scatter([], [], s=size, c='gray', alpha=0.6, 
                                              edgecolors='black', linewidth=1.5))
        legend_labels.append(f'φ={phi:.1f}')

    legend = axes[1].legend(legend_bubbles, legend_labels, 
                           scatterpoints=1, title='Virulence (φ)',
                           loc='upper left', frameon=True, fontsize=10,
                           title_fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved bubble chart figure: {out_path}")
    plt.close()

def compute_R0_v8(phi_t: float, restore: float, mr: float, S_prop: float = 0.99) -> Tuple[float, float]:
    """
    Compute R0 for low and high strains under given parameters.
    
    R0 = (transmission rate) / (recovery rate)
    
    For two-strain model:
    - R0_low = (c_low * r_low) / sigma_l
    - R0_high = (c_high * r_high * m_r) / sigma_h
    
    Args:
        phi_t: High-strain transmission multiplier
        restore: Contact restoration efficiency
        mr: Transmission multiplier for drug effect
        S_prop: Proportion susceptible (for transmission scaling)
    
    Returns:
        (R0_low, R0_high)
    """
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
    )
    kappa_base = float(getattr(P, "kappa_base", 1.0))
    kappa_scale = float(getattr(P, "kappa_scale", 1.0))
    sigma = float(getattr(P, "sigma", 1.0 / 5.0))
    
    # High-strain contact rate with virulence penalty
    vir_excess_pos = max(0.0, float(phi_t) - 1.0)
    c_high = c_low * np.exp(-0.5 * vir_excess_pos)
    
    # High-strain transmission probability
    r_high = r_low * float(phi_t)
    
    # Recovery rates
    sigma_l = sigma
    sigma_h = sigma * (1.0 - vir_excess_pos)  # slowed recovery with virulence
    sigma_h = max(sigma_h, 1e-8)  # ensure strictly positive
    
    # Transmission rates including drug effect
    beta_l = c_low * r_low * S_prop
    beta_h = c_high * r_high * S_prop
    
    # Apply drug transmission multiplier
    beta_h_drug = beta_h * float(mr)
    
    # R0 values
    R0_low = beta_l / sigma_l
    R0_high = beta_h_drug / sigma_h
    
    return float(R0_low), float(R0_high)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parameter sweep: peak infections vs (restoration_efficiency, mr, phi)"
    )

    # Existing list mode
    parser.add_argument("--phi", type=float, nargs="+", default=[1.0, 1.2, 1.5, 1.8, 2.2])
    parser.add_argument("--restore", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9])
    parser.add_argument("--mr", type=float, nargs="+", default=[0.5, 0.7, 1.0])

    # New range mode (Option 1)
    parser.add_argument("--phi-min", type=float, default=None)
    parser.add_argument("--phi-max", type=float, default=None)
    parser.add_argument("--n-phi", type=int, default=9)

    parser.add_argument("--restore-min", type=float, default=None)
    parser.add_argument("--restore-max", type=float, default=None)
    parser.add_argument("--n-restore", type=int, default=9)

    parser.add_argument("--mr-min", type=float, default=None)
    parser.add_argument("--mr-max", type=float, default=None)
    parser.add_argument("--n-mr", type=int, default=9)

    parser.add_argument("--max-phi-panels", type=int, default=8)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--steps", type=int, default=365)

    args = parser.parse_args(argv)

    print("=" * 70)
    print("SEIRS v8: Unified Contact Restoration Sweep (2D Heatmaps)")
    print("=" * 70)

    phi_vals = (
        np.linspace(args.phi_min, args.phi_max, args.n_phi).tolist()
        if args.phi_min is not None and args.phi_max is not None else args.phi
    )
    restore_vals = (
        np.linspace(args.restore_min, args.restore_max, args.n_restore).tolist()
        if args.restore_min is not None and args.restore_max is not None else args.restore
    )
    mr_vals = (
        np.linspace(args.mr_min, args.mr_max, args.n_mr).tolist()
        if args.mr_min is not None and args.mr_max is not None else args.mr
    )

    print(f"φ_transmission: {phi_vals}")
    print(f"restoration_efficiency (ρ): {restore_vals}")
    print(f"mr (transmission reduction): {mr_vals}")
    print(f"Days: {args.days}, Steps: {args.steps}")
    print("=" * 70)

    # ────────────────────────────────────────────────────────────────
    # Baseline simulation (no drug: restoration=0, mr=1.0)
    # ────────────────────────────────────────────────────────────────
    print("\nRunning baseline simulation (restoration=0.0, mr=1.0)...")
    baseline_phi = phi_vals[0] if len(phi_vals) > 0 else 1.0
    t_baseline, sim_baseline = run_sim(baseline_phi, 0.0, 1.0, args.days, args.steps)
    m_baseline = metrics(t_baseline, sim_baseline)
    baseline_peak_I_high = m_baseline["peak_I_high"]
    baseline_peak_I_low = m_baseline["peak_I_low"]
    print(f"Baseline peak_I_high: {baseline_peak_I_high:.6f}")
    print(f"Baseline peak_I_low:  {baseline_peak_I_low:.6f}")
    print("=" * 70)

    n_total = len(phi_vals) * len(restore_vals) * len(mr_vals)
    results = []
    count = 0

    for phi_t in phi_vals:
        for restore in restore_vals:
            for mr in mr_vals:
                count += 1
                print(f"[{count}/{n_total}] φ={phi_t:.3f}, ρ={restore:.3f}, mr={mr:.3f}...", end=" ")
                t, sim = run_sim(phi_t, restore, mr, args.days, args.steps)
                m = metrics(t, sim)
                results.append({
                    "phi_transmission": float(phi_t),
                    "restoration_efficiency": float(restore),
                    "mr": float(mr),
                    "peak_I_high": m["peak_I_high"],
                    "peak_I_low": m["peak_I_low"],
                    "peak_I_high_total": m["peak_I_high_total"],
                    "peak_I_low_total": m["peak_I_low_total"],
                    "delta_peak_I_high": m["peak_I_high"] - baseline_peak_I_high,
                    "delta_peak_I_low": m["peak_I_low"] - baseline_peak_I_low,
                })
                print("✓")

    df = pd.DataFrame(results)
    csv_path = "Results/drug_restoration_sweep.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    print("\nFirst 10 rows:")
    print(df.head(10))

    peak_infection_heatmaps(
        df,
        "Figures/drug_restoration_sweep/peak_infection_heatmaps.png",
        max_phi_panels=args.max_phi_panels,
    )

    # Optional: keep bubble chart only for sparse grids
    if len(phi_vals) <= 10 and len(restore_vals) <= 10 and len(mr_vals) <= 10:
        bubble_chart(df, "Figures/drug_restoration_sweep/peak_infection_bubble_chart.png")

    # ────────────────────────────────────────────────────────────────
    # Print R0 values for each phi scenario
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("R0 VALUES BY VIRULENCE (φ_transmission)")
    print("=" * 70)
    print(f"{'φ_transmission':<15} {'R0_low':<15} {'R0_high':<15}")
    print("-" * 70)
    
    for phi_t in sorted(set(phi_vals)):
        R0_low, R0_high = compute_R0_v8(phi_t, restore=0.0, mr=1.0, S_prop=0.99)
        print(f"{phi_t:<15.3f} {R0_low:<15.4f} {R0_high:<15.4f}")
    
    print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())