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
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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
    """SEIRS_model_v8 expects 14 params."""
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
    )

    birth_rate = float(getattr(P, "birth_rate", 0.0))
    death_rate = float(getattr(P, "death_rate", 0.0))
    delta = float(getattr(P, "delta", 0.0))
    kappa_base = float(getattr(P, "kappa_base", 1.0))
    kappa_scale = float(getattr(P, "kappa_scale", 1.0))
    phi_recover = float(getattr(P, "phi_recover", 1.0))
    sigma = float(getattr(P, "sigma", 1.0 / 5.0))
    tau = float(getattr(P, "tau", 1.0 / 3.0))
    theta = float(getattr(P, "theta", 0.3))

    return (
        c_low, r_low, float(phi_t),
        float(restore), float(mr),
        birth_rate, death_rate, delta,
        kappa_base, kappa_scale,
        phi_recover, sigma, tau, theta
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


def peak_infection_heatmaps(df: pd.DataFrame, out_path: str) -> None:
    """Create 2D heatmaps: (restoration, mr) → peak_I, faceted by phi.

    Layout: 2 rows × n_phi columns
    - Row 1: High-strain peaks
    - Row 2: Low-strain peaks
    Each column is a different phi value.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    phi_vals = sorted(df["phi_transmission"].unique())
    n_phi = len(phi_vals)

    fig, axes = plt.subplots(
        2, n_phi, figsize=(4.2 * n_phi, 7.5), squeeze=False, constrained_layout=True
    )

    # Global scales for comparability across phi panels
    vmin_high = df["peak_I_high"].min()
    vmax_high = df["peak_I_high"].max()
    vmin_low = df["peak_I_low"].min()
    vmax_low = df["peak_I_low"].max()

    im_h_ref = None
    im_l_ref = None

    for phi_idx, phi_t in enumerate(phi_vals):
        df_phi = df[df["phi_transmission"] == phi_t]

        # High-strain panel
        ax_h = axes[0, phi_idx]
        pivot_high = df_phi.pivot(
            index="mr",
            columns="restoration_efficiency",
            values="peak_I_high"
        ).sort_index().sort_index(axis=1)

        im_h = ax_h.imshow(
            pivot_high.values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=vmin_high,
            vmax=vmax_high,
            extent=[
                pivot_high.columns.min(), pivot_high.columns.max(),
                pivot_high.index.min(), pivot_high.index.max()
            ]
        )
        if im_h_ref is None:
            im_h_ref = im_h

        ax_h.set_title(f"φ={phi_t:.1f} — High-strain", fontsize=11, fontweight="bold")
        ax_h.set_xlabel("Restoration Efficiency (ρ)", fontsize=10)
        if phi_idx == 0:
            ax_h.set_ylabel("Transmission Multiplier (mr)", fontsize=10)

        ax_h.set_xticks(pivot_high.columns.values)
        ax_h.set_yticks(pivot_high.index.values)

        # Low-strain panel
        ax_l = axes[1, phi_idx]
        pivot_low = df_phi.pivot(
            index="mr",
            columns="restoration_efficiency",
            values="peak_I_low"
        ).sort_index().sort_index(axis=1)

        im_l = ax_l.imshow(
            pivot_low.values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="plasma",
            vmin=vmin_low,
            vmax=vmax_low,
            extent=[
                pivot_low.columns.min(), pivot_low.columns.max(),
                pivot_low.index.min(), pivot_low.index.max()
            ]
        )
        if im_l_ref is None:
            im_l_ref = im_l

        ax_l.set_title(f"φ={phi_t:.1f} — Low-strain", fontsize=11, fontweight="bold")
        ax_l.set_xlabel("Restoration Efficiency (ρ)", fontsize=10)
        if phi_idx == 0:
            ax_l.set_ylabel("Transmission Multiplier (mr)", fontsize=10)

        ax_l.set_xticks(pivot_low.columns.values)
        ax_l.set_yticks(pivot_low.index.values)

    # Shared colorbar for top row
    cbar_h = fig.colorbar(
        im_h_ref, ax=axes[0, :], orientation="vertical", fraction=0.02, pad=0.02
    )
    cbar_h.set_label("Peak I_high (fraction)", fontsize=10)

    # Shared colorbar for bottom row
    cbar_l = fig.colorbar(
        im_l_ref, ax=axes[1, :], orientation="vertical", fraction=0.02, pad=0.02
    )
    cbar_l.set_label("Peak I_low (fraction)", fontsize=10)

    # Remove tight_layout() to prevent warning
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved heatmap figure: {out_path}")
    plt.close(fig)


def bubble_chart(df: pd.DataFrame, out_path: str) -> None:
    """Bubble chart with jitter to separate overlapping bubbles."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Add jitter to separate overlapping points
    np.random.seed(42)
    jitter_strength = 0.035  # ← INCREASED from 0.02 for better separation
    df_jittered = df.copy()
    df_jittered["restoration_efficiency_j"] = df["restoration_efficiency"] + np.random.normal(0, jitter_strength, len(df))
    df_jittered["mr_j"] = df["mr"] + np.random.normal(0, jitter_strength, len(df))

    # Amplify bubble size differences
    phi_min = df["phi_transmission"].min()
    phi_max = df["phi_transmission"].max()
    phi_normalized = (df["phi_transmission"] - phi_min) / (phi_max - phi_min)
    size_scale = 500
    sizes = (phi_normalized ** 1.5) * size_scale + 100

    # ─────────────────────────────────────────────────────────────────
    # HIGH-STRAIN BUBBLE CHART
    # ─────────────────────────────────────────────────────────────────
    sc1 = axes[0].scatter(
        df_jittered["restoration_efficiency_j"],
        df_jittered["mr_j"],
        s=sizes,
        c=df["peak_I_high"],
        cmap="viridis",
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5,
        vmin=df["peak_I_high"].min(),
        vmax=df["peak_I_high"].max()
    )
    axes[0].set_xlabel("Restoration Efficiency (ρ)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Transmission Multiplier (mr)", fontsize=12, fontweight="bold")
    axes[0].set_title("High-Strain Peak Infections\n(Bubble size = Virulence φ)", 
                      fontsize=13, fontweight="bold")
    axes[0].grid(alpha=0.3, linestyle='--')
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label("Peak I_high (fraction)", fontsize=11, fontweight="bold")

    # ─────────────────────────────────────────────────────────────────
    # LOW-STRAIN BUBBLE CHART
    # ─────────────────────────────────────────────────────────────────
    sc2 = axes[1].scatter(
        df_jittered["restoration_efficiency_j"],
        df_jittered["mr_j"],
        s=sizes,
        c=df["peak_I_low"],
        cmap="plasma",
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5,
        vmin=df["peak_I_low"].min(),
        vmax=df["peak_I_low"].max()
    )
    axes[1].set_xlabel("Restoration Efficiency (ρ)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Transmission Multiplier (mr)", fontsize=12, fontweight="bold")
    axes[1].set_title("Low-Strain Peak Infections\n(Bubble size = Virulence φ)", 
                      fontsize=13, fontweight="bold")
    axes[1].grid(alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(sc2, ax=axes[1])
    cbar2.set_label("Peak I_low (fraction)", fontsize=11, fontweight="bold")

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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parameter sweep: peak infections vs (restoration_efficiency, mr, phi)"
    )
    parser.add_argument("--phi", type=float, nargs="+", default=[1.0, 1.2, 1.5, 1.8, 2.2],
                       help="phi_transmission values")
    parser.add_argument("--restore", type=float, nargs="+", default=[0.0, 0.3, 0.6, 0.9],
                       help="restoration_efficiency values")
    parser.add_argument("--mr", type=float, nargs="+", default=[0.5, 0.7, 1.0],
                       help="drug_transmission_multiplier values")
    parser.add_argument("--days", type=int, default=365, help="simulation duration")
    parser.add_argument("--steps", type=int, default=365, help="number of time steps")
    
    args = parser.parse_args(argv)
    
    print("=" * 70)
    print("SEIRS v8: Unified Contact Restoration Sweep (2D Heatmaps)")
    print("=" * 70)
    print(f"φ_transmission: {args.phi}")
    print(f"restoration_efficiency (ρ): {args.restore}")
    print(f"mr (transmission reduction): {args.mr}")
    print(f"Days: {args.days}, Steps: {args.steps}")
    print("=" * 70)
    print("NOTE: All results in FRACTIONS of population (normalized initial conditions)")
    print("=" * 70)
    
    n_total = len(args.phi) * len(args.restore) * len(args.mr)
    results = []
    
    count = 0
    for phi_t in args.phi:
        for restore in args.restore:
            for mr in args.mr:
                count += 1
                print(f"[{count}/{n_total}] φ={phi_t:.2f}, ρ={restore:.2f}, mr={mr:.2f}...", end=" ")
                
                t, sim = run_sim(phi_t, restore, mr, args.days, args.steps)
                m = metrics(t, sim)
                
                results.append({
                    "phi_transmission": phi_t,
                    "restoration_efficiency": restore,
                    "mr": mr,
                    "peak_I_high": m["peak_I_high"],
                    "peak_I_low": m["peak_I_low"],
                    "peak_I_high_total": m["peak_I_high_total"],
                    "peak_I_low_total": m["peak_I_low_total"],
                })
                print("✓")
    
    df = pd.DataFrame(results)
    csv_path = "Results/drug_restoration_sweep.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    print("\nFirst 10 rows (all values are fractions of population):")
    print(df.head(10))
    
    fig_path_heatmap = "Figures/drug_restoration_sweep/peak_infection_heatmaps.png"
    peak_infection_heatmaps(df, fig_path_heatmap)
    
    fig_path_bubble = "Figures/drug_restoration_sweep/peak_infection_bubble_chart.png"
    bubble_chart(df, fig_path_bubble)
    
    print("\n" + "=" * 70)
    print("Done! All figures generated successfully.")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())