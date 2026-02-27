#!/usr/bin/env python3
"""3D parameter sweep: peak infections vs (restoration_efficiency, mr, phi_transmission).

Axes
- restoration_efficiency: drug efficacy at masking symptoms & restoring contacts
- mr: drug_transmission_multiplier
- peak_I_high / peak_I_low: z-axis (height)

Outputs
- Figures/drug_restoration_sweep/peak_infection_landscape_3d.png

Run (from repo root)
    python Scripts/Drug_restoration_sweep_3D.py \
        --phi 1.0 1.2 1.5 1.8 2.2 \
        --restore 0.0 0.3 0.6 0.9 \
        --mr 0.5 0.7 1.0 \
        --days 365 --steps 365

Notes
- Uses SEIRS_model_v8: unified contact restoration mechanism.
- Produces TRUE 3D surface landscapes: (restoration_efficiency, mr) → peak_I
- One row per phi: high-strain peak (top), low-strain peak (bottom)
- Much cleaner: 2 rows × n_phi panels instead of 2n_phi × n_mr
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
from scipy.interpolate import griddata

matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

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


def peak_infection_landscape_3d(df: pd.DataFrame, out_path: str) -> None:
    """Create TRUE 3D surface plots: (restoration, mr) → peak_I, faceted by phi.
    
    FIXED: Z-axis labels now clarify units (fraction of population).
    CLEANED: Removed redundant colorbars (Z-axis shows values).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    phi_vals = sorted(df["phi_transmission"].unique())
    n_phi = len(phi_vals)
    
    # Two rows: high-strain (top), low-strain (bottom)
    fig = plt.figure(figsize=(6 * n_phi, 10))
    
    for phi_idx, phi_t in enumerate(phi_vals):
        df_phi = df[df["phi_transmission"] == phi_t]
        
        # Extract grid data
        restore_vals = np.array(sorted(df_phi["restoration_efficiency"].unique()))
        mr_vals = np.array(sorted(df_phi["mr"].unique()))
        
        # Create meshgrid
        restore_grid, mr_grid = np.meshgrid(restore_vals, mr_vals, indexing="ij")
        
        # ─────────────────────────────────────────────────────────────────
        # HIGH-STRAIN PEAK
        # ─────────────────────────────────────────────────────────────────
        ax_h = fig.add_subplot(2, n_phi, phi_idx + 1, projection="3d")
        
        peak_h_grid = np.zeros_like(restore_grid)
        for i, r in enumerate(restore_vals):
            for j, m in enumerate(mr_vals):
                subset = df_phi[(df_phi["restoration_efficiency"] == r) & (df_phi["mr"] == m)]
                peak_h_grid[i, j] = subset["peak_I_high"].values[0] if len(subset) > 0 else 0.0
        
        # Surface plot
        surf_h = ax_h.plot_surface(
            restore_grid, mr_grid, peak_h_grid,
            cmap=cm.viridis, alpha=0.8, edgecolor="none"
        )
        ax_h.set_xlabel("Restoration Efficiency (ρ)", fontsize=10)
        ax_h.set_ylabel("Transmission Multiplier (mr)", fontsize=10)
        ax_h.set_zlabel("Peak I_high (fraction)", fontsize=10)
        ax_h.set_title(f"φ={phi_t:.1f} — High-strain Peak", fontsize=11, fontweight="bold")
        ax_h.view_init(elev=25, azim=45)
        # ← REMOVED: colorbar (redundant with Z-axis)
        
        # ─────────────────────────────────────────────────────────────────
        # LOW-STRAIN PEAK
        # ─────────────────────────────────────────────────────────────────
        ax_l = fig.add_subplot(2, n_phi, n_phi + phi_idx + 1, projection="3d")
        
        peak_l_grid = np.zeros_like(restore_grid)
        for i, r in enumerate(restore_vals):
            for j, m in enumerate(mr_vals):
                subset = df_phi[(df_phi["restoration_efficiency"] == r) & (df_phi["mr"] == m)]
                peak_l_grid[i, j] = subset["peak_I_low"].values[0] if len(subset) > 0 else 0.0
        
        # Surface plot
        surf_l = ax_l.plot_surface(
            restore_grid, mr_grid, peak_l_grid,
            cmap=cm.plasma, alpha=0.8, edgecolor="none"
        )
        ax_l.set_xlabel("Restoration Efficiency (ρ)", fontsize=10)
        ax_l.set_ylabel("Transmission Multiplier (mr)", fontsize=10)
        ax_l.set_zlabel("Peak I_low (fraction)", fontsize=10)
        ax_l.set_title(f"φ={phi_t:.1f} — Low-strain Peak", fontsize=11, fontweight="bold")
        ax_l.view_init(elev=25, azim=45)
        # ← REMOVED: colorbar (redundant with Z-axis)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved 3D surface figure: {out_path}")
    plt.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="3D parameter sweep: peak infections vs (restoration_efficiency, mr, phi)"
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
    print("SEIRS v8: Unified Contact Restoration Sweep (3D Surfaces)")
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
    
    fig_path = "Figures/drug_restoration_sweep/peak_infection_landscape_3d.png"
    peak_infection_landscape_3d(df, fig_path)
    
    print("\n" + "=" * 70)
    print("Done! All results now in consistent units (fractions).")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())