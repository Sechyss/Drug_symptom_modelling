#!/usr/bin/env python3
"""
Explore virulence effects using SEIRS_model_v6 by sweeping phi_transmission.

Outputs:
- Figures/virulence_v6_timeseries.png
- Tables/virulence_v6_summary.csv

Run (from repo root):
  python Scripts/Exploration_virulence.py --phi 1.0 1.2 1.5 1.8 2.2 --days 365
"""
import os
import sys
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.integrate import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.SEIRS_Models import SEIRS_model_v6
from Models import params as P

COLS = ["S", "Eh", "Indh", "Idh", "Rh", "El", "Indl", "Idl", "Rl"]


def initial_conditions(normalize: bool = True) -> np.ndarray:
    y0 = np.array([
        getattr(P, "S", 10000),
        getattr(P, "Eh", 0),
        getattr(P, "Indh", 5),
        getattr(P, "Idh", 0),
        getattr(P, "Rh", 0),
        getattr(P, "El", 0),
        getattr(P, "Indl", 5),
        getattr(P, "Idl", 0),
        getattr(P, "Rl", 0),
    ], dtype=float)
    y0 = np.maximum(y0, 0.0)
    if normalize:
        s = y0.sum()
        if s <= 0 or not np.isfinite(s):
            raise ValueError("Initial conditions must have positive finite sum.")
        y0 = y0 / s
    return y0


def params_tuple_v6(phi_t: float) -> Tuple[float, ...]:
    """
    SEIRS_model_v6 expects (14):
      (contact_rate_low, transmission_probability_low, phi_transmission,
       drug_contact_multiplier, drug_transmission_multiplier,
       birth_rate, death_rate, delta,
       kappa_base, kappa_scale, phi_recover, sigma, tau, theta)
    """
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low",
                getattr(P, "transmission_probability", 0.025))
    )

    # Drug effects (read from params; default to 1.0 if not present)
    drug_contact_multiplier = float(getattr(P, "drug_contact_multiplier", 1.0))
    drug_transmission_multiplier = float(getattr(P, "drug_transmission_multiplier", 1.0))

    birth_rate = float(getattr(P, "birth_rate", 0.0))
    death_rate = float(getattr(P, "death_rate", 0.0))
    delta = float(getattr(P, "delta", 0.0))
    kappa_base = float(getattr(P, "kappa_base", 1.0))
    kappa_scale = float(getattr(P, "kappa_scale", 1.0))
    phi_recover = float(getattr(P, "phi_recover", 1.0))
    sigma = float(getattr(P, "sigma", 1 / 5))
    tau = float(getattr(P, "tau", 1 / 3))
    theta = float(getattr(P, "theta", 0.3))

    return (
        c_low,
        r_low,
        float(phi_t),
        drug_contact_multiplier,
        drug_transmission_multiplier,
        birth_rate,
        death_rate,
        delta,
        kappa_base,
        kappa_scale,
        phi_recover,
        sigma,
        tau,
        theta,
    )


def run_sim(phi_t: float, days: int, steps: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t = np.linspace(0, days, steps)
    y0 = initial_conditions(normalize=True)
    sol = odeint(SEIRS_model_v6, y0, t, args=(params_tuple_v6(phi_t),))
    sim = {k: sol[:, i] for i, k in enumerate(COLS)}
    sim["I_high"] = sim["Indh"] + sim["Idh"]
    sim["I_low"] = sim["Indl"] + sim["Idl"]
    sim["I_total"] = sim["I_high"] + sim["I_low"]
    sim["high_frac"] = sim["I_high"] / (sim["I_total"] + 1e-12)
    return t, sim


def metrics(t: np.ndarray, sim: Dict[str, np.ndarray]) -> Dict[str, float]:
    I = sim["I_total"]
    S = sim["S"]
    ih = sim["I_high"]
    il = sim["I_low"]
    out = {
        "peak_I_total": float(np.max(I)),
        "t_peak_I_total": float(t[int(np.argmax(I))]),
        "S_end": float(S[-1]),
        "attack_rate": float(1.0 - S[-1]),
        "peak_I_high": float(np.max(ih)),
        "peak_I_low": float(np.max(il)),
        "high_frac_end": float(sim["high_frac"][-1]),
        "high_frac_peak_time": float(sim["high_frac"][int(np.argmax(I))]),
    }
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Virulence sweep using SEIRS_model_v6")
    ap.add_argument("--phi", nargs="+", type=float, default=None,
                    help="Virulence values for phi_transmission (space-separated).")
    ap.add_argument("--days", type=int, default=int(getattr(P, "t_max", 365)))
    ap.add_argument("--steps", type=int, default=int(getattr(P, "t_steps", 365)))
    ap.add_argument("--baseline-phi", type=float, default=float(getattr(P, "phi_transmission", 1.5)))
    args = ap.parse_args(argv)

    phi_vals = args.phi if args.phi is not None else [1.0, 1.2, args.baseline_phi, 1.8, 2.2, 2.8]
    # ensure unique + sorted, keep baseline included
    phi_vals = sorted(set([float(x) for x in phi_vals] + [float(args.baseline_phi)]))

    fig_dir = os.path.join("Figures")
    tab_dir = os.path.join("Tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # Baseline
    t0, sim0 = run_sim(args.baseline_phi, args.days, args.steps)
    m0 = metrics(t0, sim0)

    rows = []
    sims = {}

    for phi in phi_vals:
        t, sim = run_sim(phi, args.days, args.steps)
        m = metrics(t, sim)
        row = {"phi_transmission": phi, **m}
        # deltas vs baseline
        row.update({
            "d_peak_I_total": row["peak_I_total"] - m0["peak_I_total"],
            "d_attack_rate": row["attack_rate"] - m0["attack_rate"],
            "d_S_end": row["S_end"] - m0["S_end"],
            "d_high_frac_end": row["high_frac_end"] - m0["high_frac_end"],
        })
        rows.append(row)
        sims[phi] = (t, sim)

    df = pd.DataFrame(rows).sort_values("phi_transmission")
    out_csv = os.path.join(tab_dir, "virulence_v6_summary.csv")
    df.to_csv(out_csv, index=False)

    phi_list = df["phi_transmission"].tolist()
    n_phi = len(phi_list)
    cmap = plt.get_cmap("viridis")

    # Plot 1: total infected + high fraction
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i, phi in enumerate(phi_list):
        t, sim = sims[phi]
        is_baseline = np.isclose(phi, args.baseline_phi)
        lw = 2.5 if is_baseline else 1.5
        alpha = 1.0 if is_baseline else 0.8
        color = cmap(i / (n_phi - 1)) if n_phi > 1 else cmap(0.5)

        label = f"phi={phi:g}" + (" (baseline)" if is_baseline else "")
        axes[0].plot(t, sim["I_total"], lw=lw, alpha=alpha, label=label, color=color)
        axes[1].plot(t, sim["high_frac"], lw=lw, alpha=alpha, color=color)

    axes[0].set_title("SEIRS v6: Total infectious vs virulence (phi_transmission)")
    axes[0].set_ylabel("I_total (proportion)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("High-strain fraction among infectious")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("I_high / (I_high + I_low)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    out_fig_I = os.path.join(fig_dir, "virulence_v6_I_timeseries.png")
    plt.tight_layout()
    plt.savefig(out_fig_I, dpi=600)
    plt.close()

    # Plot 2: exposed (Eh/El) + high fraction
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for i, phi in enumerate(phi_list):
        t, sim = sims[phi]
        is_baseline = np.isclose(phi, args.baseline_phi)
        lw = 2.5 if is_baseline else 1.5
        alpha = 1.0 if is_baseline else 0.8
        color = cmap(i / (n_phi - 1)) if n_phi > 1 else cmap(0.5)

        # Same color for a given phi; Eh solid, El dashed
        axes[0].plot(t, sim["Eh"], lw=lw, alpha=alpha, color=color,
                     label=f"phi={phi:g} (Eh)", linestyle="-")
        axes[0].plot(t, sim["El"], lw=lw, alpha=alpha, color=color,
                     label=f"phi={phi:g} (El)", linestyle="--")

        axes[1].plot(t, sim["high_frac"], lw=lw, alpha=alpha, color=color)

    axes[0].set_title("SEIRS v6: Exposed (Eh/El) vs virulence (phi_transmission)")
    axes[0].set_ylabel("E (proportion)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("High-strain fraction among infectious")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("I_high / (I_high + I_low)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    out_fig_E = os.path.join(fig_dir, "virulence_v6_E_timeseries.png")
    plt.tight_layout()
    plt.savefig(out_fig_E, dpi=600)
    plt.close()

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_fig_I}")
    print(f"Saved: {out_fig_E}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())