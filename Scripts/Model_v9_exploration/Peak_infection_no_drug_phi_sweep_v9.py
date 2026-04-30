#!/usr/bin/env python3
"""Sweep virulence in SEIRS v9 and plot no-drug peak infection by strain.

Outputs:
  Tables/peak_infection_no_drug_phi_sweep_v9.csv
  Figures/Model_v9_exploration/peak_infection_no_drug_phi_sweep_v9.png
  Figures/Model_v9_exploration/peak_infection_no_drug_phi_sweep_v9.svg

Run from repo root:
  python Scripts/Model_v9_exploration/Peak_infection_no_drug_phi_sweep_v9.py \
      --phi 1.0 1.33 1.67 2.0 --days 365 --steps 365
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from scipy.integrate import odeint

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Models.SEIRS_Models import SEIRS_model_v9
from Models import params as P


COLS = ["S", "Eh", "Indh", "Idh", "Rh", "El", "Indl", "Idl", "Rl"]
DEFAULT_PHI = [1.0, 2.0, 3.0, 4.0]


def initial_conditions(normalize: bool = True) -> np.ndarray:
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
        total = y0.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError("Initial conditions must have positive finite sum.")
        y0 = y0 / total
    return y0


def params_tuple_v9(phi_t: float) -> Tuple[float, ...]:
    return (
        float(getattr(P, "contact_rate", 10.0)),
        float(getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))),
        float(phi_t),
        0.0,
        1.0,
        float(getattr(P, "kappa_base", 1.0)),
        float(getattr(P, "kappa_scale", 1.0)),
        float(getattr(P, "sigma", 1.0 / 5.0)),
        float(getattr(P, "tau", 1.0 / 3.0)),
        0.0,
    )


def run_sim(phi_t: float, days: int, steps: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t = np.linspace(0.0, float(days), int(steps))
    sol = odeint(SEIRS_model_v9, initial_conditions(normalize=True), t, args=(params_tuple_v9(phi_t),))
    sim = {col: sol[:, idx] for idx, col in enumerate(COLS)}
    sim["I_high"] = sim["Indh"] + sim["Idh"]
    sim["I_low"] = sim["Indl"] + sim["Idl"]
    sim["I_total"] = sim["I_high"] + sim["I_low"]
    return t, sim


def metrics(t: np.ndarray, sim: Dict[str, np.ndarray], phi_t: float) -> Dict[str, float]:
    i_high = sim["I_high"]
    i_low = sim["I_low"]
    i_total = sim["I_total"]
    return {
        "phi_transmission": float(phi_t),
        "peak_I_high": float(np.max(i_high)),
        "time_peak_I_high": float(t[int(np.argmax(i_high))]),
        "peak_I_low": float(np.max(i_low)),
        "time_peak_I_low": float(t[int(np.argmax(i_low))]),
        "peak_I_total": float(np.max(i_total)),
        "time_peak_I_total": float(t[int(np.argmax(i_total))]),
        "final_S": float(sim["S"][-1]),
        "attack_rate": float(1.0 - sim["S"][-1]),
    }


def plot_results(results: List[Tuple[float, np.ndarray, Dict[str, np.ndarray]]], summary: pd.DataFrame, out_base: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,
            "savefig.bbox": "tight",
        }
    )

    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(14.0, 10.0), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax_low = fig.add_subplot(gs[0, 0])
    ax_high = fig.add_subplot(gs[0, 1], sharey=ax_low)
    ax_peak = fig.add_subplot(gs[1, :])
    
    cmap = plt.get_cmap("viridis")
    n_phi = max(len(results), 1)

    # Plot low virulence strain on left panel, high virulence on right panel
    for idx, (phi_t, t, sim) in enumerate(results):
        color = cmap(idx / max(n_phi - 1, 1))
        ax_low.plot(t, sim["I_low"], color=color, ls="-", label=f"phi={phi_t:.2f}")
        ax_high.plot(t, sim["I_high"], color=color, ls="-", label=f"phi={phi_t:.2f}")

    ax_low.set_title("Low virulence strain")
    ax_high.set_title("High virulence strain")
    ax_low.set_xlabel("Time (days)")
    ax_high.set_xlabel("Time (days)")
    ax_low.set_ylabel("Infectious proportion")
    
    # Panel labels
    ax_low.text(0.01, 0.98, "A", transform=ax_low.transAxes, ha="left", va="top", fontweight="bold", fontsize=14)
    ax_high.text(0.01, 0.98, "B", transform=ax_high.transAxes, ha="left", va="top", fontweight="bold", fontsize=14)

    for ax in (ax_low, ax_high):
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    phi_handles = [
        Line2D([0], [0], color=cmap(idx / max(n_phi - 1, 1)), lw=2.5, label=f"phi={phi_t:.2f}")
        for idx, (phi_t, _, _) in enumerate(results)
    ]
    ax_high.legend(handles=phi_handles, title="Virulence", loc="center right", frameon=False)

    ax_peak.plot(
        summary["phi_transmission"],
        summary["peak_I_high"],
        color="#d95f02",
        marker="o",
        label="High strain peak",
    )
    ax_peak.plot(
        summary["phi_transmission"],
        summary["peak_I_low"],
        color="#1b9e77",
        marker="s",
        label="Low strain peak",
    )
    ax_peak.set_title("Peak infection vs virulence (no drug)")
    ax_peak.set_xlabel("phi_transmission")
    ax_peak.set_ylabel("Peak infectious proportion")
    ax_peak.set_xticks(summary["phi_transmission"])
    ax_peak.grid(True, alpha=0.25)
    ax_peak.spines["top"].set_visible(False)
    ax_peak.spines["right"].set_visible(False)
    ax_peak.legend(frameon=False, loc="best")
    
    # Panel label C
    ax_peak.text(0.01, 0.98, "C", transform=ax_peak.transAxes, ha="left", va="top", fontweight="bold", fontsize=14)

    fig.savefig(out_base + ".png", dpi=700)
    fig.savefig(out_base + ".svg")
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="No-drug virulence sweep for SEIRS v9")
    parser.add_argument("--phi", nargs="+", type=float, default=None, help="phi_transmission values")
    parser.add_argument("--days", type=int, default=int(getattr(P, "t_max", 365)))
    parser.add_argument("--steps", type=int, default=int(getattr(P, "t_steps", 365)))
    args = parser.parse_args(argv)

    phi_vals = sorted(set(float(phi) for phi in (args.phi or DEFAULT_PHI)))
    results: List[Tuple[float, np.ndarray, Dict[str, np.ndarray]]] = []
    rows: List[Dict[str, float]] = []

    print("=" * 72)
    print("No-drug virulence sweep")
    print("=" * 72)
    print(f"phi_transmission values: {[f'{phi:.2f}' for phi in phi_vals]}")
    print("No-drug settings: restoration_efficiency=0.0, m_r=1.0, theta=0.0")
    print("=" * 72)

    for phi_t in phi_vals:
        t, sim = run_sim(phi_t, args.days, args.steps)
        row = metrics(t, sim, phi_t)
        results.append((phi_t, t, sim))
        rows.append(row)
        print(
            f"phi={phi_t:.2f} | peak_I_high={row['peak_I_high']:.6f} at day {row['time_peak_I_high']:.2f} | "
            f"peak_I_low={row['peak_I_low']:.6f} at day {row['time_peak_I_low']:.2f}"
        )

    summary = pd.DataFrame(rows).sort_values("phi_transmission")

    table_path = os.path.join(ROOT_DIR, "Tables", "peak_infection_no_drug_phi_sweep_v9.csv")
    figure_base = os.path.join(ROOT_DIR, "Figures", "Model_v9_exploration", "peak_infection_no_drug_phi_sweep_v9")
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_base), exist_ok=True)

    summary.to_csv(table_path, index=False)
    plot_results(results, summary, figure_base)

    print(f"Saved summary table: {table_path}")
    print(f"Saved figures: {figure_base}.png and {figure_base}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())