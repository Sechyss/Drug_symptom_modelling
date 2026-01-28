#!/usr/bin/env python3
"""
Drug exploration with SEIRS_model_v3.

Scenario:
- A drug increases contact rate (behavior) but decreases per-contact transmission
  for treated infectious individuals (Idh, Idl). Untreated infectious (Indh, Indl)
  keep baseline behavior/transmissibility.

Parameters:
- All base epidemiological parameters are loaded from Models/params.py.
- Drug effect multipliers can be provided in params.py (defaults),
  or overridden on the CLI for sweeps.

Usage (from repo root):
  # Run baseline vs drug-from-params scenario
  python Scripts/Drug_exploration.py run --days 200

  # Sweep drug multipliers and visualize peak force of infection
  python Scripts/Drug_exploration.py sweep --days 200

Outputs:
  - Figures/drug_v3_time_series.png
  - Figures/drug_v3_sweep_heatmap_foi.png (for sweep)
  - Tables/drug_v3_summary.csv
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Tuple, List, Optional

# Import model definitions and parameter defaults
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Models.SEIRS_Models import SEIRS_model_v3
from Models import params as P

# Column order for state vector consistency
COLS = ['S','Eh','Indh','Idh','Rh','El','Indl','Idl','Rl']


def params_tuple_v3(m_c: float, m_r: float) -> Tuple[float, ...]:
    """
    Build the ordered parameter tuple expected by SEIRS_model_v3.

    Order (length=15):
      (contact_rate_low, transmission_probability_low, contact_rate_high, phi_transmission,
       drug_contact_multiplier, drug_transmission_multiplier,
       birth_rate, death_rate, delta, kappa_base, kappa_scale, phi_recover, sigma, tau, theta)

    Args:
      m_c: drug_contact_multiplier (>=0), multiplies contact rate of treated infectious.
      m_r: drug_transmission_multiplier (>=0), multiplies per-contact transmission prob of treated infectious.

    Notes:
      - transmission_probability_low falls back to P.transmission_probability if alias not defined.
    """
    return (
        P.contact_rate,  # c_low
        getattr(P, 'transmission_probability_low', P.transmission_probability),  # r_low
        P.contact_rate_high,            # c_high (untreated high-virulence)
        P.phi_transmission,             # phi_t
        m_c,                            # drug_contact_multiplier
        m_r,                            # drug_transmission_multiplier
        P.birth_rate, P.death_rate, P.delta,
        getattr(P, 'kappa_base', 1.0), getattr(P, 'kappa_scale', 1.0),
        P.phi_recover, P.sigma, P.tau, P.theta
    )


def initial_conditions() -> np.ndarray:
    """
    Construct initial condition vector and normalize to proportions (sum=1).

    Returns:
      y0: np.ndarray shape (9,), ordered as in COLS.
    """
    y0 = np.array([P.S, P.Eh, P.Indh, P.Idh, P.Rh, P.El, P.Indl, P.Idl, P.Rl], dtype=float)
    return y0 / y0.sum()


def run_sim(m_c: float, m_r: float, days: int = 200) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Integrate model v3 for given drug multipliers over specified horizon.

    Args:
      m_c: treated contact multiplier.
      m_r: treated transmission multiplier.
      days: simulation horizon in days.

    Returns:
      t: time array of length 'days' (1 step/day).
      sim: dict mapping each compartment name to its trajectory.
    """
    t = np.linspace(0, days, days)
    sol = odeint(SEIRS_model_v3, initial_conditions(), t, args=(params_tuple_v3(m_c, m_r),))
    sim = {k: sol[:, i] for i, k in enumerate(COLS)}
    return t, sim


def theta_effects() -> Tuple[float, float]:
    """
    Compute effective split-at-onset treatment coverages for low/high strains:
      theta_low_eff  = min(kappa_base, 1/theta) * theta
      theta_high_eff = min(kappa_base*(1+kappa_scale*(phi_t-1)), 1/theta) * theta

    Returns:
      (theta_low_eff, theta_high_eff)
    """
    theta = P.theta
    kb = getattr(P, 'kappa_base', 1.0)
    ks = getattr(P, 'kappa_scale', 1.0)
    k_low = kb
    k_high = kb * (1.0 + ks * (P.phi_transmission - 1.0))
    cap = lambda k: min(k, 1.0 / max(theta, 1e-12))  # cap ensures k*theta <= 1
    return cap(k_low) * theta, cap(k_high) * theta


def compute_R0_v3(m_c: float, m_r: float) -> Tuple[float, float]:
    """
    Compute R0 for low / high strains under split-at-onset with drug-modified treated classes.

    Effective betas (mixture of untreated and treated at onset):
      β_low_eff  = (1-θ_l)*β_l_u + θ_l*β_l_t
      β_high_eff = (1-θ_h)*β_h_u + θ_h*β_h_t

    Returns:
      (R0_low, R0_high)
    """
    th_l, th_h = theta_effects()
    c_l, r_l, c_h, phi = P.contact_rate, P.transmission_probability, P.contact_rate_high, P.phi_transmission
    beta_l_u = c_l * r_l
    beta_l_t = (c_l * m_c) * (r_l * m_r)
    beta_h_u = c_h * r_l * phi
    beta_h_t = (c_h * m_c) * (r_l * m_r) * phi
    beta_l_eff = (1.0 - th_l) * beta_l_u + th_l * beta_l_t
    beta_h_eff = (1.0 - th_h) * beta_h_u + th_h * beta_h_t
    R0_l = beta_l_eff / P.sigma
    R0_h = beta_h_eff / (P.phi_recover * P.sigma)
    return R0_l, R0_h


def foi_series(sim: Dict[str, np.ndarray], m_c: float, m_r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Force-of-infection series with drug effects applied to treated classes.

    λ_low(t)  = β_l_u * Indl + β_l_t * Idl
    λ_high(t) = β_h_u * Indh + β_h_t * Idh
    λ_total   = λ_low + λ_high

    Args:
      sim: time series for all compartments.
      m_c, m_r: drug multipliers for treated infectious.

    Returns:
      (λ_total, λ_high, λ_low)
    """
    c_l, r_l, c_h, phi = P.contact_rate, P.transmission_probability, P.contact_rate_high, P.phi_transmission
    beta_l_u = c_l * r_l
    beta_l_t = (c_l * m_c) * (r_l * m_r)
    beta_h_u = c_h * r_l * phi
    beta_h_t = (c_h * m_c) * (r_l * m_r) * phi
    lam_l = beta_l_u * sim['Indl'] + beta_l_t * sim['Idl']
    lam_h = beta_h_u * sim['Indh'] + beta_h_t * sim['Idh']
    return lam_l + lam_h, lam_h, lam_l


def cmd_run(args: argparse.Namespace) -> None:
    """
    Run two scenarios and save plots/summary:
      1) Baseline (no drug effect on treated: m_c=1, m_r=1).
      2) Drug scenario using params.py: drug_contact_multiplier, drug_transmission_multiplier.
    """
    os.makedirs('../../Figures', exist_ok=True)
    os.makedirs('../../Tables', exist_ok=True)

    # Baseline (treated behave like untreated)
    t0, sim0 = run_sim(m_c=1.0, m_r=1.0, days=args.days)
    R0l0, R0h0 = compute_R0_v3(1.0, 1.0)

    # Drug scenario from params.py (contact up, transmission down for treated)
    m_c = getattr(P, 'drug_contact_multiplier', 1.2)
    m_r = getattr(P, 'drug_transmission_multiplier', 0.5)
    t1, sim1 = run_sim(m_c=m_c, m_r=m_r, days=args.days)
    R0l1, R0h1 = compute_R0_v3(m_c, m_r)

    # Plot: untreated infectious time series and λ(t)
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    # Untreated infectious (Ind) by strain
    axes[0].plot(t0, sim0['Indh'], label='High, baseline')
    axes[0].plot(t1, sim1['Indh'], label=f'High, drug mc={m_c}, mr={m_r}')
    axes[0].plot(t0, sim0['Indl'], '--', label='Low, baseline')
    axes[0].plot(t1, sim1['Indl'], '--', label='Low, drug')
    axes[0].set_title('Untreated infectious (Ind)')
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Proportion')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Force of infection
    lam0, _, _ = foi_series(sim0, 1.0, 1.0)
    lam1, _, _ = foi_series(sim1, m_c, m_r)
    axes[1].plot(t0, lam0, label=f'λ baseline (R0l={R0l0:.2f}, R0h={R0h0:.2f})')
    axes[1].plot(t1, lam1, label=f'λ drug (R0l={R0l1:.2f}, R0h={R0h1:.2f})')
    axes[1].set_title('Force of infection λ(t)')
    axes[1].set_xlabel('Days')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    f = '../Figures/drug_v3_time_series.png'
    plt.savefig(f, dpi=600)
    plt.close()
    print(f"Saved: {f}")

    # Summary CSV
    df = pd.DataFrame([
        {'scenario': 'baseline', 'm_c': 1.0, 'm_r': 1.0, 'R0_low': R0l0, 'R0_high': R0h0,
         'peak_Indh': float(np.max(sim0['Indh'])), 'peak_Indl': float(np.max(sim0['Indl']))},
        {'scenario': 'drug', 'm_c': m_c, 'm_r': m_r, 'R0_low': R0l1, 'R0_high': R0h1,
         'peak_Indh': float(np.max(sim1['Indh'])), 'peak_Indl': float(np.max(sim1['Indl']))},
    ])
    out = '../Tables/drug_v3_summary.csv'
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """
    Sweep drug multipliers over grids of m_c and m_r and plot a heatmap of
    the peak force of infection λ(t).
    """
    os.makedirs('../../Figures', exist_ok=True)

    M_c = [float(x) for x in args.m_c]
    M_r = [float(x) for x in args.m_r]
    peak_foi = np.zeros((len(M_c), len(M_r)))  # peak λ(t)

    for i, mc in enumerate(M_c):
        for j, mr in enumerate(M_r):
            _, sim = run_sim(mc, mr, days=args.days)
            lam, _, _ = foi_series(sim, mc, mr)
            peak_foi[i, j] = float(np.max(lam))

    # Single heatmap
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(
        peak_foi,
        origin='lower',
        aspect='auto',
        extent=[min(M_r), max(M_r), min(M_c), max(M_c)],
        cmap='viridis'
    )
    ax.set_title('Peak force of infection λ(t)')
    ax.set_xlabel('drug_transmission_multiplier (m_r)')
    ax.set_ylabel('drug_contact_multiplier (m_c)')
    fig.colorbar(im, ax=ax, label='peak λ')

    out = '../Figures/drug_v3_sweep_heatmap_foi.png'
    plt.savefig(out, dpi=600)
    plt.close()
    print(f"Saved: {out}")


def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI entry point.

    Subcommands:
      - run   : baseline vs params.py drug scenario
      - sweep : grid sweep over m_c and m_r with peak λ heatmap

    Examples:
      main(['run', '--days', '200'])
      main(['sweep', '--days', '200'])
    """
    parser = argparse.ArgumentParser(description='Drug exploration with Model v3')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='Run baseline vs params.py drug scenario')
    p_run.add_argument('--days', type=int, default=200, help='Simulation horizon in days')
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser('sweep', help='Sweep drug multipliers and plot peak λ heatmap')
    # Wider default grids
    p_sweep.add_argument('--m-c', nargs='+',
                         default=['0.6','0.8','1.0','1.2','1.4','1.6'],
                         help='Values for drug_contact_multiplier (treated contacts)')
    p_sweep.add_argument('--m-r', nargs='+',
                         default=['0.2','0.4','0.6','0.8','1.0'],
                         help='Values for drug_transmission_multiplier (treated per-contact transmission)')
    p_sweep.add_argument('--days', type=int, default=200, help='Simulation horizon in days')
    p_sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()