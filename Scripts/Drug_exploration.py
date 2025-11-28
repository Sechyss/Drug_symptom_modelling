#!/usr/bin/env python3
"""
Drug exploration with SEIRS_model_v3.

Scenario: a drug increases contact rate (behavior) but decreases per-contact transmission
for treated infectious individuals. Parameters are loaded from Models/params.py.

Usage (from repo root):
  python Scripts/Drug_exploration.py run --days 200
  python Scripts/Drug_exploration.py sweep --m-c 0.8 1.0 1.2 --m-r 0.3 0.5 0.8 --days 200

Outputs:
  - Figures/drug_v3_time_series.png
  - Figures/drug_v3_sweep_heatmap.png (for sweep)
  - Tables/drug_v3_summary.csv
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.SEIRS_Models import SEIRS_model_v3
from Models import params as P

def params_tuple_v3(m_c: float, m_r: float):
    """
    Build ordered param tuple for SEIRS_model_v3 from Models/params.py and
    supplied drug multipliers (treated contact and transmission multipliers).
    """
    return (
        P.contact_rate,                 # c_low
        getattr(P, 'transmission_probability_low', P.transmission_probability),  # r_low
        P.contact_rate_high,            # c_high (untreated high-virulence)
        P.phi_transmission,             # phi_t
        m_c,                            # drug_contact_multiplier
        m_r,                            # drug_transmission_multiplier
        P.birth_rate, P.death_rate, P.delta,
        getattr(P, 'kappa_base', 1.0), getattr(P, 'kappa_scale', 1.0),
        P.phi_recover, P.sigma, P.tau, P.theta
    )

def initial_conditions():
    y0 = np.array([P.S, P.Eh, P.Indh, P.Idh, P.Rh, P.El, P.Indl, P.Idl, P.Rl], dtype=float)
    return y0 / y0.sum()

def run_sim(m_c: float, m_r: float, days: int = 200):
    t = np.linspace(0, days, days)
    sol = odeint(SEIRS_model_v3, initial_conditions(), t, args=(params_tuple_v3(m_c, m_r),))
    cols = ['S','Eh','Indh','Idh','Rh','El','Indl','Idl','Rl']
    sim = {k: sol[:, i] for i, k in enumerate(cols)}
    return t, sim

def theta_effects():
    theta = P.theta
    kb = getattr(P, 'kappa_base', 1.0); ks = getattr(P, 'kappa_scale', 1.0)
    k_low = kb
    k_high = kb * (1.0 + ks * (P.phi_transmission - 1.0))
    cap = lambda k: min(k, 1.0 / max(theta, 1e-12))
    return cap(k_low) * theta, cap(k_high) * theta

def compute_R0_v3(m_c: float, m_r: float):
    """
    Effective R0 for each strain under split-at-onset using treated/untreated betas.
    β_low_eff  = (1-θ_l)*β_l_u + θ_l*β_l_t
    β_high_eff = (1-θ_h)*β_h_u + θ_h*β_h_t
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

def foi_series(sim, m_c: float, m_r: float):
    c_l, r_l, c_h, phi = P.contact_rate, P.transmission_probability, P.contact_rate_high, P.phi_transmission
    beta_l_u = c_l * r_l
    beta_l_t = (c_l * m_c) * (r_l * m_r)
    beta_h_u = c_h * r_l * phi
    beta_h_t = (c_h * m_c) * (r_l * m_r) * phi
    lam_l = beta_l_u * sim['Indl'] + beta_l_t * sim['Idl']
    lam_h = beta_h_u * sim['Indh'] + beta_h_t * sim['Idh']
    return lam_l + lam_h, lam_h, lam_l

def cmd_run(args):
    os.makedirs('../Figures', exist_ok=True)
    os.makedirs('../Tables', exist_ok=True)
    # Baseline (no drug effect on treated)
    t0, sim0 = run_sim(m_c=1.0, m_r=1.0, days=args.days)
    R0l0, R0h0 = compute_R0_v3(1.0, 1.0)
    # Drug scenario from params.py
    m_c = getattr(P, 'drug_contact_multiplier', 1.2)
    m_r = getattr(P, 'drug_transmission_multiplier', 0.5)
    t1, sim1 = run_sim(m_c=m_c, m_r=m_r, days=args.days)
    R0l1, R0h1 = compute_R0_v3(m_c, m_r)

    # Plot time series (high/low untreated infectious)
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    axes[0].plot(t0, sim0['Indh'], label='High, baseline'); axes[0].plot(t1, sim1['Indh'], label=f'High, drug mc={m_c}, mr={m_r}')
    axes[0].plot(t0, sim0['Indl'], '--', label='Low, baseline'); axes[0].plot(t1, sim1['Indl'], '--', label='Low, drug')
    axes[0].set_title('Untreated infectious (Ind)')
    axes[0].set_xlabel('Days'); axes[0].set_ylabel('Proportion'); axes[0].legend(); axes[0].grid(alpha=0.3)

    # FOI
    lam0, _, _ = foi_series(sim0, 1.0, 1.0)
    lam1, _, _ = foi_series(sim1, m_c, m_r)
    axes[1].plot(t0, lam0, label=f'λ baseline (R0l={R0l0:.2f}, R0h={R0h0:.2f})')
    axes[1].plot(t1, lam1, label=f'λ drug (R0l={R0l1:.2f}, R0h={R0h1:.2f})')
    axes[1].set_title('Force of infection λ(t)')
    axes[1].set_xlabel('Days'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    f = '../Figures/drug_v3_time_series.png'
    plt.savefig(f, dpi=600); plt.close()
    print(f"Saved: {f}")

    # Save summary
    df = pd.DataFrame([
        {'scenario': 'baseline', 'm_c': 1.0, 'm_r': 1.0, 'R0_low': R0l0, 'R0_high': R0h0,
         'peak_Indh': float(np.max(sim0['Indh'])), 'peak_Indl': float(np.max(sim0['Indl']))},
        {'scenario': 'drug', 'm_c': m_c, 'm_r': m_r, 'R0_low': R0l1, 'R0_high': R0h1,
         'peak_Indh': float(np.max(sim1['Indh'])), 'peak_Indl': float(np.max(sim1['Indl']))},
    ])
    out = '../Tables/drug_v3_summary.csv'
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

def cmd_sweep(args):
    os.makedirs('../Figures', exist_ok=True)
    M_c = [float(x) for x in args.m_c]
    M_r = [float(x) for x in args.m_r]
    peak = np.zeros((len(M_c), len(M_r)))
    for i, mc in enumerate(M_c):
        for j, mr in enumerate(M_r):
            _, sim = run_sim(mc, mr, days=args.days)
            peak[i, j] = float(np.max(sim['Indh'] + sim['Indl']))  # total untreated peak

    # Heatmap
    plt.figure(figsize=(8,6))
    im = plt.imshow(peak, origin='lower', aspect='auto',
                    extent=[min(M_r), max(M_r), min(M_c), max(M_c)],
                    cmap='viridis')
    plt.colorbar(im, label='Peak untreated infectious (Indh+Indl)')
    plt.xlabel('drug_transmission_multiplier (m_r)')
    plt.ylabel('drug_contact_multiplier (m_c)')
    plt.title('Model v3: peak untreated infectious vs drug multipliers')
    f = '../Figures/drug_v3_sweep_heatmap.png'
    plt.tight_layout(); plt.savefig(f, dpi=600); plt.close()
    print(f"Saved: {f}")

def main(argv=None):
    parser = argparse.ArgumentParser(description='Drug exploration with Model v3')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='Run baseline vs params.py drug scenario')
    p_run.add_argument('--days', type=int, default=200)
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser('sweep', help='Sweep drug multipliers and plot heatmap')
    p_sweep.add_argument('--m-c', nargs='+', default=['0.8','1.0','1.2'])
    p_sweep.add_argument('--m-r', nargs='+', default=['0.3','0.5','0.8'])
    p_sweep.add_argument('--days', type=int, default=200)
    p_sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == '__main__':
    main()