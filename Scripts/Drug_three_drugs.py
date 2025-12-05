#!/usr/bin/env python3
"""
Three-drug scenarios with SEIRS_model_v3.

Drugs (affect treated infectious only: Idh, Idl):
  A) Increase contact rate, decrease transmission probability per contact
     -> m_c > 1, m_r < 1
  B) Increase contact rate, keep transmission probability per contact same
     -> m_c > 1, m_r = 1
  C) Keep contact rate same, reduce transmission probability per contact
     -> m_c = 1, m_r < 1

Baseline: m_c = 1, m_r = 1
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Tuple, List, Optional

# Import model v3 and params
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.SEIRS_Models import SEIRS_model_v3
from Models import params as P

COLS = ['S','Eh','Indh','Idh','Rh','El','Indl','Idl','Rl']


def params_tuple_v3(m_c: float, m_r: float) -> Tuple[float, ...]:
    return (
        P.contact_rate,                              # c_low
        getattr(P, 'transmission_probability_low',   # r_low
                P.transmission_probability),
        P.contact_rate_high,                         # c_high
        P.phi_transmission,                          # phi_t
        m_c, m_r,                                    # drug multipliers
        P.birth_rate, P.death_rate, P.delta,
        getattr(P, 'kappa_base', 1.0),
        getattr(P, 'kappa_scale', 1.0),
        P.phi_recover, P.sigma, P.tau, P.theta
    )


def initial_conditions() -> np.ndarray:
    y0 = np.array([P.S, P.Eh, P.Indh, P.Idh, P.Rh, P.El, P.Indl, P.Idl, P.Rl], dtype=float)
    return y0 / y0.sum()


def run_sim(m_c: float, m_r: float, days: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t = np.linspace(0, days, days)
    sol = odeint(SEIRS_model_v3, initial_conditions(), t, args=(params_tuple_v3(m_c, m_r),))
    return t, {k: sol[:, i] for i, k in enumerate(COLS)}


def foi_series(sim: Dict[str, np.ndarray], m_c: float, m_r: float) -> np.ndarray:
    c_l, r_l, c_h, phi = P.contact_rate, P.transmission_probability, P.contact_rate_high, P.phi_transmission
    beta_l_u = c_l * r_l
    beta_l_t = (c_l * m_c) * (r_l * m_r)
    beta_h_u = c_h * r_l * phi
    beta_h_t = (c_h * m_c) * (r_l * m_r) * phi
    lam_l = beta_l_u * sim['Indl'] + beta_l_t * sim['Idl']
    lam_h = beta_h_u * sim['Indh'] + beta_h_t * sim['Idh']
    return lam_l + lam_h


def compute_R0_v3(m_c: float, m_r: float) -> Tuple[float, float]:
    kb = getattr(P, 'kappa_base', 1.0)
    ks = getattr(P, 'kappa_scale', 1.0)
    theta = P.theta
    phi = P.phi_transmission
    # effective split-at-onset coverage caps
    k_low = min(kb, 1.0 / max(theta, 1e-12))
    k_high = min(kb * (1.0 + ks * (phi - 1.0)), 1.0 / max(theta, 1e-12))
    th_l = k_low * theta
    th_h = k_high * theta

    c_l, r_l, c_h = P.contact_rate, P.transmission_probability, P.contact_rate_high
    beta_l_u = c_l * r_l
    beta_l_t = (c_l * m_c) * (r_l * m_r)
    beta_h_u = c_h * r_l * phi
    beta_h_t = (c_h * m_c) * (r_l * m_r) * phi

    beta_l_eff = (1.0 - th_l) * beta_l_u + th_l * beta_l_t
    beta_h_eff = (1.0 - th_h) * beta_h_u + th_h * beta_h_t

    R0_l = beta_l_eff / P.sigma
    R0_h = beta_h_eff / (P.phi_recover * P.sigma)
    return R0_l, R0_h


def run_three_drugs(days: int,
                    mc_inc: float = 1.2,
                    mr_dec: float = 0.6,
                    mc_inc_same_r: float = 1.2,
                    mr_dec_only: float = 0.6) -> None:
    """
    Scenarios:
      baseline: m_c=1.0, m_r=1.0
      drug A (contact↑, transmission↓): m_c=mc_inc, m_r=mr_dec
      drug B (contact↑, transmission=): m_c=mc_inc_same_r, m_r=1.0
      drug C (contact=, transmission↓): m_c=1.0, m_r=mr_dec_only

    This version only plots and saves the force of infection λ(t).
    """
    os.makedirs('../Figures', exist_ok=True)

    scenarios = [
        ('baseline', 1.0, 1.0),
        ('drug_A_contact_up_trans_down', mc_inc, mr_dec),
        ('drug_B_contact_up_trans_same', mc_inc_same_r, 1.0),
        ('drug_C_contact_same_trans_down', 1.0, mr_dec_only),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, mc, mr in scenarios:
        t, sim = run_sim(mc, mr, days)
        lam = foi_series(sim, mc, mr)
        R0l, R0h = compute_R0_v3(mc, mr)
        ax.plot(
            t, lam,
            label=f'{name} (m_c={mc:.2f}, m_r={mr:.2f}; R0l={R0l:.2f}, R0h={R0h:.2f})'
        )

    ax.set_title('Force of infection λ(t) with contact (m_c) and transmission (m_r) multipliers')
    ax.set_xlabel('Days')
    ax.set_ylabel('λ(t)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = '../Figures/drug_v3_force_of_infection.png'
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(f"Saved: {fig_path}")


def main(argv: Optional[List[str]] = None) -> None:
    """
    Command line usage (run from repo root):

      python Scripts/Drug_three_drugs.py --days 200
      python Scripts/Drug_three_drugs.py --mc-inc 1.3 --mr-dec 0.7 --mc-inc-same-r 1.4 --mr-dec-only 0.6

    Arguments:
      --days            Simulation horizon in days (default: 200)
      --mc-inc         Contact multiplier for drugs A and B (default: 1.2)
      --mr-dec         Transmission multiplier for drug A (default: 0.8)
      --mc-inc-same-r  Contact multiplier for drug B (default: 1.2; transmission fixed at 1.0)
      --mr-dec-only    Transmission multiplier for drug C (default: 0.8; contact fixed at 1.0)
    """
    parser = argparse.ArgumentParser(description='Three drug scenarios with Model v3')
    parser.add_argument('--days', type=int, default=200, help='Simulation horizon in days')
    parser.add_argument('--mc-inc', type=float, default=1.2, help='Contact multiplier for drugs A/B')
    parser.add_argument('--mr-dec', type=float, default=0.8, help='Transmission multiplier for drug A')
    parser.add_argument('--mc-inc-same-r', type=float, default=1.2, help='Contact multiplier for drug B')
    parser.add_argument('--mr-dec-only', type=float, default=0.8, help='Transmission multiplier for drug C')
    args = parser.parse_args(argv)
    run_three_drugs(days=args.days,
                    mc_inc=args.mc_inc,
                    mr_dec=args.mr_dec,
                    mc_inc_same_r=args.mc_inc_same_r,
                    mr_dec_only=args.mr_dec_only)


if __name__ == '__main__':
    main()