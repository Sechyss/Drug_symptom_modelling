#!/usr/bin/env python3
"""
Three-panel force-of-infection plots for SEIRS_model_v3.

Panels:
  A) Contact↑, Transmission↓ (vary m_c list, m_r list)
  B) Contact↑, Transmission= (vary m_c list, m_r fixed = 1)
  C) Contact=, Transmission↓ (m_c fixed = 1, vary m_r list)

Saves: ../Figures/drug_v3_three_panels.png
"""

import os, sys, argparse
import numpy as np
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
    tot = max(y0.sum(), 1e-12)
    return y0 / tot


def run_sim(m_c: float, m_r: float, days: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t = np.linspace(0, days, days)
    sol = odeint(SEIRS_model_v3, initial_conditions(), t, args=(params_tuple_v3(m_c, m_r),))
    return t, {k: sol[:, i] for i, k in enumerate(COLS)}


def foi_series(sim: Dict[str, np.ndarray], m_c: float, m_r: float) -> np.ndarray:
    c_l = P.contact_rate
    r_l = getattr(P, 'transmission_probability_low', P.transmission_probability)
    c_h = P.contact_rate_high
    phi = P.phi_transmission
    beta_l_u = c_l * r_l
    beta_l_t = (c_l * m_c) * (r_l * m_r)
    beta_h_u = c_h * r_l * phi
    beta_h_t = (c_h * m_c) * (r_l * m_r) * phi
    lam_l = beta_l_u * sim['Indl'] + beta_l_t * sim['Idl']
    lam_h = beta_h_u * sim['Indh'] + beta_h_t * sim['Idh']
    return lam_l + lam_h


def plot_panel(ax, days: int, name: str, mcs: List[float], mrs: List[float], fix_mc: Optional[float]=None, fix_mr: Optional[float]=None) -> None:
    """
    Plot λ(t) curves on given axis for combinations of multipliers.
    - If fix_mc is provided, use that m_c for all, vary m_r in mrs.
    - If fix_mr is provided, use that m_r for all, vary m_c in mcs.
    - Else, iterate over paired mcs and mrs lists.
    """
    if fix_mc is not None:
        for mr in mrs:
            t, sim = run_sim(fix_mc, mr, days)
            lam = foi_series(sim, fix_mc, mr)
            ax.plot(t, lam, label=f'm_c={fix_mc:.2f}, m_r={mr:.2f}')
    elif fix_mr is not None:
        for mc in mcs:
            t, sim = run_sim(mc, fix_mr, days)
            lam = foi_series(sim, mc, fix_mr)
            ax.plot(t, lam, label=f'm_c={mc:.2f}, m_r={fix_mr:.2f}')
    else:
        # Pairwise combos
        n = max(len(mcs), len(mrs))
        for i in range(n):
            mc = mcs[i % len(mcs)]
            mr = mrs[i % len(mrs)]
            t, sim = run_sim(mc, mr, days)
            lam = foi_series(sim, mc, mr)
            ax.plot(t, lam, label=f'm_c={mc:.2f}, m_r={mr:.2f}')

    ax.set_title(name)
    ax.set_xlabel('Days')
    ax.set_ylabel('λ(t)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def run_three_panels(days: int,
                     mcs_A: List[float],
                     mrs_A: List[float],
                     mcs_B: List[float],
                     mrs_C: List[float]) -> None:
    """
    Build a 1x3 figure:
      Panel A: Contact↑ Transmission↓ (vary both lists)
      Panel B: Contact↑ Transmission= (vary m_c list, fix m_r=1)
      Panel C: Contact= Transmission↓ (fix m_c=1, vary m_r list)
    """
    os.makedirs('../Figures', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Panel A: contact up, transmission down
    plot_panel(axes[0], days, 'Drug A: contact↑, transmission↓', mcs_A, mrs_A)

    # Panel B: contact up, transmission same
    plot_panel(axes[1], days, 'Drug B: contact↑, transmission=', mcs_B, [], fix_mr=1.0)

    # Panel C: contact same, transmission down
    plot_panel(axes[2], days, 'Drug C: contact=, transmission↓', [], mrs_C, fix_mc=1.0)

    plt.tight_layout()
    out = '../Figures/drug_v3_three_panels.png'
    plt.savefig(out, dpi=600)
    plt.close()
    print(f"Saved: {out}")


def parse_float_list(s: str) -> List[float]:
    """
    Parse a comma-separated list of floats from CLI.
    Example: "1.0,1.2,1.4"
    """
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(',')]


def main(argv: Optional[List[str]] = None) -> None:
    """
    Command-line usage examples (run from repo root):

      # Default run (days=200) with preset multiplier lists
      python Scripts/Drug_three_panels.py --days 200

      # Customize Drug A: multiple contact and transmission multipliers
      python Scripts/Drug_three_panels.py --A-mc 1.0,1.2,1.4 --A-mr 0.6,0.8

      # Customize Drug B: contact multipliers, transmission fixed at 1
      python Scripts/Drug_three_panels.py --B-mc 1.0,1.3,1.6

      # Customize Drug C: transmission multipliers, contact fixed at 1
      python Scripts/Drug_three_panels.py --C-mr 0.5,0.7,0.9

      # Full custom
      python Scripts/Drug_three_panels.py --days 300 --A-mc 1.1,1.3 --A-mr 0.5,0.8 --B-mc 1.0,1.2,1.4 --C-mr 0.6,0.85
    """
    parser = argparse.ArgumentParser(description='Three-panel λ(t) plots for three drug scenarios (Model v3)')
    parser.add_argument('--days', type=int, default=200, help='Simulation horizon in days')

    # Drug A lists
    parser.add_argument('--A-mc', type=str, default='1.0,1.2,1.4', help='Comma list of m_c for Drug A (contact↑)')
    parser.add_argument('--A-mr', type=str, default='0.6,0.8,1.0', help='Comma list of m_r for Drug A (transmission↓)')

    # Drug B m_c list (m_r = 1)
    parser.add_argument('--B-mc', type=str, default='1.0,1.2,1.4', help='Comma list of m_c for Drug B (contact↑, m_r=1)')

    # Drug C m_r list (m_c = 1)
    parser.add_argument('--C-mr', type=str, default='0.6,0.8,1.0', help='Comma list of m_r for Drug C (transmission↓, m_c=1)')

    args = parser.parse_args(argv)

    mcs_A = parse_float_list(args.A_mc)
    mrs_A = parse_float_list(args.A_mr)
    mcs_B = parse_float_list(args.B_mc)
    mrs_C = parse_float_list(args.C_mr)

    run_three_panels(days=args.days,
                     mcs_A=mcs_A,
                     mrs_A=mrs_A,
                     mcs_B=mcs_B,
                     mrs_C=mrs_C)


if __name__ == '__main__':
    main()