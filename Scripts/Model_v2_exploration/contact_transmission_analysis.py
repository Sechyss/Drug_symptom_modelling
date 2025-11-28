import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import Dict, Tuple, List, Optional

"""
Unified analyses for contact rate vs transmission probability using SEIRS_model_v2.

Usage (run from the repo root or Scripts/):
- Compare strategies that reach a target R0 (low strain):
    python Scripts/contact_transmission_analysis.py compare --targets 0.8 1.0 1.5 2.0 2.5 3.0 --days 200

- Sweep φ_transmission under balanced scaling:
    python Scripts/contact_transmission_analysis.py sweep --targets 0.5 1.0 1.5 2.0 2.5 3.0 --phi-values 1.00 1.05 1.10 1.15 1.20

- Force-of-infection sensitivity (±20% changes in c vs p):
    python Scripts/contact_transmission_analysis.py foi --days 200

Outputs:
- Figures/ and Tables/ CSV files summarizing parameters, R0, peaks, and λ(t).
Notes:
- SEIRS_model_v2 assumes proportions for state variables (ICs are normalized).
- p_recover affects transmission only; treatment split occurs at symptom onset (no Ind→Id lag).
"""

# allow Models import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.SEIRS_Models import SEIRS_model_v2
from Models import params as model_params

# ---------------------------- shared helpers ----------------------------

def load_base_params() -> Dict[str, float]:
    """
    Load default model parameters from Models/params.py with sensible fallbacks.

    Returns:
        dict with keys:
        - contact_rate (c), transmission_probability (p)
        - birth_rate, death_rate, delta (waning)
        - kappa_base, kappa_scale (detection scaling)
        - p_recover (treated transmission multiplier), phi_recover (recovery modifier for high)
        - phi_transmission (φ), sigma (recovery), tau (incubation), theta (coverage)
    """
    return {
        'contact_rate': getattr(model_params, 'contact_rate', 10.0),
        'transmission_probability': getattr(model_params, 'transmission_probability', 0.025),
        'birth_rate': getattr(model_params, 'birth_rate', 0.0),
        'death_rate': getattr(model_params, 'death_rate', 0.0),
        'delta': getattr(model_params, 'delta', 1/90),
        'kappa_base': getattr(model_params, 'kappa_base', 1.0),
        'kappa_scale': getattr(model_params, 'kappa_scale', 1.0),
        'p_recover': getattr(model_params, 'p_recover', 0.5),
        'phi_recover': getattr(model_params, 'phi_recover', 1.0),
        'phi_transmission': getattr(model_params, 'phi_transmission', 1.05),
        'sigma': getattr(model_params, 'sigma', 1/10),
        'tau': getattr(model_params, 'tau', 1/3),
        'theta': getattr(model_params, 'theta', 0.3),
    }

def params_tuple(d: Dict[str, float]) -> Tuple[float, ...]:
    """
    Convert a parameter dict to the ordered tuple expected by SEIRS_model_v2.

    Order:
        (contact_rate, transmission_probability, birth_rate, death_rate,
         delta, kappa_base, kappa_scale, p_recover, phi_recover,
         phi_transmission, sigma, tau, theta)
    """
    return (d['contact_rate'], d['transmission_probability'], d['birth_rate'], d['death_rate'],
            d['delta'], d['kappa_base'], d['kappa_scale'], d['p_recover'], d['phi_recover'],
            d['phi_transmission'], d['sigma'], d['tau'], d['theta'])

def initial_conditions() -> np.ndarray:
    """
    Build initial state vector and normalize to proportions (sum=1).

    Returns:
        np.ndarray of shape (9,) for [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl].
    """
    y0 = np.array([
        getattr(model_params, 'S', 10000),
        getattr(model_params, 'Eh', 0),
        getattr(model_params, 'Indh', 5),
        getattr(model_params, 'Idh', 0),
        getattr(model_params, 'Rh', 0),
        getattr(model_params, 'El', 0),
        getattr(model_params, 'Indl', 5),
        getattr(model_params, 'Idl', 0),
        getattr(model_params, 'Rl', 0)
    ], dtype=float)
    return y0 / y0.sum()

def run_sim(d: Dict[str, float], days: int = 200, steps: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Integrate SEIRS_model_v2 over a time grid.

    Args:
        d: parameter dict matching load_base_params keys.
        days: total simulation horizon in days.
        steps: number of time points (defaults to 'days', i.e., 1 step per day).

    Returns:
        t: time vector (steps,)
        sim: dict of trajectories for compartments: S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl
    """
    steps = steps or days
    t = np.linspace(0, days, steps)
    sol = odeint(SEIRS_model_v2, initial_conditions(), t, args=(params_tuple(d),))
    cols = ['S','Eh','Indh','Idh','Rh','El','Indl','Idl','Rl']
    sim = {k: sol[:, i] for i, k in enumerate(cols)}
    return t, sim

def theta_effects(d: Dict[str, float]) -> Tuple[float, float, float, float]:
    """
    Compute effective treatment coverages θ_low and θ_high, with κ caps.

    Args:
        d: parameter dict.

    Returns:
        (theta_low_eff, theta_high_eff, kappa_low, kappa_high)
    """
    theta = d['theta']
    kb, ks = d['kappa_base'], d['kappa_scale']
    phi_t = d['phi_transmission']
    k_low = kb
    k_high = kb * (1.0 + ks * (phi_t - 1.0))
    # cap so that k*theta <= 1
    cap = lambda k: min(k, 1.0 / max(theta, 1e-12))
    th_low = cap(k_low) * theta
    th_high = cap(k_high) * theta
    return th_low, th_high, k_low, k_high

def compute_R0(d: Dict[str, float]) -> Tuple[float, float]:
    """
    Compute R0 for low and high strains under split-at-onset with treated transmission.

    Formulas:
        β_l = c·p, β_h = φ·β_l
        infectiousness factor = (1 - θ_eff) + p_recover·θ_eff
        R0_low  = β_l * inf_low  / σ
        R0_high = β_h * inf_high / (φ_recover * σ)

    Args:
        d: parameter dict.

    Returns:
        (R0_low, R0_high)
    """
    th_low, th_high, _, _ = theta_effects(d)
    inf_low = (1.0 - th_low) + d['p_recover'] * th_low
    inf_high = (1.0 - th_high) + d['p_recover'] * th_high
    beta_l = d['contact_rate'] * d['transmission_probability']
    beta_h = d['phi_transmission'] * beta_l
    R0_low = beta_l * inf_low / d['sigma']
    R0_high = beta_h * inf_high / (d['phi_recover'] * d['sigma'])
    return R0_low, R0_high

def foi_series(sim: Dict[str, np.ndarray], d: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the total and per-strain forces of infection λ(t).

    λ_h = β_h (Indh + p_recover·Idh)
    λ_l = β_l (Indl + p_recover·Idl)

    Args:
        sim: trajectories for compartments.
        d: parameter dict.

    Returns:
        (λ_total, λ_high, λ_low) each as np.ndarray over time.
    """
    beta_l = d['contact_rate'] * d['transmission_probability']
    beta_h = d['phi_transmission'] * beta_l
    p = d['p_recover']
    lam_h = beta_h * (sim['Indh'] + p * sim['Idh'])
    lam_l = beta_l * (sim['Indl'] + p * sim['Idl'])
    return lam_h + lam_l, lam_h, lam_l

# ---------------------------- strategies ----------------------------

def vary_contact_rate(base: Dict[str, float], R0_target: float) -> Dict[str, float]:
    """
    Keep transmission_probability fixed; solve for contact_rate to hit target R0_low.

    Args:
        base: base parameters.
        R0_target: desired low-strain R0.

    Returns:
        New parameter dict with updated contact_rate.
    """
    d = base.copy()
    R0_l, _ = compute_R0(d)
    if R0_l > 0:
        inf_low = R0_l * d['sigma'] / (d['contact_rate'] * d['transmission_probability'])
    else:
        th_low, _, _, _ = theta_effects(d)
        inf_low = (1.0 - th_low) + d['p_recover'] * th_low
    d['contact_rate'] = R0_target * d['sigma'] / (d['transmission_probability'] * inf_low)
    return d

def vary_trans_prob(base: Dict[str, float], R0_target: float) -> Dict[str, float]:
    """
    Keep contact_rate fixed; solve for transmission_probability to hit target R0_low.

    Args:
        base: base parameters.
        R0_target: desired low-strain R0.

    Returns:
        New parameter dict with updated transmission_probability.
    """
    d = base.copy()
    R0_l, _ = compute_R0(d)
    if R0_l > 0:
        inf_low = R0_l * d['sigma'] / (d['contact_rate'] * d['transmission_probability'])
    else:
        th_low, _, _, _ = theta_effects(d)
        inf_low = (1.0 - th_low) + d['p_recover'] * th_low
    d['transmission_probability'] = R0_target * d['sigma'] / (d['contact_rate'] * inf_low)
    return d

def vary_both_balanced(base: Dict[str, float], R0_target: float) -> Dict[str, float]:
    """
    Scale contact_rate and transmission_probability equally (geometric) to hit target R0_low.

    Args:
        base: base parameters.
        R0_target: desired low-strain R0.

    Returns:
        New parameter dict with scaled contact_rate and transmission_probability.
    """
    d = base.copy()
    c0, p0 = d['contact_rate'], d['transmission_probability']
    R0_l0, _ = compute_R0(d)
    inf_low = R0_l0 * d['sigma'] / (c0 * p0) if R0_l0 > 0 else (
        (1.0 - theta_effects(d)[0]) + d['p_recover'] * theta_effects(d)[0]
    )
    beta_target = R0_target * d['sigma'] / inf_low
    scale = np.sqrt(beta_target / (c0 * p0))
    d['contact_rate'] = c0 * scale
    d['transmission_probability'] = p0 * scale
    return d

# ---------------------------- subcommands ----------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """
    Generate time-series plots comparing strategies and save a summary CSV.

    Saves:
        - Figures/contact_vs_transmission_comparison_v2.png
        - Tables/contact_vs_transmission_comparison_v2.csv
        - Figures/parameter_strategy_summary_v2.png
    """
    os.makedirs('../Figures', exist_ok=True)
    os.makedirs('../Tables', exist_ok=True)

    base = load_base_params()
    R0_targets = [float(x) for x in args.targets]
    strategies = {
        'vary_contact': vary_contact_rate,
        'vary_trans_prob': vary_trans_prob,
        'balanced': vary_both_balanced,
    }

    results: List[Dict[str, float]] = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col_idx, (name, fn) in enumerate(strategies.items()):
        ax_h = axes[0, col_idx]; ax_l = axes[1, col_idx]
        ax_h.set_title(f"{name.replace('_',' ')}\nHigh-virulence infections")
        ax_l.set_title("Low-virulence infections")
        for R0_t in R0_targets:
            d = fn(base, R0_t)
            t, sim = run_sim(d, days=args.days)
            ax_h.plot(t, sim['Indh'], lw=2, label=f"R0={R0_t}")
            ax_l.plot(t, sim['Indl'], lw=2, label=f"R0={R0_t}")

            R0_l, R0_h = compute_R0(d)
            beta_l = d['contact_rate'] * d['transmission_probability']
            beta_h = d['phi_transmission'] * beta_l
            lam, _, _ = foi_series(sim, d)
            results.append({
                'strategy': name, 'R0_target': R0_t,
                'contact_rate': d['contact_rate'],
                'transmission_probability': d['transmission_probability'],
                'beta_l': beta_l, 'beta_h': beta_h,
                'R0_low': R0_l, 'R0_high': R0_h,
                'peak_Indh': float(np.max(sim['Indh'])),
                'peak_Indl': float(np.max(sim['Indl'])),
                'peak_lambda': float(np.max(lam)),
            })
        ax_h.legend(fontsize=8); ax_h.grid(alpha=0.3)
        ax_l.legend(fontsize=8); ax_l.grid(alpha=0.3)

    plt.tight_layout()
    f1 = '../Figures/contact_vs_transmission_comparison_v2.png'
    plt.savefig(f1, dpi=600); plt.close()
    print(f"Saved: {f1}")

    df = pd.DataFrame(results)
    c1 = '../Tables/contact_vs_transmission_comparison_v2.csv'
    df.to_csv(c1, index=False)
    print(f"Saved: {c1}")

    # summary
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for name in strategies.keys():
        sub = df[df['strategy'] == name]
        axes[0, 0].plot(sub['R0_target'], sub['contact_rate'], 'o-', label=name)
        axes[0, 1].plot(sub['R0_target'], sub['transmission_probability'], 'o-', label=name)
        axes[1, 0].plot(sub['R0_target'], sub['peak_Indh'], 'o-', label=name)
        axes[1, 1].plot(sub['R0_target'], sub['peak_Indl'], 'o-', label=name)
    axes[0,0].set_title('Contact Rate vs R0'); axes[0,1].set_title('Transmission Probability vs R0')
    axes[1,0].set_title('High-Virulence Peak vs R0'); axes[1,1].set_title('Low-Virulence Peak vs R0')
    for ax in axes.ravel(): ax.set_xlabel('Target R0'); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    f2 = '../Figures/parameter_strategy_summary_v2.png'
    plt.savefig(f2, dpi=600); plt.close()
    print(f"Saved: {f2}")

def cmd_sweep(args: argparse.Namespace) -> None:
    """
    Sweep φ_transmission values with balanced scaling to hit given R0 targets.

    Args:
        args.targets: list of R0 targets (strings convertible to float).
        args.phi_values: list of φ values (strings convertible to float).

    Saves:
        - Figures/sweep_phi_balanced.png
    """
    os.makedirs('../Figures', exist_ok=True)
    base = load_base_params()
    R0_targets = np.asarray([float(x) for x in args.targets])
    phis = [float(x) for x in args.phi_values]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_c, ax_p, ax_bl, ax_bh = axes[0,0], axes[0,1], axes[1,0], axes[1,1]
    for phi in phis:
        d = base.copy(); d['phi_transmission'] = phi
        c_vals, p_vals, b_low, b_high = [], [], [], []
        for R0_t in R0_targets:
            dd = vary_both_balanced(d, R0_t)
            c_vals.append(dd['contact_rate'])
            p_vals.append(dd['transmission_probability'])
            bl = dd['contact_rate'] * dd['transmission_probability']
            bh = dd['phi_transmission'] * bl
            b_low.append(bl); b_high.append(bh)
        lab = f'φ={phi:.2f}'
        ax_c.plot(R0_targets, c_vals, 'o-', label=lab)
        ax_p.plot(R0_targets, p_vals, 'o-', label=lab)
        ax_bl.plot(R0_targets, b_low, 'o-', label=lab)
        ax_bh.plot(R0_targets, b_high, 'o-', label=lab)

    ax_c.set_title('Contact Rate vs R0 (balanced)')
    ax_p.set_title('Transmission Probability vs R0 (balanced)')
    ax_bl.set_title('β_low vs R0 (balanced)')
    ax_bh.set_title('β_high vs R0 (balanced)')
    for ax in axes.ravel():
        ax.set_xlabel('R0 target'); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    f = '../Figures/sweep_phi_balanced.png'
    plt.savefig(f, dpi=600); plt.close()
    print(f"Saved: {f}")

def cmd_foi(args: argparse.Namespace) -> None:
    """
    Compare λ(t) after ±20% changes in contact_rate vs transmission_probability.

    Saves:
        - Figures/foi_contact_vs_prob.png
        - Figures/foi_diff_contact_minus_prob.png (numerical differences)
    """
    os.makedirs('../Figures', exist_ok=True)
    base = load_base_params()
    factors = [0.8, 1.0, 1.2]
    style = {
        'contact_rate': {'ls': '--', 'alpha': 1.0, 'z': 3},
        'transmission_probability': {'ls': '-',  'alpha': 0.8, 'z': 2},
    }
    lam_series: Dict[Tuple[str, float], np.ndarray] = {}
    t_ref: Optional[np.ndarray] = None

    plt.figure(figsize=(12,5))
    for pname,label in [('contact_rate','contact_rate'), ('transmission_probability','trans_prob')]:
        for f in factors:
            d = base.copy(); d[pname] *= f
            t, sim = run_sim(d, days=args.days)
            lam, *_ = foi_series(sim, d)
            lam_series[(pname,f)] = lam
            if t_ref is None: t_ref = t
            st = style[pname]
            plt.plot(t, lam, lw=2, ls=st['ls'], alpha=st['alpha'], zorder=st['z'], label=f'{label} × {f:.1f}')
    plt.xlabel('Days'); plt.ylabel('Force of infection λ(t)')
    plt.title('λ(t) under ±20% changes in contact_rate vs transmission_probability')
    plt.legend(ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
    f1 = '../Figures/foi_contact_vs_prob.png'
    plt.savefig(f1, dpi=600); plt.close()
    print(f"Saved: {f1}")

    # difference plot
    plt.figure(figsize=(10,4))
    for f in factors:
        diff = lam_series[('contact_rate',f)] - lam_series[('transmission_probability',f)]
        plt.plot(t_ref, diff, label=f'Δλ ×{f:.1f} max={np.max(np.abs(diff)):.2e}')
    plt.axhline(0, color='k', lw=1); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    f2 = '../Figures/foi_diff_contact_minus_prob.png'
    plt.savefig(f2, dpi=600); plt.close()
    print(f"Saved: {f2}")

# ---------------------------- main ----------------------------

def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point. Parses CLI arguments and dispatches to subcommands.

    Args:
        argv: optional list of CLI arguments (defaults to sys.argv[1:]).

    Subcommands:
        - compare: strategy comparison vs target R0 (low strain).
        - sweep: φ_transmission sweep with balanced c/p scaling.
        - foi: λ(t) sensitivity to c vs p.

    Examples:
        main(['compare', '--targets', '0.8', '1.0', '1.5', '--days', '200'])
        main(['sweep', '--targets', '0.5', '1.0', '2.0', '--phi-values', '1.00', '1.10'])
        main(['foi', '--days', '200'])
    """
    parser = argparse.ArgumentParser(
        prog='contact_transmission_analysis',
        description='Unified analyses for contact rate vs transmission probability (SEIRS_model_v2).'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_compare = sub.add_parser('compare', help='Compare parameter strategies vs target R0 (low strain).')
    p_compare.add_argument('--targets', nargs='+', default=['0.8','1.0','1.5','2.0','2.5','3.0'],
                           help='List of target R0_low values to match.')
    p_compare.add_argument('--days', type=int, default=200, help='Simulation horizon in days.')
    p_compare.set_defaults(func=cmd_compare)

    p_sweep = sub.add_parser('sweep', help='Sweep φ_transmission with balanced c/p scaling.')
    p_sweep.add_argument('--targets', nargs='+', default=['0.5','1.0','1.5','2.0','2.5','3.0'],
                         help='List of target R0_low values to match for each φ.')
    p_sweep.add_argument('--phi-values', nargs='+', dest='phi_values', default=['1.00','1.05','1.10','1.15','1.20'],
                         help='List of φ_transmission values to sweep.')
    p_sweep.set_defaults(func=cmd_sweep)

    p_foi = sub.add_parser('foi', help='Plot λ(t) for ±20%% changes in contact_rate vs transmission_probability.')
    p_foi.add_argument('--days', type=int, default=200, help='Simulation horizon in days.')
    p_foi.set_defaults(func=cmd_foi)

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == '__main__':
    main()