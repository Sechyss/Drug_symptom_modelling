import os, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import odeint

# allow Models import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.SEIRS_Models import SEIRS_model_v2
from Models import params as model_params

def load_base_params():
    return {
        'contact_rate': getattr(model_params, 'contact_rate', 10.0),
        'transmission_probability': getattr(model_params, 'transmission_probability', 0.025),
        'birth_rate': getattr(model_params, 'birth_rate', 0.0),
        'death_rate': getattr(model_params, 'death_rate', 0.0),
        'delta': getattr(model_params, 'delta', 1/120),
        'kappa_base': getattr(model_params, 'kappa_base', 1.0),
        'kappa_scale': getattr(model_params, 'kappa_scale', 1.0),
        'p_recover': getattr(model_params, 'p_recover', 0.5),
        'phi_recover': getattr(model_params, 'phi_recover', 1.0),
        'phi_transmission': getattr(model_params, 'phi_transmission', 1.05),
        'sigma': getattr(model_params, 'sigma', 1/5),
        'tau': getattr(model_params, 'tau', 1/3),
        'theta': getattr(model_params, 'theta', 0.3)
    }

def params_tuple(d):
    return (d['contact_rate'], d['transmission_probability'], d['birth_rate'], d['death_rate'],
            d['delta'], d['kappa_base'], d['kappa_scale'], d['p_recover'], d['phi_recover'],
            d['phi_transmission'], d['sigma'], d['tau'], d['theta'])

def run_sim(d, days=200):
    y0 = np.array([
        getattr(model_params,'S',10000), getattr(model_params,'Eh',0),
        getattr(model_params,'Indh',5), getattr(model_params,'Idh',0),
        getattr(model_params,'Rh',0), getattr(model_params,'El',0),
        getattr(model_params,'Indl',5), getattr(model_params,'Idl',0),
        getattr(model_params,'Rl',0)
    ], dtype=float); y0 = y0 / y0.sum()
    t = np.linspace(0, days, days)
    sol = odeint(SEIRS_model_v2, y0, t, args=(params_tuple(d),))
    cols = ['S','Eh','Indh','Idh','Rh','El','Indl','Idl','Rl']
    return t, {k: sol[:,i] for i,k in enumerate(cols)}

def foi_series(sim, d):
    beta_l = d['contact_rate'] * d['transmission_probability']
    beta_h = d['phi_transmission'] * beta_l
    p_rec = d['p_recover']
    lam_h = beta_h * (sim['Indh'] + p_rec * sim['Idh'])
    lam_l = beta_l * (sim['Indl'] + p_rec * sim['Idl'])
    lam = lam_h + lam_l
    return lam, lam_h, lam_l

def metric(lam, t):
    peak = lam.max()
    auc_60 = np.trapz(lam[t<=60], t[t<=60])  # early outbreak pressure
    lam0 = lam[0]
    return peak, auc_60, lam0

def elasticity(metric_fn, d, key, eps=1e-3):
    base = d.copy()
    t, sim = run_sim(base)
    lam, *_ = foi_series(sim, base)
    m0, *_ = metric_fn(lam, t)
    # central difference on log-scale (percent change)
    up = d.copy(); up[key] *= (1+eps)
    dn = d.copy(); dn[key] *= (1-eps)
    t_up, sim_up = run_sim(up); m_up = metric_fn(foi_series(sim_up, up)[0], t_up)[0]
    t_dn, sim_dn = run_sim(dn); m_dn = metric_fn(foi_series(sim_dn, dn)[0], t_dn)[0]
    dlogm = np.log(m_up) - np.log(m_dn)
    dlogp = np.log(up[key]) - np.log(dn[key])
    return dlogm/dlogp, m0

if __name__ == '__main__':
    os.makedirs('../Figures', exist_ok=True)
    d = load_base_params()
    factors = [0.8, 1.0, 1.2]
    style = {
        'contact_rate': {'ls': '--', 'alpha': 1.0, 'zorder': 3},
        'transmission_probability': {'ls': '-',  'alpha': 0.7, 'zorder': 2},
    }

    results = []
    plt.figure(figsize=(12,5))
    for i,(pname,label) in enumerate([('contact_rate','contact_rate'), ('transmission_probability','trans_prob')]):
        for f in factors:
            di = d.copy(); di[pname] = d[pname]*f
            t, sim = run_sim(di)
            lam, *_ = foi_series(sim, di)
            plt.plot(t, lam, lw=2, label=f'{label} × {f:.1f}')
            peak, auc, lam0 = metric(lam, t)
            results.append({'param': pname, 'factor': f, 'peak_lambda': peak, 'AUC_0_60': auc, 'lambda0': lam0})

    plt.xlabel('Days'); plt.ylabel('Force of infection λ(t)')
    plt.title('λ(t) under ±20% changes in contact_rate vs transmission_probability')
    plt.legend(ncol=2, fontsize=9); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('../Figures/foi_contact_vs_prob.png', dpi=600); plt.close()

    # Elasticities at baseline (peak λ)
    E_c, m0 = elasticity(metric, d, 'contact_rate')
    E_p, _  = elasticity(metric, d, 'transmission_probability')
    print(f'Baseline peak λ: {m0:.4f}')
    print(f'Elasticity wrt contact_rate: {E_c:.3f}')
    print(f'Elasticity wrt transmission_probability: {E_p:.3f}')

    # Save table
    pd.DataFrame(results).to_csv('../Tables/foi_contact_vs_prob.csv', index=False)
    print('Saved: ../Figures/foi_contact_vs_prob.png and ../Tables/foi_contact_vs_prob.csv')

    # Difference plot (should be numerically ~0)
    plt.figure(figsize=(10,4))
    lam_series = {}  # store to compare
    for pname, label in [('contact_rate','contact_rate'), ('transmission_probability','trans_prob')]:
        for f in factors:
            d2 = d.copy(); d2[pname] = d[pname] * f
            t, sim = run_sim(d2)
            lam, *_ = foi_series(sim, d2)
            lam_series[(pname, f)] = lam
            plt.plot(t, lam, lw=2, linestyle=style[pname]['ls'],
                     alpha=style[pname]['alpha'], zorder=style[pname]['zorder'],
                     label=f'{label} × {f:.1f}')
    plt.xlabel('Days'); plt.ylabel('Force of infection λ(t)')
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig('../Figures/foi_contact_vs_prob.png', dpi=600); plt.close()

    for f in factors:
        diff = lam_series[('contact_rate', f)] - lam_series[('transmission_probability', f)]
        plt.plot(t, diff, label=f'Δλ (×{f:.1f})  max={np.max(np.abs(diff)):.2e}')
    plt.axhline(0, color='k', lw=1); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig('../Figures/foi_diff_contact_minus_prob.png', dpi=600); plt.close()