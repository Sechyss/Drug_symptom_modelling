# python
"""
Combined runner for SEIRS v4 (single-strain) and v5 (two-strain).

Runs baseline (no drug) and drug scenarios for both models, prints metrics
(peak infectious, final susceptible, attack rate), and saves a single figure
comparing all four scenarios headlessly.
"""

#%% Imports
import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Models.SEIRS_Models import SEIRS_model_v4, SEIRS_model_v5
from Models import params as model_params

np.random.seed(42)

#%% Shared time vector
t_max = getattr(model_params, "t_max", 365)
t_steps = int(getattr(model_params, "t_steps", 365))
t = np.linspace(0, t_max, t_steps)

#%% v4 (single-strain) initial conditions (S, El, Indl, Idl, Rl) normalized
S4 = getattr(model_params, "S", 10000)
El4 = getattr(model_params, "El", 0)
Indl4 = getattr(model_params, "Indl", 5)
Idl4 = getattr(model_params, "Idl", 0)
Rl4 = getattr(model_params, "Rl", 0)
init_v4 = np.array([S4, El4, Indl4, Idl4, Rl4], dtype=float)
init_v4 = init_v4 / init_v4.sum()

#%% v5 (two-strain) initial conditions (S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl) normalized
S5  = getattr(model_params, "S", 10000)
Eh5 = getattr(model_params, "Eh", 0)
Indh5 = getattr(model_params, "Indh", 0)
Idh5  = getattr(model_params, "Idh", 0)
Rh5   = getattr(model_params, "Rh", 0)
El5   = getattr(model_params, "El", 0)
Indl5 = getattr(model_params, "Indl", 5)
Idl5  = getattr(model_params, "Idl", 0)
Rl5   = getattr(model_params, "Rl", 0)
init_v5 = np.array([S5, Eh5, Indh5, Idh5, Rh5, El5, Indl5, Idl5, Rl5], dtype=float)
init_v5 = init_v5 / init_v5.sum()

#%% v4 parameters (12):
contact_rate = getattr(model_params, "contact_rate", 10.0)
transmission_probability = getattr(model_params, "transmission_probability_low",
                                   getattr(model_params, "transmission_probability", 0.025))
phi_transmission = getattr(model_params, "phi_transmission", 1.05)
m_c_drug = getattr(model_params, "drug_contact_multiplier", 1.0)
m_r_drug = getattr(model_params, "drug_transmission_multiplier", 1.0)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
kappa_base = getattr(model_params, "kappa_base", 1.0)
kappa_scale = getattr(model_params, "kappa_scale", 1.0)
sigma = getattr(model_params, "sigma", 1/5)
tau = getattr(model_params, "tau", 1/3)
theta = getattr(model_params, "theta", 0.3)

params_v4 = [contact_rate, transmission_probability, phi_transmission,
             m_c_drug, m_r_drug,
             birth_rate, death_rate, kappa_base, kappa_scale,
             sigma, tau, theta]

# Baseline v4: no drug
params_v4_base = params_v4.copy()
params_v4_base[3] = 1.0  # m_c_drug
params_v4_base[4] = 1.0  # m_r_drug
params_v4_base[11] = 0.0 # theta
params_v4_base = tuple(params_v4_base)
params_v4_drug = tuple(params_v4)

#%% v5 parameters (14)
c_low = getattr(model_params, "contact_rate_low", getattr(model_params, "contact_rate", 10.0))
r_low = getattr(model_params, "transmission_probability_low", getattr(model_params, "transmission_probability", 0.025))
c_high = getattr(model_params, "contact_rate_high", getattr(model_params, "contact_rate", 10.0))
phi_transmission = getattr(model_params, "phi_transmission", 1.05)
m_c_drug5 = getattr(model_params, "drug_contact_multiplier", 1.0)
m_r_drug5 = getattr(model_params, "drug_transmission_multiplier", 1.0)
birth_rate5 = getattr(model_params, "birth_rate", 0.0)
death_rate5 = getattr(model_params, "death_rate", 0.0)
kappa_base5 = getattr(model_params, "kappa_base", 1.0)
kappa_scale5 = getattr(model_params, "kappa_scale", 1.0)
phi_recover5 = getattr(model_params, "phi_recover", 1.0)
sigma5 = getattr(model_params, "sigma", 1/5)
tau5 = getattr(model_params, "tau", 1/3)
theta5 = getattr(model_params, "theta", 0.3)

params_v5 = [c_low, r_low, c_high, phi_transmission,
             m_c_drug5, m_r_drug5,
             birth_rate5, death_rate5, kappa_base5, kappa_scale5,
             phi_recover5, sigma5, tau5, theta5]

# Baseline v5: no drug
params_v5_base = params_v5.copy()
params_v5_base[4] = 1.0  # m_c_drug
params_v5_base[5] = 1.0  # m_r_drug
params_v5_base[13] = 0.0 # theta
params_v5_base = tuple(params_v5_base)
params_v5_drug = tuple(params_v5)


#%% Helpers
def run_v4(params_tuple, t_grid, init_values):
    sol = odeint(SEIRS_model_v4, init_values, t_grid, args=(params_tuple,))
    S, E, Ind, Id, R = sol.T
    total_inf = Ind + Id
    metrics = {
        'peak_infectious': float(np.max(total_inf)),
        'time_of_peak': float(t_grid[int(np.argmax(total_inf))]),
        'final_susceptible': float(S[-1]),
        'attack_rate': float(1.0 - S[-1])
    }
    return sol, total_inf, metrics


def run_v5(params_tuple, t_grid, init_values):
    sol = odeint(SEIRS_model_v5, init_values, t_grid, args=(params_tuple,))
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T
    # Per-strain infectious
    inf_high = Indh + Idh
    inf_low  = Indl + Idl
    total_inf = inf_high + inf_low
    metrics = {
        'peak_infectious': float(np.max(total_inf)),
        'time_of_peak': float(t_grid[int(np.argmax(total_inf))]),
        'final_susceptible': float(S[-1]),
        'attack_rate': float(1.0 - S[-1])
    }
    return sol, total_inf, inf_high, inf_low, metrics


#%% Run all scenarios
sol_v4_base, inf_v4_base, m_v4_base = run_v4(params_v4_base, t, init_v4)
sol_v4_drug, inf_v4_drug, m_v4_drug = run_v4(params_v4_drug, t, init_v4)

sol_v5_base, inf_v5_base, inf_v5_high_base, inf_v5_low_base, m_v5_base = run_v5(params_v5_base, t, init_v5)
sol_v5_drug, inf_v5_drug, inf_v5_high_drug, inf_v5_low_drug, m_v5_drug = run_v5(params_v5_drug, t, init_v5)

S_v4_base = sol_v4_base[:, 0]
S_v4_drug = sol_v4_drug[:, 0]
S_v5_base = sol_v5_base[:, 0]
S_v5_drug = sol_v5_drug[:, 0]

#%% Print metrics
def print_metrics(title, m):
    print(title)
    print(f"  Peak infectious: {m['peak_infectious']:.6f} at day {m['time_of_peak']:.2f}")
    print(f"  Final susceptible: {m['final_susceptible']:.6f} (attack rate {m['attack_rate']:.6f})")

print("=== v4 Single-Strain (No Drug) ===")
print_metrics("", m_v4_base)
print("=== v4 Single-Strain (Drug) ===")
print_metrics("", m_v4_drug)
print("=== v5 Two-Strain (No Drug) ===")
print_metrics("", m_v5_base)
print("=== v5 Two-Strain (Drug) ===")
print_metrics("", m_v5_drug)

#%% Combined figure: top = total infectious (all), bottom = susceptible (all)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor='white', sharex=True)
ax1, ax2 = axes

# Infectious comparison: v4 single strain, v5 two strains
ax1.plot(t, inf_v4_base, 'tab:blue',  lw=2, label='No Drug (Single Strain)')
ax1.plot(t, inf_v4_drug, 'tab:blue',  lw=2, ls='--', label='Drug (Single Strain)')

ax1.plot(t, inf_v5_high_base, 'tab:red',    lw=2, label='No Drug (High Strain)')
ax1.plot(t, inf_v5_high_drug, 'tab:red',    lw=2, ls='--', label='Drug (High Strain)')
ax1.plot(t, inf_v5_low_base,  'tab:orange', lw=2, label='No Drug (Low Strain)')
ax1.plot(t, inf_v5_low_drug,  'tab:orange', lw=2, ls='--', label='Drug (Low Strain)')
ax1.set_ylabel('Proportion Infectious')
ax1.legend(ncol=2, framealpha=0.7)
ax1.grid(alpha=0.2)

# Susceptible comparison
ax2.plot(t, S_v4_base, 'tab:green', lw=2, label='No Drug - one strain (S)')
ax2.plot(t, S_v4_drug, 'tab:green', lw=2, ls='--', label='Drug - one strain (S)')
ax2.plot(t, S_v5_base, 'tab:purple', lw=2, label='No Drug - two strains (S)')
ax2.plot(t, S_v5_drug, 'tab:purple', lw=2, ls='--', label='Drug - two strains (S)')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Proportion Susceptible')
ax2.legend(ncol=2, framealpha=0.7)
ax2.grid(alpha=0.2)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '../Figures/combined_v4_v5_comparison.png')
plt.savefig(out_path, dpi=600)

print(f"Saved combined figure to {os.path.realpath(out_path)}")
