# python
"""
Combined runner for SEIRS v9 (single-strain) and v9 (two-strain).

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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from Models.SEIRS_Models import SEIRS_model_v9_singlestrain, SEIRS_model_v9
from Models import params as model_params

np.random.seed(42)

#%% Shared time vector
t_max = getattr(model_params, "t_max", 365)
t_steps = int(getattr(model_params, "t_steps", 365))
t = np.linspace(0, t_max, t_steps)

#%% v9 (single-strain) initial conditions (S, El, Indl, Idl, Rl) normalized
S_9s = getattr(model_params, "S", 10000)
E_9s = getattr(model_params, "El", 0)
Ind_9s = getattr(model_params, "Indl", 5)
Id_9s = getattr(model_params, "Idl", 0)
R_9s = getattr(model_params, "Rl", 0)
init_v9_single = np.array([S_9s, E_9s, Ind_9s, Id_9s, R_9s], dtype=float)
init_v9_single = init_v9_single / init_v9_single.sum()

#%% v9 (two-strain) initial conditions (S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl) normalized
S9  = getattr(model_params, "S", 10000)
Eh9 = getattr(model_params, "Eh", 0)
Indh9 = getattr(model_params, "Indh", 0)
Idh9  = getattr(model_params, "Idh", 0)
Rh9   = getattr(model_params, "Rh", 0)
El9   = getattr(model_params, "El", 0)
Indl9 = getattr(model_params, "Indl", 5)
Idl9  = getattr(model_params, "Idl", 0)
Rl9   = getattr(model_params, "Rl", 0)
init_v9_two = np.array([S9, Eh9, Indh9, Idh9, Rh9, El9, Indl9, Idl9, Rl9], dtype=float)
init_v9_two = init_v9_two / init_v9_two.sum()

#%% v9 single-strain parameters (9):
# (c_low, r_low, m_r_drug, birth_rate, death_rate, delta, sigma, tau, theta)
c_low = getattr(model_params, "contact_rate", 10.0)
r_low = getattr(model_params, "transmission_probability_low",
                getattr(model_params, "transmission_probability", 0.025))
m_r_drug = getattr(model_params, "drug_transmission_multiplier", 0.75)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
delta = 1/120  # immunity waning rate
sigma = getattr(model_params, "sigma", 1/5)
tau = getattr(model_params, "tau", 1/3)
theta = getattr(model_params, "theta", 0.3)

params_v9_single = [c_low, r_low, m_r_drug,
                    birth_rate, death_rate, delta,
                    sigma, tau, theta]

# Baseline v9 single: no drug
params_v9_single_base = params_v9_single.copy()
params_v9_single_base[2] = 1.0  # m_r_drug = 1.0 (no drug effect)
params_v9_single_base[8] = 0.0  # theta = 0 (no treatment)
params_v9_single_base = tuple(params_v9_single_base)
params_v9_single_drug = tuple(params_v9_single)

#%% v9 two-strain parameters (13)
# (c_low, r_low, phi_t, restoration_efficiency, m_r_drug,
#  birth_rate, death_rate, delta, kappa_base, kappa_scale,
#  sigma, tau, theta)
c_low9 = getattr(model_params, "contact_rate", 10.0)
r_low9 = getattr(model_params, "transmission_probability_low",
                 getattr(model_params, "transmission_probability", 0.025))
phi_t = getattr(model_params, "phi_transmission", 1.5)
restoration_efficiency = getattr(model_params, "drug_contact_restore", 0.5)
m_r_drug9 = getattr(model_params, "drug_transmission_multiplier", 0.75)
birth_rate9 = getattr(model_params, "birth_rate", 0.0)
death_rate9 = getattr(model_params, "death_rate", 0.0)
delta9 = 1/120  # immunity waning rate
kappa_base9 = getattr(model_params, "kappa_base", 1.0)
kappa_scale9 = getattr(model_params, "kappa_scale", 1.0)
sigma9 = getattr(model_params, "sigma", 1/5)
tau9 = getattr(model_params, "tau", 1/3)
theta9 = getattr(model_params, "theta", 0.3)

params_v9_two = [c_low9, r_low9, phi_t, restoration_efficiency, m_r_drug9,
                 birth_rate9, death_rate9, delta9,
                 kappa_base9, kappa_scale9, sigma9, tau9, theta9]

# Baseline v9 two-strain: no drug
params_v9_two_base = params_v9_two.copy()
params_v9_two_base[3] = 0.0  # restoration_efficiency = 0 (no restoration)
params_v9_two_base[4] = 1.0  # m_r_drug = 1.0 (no drug effect)
params_v9_two_base[12] = 0.0 # theta = 0 (no treatment)
params_v9_two_base = tuple(params_v9_two_base)
params_v9_two_drug = tuple(params_v9_two)


#%% Helpers
def run_v9_single(params_tuple, t_grid, init_values):
    sol = odeint(SEIRS_model_v9_singlestrain, init_values, t_grid, args=(params_tuple,))
    S, E, Ind, Id, R = sol.T
    total_inf = Ind + Id
    metrics = {
        'peak_infectious': float(np.max(total_inf)),
        'time_of_peak': float(t_grid[int(np.argmax(total_inf))]),
        'final_susceptible': float(S[-1]),
        'attack_rate': float(1.0 - S[-1])
    }
    return sol, total_inf, metrics


def run_v9_two(params_tuple, t_grid, init_values):
    sol = odeint(SEIRS_model_v9, init_values, t_grid, args=(params_tuple,))
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
sol_v9s_base, inf_v9s_base, m_v9s_base = run_v9_single(params_v9_single_base, t, init_v9_single)
sol_v9s_drug, inf_v9s_drug, m_v9s_drug = run_v9_single(params_v9_single_drug, t, init_v9_single)

sol_v9t_base, inf_v9t_base, inf_v9t_high_base, inf_v9t_low_base, m_v9t_base = run_v9_two(params_v9_two_base, t, init_v9_two)
sol_v9t_drug, inf_v9t_drug, inf_v9t_high_drug, inf_v9t_low_drug, m_v9t_drug = run_v9_two(params_v9_two_drug, t, init_v9_two)

S_v9s_base = sol_v9s_base[:, 0]
S_v9s_drug = sol_v9s_drug[:, 0]
S_v9t_base = sol_v9t_base[:, 0]
S_v9t_drug = sol_v9t_drug[:, 0]

#%% Print metrics
def print_metrics(title, m):
    print(title)
    print(f"  Peak infectious: {m['peak_infectious']:.6f} at day {m['time_of_peak']:.2f}")
    print(f"  Final susceptible: {m['final_susceptible']:.6f} (attack rate {m['attack_rate']:.6f})")

print("=== v9 Single-Strain (No Drug) ===")
print_metrics("", m_v9s_base)
print("=== v9 Single-Strain (Drug) ===")
print_metrics("", m_v9s_drug)
print("=== v9 Two-Strain (No Drug) ===")
print_metrics("", m_v9t_base)
print("=== v9 Two-Strain (Drug) ===")
print_metrics("", m_v9t_drug)

#%% Combined figure: top = total infectious (all), bottom = susceptible (all)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), facecolor='white', sharex=True)
ax1, ax2 = axes

# Infectious comparison: v9 single strain, v9 two strains
ax1.plot(t, inf_v9s_base, 'tab:blue',  lw=2, label='No Drug (Single Strain)')
ax1.plot(t, inf_v9s_drug, 'tab:blue',  lw=2, ls='--', label='Drug (Single Strain)')

ax1.plot(t, inf_v9t_high_base, 'tab:red',    lw=2, label='No Drug (High Strain)')
ax1.plot(t, inf_v9t_high_drug, 'tab:red',    lw=2, ls='--', label='Drug (High Strain)')
ax1.plot(t, inf_v9t_low_base,  'tab:orange', lw=2, label='No Drug (Low Strain)')
ax1.plot(t, inf_v9t_low_drug,  'tab:orange', lw=2, ls='--', label='Drug (Low Strain)')
ax1.set_ylabel('Proportion Infectious')
ax1.legend(ncol=2, framealpha=0.7)
ax1.grid(alpha=0.2)

# Susceptible comparison
ax2.plot(t, S_v9s_base, 'tab:green', lw=2, label='No Drug - one strain (S)')
ax2.plot(t, S_v9s_drug, 'tab:green', lw=2, ls='--', label='Drug - one strain (S)')
ax2.plot(t, S_v9t_base, 'tab:purple', lw=2, label='No Drug - two strains (S)')
ax2.plot(t, S_v9t_drug, 'tab:purple', lw=2, ls='--', label='Drug - two strains (S)')
ax2.set_xlabel('Time (days)')
ax2.set_ylabel('Proportion Susceptible')
ax2.legend(ncol=2, framealpha=0.7)
ax2.grid(alpha=0.2)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '../../Figures/Model_v9_exploration/combined_v9_single_two_comparison.png')
plt.savefig(out_path, dpi=600)

print(f"Saved combined figure to {os.path.realpath(out_path)}")
