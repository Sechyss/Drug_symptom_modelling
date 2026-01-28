# python
# Updated `Scripts/Simple_SEIRS_model_singlestrain.py` to match SEIRS_model_v4 (single strain)

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
from Models.SEIRS_Models import SEIRS_model_v4
from Models import params as model_params  # load shared defaults and initial conditions

np.random.seed(42)

#%% Initial Conditions (load from Models.params and normalize to proportions)
S = getattr(model_params, "S", 10000)
El = getattr(model_params, "El", 0)
Indl = getattr(model_params, "Indl", 5)
Idl = getattr(model_params, "Idl", 0)
Rl = getattr(model_params, "Rl", 0)

pop_values = np.array([S, El, Indl, Idl, Rl], dtype=float)
pop_values = pop_values / pop_values.sum()  # Normalize to proportions

#%% Parameters (match SEIRS_model_v4 signature: 12 params)
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

# pack parameters in the order SEIRS_model_v4 expects:
# (contact_rate, transmission_probability, phi_transmission,
#  drug_contact_multiplier, drug_transmission_multiplier,
#  birth_rate, death_rate, kappa_base, kappa_scale,
#  sigma, tau, theta)
parameters = (contact_rate, transmission_probability, phi_transmission,
              m_c_drug, m_r_drug,
              birth_rate, death_rate, kappa_base, kappa_scale,
              sigma, tau, theta)

# Time vector: use shared time grid from params if present, otherwise default to one year daily
t_max = getattr(model_params, "t_max", 365)
t_steps = int(getattr(model_params, "t_steps", 365))
t = np.linspace(0, t_max, t_steps)

#%% Helper to run model and compute metrics
def run_model(params_tuple, t_grid, init_values):
    sol = odeint(SEIRS_model_v4, init_values, t_grid, args=(params_tuple,))
    Sdt, Eldt, Indldt, Idldt, Rldt = sol.T
    total_inf = Indldt + Idldt
    metrics = {
        'peak_infectious': float(np.max(total_inf)),
        'time_of_peak': float(t_grid[int(np.argmax(total_inf))]),
        'final_susceptible': float(Sdt[-1]),
        'attack_rate': float(1.0 - Sdt[-1])
    }
    return sol, metrics

#%% Build scenario parameters
# Baseline (no drug): set theta=0 and drug multipliers to 1.0
baseline_params = list(parameters)
baseline_params[3] = 1.0  # m_c_drug
baseline_params[4] = 1.0  # m_r_drug
baseline_params[11] = 0.0 # theta
baseline_params = tuple(baseline_params)

# Drug scenario: use provided parameters as-is
drug_params = parameters

#%% Run both scenarios
solution_base, metrics_base = run_model(baseline_params, t, pop_values)
solution_drug, metrics_drug = run_model(drug_params, t, pop_values)

# Extract series for plotting
S_base, E_base, Ind_base, Id_base, R_base = solution_base.T
S_drug, E_drug, Ind_drug, Id_drug, R_drug = solution_drug.T
total_inf_base = Ind_base + Id_base
total_inf_drug = Ind_drug + Id_drug

#%% Plot comparison dynamics
fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111, facecolor='#f4f4f4', axisbelow=True)

ax.plot(t, E_base, 'c', lw=2, label='Exposed (No Drug)')
ax.plot(t, total_inf_base, 'r', lw=2, label='Total Infectious (No Drug)')
ax.plot(t, E_drug, 'c--', lw=2, label='Exposed (Drug)')
ax.plot(t, total_inf_drug, 'r--', lw=2, label='Total Infectious (Drug)')

ax.set_xlabel('Time (days)')
ax.set_ylabel('Proportion of Population')
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig('./Figures/singlestrain_v4_drug_comparison.png', dpi=300)

#%% Print summary metrics
print('=== Baseline (No Drug) ===')
print(f"Peak infectious proportion: {metrics_base['peak_infectious']:.6f} at day {metrics_base['time_of_peak']:.2f}")
print(f"Final susceptible proportion: {metrics_base['final_susceptible']:.6f} (attack rate {metrics_base['attack_rate']:.6f})")
print('=== Drug Scenario ===')
print(f"Peak infectious proportion: {metrics_drug['peak_infectious']:.6f} at day {metrics_drug['time_of_peak']:.2f}")
print(f"Final susceptible proportion: {metrics_drug['final_susceptible']:.6f} (attack rate {metrics_drug['attack_rate']:.6f})")
