# python
"""
Two-strain SEIRS model runner using `SEIRS_model_v5`.
- Runs baseline (no drug) vs drug scenarios
- Computes peak infectious and final susceptible (attack rate)
- Saves comparison plot headlessly
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
from Models.SEIRS_Models import SEIRS_model_v5
from Models import params as model_params  # shared defaults and initial conditions

np.random.seed(42)

#%% Initial Conditions (two strains) and normalize to proportions
S  = getattr(model_params, "S", 10000)
Eh = getattr(model_params, "Eh", 0)
Indh = getattr(model_params, "Indh", 0)
Idh  = getattr(model_params, "Idh", 0)
Rh   = getattr(model_params, "Rh", 0)
El   = getattr(model_params, "El", 0)
Indl = getattr(model_params, "Indl", 5)
Idl  = getattr(model_params, "Idl", 0)
Rl   = getattr(model_params, "Rl", 0)

pop_values = np.array([S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl], dtype=float)
pop_values = pop_values / np.sum(pop_values)

#%% Parameters (match SEIRS_model_v5 signature: 14 params)
c_low = getattr(model_params, "contact_rate_low", getattr(model_params, "contact_rate", 10.0))
r_low = getattr(model_params, "transmission_probability_low", getattr(model_params, "transmission_probability", 0.025))
c_high = getattr(model_params, "contact_rate_high", getattr(model_params, "contact_rate", 10.0))
phi_transmission = getattr(model_params, "phi_transmission", 1.05)
m_c_drug = getattr(model_params, "drug_contact_multiplier", 1.0)
m_r_drug = getattr(model_params, "drug_transmission_multiplier", 1.0)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
kappa_base = getattr(model_params, "kappa_base", 1.0)
kappa_scale = getattr(model_params, "kappa_scale", 1.0)
phi_recover = getattr(model_params, "phi_recover", 1.0)
sigma = getattr(model_params, "sigma", 1/5)
tau = getattr(model_params, "tau", 1/3)
theta = getattr(model_params, "theta", 0.3)

# Pack parameters: (c_low, r_low, c_high, phi_transmission, m_c_drug, m_r_drug,
#                   birth_rate, death_rate, kappa_base, kappa_scale,
#                   phi_recover, sigma, tau, theta)
parameters = (c_low, r_low, c_high, phi_transmission, m_c_drug, m_r_drug,
			  birth_rate, death_rate, kappa_base, kappa_scale,
			  phi_recover, sigma, tau, theta)

# Time vector
t_max = getattr(model_params, "t_max", 365)
t_steps = int(getattr(model_params, "t_steps", 365))
t = np.linspace(0, t_max, t_steps)

#%% Helper to run model and compute metrics
def run_model(params_tuple, t_grid, init_values):
	sol = odeint(SEIRS_model_v5, init_values, t_grid, args=(params_tuple,))
	Sdt, Ehdt, Indhdt, Idhdt, Rhdt, Eldt, Indldt, Idldt, Rldt = sol.T
	total_exposed = Ehdt + Eldt
	total_inf = Indhdt + Idhdt + Indldt + Idldt
	metrics = {
		'peak_infectious': float(np.max(total_inf)),
		'time_of_peak': float(t_grid[int(np.argmax(total_inf))]),
		'final_susceptible': float(Sdt[-1]),
		'attack_rate': float(1.0 - Sdt[-1])
	}
	return sol, metrics, total_exposed, total_inf

#%% Build scenario parameters
# Baseline (no drug): set theta=0 and drug multipliers to 1.0
baseline_params = list(parameters)
baseline_params[4] = 1.0  # m_c_drug
baseline_params[5] = 1.0  # m_r_drug
baseline_params[13] = 0.0 # theta
baseline_params = tuple(baseline_params)

# Drug scenario: use provided parameters as-is
drug_params = parameters

#%% Run both scenarios
solution_base, metrics_base, exposed_base, total_inf_base = run_model(baseline_params, t, pop_values)
solution_drug, metrics_drug, exposed_drug, total_inf_drug = run_model(drug_params, t, pop_values)

# Extract for optional per-strain plotting
S_b, Eh_b, Indh_b, Idh_b, Rh_b, El_b, Indl_b, Idl_b, Rl_b = solution_base.T
S_d, Eh_d, Indh_d, Idh_d, Rh_d, El_d, Indl_d, Idl_d, Rl_d = solution_drug.T

#%% Plot comparison dynamics
fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111, facecolor='#f4f4f4', axisbelow=True)

ax.plot(t, exposed_base, 'c', lw=2, label='Exposed (Total, No Drug)')
ax.plot(t, total_inf_base, 'r', lw=2, label='Total Infectious (No Drug)')
ax.plot(t, exposed_drug, 'c--', lw=2, label='Exposed (Total, Drug)')
ax.plot(t, total_inf_drug, 'r--', lw=2, label='Total Infectious (Drug)')

ax.set_xlabel('Time (days)')
ax.set_ylabel('Proportion of Population')
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig('./Figures/twostrains_v5_drug_comparison.png', dpi=300)

#%% Print summary metrics
print('=== Baseline (No Drug) ===')
print(f"Peak infectious proportion: {metrics_base['peak_infectious']:.6f} at day {metrics_base['time_of_peak']:.2f}")
print(f"Final susceptible proportion: {metrics_base['final_susceptible']:.6f} (attack rate {metrics_base['attack_rate']:.6f})")
print('=== Drug Scenario ===')
print(f"Peak infectious proportion: {metrics_drug['peak_infectious']:.6f} at day {metrics_drug['time_of_peak']:.2f}")
print(f"Final susceptible proportion: {metrics_drug['final_susceptible']:.6f} (attack rate {metrics_drug['attack_rate']:.6f})")

