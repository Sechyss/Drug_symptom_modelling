# python
# Updated `Scripts/Simple_SEIRS_model_singlestrain.py` to match SEIRS_model_v4 (single strain)

#%% Imports
import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
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

#%% Solve ODEs (SEIRS_model_v4 returns 5 state variables)
solution = odeint(SEIRS_model_v4, pop_values, t, args=(parameters,))
columns = ['Susceptible', 'Exposed', 'Infected_NotDrug', 'Infected_Drug', 'Recovered']
results = pd.DataFrame(solution, columns=columns)
Sdt, Eldt, Indldt, Idldt, Rldt = solution.T

#%% Plot time dynamics
fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111, facecolor='#f4f4f4', axisbelow=True)

# plot exposed and total infectious (treated + untreated)
ax.plot(t, Eldt, 'c', lw=2, label='Exposed')
total_infectious = Indldt + Idldt
ax.plot(t, total_infectious, 'r', lw=2, label='Total Infectious (NotDrug + Drug)')

ax.set_xlabel('Time (days)')
ax.set_ylabel('Proportion of Population')
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig('./Figures/first_draft_model_dynamics_v4.png', dpi=600)
plt.show()