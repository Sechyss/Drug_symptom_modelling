#%% Imports
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Models.SEIRS_Models import SEIRS_first_model
from Models import params as model_params  # load shared defaults and initial conditions

np.random.seed(42)

#%% Initial Conditions (load from Models.params and normalize to proportions)
# use getattr to provide sensible fallbacks if a name is missing in params.py
S = getattr(model_params, "S", 10000)
Eh = getattr(model_params, "Eh", 0)
Indh = getattr(model_params, "Indh", 5)
Idh = getattr(model_params, "Idh", 0)
Rh = getattr(model_params, "Rh", 0)
El = getattr(model_params, "El", 0)
Indl = getattr(model_params, "Indl", 5)
Idl = getattr(model_params, "Idl", 0)
Rl = getattr(model_params, "Rl", 0)

pop_values = np.array([S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl], dtype=float)
pop_values = pop_values / pop_values.sum()  # Normalize to proportions

#%% Parameters (load from Models.params)
# use getattr to preserve robustness if params.py is missing an entry
theta = getattr(model_params, "theta", 0.5)
p_recover = getattr(model_params, "p_recover", 1.5)
phi_transmission = getattr(model_params, "phi_transmission", 1.05)
phi_recover = getattr(model_params, "phi_recover", 1)
sigma = getattr(model_params, "sigma", 1/10)
delta = getattr(model_params, "delta", 1/90)
tau = getattr(model_params, "tau", 1/3)
delta_d = getattr(model_params, "delta_d", 1/3)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
beta_l = getattr(model_params, "beta_l", 0.25)

parameters = (beta_l, birth_rate, death_rate, delta, delta_d, p_recover,
              phi_recover, phi_transmission, sigma, tau, theta)

# Time vector: use shared time grid from params if present, otherwise default to one year daily
t_max = getattr(model_params, "t_max", 365)
t_steps = int(getattr(model_params, "t_steps", 365))
t = np.linspace(0, t_max, t_steps)

#%% Solve ODEs
solution = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))
columns = ['Susceptible', 'Exposed_High', 'Infected_NotDrug_High', 'Infected_Drug_High', 'Recovered_High',
           'Exposed_Low', 'Infected_NotDrug_Low', 'Infected_Drug_Low', 'Recovered_Low']
results = pd.DataFrame(solution, columns=columns)
Sdt, Ehdt, Indhdt, Idhdt, Rhdt, Eldt, Indldt, Idldt, Rldt = solution.T

#%% Plot time dynamics
fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111, facecolor='#f4f4f4', axisbelow=True)
#ax.plot(t, Sdt, 'b', lw=2, label='Susceptible')
ax.plot(t, Ehdt, 'y', lw=2, label='Exposed High')
#ax.plot(t, Indhdt, 'r', lw=2, label='Infected Not Drug High')
#ax.plot(t, Idhdt, 'm', lw=2, label='Infected Drug High')
#ax.plot(t, Rhdt, 'g', lw=2, label='Recovered High')
ax.plot(t, Eldt, 'c', lw=2, label='Exposed Low')
#ax.plot(t, Indldt, color='orange', lw=2, label='Infected Not Drug Low')
#ax.plot(t, Idldt, color='brown', lw=2, label='Infected Drug Low')
#ax.plot(t, Rldt, color='olive', lw=2, label='Recovered Low')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Proportion of Population')
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig('./Figures/first_draft_model_dynamics.png', dpi=300)
plt.show()