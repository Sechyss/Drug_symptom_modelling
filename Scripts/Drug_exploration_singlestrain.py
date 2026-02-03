# python
"""
Scripts/Drug_exploration_singlestrain.py

Sweep contact rate and transmission probability per contact, simulate SEIRS,
and plot peak infected proportion and final susceptible proportion.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ---------------------------
# Model definition and utils
# ---------------------------

def params_tuple_v3(m_c_drug=1.0, m_r_drug=1.0):
        """
        Build parameter dict for simplified single-strain SEIRS model.
        This emulates model v4's drug multipliers by:
            - beta = beta0 * m_c_drug * m_r_drug (drug reduces contacts/transmission)
            - gamma = gamma0 (recovery unchanged in this simplified setup)
        """
        beta0 = 0.18        # base transmission coefficient (placeholder)
        sigma = 1/5.2       # incubation rate (E -> I)
        gamma0 = 1/10.0     # recovery rate (I -> R)
        waning = 1/(365.0)  # loss of immunity (R -> S)
        mu = 0.0            # demographic turnover
        beta = beta0 * m_c_drug * m_r_drug
        gamma = gamma0
        return dict(beta=beta, sigma=sigma, gamma=gamma, waning=waning, mu=mu, beta0=beta0, gamma0=gamma0)

def initial_conditions(N=1.0, E0=0.0, I0=1e-5, R0=0.0):
    """
    Return initial condition vector [S, E, I, R] as fractions of N.
    """
    S0 = N - E0 - I0 - R0
    return [S0, E0, I0, R0]

def SEIRS_model_v4(y, t, params):
    """
    SEIRS ODE system
    y: [S, E, I, R]
    params: dict containing beta, sigma, gamma, waning, mu
    """
    S, E, I, R = y
    beta = params['beta']
    sigma = params['sigma']
    gamma = params['gamma']
    waning = params['waning']
    mu = params['mu']

    lam = beta * I

    dSdt = -lam * S + waning * R + mu * (1.0 - S)
    dEdt = lam * S - sigma * E - mu * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I - waning * R - mu * R

    return [dSdt, dEdt, dIdt, dRdt]

# ---------------------------
# Simulation settings
# ---------------------------

tmax = 200
dt = 0.5
t = np.arange(0, tmax + dt, dt)

# template parameters (keeps sigma, gamma, waning, mu)
p_template = params_tuple_v3(m_c_drug=1.0, m_r_drug=1.0)

# initial conditions
y0 = initial_conditions(N=1.0, E0=0.0, I0=1e-5, R0=0.0)
S0 = y0[0]

# Sweep ranges: drug multipliers on contact and transmission
# Non-drug baseline occurs at (m_c_drug=1, m_r_drug=1)
m_c_vals = np.linspace(0.2, 2.0, 60)      # contact multiplier
m_r_vals = np.linspace(0.2, 2.0, 60)      # transmission multiplier

peak_I = np.zeros((len(m_r_vals), len(m_c_vals)))
epi_size = np.zeros_like(peak_I)
S_end = np.zeros_like(peak_I)  # retained for reference

# ---------------------------
# Run sweep simulations
# ---------------------------

for i_r, m_r_drug in enumerate(m_r_vals):
    for i_c, m_c_drug in enumerate(m_c_vals):
        params = p_template.copy()
        params['beta'] = params['beta0'] * m_c_drug * m_r_drug

        sol = odeint(SEIRS_model_v4, y0, t, args=(params,))
        S, E, I, R = sol.T

        peak_I[i_r, i_c] = I.max()
        S_end[i_r, i_c] = S[-1]
        epi_size[i_r, i_c] = 1.0 - S[-1]

# ---------------------------
# Plot results
# ---------------------------

plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Peak infected heatmap
im0 = axes[0].imshow(peak_I, origin='lower', aspect='auto',
                     extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
                     cmap='viridis')
axes[0].set_xlabel('Drug contact multiplier m_c')
axes[0].set_ylabel('Drug transmission multiplier m_r')
axes[0].set_title('Peak infected proportion (max I)')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label('Peak I')

# Highlight non-drug baseline (m_c=1, m_r=1)
axes[0].axvline(1.0, color='red', linestyle='--', linewidth=1.0)
axes[0].axhline(1.0, color='red', linestyle='--', linewidth=1.0)

# Epidemic size heatmap (1 - S_end)
im1 = axes[1].imshow(epi_size, origin='lower', aspect='auto',
                     extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
                     cmap='magma_r')
axes[1].set_xlabel('Drug contact multiplier m_c')
axes[1].set_ylabel('Drug transmission multiplier m_r')
axes[1].set_title('Epidemic size (1 - S(t_final))')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label('Epidemic size')

# Highlight non-drug baseline (m_c=1, m_r=1)
axes[1].axvline(1.0, color='red', linestyle='--', linewidth=1.0)
axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1.0)

plt.tight_layout()
plt.savefig('../Figures/Drug_exploration_singlestrain.png', dpi=600)
plt.show()

# ---------------------------
# Example: print a few summary values for reference
# ---------------------------
# Find global maxima locations
imax = np.unravel_index(np.argmax(peak_I), peak_I.shape)
emax = np.unravel_index(np.argmax(epi_size), epi_size.shape)

print(f"Max peak I = {peak_I[imax]:.6f} at m_c={m_c_vals[imax[1]]:.3f}, m_r={m_r_vals[imax[0]]:.3f}")
print(f"Max epidemic size = {epi_size[emax]:.6f} at m_c={m_c_vals[emax[1]]:.3f}, m_r={m_r_vals[emax[0]]:.3f}")