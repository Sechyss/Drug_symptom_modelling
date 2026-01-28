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

def params_tuple_v3(m_c=1.0, m_r=1.0):
    """
    Build parameter dict for SEIRS_model_v4.
    m_c: multiplier on transmission/contact (not used directly in sweep)
    m_r: multiplier on recovery rate
    """
    beta0 = 0.18        # base transmission coefficient (placeholder)
    sigma = 1/5.2       # incubation rate (E -> I)
    gamma0 = 1/10.0     # recovery rate (I -> R)
    waning = 1/(365.0)  # loss of immunity (R -> S)
    mu = 0.0            # demographic turnover
    beta = beta0 * m_c
    gamma = gamma0 * m_r
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
p_template = params_tuple_v3(m_c=1.0, m_r=1.0)

# initial conditions
y0 = initial_conditions(N=1.0, E0=0.0, I0=1e-5, R0=0.0)
S0 = y0[0]

# Sweep ranges: contact rate c and transmission prob per contact p
c_vals = np.linspace(0.1, 20.0, 60)       # contacts per person per day
p_vals = np.linspace(0.001, 0.30, 60)     # transmission probability per contact

peak_I = np.zeros((len(p_vals), len(c_vals)))
S_end = np.zeros_like(peak_I)

# ---------------------------
# Run sweep simulations
# ---------------------------

for i_p, p_trans in enumerate(p_vals):
    for i_c, c in enumerate(c_vals):
        beta = c * p_trans
        params = p_template.copy()
        params['beta'] = beta

        sol = odeint(SEIRS_model_v4, y0, t, args=(params,))
        S, E, I, R = sol.T

        peak_I[i_p, i_c] = I.max()
        S_end[i_p, i_c] = S[-1]

# ---------------------------
# Plot results
# ---------------------------

plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Peak infected heatmap
im0 = axes[0].imshow(peak_I, origin='lower', aspect='auto',
                     extent=[c_vals[0], c_vals[-1], p_vals[0], p_vals[-1]],
                     cmap='viridis')
axes[0].set_xlabel('Contact rate c (contacts/day)')
axes[0].set_ylabel('Transmission probability p per contact')
axes[0].set_title('Peak infected proportion (max I)')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label('Peak I')

# Final susceptible heatmap
im1 = axes[1].imshow(S_end, origin='lower', aspect='auto',
                     extent=[c_vals[0], c_vals[-1], p_vals[0], p_vals[-1]],
                     cmap='magma_r')
axes[1].set_xlabel('Contact rate c (contacts/day)')
axes[1].set_ylabel('Transmission probability p per contact')
axes[1].set_title('Final susceptible proportion S(t_final)')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label('S_end')

plt.tight_layout()
plt.savefig('../Figures/Drug_exploration_singlestrain.png', dpi=600)
plt.show()

# ---------------------------
# Example: print a few summary values for reference
# ---------------------------
# Find global maxima/minima locations
imax = np.unravel_index(np.argmax(peak_I), peak_I.shape)
smin = np.unravel_index(np.argmin(S_end), S_end.shape)

print(f"Max peak I = {peak_I[imax]:.6f} at c={c_vals[imax[1]]:.3f}, p={p_vals[imax[0]]:.4f}")
print(f"Min final S = {S_end[smin]:.6f} at c={c_vals[smin[1]]:.3f}, p={p_vals[smin[0]]:.4f}")