# python
"""
Scripts/Drug_exploration_singlestrain.py

Sweep drug contact and transmission multipliers, simulate SEIRS_model_v4
(single-strain from Models/SEIRS_Models.py), and plot peak infected proportion
and final susceptible proportion using parameters from Models/params.py.
"""

import os
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Make project root importable when running as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Models.SEIRS_Models import SEIRS_model_v4
from Models import params as P

# ---------------------------
# Time grid from params
# ---------------------------
tmax = P.t_max
dt = P.t_steps
t = np.linspace(0.0, tmax, dt)

# ---------------------------
# Initial conditions from params (single-strain compartments for v4)
# y = [S, El, Indl, Idl, Rl]
# ---------------------------
S0 = float(P.S)
El0 = float(P.El)
Indl0 = float(P.Indl)
Idl0 = float(P.Idl)
Rl0 = float(P.Rl)
y0 = [S0, El0, Indl0, Idl0, Rl0]

# Total population for proportions (single-strain compartments only)
N0 = S0 + El0 + Indl0 + Idl0 + Rl0
if N0 <= 0:
    raise ValueError("Initial population (single-strain compartments) must be positive.")

# ---------------------------
# Sweep ranges: drug multipliers on contact and transmission
# ---------------------------
m_c_vals = np.linspace(0.2, 2.0, 60)      # drug_contact_multiplier
m_r_vals = np.linspace(0.2, 2.0, 60)      # drug_transmission_multiplier

peak_I = np.zeros((len(m_r_vals), len(m_c_vals)))
epi_size = np.zeros_like(peak_I)
S_end = np.zeros_like(peak_I)

# ---------------------------
# Run sweep simulations using SEIRS_model_v4
# Model v4 parameters (12):
# (contact_rate, transmission_probability, phi_transmission,
#  drug_contact_multiplier, drug_transmission_multiplier,
#  birth_rate, death_rate, kappa_base, kappa_scale,
#  sigma, tau, theta)
# ---------------------------
for i_r, m_r in enumerate(m_r_vals):
    for i_c, m_c in enumerate(m_c_vals):
        param_vec = (
            P.contact_rate,
            P.transmission_probability,
            P.phi_transmission,
            m_c,
            m_r,
            P.birth_rate,
            P.death_rate,
            P.kappa_base,
            P.kappa_scale,
            P.sigma,
            P.tau,
            P.theta,
        )

        sol = odeint(SEIRS_model_v4, y0, t, args=(param_vec,))
        S, El, Indl, Idl, Rl = sol.T

        I = Indl + Idl
        peak_I[i_r, i_c] = (I.max() / N0)
        S_end[i_r, i_c] = S[-1] / N0
        epi_size[i_r, i_c] = 1.0 - (S[-1] / N0)

# ---------------------------
# Plot results
# ---------------------------
plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Peak infected heatmap (proportion)
im0 = axes[0].imshow(
    peak_I,
    origin='lower',
    aspect='auto',
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap='viridis'
)
axes[0].set_xlabel('Drug contact multiplier m_c')
axes[0].set_ylabel('Drug transmission multiplier m_r')
axes[0].set_title('Peak infected proportion (max (Indl+Idl)/N)')
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label('Peak I / N')

# Highlight non-drug baseline (m_c=1, m_r=1)
axes[0].axvline(1.0, color='red', linestyle='--', linewidth=1.0)
axes[0].axhline(1.0, color='red', linestyle='--', linewidth=1.0)

# Epidemic size heatmap (1 - S_end/N0)
im1 = axes[1].imshow(
    epi_size,
    origin='lower',
    aspect='auto',
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap='magma_r'
)
axes[1].set_xlabel('Drug contact multiplier m_c')
axes[1].set_ylabel('Drug transmission multiplier m_r')
axes[1].set_title('Epidemic size (1 - S(t_final)/N)')
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label('Epidemic size')

# Baseline lines
axes[1].axvline(1.0, color='red', linestyle='--', linewidth=1.0)
axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1.0)

plt.tight_layout()

# Ensure Figures dir exists and save
fig_dir = os.path.join(ROOT_DIR, "Figures")
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, "Drug_exploration_singlestrain.png")
plt.savefig(out_path, dpi=600)
plt.show()

# ---------------------------
# Summary
# ---------------------------
imax = np.unravel_index(np.argmax(peak_I), peak_I.shape)
emax = np.unravel_index(np.argmax(epi_size), epi_size.shape)

print(f"Max peak I/N = {peak_I[imax]:.6f} at m_c={m_c_vals[imax[1]]:.3f}, m_r={m_r_vals[imax[0]]:.3f}")
print(f"Max epidemic size = {epi_size[emax]:.6f} at m_c={m_c_vals[emax[1]]:.3f}, m_r={m_r_vals[emax[0]]:.3f}")