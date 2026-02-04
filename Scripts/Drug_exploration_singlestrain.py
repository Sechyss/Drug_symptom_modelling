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
from matplotlib import patheffects as pe

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

# Choose a trade-off slope for Drug 3 (how much transmission decreases per unit ↑contact)
drug3_k = 0.5  # m_r = 1 - drug3_k * (m_c - 1), clipped to sweep bounds

def drug3_path(m_c_arr, k, m_r_min, m_r_max):
    y = 1.0 - k * (m_c_arr - 1.0)
    return np.clip(y, m_r_min, m_r_max)

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

# Highlight non-drug baseline (m_c=1, m_r=1) with neutral gray
axes[0].axvline(1.0, color='gray', linestyle='--', linewidth=1.2, zorder=2)
axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=1.2, zorder=2)

# Overlay drug scenarios on peak heatmap (white with black outline)
line_d1_p, = axes[0].plot(m_c_vals, np.ones_like(m_c_vals), color='white', lw=2.5, zorder=3,
                          label='Drug 1: ↑contact, m_r=1')
line_d2_p, = axes[0].plot(np.ones_like(m_r_vals), m_r_vals, color='white', lw=2.5, zorder=3,
                          label='Drug 2: m_c=1, ↓transmission')
line_d3_p, = axes[0].plot(m_c_vals, drug3_path(m_c_vals, drug3_k, m_r_vals[0], m_r_vals[-1]),
                          color='white', lw=2.5, zorder=3, label='Drug 3: ↑contact & ↓transmission')

# Add black outline for contrast
for ln in (line_d1_p, line_d2_p, line_d3_p):
    ln.set_path_effects([pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])

leg0 = axes[0].legend(loc='upper right', frameon=True)
leg0.get_frame().set_facecolor('#f0f0f0')
leg0.get_frame().set_alpha(0.95)

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

# Baseline lines on second panel
axes[1].axvline(1.0, color='gray', linestyle='--', linewidth=1.2, zorder=2)
axes[1].axhline(1.0, color='gray', linestyle='--', linewidth=1.2, zorder=2)

# Overlay drug scenarios on epidemic size heatmap (same styling)
line_d1_e, = axes[1].plot(m_c_vals, np.ones_like(m_c_vals), color='white', lw=2.5, zorder=3, label='Drug 1')
line_d2_e, = axes[1].plot(np.ones_like(m_r_vals), m_r_vals, color='white', lw=2.5, zorder=3, label='Drug 2')
line_d3_e, = axes[1].plot(m_c_vals, drug3_path(m_c_vals, drug3_k, m_r_vals[0], m_r_vals[-1]),
                          color='white', lw=2.5, zorder=3, label='Drug 3')

for ln in (line_d1_e, line_d2_e, line_d3_e):
    ln.set_path_effects([pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])

leg1 = axes[1].legend(loc='upper right', frameon=True)
leg1.get_frame().set_facecolor('#f0f0f0')
leg1.get_frame().set_alpha(0.95)

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

def leading_growth_rate(m_c, m_r, eps=1e-6):
    """
    Numeric leading eigenvalue of the exposed+infectious block at the disease-free state (y0).
    >0 ⇒ early exponential growth; <0 ⇒ decay.
    """
    param_vec = (
        P.contact_rate, P.transmission_probability, P.phi_transmission,
        m_c, m_r, P.birth_rate, P.death_rate, P.kappa_base, P.kappa_scale,
        P.sigma, P.tau, P.theta
    )

    f0 = SEIRS_model_v4(y0, 0.0, param_vec)
    n = len(y0)
    J = np.zeros((n, n))
    # forward-difference Jacobian to avoid negative states
    for j in range(n):
        y_pert = np.array(y0, dtype=float)
        y_pert[j] += eps
        fj = SEIRS_model_v4(y_pert, 0.0, param_vec)
        J[:, j] = (np.array(fj) - np.array(f0)) / eps

    # Use exposed+infectious sub-block [El, Indl, Idl] = indices [1,2,3]
    Jsub = J[1:4, 1:4]
    ev = np.linalg.eigvals(Jsub)
    return float(np.max(ev.real))

# Grid of leading growth rates
lead_rate = np.zeros_like(peak_I)
for i_r, m_r in enumerate(m_r_vals):
    for i_c, m_c in enumerate(m_c_vals):
        lead_rate[i_r, i_c] = leading_growth_rate(m_c, m_r)

print(f"Leading growth rate range: min={lead_rate.min():.3e}, max={lead_rate.max():.3e}")

# Overlay Rt≈1 boundary (leading growth rate = 0) on both heatmaps
# Draw white underlay then black overlay for contrast
axes[0].contour(
    m_c_vals, m_r_vals, lead_rate, levels=[0.0],
    colors='white', linewidths=3.5, zorder=4
)
cs0 = axes[0].contour(
    m_c_vals, m_r_vals, lead_rate, levels=[0.0],
    colors='black', linewidths=2.0, zorder=5
)
axes[0].clabel(cs0, fmt={0.0: "threshold"}, inline=True, fontsize=9)

axes[1].contour(
    m_c_vals, m_r_vals, lead_rate, levels=[0.0],
    colors='white', linewidths=3.5, zorder=4
)
cs1 = axes[1].contour(
    m_c_vals, m_r_vals, lead_rate, levels=[0.0],
    colors='black', linewidths=2.0, zorder=5
)
axes[1].clabel(cs1, fmt={0.0: "threshold"}, inline=True, fontsize=9)

plt.tight_layout()