import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.ticker import ScalarFormatter

# Workspace imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from Models.SEIRS_Models import SEIRS_model_v4
from Models import params as P

# Time grid from params
t = np.linspace(0.0, float(P.t_max), int(P.t_steps))

# Initial conditions for v4 (single strain)
S0  = float(P.S)
El0 = float(P.El)
Indl0 = float(P.Indl)
Idl0  = float(P.Idl)
Rl0   = float(P.Rl)
y0 = [S0, El0, Indl0, Idl0, Rl0]
N0 = S0 + El0 + Indl0 + Idl0 + Rl0
if N0 <= 0:
    raise ValueError("Initial total population must be positive.")

# Normalize to proportions (script-only change)
y0 = [v / N0 for v in y0]
N0 = 1.0

# Sweep ranges for drug modifiers
m_c_vals = np.linspace(0.0, 2.0, 50)  # contact multiplier
m_r_vals = np.linspace(0.0, 2.0, 50)  # transmission multiplier

def get_param_vec(m_c_drug, m_r_drug):
    """Build parameter tuple for v4 using base params and current modifiers."""
    return (
        float(P.contact_rate),
        float(P.transmission_probability),
        float(P.phi_transmission),
        float(m_c_drug),
        float(m_r_drug),
        float(P.birth_rate),
        float(P.death_rate),
        float(P.kappa_base),
        float(P.kappa_scale),
        float(P.sigma),
        float(P.tau),
        float(P.theta),
    )

# Storage
peak_I = np.zeros((len(m_r_vals), len(m_c_vals)))
S_end  = np.zeros_like(peak_I)

# Sweep and simulate
for i_r, m_r in enumerate(m_r_vals):
    for i_c, m_c in enumerate(m_c_vals):
        params = get_param_vec(m_c, m_r)
        sol = odeint(SEIRS_model_v4, y0, t, args=(params,))
        S, El, Indl, Idl, Rl = sol.T
        I = Indl + Idl
        peak_I[i_r, i_c] = np.max(I) / N0
        S_end[i_r, i_c]  = S[-1] / N0

# ---- R0 overlay grid (no mesh) ----
MC, MR = np.meshgrid(m_c_vals, m_r_vals)
c = float(P.contact_rate)
r = float(P.transmission_probability)
sigma = float(P.sigma)
theta_low = np.clip(float(P.kappa_base) * float(P.theta), 0.0, 1.0)
sfrac = float(y0[0])  # S0/N0 after normalization

beta_u = c * r
beta_t = (c * MC) * (r * MR)
R0_grid = sfrac * ((1.0 - theta_low) * beta_u + theta_low * beta_t) / sigma
levels = [0.75, 1.0, 1.25, 2.0]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Peak infected proportion
im0 = axes[0].imshow(
    peak_I,
    origin="lower",
    aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="viridis",
)
axes[0].set_xlabel(r"Drug contact multiplier $m_c$")
axes[0].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[0].set_title("Peak infected proportion (max I/N0)")
cbar0 = fig.colorbar(im0, ax=axes[0])
fmt0 = ScalarFormatter(useOffset=False); fmt0.set_scientific(False)
cbar0.formatter = fmt0; cbar0.update_ticks()
cbar0.set_label("Peak (I/N0)")

# R0 contours on peak plot
cs0 = axes[0].contour(MC, MR, R0_grid, levels=levels,
                      colors="white", linestyles="dashed", linewidths=1)
axes[0].clabel(cs0, inline=True, fontsize=8, fmt=lambda v: f"R0={v:g}")

# S at end of epidemic
im1 = axes[1].imshow(
    S_end,
    origin="lower",
    aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="plasma",
)
axes[1].set_xlabel(r"Drug contact multiplier $m_c$")
axes[1].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[1].set_title("Susceptible at end (S_end/N0)")
cbar1 = fig.colorbar(im1, ax=axes[1])
fmt1 = ScalarFormatter(useOffset=False); fmt1.set_scientific(False)
cbar1.formatter = fmt1; cbar1.update_ticks()
cbar1.set_label("S_end/N0")

# R0 contours on S_end plot
cs1 = axes[1].contour(MC, MR, R0_grid, levels=levels,
                      colors="white", linestyles="dashed", linewidths=1)
axes[1].clabel(cs1, inline=True, fontsize=8, fmt=lambda v: f"R0={v:g}")

plt.tight_layout()

# Save
fig_dir = os.path.join(ROOT_DIR, "Figures")
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, "Explore_v4_singlestrain.png")
plt.savefig(out_path, dpi=600)
print(f"Saved figure to {out_path}")