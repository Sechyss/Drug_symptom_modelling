"""
Sweep drug contact and transmission multipliers, simulate SEIRS_model_v5
(two strains from Models/SEIRS_Models.py), and plot three heatmaps:
- Peak infected proportion (Indh+Idh+Indl+Idl)
- Epidemic size (1 - S_end / N0)
- Effective R0 (dominant eigenvalue at DFE; max of strain R0s)

Optional overlays: R0_eff contour lines and three drug scenario paths.
"""

import os
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.ticker import ScalarFormatter

# Workspace paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Model + parameters
from Models.SEIRS_Models import SEIRS_model_v5
from Models import params as P

# Time grid
t = np.linspace(0.0, float(P.t_max), int(P.t_steps))

# Initial conditions (two-strain v5): [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
S0 = float(P.S)
Eh0 = float(P.Eh)
Indh0 = float(P.Indh)
Idh0 = float(P.Idh)
Rh0 = float(P.Rh)
El0 = float(P.El)
Indl0 = float(P.Indl)
Idl0 = float(P.Idl)
Rl0 = float(P.Rl)
y0 = [S0, Eh0, Indh0, Idh0, Rh0, El0, Indl0, Idl0, Rl0]
N0 = S0 + Eh0 + Indh0 + Idh0 + Rh0 + El0 + Indl0 + Idl0 + Rl0
if N0 <= 0:
    raise ValueError("Initial population (two-strain compartments) must be positive.")

# Sweep ranges
m_c_vals = np.linspace(0.2, 2.0, 60)
m_r_vals = np.linspace(0.2, 2.0, 60)

# Storage
peak_I = np.zeros((len(m_r_vals), len(m_c_vals)))
epi_size = np.zeros_like(peak_I)

# Drug 3 path parameter
DRUG3_K = 0.5

# ---------------------------------------------------------------------------

def drug3_path(m_c_arr, k, m_r_min, m_r_max):
    return np.clip(1.0 - k * (m_c_arr - 1.0), m_r_min, m_r_max)


def get_param_vec(m_c, m_r):
    """Parameter vector for v5 (two strains)."""
    return (
        P.contact_rate,                # contact_rate_low
        P.transmission_probability,    # transmission_probability_low
        P.contact_rate_high,           # contact_rate_high
        P.phi_transmission,            # phi_transmission
        m_c,                           # drug_contact_multiplier
        m_r,                           # drug_transmission_multiplier
        P.birth_rate,
        P.death_rate,
        P.kappa_base,
        P.kappa_scale,
        P.phi_recover,
        P.sigma,
        P.tau,
        P.theta,
    )


def theta_fractions():
    """Effective treated/detected fractions at onset for high vs low (v5 logic)."""
    phi_t = float(P.phi_transmission)
    kappa_base = float(P.kappa_base)
    kappa_scale = float(P.kappa_scale)
    theta = float(P.theta)

    kappa_high = kappa_base * (1.0 + kappa_scale * (phi_t - 1.0))
    kappa_low = kappa_base
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)
    return kappa_high * theta, kappa_low * theta


def R0_eff_grid_v5(m_c_arr, m_r_arr):
    """
    Effective R0 at DFE for two-strain v5.

    For each strain s ∈ {high, low}:
      R0_s = (S0/N0) * [ (1 - θ_s) * β_s_untreated / σ_s + θ_s * β_s_treated / σ_s ]

    Overall R0_eff = max(R0_high, R0_low) (dominant eigenvalue of diagonal NGM).
    """
    theta_high, theta_low = theta_fractions()
    c_low = float(P.contact_rate)
    r_low = float(P.transmission_probability)
    c_high = float(P.contact_rate_high)
    phi_t = float(P.phi_transmission)
    sigma_h = float(P.phi_recover) * float(P.sigma)
    sigma_l = float(P.sigma)
    s_over_n = S0 / N0

    # Betas depend on grid (drug modifies treated only)
    beta_l_u = c_low * r_low
    beta_l_t = (c_low * m_c_arr) * (r_low * m_r_arr)

    beta_h_u = c_high * r_low * phi_t
    beta_h_t = (c_high * m_c_arr) * (r_low * m_r_arr) * phi_t

    R0_l = s_over_n * (((1.0 - theta_low) * beta_l_u + theta_low * beta_l_t) / sigma_l)
    R0_h = s_over_n * (((1.0 - theta_high) * beta_h_u + theta_high * beta_h_t) / sigma_h)

    return np.maximum(R0_l, R0_h)


def add_R0_contours(ax, MC, MR, R0_grid, levels):
    cs = ax.contour(
        MC,
        MR,
        R0_grid,
        levels=levels,
        colors="white",
        linewidths=1.8,
        linestyles="--",
        zorder=4,
    )
    ax.clabel(cs, fmt=lambda v: f"R0_eff={v:g}", inline=True, fontsize=9)

# ---------------------------------------------------------------------------

# Precompute R0 across grid
MC, MR = np.meshgrid(m_c_vals, m_r_vals)
R0_grid = R0_eff_grid_v5(MC, MR)

# Simulations
for i_r, m_r in enumerate(m_r_vals):
    for i_c, m_c in enumerate(m_c_vals):
        sol = odeint(SEIRS_model_v5, y0, t, args=(get_param_vec(m_c, m_r),))
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T
        I_tot = Indh + Idh + Indl + Idl
        peak_I[i_r, i_c] = I_tot.max() / N0
        epi_size[i_r, i_c] = 1.0 - S[-1] / N0

# ---------------------------------------------------------------------------

plt.style.use("default")
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

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
axes[0].set_title("Peak infected proportion (two strains)")
cbar0 = fig.colorbar(im0, ax=axes[0])
fmt0 = ScalarFormatter(useOffset=False)
fmt0.set_scientific(False)
cbar0.formatter = fmt0
cbar0.update_ticks()
cbar0.set_label(r"Peak $(I/N_0)$")

# Epidemic size
im1 = axes[1].imshow(
    epi_size,
    origin="lower",
    aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="plasma",
)
axes[1].set_xlabel(r"Drug contact multiplier $m_c$")
axes[1].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[1].set_title(r"Epidemic size $(1 - S_{\mathrm{end}}/N_0)$")
cbar1 = fig.colorbar(im1, ax=axes[1])
fmt1 = ScalarFormatter(useOffset=False)
fmt1.set_scientific(False)
cbar1.formatter = fmt1
cbar1.update_ticks()
cbar1.set_label(r"Epidemic size")

# Effective R0
im2 = axes[2].imshow(
    R0_grid,
    origin="lower",
    aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="magma",
)
axes[2].set_xlabel(r"Drug contact multiplier $m_c$")
axes[2].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[2].set_title(r"Effective $R_0$ under drug modifiers (v5)")
cbar2 = fig.colorbar(im2, ax=axes[2])
fmt2 = ScalarFormatter(useOffset=False)
fmt2.set_scientific(False)
cbar2.formatter = fmt2
cbar2.update_ticks()
cbar2.set_label(r"$R_0^{eff}$")

# Overlays
SHOW_OVERLAYS = True
SHOW_R0_CONTOURS = True
R0_LEVELS = [0.75, 1.0, 1.25, 1.5, 2.0]

if SHOW_OVERLAYS and SHOW_R0_CONTOURS:
    for ax in axes:
        add_R0_contours(ax, MC, MR, R0_grid, R0_LEVELS)

if SHOW_OVERLAYS:
    m_r_min, m_r_max = float(m_r_vals[0]), float(m_r_vals[-1])

    for ax in axes:
        line_d1, = ax.plot(
            m_c_vals,
            np.ones_like(m_c_vals),
            color="chocolate",
            lw=2.5,
            zorder=5,
            label=r"Drug 1: $\uparrow$ contact, $m_r=1$",
        )
        line_d2, = ax.plot(
            np.ones_like(m_r_vals),
            m_r_vals,
            color="lightblue",
            lw=2.5,
            zorder=5,
            label=r"Drug 2: $m_c=1$, $\downarrow$ transmission",
        )
        line_d3, = ax.plot(
            m_c_vals,
            drug3_path(m_c_vals, DRUG3_K, m_r_min, m_r_max),
            color="white",
            lw=2.5,
            zorder=5,
            label=r"Drug 3: $\uparrow$ contact \& $\downarrow$ transmission",
        )
        for ln in (line_d1, line_d2, line_d3):
            ln.set_path_effects([pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])

    for ax in axes:
        leg = ax.legend(loc="upper right", frameon=True)
        leg.get_frame().set_facecolor("#f0f0f0")
        leg.get_frame().set_alpha(0.95)

plt.tight_layout()

fig_dir = os.path.join(ROOT_DIR, "Figures")
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, "Drug_exploration_twostrains_v5.png")
plt.savefig(out_path, dpi=600)
plt.show()
