import os
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
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

# Initial conditions (v5): [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
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
N0 = sum(y0)
if N0 <= 0:
    raise ValueError("Initial population must be positive.")

# Normalize to proportions (to avoid explosive mass-action on counts)
y0 = [v / N0 for v in y0]
S0_prop = y0[0]
N0 = 1.0

# Sweep ranges
m_c_vals = np.linspace(0.0, 2.0, 60)
m_r_vals = np.linspace(0.0, 2.0, 60)

# Storage
peak_I = np.zeros((len(m_r_vals), len(m_c_vals)))
S_end  = np.zeros_like(peak_I)

def get_param_vec(m_c, m_r):
    """Parameter vector for v5 (two strains)."""
    return (
        float(P.contact_rate),             # contact_rate_low
        float(P.transmission_probability), # transmission_probability_low
        float(P.contact_rate_high),        # contact_rate_high
        float(P.phi_transmission),         # phi_transmission
        float(m_c),                        # drug_contact_multiplier
        float(m_r),                        # drug_transmission_multiplier
        float(P.birth_rate),
        float(P.death_rate),
        float(P.kappa_base),
        float(P.kappa_scale),
        float(P.phi_recover),
        float(P.sigma),
        float(P.tau),
        float(P.theta),
    )

def theta_fractions():
    """Effective treated fractions at onset for high vs low (v5 logic)."""
    phi_t = float(P.phi_transmission)
    kappa_base = float(P.kappa_base)
    kappa_scale = float(P.kappa_scale)
    theta = float(P.theta)

    kappa_high = kappa_base * (1.0 + kappa_scale * (phi_t - 1.0))
    kappa_low  = kappa_base
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low  = min(kappa_low,  1.0 / theta)
    return kappa_high * theta, kappa_low * theta

def R0_eff_grid_v5(MC, MR, s_over_n):
    """
    Effective R0 at DFE for two-strain v5 (max of the two strains).
    R0_s = (S0/N0) * [ (1-θ_s) * β_untreated/σ_s + θ_s * β_treated/σ_s ].
    """
    theta_high, theta_low = theta_fractions()
    c_low  = float(P.contact_rate)
    r_low  = float(P.transmission_probability)
    c_high = float(P.contact_rate_high)
    phi_t  = float(P.phi_transmission)
    sigma_h = float(P.phi_recover) * float(P.sigma)
    sigma_l = float(P.sigma)

    beta_l_u = c_low * r_low
    beta_l_t = (c_low * MC) * (r_low * MR)

    beta_h_u = c_high * r_low * phi_t
    beta_h_t = (c_high * MC) * (r_low * MR) * phi_t

    R0_l = s_over_n * (((1.0 - theta_low)  * beta_l_u + theta_low  * beta_l_t) / sigma_l)
    R0_h = s_over_n * (((1.0 - theta_high) * beta_h_u + theta_high * beta_h_t) / sigma_h)
    return np.maximum(R0_l, R0_h)

# Precompute R0 contours
MC, MR = np.meshgrid(m_c_vals, m_r_vals)
R0_grid = R0_eff_grid_v5(MC, MR, S0_prop)
R0_LEVELS = [0.75, 1.0, 1.25, 2.0]

# Simulations
for i_r, m_r in enumerate(m_r_vals):
    for i_c, m_c in enumerate(m_c_vals):
        sol = odeint(SEIRS_model_v5, y0, t, args=(get_param_vec(m_c, m_r),))
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T
        I_tot = Indh + Idh + Indl + Idl
        peak_I[i_r, i_c] = I_tot.max() / N0
        S_end[i_r, i_c]  = S[-1] / N0

# Drug path overlays (same as single-strain)
def drug3_curve(m_c_arr, k=0.6):
    mr = 1.0 - k * (m_c_arr - 1.0)
    return np.clip(mr, 0.0, 2.0)

mc_line = np.linspace(m_c_vals[0], m_c_vals[-1], 200)
mr_line = np.linspace(m_r_vals[0], m_r_vals[-1], 200)
drug1_mc, drug1_mr = mc_line, np.full_like(mc_line, 1.0)      # ↑ contact only
drug2_mc, drug2_mr = np.full_like(mr_line, 1.0), mr_line      # ↓ transmission only
drug3_mc, drug3_mr = mc_line, drug3_curve(mc_line, k=0.6)     # ↑ contact & ↓ transmission
BASELINE = (1.0, 1.0)

# -------------------- Plotting --------------------
plt.style.use("default")
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
axes[0].set_title("Peak infected proportion (two strains)")
cbar0 = fig.colorbar(im0, ax=axes[0])
fmt0 = ScalarFormatter(useOffset=False); fmt0.set_scientific(False)
cbar0.formatter = fmt0; cbar0.update_ticks()
cbar0.set_label(r"Peak $(I/N_0)$")

# R0 contours on peak plot
cs0 = axes[0].contour(MC, MR, R0_grid, levels=R0_LEVELS,
                      colors="white", linestyles="dashed", linewidths=1)
axes[0].clabel(cs0, inline=True, fontsize=8, fmt=lambda v: f"R0_eff={v:g}")

# Susceptible at end
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
cbar1.set_label(r"$S_{end}/N_0$")

# R0 contours on S_end plot
cs1 = axes[1].contour(MC, MR, R0_grid, levels=R0_LEVELS,
                      colors="white", linestyles="dashed", linewidths=1)
axes[1].clabel(cs1, inline=True, fontsize=8, fmt=lambda v: f"R0_eff={v:g}")

# Baseline + drug paths on both plots
for ax in axes:
    ax.axvline(1.0, color="gray", lw=1, ls="solid", alpha=0.6, zorder=5)
    ax.axhline(1.0, color="gray", lw=1, ls="solid", alpha=0.6, zorder=5)
    ax.plot(*BASELINE, marker="o", color="gray", ms=4, zorder=6)

    ax.plot(drug1_mc, drug1_mr, color="orange", lw=2, label="Drug 1: ↑ contact", zorder=6)
    ax.plot(drug2_mc, drug2_mr, color="deepskyblue", lw=2, label="Drug 2: ↓ transmission", zorder=6)
    ax.plot(drug3_mc, drug3_mr, color="black", lw=2, label="Drug 3: ↑ contact & ↓ transmission", zorder=6)

# Legend (smaller, bottom-left)
axes[1].legend(
    loc="lower left",
    fontsize=8,
    frameon=True,
    framealpha=0.9,
    borderaxespad=0.3,
    handlelength=1.6,
    labelspacing=0.3,
)

plt.tight_layout()

# Save
fig_dir = os.path.join(ROOT_DIR, "Figures")
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, "Explore_v5_twostrains.png")
plt.savefig(out_path, dpi=600)
print(f"Saved figure to {out_path}")