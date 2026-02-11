import os
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import TwoSlopeNorm

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
S0_prop = float(y0[0])
N0 = 1.0

# Sweep ranges
m_c_vals = np.linspace(0.0, 2.0, 60)
m_r_vals = np.linspace(0.0, 2.0, 60)

# Storage (absolute, then deltas)
peak_I = np.zeros((len(m_r_vals), len(m_c_vals)))
S_end = np.zeros_like(peak_I)

def get_param_vec(m_c, m_r):
    """Parameter vector for v5 (two strains)."""
    return (
        float(P.contact_rate),              # contact_rate_low
        float(P.transmission_probability),  # transmission_probability_low
        float(P.contact_rate_high),         # contact_rate_high
        float(P.phi_transmission),          # phi_transmission
        float(m_c),                         # drug_contact_multiplier
        float(m_r),                         # drug_transmission_multiplier
        float(P.birth_rate),
        float(P.death_rate),
        float(P.kappa_base),
        float(P.kappa_scale),
        float(P.phi_recover),
        float(P.sigma),
        float(P.tau),
        float(P.theta),
    )

def theta_fractions_v5():
    """Effective treated fractions at onset for high vs low (v5 logic)."""
    phi_t = float(P.phi_transmission)
    kappa_base = float(P.kappa_base)
    kappa_scale = float(P.kappa_scale)
    theta = float(P.theta)

    kappa_high = kappa_base * (1.0 + kappa_scale * (phi_t - 1.0))
    kappa_low = kappa_base

    # Keep theta_* <= 1 by constraining kappa_* if theta>0
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)

    return kappa_high * theta, kappa_low * theta

def R0_grids_v5(MC, MR, s_over_n):
    """
    Effective R0 at DFE for each strain (low/high) under v5.

      R0_s = (S0/N0) * [ (1-θ_s) * β_u + θ_s * β_t ] / σ_s.
    """
    theta_high, theta_low = theta_fractions_v5()

    c_low = float(P.contact_rate)
    r_low = float(P.transmission_probability)
    c_high = float(P.contact_rate_high)
    phi_t = float(P.phi_transmission)

    sigma_l = float(P.sigma)
    sigma_h = float(P.phi_recover) * float(P.sigma)

    beta_l_u = c_low * r_low
    beta_l_t = (c_low * MC) * (r_low * MR)

    beta_h_u = c_high * r_low * phi_t
    beta_h_t = (c_high * MC) * (r_low * MR) * phi_t

    R0_low = s_over_n * (((1.0 - theta_low) * beta_l_u + theta_low * beta_l_t) / sigma_l)
    R0_high = s_over_n * (((1.0 - theta_high) * beta_h_u + theta_high * beta_h_t) / sigma_h)
    return R0_low, R0_high

# --- R0 precompute (independent: low + high) ---
MC, MR = np.meshgrid(m_c_vals, m_r_vals)
R0_low_grid, R0_high_grid = R0_grids_v5(MC, MR, S0_prop)
R0_LEVELS = [0.75, 1.0, 1.25, 2.0]

# Baseline R0 (mc=1, mr=1)
R0_low_baseline, R0_high_baseline = R0_grids_v5(np.array([[1.0]]), np.array([[1.0]]), S0_prop)
R0_low_baseline = float(R0_low_baseline[0, 0])
R0_high_baseline = float(R0_high_baseline[0, 0])
print(f"Baseline R0_low:  {R0_low_baseline:.3f}")
print(f"Baseline R0_high: {R0_high_baseline:.3f}")

# --- Baseline simulation (no drug): mc=1, mr=1 ---
params_baseline = get_param_vec(1.0, 1.0)
sol0 = odeint(SEIRS_model_v5, y0, t, args=(params_baseline,))
S_b, Eh_b, Indh_b, Idh_b, Rh_b, El_b, Indl_b, Idl_b, Rl_b = sol0.T
I_tot_b = Indh_b + Idh_b + Indl_b + Idl_b
baseline_peak_I = float(np.max(I_tot_b) / N0)
baseline_S_end = float(S_b[-1] / N0)
print(f"Baseline metrics -> Peak I/N0: {baseline_peak_I:.6f}, S_end/N0: {baseline_S_end:.6f}")

# --- Grid simulations ---
for i_r, m_r in enumerate(m_r_vals):
    for i_c, m_c in enumerate(m_c_vals):
        sol = odeint(SEIRS_model_v5, y0, t, args=(get_param_vec(m_c, m_r),))
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T
        I_tot = Indh + Idh + Indl + Idl
        peak_I[i_r, i_c] = float(I_tot.max() / N0)
        S_end[i_r, i_c] = float(S[-1] / N0)

# Deltas vs baseline
delta_peak = peak_I - baseline_peak_I
delta_S_end = S_end - baseline_S_end

# Drug path overlays
def drug3_curve(m_c_arr, k=0.6):
    mr = 1.0 - k * (m_c_arr - 1.0)
    return np.clip(mr, 0.0, 2.0)

mc_line = np.linspace(m_c_vals[0], m_c_vals[-1], 200)
mr_line = np.linspace(m_r_vals[0], m_r_vals[-1], 200)
drug1_mc, drug1_mr = mc_line, np.full_like(mc_line, 1.0)   # ↑ contact only
drug2_mc, drug2_mr = np.full_like(mr_line, 1.0), mr_line   # ↓ transmission only
drug3_mc, drug3_mr = mc_line, drug3_curve(mc_line, k=0.6)  # ↑ contact & ↓ transmission
BASELINE = (1.0, 1.0)

# -------------------- Plotting --------------------
plt.style.use("default")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Symmetric normalization around zero for deltas
max_abs_peak = float(np.max(np.abs(delta_peak)))
max_abs_send = float(np.max(np.abs(delta_S_end)))
peak_norm = TwoSlopeNorm(vmin=-max_abs_peak, vcenter=0.0, vmax=max_abs_peak)
send_norm = TwoSlopeNorm(vmin=-max_abs_send, vcenter=0.0, vmax=max_abs_send)

# Left: Δ peak infected
im0 = axes[0].imshow(
    delta_peak,
    origin="lower",
    aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="coolwarm",
    norm=peak_norm,
)
axes[0].set_xlabel(r"Drug contact multiplier $m_c$")
axes[0].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[0].set_title("Δ Peak infected proportion (two strains, vs baseline)")
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label(r"$\Delta$ Peak $(I/N_0)$")

# Right: Δ S_end
im1 = axes[1].imshow(
    delta_S_end,
    origin="lower",
    aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="coolwarm",
    norm=send_norm,
)
axes[1].set_xlabel(r"Drug contact multiplier $m_c$")
axes[1].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[1].set_title(r"$\Delta$ Susceptible at end (two strains, vs baseline)")
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label(r"$\Delta\, S_{end}/N_0$")

# R0 contours (independent overlays) on both plots:
# - Low: white dashed levels; baseline yellow solid
# - High: black dashed levels; baseline orange solid
for ax in axes:
    cs_low = ax.contour(
        MC, MR, R0_low_grid, levels=R0_LEVELS,
        colors="white", linestyles="dashed", linewidths=1
    )
    ax.clabel(cs_low, inline=True, fontsize=7, fmt=lambda v: f"R0_l={v:g}")
    ax.contour(
        MC, MR, R0_low_grid, levels=[R0_low_baseline],
        colors="white", linestyles="solid", linewidths=2
    )

    cs_high = ax.contour(
        MC, MR, R0_high_grid, levels=R0_LEVELS,
        colors="black", linestyles="dashed", linewidths=1
    )
    ax.clabel(cs_high, inline=True, fontsize=7, fmt=lambda v: f"R0_h={v:g}")
    ax.contour(
        MC, MR, R0_high_grid, levels=[R0_high_baseline],
        colors="white", linestyles="solid", linewidths=2
    )

# Baseline + drug paths
for ax in axes:
    ax.axvline(1.0, color="gray", lw=1, ls="solid", alpha=0.6, zorder=5)
    ax.axhline(1.0, color="gray", lw=1, ls="solid", alpha=0.6, zorder=5)
    ax.plot(*BASELINE, marker="o", color="gray", ms=4, zorder=6)

    ax.plot(drug1_mc, drug1_mr, color="orange", lw=2, label="Drug 1: ↑ contact", zorder=6)
    ax.plot(drug2_mc, drug2_mr, color="deepskyblue", lw=2, label="Drug 2: ↓ transmission", zorder=6)
    ax.plot(drug3_mc, drug3_mr, color="black", lw=2, label="Drug 3: ↑ contact & ↓ transmission", zorder=6)

axes[0].legend(
    loc="lower left",
    fontsize=6,
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
out_path = os.path.join(fig_dir, "Explore_v5_twostrains_R0split_vs_baseline.png")
plt.savefig(out_path, dpi=600)
print(f"Saved figure to {out_path}")
