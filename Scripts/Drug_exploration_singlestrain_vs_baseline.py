import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import TwoSlopeNorm

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

# --- Parameters for R0 and a helper ---
c = float(P.contact_rate)
r = float(P.transmission_probability)
sigma = float(P.sigma)
theta_low = np.clip(float(P.kappa_base) * float(P.theta), 0.0, 1.0)
sfrac = float(y0[0])  # S0/N0 after normalization

def compute_R0(m_c=1.0, m_r=1.0):
    beta_u = c * r
    beta_t = (c * m_c) * (r * m_r)
    return sfrac * ((1.0 - theta_low) * beta_u + theta_low * beta_t) / sigma

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

# ---- Baseline (no drug): m_c=1, m_r=1 ----
params_baseline = get_param_vec(1.0, 1.0)
sol0 = odeint(SEIRS_model_v4, y0, t, args=(params_baseline,))
S_b, El_b, Indl_b, Idl_b, Rl_b = sol0.T
I_b = Indl_b + Idl_b
baseline_peak_I = float(np.max(I_b) / N0)
baseline_S_end  = float(S_b[-1] / N0)

# Baseline R0
R0_baseline = compute_R0(1.0, 1.0)
print(f"Baseline metrics -> Peak I/N0: {baseline_peak_I:.6f}, S_end/N0: {baseline_S_end:.6f}")
print(f"Baseline R0: {R0_baseline:.3f} {'(>1)' if R0_baseline > 1.0 else '(≤1)'}")
if R0_baseline <= 1.0:
    print("Warning: Baseline R0 ≤ 1; negative deltas are unlikely because the baseline outbreak is minimal.")

# ---- Grid simulation ----
peak_I = np.zeros((len(m_r_vals), len(m_c_vals)))
S_end_grid = np.zeros_like(peak_I)

for i_r, m_r in enumerate(m_r_vals):
    for i_c, m_c in enumerate(m_c_vals):
        params = get_param_vec(m_c, m_r)
        sol = odeint(SEIRS_model_v4, y0, t, args=(params,))
        S, El, Indl, Idl, Rl = sol.T
        I = Indl + Idl
        peak_I[i_r, i_c] = np.max(I) / N0
        S_end = S[-1] / N0
        S_end_grid[i_r, i_c] = S_end

# ---- Comparisons to baseline ----
# Differences (delta) and ratios; handle potential zero baselines
eps = 1e-12
delta_peak = peak_I - baseline_peak_I
delta_S_end = S_end_grid - baseline_S_end
ratio_peak = peak_I / max(baseline_peak_I, eps)
ratio_S_end = S_end_grid / max(baseline_S_end, eps)

# After computing delta_peak/delta_final:
min_dp = float(np.min(delta_peak))
min_dS = float(np.min(delta_S_end))
ir_min, ic_min = np.unravel_index(np.argmin(delta_peak), delta_peak.shape)
mc_min = m_c_vals[ic_min]; mr_min = m_r_vals[ir_min]
print(f"Min Δpeak={min_dp:.6e} at (mc={mc_min:.3f}, mr={mr_min:.3f})")
print(f"Min ΔS_end={min_dS:.6e}")

# ---- R0 overlay grid ----
MC, MR = np.meshgrid(m_c_vals, m_r_vals)
# reuse c, r, sigma, theta_low, sfrac defined above
beta_u = c * r
beta_t = (c * MC) * (r * MR)
R0_grid = sfrac * ((1.0 - theta_low) * beta_u + theta_low * beta_t) / sigma
levels = [0.75, 1.0, 1.25, 2.0]

# Drug path definitions
DRUG1_COLOR = "orange"       # ↑ contact only
DRUG2_COLOR = "deepskyblue"  # ↓ transmission only
DRUG3_COLOR = "black"        # ↑ contact & ↓ transmission
BASE_COLOR  = "gray"

def drug3_curve(m_c_arr, k=0.6):
    """m_r decreases linearly as m_c increases; clamp to [0, 2]."""
    mr = 1.0 - k * (m_c_arr - 1.0)
    return np.clip(mr, 0.0, 2.0)

mc_line = np.linspace(m_c_vals[0], m_c_vals[-1], 200)
mr_line = np.linspace(m_r_vals[0], m_r_vals[-1], 200)
drug1_mc, drug1_mr = mc_line, np.full_like(mc_line, 1.0)
drug2_mc, drug2_mr = np.full_like(mr_line, 1.0), mr_line
drug3_mc, drug3_mr = mc_line, drug3_curve(mc_line, k=0.6)
baseline_pt = (1.0, 1.0)
drug1_end = (2.0, 1.0)
drug2_end = (1.0, 0.0)
drug3_end = (2.0, float(drug3_curve(np.array([2.0]), k=0.6)[0]))

# -------------------- Drug path definitions (dose-response) --------------------
DRUGA_COLOR = "black"        # A: ↑ contact & ↓ transmission
DRUGB_COLOR = "orange"       # B: ↑ contact only
DRUGC_COLOR = "deepskyblue"  # C: ↓ transmission only
BASE_COLOR  = "gray"

dose = np.linspace(0.0, 1.0, 200)

def emax_curve(d, Emax, EC50):
    """Classic saturating exposure-response: effect = Emax * d/(EC50 + d)."""
    d = np.asarray(d, dtype=float)
    return Emax * (d / (EC50 + d + 1e-12))

# Tune these to your biology/PKPD assumptions
Emax_mc_A, EC50_mc_A = 1.0, 0.25   # max +100% contact for Drug A (mc up to ~2.0)
Emax_mr_A, EC50_mr_A = 0.7, 0.35   # max -70% transmission for Drug A (mr down to ~0.3)

Emax_mc_B, EC50_mc_B = 1.0, 0.20   # max +100% contact for Drug B
Emax_mr_C, EC50_mr_C = 1.0, 0.30   # max -100% transmission for Drug C (mr down to ~0.0)

# Drug A: ↑ contact AND ↓ transmission (both saturate, potentially with different EC50s)
drugA_mc = 1.0 + emax_curve(dose, Emax_mc_A, EC50_mc_A)
drugA_mr = 1.0 - emax_curve(dose, Emax_mr_A, EC50_mr_A)

# Drug B: ↑ contact, transmission fixed
drugB_mc = 1.0 + emax_curve(dose, Emax_mc_B, EC50_mc_B)
drugB_mr = np.full_like(dose, 1.0)

# Drug C: ↓ transmission, contact fixed
drugC_mc = np.full_like(dose, 1.0)
drugC_mr = 1.0 - emax_curve(dose, Emax_mr_C, EC50_mr_C)

# Clamp to plotting/sweep bounds (match your m_c_vals/m_r_vals range)
def clamp_to_bounds(mc, mr, mc_min, mc_max, mr_min, mr_max):
    return (np.clip(mc, mc_min, mc_max), np.clip(mr, mr_min, mr_max))

drugA_mc, drugA_mr = clamp_to_bounds(drugA_mc, drugA_mr, m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1])
drugB_mc, drugB_mr = clamp_to_bounds(drugB_mc, drugB_mr, m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1])
drugC_mc, drugC_mr = clamp_to_bounds(drugC_mc, drugC_mr, m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1])

baseline_pt = (1.0, 1.0)

# -------------------- Plotting --------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Set symmetric normalization around zero for both plots
max_abs_peak = float(np.max(np.abs(delta_peak)))
max_abs_S_end = float(np.max(np.abs(delta_S_end)))
peak_norm = TwoSlopeNorm(vmin=-max_abs_peak, vcenter=0.0, vmax=max_abs_peak)
S_end_norm = TwoSlopeNorm(vmin=-max_abs_S_end, vcenter=0.0, vmax=max_abs_S_end)

# Left: Δ Peak infected proportion (vs baseline)
im0 = axes[0].imshow(
    delta_peak, origin="lower", aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="coolwarm", norm=peak_norm,
)
axes[0].set_xlabel(r"Drug contact multiplier $m_c$")
axes[0].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[0].set_title("Δ Peak infected proportion (vs baseline)")
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.set_label(r"$\Delta$ Peak $(I/N_0)$")
cs0 = axes[0].contour(MC, MR, R0_grid, levels=levels,
                      colors="white", linestyles="dashed", linewidths=1)
axes[0].clabel(cs0, inline=True, fontsize=8, fmt=lambda v: f"R0={v:g}")
# Equal-to-baseline contour
axes[0].contour(MC, MR, delta_peak, levels=[0.0], colors="yellow", linewidths=1.5)

# Right: Δ Final epidemic size (vs baseline)
im1 = axes[1].imshow(
    delta_S_end, origin="lower", aspect="auto",
    extent=[m_c_vals[0], m_c_vals[-1], m_r_vals[0], m_r_vals[-1]],
    cmap="coolwarm", norm=S_end_norm,
)
axes[1].set_xlabel(r"Drug contact multiplier $m_c$")
axes[1].set_ylabel(r"Drug transmission multiplier $m_r$")
axes[1].set_title("Δ Susceptible at end (vs baseline)")
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.set_label(r"$\Delta\, S_{end}/N_0$")
cs1 = axes[1].contour(MC, MR, R0_grid, levels=levels,
                      colors="white", linestyles="dashed", linewidths=1)
axes[1].clabel(cs1, inline=True, fontsize=8, fmt=lambda v: f"R0={v:g}")
# Equal-to-baseline contour
axes[1].contour(MC, MR, delta_S_end, levels=[0.0], colors="yellow", linewidths=1.5)

# ---- Overlay drug paths AFTER axes exist ----
dose_marks = [0.0, 0.25, 0.5, 0.75, 1.0]
mark_idx = [int(p * (len(dose) - 1)) for p in dose_marks]

for ax in np.ravel(axes):
    ax.axvline(1.0, color=BASE_COLOR, lw=1, ls="solid", alpha=0.6, zorder=5)
    ax.axhline(1.0, color=BASE_COLOR, lw=1, ls="solid", alpha=0.6, zorder=5)
    ax.plot(*baseline_pt, marker="o", color=BASE_COLOR, ms=4, zorder=6)

    # Drug A: curved trajectory in (mc, mr)
    ax.plot(drugA_mc, drugA_mr, color=DRUGA_COLOR, lw=2, label="Drug A: ↑ contact & ↓ transmission", zorder=6)
    ax.plot(drugA_mc[mark_idx], drugA_mr[mark_idx], "o", color=DRUGA_COLOR, ms=3, zorder=7)

    # Drug B: saturating increase in mc (still horizontal in mr, but nonlinear in dose)
    ax.plot(drugB_mc, drugB_mr, color=DRUGB_COLOR, lw=2, label="Drug B: ↑ contact", zorder=6)
    ax.plot(drugB_mc[mark_idx], drugB_mr[mark_idx], "o", color=DRUGB_COLOR, ms=3, zorder=7)

    # Drug C: saturating decrease in mr (still vertical in mc, but nonlinear in dose)
    ax.plot(drugC_mc, drugC_mr, color=DRUGC_COLOR, lw=2, label="Drug C: ↓ transmission", zorder=6)
    ax.plot(drugC_mc[mark_idx], drugC_mr[mark_idx], "o", color=DRUGC_COLOR, ms=3, zorder=7)

axes[0].legend(
    loc="lower left",
    fontsize=6,
    frameon=True,
    framealpha=0.85,
    borderaxespad=0.3,
    handlelength=1.6,
    labelspacing=0.3,
)
plt.tight_layout()

# Save
fig_dir = os.path.join(ROOT_DIR, "Figures")
os.makedirs(fig_dir, exist_ok=True)
out_path = os.path.join(fig_dir, "Explore_v4_singlestrain_vs_baseline.png")
plt.savefig(out_path, dpi=600)
print(f"Saved figure to {out_path}")

# Optional: quick screen for near-baseline points
tol = 1e-3
mask_baseline_peak = np.abs(delta_peak) <= tol
mask_baseline_S_end = np.abs(delta_S_end) <= tol
count_both = np.count_nonzero(mask_baseline_peak & mask_baseline_S_end)
print(f"Grid points ~baseline (both metrics, tol={tol}): {count_both}")

# Test points
test_points = [(1.0, 0.5), (0.5, 1.0), (0.5, 0.5)]
for mc, mr in test_points:
    sol = odeint(SEIRS_model_v4, y0, t, args=(get_param_vec(mc, mr),))
    S, El, Indl, Idl, Rl = sol.T
    I = Indl + Idl
    peak = float(np.max(I))
    S_end = float(S[-1])
    print(f"(mc={mc}, mr={mr}) -> Δpeak={peak - baseline_peak_I:.6f}, ΔS_end={S_end - baseline_S_end:.6f}")