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
import matplotlib.ticker as mticker

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
        peak_I[i_r, i_c] = I.max() / N0            # max fraction infectious
        S_end[i_r, i_c] = S[-1] / N0
        epi_size[i_r, i_c] = 1.0 - S[-1] / N0      # cumulative infections over [0, t_max]

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

# --- Iso-R0 lines (analytic approximation at t=0) ---
R0_base = (P.contact_rate * P.transmission_probability / P.sigma) * (S0 / N0)
levels = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
m_c_min, m_c_max = m_c_vals[0], m_c_vals[-1]
m_r_min, m_r_max = m_r_vals[0], m_r_vals[-1]

def plot_iso_R0(ax, levels):
    for L in levels:
        mc_line = m_c_vals
        mr_line = L / (R0_base * mc_line)
        mask = (mr_line >= m_r_min) & (mr_line <= m_r_max)
        if not np.any(mask):
            continue

        # plot the iso-R0 line
        ln, = ax.plot(mc_line[mask], mr_line[mask],
                      color='white', lw=1.8, ls='--', zorder=4)
        ln.set_path_effects([pe.Stroke(linewidth=3.2, foreground='black'), pe.Normal()])

        # label at the middle of the visible segment, rotated along slope
        idxs = np.where(mask)[0]
        mid_idx = idxs[len(idxs) // 2]
        x_mid = mc_line[mid_idx]
        y_mid = mr_line[mid_idx]

        # slope dy/dx for mr = L / (R0_base * mc)
        dy_dx = -L / (R0_base * (x_mid**2))
        angle_deg = np.degrees(np.arctan(dy_dx))

        ax.text(
            x_mid, y_mid, f"R0={L}",
            color='white', fontsize=9,
            ha='center', va='center',
            rotation=angle_deg, rotation_mode='anchor',
            zorder=5,
            bbox=dict(facecolor='black', alpha=0.6, pad=0.8,
                      edgecolor='white', boxstyle='round,pad=0.2')
        )

# Add iso-R0 to both panels
plot_iso_R0(axes[0], levels)

# Overlay drug scenarios on peak heatmap (white with black outline)
line_d1_p, = axes[0].plot(m_c_vals, np.ones_like(m_c_vals), color='chocolate', lw=2.5, zorder=3,
                          label='Drug 1: ↑contact, m_r=1')
line_d2_p, = axes[0].plot(np.ones_like(m_r_vals), m_r_vals, color='lightblue', lw=2.5, zorder=3,
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

# Iso-R0 on second panel
plot_iso_R0(axes[1], levels)

# Overlay drug scenarios on epidemic size heatmap (same styling)
line_d1_e, = axes[1].plot(m_c_vals, np.ones_like(m_c_vals), color='chocolate', lw=2.5, zorder=3, label='Drug 1')
line_d2_e, = axes[1].plot(np.ones_like(m_r_vals), m_r_vals, color='lightblue', lw=2.5, zorder=3, label='Drug 2')
line_d3_e, = axes[1].plot(m_c_vals, drug3_path(m_c_vals, drug3_k, m_r_vals[0], m_r_vals[-1]),
                          color='white', lw=2.5, zorder=3, label='Drug 3')

for ln in (line_d1_e, line_d2_e, line_d3_e):
    ln.set_path_effects([pe.Stroke(linewidth=4, foreground='black'), pe.Normal()])

leg1 = axes[1].legend(loc='upper right', frameon=True)
leg1.get_frame().set_facecolor('#f0f0f0')
leg1.get_frame().set_alpha(0.95)

# Console info
print(f"Baseline R0 at t=0 (m_c=m_r=1): {R0_base:.3f}")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

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

# Print ranges to console
print(f"Peak I/N range: [{peak_I.min():.6e}, {peak_I.max():.6e}]")
print(f"Epidemic size range: [{epi_size.min():.6e}, {epi_size.max():.6e}]")

# Disable offset and force plain formatting
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    cb.formatter.set_powerlimits((0, 0))  # no 1e±k scaling
    cb.update_ticks()

# Or use explicit formatting:
# cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
# cb.update_ticks()

# Show absolute numbers (disable scientific offset)
for cb in (cbar0, cbar1):
    cb.formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    cb.formatter.set_powerlimits((0, 0))  # never show 1e±k header
    cb.update_ticks()

# Also plot differences vs baseline for clearer colorbars
# Baseline indices (closest to m_c=1, m_r=1)
i_c0 = (np.abs(m_c_vals - 1.0)).argmin()
i_r0 = (np.abs(m_r_vals - 1.0)).argmin()
peak_base = peak_I[i_r0, i_c0]
epi_base = epi_size[i_r0, i_c0]

# Replace heatmaps with deltas (centered at 0)
im0.set_data(peak_I - peak_base)
cbar0.set_label('Δ Peak I/N (vs baseline)')
im1.set_data(epi_size - epi_base)
cbar1.set_label('Δ Epidemic size (vs baseline)')

# Use a diverging colormap centered at 0
for im in (im0, im1):
    im.set_cmap('coolwarm')
    # auto vmin/vmax symmetric around zero
    data = im.get_array()
    vmax = np.nanmax(np.abs(data))
    im.set_clim(-vmax, vmax)

# Print actual ranges for context
print(f"Peak I/N absolute range: [{peak_I.min():.6e}, {peak_I.max():.6e}] (baseline={peak_base:.6e})")
print(f"Δ Peak I/N range: [{(peak_I-peak_base).min():.6e}, {(peak_I-peak_base).max():.6e}]")
print(f"Epidemic size absolute range: [{epi_size.min():.6e}, {epi_size.max():.6e}] (baseline={epi_base:.6e})")
print(f"Δ Epidemic size range: [{(epi_size-epi_base).min():.6e}, {(epi_size-epi_base).max():.6e}]")

# ---------------------------
# Summary
# ---------------------------
imax = np.unravel_index(np.argmax(peak_I), peak_I.shape)
emax = np.unravel_index(np.argmax(epi_size), epi_size.shape)

print(f"Max peak I/N = {peak_I[imax]:.6f} at m_c={m_c_vals[imax[1]]:.3f}, m_r={m_r_vals[imax[0]]:.3f}")
print(f"Max epidemic size = {epi_size[emax]:.6f} at m_c={m_c_vals[emax[1]]:.3f}, m_r={m_r_vals[emax[0]]:.3f}")
