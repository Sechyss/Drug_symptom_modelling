# sweep_p_recover_theta_heatmap.py
# This script performs a 2D parameter sweep of p_recover vs theta for the SEIRS model,
# collects metrics (peak and equilibrium prevalence of the high-virulence strain) and
# produces heatmaps and a CSV table of results.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from Models.SEIRS_Models import SEIRS_first_model
import pandas as pd
from Models import params as model_params  # load shared parameters and initial conditions
import os

# Make any stochastic behaviour reproducible
np.random.seed(42)

#%% --- Parameter Sweep: p_recover vs theta ---
# --- baseline parameters (loaded from Models.params, with sensible defaults) ---
# use getattr so script still runs if a name is missing in model_params
beta_l = getattr(model_params, "beta_l", 0.25)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
delta = getattr(model_params, "delta", 1/90)
delta_d = getattr(model_params, "delta_d", 1/3)
p_recover = getattr(model_params, "p_recover", 1.5)
phi_recover = getattr(model_params, "phi_recover", 1.0)
sigma = getattr(model_params, "sigma", 1/10)
tau = getattr(model_params, "tau", 1/3)
phi_transmission = getattr(model_params, "phi_transmission", 1.05)
theta = getattr(model_params, "theta", 0.3)
t_max = getattr(model_params, "t_max", 365)
t_steps = getattr(model_params, "t_steps", 365)

# --- initial conditions (loaded from Models.params) ---
# load initial compartment counts from params.py (fall back to previous defaults if missing)
S0 = getattr(model_params, "S", 10000)
Eh0 = getattr(model_params, "Eh", 0)
Indh0 = getattr(model_params, "Indh", 5)
Idh0 = getattr(model_params, "Idh", 0)
Rh0 = getattr(model_params, "Rh", 0)
El0 = getattr(model_params, "El", 0)
Indl0 = getattr(model_params, "Indl", 5)
Idl0 = getattr(model_params, "Idl", 0)
Rl0 = getattr(model_params, "Rl", 0)
y0 = np.array([S0, Eh0, Indh0, Idh0, Rh0, El0, Indl0, Idl0, Rl0])
# Normalize to proportions (model expects fractions/proportions rather than absolute counts)
y0 = y0 / y0.sum()

# --- time vector ---
# Use t_max / t_steps from params.py when available, otherwise daily for one year
t = np.linspace(0, t_max, int(t_steps))

# --- Define parameter ranges for sweep ---
theta_vals = np.linspace(0.0, 1.0, 20)  # fraction using symptom-blocking drug
p_recover_vals = np.linspace(1.0, 2.0, 20)  # recovery rate multiplier (wider range)

# storage arrays: shape (len(theta_vals), len(p_recover_vals))
# peak_high: maximum prevalence of high-virulence infection over the simulation (peak)
peak_high = np.zeros((len(theta_vals), len(p_recover_vals)))
# eq_high_abs: mean absolute prevalence of high-virulence infection near the end of the simulation (approx equilibrium)
eq_high_abs = np.zeros_like(peak_high)
# eq_high_frac: fraction of total infections that are high-virulence at equilibrium
eq_high_frac = np.zeros_like(peak_high)

# --- main parameter sweep ---
# Loop over theta (rows) and p_recover (columns). For every pair, run the ODE model and compute metrics.
for i, theta_val in enumerate(theta_vals):
    for j, p_recover_val in enumerate(p_recover_vals):
        # Pack parameters to match SEIRS_first_model signature:
        # (beta_l, birth_rate, death_rate, delta, delta_d, p_recover, phi_recover, phi_transmission, sigma, tau, theta)
        params = (beta_l, birth_rate, death_rate, delta, delta_d, p_recover_val,
                  phi_recover, phi_transmission, sigma, tau, theta_val)

        # Integrate the ODE system for the given parameter set
        sol = odeint(SEIRS_first_model, y0, t, args=(params,))

        # Unpack solution columns into named variables for clarity
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

        # total_high: combined prevalence of high-virulence infecteds (both drug and not-on-drug compartments)
        total_high = Indh + Idh
        # total_low: combined prevalence of low-virulence infecteds
        total_low = Indl + Idl
        # total_inf: total infected prevalence (high + low)
        total_inf = total_high + total_low

        # Compute metrics:
        # 1) peak_high: maximum value of total_high during the simulation
        peak_high[i, j] = total_high.max()

        # 2) eq_high_abs: approximate equilibrium absolute prevalence of high strain
        #    Use last 30 timepoints (last ~30 days) mean as a crude equilibrium estimate
        eq_high_abs[i, j] = total_high[-30:].mean()

        # 3) eq_high_frac: fraction of infections due to high strain at equilibrium
        #    add small epsilon in denominator to avoid division by zero
        eq_high_frac[i, j] = eq_high_abs[i, j] / (total_inf[-30:].mean() + 1e-12)

# --- plotting ---
# Create 3 side-by-side heatmaps: peak, equilibrium absolute, equilibrium fraction
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def plot_heatmap(ax, data, title, cmap='viridis'):
    """
    Helper to draw a heatmap with proper axis labels and colorbar.

    - ax: matplotlib Axes
    - data: 2D numpy array with shape (len(theta_vals), len(p_recover_vals))
    - title: plot title
    - cmap: colormap
    """
    # imshow with origin='lower' so that smaller theta is at the bottom (natural orientation)
    im = ax.imshow(data, aspect='auto', origin='lower', cmap=cmap,
                   extent=[p_recover_vals[0], p_recover_vals[-1], theta_vals[0], theta_vals[-1]])
    # Axis labels describe what each axis means
    ax.set_xlabel('p_recover (recovery rate multiplier)')
    ax.set_ylabel('theta (fraction using symptom-blocking drug)')
    ax.set_title(title)
    # Add a small colorbar to the axis
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Plot each metric with a visually distinct colormap
plot_heatmap(axes[0], peak_high, 'Peak high-virulence infection\n(absolute prevalence)', cmap='plasma')
plot_heatmap(axes[1], eq_high_abs, 'Equilibrium high-virulence infection\n(absolute prevalence)', cmap='magma')
plot_heatmap(axes[2], eq_high_frac, 'High strain fraction at equilibrium\n(proportion of all infections)', cmap='inferno')

# Create directories if they don't exist (add this before saving files)
os.makedirs('./Figures', exist_ok=True)
os.makedirs('./Tables', exist_ok=True)

plt.tight_layout()
# Save figure to disk (Figures folder expected to exist)
plt.savefig('./Figures/p_recover_theta_heatmap_firstmodel.png', dpi=300)
# Show interactively if possible (depending on backend); for scripts this may save only
plt.show()

# --- optional: export results as table ---
# Flatten grids into long-form table for downstream analysis / plotting in R / Excel
df = pd.DataFrame({
    'theta': np.repeat(theta_vals, len(p_recover_vals)),               # repeat theta for each p_recover
    'p_recover': np.tile(p_recover_vals, len(theta_vals)),             # tile p_recover across theta rows
    'peak_high': peak_high.flatten(),                                  # flatten 2D -> 1D
    'eq_high_abs': eq_high_abs.flatten(),
    'eq_high_frac': eq_high_frac.flatten()
})
# Save numeric results to a CSV table for reproducibility
df.to_csv('./Tables/p_recover_theta_heatmap_results_firstmodel.csv', index=False)
print("Saved numeric results to: ./Tables/p_recover_theta_heatmap_results_firstmodel.csv")
