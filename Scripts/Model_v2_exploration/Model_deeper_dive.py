# Model_deeper_dive.py
# Detailed annotated script exploring the SEIRS model behaviour and stability.

#%% Imports
import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

# Add parent directory to path to allow imports from Models/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.SEIRS_Models import SEIRS_model_v2
from Models import params as model_params  # import default parameters

# Make any stochastic behaviour reproducible
np.random.seed(42)

#%% --- Load Initial Conditions from module ---
# --- baseline parameters (loaded from Models.params, with sensible defaults) ---
# use getattr so script still runs if a name is missing in model_params
contact_rate = getattr(model_params, "contact_rate", 10.0)
transmission_probability = getattr(model_params, "transmission_probability", 0.025)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
delta = getattr(model_params, "delta", 1/120)
kappa_base = getattr(model_params, "kappa_base", 1.0)
kappa_scale = getattr(model_params, "kappa_scale", 1.0)
p_recover = getattr(model_params, "p_recover", 0.5)
phi_recover = getattr(model_params, "phi_recover", 1.0)
phi_transmission = getattr(model_params, "phi_transmission", 1.05)
sigma = getattr(model_params, "sigma", 1/5)
tau = getattr(model_params, "tau", 1/3)
theta = getattr(model_params, "theta", 0.3)
t_max = getattr(model_params, "t_max", 365)
t_steps = getattr(model_params, "t_steps", 365)

# Compute beta_l from contact_rate and transmission_probability
beta_l = contact_rate * transmission_probability

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

# Updated parameters tuple for SEIRS_model_v2 (13 parameters)
parameters = (contact_rate, transmission_probability, birth_rate, death_rate, delta, 
              kappa_base, kappa_scale, p_recover, phi_recover, phi_transmission, 
              sigma, tau, theta)

# --- time vector ---
# Use t_max / t_steps from params.py when available, otherwise daily for one year
t = np.linspace(0, t_max, int(t_steps))

#%% Solve ODEs (baseline run)
# Integrate the ODE system with baseline parameters to obtain time series for each compartment.
# SEIRS_model_v2 signature: f(y, t, params) -> derivatives
solution = odeint(SEIRS_model_v2, y0, t, args=(parameters,))

# Column labels matching the ordering returned by SEIRS_model_v2
columns = [
    'Susceptible',
    'Exposed_High',
    'Infected_NotDrug_High',
    'Infected_Drug_High',
    'Recovered_High',
    'Exposed_Low',
    'Infected_NotDrug_Low',
    'Infected_Drug_Low',
    'Recovered_Low'
]

# Store solution in a DataFrame for convenient inspection and save/plot
results = pd.DataFrame(solution, columns=columns)

# Unpack solution columns into named arrays for plotting convenience
Sdt, Ehdt, Indhdt, Idhdt, Rhdt, Eldt, Indldt, Idldt, Rldt = solution.T

#%% Plot time dynamics (baseline)
# Visualise time series for each compartment.
fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111, facecolor='#f4f4f4', axisbelow=True)

# Plot each compartment with distinct colours and labels
# ax.plot(t, Sdt, 'b', lw=2, label='Susceptible')
ax.plot(t, Ehdt, 'y', lw=2, label='Exposed High')
# ax.plot(t, Indhdt, 'r', lw=2, label='Infected Not Drug High')
# ax.plot(t, Idhdt, 'm', lw=2, label='Infected Drug High')
# ax.plot(t, Rhdt, 'g', lw=2, label='Recovered High')
ax.plot(t, Eldt, 'c', lw=2, label='Exposed Low')
# ax.plot(t, Indldt, color='orange', lw=2, label='Infected Not Drug Low')
# ax.plot(t, Idldt, color='brown', lw=2, label='Infected Drug Low')
# ax.plot(t, Rldt, color='olive', lw=2, label='Recovered Low')

# Axis labels, legend, layout and save figure
ax.set_xlabel('Time (days)')
ax.set_ylabel('Proportion of Population')
ax.legend(framealpha=0.7)
plt.tight_layout()
plt.savefig('../Figures/model_deeper_dive_dynamics_model_v2.png', dpi=600)
plt.show()

#%% 1️⃣ Check approximate equilibrium (final timepoint)
# Inspect the final row of the simulation as an approximation to equilibrium (if reached)
steady_state = results.iloc[-1]
print("\n--- Approximate Equilibrium (final day) ---")
print(steady_state)

#%% 2️⃣ Compute R0 approximations accounting for treatment and kappa
# Calculate effective kappa values
virulence_excess = phi_transmission - 1.0
kappa_high = kappa_base * (1 + kappa_scale * virulence_excess)
kappa_low = kappa_base

# Safety: ensure kappa * theta ≤ 1
if theta > 0:
    kappa_high = min(kappa_high, 1.0 / theta)
    kappa_low = min(kappa_low, 1.0 / theta)

# Effective treatment fractions
theta_eff_high = kappa_high * theta
theta_eff_low = kappa_low * theta

# Compute beta values
beta_h = phi_transmission * beta_l

# Effective transmission rates (accounting for treatment reduction)
beta_eff_high = beta_h * (1 - theta_eff_high + p_recover * theta_eff_high)
beta_eff_low = beta_l * (1 - theta_eff_low + p_recover * theta_eff_low)

# Effective recovery rates
sigma_eff_high = phi_recover * sigma
sigma_eff_low = sigma

# R0 calculations
R0_low = beta_eff_low / sigma_eff_low
R0_high = beta_eff_high / sigma_eff_high

print(f"\n--- R0 Calculations (accounting for treatment effects) ---")
print(f"kappa_high = {kappa_high:.3f}, kappa_low = {kappa_low:.3f}")
print(f"Effective treatment coverage: high={theta_eff_high:.3f}, low={theta_eff_low:.3f}")
print(f"R0_low = {R0_low:.3f}")
print(f"R0_high = {R0_high:.3f}")
print(f"R0 ratio (high/low) = {R0_high/R0_low:.3f}")

#%% 3️⃣ Jacobian-based local stability at the disease-free equilibrium (DFE)
def jacobian_at_dfe(params):
    """
    Construct a reduced Jacobian matrix evaluated at the Disease-Free Equilibrium.
    This Jacobian is built for the infected/exposed subspace to assess local stability.
    The ordering used here: [Eh, El, Indh, Indl, Idh, Idl]
    Note: This is a simplified Jacobian for intuition; a full Jacobian would include S,R compartments.
    """
    (contact_rate, trans_prob, br, dr, delta, 
     kappa_base, kappa_scale, p_rec, phi_rec, phi_trans, 
     sigma, tau, theta) = params

    beta_l = contact_rate * trans_prob
    beta_h = phi_trans * beta_l
    
    # Compute kappa values
    virulence_excess = phi_trans - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * virulence_excess)
    kappa_low = kappa_base
    
    # Safety caps
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)
    
    J = np.zeros((6, 6))

    # dEh/dt terms
    J[0, 0] = -tau - dr           # Eh self
    J[0, 2] = beta_h              # from Indh
    J[0, 4] = beta_h * p_rec      # from Idh (reduced transmission)

    # dEl/dt terms
    J[1, 1] = -tau - dr           # El self
    J[1, 3] = beta_l              # from Indl
    J[1, 5] = beta_l * p_rec      # from Idl (reduced transmission)

    # dIndh/dt terms (high, untreated)
    J[2, 0] = (1 - kappa_high * theta) * tau  # from Eh
    J[2, 2] = -(phi_rec * sigma + dr)

    # dIndl/dt terms (low, untreated)
    J[3, 1] = (1 - kappa_low * theta) * tau   # from El
    J[3, 3] = -(sigma + dr)

    # dIdh/dt terms (high, treated)
    J[4, 0] = kappa_high * theta * tau        # from Eh
    J[4, 4] = -(phi_rec * sigma + dr)

    # dIdl/dt terms (low, treated)
    J[5, 1] = kappa_low * theta * tau         # from El
    J[5, 5] = -(sigma + dr)

    return J

# Compute Jacobian and eigenvalues at DFE using current parameters
J = jacobian_at_dfe(parameters)
eigs = eigvals(J)
print("\n--- Jacobian Analysis at DFE ---")
print("Eigenvalues:", np.round(eigs, 4))
print("Stable equilibrium?", np.all(np.real(eigs) < 0))  # stable if all real parts < 0
print("Leading eigenvalue (real part):", np.max(np.real(eigs)))

#%% 4️⃣ Phase plane: competition between strains
# Plot trajectory in plane (low infection on x-axis vs high infection on y-axis)
plt.figure(figsize=(7, 6))
plt.plot(Indldt + Idldt, Indhdt + Idhdt, color='purple', lw=2)
plt.xlabel("Low-virulence infection proportion")
plt.ylabel("High-virulence infection proportion")
plt.title("Phase plane: competition between strains")
plt.grid(True, alpha=0.4)
plt.savefig('../Figures/phase_plane_high_low_infection_model_v2.png', dpi=300)
plt.show()

#%% 5️⃣ Scenario comparison: vary theta (drug usage coverage)
# Sweep theta from 0 -> 1 and record peak and equilibrium prevalence for each strain.
thetas = np.linspace(0, 1, 10)
peak_high, peak_low = [], []
eq_high, eq_low = [], []
R0_highs, R0_lows = [], []

for th in thetas:
    # Update parameters tuple for this scenario (only theta changes)
    params_var = (contact_rate, transmission_probability, birth_rate, death_rate, delta, 
                  kappa_base, kappa_scale, p_recover, phi_recover, phi_transmission, 
                  sigma, tau, th)
    
    sol = odeint(SEIRS_model_v2, y0, t, args=(params_var,))
    # Unpack solution; only need infected compartments for metrics
    _, _, Indh, Idh, _, _, Indl, Idl, _ = sol.T

    # Compute totals (drug and not-on-drug)
    total_high = Indh + Idh
    total_low = Indl + Idl

    # Record peak prevalence and approximate equilibrium (mean of last 30 days)
    peak_high.append(np.max(total_high))
    peak_low.append(np.max(total_low))
    eq_high.append(np.mean(total_high[-30:]))  # last 30 timepoints as equilibrium proxy
    eq_low.append(np.mean(total_low[-30:]))
    
    # Compute R0 for this theta
    virulence_excess = phi_transmission - 1.0
    kappa_high_tmp = kappa_base * (1 + kappa_scale * virulence_excess)
    kappa_low_tmp = kappa_base
    if th > 0:
        kappa_high_tmp = min(kappa_high_tmp, 1.0 / th)
        kappa_low_tmp = min(kappa_low_tmp, 1.0 / th)
    
    theta_eff_high_tmp = kappa_high_tmp * th
    theta_eff_low_tmp = kappa_low_tmp * th
    
    beta_eff_high_tmp = beta_h * (1 - theta_eff_high_tmp + p_recover * theta_eff_high_tmp)
    beta_eff_low_tmp = beta_l * (1 - theta_eff_low_tmp + p_recover * theta_eff_low_tmp)
    
    R0_high_tmp = beta_eff_high_tmp / (phi_recover * sigma)
    R0_low_tmp = beta_eff_low_tmp / sigma
    
    R0_highs.append(R0_high_tmp)
    R0_lows.append(R0_low_tmp)

#%% 6️⃣ Plot: effect of theta on infection peaks
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: infection peaks
axes[0].plot(thetas, peak_low, 'o-', label='Low virulence peak', color='steelblue')
axes[0].plot(thetas, peak_high, 's-', label='High virulence peak', color='crimson')
axes[0].set_xlabel("θ (treatment coverage)")
axes[0].set_ylabel("Peak infection proportion")
axes[0].set_title("Effect of treatment coverage on infection peaks")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right panel: R0 values
axes[1].plot(thetas, R0_lows, 'o-', label='R0 low', color='steelblue')
axes[1].plot(thetas, R0_highs, 's-', label='R0 high', color='crimson')
axes[1].axhline(1.0, color='gray', ls='--', lw=1, label='R0=1')
axes[1].set_xlabel("θ (treatment coverage)")
axes[1].set_ylabel("R0")
axes[1].set_title("Effect of treatment coverage on R0")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../Figures/drug_coverage_effects_model_v2.png', dpi=600)
plt.show()

#%% 7️⃣ Bifurcation-like plot: dominance at equilibrium as theta varies
# Fraction of infections due to high strain at equilibrium for each theta
frac_high = np.array(eq_high) / (np.array(eq_high) + np.array(eq_low) + 1e-12)  # avoid divide by zero

plt.figure(figsize=(7, 5))
plt.plot(thetas, frac_high, 'd-', color='crimson', lw=2)
plt.axhline(0.5, color='gray', ls='--', lw=1)  # 50% threshold line for dominance
plt.xlabel("θ (treatment coverage)")
plt.ylabel("High virulence fraction at equilibrium")
plt.title("Selection for virulence vs treatment coverage\n(kappa-adjusted model)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../Figures/bifurcation_virulence_drug_coverage_model_v2.png', dpi=600)
plt.show()

# Print a concise summary table of dominance across tested theta values
print("\n--- Summary of equilibrium virulence dominance ---")
print(f"{'θ':>6} {'High%':>8} {'Dominant':>10} {'R0_low':>8} {'R0_high':>8}")
print("-" * 50)
for th, f, r0l, r0h in zip(thetas, frac_high, R0_lows, R0_highs):
    dom = "High" if f > 0.5 else "Low"
    print(f"{th:>6.2f} {f*100:>8.1f} {dom:>10} {r0l:>8.3f} {r0h:>8.3f}")

#%% 8️⃣ Additional analysis: kappa effect visualization
# Show how kappa_high varies with phi_transmission
phi_trans_range = np.linspace(1.0, 1.2, 21)
kappa_high_vals = [kappa_base * (1 + kappa_scale * (pt - 1.0)) for pt in phi_trans_range]

plt.figure(figsize=(7, 5))
plt.plot(phi_trans_range, kappa_high_vals, 'o-', color='darkgreen', lw=2)
plt.axhline(kappa_base, color='gray', ls='--', lw=1, label=f'kappa_base={kappa_base}')
plt.xlabel("phi_transmission (virulence)")
plt.ylabel("kappa_high (detection multiplier)")
plt.title(f"Detection scaling with virulence\n(kappa_scale={kappa_scale})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../Figures/kappa_virulence_relationship.png', dpi=300)
plt.show()

print("\n" + "="*70)
print("Model analysis complete! Figures saved to ../Figures/")
print("="*70)
