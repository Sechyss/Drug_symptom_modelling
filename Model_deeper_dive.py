# Model_deeper_dive.py
# Detailed annotated script exploring the SEIRS model behaviour and stability.

#%% Imports
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from Models.SEIRS_Models import SEIRS_first_model
from Models import params as model_params  # import default parameters

# Make any stochastic behaviour reproducible
np.random.seed(42)

#%% --- Load Initial Conditions from module ---
# --- baseline parameters (loaded from Models.params, with sensible defaults) ---
# use getattr so script still runs if a name is missing in model_params
beta_l = getattr(model_params, "beta_l", 0.25)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
delta = getattr(model_params, "delta", 1/90)
delta_d = getattr(model_params, "delta_d", 1/3)
p_recover = getattr(model_params, "p_recover", 0.5)
phi_recover = getattr(model_params, "phi_recover", 1)
sigma = getattr(model_params, "sigma", 1/10)
tau = getattr(model_params, "tau", 1/3)

phi_transmission = getattr(model_params, "phi_transmission", 1.05)
theta = getattr(model_params, "theta", 0.25)
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

parameters = (beta_l, birth_rate, death_rate, delta, delta_d, p_recover,
              phi_recover, phi_transmission, sigma, tau, theta)
# --- time vector ---
# Use t_max / t_steps from params.py when available, otherwise daily for one year
t = np.linspace(0, t_max, int(t_steps))
#%% Solve ODEs (baseline run)
# Integrate the ODE system with baseline parameters to obtain time series for each compartment.
# SEIRS_first_model signature: f(y, t, params) -> derivatives
solution = odeint(SEIRS_first_model, y0, t, args=(parameters,))

# Column labels matching the ordering returned by SEIRS_first_model
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
plt.savefig('./Figures/model_deeper_dive_dynamics_model.png', dpi=300)
plt.show()

#%% 1️⃣ Check approximate equilibrium (final timepoint)
# Inspect the final row of the simulation as an approximation to equilibrium (if reached)
steady_state = results.iloc[-1]
print("\n--- Approximate Equilibrium (final day) ---")
print(steady_state)

#%% 2️⃣ Compute simple R0 approximations for intuition
# These are heuristic approximations (not full NGM for complex models), useful for quick checks:
R0_low = beta_l / sigma
R0_high = (phi_transmission * beta_l) / (phi_recover * sigma)
print(f"\nEstimated R0_low = {R0_low:.2f}")
print(f"Estimated R0_high = {R0_high:.2f}")

#%% 3️⃣ Jacobian-based local stability at the disease-free equilibrium (DFE)
def jacobian_at_dfe(params):
    """
    Construct a reduced Jacobian matrix evaluated at the Disease-Free Equilibrium.
    This Jacobian is built for the infected/exposed subspace to assess local stability.
    The ordering used here: [Eh, El, Indh, Indl]
    Note: This is a simplified Jacobian for intuition; a full Jacobian would include S,R compartments.
    """
    (beta_l, br, dr, delta, delta_d, p_rec,
     phi_rec, phi_trans, sigma, tau, theta) = params

    beta_h = phi_trans * beta_l  # high-strain transmission rate
    J = np.zeros((4, 4))

    # Fill Jacobian entries based on linearised system around DFE
    # dEh/dt terms
    J[0, 0] = -tau - dr       # Eh -> leaves at rate tau and death dr
    J[0, 2] = beta_h          # infection contribution from high infecteds to Eh

    # dEl/dt terms
    J[1, 1] = -tau - dr       # El -> leaves at rate tau and death dr
    J[1, 3] = beta_l          # infection contribution from low infecteds to El

    # dIndh/dt terms (infectious high, not-on-drug)
    J[2, 0] = tau             # Eh progresses to Indh at rate tau
    J[2, 2] = -(phi_rec * sigma + dr)  # leaves due to recovery (modified) and death

    # dIndl/dt terms (infectious low)
    J[3, 1] = tau
    J[3, 3] = -(sigma + dr)

    return J

# Compute Jacobian and eigenvalues at DFE using current parameters
J = jacobian_at_dfe(parameters)
eigs = eigvals(J)
print("\nJacobian eigenvalues at DFE:", np.round(eigs, 4))
print("Stable equilibrium?", np.all(np.real(eigs) < 0))  # stable if all real parts < 0

#%% 4️⃣ Phase plane: competition between strains
# Plot trajectory in plane (low infection on x-axis vs high infection on y-axis)
plt.figure(figsize=(7, 6))
plt.plot(Indldt + Idldt, Indhdt + Idhdt, color='purple', lw=2)
plt.xlabel("Low-virulence infection proportion")
plt.ylabel("High-virulence infection proportion")
plt.title("Phase plane: competition between strains")
plt.grid(True, alpha=0.4)
plt.savefig('./Figures/phase_plane_high_low_infection_model.png', dpi=300)
plt.show()

#%% 5️⃣ Scenario comparison: vary theta (drug usage coverage)
# Sweep theta from 0 -> 1 and record peak and equilibrium prevalence for each strain.
thetas = np.linspace(0, 1, 10)
peak_high, peak_low = [], []
eq_high, eq_low = [], []

for th in thetas:
    # Update parameters tuple for this scenario (only theta changes)
    params_var = (beta_l, birth_rate, death_rate, delta, delta_d, p_recover,
                  phi_recover, phi_transmission, sigma, tau, th)
    # NOTE: use SEIRS_first_model as the ODE function (was 'model' in earlier drafts, which is undefined)
    sol = odeint(SEIRS_first_model, y0, t, args=(params_var,))
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

#%% 6️⃣ Plot: effect of theta on infection peaks
plt.figure(figsize=(7, 5))
plt.plot(thetas, peak_low, 'o-', label='Low virulence peak')
plt.plot(thetas, peak_high, 's-', label='High virulence peak')
plt.xlabel("θ (fraction using symptom-blocking drug)")
plt.ylabel("Peak infection proportion")
plt.title("Effect of drug coverage on infection peaks")
plt.legend()
plt.tight_layout()
plt.savefig('./Figures/drug_coverage_infection_peaks_model.png', dpi=300)
plt.show()

#%% 7️⃣ Bifurcation-like plot: dominance at equilibrium as theta varies
# Fraction of infections due to high strain at equilibrium for each theta
frac_high = np.array(eq_high) / (np.array(eq_high) + np.array(eq_low) + 1e-12)  # avoid divide by zero

plt.figure(figsize=(7, 5))
plt.plot(thetas, frac_high, 'd-', color='crimson', lw=2)
plt.axhline(0.5, color='gray', ls='--', lw=1)  # 50% threshold line for dominance
plt.xlabel("θ (fraction using symptom-blocking drug)")
plt.ylabel("High virulence fraction at equilibrium")
plt.title("Bifurcation-like diagram: selection for virulence vs drug coverage")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Figures/bifurcation_virulence_drug_coverage_secondmodel.png', dpi=300)
plt.show()

# Print a concise summary table of dominance across tested theta values
print("\n--- Summary of equilibrium virulence dominance ---")
for th, f in zip(thetas, frac_high):
    dom = "High" if f > 0.5 else "Low"
    print(f"θ={th:.2f} → {f*100:.1f}% high-virulence (dominant: {dom})")
