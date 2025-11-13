# filepath: /home/albertotr/PycharmProjects/MGall_Analysis/Scripts/MG_modelling/Parameter_testing.py
"""
Parameter sensitivity analysis for SEIRS virulence-transmission trade-off model.

Model Hypothesis:
-----------------
Tests whether symptom-targeting drugs can facilitate evolution of "super-virulent" 
strains by removing the constraint that high virulence = immobile/dead hosts.

Key Biological Assumptions:
1. Transmission requires symptoms (sneezing, coughing, etc.)
2. Drug reduces symptoms but doesn't eliminate pathogen
3. High-virulence strain:
   - Produces strong symptoms → both treated AND untreated transmit
   - B_h = beta_h * (Indh + Idh)
   - Drug reduces symptoms but can't eliminate transmission completely
4. Low-virulence strain:
   - Produces mild symptoms → ONLY untreated transmit
   - B_l = beta_l * Indl
   - Drug eliminates weak symptoms → no transmission from treated cases (Idl)

Parameter Sweeps Performed:
---------------------------
1. R0_low baseline: Tests epidemic dynamics across transmission intensities (0.5-3.0)
2. p_recover sweep: Treatment effectiveness (1.2-1.8) - how much faster treated recover
3. theta sweep: Treatment coverage (0.0-1.0) - tests dose-response relationship
4. theta * p_recover interaction: 2D heatmap showing joint effects on peak exposure
5. Strain competition: High/Low dominance ratio across (theta, p_recover) grid
6. Outbreak timing: When do peak exposures occur for each strain?
7. Attack rate: Final epidemic size vs R0 and treatment coverage

Outputs:
--------
- ./Figures/: Time series plots and heatmaps for each parameter sweep
- ./Tables/: CSV files with full time series data for reproducibility
"""
#%% Imports
# import operating-system utilities (used to create output folders)
import os
# import numpy for numeric arrays and math utilities
import numpy as np
# import pandas for DataFrame creation and CSV export
import pandas as pd
# import ODE solver
from scipy.integrate import odeint
# import plotting library
import matplotlib.pyplot as plt
# import the ODE model function (first variant)
from Models.SEIRS_Models import SEIRS_first_model
# import shared parameter defaults and initial conditions
from Models import params as model_params  # load shared defaults and initial conditions

# make numpy's random behavior deterministic for reproducibility of any RNG usage
np.random.seed(42)

#%% Initial Conditions (load from Models.params and normalize to proportions)
# use getattr to read S from params.py if present, otherwise fall back to 10000
S = getattr(model_params, "S", 10000)
# Eh: exposed high strain initial count (fallback 0)
Eh = getattr(model_params, "Eh", 0)
# Indh: infected high strain NOT on drug initial count (fallback 5)
Indh = getattr(model_params, "Indh", 5)
# Idh: infected high strain ON drug initial count (fallback 0)
Idh = getattr(model_params, "Idh", 0)
# Rh: recovered high strain initial count (fallback 0)
Rh = getattr(model_params, "Rh", 0)
# El: exposed low strain initial count (fallback 0)
El = getattr(model_params, "El", 0)
# Indl: infected low strain NOT on drug initial count (fallback 5)
Indl = getattr(model_params, "Indl", 5)
# Idl: infected low strain ON drug initial count (fallback 0)
Idl = getattr(model_params, "Idl", 0)
# Rl: recovered low strain initial count (fallback 0)
Rl = getattr(model_params, "Rl", 0)

# build the initial-state vector in the same ordering that the model expects
pop_values = np.array([S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl], dtype=float)
# normalize the initial vector so compartments are proportions summing to 1
pop_values = pop_values / pop_values.sum()  # Normalize to proportions

#%% Parameters (load from Models.params with corrected comments)
# Treatment parameters
theta = getattr(model_params, "theta", 0.3)
# theta: fraction of DETECTED cases that receive treatment (0..1)
# Effective treatment rate = delta_d * theta

p_recover = getattr(model_params, "p_recover", 1.5)
# p_recover: recovery rate multiplier for treated individuals (>1 = faster recovery)
# Treated recover at: p_recover * sigma
# Untreated recover at: sigma

# Virulence modifiers
phi_transmission = getattr(model_params, "phi_transmission", 1.05)
# phi_transmission: transmission multiplier for high-virulence (beta_h = phi_transmission * beta_l)
# Default 1.05 = 5% higher transmission for high-virulence strain

phi_recover = getattr(model_params, "phi_recover", 1.0)
# phi_recover: recovery rate modifier for high-virulence (currently 1.0 = no effect)
# Future: <1 = slower recovery (longer infectious period), >1 = faster recovery

# Disease progression rates (per day)
sigma = getattr(model_params, "sigma", 1/10)       # Recovery rate (10-day infectious period)
delta = getattr(model_params, "delta", 1/90)       # Immunity waning rate (90-day immunity)
tau = getattr(model_params, "tau", 1/3)            # Exposed → Infectious rate (3-day incubation)
delta_d = getattr(model_params, "delta_d", 1/3)    # Detection rate (3 days to detection)

# Demography (set to 0 for short-term epidemic focus)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)

# Time grid
t_max = getattr(model_params, "t_max", 365)
t_steps = int(getattr(model_params, "t_steps", 365))
t = np.linspace(0, t_max, t_steps)

# Ensure output directories exist
os.makedirs('./Figures', exist_ok=True)
os.makedirs('./Tables', exist_ok=True)

#%% R0 sweep settings (corrected comment about R0 approximation)
r0_values = getattr(model_params, "r0_values", np.array([0.5, 1.0, 1.5, 2.0, 3.0]))
r0_values = np.atleast_1d(r0_values)

# Compute beta_l from target R0_low
# For untreated population: R0 ≈ beta / sigma (assuming negligible waning during outbreak)
# Therefore: beta_l = R0_low * sigma
beta_from_r0 = lambda R0: (R0 * sigma)

# Storage for time series across R0 values
Eh_matrix = np.zeros((len(r0_values), len(t)))
El_matrix = np.zeros((len(r0_values), len(t)))
R0_high_values = np.zeros(len(r0_values))

#%% Run simulations for each target R0_low (baseline p_recover and theta)
# loop over target low-strain R0 values
for idx, R0_low_target in enumerate(r0_values):
    # compute beta_l that would produce target R0_low under the simple heuristic
    beta_l_i = beta_from_r0(R0_low_target)
    # pack parameters in the order expected by SEIRS_first_model
    parameters = (beta_l_i, birth_rate, death_rate, delta, delta_d, p_recover,
                  phi_recover, phi_transmission, sigma, tau, theta)

    # call the ODE integrator: solve the system with initial state pop_values, times t, args parameters
    sol = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))
    # attempt to unpack the solution columns into named time series arrays
    try:
        S_ts, Eh_ts, Indh_ts, Idh_ts, Rh_ts, El_ts, Indl_ts, Idl_ts, Rl_ts = sol.T
    except Exception:
        # if unpacking fails because shape/order differs, convert to DataFrame and extract by index
        df = pd.DataFrame(sol)
        S_ts = df.iloc[:, 0].values
        Eh_ts = df.iloc[:, 1].values
        Indh_ts = df.iloc[:, 2].values
        Idh_ts = df.iloc[:, 3].values
        Rh_ts = df.iloc[:, 4].values
        El_ts = df.iloc[:, 5].values
        Indl_ts = df.iloc[:, 6].values
        Idl_ts = df.iloc[:, 7].values
        Rl_ts = df.iloc[:, 8].values

    # store exposed time series in the preallocated matrices
    Eh_matrix[idx, :] = Eh_ts
    El_matrix[idx, :] = El_ts

    # compute a simple heuristic for R0 of the high-virulence strain for reporting only
    R0_high_values[idx] = (phi_transmission * beta_l_i) / (phi_recover * sigma)

    # --- Diagnostic output: check treated compartment sizes and treated fraction ---
    # compute the fraction of infectious individuals that are on treatment (treated fraction)
    treated_frac = (Idh_ts + Idl_ts) / (Indh_ts + Idh_ts + Indl_ts + Idl_ts + 1e-12)
    # window used for computing means near the end of the simulation
    window = min(30, len(t))
    # print a short diagnostic line with key parameter values for this run
    print(f"[diag] R0_low={R0_low_target:.3f}  beta_l={beta_l_i:.3e}  "
          f"theta={theta:.3f}  delta_d={delta_d:.3f}  p_recover={p_recover:.3f}")
    # print maxima of treated infectious compartments and maxima of combined infectious pools
    print(f"[diag] max(Idh)={Idh_ts.max():.3e}  max(Idl)={Idl_ts.max():.3e}  "
          f"max(Indh+Idh)={(Indh_ts+Idh_ts).max():.3e}  max(Indl+Idl)={(Indl_ts+Idl_ts).max():.3e}")
    # print mean treated fraction over the final window
    print(f"[diag] mean treated fraction (last {window} steps) = {np.nanmean(treated_frac[-window:]):.4f}")

#%% Plot exposed compartments for each R0_low (baseline p_recover)
# create a new figure sized for two subplots side-by-side
plt.figure(figsize=(12, 5))

# choose a colormap and sample colors evenly for each R0 curve
colors = plt.cm.viridis(np.linspace(0, 1, len(r0_values)))

# Left subplot: Exposed High (Eh)
ax1 = plt.subplot(1, 2, 1)
# plot Eh time series for each R0 value, using assigned colors and labels
for i, R0_low_target in enumerate(r0_values):
    ax1.plot(t, Eh_matrix[i, :], color=colors[i],
             label=f"R0_low={R0_low_target:.2f}, R0_high≈{R0_high_values[i]:.2f}")
# label axes and add title for context
ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Exposed (high virulence)")
ax1.set_title("Exposed (High) — varying R0_low (baseline p_recover)")
# add legend and a subtle grid for readability
ax1.legend(fontsize='small', loc='upper right')
ax1.grid(alpha=0.3)

# Right subplot: Exposed Low (El)
ax2 = plt.subplot(1, 2, 2)
# plot El time series for each R0 value
for i, R0_low_target in enumerate(r0_values):
    ax2.plot(t, El_matrix[i, :], color=colors[i],
             label=f"R0_low={R0_low_target:.2f}")
# label axes and add title
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Exposed (low virulence)")
ax2.set_title("Exposed (Low) — varying R0_low (baseline p_recover)")
ax2.legend(fontsize='small', loc='upper right')
ax2.grid(alpha=0.3)

# adjust layout so subplots don't overlap
plt.tight_layout()
# path where the combined figure will be saved
out_path = "./Figures/R0_exposed_comparison.png"
# save the current figure to disk at high resolution
plt.savefig(out_path, dpi=600)
# print the saved location to console for confirmation
print(f"Saved R0 exposed comparison plot to: {out_path}")
# display the figure in interactive environments (may not show in headless scripts)
plt.show()

#%% Optional: save numeric results to CSV for further analysis (baseline)
# build a list of DataFrames, one per R0 setting, then concatenate
df_list = []
for i, R0_low_target in enumerate(r0_values):
    # assemble a DataFrame for this R0 run with time and exposed time series
    df_tmp = pd.DataFrame({
        "time": t,
        "Eh": Eh_matrix[i, :],
        "El": El_matrix[i, :],
        "R0_low": R0_low_target,
        "R0_high_approx": R0_high_values[i],
        "p_recover": p_recover
    })
    # append to list
    df_list.append(df_tmp)
# combine all per-run DataFrames into one long-form table
df_all = pd.concat(df_list, ignore_index=True)
# CSV filepath for baseline results
csv_out = "./Tables/R0_exposed_timeseries_baseline_p_recover.csv"
# write the table to disk (index not included)
df_all.to_csv(csv_out, index=False)
# confirmation print
print(f"Saved time series data to: {csv_out}")

#%% New section: sweep p_recover values and assess effect on exposed compartments across R0 values
# allow p_recover_values to be specified in params.py (fallback to example values)
p_recover_values = getattr(model_params, "p_recover_values", np.array([1.2, 1.5, 1.8]))
p_recover_values = np.atleast_1d(p_recover_values)  # ensure array-like

# iterate over candidate p_recover multipliers
for p_idx, p_val in enumerate(p_recover_values):
    # preallocate matrices to store exposed timeseries for this p_recover across all R0 values
    Eh_mat_p = np.zeros((len(r0_values), len(t)))
    El_mat_p = np.zeros((len(r0_values), len(t)))
    # storage for adjusted R0 estimates under current p_recover
    R0_high_adj = np.zeros(len(r0_values))
    R0_low_adj = np.zeros(len(r0_values))

    # loop over R0 values for this p_recover
    for idx, R0_low_target in enumerate(r0_values):
        # compute beta_l from the R0 target
        beta_l_i = beta_from_r0(R0_low_target)
        # assemble parameters tuple with the current p_recover value
        parameters = (beta_l_i, birth_rate, death_rate, delta, delta_d, p_val,
                      phi_recover, phi_transmission, sigma, tau, theta)

        # integrate the ODE with these parameters
        sol = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))

        # attempt to unpack expected columns into named arrays
        try:
            S_ts, Eh_ts, Indh_ts, Idh_ts, Rh_ts, El_ts, Indl_ts, Idl_ts, Rl_ts = sol.T
        except Exception:
            # fallback to DataFrame extraction by column index
            df = pd.DataFrame(sol)
            S_ts = df.iloc[:, 0].values
            Eh_ts = df.iloc[:, 1].values
            Indh_ts = df.iloc[:, 2].values
            Idh_ts = df.iloc[:, 3].values
            Rh_ts = df.iloc[:, 4].values
            El_ts = df.iloc[:, 5].values
            Indl_ts = df.iloc[:, 6].values
            Idl_ts = df.iloc[:, 7].values
            Rl_ts = df.iloc[:, 8].values

        # save exposed time series into matrices for plotting later
        Eh_mat_p[idx, :] = Eh_ts
        El_mat_p[idx, :] = El_ts

        # compute approximate R0s for reporting that incorporate treatment fraction (theta)
        avg_rec_high = phi_recover * sigma * ((1 - theta) + theta * p_val)
        avg_rec_low = sigma * ((1 - theta) + theta * p_val)
        R0_high_adj[idx] = (phi_transmission * beta_l_i) / (avg_rec_high + 1e-12)
        R0_low_adj[idx] = beta_l_i / (avg_rec_low + 1e-12)

        # --- Diagnostic output for p_recover sweep ---
        # fraction of infectious that are treated (Idh + Idl) / total infectious
        treated_frac = (Idh_ts + Idl_ts) / (Indh_ts + Idh_ts + Indl_ts + Idl_ts + 1e-12)
        # window of recent timesteps to average diagnostics over
        window = min(30, len(t))
        # print summary diagnostics for this simulation
        print(f"[diag p_recover={p_val:.3f}] R0_low={R0_low_target:.3f}  beta_l={beta_l_i:.3e}  "
              f"theta={theta:.3f}  delta_d={delta_d:.3f}")
        print(f"[diag p_recover={p_val:.3f}] max(Idh)={Idh_ts.max():.3e}  max(Idl)={Idl_ts.max():.3e}  "
              f"mean treated fraction (last {window} steps) = {np.nanmean(treated_frac[-window:]):.4f}")

    # Plotting section for the current p_recover value: two subplots side-by-side
    plt.figure(figsize=(12, 5))
    # choose a color gradient for the different R0 curves
    colors = plt.cm.plasma(np.linspace(0, 1, len(r0_values)))

    # subplot: Exposed High time series across R0 values
    ax1 = plt.subplot(1, 2, 1)
    for i, R0_low_target in enumerate(r0_values):
        ax1.plot(t, Eh_mat_p[i, :], color=colors[i],
                 label=f"R0_low={R0_low_target:.2f}, R0_high≈{R0_high_adj[i]:.2f}")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Exposed (high virulence)")
    ax1.set_title(f"Exposed (High) — varying R0_low (p_recover={p_val:.2f})")
    ax1.legend(fontsize='x-small', loc='upper right')
    ax1.grid(alpha=0.3)

    # subplot: Exposed Low time series across R0 values
    ax2 = plt.subplot(1, 2, 2)
    for i, R0_low_target in enumerate(r0_values):
        ax2.plot(t, El_mat_p[i, :], color=colors[i],
                 label=f"R0_low={R0_low_target:.2f}, R0_low_adj≈{R0_low_adj[i]:.2f}")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Exposed (low virulence)")
    ax2.set_title(f"Exposed (Low) — varying R0_low (p_recover={p_val:.2f})")
    ax2.legend(fontsize='x-small', loc='upper right')
    ax2.grid(alpha=0.3)

    # layout and save the figure for this p_recover
    plt.tight_layout()
    fig_name = f"./Figures/R0_exposed_p_recover_{p_val:.2f}.png"
    plt.savefig(fig_name, dpi=300)
    print(f"Saved figure for p_recover={p_val:.2f} to: {fig_name}")
    plt.show()

    # Save numeric results for this p_recover to a CSV table
    df_list_p = []
    for i, R0_low_target in enumerate(r0_values):
        # create per-run DataFrame with time and exposed series plus metadata
        df_tmp = pd.DataFrame({
            "time": t,
            "Eh": Eh_mat_p[i, :],
            "El": El_mat_p[i, :],
            "R0_low": R0_low_target,
            "R0_high_adj": R0_high_adj[i],
            "R0_low_adj": R0_low_adj[i],
            "p_recover": p_val
        })
        df_list_p.append(df_tmp)
    # combine all runs for this p_recover into a single table
    df_all_p = pd.concat(df_list_p, ignore_index=True)
    # write CSV to Tables directory
    csv_out_p = f"./Tables/R0_exposed_timeseries_p_recover_{p_val:.2f}.csv"
    df_all_p.to_csv(csv_out_p, index=False)
    print(f"Saved numeric results for p_recover={p_val:.2f} to: {csv_out_p}")

#%% New section: sweep theta values (treatment coverage) and assess effect on exposed compartments
# allow theta_values to be specified in params.py (fallback to example values)
# theta represents the fraction of exposed individuals who will take treatment
# range from 0.0 (no one takes treatment) to 1.0 (everyone takes treatment)
theta_values = getattr(model_params, "theta_values", np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
theta_values = np.atleast_1d(theta_values)  # ensure array-like even if single value provided

# iterate over candidate theta values (treatment coverage levels)
for theta_idx, theta_val in enumerate(theta_values):
    # print section header for clarity in console output during long runs
    print(f"\n{'='*60}")
    print(f"Starting theta sweep: theta = {theta_val:.2f} ({theta_idx+1}/{len(theta_values)})")
    print(f"{'='*60}")
    
    # preallocate matrices to store exposed timeseries for this theta across all R0 values
    # each row corresponds to one R0 value, columns are time points
    Eh_mat_theta = np.zeros((len(r0_values), len(t)))  # Exposed high-virulence strain
    El_mat_theta = np.zeros((len(r0_values), len(t)))  # Exposed low-virulence strain
    
    # storage arrays for R0 estimates under current theta
    # these will be computed based on effective recovery rates that account for treatment
    R0_high_theta = np.zeros(len(r0_values))  # adjusted R0 for high-virulence strain
    R0_low_theta = np.zeros(len(r0_values))   # adjusted R0 for low-virulence strain

    # loop over R0 values for this theta: for each theta, test all R0 scenarios
    for idx, R0_low_target in enumerate(r0_values):
        # compute beta_l (transmission rate) from target R0_low using heuristic formula
        # R0 ≈ beta / sigma, so beta = R0 * sigma
        beta_l_i = beta_from_r0(R0_low_target)
        
        # pack parameters tuple with current theta_val (use baseline p_recover from params)
        # parameter order: beta_l, birth_rate, death_rate, delta, delta_d, p_recover,
        #                  phi_recover, phi_transmission, sigma, tau, theta
        parameters = (beta_l_i, birth_rate, death_rate, delta, delta_d, p_recover,
                      phi_recover, phi_transmission, sigma, tau, theta_val)

        # solve ODE system: integrate from initial conditions over time grid t
        sol = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))
        
        # unpack solution array into named compartment time series
        # expected order: S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl
        try:
            # attempt direct unpacking if columns match expected format
            S_ts, Eh_ts, Indh_ts, Idh_ts, Rh_ts, El_ts, Indl_ts, Idl_ts, Rl_ts = sol.T
        except Exception:
            # fallback: extract columns by index in case of shape mismatch
            S_ts = sol[:, 0]      # Susceptible
            Eh_ts = sol[:, 1]     # Exposed high-virulence
            Indh_ts = sol[:, 2]   # Infected high-virulence, not on drug
            Idh_ts = sol[:, 3]    # Infected high-virulence, on drug
            Rh_ts = sol[:, 4]     # Recovered from high-virulence
            El_ts = sol[:, 5]     # Exposed low-virulence
            Indl_ts = sol[:, 6]   # Infected low-virulence, not on drug
            Idl_ts = sol[:, 7]    # Infected low-virulence, on drug
            Rl_ts = sol[:, 8]     # Recovered from low-virulence

        # store exposed time series
        Eh_mat_theta[idx, :] = Eh_ts
        El_mat_theta[idx, :] = El_ts

        # compute R0 estimates (accounting for theta effect on effective recovery rate)
        # individuals taking drug recover faster by factor p_recover
        # effective recovery for high: phi_recover * sigma * (1 - theta_val + theta_val * p_recover)
        sigma_eff_high = phi_recover * sigma * (1 - theta_val + theta_val * p_recover)
        sigma_eff_low = sigma * (1 - theta_val + theta_val * p_recover)
        
        R0_high_theta[idx] = (phi_transmission * beta_l_i) / sigma_eff_high
        R0_low_theta[idx] = beta_l_i / sigma_eff_low

        # --- Diagnostic output ---
        treated_frac = (Idh_ts + Idl_ts) / (Indh_ts + Idh_ts + Indl_ts + Idl_ts + 1e-12)
        window = min(30, len(t))
        print(f"[diag] theta={theta_val:.3f}  R0_low_target={R0_low_target:.3f}  "
              f"R0_low_adj={R0_low_theta[idx]:.3f}  R0_high_adj={R0_high_theta[idx]:.3f}")
        print(f"[diag] max(Idh)={Idh_ts.max():.3e}  max(Idl)={Idl_ts.max():.3e}")
        print(f"[diag] mean treated fraction (last {window} steps) = {np.nanmean(treated_frac[-window:]):.4f}")

    # Plotting section for the current theta value: visualize exposed compartments across R0 values
    plt.figure(figsize=(12, 5))  # create figure with two side-by-side subplots
    # generate color gradient for different R0 curves using plasma colormap
    colors = plt.cm.plasma(np.linspace(0, 1, len(r0_values)))

    # Left subplot: Exposed High (Eh) time series across R0 values
    ax1 = plt.subplot(1, 2, 1)
    for i, R0_low_target in enumerate(r0_values):
        ax1.plot(t, Eh_mat_theta[i, :], color=colors[i],
                 label=f"R0_low_target={R0_low_target:.2f}, R0_low_adj={R0_low_theta[i]:.2f}")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Exposed (high virulence)")
    ax1.set_title(f"Exposed (High) — varying R0_low (theta={theta_val:.2f})")
    ax1.legend(fontsize='small', loc='best')  # automatic legend positioning
    ax1.grid(alpha=0.3)  # subtle grid for easier reading

    # Right subplot: Exposed Low (El) time series across R0 values
    ax2 = plt.subplot(1, 2, 2)
    for i, R0_low_target in enumerate(r0_values):
        ax2.plot(t, El_mat_theta[i, :], color=colors[i],
                 label=f"R0_low_target={R0_low_target:.2f}, R0_low_adj={R0_low_theta[i]:.2f}")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Exposed (low virulence)")
    ax2.set_title(f"Exposed (Low) — varying R0_low (theta={theta_val:.2f})")
    ax2.legend(fontsize='small', loc='best')
    ax2.grid(alpha=0.3)

    # adjust layout to prevent subplot overlap and label clipping
    plt.tight_layout()
    
    # construct output path with theta value embedded in filename for unique identification
    out_path = f"./Figures/R0_exposed_theta_{theta_val:.2f}.png"
    # save figure to disk at high resolution (600 dpi)
    plt.savefig(out_path, dpi=600)
    # print confirmation message with file location
    print(f"Saved theta={theta_val:.2f} plot to: {out_path}")
    # display figure in interactive environments (Jupyter, IDE with plot viewer)
    plt.show()

    # Save numeric results to CSV for further statistical analysis or reproducibility
    df_list_theta = []  # accumulator for per-R0 DataFrames
    for i, R0_low_target in enumerate(r0_values):
        # create DataFrame with time series and metadata for this R0 scenario
        df_tmp = pd.DataFrame({
            "time": t,                          # time points (days)
            "Eh": Eh_mat_theta[i, :],          # exposed high-virulence time series
            "El": El_mat_theta[i, :],          # exposed low-virulence time series
            "R0_low_target": R0_low_target,    # target R0 used to compute beta_l
            "R0_low_adj": R0_low_theta[i],     # adjusted R0_low accounting for treatment
            "R0_high_adj": R0_high_theta[i],   # adjusted R0_high accounting for treatment
            "theta": theta_val,                 # treatment coverage fraction for this sweep
            "p_recover": p_recover              # recovery multiplier (from baseline params)
        })
        df_list_theta.append(df_tmp)
    
    # concatenate all R0 scenarios into single long-form DataFrame
    df_theta = pd.concat(df_list_theta, ignore_index=True)
    # construct CSV output path with theta value in filename
    csv_out = f"./Tables/R0_exposed_timeseries_theta_{theta_val:.2f}.csv"
    # write DataFrame to CSV without row indices
    df_theta.to_csv(csv_out, index=False)
    # print confirmation with file location
    print(f"Saved time series data to: {csv_out}")

# print final completion message after all theta values processed
print("\n" + "="*60)
print("Theta sweep analysis complete!")
print("="*60)

#%% Combined theta × p_recover interaction analysis
# Objective: visualize how treatment coverage (theta) and treatment effectiveness (p_recover)
# jointly influence disease dynamics using 2D heatmaps of peak exposed compartments

# Create parameter grids for systematic exploration
# theta_grid: treatment coverage ranging from 0% to 100% in 6 steps
theta_grid = np.linspace(0, 1, 6)  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# p_recover_grid: treatment effectiveness multiplier from 1.0 (no benefit) to 2.0 (doubles recovery)
p_recover_grid = np.linspace(1.0, 2.0, 6)  # [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# Preallocate 2D arrays to store peak exposed values across parameter combinations
# rows: different theta values, columns: different p_recover values
peak_Eh_grid = np.zeros((len(theta_grid), len(p_recover_grid)))  # peak exposed high-virulence
peak_El_grid = np.zeros((len(theta_grid), len(p_recover_grid)))  # peak exposed low-virulence

# Nested loop: iterate over all (theta, p_recover) combinations
for i, theta_val in enumerate(theta_grid):
    for j, p_val in enumerate(p_recover_grid):
        # Use moderate baseline R0 = 1.5 for this comparison
        # (epidemic threshold, neither too weak nor too explosive)
        beta_l_i = beta_from_r0(1.5)
        
        # Pack parameters with current theta and p_recover values
        # order: beta_l, birth_rate, death_rate, delta, delta_d, p_recover,
        #        phi_recover, phi_transmission, sigma, tau, theta
        parameters = (beta_l_i, birth_rate, death_rate, delta, delta_d, p_val,
                      phi_recover, phi_transmission, sigma, tau, theta_val)
        
        # Solve ODE system from initial conditions over time grid t
        sol = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))
        
        # Extract exposed compartment time series
        Eh_ts = sol[:, 1]  # exposed high-virulence (column index 1)
        El_ts = sol[:, 5]  # exposed low-virulence (column index 5)
        
        # Store maximum (peak) values in the grid
        # this represents worst-case exposure burden for each scenario
        peak_Eh_grid[i, j] = Eh_ts.max()
        peak_El_grid[i, j] = El_ts.max()

# Visualization: create side-by-side heatmaps for both strains
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left heatmap: Peak Exposed High-Virulence
# imshow displays 2D array as color-coded image
# origin='lower' ensures theta=0 is at bottom (intuitive orientation)
im1 = ax1.imshow(peak_Eh_grid, aspect='auto', cmap='YlOrRd', origin='lower')
ax1.set_xlabel('p_recover (treatment effectiveness)')
ax1.set_ylabel('theta (treatment coverage)')
ax1.set_title('Peak Exposed (High Virulence)')
# Set tick positions and labels to show actual parameter values
ax1.set_xticks(range(len(p_recover_grid)))
ax1.set_xticklabels([f'{p:.2f}' for p in p_recover_grid])  # format as 2 decimals
ax1.set_yticks(range(len(theta_grid)))
ax1.set_yticklabels([f'{t:.2f}' for t in theta_grid])
# Add color bar to show peak value scale
plt.colorbar(im1, ax=ax1, label='Peak Exposed Fraction')

# Right heatmap: Peak Exposed Low-Virulence
# same structure as left plot but for low-virulence strain
im2 = ax2.imshow(peak_El_grid, aspect='auto', cmap='YlGnBu', origin='lower')
ax2.set_xlabel('p_recover (treatment effectiveness)')
ax2.set_ylabel('theta (treatment coverage)')
ax2.set_title('Peak Exposed (Low Virulence)')
ax2.set_xticks(range(len(p_recover_grid)))
ax2.set_xticklabels([f'{p:.2f}' for p in p_recover_grid])
ax2.set_yticks(range(len(theta_grid)))
ax2.set_yticklabels([f'{t:.2f}' for t in theta_grid])
plt.colorbar(im2, ax=ax2, label='Peak Exposed Fraction')

# Adjust spacing to prevent overlap
plt.tight_layout()
# Save high-quality figure
plt.savefig('./Figures/theta_p_recover_heatmap.png', dpi=300)
plt.show()

print("Heatmap analysis complete: check ./Figures/theta_p_recover_heatmap.png")

#%% Strain competition analysis
# Objective: quantify which strain dominates for theta∈[0,1] and p_recover∈[1,2]

# Use fine grids strictly within requested ranges
n_theta, n_p = 21, 21  # adjust resolution as needed
theta_grid = np.linspace(0.0, 1.0, n_theta)         # [0.00 ... 1.00]
p_recover_grid = np.linspace(1.0, 2.0, n_p)         # [1.00 ... 2.00]

def compute_dominance_ratio(theta_val, p_val, R0_low_target):
    """Returns ratio of peak high- to low-strain infectious prevalence."""
    beta_l_i = beta_from_r0(R0_low_target)
    parameters = (beta_l_i, birth_rate, death_rate, delta, delta_d, p_val,
                  phi_recover, phi_transmission, sigma, tau, theta_val)
    sol = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))
    I_high = sol[:, 2] + sol[:, 3]  # Indh + Idh
    I_low  = sol[:, 6] + sol[:, 7]  # Indl + Idl
    return I_high.max() / (I_low.max() + 1e-12)

dominance_surface = np.zeros((len(theta_grid), len(p_recover_grid)))
for i, theta_val in enumerate(theta_grid):
    for j, p_val in enumerate(p_recover_grid):
        dominance_surface[i, j] = compute_dominance_ratio(theta_val, p_val, 1.5)

plt.figure(figsize=(8, 6))
im = plt.imshow(dominance_surface, aspect='auto', cmap='RdBu_r', origin='lower',
                vmin=0, vmax=2)
plt.colorbar(im, label='High/Low strain dominance ratio')
plt.xlabel('p_recover (1.0–2.0)')
plt.ylabel('theta (0.0–1.0)')
plt.title('Strain Competition: High/Low Ratio (ratio > 1: high strain wins)')

# Clean ticks: show 6 evenly spaced labels across the actual value ranges
x_idx = np.linspace(0, len(p_recover_grid) - 1, 6).round().astype(int)
y_idx = np.linspace(0, len(theta_grid) - 1, 6).round().astype(int)
plt.xticks(x_idx, [f'{p_recover_grid[k]:.2f}' for k in x_idx])
plt.yticks(y_idx, [f'{theta_grid[k]:.2f}' for k in y_idx])

plt.tight_layout()
plt.savefig('./Figures/strain_dominance_heatmap.png', dpi=300)
plt.show()
print("Strain dominance analysis complete: check ./Figures/strain_dominance_heatmap.png")

#%% Outbreak timing analysis
def analyze_outbreak_timing(theta_val, R0_low_target):
    beta_l_i = beta_from_r0(R0_low_target)
    parameters = (beta_l_i, birth_rate, death_rate, delta, delta_d, p_recover,
                  phi_recover, phi_transmission, sigma, tau, theta_val)
    
    sol = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))
    
    Eh_ts = sol[:, 1]
    El_ts = sol[:, 5]
    
    return t[Eh_ts.argmax()], t[El_ts.argmax()]

# Compare timing across theta and R0
results = []
for theta_val in theta_values:
    for R0_val in r0_values:
        peak_time_high, peak_time_low = analyze_outbreak_timing(theta_val, R0_val)
        results.append({
            'theta': theta_val,
            'R0': R0_val,
            'peak_time_high': peak_time_high,
            'peak_time_low': peak_time_low,
            'time_difference': peak_time_high - peak_time_low
        })

df_timing = pd.DataFrame(results)
df_timing.to_csv('./Tables/outbreak_timing_analysis.csv', index=False)

#%% Final size / attack rate analysis
def compute_attack_rate(theta_val, p_val, R0_val):
    beta_l_i = beta_from_r0(R0_val)
    parameters = (beta_l_i, birth_rate, death_rate, delta, delta_d, p_val,
                  phi_recover, phi_transmission, sigma, tau, theta_val)
    
    sol = odeint(SEIRS_first_model, pop_values, t, args=(parameters,))
    
    # Final recovered = final S subtracted from initial S
    S_initial = sol[0, 0]
    S_final = sol[-1, 0]
    attack_rate = (S_initial - S_final) / S_initial
    
    return attack_rate

# Calculate for grid
attack_grid = np.zeros((len(theta_grid), len(r0_values)))
for i, theta_val in enumerate(theta_grid):
    for j, R0_val in enumerate(r0_values):
        attack_grid[i, j] = compute_attack_rate(theta_val, p_recover, R0_val)

plt.figure(figsize=(10, 6))
for i, theta_val in enumerate(theta_grid):
    plt.plot(r0_values, attack_grid[i, :], marker='o', label=f'θ={theta_val:.2f}')
plt.xlabel('R₀')
plt.ylabel('Attack Rate (fraction infected)')
plt.title('Final Attack Rate vs R₀ and Treatment Coverage')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('./Figures/attack_rate_analysis.png', dpi=300)
plt.show()