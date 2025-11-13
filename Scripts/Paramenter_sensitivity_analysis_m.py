#%% Imports
import numpy as np                # numerical arrays and math
import pandas as pd               # dataframes for results storage
from scipy.integrate import odeint # ODE integrator
import matplotlib.pyplot as plt   # plotting
import seaborn as sns             # nicer plotting styles
from Models.SEIRS_Models import SEIRS_first_model  # the ODE model function to integrate
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
# Integrate the ODE system once at baseline parameters to produce a results DataFrame
solution = odeint(SEIRS_first_model, y0, t, args=(parameters,))

# Column names corresponding to model state ordering returned by SEIRS_first_model
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

# Store solution in a DataFrame for easy downstream analysis and plotting
results = pd.DataFrame(solution, columns=columns)

# Unpack solution columns into named arrays for plotting convenience
Sdt, Ehdt, Indhdt, Idhdt, Rhdt, Eldt, Indldt, Idldt, Rldt = solution.T

#%% --- Sensitivity Analysis Section ---

def outcome(df):
    """
    Define the scalar model outcome used for sensitivity analysis.
    Here: peak total infection (max across time of all infected compartments).
    """
    total_infected = (
        df['Infected_NotDrug_High'] + df['Infected_Drug_High'] +
        df['Infected_NotDrug_Low'] + df['Infected_Drug_Low']
    )
    return total_infected.max()

def run_model(param_dict):
    """
    Run the ODE model with parameters provided in param_dict and return the scalar outcome.
    - Build the params tuple in the same order SEIRS_first_model expects.
    - Integrate ODE and compute the chosen outcome from the resulting DataFrame.
    """
    # Unpack parameters from input dictionary into a tuple in the correct order
    params = (
        param_dict['beta_l'],
        param_dict['birth_rate'],
        param_dict['death_rate'],
        param_dict['delta'],
        param_dict['delta_d'],
        param_dict['p_recover'],
        param_dict['phi_recover'],
        param_dict['phi_transmission'],
        param_dict['sigma'],
        param_dict['tau'],
        param_dict['theta']
    )

    # IMPORTANT: use the imported model function (SEIRS_first_model).
    # Original code used 'model' which is undefined here — that is a bug.
    sol = odeint(SEIRS_first_model, y0, t, args=(params,))

    # Convert to DataFrame using the same columns defined earlier
    df = pd.DataFrame(sol, columns=columns)

    # Return scalar metric for sensitivity analysis
    return outcome(df)

# Baseline parameters as a dictionary used for sensitivity reference
baseline = {
    'beta_l': 0.25,
    'birth_rate': 0.0,
    'death_rate': 0.0,
    'delta': 1/90,
    'delta_d': 1/3,
    'p_recover': 1.5,
    'phi_recover': 1,
    'phi_transmission': 1.05,
    'sigma': 1/10,
    'tau': 1/3,
    'theta': 0.3
}

def sensitivity_analysis(param_dict, personalized_ranges):
    """
    One-at-a-time sensitivity analysis with personalized ranges.
    Uses midpoint as baseline for fair comparison.
    """
    sens_results = {}

    # Iterate through each parameter
    for key in param_dict:
        p_low = personalized_ranges[key]['low']
        p_high = personalized_ranges[key]['high']
        p_mid = (p_low + p_high) / 2  # midpoint as baseline
        
        # Run model at low, mid, and high values
        test_params_low = param_dict.copy()
        test_params_low[key] = p_low
        outcome_low = run_model(test_params_low)
        
        test_params_mid = param_dict.copy()
        test_params_mid[key] = p_mid
        outcome_mid = run_model(test_params_mid)
        
        test_params_high = param_dict.copy()
        test_params_high[key] = p_high
        outcome_high = run_model(test_params_high)

        # Calculate sensitivity index (normalized by parameter range)
        param_range = p_high - p_low
        outcome_range = outcome_high - outcome_low
        
        # Normalized sensitivity: how much does outcome change per unit change in parameter?
        if param_range != 0:
            sensitivity_index = outcome_range / param_range
        else:
            sensitivity_index = 0

        sens_results[key] = {
            'param_low': p_low,
            'param_mid': p_mid,
            'param_high': p_high,
            'outcome_low': outcome_low,
            'outcome_mid': outcome_mid,
            'outcome_high': outcome_high,
            'sensitivity_index': sensitivity_index,
            'percent_change': (outcome_range / outcome_mid * 100) if outcome_mid != 0 else 0
        }

    return pd.DataFrame(sens_results).T

# Personalized ranges for sensitivity analysis
personalized_ranges = {
    'beta_l': {'low': 0.2, 'high': 0.3},
    'birth_rate': {'low': 0.0, 'high': 0.0},
    'death_rate': {'low': 0.0, 'high': 0.0},
    'delta': {'low': 1/100, 'high': 1/80},
    'delta_d': {'low': 1/4, 'high': 1/2},
    'p_recover': {'low': 1.0, 'high': 2.0},
    'phi_recover': {'low': 0.5, 'high': 1.0},
    'phi_transmission': {'low': 1.0, 'high': 1.15},
    'sigma': {'low': 1/15, 'high': 1/5},
    'tau': {'low': 1/4, 'high': 1/2},
    'theta': {'low': 0.0, 'high': 1.0}
}

# Run the sensitivity analysis with personalized ranges
sens_df = sensitivity_analysis(baseline, personalized_ranges)

# Print concise summary table showing midpoint baseline, outcome_low, outcome_high and sensitivity index
print("\n=== Sensitivity Analysis Results ===")
print(sens_df[['outcome_mid', 'outcome_low', 'outcome_high', 'sensitivity_index', 'percent_change']])

# Print parameter low, mid, and high values
print("\n=== Parameter Values Used ===")
print(sens_df[['param_low', 'param_mid', 'param_high']])

#%% --- Plot sensitivity results as a horizontal bar chart (beautified) ---

# Create figure and set seaborn theme for nicer style
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Sort parameters by absolute sensitivity (descending) to make the tornado plot readable
sens_sorted = sens_df.sort_values(by='sensitivity_index', key=np.abs, ascending=False)

# Draw horizontal bars centered at zero; use a diverging colormap mapped to number of params
plt.barh(
    sens_sorted.index,
    sens_sorted['sensitivity_index'],
    color=sns.color_palette('coolwarm', len(sens_sorted)),  # color for each bar
    edgecolor='black'
)

# Add vertical zero line for reference (positive effect vs negative effect)
plt.axvline(0, color='black', lw=1)
plt.xlabel('Sensitivity Index (Effect on Peak Infection)', fontsize=13)
plt.ylabel('Parameter', fontsize=13)
plt.title('Tornado Plot of Parameter Sensitivity', fontsize=15, weight='bold')

# Annotate each bar with its numeric sensitivity value
for i, (idx, val) in enumerate(zip(sens_sorted.index, sens_sorted['sensitivity_index'])):
    # place label slightly offset from bar end; choose alignment based on sign
    plt.text(val + np.sign(val) * 0.01, i, f"{val:.2f}", va='center',
             ha='left' if val >= 0 else 'right', fontsize=10)

plt.tight_layout()

# Save plot to file for reproducibility; plt.show() will also display if backend is interactive
plt.savefig('./Figures/sensitivity_tornado_plot.png', dpi=300)
plt.show()

# %%
# End of script
