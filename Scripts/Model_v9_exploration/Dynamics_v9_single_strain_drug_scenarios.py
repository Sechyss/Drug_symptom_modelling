"""
Dynamics comparison for SEIRS v9 (single-strain) under two intervention setups.

Scenarios:
1) No drug (m_r=1.0, theta=0.0)
2) Baseline drug effect (m_r=0.75, theta=0.3)

The script prints core metrics and saves a publication-style figure
showing infectious dynamics over time.
"""

# %% Imports
import os
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from Models.SEIRS_Models import SEIRS_model_v9_singlestrain
from Models import params as model_params


# %% Shared time vector
T_MAX = getattr(model_params, "t_max", 365)
T_STEPS = int(getattr(model_params, "t_steps", 365))
T = np.linspace(0, T_MAX, T_STEPS)


# %% Initial conditions (S, El, Indl, Idl, Rl), normalized
S0 = getattr(model_params, "S", 10000)
El0 = getattr(model_params, "El", 0)
Indl0 = getattr(model_params, "Indl", 5)
Idl0 = getattr(model_params, "Idl", 0)
Rl0 = getattr(model_params, "Rl", 0)

init = np.array([S0, El0, Indl0, Idl0, Rl0], dtype=float)
init = init / init.sum()


# %% Baseline parameters for v9 single-strain
# (c_low, r_low, m_r_drug, birth_rate, death_rate, delta, sigma, tau, theta)
c_low = getattr(model_params, "contact_rate", 10.0)
r_low = getattr(
    model_params,
    "transmission_probability_low",
    getattr(model_params, "transmission_probability", 0.025),
)
m_r_baseline = getattr(model_params, "drug_transmission_multiplier", 0.75)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
delta = getattr(model_params, "delta", 1 / 120)
sigma = getattr(model_params, "sigma", 1 / 5)
tau = getattr(model_params, "tau", 1 / 3)
theta_baseline = getattr(model_params, "theta", 0.3)


def pack_params(m_r_drug, theta):
    """Build parameter tuple with the v9 single-strain expected order."""
    return (
        c_low,
        r_low,
        m_r_drug,
        birth_rate,
        death_rate,
        delta,
        sigma,
        tau,
        theta,
    )


def run_scenario(name, params_tuple):
    """Run ODE and return trajectories and summary metrics."""
    sol = odeint(SEIRS_model_v9_singlestrain, init, T, args=(params_tuple,))
    S, El, Indl, Idl, Rl = sol.T

    inf_total = Indl + Idl

    peak_idx = int(np.argmax(inf_total))
    metrics = {
        "scenario": name,
        "peak_infectious": float(np.max(inf_total)),
        "time_of_peak": float(T[peak_idx]),
        "final_susceptible": float(S[-1]),
        "attack_rate": float(1.0 - S[-1]),
    }

    return {
        "S": S,
        "inf_total": inf_total,
        "metrics": metrics,
    }


# %% Define the two requested scenarios
scenarios = {
    "No drug": pack_params(m_r_drug=1.0, theta=0.0),
    "Baseline drug": pack_params(m_r_drug=m_r_baseline, theta=theta_baseline),
}


# %% Run all scenarios
results = {name: run_scenario(name, p) for name, p in scenarios.items()}


# %% Print metrics
print("=== v9 Single-Strain Dynamics Comparison ===")
for name, out in results.items():
    m = out["metrics"]
    print(f"\n{name}")
    print(f"  Peak infectious: {m['peak_infectious']:.6f} at day {m['time_of_peak']:.2f}")
    print(f"  Final susceptible: {m['final_susceptible']:.6f}")
    print(f"  Attack rate: {m['attack_rate']:.6f}")


# %% Plot dynamics (single strain, two scenarios)
from matplotlib.lines import Line2D

# Scientific style settings
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.3,
        "savefig.bbox": "tight",
    }
)

scenario_styles = {
    "No drug": {"color": "#1b9e77"},
    "Baseline drug": {"color": "#d95f02"},
}

fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)

# Plot infectious dynamics for each scenario
for name, out in results.items():
    color = scenario_styles[name]["color"]
    ax.plot(T, out["inf_total"], color=color, lw=2.5, label=name)

ax.set_xlabel("Time (days)")
ax.set_ylabel("Infectious proportion")
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper right")

out_base = os.path.join(
    os.path.dirname(__file__),
    "../../Figures/Model_v9_exploration/v9_single_strain_dynamics_publication",
)

plt.savefig(out_base + ".png", dpi=700)  # high-res raster
plt.savefig(out_base + ".svg")           # vector for publication

print(f"\nSaved figure to {os.path.realpath(out_base + '.png')}")
print(f"Saved vector figure to {os.path.realpath(out_base + '.svg')}")