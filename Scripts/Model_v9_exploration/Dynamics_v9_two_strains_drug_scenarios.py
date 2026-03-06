"""
Dynamics comparison for SEIRS v9 (two-strain) under four intervention setups.

Scenarios:
1) No drug
2) Baseline drug effect
3) No contact restoration (restoration=0) with baseline m_r
4) No transmission reduction (m_r=1) with baseline restoration

The script prints core metrics and saves a single publication-style figure
combining both strains with solid lines for high strain and dashed for low strain.
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
from Models.SEIRS_Models import SEIRS_model_v9
from Models import params as model_params


# %% Shared time vector
T_MAX = getattr(model_params, "t_max", 365)
T_STEPS = int(getattr(model_params, "t_steps", 365))
T = np.linspace(0, T_MAX, T_STEPS)


# %% Initial conditions (S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl), normalized
S0 = getattr(model_params, "S", 10000)
Eh0 = getattr(model_params, "Eh", 0)
Indh0 = getattr(model_params, "Indh", 0)
Idh0 = getattr(model_params, "Idh", 0)
Rh0 = getattr(model_params, "Rh", 0)
El0 = getattr(model_params, "El", 0)
Indl0 = getattr(model_params, "Indl", 5)
Idl0 = getattr(model_params, "Idl", 0)
Rl0 = getattr(model_params, "Rl", 0)

init = np.array([S0, Eh0, Indh0, Idh0, Rh0, El0, Indl0, Idl0, Rl0], dtype=float)
init = init / init.sum()


# %% Baseline parameters for v9 two-strain
# (c_low, r_low, phi_t, restoration_efficiency, m_r_drug,
#  birth_rate, death_rate, delta, kappa_base, kappa_scale, sigma, tau, theta)
c_low = getattr(model_params, "contact_rate", 10.0)
r_low = getattr(
    model_params,
    "transmission_probability_low",
    getattr(model_params, "transmission_probability", 0.025),
)
phi_t = getattr(model_params, "phi_transmission", 1.5)
restoration_baseline = getattr(model_params, "drug_contact_restore", 0.5)
m_r_baseline = getattr(model_params, "drug_transmission_multiplier", 0.75)
birth_rate = getattr(model_params, "birth_rate", 0.0)
death_rate = getattr(model_params, "death_rate", 0.0)
delta = getattr(model_params, "delta", 1 / 120)
kappa_base = getattr(model_params, "kappa_base", 1.0)
kappa_scale = getattr(model_params, "kappa_scale", 1.0)
sigma = getattr(model_params, "sigma", 1 / 5)
tau = getattr(model_params, "tau", 1 / 3)
theta_baseline = getattr(model_params, "theta", 0.3)


def pack_params(restoration_efficiency, m_r_drug, theta):
    """Build parameter tuple with the v9 expected order."""
    return (
        c_low,
        r_low,
        phi_t,
        restoration_efficiency,
        m_r_drug,
        birth_rate,
        death_rate,
        delta,
        kappa_base,
        kappa_scale,
        sigma,
        tau,
        theta,
    )


def run_scenario(name, params_tuple):
    """Run ODE and return trajectories and summary metrics."""
    sol = odeint(SEIRS_model_v9, init, T, args=(params_tuple,))
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

    inf_high = Indh + Idh
    inf_low = Indl + Idl
    inf_total = inf_high + inf_low

    peak_idx = int(np.argmax(inf_total))
    metrics = {
        "scenario": name,
        "peak_infectious_total": float(np.max(inf_total)),
        "time_of_peak": float(T[peak_idx]),
        "final_susceptible": float(S[-1]),
        "attack_rate": float(1.0 - S[-1]),
    }

    return {
        "S": S,
        "inf_high": inf_high,
        "inf_low": inf_low,
        "inf_total": inf_total,
        "metrics": metrics,
    }


# %% Define the four requested scenarios
scenarios = {
    "No drug": pack_params(restoration_efficiency=0.0, m_r_drug=1.0, theta=0.0),
    "Baseline drug": pack_params(
        restoration_efficiency=restoration_baseline,
        m_r_drug=m_r_baseline,
        theta=theta_baseline,
    ),
    "No restoration, baseline m_r": pack_params(
        restoration_efficiency=0.0,
        m_r_drug=m_r_baseline,
        theta=theta_baseline,
    ),
    "m_r = 1, baseline restoration": pack_params(
        restoration_efficiency=restoration_baseline,
        m_r_drug=1.0,
        theta=theta_baseline,
    ),
}


# %% Run all scenarios
results = {name: run_scenario(name, p) for name, p in scenarios.items()}


# %% Print metrics
print("=== v9 Two-Strain Dynamics Comparison ===")
for name, out in results.items():
    m = out["metrics"]
    print(f"\n{name}")
    print(f"  Peak infectious (total): {m['peak_infectious_total']:.6f} at day {m['time_of_peak']:.2f}")
    print(f"  Final susceptible: {m['final_susceptible']:.6f}")
    print(f"  Attack rate: {m['attack_rate']:.6f}")


# %% Plot dynamics (single combined panel, publication-style)
from matplotlib.lines import Line2D

# Scientific style settings (portable, no external font dependency)
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
    "No restoration, baseline m_r": {"color": "#7570b3"},
    "m_r = 1, baseline restoration": {"color": "#e7298a"},
}

fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)

# Plot both strains in one panel:
# high strain = solid, low strain = dashed
for name, out in results.items():
    color = scenario_styles[name]["color"]
    ax.plot(T, out["inf_high"], color=color, ls="-")
    ax.plot(T, out["inf_low"], color=color, ls="--")

ax.set_xlabel("Time (days)")
ax.set_ylabel("Infectious proportion")
ax.grid(True, alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend 1: scenario colors (moved to center right, single column)
scenario_handles = [
    Line2D([0], [0], color=scenario_styles[name]["color"], lw=2.5, label=name)
    for name in scenario_styles
]
leg1 = ax.legend(
    handles=scenario_handles,
    title="Scenario",
    loc="center right",
    frameon=False,
    ncol=1,
)
ax.add_artist(leg1)

# Legend 2: strain line styles
strain_handles = [
    Line2D([0], [0], color="black", lw=2.5, ls="-", label="High strain"),
    Line2D([0], [0], color="black", lw=2.5, ls="--", label="Low strain"),
]
ax.legend(
    handles=strain_handles,
    title="Strain",
    loc="upper right",
    frameon=False,
)

out_base = os.path.join(
    os.path.dirname(__file__),
    "../../Figures/Model_v9_exploration/v9_two_strain_dynamics_combined_publication",
)

plt.savefig(out_base + ".png", dpi=700)  # high-res raster
plt.savefig(out_base + ".svg")           # vector for publication

print(f"\nSaved figure to {os.path.realpath(out_base + '.png')}")
print(f"Saved vector figure to {os.path.realpath(out_base + '.svg')}")
