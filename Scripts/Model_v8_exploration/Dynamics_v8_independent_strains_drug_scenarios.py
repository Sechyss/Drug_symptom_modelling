"""
Independent-strain dynamics for SEIRS v8 (two-strain model, no competition).

This script uses SEIRS_model_v8 but runs each strain independently by zeroing
initial conditions of the other strain:
- Low-only run: high-strain initial compartments set to zero
- High-only run: low-strain initial compartments set to zero

For each intervention scenario and each independent run, the script prints:
- Peak infectious and time of peak
- Epidemic size (defined here as final susceptible)
- Attack rate
- Strain-specific R0 and Reff
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
from Models.SEIRS_Models import SEIRS_model_v8
from Models import params as model_params


# %% Shared time vector
T_MAX = getattr(model_params, "t_max", 365)
T_STEPS = int(getattr(model_params, "t_steps", 365))
T = np.linspace(0, T_MAX, T_STEPS)


# %% Baseline parameters for v8 two-strain
# (c_low, r_low, phi_t, restoration_efficiency, m_r_drug,
#  kappa_base, kappa_scale, sigma, tau, theta)
c_low = getattr(model_params, "contact_rate", 10.0)
r_low = getattr(
    model_params,
    "transmission_probability_low",
    getattr(model_params, "transmission_probability", 0.025),
)
phi_t = getattr(model_params, "phi_transmission", 1.5)
restoration_baseline = getattr(
    model_params,
    "restoration_efficiency",
    getattr(model_params, "drug_contact_restore", 0.5),
)
m_r_baseline = getattr(model_params, "drug_transmission_multiplier", 0.75)
kappa_base = getattr(model_params, "kappa_base", 1.0)
kappa_scale = getattr(model_params, "kappa_scale", 1.0)
sigma = getattr(model_params, "sigma", 1 / 5)
tau = getattr(model_params, "tau", 1 / 3)
theta_baseline = getattr(model_params, "theta", 0.3)


def pack_params(restoration_efficiency, m_r_drug, theta):
    """Build parameter tuple with the v8 expected order."""
    return (
        c_low,
        r_low,
        phi_t,
        restoration_efficiency,
        m_r_drug,
        kappa_base,
        kappa_scale,
        sigma,
        tau,
        theta,
    )


def compute_reproduction_numbers(params_tuple):
    """Return strain-specific R0 and Reff for v8 parameterization."""
    (
        c_low_i,
        r_low_i,
        phi_t_i,
        restoration_efficiency_i,
        m_r_drug_i,
        kappa_base_i,
        kappa_scale_i,
        sigma_i,
        _tau_i,
        theta_i,
    ) = params_tuple

    phi_safe = max(phi_t_i, 1e-8)
    c_high_untreated = c_low_i / phi_safe
    c_high_treated = c_high_untreated + restoration_efficiency_i * (c_low_i - c_high_untreated)

    beta_h_u = c_high_untreated * r_low_i * phi_t_i
    beta_h_t = c_high_treated * (r_low_i * m_r_drug_i) * phi_t_i
    beta_l_u = c_low_i * r_low_i
    beta_l_t = c_low_i * (r_low_i * m_r_drug_i)

    vir_excess_pos = max(0.0, phi_t_i - 1.0)
    kappa_high = kappa_base_i * (1 + kappa_scale_i * vir_excess_pos)
    kappa_low = kappa_base_i
    if theta_i > 0:
        kappa_high = min(kappa_high, 1.0 / theta_i)
        kappa_low = min(kappa_low, 1.0 / theta_i)

    theta_high = kappa_high * theta_i
    theta_low = kappa_low * theta_i

    sigma_safe = max(sigma_i, 1e-8)
    r0_high = beta_h_u / sigma_safe
    r0_low = beta_l_u / sigma_safe

    re_high = ((1.0 - theta_high) * beta_h_u + theta_high * beta_h_t) / sigma_safe
    re_low = ((1.0 - theta_low) * beta_l_u + theta_low * beta_l_t) / sigma_safe

    return {
        "R0_high": float(r0_high),
        "R0_low": float(r0_low),
        "Re_high": float(re_high),
        "Re_low": float(re_low),
    }


def build_initial_conditions(seed_strain):
    """Build normalized initial condition vector for a single seeded strain."""
    S0 = float(getattr(model_params, "S", 10000))

    Eh0 = float(getattr(model_params, "Eh", 0))
    Indh0 = float(getattr(model_params, "Indh", 0))
    Idh0 = float(getattr(model_params, "Idh", 0))
    Rh0 = float(getattr(model_params, "Rh", 0))

    El0 = float(getattr(model_params, "El", 0))
    Indl0 = float(getattr(model_params, "Indl", 5))
    Idl0 = float(getattr(model_params, "Idl", 0))
    Rl0 = float(getattr(model_params, "Rl", 0))

    if seed_strain == "low":
        low_seed = Indl0 if Indl0 > 0 else max(Indh0, 1.0)
        init = np.array([S0, 0.0, 0.0, 0.0, 0.0, El0, low_seed, Idl0, Rl0], dtype=float)
    elif seed_strain == "high":
        high_seed = Indh0 if Indh0 > 0 else max(Indl0, 1.0)
        init = np.array([S0, Eh0, high_seed, Idh0, Rh0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        raise ValueError("seed_strain must be 'low' or 'high'.")

    init_sum = init.sum()
    if init_sum <= 0 or not np.isfinite(init_sum):
        raise ValueError("Initial conditions must have positive finite sum.")

    return init / init_sum


def run_independent_scenario(seed_strain, params_tuple):
    """Run one independent-strain simulation using SEIRS_model_v8."""
    init = build_initial_conditions(seed_strain)
    sol = odeint(SEIRS_model_v8, init, T, args=(params_tuple,))
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

    inf_high = Indh + Idh
    inf_low = Indl + Idl

    if seed_strain == "high":
        inf = inf_high
    else:
        inf = inf_low

    peak_idx = int(np.argmax(inf))
    repro = compute_reproduction_numbers(params_tuple)

    metrics = {
        "peak_infectious": float(np.max(inf)),
        "time_of_peak": float(T[peak_idx]),
        "epidemic_size": float(S[-1]),  # Defined by user: final susceptible
        "attack_rate": float(1.0 - S[-1]),
    }

    if seed_strain == "high":
        metrics["R0"] = repro["R0_high"]
        metrics["Reff"] = repro["Re_high"]
    else:
        metrics["R0"] = repro["R0_low"]
        metrics["Reff"] = repro["Re_low"]

    return {
        "S": S,
        "metrics": metrics,
    }


# %% Define scenarios
scenarios = {
    "No drug": pack_params(restoration_efficiency=0.0, m_r_drug=1.0, theta=0.0),
    "Drug A": pack_params(
        restoration_efficiency=0.0,
        m_r_drug=m_r_baseline,
        theta=theta_baseline,
    ),
    "Drug B": pack_params(
        restoration_efficiency=restoration_baseline,
        m_r_drug=1.0,
        theta=theta_baseline,
    ),
    "Drug C": pack_params(
        restoration_efficiency=restoration_baseline,
        m_r_drug=m_r_baseline,
        theta=theta_baseline,
    ),
}


# %% Run and print metrics
print("=== v8 Independent-Strain Dynamics (No Competition) ===")
results = {}
for scenario_name, p in scenarios.items():
    low_metrics = run_independent_scenario("low", p)
    high_metrics = run_independent_scenario("high", p)
    results[scenario_name] = {"low": low_metrics, "high": high_metrics}

    low_m = low_metrics["metrics"]
    high_m = high_metrics["metrics"]

    print(f"\n{scenario_name}")

    print("  Low strain only")
    print(
        f"    Peak infectious: {low_m['peak_infectious']:.6f} "
        f"at day {low_m['time_of_peak']:.2f}"
    )
    print(f"    Epidemic size (final susceptible): {low_m['epidemic_size']:.6f}")
    print(f"    Attack rate: {low_m['attack_rate']:.6f}")
    print(f"    R0: {low_m['R0']:.4f}")
    print(f"    Reff: {low_m['Reff']:.4f}")

    print("  High strain only")
    print(
        f"    Peak infectious: {high_m['peak_infectious']:.6f} "
        f"at day {high_m['time_of_peak']:.2f}"
    )
    print(f"    Epidemic size (final susceptible): {high_m['epidemic_size']:.6f}")
    print(f"    Attack rate: {high_m['attack_rate']:.6f}")
    print(f"    R0: {high_m['R0']:.4f}")
    print(f"    Reff: {high_m['Reff']:.4f}")


# %% Plot susceptible trajectories for independent runs
from matplotlib.lines import Line2D

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
    "Drug A": {"color": "#d95f02"},
    "Drug B": {"color": "#7570b3"},
    "Drug C": {"color": "#e7298a"},
}

fig, (ax_low, ax_high) = plt.subplots(
    1, 2, figsize=(12.2, 5.6), sharey=True, constrained_layout=True
)

for scenario_name, out in results.items():
    color = scenario_styles[scenario_name]["color"]
    ax_low.plot(T, out["low"]["S"], color=color, ls="-")
    ax_high.plot(T, out["high"]["S"], color=color, ls="-")

ax_low.set_title("Low strain only")
ax_high.set_title("High strain only")
ax_low.set_xlabel("Time (days)")
ax_high.set_xlabel("Time (days)")
ax_low.set_ylabel("Susceptible proportion")

ax_low.text(0.01, 0.98, "A", transform=ax_low.transAxes, ha="left", va="top", fontweight="bold", fontsize=14)
ax_high.text(0.01, 0.98, "B", transform=ax_high.transAxes, ha="left", va="top", fontweight="bold", fontsize=14)

for ax in (ax_low, ax_high):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

scenario_handles = [
    Line2D([0], [0], color=scenario_styles[name]["color"], lw=2.5, label=name)
    for name in scenarios
]
ax_high.legend(
    handles=scenario_handles,
    title="Scenario",
    loc="center right",
    frameon=False,
    ncol=1,
)

out_base = os.path.join(
    os.path.dirname(__file__),
    "../../Figures/Model_v8_exploration/v8_independent_strains_susceptible_dynamics",
)

plt.savefig(out_base + ".png", dpi=700)
plt.savefig(out_base + ".svg")

print(f"\nSaved figure to {os.path.realpath(out_base + '.png')}")
print(f"Saved vector figure to {os.path.realpath(out_base + '.svg')}")
