"""
Dynamics comparison for SEIRS v8 (two-strain) under four intervention setups.

Scenarios:
1) No drug
2) Drug A: transmission reduction only
3) Drug B: contact restoration only
4) Drug C: combined transmission reduction + contact restoration

The script prints core metrics and saves a publication-style figure with one
panel per scenario, comparing high- and low-virulence trajectories within each.
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
init_sum = init.sum()
if init_sum <= 0 or not np.isfinite(init_sum):
    raise ValueError("Initial conditions must have positive finite sum.")
init = init / init_sum


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
    """Return strain-specific and dominant R0/Re for the v8 parameterization.

    R0: no-drug baseline (untreated transmission only).
    Re: effective reproduction number after drug effects are included.
    """
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
    r0_dominant = max(r0_high, r0_low)

    re_high = ((1.0 - theta_high) * beta_h_u + theta_high * beta_h_t) / sigma_safe
    re_low = ((1.0 - theta_low) * beta_l_u + theta_low * beta_l_t) / sigma_safe
    re_dominant = max(re_high, re_low)

    return {
        "R0_high": float(r0_high),
        "R0_low": float(r0_low),
        "R0_dominant": float(r0_dominant),
        "Re_high": float(re_high),
        "Re_low": float(re_low),
        "Re_dominant": float(re_dominant),
    }


def run_scenario(name, params_tuple):
    """Run ODE and return trajectories and summary metrics."""
    sol = odeint(SEIRS_model_v8, init, T, args=(params_tuple,))
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

    inf_high = Indh + Idh
    inf_low = Indl + Idl
    inf_total = inf_high + inf_low

    peak_idx = int(np.argmax(inf_total))
    repro = compute_reproduction_numbers(params_tuple)
    metrics = {
        "scenario": name,
        "peak_infectious_total": float(np.max(inf_total)),
        "time_of_peak": float(T[peak_idx]),
        "final_susceptible": float(S[-1]),
        "attack_rate": float(1.0 - S[-1]),
        **repro,
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


# %% Run all scenarios
results = {name: run_scenario(name, p) for name, p in scenarios.items()}


# %% Print metrics
print("=== v8 Two-Strain Dynamics Comparison ===")
for name, out in results.items():
    m = out["metrics"]
    print(f"\n{name}")
    print(f"  Peak infectious (total): {m['peak_infectious_total']:.6f} at day {m['time_of_peak']:.2f}")
    print(f"  Epidemic size (final susceptible): {m['final_susceptible']:.6f}")
    print(f"  Attack rate: {m['attack_rate']:.6f}")
    print(
        "  R0 (dominant / high / low): "
        f"{m['R0_dominant']:.4f} / {m['R0_high']:.4f} / {m['R0_low']:.4f}"
    )
    print(
        "  Reff (dominant / high / low, with drug): "
        f"{m['Re_dominant']:.4f} / {m['Re_high']:.4f} / {m['Re_low']:.4f}"
    )


# %% Plot dynamics (four panels, one per scenario)
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
    "Drug A": {"color": "#d95f02"},
    "Drug B": {"color": "#7570b3"},
    "Drug C": {"color": "#e7298a"},
}

fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.4), sharex=True, sharey=True, constrained_layout=True)

panel_labels = ["A", "B", "C", "D"]
for ax, panel_label, (name, out) in zip(axes.flat, panel_labels, results.items()):
    color = scenario_styles[name]["color"]
    ax.plot(T, out["inf_high"], color=color, ls="-", label="High virulence")
    ax.plot(T, out["inf_low"], color=color, ls="--", label="Low virulence")
    ax.set_title(name)
    ax.text(
        0.01,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=14,
    )
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for ax in axes[1, :]:
    ax.set_xlabel("Time (days)")

for ax in axes[:, 0]:
    ax.set_ylabel("Infectious proportion")

scenario_handles = [
    Line2D([0], [0], color=scenario_styles[name]["color"], lw=2.5, label=name)
    for name in scenarios
]
strain_handles = [
    Line2D([0], [0], color="black", lw=2.5, ls="-", label="High virulence"),
    Line2D([0], [0], color="black", lw=2.5, ls="--", label="Low virulence"),
]

combined_handles = scenario_handles + strain_handles
fig.legend(
    handles=combined_handles,
    title="Scenario and strain",
    loc="outside right center",
    frameon=False,
)

out_base = os.path.join(
    os.path.dirname(__file__),
    "../../Figures/Model_v8_exploration/v8_two_strain_dynamics_combined_publication",
)

plt.savefig(out_base + ".png", dpi=700)  # high-res raster
plt.savefig(out_base + ".svg")  # vector for publication

print(f"\nSaved figure to {os.path.realpath(out_base + '.png')}")
print(f"Saved vector figure to {os.path.realpath(out_base + '.svg')}")
