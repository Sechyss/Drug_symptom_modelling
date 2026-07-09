"""
Figure 1 for SEIRS v9 (two-strain): final epidemic size under four intervention setups.

Scenarios:
1) No drug
2) Drug A: transmission reduction only
3) Drug B: contact restoration only
4) Drug C: combined transmission reduction + contact restoration

The script computes final epidemic size as attack rate = 1 - S(T_end),
then saves a publication-style grouped bar figure named Fig1.
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
init_sum = init.sum()
if init_sum <= 0 or not np.isfinite(init_sum):
    raise ValueError("Initial conditions must have positive finite sum.")
init = init / init_sum


# %% Baseline parameters for v9 two-strain
# (c_low, r_low, phi_t, restoration_efficiency, m_r_drug,
#  kappa_base, kappa_scale, sigma, tau, theta)
c_low = getattr(model_params, "contact_rate", 10.0)
r_low = getattr(
    model_params,
    "transmission_probability_low",
    getattr(model_params, "transmission_probability", 0.025),
)
phi_t = getattr(model_params, "phi_transmission", 1.5)
restoration_baseline = getattr(model_params, "drug_contact_restore", 0.5)
m_r_baseline = getattr(model_params, "drug_transmission_multiplier", 0.75)
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
        kappa_base,
        kappa_scale,
        sigma,
        tau,
        theta,
    )


def run_scenario(name, params_tuple):
    """Run ODE and return summary metrics for final epidemic size."""
    sol = odeint(SEIRS_model_v9, init, T, args=(params_tuple,))
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

    inf_high = Indh + Idh
    inf_low = Indl + Idl
    inf_total = inf_high + inf_low

    final_size_high = Eh[-1] + Indh[-1] + Idh[-1] + Rh[-1]
    final_size_low = El[-1] + Indl[-1] + Idl[-1] + Rl[-1]

    peak_idx = int(np.argmax(inf_total))
    metrics = {
        "scenario": name,
        "final_susceptible": float(S[-1]),
        "final_epidemic_size_high": float(final_size_high),
        "final_epidemic_size_low": float(final_size_low),
        "final_epidemic_size": float(final_size_high + final_size_low),
        "peak_infectious_total": float(np.max(inf_total)),
        "time_of_peak": float(T[peak_idx]),
    }

    return metrics


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
results = [run_scenario(name, p) for name, p in scenarios.items()]


# %% Print metrics
print("=== v9 Two-Strain Final Epidemic Size (Figure 1) ===")
for m in results:
    print(f"\n{m['scenario']}")
    print(f"  Final susceptible: {m['final_susceptible']:.6f}")
    print(f"  Final epidemic size, high virulence: {m['final_epidemic_size_high']:.6f}")
    print(f"  Final epidemic size, low virulence:  {m['final_epidemic_size_low']:.6f}")
    print(f"  Final epidemic size (total):         {m['final_epidemic_size']:.6f}")
    print(f"  Peak infectious (total): {m['peak_infectious_total']:.6f} at day {m['time_of_peak']:.2f}")


# %% Plot Figure 1: stacked final epidemic size (high + low virulence)
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
        "savefig.bbox": "tight",
    }
)

scenario_names = [m["scenario"] for m in results]
final_epidemic_size_high = np.array([m["final_epidemic_size_high"] for m in results])
final_epidemic_size_low = np.array([m["final_epidemic_size_low"] for m in results])
final_epidemic_size_total = np.array([m["final_epidemic_size"] for m in results])

x = np.arange(len(scenario_names))
width = 0.58

fig, ax = plt.subplots(figsize=(10.2, 6.8), constrained_layout=True)

bars_h = ax.bar(
    x,
    final_epidemic_size_high,
    width,
    color="#d95f02",
    alpha=0.9,
    label="High virulence contribution",
)
bars_l = ax.bar(
    x,
    final_epidemic_size_low,
    width,
    bottom=final_epidemic_size_high,
    color="#1b9e77",
    alpha=0.9,
    label="Low virulence contribution",
)

ax.set_xticks(x)
ax.set_xticklabels(scenario_names)
ax.set_ylabel("Population proportion")
ax.set_ylim(0, 1.0)
ax.grid(True, axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, b in enumerate(bars_h):
    h = b.get_height()
    if h > 0.02:
        ax.text(
            b.get_x() + b.get_width() / 2,
            h / 2,
            f"{h:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )

for i, b in enumerate(bars_l):
    h = b.get_height()
    base = final_epidemic_size_high[i]
    if h > 0.02:
        ax.text(
            b.get_x() + b.get_width() / 2,
            base + h / 2,
            f"{h:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
        )

for i, total in enumerate(final_epidemic_size_total):
    ax.text(
        x[i],
        total + 0.012,
        f"Total: {total:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        weight="bold",
    )

ax.legend(frameon=False, loc="upper right")

out_base = os.path.join(
    os.path.dirname(__file__),
    "../../Figures/Model_v9_exploration/Fig1",
)

plt.savefig(out_base + ".png", dpi=700)
plt.savefig(out_base + ".svg")

print(f"\nSaved figure to {os.path.realpath(out_base + '.png')}")
print(f"Saved vector figure to {os.path.realpath(out_base + '.svg')}")
