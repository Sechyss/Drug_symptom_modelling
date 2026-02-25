import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Workspace paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Models.SEIRS_Models import SEIRS_model_v7
from Models import params as P


# -------------------------
# Configuration
# -------------------------
METRIC = "peak_I"          # "peak_I", "peak_I_high", "peak_I_low", or "S_end"
REL_PERTURB = 0.20         # +/- 20%
FIG_DPI = 600
STRAIN_COMPARISON = True   # If True, generate separate plots for high/low strains

# Parameter keys for v7 tornado analysis
# Note: v7 calculates c_high dynamically from c_low and phi_t,
# so we don't include contact_rate_high here
PARAM_KEYS = [
    "contact_rate",
    "transmission_probability_low",
    "phi_transmission",
    "drug_contact_multiplier",
    "drug_transmission_multiplier",
    "drug_contact_restore",          # NEW in v7
    "birth_rate",
    "death_rate",
    "delta",
    "kappa_base",
    "kappa_scale",
    "phi_recover",
    "sigma",
    "tau",
    "theta",
]


# -------------------------
# Helpers
# -------------------------
def time_grid():
    return np.linspace(0.0, float(P.t_max), int(P.t_steps))


def initial_state_normalized():
    # v7 state: [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
    y0 = np.array(
        [
            float(P.S),
            float(P.Eh),
            float(P.Indh),
            float(P.Idh),
            float(P.Rh),
            float(P.El),
            float(P.Indl),
            float(P.Idl),
            float(P.Rl),
        ],
        dtype=float,
    )
    N0 = float(np.sum(y0))
    if not np.isfinite(N0) or N0 <= 0:
        raise ValueError("Initial total population must be positive.")
    y0 = y0 / N0
    return y0, 1.0  # return N0=1.0 after normalization


def get_base_param_dict():
    # Map script keys -> values in params.py
    return {
        "contact_rate": float(P.contact_rate),
        "transmission_probability_low": float(getattr(P, "transmission_probability_low", P.transmission_probability)),
        "phi_transmission": float(P.phi_transmission),
        "drug_contact_multiplier": float(P.drug_contact_multiplier),
        "drug_transmission_multiplier": float(P.drug_transmission_multiplier),
        "drug_contact_restore": float(P.drug_contact_restore),
        "birth_rate": float(P.birth_rate),
        "death_rate": float(P.death_rate),
        "delta": float(P.delta),
        "kappa_base": float(P.kappa_base),
        "kappa_scale": float(P.kappa_scale),
        "phi_recover": float(P.phi_recover),
        "sigma": float(P.sigma),
        "tau": float(P.tau),
        "theta": float(P.theta),
    }


def clamp_param(key, value):
    # Basic safety clamps (keep model in valid region)
    if key in {"contact_rate", "sigma", "tau", "delta", "kappa_base"}:
        return max(float(value), 1e-6)  # Avoid exactly zero for rates
    if key in {"kappa_scale", "phi_transmission", "phi_recover"}:
        return max(float(value), 0.0)
    if key in {"drug_contact_multiplier", "drug_transmission_multiplier", "transmission_probability_low"}:
        return max(float(value), 0.0)
    if key in {"theta", "drug_contact_restore"}:
        return float(np.clip(value, 0.0, 1.0))
    if key in {"birth_rate", "death_rate"}:
        return max(float(value), 0.0)
    return float(value)


def build_param_tuple(d):
    # v7 expects (15):
    # (c_low, r_low, phi_t, m_c_drug, m_r_drug, drug_contact_restore,
    #  birth_rate, death_rate, delta, kappa_base, kappa_scale, 
    #  phi_recover, sigma, tau, theta)
    return (
        float(d["contact_rate"]),
        float(d["transmission_probability_low"]),
        float(d["phi_transmission"]),
        float(d["drug_contact_multiplier"]),
        float(d["drug_transmission_multiplier"]),
        float(d["drug_contact_restore"]),
        float(d["birth_rate"]),
        float(d["death_rate"]),
        float(d["delta"]),
        float(d["kappa_base"]),
        float(d["kappa_scale"]),
        float(d["phi_recover"]),
        float(d["sigma"]),
        float(d["tau"]),
        float(d["theta"]),
    )


def simulate_metric(param_dict, metric):
    t = time_grid()
    y0, N0 = initial_state_normalized()
    sol = odeint(SEIRS_model_v7, y0, t, args=(build_param_tuple(param_dict),))
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

    I_tot = Indh + Idh + Indl + Idl
    I_high = Indh + Idh
    I_low = Indl + Idl
    
    if metric == "peak_I":
        return float(np.max(I_tot) / N0)
    if metric == "peak_I_high":
        return float(np.max(I_high) / N0)
    if metric == "peak_I_low":
        return float(np.max(I_low) / N0)
    if metric == "S_end":
        return float(S[-1] / N0)

    raise ValueError(f"Unknown METRIC={metric!r}. Use 'peak_I', 'peak_I_high', 'peak_I_low', or 'S_end'.")


def perturb(value, rel, direction):
    # direction: -1 or +1
    return float(value * (1.0 + direction * rel))


def create_tornado_plot(rows, metric_name, output_suffix=""):
    """Create and save a tornado plot from sensitivity data."""
    labels = [r["param"] for r in rows]
    neg = np.array([r["delta_low"] for r in rows], dtype=float)
    pos = np.array([r["delta_high"] for r in rows], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(labels))))

    # Negative side
    ax.barh(y, neg, color="#4C72B0", alpha=0.85, label=f"-{int(REL_PERTURB*100)}%")
    # Positive side
    ax.barh(y, pos, color="#DD8452", alpha=0.85, label=f"+{int(REL_PERTURB*100)}%")

    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    # Set metric label
    if "high" in metric_name.lower():
        metric_label = r"$\Delta$ Peak $I_{high}/N_0$"
        title_suffix = " (High-virulence strain)"
    elif "low" in metric_name.lower():
        metric_label = r"$\Delta$ Peak $I_{low}/N_0$"
        title_suffix = " (Low-virulence strain)"
    elif "S_end" in metric_name:
        metric_label = r"$\Delta\, S_{end}/N_0$"
        title_suffix = ""
    else:
        metric_label = r"$\Delta$ Peak $(I/N_0)$"
        title_suffix = " (Total infections)"
    
    ax.set_xlabel(f"{metric_label} (vs baseline)")
    ax.set_title(f"Tornado sensitivity (SEIRS_model_v7){title_suffix}")

    ax.legend(loc="lower right", frameon=True)

    # Tight x-limits around data
    xmax = float(np.max(np.abs(np.concatenate([neg, pos, np.array([0.0])]))))
    if xmax > 0:
        ax.set_xlim(-1.05 * xmax, 1.05 * xmax)

    plt.tight_layout()

    # Save figure
    out_dir = os.path.join(ROOT_DIR, "Figures")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"Tornado_v7_{metric_name}{output_suffix}.png")
    plt.savefig(fig_path, dpi=FIG_DPI)
    print(f"Saved: {fig_path}")
    plt.close(fig)
    
    return fig_path


def save_csv_results(rows, metric_name, output_suffix=""):
    """Save tornado results to CSV."""
    csv_dir = os.path.join(ROOT_DIR, "Outputs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"Tornado_v7_{metric_name}{output_suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"Saved: {csv_path}")
    return csv_path


def print_summary(rows, metric_name):
    """Print summary of top influential parameters."""
    print("\n" + "="*60)
    print(f"PARAMETER SENSITIVITY SUMMARY: {metric_name}")
    print("="*60)
    for i, r in enumerate(rows[:10], 1):  # Top 10 most influential
        print(f"{i:2d}. {r['param']:30s} | max Δ = {r['max_abs_delta']:+.6f}")
    print("="*60)


def run_tornado_analysis(param_keys, base_params, metric_name):
    """Run tornado analysis for a given metric."""
    baseline_val = simulate_metric(base_params, metric_name)
    print(f"\nBaseline {metric_name}: {baseline_val:.8f}")

    rows = []
    for key in param_keys:
        if key not in base_params:
            continue

        v0 = float(base_params[key])

        v_low = clamp_param(key, perturb(v0, REL_PERTURB, -1))
        v_high = clamp_param(key, perturb(v0, REL_PERTURB, +1))

        d_low = dict(base_params)
        d_high = dict(base_params)
        d_low[key] = v_low
        d_high[key] = v_high

        out_low = simulate_metric(d_low, metric_name)
        out_high = simulate_metric(d_high, metric_name)

        neg = out_low - baseline_val
        pos = out_high - baseline_val

        rows.append(
            {
                "param": key,
                "base": v0,
                "low": v_low,
                "high": v_high,
                "metric_base": baseline_val,
                "metric_low": out_low,
                "metric_high": out_high,
                "delta_low": neg,
                "delta_high": pos,
                "max_abs_delta": max(abs(neg), abs(pos)),
            }
        )

    # Sort by influence (largest first)
    rows.sort(key=lambda r: r["max_abs_delta"], reverse=True)
    
    return rows, baseline_val


# -------------------------
# Run baseline + tornado
# -------------------------
base = get_base_param_dict()
base = {k: clamp_param(k, v) for k, v in base.items()}

print("="*60)
print("TORNADO SENSITIVITY ANALYSIS - SEIRS Model v7")
print("="*60)

# Main analysis for specified METRIC
rows, baseline_val = run_tornado_analysis(PARAM_KEYS, base, METRIC)
create_tornado_plot(rows, METRIC)
save_csv_results(rows, METRIC)
print_summary(rows, METRIC)

# If strain comparison requested, analyze each strain separately
if STRAIN_COMPARISON and METRIC == "peak_I":
    print("\n" + "="*60)
    print("STRAIN-SPECIFIC ANALYSIS")
    print("="*60)
    
    # High-virulence strain
    print("\n--- HIGH-VIRULENCE STRAIN ---")
    rows_high, baseline_high = run_tornado_analysis(PARAM_KEYS, base, "peak_I_high")
    create_tornado_plot(rows_high, "peak_I_high", "_strain_comparison")
    save_csv_results(rows_high, "peak_I_high", "_strain_comparison")
    print_summary(rows_high, "peak_I_high")
    
    # Low-virulence strain
    print("\n--- LOW-VIRULENCE STRAIN ---")
    rows_low, baseline_low = run_tornado_analysis(PARAM_KEYS, base, "peak_I_low")
    create_tornado_plot(rows_low, "peak_I_low", "_strain_comparison")
    save_csv_results(rows_low, "peak_I_low", "_strain_comparison")
    print_summary(rows_low, "peak_I_low")
    
    # Create comparison plot
    print("\n--- CREATING COMPARISON FIGURE ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(5, 0.35 * len(PARAM_KEYS))))
    
    # High strain (left panel)
    labels_high = [r["param"] for r in rows_high]
    neg_high = np.array([r["delta_low"] for r in rows_high], dtype=float)
    pos_high = np.array([r["delta_high"] for r in rows_high], dtype=float)
    y_high = np.arange(len(labels_high))
    
    ax1.barh(y_high, neg_high, color="#4C72B0", alpha=0.85, label=f"-{int(REL_PERTURB*100)}%")
    ax1.barh(y_high, pos_high, color="#DD8452", alpha=0.85, label=f"+{int(REL_PERTURB*100)}%")
    ax1.axvline(0.0, color="k", lw=1)
    ax1.set_yticks(y_high)
    ax1.set_yticklabels(labels_high)
    ax1.invert_yaxis()
    ax1.set_xlabel(r"$\Delta$ Peak $I_{high}/N_0$ (vs baseline)")
    ax1.set_title("High-virulence strain")
    ax1.legend(loc="lower right", frameon=True)
    xmax_high = float(np.max(np.abs(np.concatenate([neg_high, pos_high, np.array([0.0])]))))
    if xmax_high > 0:
        ax1.set_xlim(-1.05 * xmax_high, 1.05 * xmax_high)
    
    # Low strain (right panel)
    labels_low = [r["param"] for r in rows_low]
    neg_low = np.array([r["delta_low"] for r in rows_low], dtype=float)
    pos_low = np.array([r["delta_high"] for r in rows_low], dtype=float)
    y_low = np.arange(len(labels_low))
    
    ax2.barh(y_low, neg_low, color="#4C72B0", alpha=0.85, label=f"-{int(REL_PERTURB*100)}%")
    ax2.barh(y_low, pos_low, color="#DD8452", alpha=0.85, label=f"+{int(REL_PERTURB*100)}%")
    ax2.axvline(0.0, color="k", lw=1)
    ax2.set_yticks(y_low)
    ax2.set_yticklabels(labels_low)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"$\Delta$ Peak $I_{low}/N_0$ (vs baseline)")
    ax2.set_title("Low-virulence strain")
    ax2.legend(loc="lower right", frameon=True)
    xmax_low = float(np.max(np.abs(np.concatenate([neg_low, pos_low, np.array([0.0])]))))
    if xmax_low > 0:
        ax2.set_xlim(-1.05 * xmax_low, 1.05 * xmax_low)
    
    fig.suptitle("Tornado Sensitivity Comparison (SEIRS_model_v7)", fontsize=14, y=0.995)
    plt.tight_layout()
    
    comparison_path = os.path.join(ROOT_DIR, "Figures", "Tornado_v7_strain_comparison.png")
    plt.savefig(comparison_path, dpi=FIG_DPI)
    print(f"Saved: {comparison_path}")
    plt.close(fig)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
