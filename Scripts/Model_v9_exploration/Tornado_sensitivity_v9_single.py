import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Workspace paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Models.SEIRS_Models import SEIRS_model_v9_singlestrain
from Models import params as P


# -------------------------
# Configuration
# -------------------------
METRIC = "peak_I"   # "peak_I" or "S_end"
REL_PERTURB = 0.20   # +/- 20%
FIG_DPI = 600

# Parameter keys for v9 single-strain tornado analysis
# v9 single expects:
# (c_low, r_low, m_r_drug, birth_rate, death_rate, delta, sigma, tau, theta)
PARAM_KEYS = [
    "contact_rate",
    "transmission_probability_low",
    "drug_transmission_multiplier",
    "birth_rate",
    "death_rate",
    "delta",
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
    # v9 single-strain state: [S, El, Indl, Idl, Rl]
    y0 = np.array(
        [
            float(P.S),
            float(P.El),
            float(P.Indl),
            float(P.Idl),
            float(P.Rl),
        ],
        dtype=float,
    )
    n0 = float(np.sum(y0))
    if not np.isfinite(n0) or n0 <= 0:
        raise ValueError("Initial total population must be positive.")
    y0 = y0 / n0
    return y0, 1.0  # return N0=1.0 after normalization


def get_base_param_dict():
    # Map script keys -> values in params.py
    return {
        "contact_rate": float(getattr(P, "contact_rate", 10.0)),
        "transmission_probability_low": float(
            getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
        ),
        "drug_transmission_multiplier": float(getattr(P, "drug_transmission_multiplier", 0.75)),
        "birth_rate": float(getattr(P, "birth_rate", 0.0)),
        "death_rate": float(getattr(P, "death_rate", 0.0)),
        "delta": float(getattr(P, "delta", 1 / 120)),
        "sigma": float(getattr(P, "sigma", 1 / 5)),
        "tau": float(getattr(P, "tau", 1 / 3)),
        "theta": float(getattr(P, "theta", 0.3)),
    }


def clamp_param(key, value):
    # Basic safety clamps (keep model in valid region)
    if key in {"contact_rate", "sigma", "tau", "delta"}:
        return max(float(value), 1e-6)  # Avoid exactly zero for rates
    if key in {"drug_transmission_multiplier", "transmission_probability_low"}:
        return max(float(value), 0.0)
    if key in {"theta"}:
        return float(np.clip(value, 0.0, 1.0))
    if key in {"birth_rate", "death_rate"}:
        return max(float(value), 0.0)
    return float(value)


def build_param_tuple(d):
    # v9 single expects (9):
    # (c_low, r_low, m_r_drug, birth_rate, death_rate, delta, sigma, tau, theta)
    return (
        float(d["contact_rate"]),
        float(d["transmission_probability_low"]),
        float(d["drug_transmission_multiplier"]),
        float(d["birth_rate"]),
        float(d["death_rate"]),
        float(d["delta"]),
        float(d["sigma"]),
        float(d["tau"]),
        float(d["theta"]),
    )


def simulate_metric(param_dict, metric):
    t = time_grid()
    y0, n0 = initial_state_normalized()
    sol = odeint(SEIRS_model_v9_singlestrain, y0, t, args=(build_param_tuple(param_dict),))
    s, el, indl, idl, rl = sol.T

    i_tot = indl + idl

    if metric == "peak_I":
        return float(np.max(i_tot) / n0)
    if metric == "S_end":
        return float(s[-1] / n0)

    raise ValueError(f"Unknown METRIC={metric!r}. Use 'peak_I' or 'S_end'.")


def perturb(value, rel, direction):
    # direction: -1 or +1
    return float(value * (1.0 + direction * rel))


def create_tornado_plot(rows, metric_name):
    labels = [r["param"] for r in rows]
    neg = np.array([r["delta_low"] for r in rows], dtype=float)
    pos = np.array([r["delta_high"] for r in rows], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(labels))))

    ax.barh(y, neg, color="#4C72B0", alpha=0.85, label=f"-{int(REL_PERTURB*100)}%")
    ax.barh(y, pos, color="#DD8452", alpha=0.85, label=f"+{int(REL_PERTURB*100)}%")

    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    if metric_name == "S_end":
        metric_label = r"$\Delta\, S_{end}/N_0$"
    else:
        metric_label = r"$\Delta$ Peak $(I/N_0)$"

    ax.set_xlabel(f"{metric_label} (vs baseline)")
    ax.set_title("Tornado sensitivity (SEIRS_model_v9_singlestrain)")
    ax.legend(loc="lower right", frameon=True)

    # Tight x-limits around data
    xmax = float(np.max(np.abs(np.concatenate([neg, pos, np.array([0.0])]))))
    if xmax > 0:
        ax.set_xlim(-1.05 * xmax, 1.05 * xmax)

    plt.tight_layout()

    out_dir = os.path.join(ROOT_DIR, "Figures", "Model_v9_exploration")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"Tornado_v9_single_{metric_name}.svg")
    plt.savefig(fig_path, dpi=FIG_DPI)
    print(f"Saved: {fig_path}")
    plt.close(fig)

    return fig_path


def save_csv_results(rows, metric_name):
    csv_dir = os.path.join(ROOT_DIR, "Outputs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"Tornado_v9_single_{metric_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"Saved: {csv_path}")
    return csv_path


def print_summary(rows, metric_name):
    print("\n" + "=" * 60)
    print(f"PARAMETER SENSITIVITY SUMMARY: {metric_name}")
    print("=" * 60)
    for i, r in enumerate(rows[:10], 1):
        print(f"{i:2d}. {r['param']:30s} | max Δ = {r['max_abs_delta']:+.6f}")
    print("=" * 60)


def run_tornado_analysis(param_keys, base_params, metric_name):
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

    rows.sort(key=lambda r: r["max_abs_delta"], reverse=True)
    return rows, baseline_val


# -------------------------
# Run baseline + tornado
# -------------------------
base = get_base_param_dict()
base = {k: clamp_param(k, v) for k, v in base.items()}

print("=" * 60)
print("TORNADO SENSITIVITY ANALYSIS - SEIRS Model v9 Single Strain")
print("=" * 60)

rows, baseline_val = run_tornado_analysis(PARAM_KEYS, base, METRIC)
create_tornado_plot(rows, METRIC)
save_csv_results(rows, METRIC)
print_summary(rows, METRIC)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
