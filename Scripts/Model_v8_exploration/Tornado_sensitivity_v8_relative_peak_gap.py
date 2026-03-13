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

from Models.SEIRS_Models import SEIRS_model_v8
from Models import params as P


# -------------------------
# Configuration
# -------------------------
REL_PERTURB = 0.20  # +/- 20%
FIG_DPI = 600
REL_DIFF_EPS = 1e-12

# v8 two-strain expects:
# (c_low, r_low, phi_t, restoration_efficiency, m_r_drug,
#  kappa_base, kappa_scale, sigma, tau, theta)
PARAM_KEYS = [
    "contact_rate",
    "transmission_probability_low",
    "phi_transmission",
    "restoration_efficiency",
    "drug_transmission_multiplier",
    "kappa_base",
    "kappa_scale",
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
    # v8 two-strain state: [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
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
    n0 = float(np.sum(y0))
    if not np.isfinite(n0) or n0 <= 0:
        raise ValueError("Initial total population must be positive.")
    y0 = y0 / n0
    return y0, 1.0  # return N0=1.0 after normalization


def get_base_param_dict():
    restoration_eff = getattr(P, "restoration_efficiency", getattr(P, "drug_contact_restore", 0.8))

    return {
        "contact_rate": float(getattr(P, "contact_rate", 10.0)),
        "transmission_probability_low": float(
            getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
        ),
        "phi_transmission": float(getattr(P, "phi_transmission", 1.5)),
        "restoration_efficiency": float(restoration_eff),
        "drug_transmission_multiplier": float(getattr(P, "drug_transmission_multiplier", 0.75)),
        "kappa_base": float(getattr(P, "kappa_base", 1.0)),
        "kappa_scale": float(getattr(P, "kappa_scale", 1.0)),
        "sigma": float(getattr(P, "sigma", 1 / 5)),
        "tau": float(getattr(P, "tau", 1 / 3)),
        "theta": float(getattr(P, "theta", 0.3)),
    }


def clamp_param(key, value):
    if key in {"contact_rate", "sigma", "tau", "delta", "kappa_base"}:
        return max(float(value), 1e-6)
    if key in {"kappa_scale", "phi_transmission"}:
        return max(float(value), 0.0)
    if key in {"drug_transmission_multiplier", "transmission_probability_low"}:
        return max(float(value), 0.0)
    if key in {"theta", "restoration_efficiency"}:
        return float(np.clip(value, 0.0, 1.0))
    return float(value)


def build_param_tuple(d):
    return (
        float(d["contact_rate"]),
        float(d["transmission_probability_low"]),
        float(d["phi_transmission"]),
        float(d["restoration_efficiency"]),
        float(d["drug_transmission_multiplier"]),
        float(d["kappa_base"]),
        float(d["kappa_scale"]),
        float(d["sigma"]),
        float(d["tau"]),
        float(d["theta"]),
    )


def simulate_peaks(param_dict):
    t = time_grid()
    y0, n0 = initial_state_normalized()
    sol = odeint(SEIRS_model_v8, y0, t, args=(build_param_tuple(param_dict),))
    _, _, indh, idh, _, _, indl, idl, _ = sol.T

    peak_high = float(np.max(indh + idh) / n0)
    peak_low = float(np.max(indl + idl) / n0)
    return peak_high, peak_low


def relative_peak_gap(peak_high, peak_low):
    denom = max(float(peak_low), REL_DIFF_EPS)
    return float((peak_high - peak_low) / denom)


def perturb(value, rel, direction):
    # direction: -1 or +1
    return float(value * (1.0 + direction * rel))


def run_tornado_analysis(param_keys, base_params):
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

        peak_h_low, peak_l_low = simulate_peaks(d_low)
        peak_h_high, peak_l_high = simulate_peaks(d_high)

        rel_low = relative_peak_gap(peak_h_low, peak_l_low)
        rel_high = relative_peak_gap(peak_h_high, peak_l_high)

        rows.append(
            {
                "param": key,
                "base": v0,
                "low": v_low,
                "high": v_high,
                "peak_high_low_perturb": peak_h_low,
                "peak_low_low_perturb": peak_l_low,
                "peak_high_high_perturb": peak_h_high,
                "peak_low_high_perturb": peak_l_high,
                "relative_gap_low_perturb": rel_low,
                "relative_gap_high_perturb": rel_high,
                "max_abs_relative_gap": max(abs(rel_low), abs(rel_high)),
            }
        )

    rows.sort(key=lambda r: r["max_abs_relative_gap"], reverse=True)
    return rows


def create_tornado_plot(rows, output_suffix=""):
    labels = [r["param"] for r in rows]

    # Keep tornado geometry: low perturbations on left, high perturbations on right.
    neg = -np.array([r["relative_gap_low_perturb"] for r in rows], dtype=float)
    pos = np.array([r["relative_gap_high_perturb"] for r in rows], dtype=float)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(labels))))

    ax.barh(y, neg, color="#4C72B0", alpha=0.85, label=f"-{int(REL_PERTURB*100)}%")
    ax.barh(y, pos, color="#DD8452", alpha=0.85, label=f"+{int(REL_PERTURB*100)}%")

    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel(r"Relative peak gap: $(I_{high}^{peak} - I_{low}^{peak}) / I_{low}^{peak}$")
    ax.set_title("Tornado sensitivity (SEIRS_model_v8): strain peak relative gap")
    ax.legend(loc="lower right", frameon=True)

    xmax = float(np.max(np.abs(np.concatenate([neg, pos, np.array([0.0])]))))
    if xmax > 0:
        ax.set_xlim(-1.05 * xmax, 1.05 * xmax)

    plt.tight_layout()

    out_dir = os.path.join(ROOT_DIR, "Figures", "Model_v8_exploration")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"Tornado_v8_relative_peak_gap{output_suffix}.svg")
    plt.savefig(fig_path, dpi=FIG_DPI)
    print(f"Saved: {fig_path}")
    plt.close(fig)

    return fig_path


def save_csv_results(rows, output_suffix=""):
    csv_dir = os.path.join(ROOT_DIR, "Outputs")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"Tornado_v8_relative_peak_gap{output_suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    print(f"Saved: {csv_path}")
    return csv_path


def print_summary(rows):
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY SUMMARY: RELATIVE PEAK GAP")
    print("=" * 70)
    for i, r in enumerate(rows[:10], 1):
        print(f"{i:2d}. {r['param']:30s} | max |gap| = {r['max_abs_relative_gap']:+.6f}")
    print("=" * 70)


# -------------------------
# Run
# -------------------------
base = get_base_param_dict()
base = {k: clamp_param(k, v) for k, v in base.items()}

print("=" * 70)
print("TORNADO SENSITIVITY - SEIRS Model v8 Two Strains")
print("Metric: (peak_I_high - peak_I_low) / peak_I_low")
print("=" * 70)

rows = run_tornado_analysis(PARAM_KEYS, base)
create_tornado_plot(rows)
save_csv_results(rows)
print_summary(rows)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)