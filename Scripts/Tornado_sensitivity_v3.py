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

from Models.SEIRS_Models import SEIRS_model_v3
from Models import params as P


# -------------------------
# Configuration
# -------------------------
METRIC = "peak_I"          # "peak_I" or "S_end"
REL_PERTURB = 0.20         # +/- 20%
FIG_DPI = 600

# (Optional) restrict which params to include in tornado
PARAM_KEYS = [
    "contact_rate",
    "transmission_probability_low",
    "contact_rate_high",
    "phi_transmission",
    "drug_contact_multiplier",
    "drug_transmission_multiplier",
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
    # v3 state: [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
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
        "contact_rate_high": float(P.contact_rate_high),
        "phi_transmission": float(P.phi_transmission),
        "drug_contact_multiplier": float(P.drug_contact_multiplier),
        "drug_transmission_multiplier": float(P.drug_transmission_multiplier),
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
    if key in {"contact_rate", "contact_rate_high", "sigma", "tau", "delta", "kappa_base"}:
        return max(float(value), 0.0)
    if key in {"kappa_scale", "phi_transmission", "phi_recover"}:
        return max(float(value), 0.0)
    if key in {"drug_contact_multiplier", "drug_transmission_multiplier", "transmission_probability_low"}:
        return max(float(value), 0.0)
    if key == "theta":
        return float(np.clip(value, 0.0, 1.0))
    # birth_rate/death_rate can be 0; allow small negatives if desired, but clamp to >=0 by default
    if key in {"birth_rate", "death_rate"}:
        return max(float(value), 0.0)
    return float(value)


def build_param_tuple(d):
    # v3 expects (15):
    # (contact_rate_low, transmission_probability_low, contact_rate_high, phi_transmission,
    #  drug_contact_multiplier, drug_transmission_multiplier,
    #  birth_rate, death_rate, delta, kappa_base, kappa_scale, phi_recover, sigma, tau, theta)
    return (
        float(d["contact_rate"]),
        float(d["transmission_probability_low"]),
        float(d["contact_rate_high"]),
        float(d["phi_transmission"]),
        float(d["drug_contact_multiplier"]),
        float(d["drug_transmission_multiplier"]),
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
    sol = odeint(SEIRS_model_v3, y0, t, args=(build_param_tuple(param_dict),))
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = sol.T

    I_tot = Indh + Idh + Indl + Idl
    if metric == "peak_I":
        return float(np.max(I_tot) / N0)
    if metric == "S_end":
        return float(S[-1] / N0)

    raise ValueError(f"Unknown METRIC={metric!r}. Use 'peak_I' or 'S_end'.")


def perturb(value, rel, direction):
    # direction: -1 or +1
    return float(value * (1.0 + direction * rel))


# -------------------------
# Run baseline + tornado
# -------------------------
base = get_base_param_dict()
base = {k: clamp_param(k, v) for k, v in base.items()}

baseline_val = simulate_metric(base, METRIC)
print(f"Baseline {METRIC}: {baseline_val:.8f}")

rows = []
for key in PARAM_KEYS:
    if key not in base:
        continue

    v0 = float(base[key])

    v_low = clamp_param(key, perturb(v0, REL_PERTURB, -1))
    v_high = clamp_param(key, perturb(v0, REL_PERTURB, +1))

    d_low = dict(base)
    d_high = dict(base)
    d_low[key] = v_low
    d_high[key] = v_high

    out_low = simulate_metric(d_low, METRIC)
    out_high = simulate_metric(d_high, METRIC)

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

# -------------------------
# Plot tornado
# -------------------------
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

metric_label = r"$\Delta$ Peak $(I/N_0)$" if METRIC == "peak_I" else r"$\Delta\, S_{end}/N_0$"
ax.set_xlabel(f"{metric_label} (vs baseline)")
ax.set_title(f"Tornado sensitivity (SEIRS_model_v3) — metric: {metric_label}")

ax.legend(loc="lower right", frameon=True)

# Tight x-limits around data
xmax = float(np.max(np.abs(np.concatenate([neg, pos, np.array([0.0])]))))
if xmax > 0:
    ax.set_xlim(-1.05 * xmax, 1.05 * xmax)

plt.tight_layout()

# Save outputs
out_dir = os.path.join(ROOT_DIR, "Figures")
os.makedirs(out_dir, exist_ok=True)
fig_path = os.path.join(out_dir, f"Tornado_v3_{METRIC}.png")
plt.savefig(fig_path, dpi=FIG_DPI)
print(f"Saved: {fig_path}")

# Also save a CSV table for inspection
csv_dir = os.path.join(ROOT_DIR, "Outputs")
os.makedirs(csv_dir, exist_ok=True)
csv_path = os.path.join(csv_dir, f"Tornado_v3_{METRIC}.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        w.writeheader()
        w.writerows(rows)
print(f"Saved: {csv_path}")