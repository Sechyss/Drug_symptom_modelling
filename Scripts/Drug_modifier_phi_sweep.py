#!/usr/bin/env python3
"""3D parameter sweep: peak infections vs (mc, mr, phi_transmission).

Axes
- mc: drug_contact_multiplier
- mr: drug_transmission_multiplier
- phi_transmission: virulence/transmission multiplier

Outputs
- Figures/drug_modifier_phi_sweep/peak_infection_landscape.png

Run (from repo root)
    python Scripts/Drug_modifier_phi_sweep.py \
        --phi 1.0 1.2 1.5 1.8 2.2 \
        --mc 0.8 1.0 1.2 \
        --mr 0.5 0.7 1.0 \
        --days 365 --steps 365

Notes
- Uses SEIRS_model_v6, i.e. R0 varies naturally under the model's built-in
    virulence-contact trade-off (c_high decreases with phi).
- Produces 3D surface “landscapes” over (mc, mr), faceted by phi_transmission.
- Two rows: high-strain peak and low-strain peak.
"""

import os
import sys
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.integrate import odeint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
from matplotlib.colors import Normalize

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.SEIRS_Models import SEIRS_model_v6
from Models import params as P

COLS = ["S", "Eh", "Indh", "Idh", "Rh", "El", "Indl", "Idl", "Rl"]


def initial_conditions(normalize: bool = True) -> np.ndarray:
    y0 = np.array(
        [
            getattr(P, "S", 10000),
            getattr(P, "Eh", 0),
            getattr(P, "Indh", 5),
            getattr(P, "Idh", 0),
            getattr(P, "Rh", 0),
            getattr(P, "El", 0),
            getattr(P, "Indl", 5),
            getattr(P, "Idl", 0),
            getattr(P, "Rl", 0),
        ],
        dtype=float,
    )
    y0 = np.maximum(y0, 0.0)
    if normalize:
        s = y0.sum()
        if s <= 0 or not np.isfinite(s):
            raise ValueError("Initial conditions must have positive finite sum.")
        y0 = y0 / s
    return y0


def c_high_from_phi_v6(c_low: float, phi_t: float) -> float:
    """Replicates the built-in v6 rule for the high-strain contact rate."""
    vir_excess_pos = max(0.0, float(phi_t) - 1.0)
    c_high = float(c_low) * (1.0 - vir_excess_pos)
    return max(c_high, 0.0)


def theta_high(phi_t: float, theta: float, kappa_base: float, kappa_scale: float) -> float:
    kappa_high = float(kappa_base) * (1.0 + float(kappa_scale) * (float(phi_t) - 1.0))
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / float(theta))
    return float(theta) * kappa_high


def r0_proxy_high(
    phi_t: float,
    c_low: float,
    r_low: float,
    mc: float,
    mr: float,
    phi_recover: float,
    sigma: float,
    theta: float,
    kappa_base: float,
    kappa_scale: float,
) -> float:
    """A simple high-strain R0 proxy consistent with the analytic weighting logic."""
    c_high = c_high_from_phi_v6(c_low, phi_t)
    th = theta_high(phi_t, theta, kappa_base, kappa_scale)
    weight = (1.0 - th) + th * float(mc) * float(mr)
    avg_beta_h = float(phi_t) * c_high * float(r_low) * weight
    denom = max(float(phi_recover) * float(sigma), 1e-12)
    return avg_beta_h / denom


def params_tuple_v6(phi_t: float, mc: float, mr: float) -> Tuple[float, ...]:
    """SEIRS_model_v6 expects (14) params."""
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
    )

    birth_rate = float(getattr(P, "birth_rate", 0.0))
    death_rate = float(getattr(P, "death_rate", 0.0))
    delta = float(getattr(P, "delta", 0.0))
    kappa_base = float(getattr(P, "kappa_base", 1.0))
    kappa_scale = float(getattr(P, "kappa_scale", 1.0))
    phi_recover = float(getattr(P, "phi_recover", 1.0))
    sigma = float(getattr(P, "sigma", 1 / 5))
    tau = float(getattr(P, "tau", 1 / 3))
    theta = float(getattr(P, "theta", 0.3))

    return (
        c_low,
        r_low,
        float(phi_t),
        float(mc),
        float(mr),
        birth_rate,
        death_rate,
        delta,
        kappa_base,
        kappa_scale,
        phi_recover,
        sigma,
        tau,
        theta,
    )


def run_sim(phi_t: float, mc: float, mr: float, days: int, steps: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    t = np.linspace(0, days, steps)
    y0 = initial_conditions(normalize=True)
    sol = odeint(SEIRS_model_v6, y0, t, args=(params_tuple_v6(phi_t, mc, mr),))
    sim = {k: sol[:, i] for i, k in enumerate(COLS)}
    sim["I_high"] = sim["Indh"] + sim["Idh"]
    sim["I_low"] = sim["Indl"] + sim["Idl"]
    sim["I_total"] = sim["I_high"] + sim["I_low"]
    sim["high_frac"] = sim["I_high"] / (sim["I_total"] + 1e-12)
    return t, sim


def metrics(t: np.ndarray, sim: Dict[str, np.ndarray]) -> Dict[str, float]:
    I = sim["I_total"]
    S = sim["S"]
    ih = sim["I_high"]
    il = sim["I_low"]
    return {
        "peak_I_total": float(np.max(I)),
        "t_peak_I_total": float(t[int(np.argmax(I))]),
        "S_end": float(S[-1]),
        "attack_rate": float(1.0 - S[-1]),
        "peak_I_high": float(np.max(ih)),
        "peak_I_low": float(np.max(il)),
        "high_frac_end": float(sim["high_frac"][-1]),
        "high_frac_peak_time": float(sim["high_frac"][int(np.argmax(I))]),
    }


def peak_infection_landscape(df: pd.DataFrame, out_path: str) -> None:
    phi_vals = sorted(df["phi_transmission"].unique().tolist())
    mc_vals = sorted(df["mc"].unique().tolist())
    mr_vals = sorted(df["mr"].unique().tolist())

    if len(mc_vals) < 2 or len(mr_vals) < 2:
        raise ValueError("Need at least 2 values each for --mc and --mr to plot a surface.")

    X, Y = np.meshgrid(mr_vals, mc_vals)  # columns=mr, rows=mc

    hi_min = float(df["peak_I_high"].min())
    hi_max = float(df["peak_I_high"].max())
    lo_min = float(df["peak_I_low"].min())
    lo_max = float(df["peak_I_low"].max())

    cmap = matplotlib.colormaps["viridis"]
    norm_hi = Normalize(vmin=hi_min, vmax=hi_max)
    norm_lo = Normalize(vmin=lo_min, vmax=lo_max)

    ncols = len(phi_vals)
    fig = plt.figure(figsize=(4.2 * ncols, 8.0))

    for j, phi_t in enumerate(phi_vals):
        sub = df[np.isclose(df["phi_transmission"], float(phi_t))]
        z_hi = (
            sub.pivot(index="mc", columns="mr", values="peak_I_high")
            .reindex(index=mc_vals, columns=mr_vals)
            .to_numpy(dtype=float)
        )
        z_lo = (
            sub.pivot(index="mc", columns="mr", values="peak_I_low")
            .reindex(index=mc_vals, columns=mr_vals)
            .to_numpy(dtype=float)
        )

        ax_hi = fig.add_subplot(2, ncols, 1 + j, projection="3d")
        ax_lo = fig.add_subplot(2, ncols, 1 + ncols + j, projection="3d")

        ax_hi.plot_surface(
            X,
            Y,
            z_hi,
            facecolors=cmap(norm_hi(z_hi)),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        ax_lo.plot_surface(
            X,
            Y,
            z_lo,
            facecolors=cmap(norm_lo(z_lo)),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )

        ax_hi.set_title(f"phi={phi_t:g}")
        ax_lo.set_title(f"phi={phi_t:g}")

        ax_hi.set_xlabel("mr")
        ax_hi.set_ylabel("mc")
        ax_hi.set_zlabel("peak_I_high")

        ax_lo.set_xlabel("mr")
        ax_lo.set_ylabel("mc")
        ax_lo.set_zlabel("peak_I_low")

        ax_hi.set_zlim(hi_min, hi_max)
        ax_lo.set_zlim(lo_min, lo_max)

    sm_hi = cm.ScalarMappable(norm=norm_hi, cmap=cmap)
    sm_hi.set_array([])
    sm_lo = cm.ScalarMappable(norm=norm_lo, cmap=cmap)
    sm_lo.set_array([])

    fig.suptitle("Peak infection landscapes over (mc, mr), faceted by phi_transmission", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05, wspace=0.3, hspace=0.25)
    plt.savefig(out_path, dpi=600)
    plt.close(fig)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sweep drug modifiers vs phi_transmission (SEIRS v6)")
    ap.add_argument("--phi", nargs="+", type=float, default=None, help="phi_transmission values")
    ap.add_argument("--mc", nargs="+", type=float, default=None, help="drug_contact_multiplier values")
    ap.add_argument("--mr", nargs="+", type=float, default=None, help="drug_transmission_multiplier values")
    ap.add_argument("--days", type=int, default=int(getattr(P, "t_max", 365)))
    ap.add_argument("--steps", type=int, default=int(getattr(P, "t_steps", 365)))
    ap.add_argument("--baseline-phi", type=float, default=float(getattr(P, "phi_transmission", 1.5)))
    args = ap.parse_args(argv)

    phi_vals = args.phi if args.phi is not None else [1.0, 1.2, args.baseline_phi, 1.8, 2.2]
    mc_vals = args.mc if args.mc is not None else [0.8, 1.0, 1.2]
    mr_vals = args.mr if args.mr is not None else [0.5, 0.7, 1.0]

    phi_vals = sorted(set(float(x) for x in phi_vals))
    mc_vals = sorted(set(float(x) for x in mc_vals))
    mr_vals = sorted(set(float(x) for x in mr_vals))

    fig_dir = os.path.join( "Figures", "drug_modifier_phi_sweep")
    os.makedirs(fig_dir, exist_ok=True)

    # Pull shared params for proxy computations
    c_low = float(getattr(P, "contact_rate", 10.0))
    r_low = float(
        getattr(P, "transmission_probability_low", getattr(P, "transmission_probability", 0.025))
    )
    kappa_base = float(getattr(P, "kappa_base", 1.0))
    kappa_scale = float(getattr(P, "kappa_scale", 1.0))
    phi_recover = float(getattr(P, "phi_recover", 1.0))
    sigma = float(getattr(P, "sigma", 1 / 5))
    theta = float(getattr(P, "theta", 0.3))

    rows: List[Dict[str, float]] = []

    for phi_t in phi_vals:
        c_high = c_high_from_phi_v6(c_low, phi_t)
        th = theta_high(phi_t, theta, kappa_base, kappa_scale)
        for mc in mc_vals:
            for mr in mr_vals:
                t, sim = run_sim(phi_t, mc, mr, args.days, args.steps)
                m = metrics(t, sim)

                row: Dict[str, float] = {
                    "phi_transmission": float(phi_t),
                    "mc": float(mc),
                    "mr": float(mr),
                    "c_low": float(c_low),
                    "c_high_v6": float(c_high),
                    "theta_high": float(th),
                    "R0_proxy_high": float(
                        r0_proxy_high(
                            phi_t=phi_t,
                            c_low=c_low,
                            r_low=r_low,
                            mc=mc,
                            mr=mr,
                            phi_recover=phi_recover,
                            sigma=sigma,
                            theta=theta,
                            kappa_base=kappa_base,
                            kappa_scale=kappa_scale,
                        )
                    ),
                }
                row.update({
                    "peak_I_high": float(m["peak_I_high"]),
                    "peak_I_low": float(m["peak_I_low"]),
                })
                rows.append(row)

    df = pd.DataFrame(rows).sort_values(["phi_transmission", "mc", "mr"])\
        .reset_index(drop=True)

    out_fig = os.path.join(fig_dir, "peak_infection_landscape.png")
    peak_infection_landscape(df, out_fig)

    print(f"Saved: {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
