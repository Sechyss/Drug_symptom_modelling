import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# Workspace paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "../.."))

# Inputs produced by existing tornado scripts
CSV_HIGH = os.path.join(ROOT_DIR, "Outputs", "Tornado_v9_peak_I_high_separated.csv")
CSV_GAP = os.path.join(ROOT_DIR, "Outputs", "Tornado_v9_relative_peak_gap.csv")

# Output
OUT_DIR = os.path.join(ROOT_DIR, "Figures", "Model_v9_exploration")
OUT_SVG = os.path.join(OUT_DIR, "Figure_S4.svg")
OUT_PNG = os.path.join(OUT_DIR, "Figure_S4.png")

REL_PERTURB = 0.20
FIG_DPI = 600


def read_csv_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input file not found: {path}")

    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Input CSV is empty: {path}")

    return rows


def plot_high_virulence_panel(ax, rows):
    labels = [r["param"] for r in rows]
    neg = np.array([float(r["delta_low"]) for r in rows], dtype=float)
    pos = np.array([float(r["delta_high"]) for r in rows], dtype=float)
    y = np.arange(len(labels))

    ax.barh(y, neg, color="#4C72B0", alpha=0.85, label=f"-{int(REL_PERTURB * 100)}%")
    ax.barh(y, pos, color="#DD8452", alpha=0.85, label=f"+{int(REL_PERTURB * 100)}%")

    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel(r"$\Delta$ Peak $I_{high}/N_0$ (vs baseline)")
    ax.set_title("High-virulence strain tornado sensitivity")
    ax.legend(loc="lower right", frameon=True)

    xmax = float(np.max(np.abs(np.concatenate([neg, pos, np.array([0.0])]))))
    if xmax > 0:
        ax.set_xlim(-1.05 * xmax, 1.05 * xmax)


def plot_relative_gap_panel(ax, rows):
    labels = [r["param"] for r in rows]

    # Keep original tornado geometry: low perturbation values to the left.
    neg = -np.array([float(r["relative_gap_low_perturb"]) for r in rows], dtype=float)
    pos = np.array([float(r["relative_gap_high_perturb"]) for r in rows], dtype=float)
    y = np.arange(len(labels))

    ax.barh(y, neg, color="#4C72B0", alpha=0.85, label=f"-{int(REL_PERTURB * 100)}%")
    ax.barh(y, pos, color="#DD8452", alpha=0.85, label=f"+{int(REL_PERTURB * 100)}%")

    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel(r"Relative peak gap: $(I_{high}^{peak} - I_{low}^{peak}) / I_{low}^{peak}$")
    ax.set_title("Relative peak gap tornado sensitivity")
    ax.legend(loc="lower right", frameon=True)

    xmax = float(np.max(np.abs(np.concatenate([neg, pos, np.array([0.0])]))))
    if xmax > 0:
        ax.set_xlim(-1.05 * xmax, 1.05 * xmax)


def add_panel_label(ax, label):
    ax.text(
        -0.12,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def main():
    rows_high = read_csv_rows(CSV_HIGH)
    rows_gap = read_csv_rows(CSV_GAP)

    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 14))

    plot_high_virulence_panel(axes[0], rows_high)
    add_panel_label(axes[0], "A")

    plot_relative_gap_panel(axes[1], rows_gap)
    add_panel_label(axes[1], "B")

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    plt.savefig(OUT_SVG, dpi=FIG_DPI)
    plt.savefig(OUT_PNG, dpi=FIG_DPI)
    plt.close(fig)

    print(f"Saved: {OUT_SVG}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
