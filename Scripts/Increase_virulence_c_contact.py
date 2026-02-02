import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, AutoMinorLocator, LogLocator, ScalarFormatter, LogFormatterMathtext

# Scientific styling
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 11,
    "grid.alpha": 0.25,
    # Ensure vector-friendly text (no Type 3 fonts)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Consistent fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

def theta_high(phi_t, theta, kappa_base, kappa_scale):
    kappa_h = kappa_base * (1.0 + kappa_scale * (phi_t - 1.0))
    # Cap so that theta_high <= 1
    if theta > 0:
        kappa_h = min(kappa_h, 1.0 / theta)
    return theta * kappa_h

def required_c_high(phi_t, R0_target, r_low, m_c_drug, m_r_drug, phi_recover, sigma, theta, kappa_base, kappa_scale):
    th = theta_high(phi_t, theta, kappa_base, kappa_scale)
    weight = (1.0 - th) + th * m_c_drug * m_r_drug
    denom = max(phi_t * r_low * weight, 1e-12)  # guard against divide-by-zero
    return R0_target * (phi_recover * sigma) / denom

def main():
    # Fixed parameters (adjust as needed)
    R0_target = 2.0
    r_low = 0.05
    # If treatment masks symptoms, treated people mix more: m_c_drug > 1
    m_c_drug = 1.25   # treated contact multiplier (>1 = increased mixing)
    m_r_drug = 0.7    # per-contact transmission modifier for treated (<=1 if drug reduces shedding)
    phi_recover = 1.0
    sigma = 1/7.0
    theta = 0.6
    kappa_base = 0.5
    kappa_scale = 2.0

    # Sweep phi_t values on a log grid up to 1e4 (absurdly high virulence multiplier)
    phi_vals = np.logspace(0, 4, 120)  # 10^0=1 to 10^4

    c_high_vals, R0_check = [], []
    for phi_t in phi_vals:
        c_h = required_c_high(phi_t, R0_target, r_low, m_c_drug, m_r_drug,
                              phi_recover, sigma, theta, kappa_base, kappa_scale)
        c_high_vals.append(c_h)

        th = theta_high(phi_t, theta, kappa_base, kappa_scale)
        # Treated fraction th has both contact and transmission modifiers.
        avg_beta_h = phi_t * c_h * r_low * ((1.0 - th) + th * m_c_drug * m_r_drug)
        R0_h_approx = avg_beta_h / (phi_recover * sigma)
        R0_check.append(R0_h_approx)

    c_high_vals = np.array(c_high_vals)
    R0_check = np.array(R0_check)
    R0_pct_err = 100.0 * (R0_check / R0_target - 1.0)

    # Asymptotic behavior for large phi_t (theta_high → 1):
    # c_high ≈ K / phi_t, where K = R0_target * (phi_recover * sigma) / (r_low * m_c_drug * m_r_drug)
    K_lim = R0_target * (phi_recover * sigma) / (r_low * m_c_drug * m_r_drug)
    c_high_asym = K_lim / phi_vals

    # Figure with two panels (paper-friendly)
    fig, axes = plt.subplots(
        2, 1, figsize=(3.6, 4.2), sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    colors = sns.color_palette()

    # Top panel: required c_high (log-log to show approach to ~0)
    ax = axes[0]
    ax.plot(phi_vals, c_high_vals, color=colors[0], lw=2.2, marker='o', ms=3.0, label='Required c_high')
    ax.plot(phi_vals, c_high_asym, color=colors[2], lw=1.6, linestyle=':', label='Asymptote ~ K/φ_t')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('c_high (contacts/day, log scale)')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    # Replace AutoMinorLocator with LogLocator for minor ticks on log axes
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    # Log tick labels in 10^n format
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(True, which='major', axis='both')
    ax.grid(True, which='minor', axis='both', alpha=0.12)
    ax.legend(frameon=True, loc='best')
    ax.tick_params(direction='in', top=True, right=True, length=4)

    # Bottom panel: percent deviation of R0 from target (x log-scale)
    ax_err = axes[1]
    ax_err.axhline(0.0, color='k', lw=1.0, alpha=0.6, linestyle='-')
    ax_err.plot(phi_vals, R0_pct_err, color=colors[1], lw=2.0, linestyle='--', label='R0 deviation (%)')
    ax_err.set_xscale('log')
    ax_err.set_xlabel('φ_t (virulence multiplier, high strain, log scale)')
    ax_err.set_ylabel('ΔR0 (%)')
    ax_err.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_err.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    # Minor ticks for log x-axis
    ax_err.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    # Keep AutoMinorLocator for linear y-axis
    ax_err.yaxis.set_minor_locator(AutoMinorLocator())
    ax_err.xaxis.set_major_formatter(LogFormatterMathtext())
    ax_err.grid(True, axis='y', which='major')
    ax_err.grid(True, axis='y', which='minor', alpha=0.15)
    ax_err.tick_params(direction='in', top=True, right=True, length=4)
    ax_err.legend(frameon=True, loc='best')

    # Symmetric tight bounds around zero for the error panel
    err_max = np.nanmax(np.abs(R0_pct_err))
    pad = max(0.5, 0.1 * err_max)
    ax_err.set_ylim(-err_max - pad, err_max + pad)

    sns.despine(fig=fig, trim=True)

    # Save high-quality outputs
    outdir = os.path.join(os.path.dirname(__file__), '..', 'Figures')
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, 'increase_virulence_c_contact_extreme.png')
    pdf_path = os.path.join(outdir, 'increase_virulence_c_contact_extreme.pdf')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}\nSaved: {pdf_path}")

    plt.show()

if __name__ == "__main__":
    main()