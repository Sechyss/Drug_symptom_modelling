# Standard library imports
import os

# Third-party imports for numerical computing and visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, AutoMinorLocator, LogLocator, ScalarFormatter, LogFormatterMathtext

# Configure scientific styling for publication-quality figures
sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
plt.rcParams.update({
    # Display and output resolution settings
    "figure.dpi": 120,        # Screen display resolution
    "savefig.dpi": 600,       # High-resolution output for publications
    # Axes and label styling
    "axes.linewidth": 1.0,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 11,
    "grid.alpha": 0.25,       # Subtle grid lines
    # Ensure vector-friendly text (no Type 3 fonts for journals)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    # Consistent font family
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
})

def theta_high(phi_t, theta, kappa_base, kappa_scale):
    """
    Calculate the fraction of high-strain infected individuals receiving treatment.
    
    The treatment rate increases with virulence (phi_t), assuming more virulent
    strains cause more severe symptoms that are more likely to be treated.
    
    Parameters:
    -----------
    phi_t : float
        Virulence multiplier for the high strain (relative to baseline)
    theta : float
        Base treatment rate
    kappa_base : float
        Baseline treatment scaling factor
    kappa_scale : float
        Rate at which treatment increases with virulence
    
    Returns:
    --------
    float : Effective treatment fraction (capped at 1.0)
    """
    # Treatment scaling increases linearly with virulence above baseline
    kappa_h = kappa_base * (1.0 + kappa_scale * (phi_t - 1.0))
    # Cap the treatment fraction at 1.0 (100% treatment)
    if theta > 0:
        kappa_h = min(kappa_h, 1.0 / theta)
    return theta * kappa_h

def required_c_high(phi_t, R0_target, r_low, m_c_drug, m_r_drug, phi_recover, sigma, theta, kappa_base, kappa_scale):
    """
    Calculate the required contact rate for the high-virulence strain to achieve target R0.
    
    This function accounts for the fact that treated individuals may have different
    contact patterns (m_c_drug) and transmission rates (m_r_drug) than untreated individuals.
    
    Parameters:
    -----------
    phi_t : float
        Virulence multiplier for the high strain
    R0_target : float
        Target basic reproduction number
    r_low : float
        Per-contact transmission probability for low-virulence strain
    m_c_drug : float
        Contact rate multiplier for treated individuals (>1 if treatment masks symptoms)
    m_r_drug : float
        Transmission rate multiplier for treated individuals (typically <1)
    phi_recover : float
        Recovery rate multiplier
    sigma : float
        Base recovery rate (1/infectious_period)
    theta : float
        Base treatment rate
    kappa_base : float
        Baseline treatment scaling factor
    kappa_scale : float
        Rate at which treatment increases with virulence
    
    Returns:
    --------
    float : Required contact rate to achieve target R0
    """
    # Calculate effective treatment fraction for this virulence level
    th = theta_high(phi_t, theta, kappa_base, kappa_scale)
    
    # Weighted average of transmission for untreated and treated populations
    # Untreated: (1-th) with baseline contact and transmission
    # Treated: th with modified contact (m_c_drug) and transmission (m_r_drug)
    weight = (1.0 - th) + th * m_c_drug * m_r_drug
    
    # Denominator for R0 equation: virulence * transmission * contact weight
    denom = max(phi_t * r_low * weight, 1e-12)  # Guard against divide-by-zero
    
    # Solve for required contact rate: R0 = (beta * c) / (recovery_rate)
    return R0_target * (phi_recover * sigma) / denom

def main():
    """Main function to analyze and visualize virulence-contact rate trade-offs."""
    
    # ============================================================
    # EPIDEMIOLOGICAL PARAMETERS
    # ============================================================
    
    # Target basic reproduction number to maintain across virulence levels
    R0_target = 2.0
    
    # Baseline transmission and recovery parameters
    r_low = 0.05          # Per-contact transmission probability (baseline strain)
    sigma = 1/7.0         # Recovery rate (1/infectious_period in days)
    phi_recover = 1.0     # Recovery rate multiplier
    
    # Treatment parameters
    theta = 0.6           # Base treatment rate
    kappa_base = 0.5      # Baseline treatment scaling factor
    kappa_scale = 2.0     # How quickly treatment uptake increases with virulence
    
    # Drug impact on behavior and transmission
    # Note: If treatment masks symptoms, treated individuals may mix more socially
    m_c_drug = 1.25       # Contact rate multiplier for treated individuals (>1 = increased mixing)
    m_r_drug = 0.7        # Transmission rate multiplier for treated individuals (<=1 if drug reduces shedding)

    # ============================================================
    # CALCULATE REQUIRED CONTACT RATES ACROSS VIRULENCE SPECTRUM
    # ============================================================
    
    # Sweep virulence multiplier on logarithmic grid from 1 to 10,000
    # (exploring extreme virulence scenarios)
    phi_vals = np.logspace(0, 4, 120)  # 10^0=1 to 10^4

    # Storage for results
    c_high_vals, R0_check = [], []
    
    for phi_t in phi_vals:
        # Calculate the required contact rate to maintain R0_target at this virulence
        c_h = required_c_high(phi_t, R0_target, r_low, m_c_drug, m_r_drug,
                              phi_recover, sigma, theta, kappa_base, kappa_scale)
        c_high_vals.append(c_h)

        # Verify that the calculated contact rate produces the target R0
        th = theta_high(phi_t, theta, kappa_base, kappa_scale)
        # Average transmission rate accounting for treated and untreated populations
        avg_beta_h = phi_t * c_h * r_low * ((1.0 - th) + th * m_c_drug * m_r_drug)
        # Calculate actual R0 from this transmission rate
        R0_h_approx = avg_beta_h / (phi_recover * sigma)
        R0_check.append(R0_h_approx)

    # Convert to numpy arrays for vectorized operations
    c_high_vals = np.array(c_high_vals)
    R0_check = np.array(R0_check)
    
    # Calculate percent deviation from target R0 (should be near zero)
    R0_pct_err = 100.0 * (R0_check / R0_target - 1.0)

    # ============================================================
    # ASYMPTOTIC ANALYSIS
    # ============================================================
    
    # For very high virulence (theta_high → 1, everyone treated):
    # Contact rate approaches c_high ≈ K / phi_t, where:
    # K = R0_target * (phi_recover * sigma) / (r_low * m_c_drug * m_r_drug)
    K_lim = R0_target * (phi_recover * sigma) / (r_low * m_c_drug * m_r_drug)
    c_high_asym = K_lim / phi_vals

    # ============================================================
    # VISUALIZATION: TWO-PANEL FIGURE
    # ============================================================
    
    # Create figure with two panels optimized for publication
    fig, axes = plt.subplots(
        2, 1, figsize=(3.6, 4.2), sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]}  # Top panel larger than bottom
    )
    colors = sns.color_palette()

    # -----------------------------------------------------------
    # PANEL 1: Required contact rate vs virulence (log-log scale)
    # -----------------------------------------------------------
    ax = axes[0]
    
    # Plot calculated required contact rates
    ax.plot(phi_vals, c_high_vals, color=colors[0], lw=2.2, marker='o', ms=3.0, label='Required c_high')
    # Plot asymptotic approximation for comparison
    ax.plot(phi_vals, c_high_asym, color=colors[2], lw=1.6, linestyle=':', label='Asymptote ~ K/φ_t')
    
    # Configure log-log scale to show exponential decay
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('c_high (contacts/day, log scale)')
    
    # Set up logarithmic tick marks and labels
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())  # Format as 10^n
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    
    # Add grid lines for readability
    ax.grid(True, which='major', axis='both')
    ax.grid(True, which='minor', axis='both', alpha=0.12)
    ax.legend(frameon=True, loc='best')
    ax.tick_params(direction='in', top=True, right=True, length=4)

    # -----------------------------------------------------------
    # PANEL 2: Verification of R0 accuracy (error plot)
    # -----------------------------------------------------------
    ax_err = axes[1]
    
    # Add reference line at zero (perfect match to target)
    ax_err.axhline(0.0, color='k', lw=1.0, alpha=0.6, linestyle='-')
    # Plot percent deviation from target R0
    ax_err.plot(phi_vals, R0_pct_err, color=colors[1], lw=2.0, linestyle='--', label='R0 deviation (%)')
    
    # Configure axes (log x-axis, linear y-axis)
    ax_err.set_xscale('log')
    ax_err.set_xlabel('φ_t (virulence multiplier, high strain, log scale)')
    ax_err.set_ylabel('ΔR0 (%)')
    
    # Format tick labels
    ax_err.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # One decimal place
    ax_err.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax_err.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax_err.yaxis.set_minor_locator(AutoMinorLocator())  # Linear y-axis uses auto minor ticks
    ax_err.xaxis.set_major_formatter(LogFormatterMathtext())
    
    # Add grid for readability
    ax_err.grid(True, axis='y', which='major')
    ax_err.grid(True, axis='y', which='minor', alpha=0.15)
    ax_err.tick_params(direction='in', top=True, right=True, length=4)
    ax_err.legend(frameon=True, loc='best')

    # Set symmetric y-axis limits centered on zero
    err_max = np.nanmax(np.abs(R0_pct_err))
    pad = max(0.5, 0.1 * err_max)  # Add small padding
    ax_err.set_ylim(-err_max - pad, err_max + pad)

    # Remove top and right spines for cleaner look
    sns.despine(fig=fig, trim=True)

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    
    # Create output directory if it doesn't exist
    outdir = os.path.join(os.path.dirname(__file__), '..', 'Figures')
    os.makedirs(outdir, exist_ok=True)
    
    # Save in both PNG (for quick viewing) and PDF (for publication)
    png_path = os.path.join(outdir, 'increase_virulence_c_contact_extreme.png')
    pdf_path = os.path.join(outdir, 'increase_virulence_c_contact_extreme.pdf')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {png_path}\nSaved: {pdf_path}")

    # Display the figure
    plt.show()

# Entry point: run main function when script is executed directly
if __name__ == "__main__":
    main()