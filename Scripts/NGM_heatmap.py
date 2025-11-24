import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to allow imports from Models/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Scripts.NGM import compute_NGM, load_params

# Make reproducible
np.random.seed(42)

"""
NGM_heatmap.py

Create heatmap of R0 values across p_recover vs theta parameter space.
This complements the equilibrium analysis in Sweep_phi_theta_heatmap.py
Updated for SEIRS_model_v2 with kappa-based detection mechanism.
"""

# Create directories if needed
os.makedirs('../Figures', exist_ok=True)
os.makedirs('../Tables', exist_ok=True)

# Load baseline parameters
params = load_params()

# Define parameter ranges
theta_vals = np.linspace(0.0, 1.0, 20)
p_recover_vals = np.linspace(0.0, 1.0, 20)  # Updated: now 0-1 (transmission reduction)

# Storage arrays
R0_grid = np.zeros((len(theta_vals), len(p_recover_vals)))
R0_high_grid = np.zeros_like(R0_grid)  # High-virulence specific R0
R0_low_grid = np.zeros_like(R0_grid)   # Low-virulence specific R0

print("Computing R0 across parameter space...")
print(f"Grid size: {len(theta_vals)} × {len(p_recover_vals)} = {len(theta_vals) * len(p_recover_vals)} simulations")
print(f"theta range: [{theta_vals[0]:.2f}, {theta_vals[-1]:.2f}]")
print(f"p_recover range: [{p_recover_vals[0]:.2f}, {p_recover_vals[-1]:.2f}]")
print(f"  (Note: p_recover now represents transmission REDUCTION, not recovery boost)")

# Parameter sweep
for i, theta_val in enumerate(theta_vals):
    for j, p_val in enumerate(p_recover_vals):
        # Update parameters
        params['theta'] = theta_val
        params['p_recover'] = p_val
        
        # Compute NGM
        try:
            result = compute_NGM(params)
            R0_grid[i, j] = result['R0']
            
            # Extract strain-specific R0 from eigenvalues
            # The NGM eigenvalues correspond to different transmission pathways
            # For our 2-strain model, the top 2 eigenvalues roughly correspond to each strain
            eigs_real = np.real(result['eigenvalues'])
            eigs_sorted = sorted(eigs_real, reverse=True)
            
            # Note: These are approximate strain-specific R0s
            # The actual coupling between strains makes this interpretation fuzzy
            R0_high_grid[i, j] = eigs_sorted[0] if len(eigs_sorted) > 0 else 0
            R0_low_grid[i, j] = eigs_sorted[1] if len(eigs_sorted) > 1 else 0
            
        except Exception as e:
            print(f"Error at theta={theta_val:.2f}, p_recover={p_val:.2f}: {e}")
            R0_grid[i, j] = np.nan
            R0_high_grid[i, j] = np.nan
            R0_low_grid[i, j] = np.nan
    
    # Progress indicator
    if (i + 1) % 5 == 0:
        print(f"  Completed {i + 1}/{len(theta_vals)} rows...")

print("Done!")

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

def plot_R0_heatmap(ax, data, title, vmin=0, vmax=None):
    """Plot R0 heatmap with R0=1 threshold contour"""
    if vmax is None:
        vmax = np.nanmax(data)
    
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   extent=[p_recover_vals[0], p_recover_vals[-1], 
                          theta_vals[0], theta_vals[-1]],
                   vmin=vmin, vmax=vmax)
    
    # Add R0=1 threshold contour
    try:
        contour = ax.contour(p_recover_vals, theta_vals, data, levels=[1.0], 
                             colors='black', linewidths=2, linestyles='--')
        ax.clabel(contour, inline=True, fontsize=10, fmt='R₀=1')
    except:
        pass  # Skip if contour fails (e.g., no R0=1 crossing)
    
    ax.set_xlabel('p_recover (transmission multiplier for treated, 0=no transmission)', fontsize=11)
    ax.set_ylabel('theta (treatment coverage)', fontsize=11)
    ax.set_title(title, fontsize=12)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('R₀', fontsize=11)
    
    return im

# Plot all three R0 metrics
plot_R0_heatmap(axes[0], R0_grid, 'Overall R₀ (spectral radius)')
plot_R0_heatmap(axes[1], R0_high_grid, 'Dominant eigenvalue\n(~High-virulence R₀)')
plot_R0_heatmap(axes[2], R0_low_grid, 'Second eigenvalue\n(~Low-virulence R₀)')

plt.tight_layout()
plt.savefig('../Figures/R0_heatmap_p_recover_theta_v2.png', dpi=600)
print("\nSaved figure: ../Figures/R0_heatmap_p_recover_theta_v2.png")
plt.close()

# Export data
df = pd.DataFrame({
    'theta': np.repeat(theta_vals, len(p_recover_vals)),
    'p_recover': np.tile(p_recover_vals, len(theta_vals)),
    'R0_overall': R0_grid.flatten(),
    'R0_high': R0_high_grid.flatten(),
    'R0_low': R0_low_grid.flatten()
})
df.to_csv('../Tables/R0_heatmap_results_v2.csv', index=False)
print("Saved data: ../Tables/R0_heatmap_results_v2.csv")

# Summary statistics
print("\n" + "="*70)
print("Summary Statistics:")
print("="*70)
print(f"Overall R₀ range: [{np.nanmin(R0_grid):.3f}, {np.nanmax(R0_grid):.3f}]")
print(f"High-virulence R₀ range: [{np.nanmin(R0_high_grid):.3f}, {np.nanmax(R0_high_grid):.3f}]")
print(f"Low-virulence R₀ range: [{np.nanmin(R0_low_grid):.3f}, {np.nanmax(R0_low_grid):.3f}]")

# Find parameter combinations with R0 < 1 (disease elimination possible)
elimination_zone = R0_grid < 1.0
n_elimination = np.sum(elimination_zone)
pct_elimination = 100 * n_elimination / R0_grid.size
print(f"\nParameter combinations with R₀ < 1: {n_elimination}/{R0_grid.size} ({pct_elimination:.1f}%)")

if n_elimination > 0:
    print("\n✓ Disease elimination is possible in some parameter regimes")
    
    # Find boundaries of elimination zone
    elim_indices = np.where(elimination_zone)
    theta_elim_min = theta_vals[elim_indices[0]].min()
    theta_elim_max = theta_vals[elim_indices[0]].max()
    p_recover_elim_min = p_recover_vals[elim_indices[1]].min()
    p_recover_elim_max = p_recover_vals[elim_indices[1]].max()
    
    print(f"  Elimination zone:")
    print(f"    theta: [{theta_elim_min:.2f}, {theta_elim_max:.2f}]")
    print(f"    p_recover: [{p_recover_elim_min:.2f}, {p_recover_elim_max:.2f}]")
else:
    print("\n⚠️ Disease will spread (R₀ > 1) across all tested parameters")

# Additional analysis: Effect of treatment parameters
print("\n" + "="*70)
print("Treatment Effect Analysis:")
print("="*70)

# Compare R0 at different treatment scenarios
scenarios = [
    ('No treatment', 0.0, 0.5),
    ('Low coverage, low efficacy', 0.3, 0.25),
    ('Low coverage, high efficacy', 0.3, 0.0),
    ('High coverage, low efficacy', 0.9, 0.25),
    ('High coverage, high efficacy', 0.9, 0.0),
]

print(f"\n{'Scenario':<30s} {'theta':>6s} {'p_rec':>6s} {'R0':>8s}")
print("-" * 55)

for scenario_name, theta_val, p_rec_val in scenarios:
    params_test = params.copy()
    params_test['theta'] = theta_val
    params_test['p_recover'] = p_rec_val
    
    result = compute_NGM(params_test)
    print(f"{scenario_name:<30s} {theta_val:>6.2f} {p_rec_val:>6.2f} {result['R0']:>8.3f}")

# Marginal effects plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Effect of theta (averaged over p_recover)
R0_theta_mean = np.nanmean(R0_grid, axis=1)
R0_theta_std = np.nanstd(R0_grid, axis=1)

axes[0].plot(theta_vals, R0_theta_mean, 'b-', linewidth=2, label='Mean R₀')
axes[0].fill_between(theta_vals, 
                      R0_theta_mean - R0_theta_std, 
                      R0_theta_mean + R0_theta_std, 
                      alpha=0.3, label='± 1 std')
axes[0].axhline(1.0, color='red', linestyle='--', linewidth=1, label='R₀=1')
axes[0].set_xlabel('Treatment coverage (θ)', fontsize=11)
axes[0].set_ylabel('R₀', fontsize=11)
axes[0].set_title('Marginal effect of treatment coverage', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Effect of p_recover (averaged over theta)
R0_p_recover_mean = np.nanmean(R0_grid, axis=0)
R0_p_recover_std = np.nanstd(R0_grid, axis=0)

axes[1].plot(p_recover_vals, R0_p_recover_mean, 'b-', linewidth=2, label='Mean R₀')
axes[1].fill_between(p_recover_vals, 
                      R0_p_recover_mean - R0_p_recover_std, 
                      R0_p_recover_mean + R0_p_recover_std, 
                      alpha=0.3, label='± 1 std')
axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1, label='R₀=1')
axes[1].set_xlabel('p_recover (treated transmission multiplier)', fontsize=11)
axes[1].set_ylabel('R₀', fontsize=11)
axes[1].set_title('Marginal effect of treatment efficacy', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../Figures/R0_marginal_effects_v2.png', dpi=300)
print("\nSaved figure: ../Figures/R0_marginal_effects_v2.png")
plt.close()

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)