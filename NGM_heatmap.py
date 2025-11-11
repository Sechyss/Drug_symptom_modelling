import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from NGM import compute_NGM, load_params

# Make reproducible
np.random.seed(42)

"""
NGM_heatmap.py

Create heatmap of R0 values across p_recover vs theta parameter space.
This complements the equilibrium analysis in Sweep_phi_theta_heatmap.py
"""

# Create directories if needed
os.makedirs('./Figures', exist_ok=True)
os.makedirs('./Tables', exist_ok=True)

# Load baseline parameters
params = load_params()

# Define parameter ranges (matching your other scripts)
theta_vals = np.linspace(0.0, 1.0, 20)
p_recover_vals = np.linspace(1.0, 2.0, 20)

# Storage arrays
R0_grid = np.zeros((len(theta_vals), len(p_recover_vals)))
R0_high_grid = np.zeros_like(R0_grid)  # High-virulence specific R0
R0_low_grid = np.zeros_like(R0_grid)   # Low-virulence specific R0

print("Computing R0 across parameter space...")
print(f"Grid size: {len(theta_vals)} × {len(p_recover_vals)} = {len(theta_vals) * len(p_recover_vals)} simulations")

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
            eigs_sorted = sorted(np.real(result['eigenvalues']), reverse=True)
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
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_R0_heatmap(ax, data, title, vmin=0, vmax=None):
    """Plot R0 heatmap with R0=1 threshold contour"""
    if vmax is None:
        vmax = np.nanmax(data)
    
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   extent=[p_recover_vals[0], p_recover_vals[-1], 
                          theta_vals[0], theta_vals[-1]],
                   vmin=vmin, vmax=vmax)
    
    # Add R0=1 threshold contour
    contour = ax.contour(p_recover_vals, theta_vals, data, levels=[1.0], 
                         colors='black', linewidths=2, linestyles='--')
    ax.clabel(contour, inline=True, fontsize=10, fmt='R₀=1')
    
    ax.set_xlabel('p_recover (recovery rate multiplier)')
    ax.set_ylabel('theta (fraction using symptom-blocking drug)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('R₀')
    
    return im

# Plot all three R0 metrics
plot_R0_heatmap(axes[0], R0_grid, 'Overall R₀')
plot_R0_heatmap(axes[1], R0_high_grid, 'High-virulence R₀')
plot_R0_heatmap(axes[2], R0_low_grid, 'Low-virulence R₀')

plt.tight_layout()
plt.savefig('./Figures/R0_heatmap_p_recover_theta.png', dpi=300)
print("\nSaved figure: ./Figures/R0_heatmap_p_recover_theta.png")
plt.show()

# Export data
df = pd.DataFrame({
    'theta': np.repeat(theta_vals, len(p_recover_vals)),
    'p_recover': np.tile(p_recover_vals, len(theta_vals)),
    'R0_overall': R0_grid.flatten(),
    'R0_high': R0_high_grid.flatten(),
    'R0_low': R0_low_grid.flatten()
})
df.to_csv('./Tables/R0_heatmap_results.csv', index=False)
print("Saved data: ./Tables/R0_heatmap_results.csv")

# Summary statistics
print("\n" + "="*60)
print("Summary Statistics:")
print("="*60)
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
else:
    print("\n⚠️ Disease will spread (R₀ > 1) across all tested parameters")