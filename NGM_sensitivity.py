import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from NGM import compute_NGM, load_params

"""
NGM_sensitivity.py

Sensitivity analysis of R0 to individual parameter changes.
Shows which parameters most strongly affect disease transmission.
"""

params = load_params()
baseline_R0 = compute_NGM(params)['R0']

print("="*60)
print("R₀ Sensitivity Analysis")
print("="*60)
print(f"Baseline R₀: {baseline_R0:.4f}\n")

# Define parameter ranges for sensitivity
param_ranges = {
    'beta_l': np.linspace(0.1, 0.4, 30),
    'p_recover': np.linspace(0.5, 2.5, 30),
    'theta': np.linspace(0.0, 1.0, 30),
    'phi_transmission': np.linspace(1.0, 1.2, 30),
    'phi_recover': np.linspace(0.8, 1.0, 30),
    'sigma': np.linspace(1/20, 1/5, 30),
    'tau': np.linspace(1/5, 1/1, 30),
    'delta': np.linspace(1/180, 1/30, 30)
}

# Compute R0 for each parameter variation
results = {}

for param_name, param_values in param_ranges.items():
    R0_values = []
    
    for val in param_values:
        test_params = params.copy()
        test_params[param_name] = val
        
        try:
            result = compute_NGM(test_params)
            R0_values.append(result['R0'])
        except:
            R0_values.append(np.nan)
    
    results[param_name] = {
        'values': param_values,
        'R0': np.array(R0_values)
    }

# Plotting
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (param_name, data) in enumerate(results.items()):
    ax = axes[idx]
    
    ax.plot(data['values'], data['R0'], 'b-', linewidth=2)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='R₀=1')
    ax.axhline(baseline_R0, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Baseline')
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('R₀')
    ax.set_title(f'Effect of {param_name} on R₀')
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('./Figures/R0_sensitivity_analysis.png', dpi=600)
print("Saved: ./Figures/R0_sensitivity_analysis.png")
plt.show()

# Compute normalized sensitivity indices
print("\nNormalized Sensitivity Indices:")
print("(Change in R₀ per 10% change in parameter)\n")

sensitivity_indices = {}
for param_name, data in results.items():
    # Find R0 at ±10% of baseline parameter value
    baseline_val = params[param_name]
    
    # Interpolate R0 at baseline ± 10%
    low_val = baseline_val * 0.9
    high_val = baseline_val * 1.1
    
    R0_low = np.interp(low_val, data['values'], data['R0'])
    R0_high = np.interp(high_val, data['values'], data['R0'])
    
    # Sensitivity = (ΔR0 / R0) / (Δparam / param)
    sensitivity = (R0_high - R0_low) / baseline_R0 / 0.2  # 0.2 = 20% total change
    sensitivity_indices[param_name] = sensitivity
    
    print(f"  {param_name:20s}: {sensitivity:+.4f}")

# Save sensitivity results
df_sens = pd.DataFrame({
    'parameter': list(sensitivity_indices.keys()),
    'sensitivity_index': list(sensitivity_indices.values())
}).sort_values('sensitivity_index', key=abs, ascending=False)

df_sens.to_csv('./Tables/R0_sensitivity_indices.csv', index=False)
print("\nSaved: ./Tables/R0_sensitivity_indices.csv")

# Bar plot of sensitivities
plt.figure(figsize=(10, 6))
colors = ['red' if s < 0 else 'green' for s in df_sens['sensitivity_index']]
plt.barh(df_sens['parameter'], df_sens['sensitivity_index'], color=colors, alpha=0.7)
plt.xlabel('Normalized Sensitivity Index')
plt.title('Parameter Sensitivity of R₀\n(Positive = increasing parameter increases R₀)')
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('./Figures/R0_sensitivity_barplot.png', dpi=600)
print("Saved: ./Figures/R0_sensitivity_barplot.png")
plt.show()