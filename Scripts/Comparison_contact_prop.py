import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Add parent directory to path to allow imports from Models/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.SEIRS_Models import SEIRS_model_v2
from Models import params as model_params

"""
Comparison_contact_prop.py

Compare different strategies for achieving target R0 values:
1. Varying contact_rate (fixed transmission_probability)
2. Varying transmission_probability (fixed contact_rate)
3. Varying both proportionally (balanced)

This helps understand which epidemiological pathway (contact patterns vs 
transmission efficiency) has stronger effects on disease dynamics.

Updated for SEIRS_model_v2 with kappa-based detection.
"""

# Create output directories
os.makedirs('../Figures', exist_ok=True)
os.makedirs('../Tables', exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_base_params():
    """Load baseline parameters from params.py"""
    return {
        'contact_rate': getattr(model_params, 'contact_rate', 10.0),
        'transmission_probability': getattr(model_params, 'transmission_probability', 0.025),
        'birth_rate': getattr(model_params, 'birth_rate', 0.0),
        'death_rate': getattr(model_params, 'death_rate', 0.0),
        'delta': getattr(model_params, 'delta', 1/120),
        'kappa_base': getattr(model_params, 'kappa_base', 1.0),
        'kappa_scale': getattr(model_params, 'kappa_scale', 1.0),
        'p_recover': getattr(model_params, 'p_recover', 0.5),
        'phi_recover': getattr(model_params, 'phi_recover', 1.0),
        'phi_transmission': getattr(model_params, 'phi_transmission', 1.05),
        'sigma': getattr(model_params, 'sigma', 1/5),
        'tau': getattr(model_params, 'tau', 1/3),
        'theta': getattr(model_params, 'theta', 0.3)
    }

def params_dict_to_tuple(params_dict):
    """Convert parameter dictionary to tuple for ODE solver."""
    return (
        params_dict['contact_rate'],
        params_dict['transmission_probability'],
        params_dict['birth_rate'],
        params_dict['death_rate'],
        params_dict['delta'],
        params_dict['kappa_base'],
        params_dict['kappa_scale'],
        params_dict['p_recover'],
        params_dict['phi_recover'],
        params_dict['phi_transmission'],
        params_dict['sigma'],
        params_dict['tau'],
        params_dict['theta']
    )

def compute_R0(params_dict):
    """Compute R0 accounting for treatment effects and kappa."""
    beta_l = params_dict['contact_rate'] * params_dict['transmission_probability']
    beta_h = params_dict['phi_transmission'] * beta_l
    
    theta = params_dict['theta']
    p_recover = params_dict['p_recover']
    sigma = params_dict['sigma']
    phi_recover = params_dict['phi_recover']
    kappa_base = params_dict['kappa_base']
    kappa_scale = params_dict['kappa_scale']
    
    # Compute kappa values
    virulence_excess = params_dict['phi_transmission'] - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * virulence_excess)
    kappa_low = kappa_base
    
    # Safety caps
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)
    
    # Effective treatment fractions
    theta_eff_high = kappa_high * theta
    theta_eff_low = kappa_low * theta
    
    # Effective transmission rates
    beta_eff_high = beta_h * (1 - theta_eff_high + p_recover * theta_eff_high)
    beta_eff_low = beta_l * (1 - theta_eff_low + p_recover * theta_eff_low)
    
    # Recovery rates
    sigma_eff_high = phi_recover * sigma
    sigma_eff_low = sigma
    
    R0_high = beta_eff_high / sigma_eff_high
    R0_low = beta_eff_low / sigma_eff_low
    
    return R0_low, R0_high

def run_simulation(params_dict, days=100):
    """Run SEIRS simulation and return time series."""
    # Initial conditions
    y0 = np.array([
        getattr(model_params, 'S', 10000),
        getattr(model_params, 'Eh', 0),
        getattr(model_params, 'Indh', 5),
        getattr(model_params, 'Idh', 0),
        getattr(model_params, 'Rh', 0),
        getattr(model_params, 'El', 0),
        getattr(model_params, 'Indl', 5),
        getattr(model_params, 'Idl', 0),
        getattr(model_params, 'Rl', 0)
    ], dtype=float)
    y0 = y0 / y0.sum()  # Normalize to proportions
    
    t = np.linspace(0, days, days)
    params_tuple = params_dict_to_tuple(params_dict)
    
    sol = odeint(SEIRS_model_v2, y0, t, args=(params_tuple,))
    
    return {
        't': t,
        'S': sol[:, 0],
        'Eh': sol[:, 1],
        'Indh': sol[:, 2],
        'Idh': sol[:, 3],
        'Rh': sol[:, 4],
        'El': sol[:, 5],
        'Indl': sol[:, 6],
        'Idl': sol[:, 7],
        'Rl': sol[:, 8],
        'I_high': sol[:, 2] + sol[:, 3],
        'I_low': sol[:, 6] + sol[:, 7],
        'I_total': sol[:, 2] + sol[:, 3] + sol[:, 6] + sol[:, 7]
    }

# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

def vary_contact_rate(base_params, R0_target):
    """Strategy 1: Fix transmission_probability, vary contact_rate"""
    params = base_params.copy()
    
    # Target R0 for low-virulence strain
    sigma = params['sigma']
    theta = params['theta']
    p_recover = params['p_recover']
    kappa_low = params['kappa_base']
    
    # Effective recovery accounting for treatment
    theta_eff_low = kappa_low * theta
    if theta > 0:
        theta_eff_low = min(theta_eff_low, 1.0)
    
    sigma_eff = sigma * (1 - theta_eff_low + p_recover * theta_eff_low)
    
    # Required beta_l
    beta_l_target = R0_target * sigma_eff
    
    # Solve for contact_rate
    params['contact_rate'] = beta_l_target / params['transmission_probability']
    
    return params

def vary_transmission_prob(base_params, R0_target):
    """Strategy 2: Fix contact_rate, vary transmission_probability"""
    params = base_params.copy()
    
    # Same calculation as above
    sigma = params['sigma']
    theta = params['theta']
    p_recover = params['p_recover']
    kappa_low = params['kappa_base']
    
    theta_eff_low = kappa_low * theta
    if theta > 0:
        theta_eff_low = min(theta_eff_low, 1.0)
    
    sigma_eff = sigma * (1 - theta_eff_low + p_recover * theta_eff_low)
    beta_l_target = R0_target * sigma_eff
    
    # Solve for transmission_probability
    params['transmission_probability'] = beta_l_target / params['contact_rate']
    
    return params

def vary_both_balanced(base_params, R0_target):
    """Strategy 3: Vary both proportionally (maintain ratio)"""
    params = base_params.copy()
    
    # Calculate target beta
    sigma = params['sigma']
    theta = params['theta']
    p_recover = params['p_recover']
    kappa_low = params['kappa_base']
    
    theta_eff_low = kappa_low * theta
    if theta > 0:
        theta_eff_low = min(theta_eff_low, 1.0)
    
    sigma_eff = sigma * (1 - theta_eff_low + p_recover * theta_eff_low)
    beta_l_target = R0_target * sigma_eff
    
    # Current beta
    beta_l_current = params['contact_rate'] * params['transmission_probability']
    
    # Scale factor
    scale = np.sqrt(beta_l_target / beta_l_current)
    
    params['contact_rate'] *= scale
    params['transmission_probability'] *= scale
    
    return params

# ============================================================================
# MAIN COMPARISON
# ============================================================================

print("="*70)
print("Contact Rate vs Transmission Probability Comparison")
print("="*70)

base_params = load_base_params()

# Test R0 values
R0_targets = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]

strategies = {
    'vary_contact': vary_contact_rate,
    'vary_trans_prob': vary_transmission_prob,
    'balanced': vary_both_balanced
}

results_data = []

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for col_idx, (strategy_name, strategy_func) in enumerate(strategies.items()):
    print(f"\n{strategy_name.upper()}")
    print("-" * 40)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(R0_targets)))
    
    for idx, R0_target in enumerate(R0_targets):
        # Get parameters for this strategy and R0
        params = strategy_func(base_params, R0_target)
        
        # Verify actual R0
        R0_low_actual, R0_high_actual = compute_R0(params)
        
        # Run simulation
        sim = run_simulation(params, days=200)
        
        # Store results
        results_data.append({
            'strategy': strategy_name,
            'R0_target': R0_target,
            'R0_low_actual': R0_low_actual,
            'R0_high_actual': R0_high_actual,
            'contact_rate': params['contact_rate'],
            'transmission_probability': params['transmission_probability'],
            'beta_l': params['contact_rate'] * params['transmission_probability'],
            'peak_I_high': sim['I_high'].max(),
            'peak_I_low': sim['I_low'].max(),
            'peak_I_total': sim['I_total'].max(),
            'final_R_high': sim['Rh'][-1],
            'final_R_low': sim['Rl'][-1]
        })
        
        # Plot dynamics
        axes[0, col_idx].plot(sim['t'], sim['I_high'], color=colors[idx], 
                             linewidth=2, label=f'R0={R0_target:.1f}')
        axes[1, col_idx].plot(sim['t'], sim['I_low'], color=colors[idx], 
                             linewidth=2, label=f'R0={R0_target:.1f}')
        
        print(f"  R0={R0_target:.1f}: c={params['contact_rate']:.2f}, "
              f"p={params['transmission_probability']:.4f}, "
              f"actual R0_low={R0_low_actual:.3f}")
    
    # Format plots
    axes[0, col_idx].set_title(f'{strategy_name}\nHigh-virulence infections', fontsize=11)
    axes[0, col_idx].set_xlabel('Days')
    axes[0, col_idx].set_ylabel('Proportion infected (high)')
    axes[0, col_idx].legend(fontsize=8)
    axes[0, col_idx].grid(alpha=0.3)
    
    axes[1, col_idx].set_title(f'Low-virulence infections', fontsize=11)
    axes[1, col_idx].set_xlabel('Days')
    axes[1, col_idx].set_ylabel('Proportion infected (low)')
    axes[1, col_idx].legend(fontsize=8)
    axes[1, col_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../Figures/contact_vs_transmission_comparison_v2.png', dpi=600)
print("\nSaved: ../Figures/contact_vs_transmission_comparison_v2.png")
plt.close()

# Save results
df = pd.DataFrame(results_data)
df.to_csv('../Tables/contact_vs_transmission_comparison_v2.csv', index=False)
print("Saved: ../Tables/contact_vs_transmission_comparison_v2.csv")

# Summary comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for strategy_name in strategies.keys():
    df_strat = df[df['strategy'] == strategy_name]
    
    axes[0, 0].plot(df_strat['R0_target'], df_strat['contact_rate'], 
                   'o-', label=strategy_name, linewidth=2, markersize=6)
    axes[0, 1].plot(df_strat['R0_target'], df_strat['transmission_probability'], 
                   'o-', label=strategy_name, linewidth=2, markersize=6)
    axes[1, 0].plot(df_strat['R0_target'], df_strat['peak_I_high'], 
                   'o-', label=strategy_name, linewidth=2, markersize=6)
    axes[1, 1].plot(df_strat['R0_target'], df_strat['peak_I_low'], 
                   'o-', label=strategy_name, linewidth=2, markersize=6)

axes[0, 0].set_xlabel('Target R0')
axes[0, 0].set_ylabel('Contact rate')
axes[0, 0].set_title('Contact Rate vs R0')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_xlabel('Target R0')
axes[0, 1].set_ylabel('Transmission probability')
axes[0, 1].set_title('Transmission Probability vs R0')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].set_xlabel('Target R0')
axes[1, 0].set_ylabel('Peak infection proportion')
axes[1, 0].set_title('High-Virulence Peak vs R0')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].set_xlabel('Target R0')
axes[1, 1].set_ylabel('Peak infection proportion')
axes[1, 1].set_title('Low-Virulence Peak vs R0')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../Figures/parameter_strategy_summary_v2.png', dpi=600)
print("Saved: ../Figures/parameter_strategy_summary_v2.png")
plt.close()

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)