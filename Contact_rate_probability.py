import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Models.SEIRS_Models import SEIRS_model_v2
from Models import params as model_params

np.random.seed(42)

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Base parameters (loaded from params.py with fallbacks)
BASE_PARAMS = {
    'contact_rate': getattr(model_params, 'contact_rate', 10.0),
    'transmission_probability': getattr(model_params, 'transmission_probability', 0.025),
    'birth_rate': getattr(model_params, 'birth_rate', 0.0),
    'death_rate': getattr(model_params, 'death_rate', 0.0),
    'delta': getattr(model_params, 'delta', 1/90),
    'kappa_base': getattr(model_params, 'kappa_base', 1.0),
    'kappa_scale': getattr(model_params, 'kappa_scale', 1.0),
    'p_recover': getattr(model_params, 'p_recover', 0.5),
    'phi_recover': getattr(model_params, 'phi_recover', 1.0),
    'phi_transmission': getattr(model_params, 'phi_transmission', 1.05),
    'sigma': getattr(model_params, 'sigma', 1/10),
    'tau': getattr(model_params, 'tau', 1/3),
    'theta': getattr(model_params, 'theta', 0.3)
}

# Initial conditions (proportions)
INITIAL_CONDITIONS = np.array([
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
INITIAL_CONDITIONS = INITIAL_CONDITIONS / INITIAL_CONDITIONS.sum()

# Time grid
T_MAX = getattr(model_params, 't_max', 365)
T_STEPS = int(getattr(model_params, 't_steps', 365))
TIME_GRID = np.linspace(0, T_MAX, T_STEPS)

# Output directories
os.makedirs('./Figures', exist_ok=True)
os.makedirs('./Tables', exist_ok=True)

# ============================================================================
# R0-TARGETED PARAMETER COMBINATIONS
# ============================================================================

# Target R0 values for low-virulence strain
R0_TARGETS = np.linspace(0.5, 3.0, 11)

# Virulence multipliers to test
PHI_TRANSMISSION_VALUES = [1.0, 1.05, 1.1, 1.15, 1.2]

# Different parameter combination strategies
COMBINATION_STRATEGIES = {
    'vary_contact_rate': {
        'description': 'Fix transmission_probability, vary contact_rate',
        'fixed_param': 'transmission_probability',
        'fixed_value': 0.025,
        'vary_param': 'contact_rate'
    },
    'vary_transmission_prob': {
        'description': 'Fix contact_rate, vary transmission_probability',
        'fixed_param': 'contact_rate',
        'fixed_value': 10.0,
        'vary_param': 'transmission_probability'
    },
    'balanced': {
        'description': 'Vary both proportionally (keep ratio constant)',
        'fixed_param': None,
        'fixed_value': None,
        'vary_param': 'both'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_params_tuple(param_dict):
    """Convert parameter dictionary to tuple for ODE solver."""
    return (
        param_dict['contact_rate'],
        param_dict['transmission_probability'],
        param_dict['birth_rate'],
        param_dict['death_rate'],
        param_dict['delta'],
        param_dict['kappa_base'],
        param_dict['kappa_scale'],
        param_dict['p_recover'],
        param_dict['phi_recover'],
        param_dict['phi_transmission'],
        param_dict['sigma'],
        param_dict['tau'],
        param_dict['theta']
    )

def compute_params_for_R0(R0_target, sigma, theta, p_recover, kappa_base, strategy_key):
    """
    Compute contact_rate and transmission_probability to achieve target R0.
    
    R0_low ≈ beta_l / sigma_eff
    where beta_l = contact_rate * transmission_probability
    and sigma_eff accounts for treatment effect
    """
    strategy = COMBINATION_STRATEGIES[strategy_key]
    
    # Effective recovery rate (accounting for treatment)
    theta_eff_low = kappa_base * theta
    sigma_eff = sigma * (1 - theta_eff_low + p_recover * theta_eff_low)
    
    # Required beta_l to achieve R0_target
    beta_l_target = R0_target * sigma_eff
    
    if strategy['vary_param'] == 'contact_rate':
        # Fix transmission probability, solve for contact rate
        trans_prob = strategy['fixed_value']
        contact_rate = beta_l_target / trans_prob
        return contact_rate, trans_prob
    
    elif strategy['vary_param'] == 'transmission_probability':
        # Fix contact rate, solve for transmission probability
        contact_rate = strategy['fixed_value']
        trans_prob = beta_l_target / contact_rate
        return contact_rate, trans_prob
    
    elif strategy['vary_param'] == 'both':
        # Keep baseline ratio, scale both proportionally
        baseline_contact = BASE_PARAMS['contact_rate']
        baseline_trans = BASE_PARAMS['transmission_probability']
        baseline_beta = baseline_contact * baseline_trans
        
        scale_factor = beta_l_target / baseline_beta
        contact_rate = baseline_contact * np.sqrt(scale_factor)
        trans_prob = baseline_trans * np.sqrt(scale_factor)
        return contact_rate, trans_prob

def run_simulation(params_dict, y0, t):
    """Run single simulation with given parameters."""
    params_tuple = build_params_tuple(params_dict)
    sol = odeint(SEIRS_model_v2, y0, t, args=(params_tuple,))
    
    compartments = {
        'S': sol[:, 0],
        'Eh': sol[:, 1],
        'Indh': sol[:, 2],
        'Idh': sol[:, 3],
        'Rh': sol[:, 4],
        'El': sol[:, 5],
        'Indl': sol[:, 6],
        'Idl': sol[:, 7],
        'Rl': sol[:, 8]
    }
    
    compartments['I_high'] = compartments['Indh'] + compartments['Idh']
    compartments['I_low'] = compartments['Indl'] + compartments['Idl']
    compartments['I_total'] = compartments['I_high'] + compartments['I_low']
    
    return compartments

def compute_actual_R0(params_dict):
    """Compute actual R0 values from parameters."""
    beta_l = params_dict['contact_rate'] * params_dict['transmission_probability']
    beta_h = params_dict['phi_transmission'] * beta_l
    
    theta = params_dict['theta']
    p_recover = params_dict['p_recover']
    sigma = params_dict['sigma']
    phi_recover = params_dict['phi_recover']
    kappa_base = params_dict['kappa_base']
    kappa_scale = params_dict['kappa_scale']
    
    # Compute kappa for both strains
    virulence_excess = params_dict['phi_transmission'] - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * virulence_excess)
    kappa_low = kappa_base
    
    # Safety: ensure kappa * theta ≤ 1
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)
    
    theta_eff_high = kappa_high * theta
    theta_eff_low = kappa_low * theta
    
    beta_eff_high = beta_h * (1 - theta_eff_high + p_recover * theta_eff_high)
    beta_eff_low = beta_l * (1 - theta_eff_low + p_recover * theta_eff_low)
    
    sigma_eff_high = phi_recover * sigma
    sigma_eff_low = sigma
    
    R0_high = beta_eff_high / sigma_eff_high
    R0_low = beta_eff_low / sigma_eff_low
    
    return R0_low, R0_high

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_R0_phi_sweep(strategy_key):
    """Run sweep over R0 targets and phi_transmission values."""
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy_key}")
    print(f"Description: {COMBINATION_STRATEGIES[strategy_key]['description']}")
    print(f"{'='*70}")
    
    results_data = []
    
    for phi_trans in PHI_TRANSMISSION_VALUES:
        print(f"\n  phi_transmission = {phi_trans:.2f}")
        
        for R0_target in R0_TARGETS:
            # Build parameter dictionary
            params_dict = BASE_PARAMS.copy()
            params_dict['phi_transmission'] = phi_trans
            
            # Compute contact_rate and transmission_probability for target R0
            contact_rate, trans_prob = compute_params_for_R0(
                R0_target,
                params_dict['sigma'],
                params_dict['theta'],
                params_dict['p_recover'],
                params_dict['kappa_base'],
                strategy_key
            )
            
            params_dict['contact_rate'] = contact_rate
            params_dict['transmission_probability'] = trans_prob
            
            # Verify actual R0
            R0_low_actual, R0_high_actual = compute_actual_R0(params_dict)
            
            # Run simulation
            compartments = run_simulation(params_dict, INITIAL_CONDITIONS, TIME_GRID)
            
            # Store results
            results_data.append({
                'strategy': strategy_key,
                'R0_target': R0_target,
                'R0_low_actual': R0_low_actual,
                'R0_high_actual': R0_high_actual,
                'phi_transmission': phi_trans,
                'contact_rate': contact_rate,
                'transmission_probability': trans_prob,
                'beta_l': contact_rate * trans_prob,
                'beta_h': contact_rate * trans_prob * phi_trans,
                'Eh_peak': compartments['Eh'].max(),
                'El_peak': compartments['El'].max(),
                'Ih_peak': compartments['I_high'].max(),
                'Il_peak': compartments['I_low'].max(),
                'Eh_final': compartments['Eh'][-1],
                'El_final': compartments['El'][-1],
                'dominance_ratio': compartments['Eh'].max() / (compartments['El'].max() + 1e-10)
            })
            
            print(f"    R0_target={R0_target:.2f} → R0_low={R0_low_actual:.3f}, "
                  f"R0_high={R0_high_actual:.3f}, "
                  f"c={contact_rate:.2f}, p={trans_prob:.4f}")
    
    return pd.DataFrame(results_data)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_R0_phi_heatmaps(df, strategy_key):
    """Create heatmaps showing outcomes across R0 and phi_transmission."""
    
    # Pivot data for heatmaps
    outcomes = ['Eh_peak', 'El_peak', 'dominance_ratio']
    n_outcomes = len(outcomes)
    
    fig, axes = plt.subplots(1, n_outcomes, figsize=(7*n_outcomes, 6))
    
    for idx, outcome in enumerate(outcomes):
        pivot = df.pivot(index='phi_transmission', columns='R0_target', values=outcome)
        
        im = axes[idx].imshow(pivot, aspect='auto', cmap='YlOrRd', origin='lower')
        axes[idx].set_xlabel('R0 target (low strain)')
        axes[idx].set_ylabel('phi_transmission')
        axes[idx].set_title(f'{outcome}\n({strategy_key})')
        
        # Set ticks
        x_ticks = np.arange(len(pivot.columns))
        y_ticks = np.arange(len(pivot.index))
        axes[idx].set_xticks(x_ticks[::2])
        axes[idx].set_xticklabels([f'{pivot.columns[i]:.2f}' for i in x_ticks[::2]])
        axes[idx].set_yticks(y_ticks)
        axes[idx].set_yticklabels([f'{pivot.index[i]:.2f}' for i in y_ticks])
        
        plt.colorbar(im, ax=axes[idx], label=outcome)
    
    plt.tight_layout()
    plt.savefig(f'./Figures/R0_phi_heatmap_{strategy_key}.png', dpi=300)
    print(f"Saved: ./Figures/R0_phi_heatmap_{strategy_key}.png")
    plt.close()

def plot_parameter_relationships(df, strategy_key):
    """Plot how contact_rate and transmission_probability vary with R0."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for phi_trans in PHI_TRANSMISSION_VALUES:
        df_phi = df[df['phi_transmission'] == phi_trans]
        
        # Contact rate vs R0
        axes[0, 0].plot(df_phi['R0_target'], df_phi['contact_rate'], 
                       marker='o', label=f'φ={phi_trans:.2f}')
        
        # Transmission probability vs R0
        axes[0, 1].plot(df_phi['R0_target'], df_phi['transmission_probability'], 
                       marker='o', label=f'φ={phi_trans:.2f}')
        
        # Beta_l vs R0
        axes[1, 0].plot(df_phi['R0_target'], df_phi['beta_l'], 
                       marker='o', label=f'φ={phi_trans:.2f}')
        
        # Beta_h vs R0
        axes[1, 1].plot(df_phi['R0_target'], df_phi['beta_h'], 
                       marker='o', label=f'φ={phi_trans:.2f}')
    
    axes[0, 0].set_xlabel('R0 target')
    axes[0, 0].set_ylabel('Contact rate')
    axes[0, 0].set_title(f'Contact Rate vs R0\n({strategy_key})')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_xlabel('R0 target')
    axes[0, 1].set_ylabel('Transmission probability')
    axes[0, 1].set_title(f'Transmission Probability vs R0\n({strategy_key})')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].set_xlabel('R0 target')
    axes[1, 0].set_ylabel('β_low')
    axes[1, 0].set_title(f'β_low vs R0\n({strategy_key})')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_xlabel('R0 target')
    axes[1, 1].set_ylabel('β_high')
    axes[1, 1].set_title(f'β_high vs R0\n({strategy_key})')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./Figures/parameter_relationships_{strategy_key}.png', dpi=300)
    print(f"Saved: ./Figures/parameter_relationships_{strategy_key}.png")
    plt.close()

# ============================================================================
# RUN ALL STRATEGIES
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("R0-Targeted Parameter Sweep with Virulence Variations")
    print("="*70)
    
    all_results = []
    
    for strategy_key in COMBINATION_STRATEGIES.keys():
        # Run sweep
        df = run_R0_phi_sweep(strategy_key)
        all_results.append(df)
        
        # Save data
        df.to_csv(f'./Tables/R0_phi_sweep_{strategy_key}.csv', index=False)
        print(f"Saved: ./Tables/R0_phi_sweep_{strategy_key}.csv")
        
        # Create visualizations
        plot_R0_phi_heatmaps(df, strategy_key)
        plot_parameter_relationships(df, strategy_key)
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv('./Tables/R0_phi_sweep_all_strategies.csv', index=False)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - Figures/R0_phi_heatmap_*.png")
    print("  - Figures/parameter_relationships_*.png")
    print("  - Tables/R0_phi_sweep_*.csv")