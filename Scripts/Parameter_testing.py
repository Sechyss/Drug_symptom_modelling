# filepath: /home/albertotr/PycharmProjects/MGall_Analysis/Scripts/MG_modelling/Parameter_testing.py
"""
Generic parameter sensitivity analysis for SEIRS virulence-transmission trade-off model.

This script provides a flexible framework for testing any parameter combination without
hardcoded sweeps. Configure your experiments in the PARAMETER_EXPERIMENTS dictionary below.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from Models.SEIRS_Models import SEIRS_model_v2
from Models import params as model_params

np.random.seed(42)

# ============================================================================
# CONFIGURATION SECTION - Edit this to customize your experiments
# ============================================================================

# Base parameters (loaded from params.py with fallbacks)
BASE_PARAMS = {
    'contact_rate': getattr(model_params, 'contact_rate', 10.0),
    'transmission_probability': getattr(model_params, 'transmission_probability', 0.025),
    'birth_rate': getattr(model_params, 'birth_rate', 0.0),
    'death_rate': getattr(model_params, 'death_rate', 0.0),
    'delta': getattr(model_params, 'delta', 1/90),
    'delta_d': getattr(model_params, 'delta_d', 1/3),
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
os.makedirs('../Figures', exist_ok=True)
os.makedirs('../Tables', exist_ok=True)

# ============================================================================
# DEFINE YOUR EXPERIMENTS HERE
# ============================================================================
# Each experiment is a dictionary with:
#   - 'name': short identifier for filenames
#   - 'param_name': parameter to sweep (must match key in BASE_PARAMS)
#   - 'param_values': array of values to test
#   - 'outputs': list of compartments to plot (e.g., ['Eh', 'El', 'Indh'])
#   - 'plot_type': '1D' (line plots) or '2D' (heatmap, requires second parameter)
#   - 'secondary_param': (optional) for 2D sweeps
#   - 'secondary_values': (optional) for 2D sweeps
#   - 'R0_sweep': (optional) if True, converts param_values as R0 targets

PARAMETER_EXPERIMENTS = [
    # Example 1: Sweep treatment coverage (theta)
    {
        'name': 'theta_sweep',
        'param_name': 'theta',
        'param_values': np.linspace(0.0, 1.0, 11),
        'outputs': ['Eh', 'El'],
        'plot_type': '1D'
    },
    
    # Example 2: Sweep treatment efficacy (p_recover)
    {
        'name': 'p_recover_sweep',
        'param_name': 'p_recover',
        'param_values': np.linspace(1.0, 2.0, 11),
        'outputs': ['Eh', 'El', 'Indh', 'Indl'],
        'plot_type': '1D'
    },
    
    # Example 3: 2D heatmap of theta × p_recover
    {
        'name': 'theta_p_recover_heatmap',
        'param_name': 'theta',
        'param_values': np.linspace(0.0, 1.0, 21),
        'secondary_param': 'p_recover',
        'secondary_values': np.linspace(1.0, 2.0, 21),
        'outputs': ['Eh_peak', 'El_peak'],  # special: peak values
        'plot_type': '2D'
    },
    
    # Example 4: R0 sweep (special handling)
    {
        'name': 'R0_sweep',
        'param_name': 'sigma',  # will be computed from R0 target
        'param_values': np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        'outputs': ['Eh', 'El'],
        'plot_type': '1D',
        'R0_sweep': True  # special flag
    },
    
    # Example 5: Virulence transmission advantage
    {
        'name': 'phi_transmission_sweep',
        'param_name': 'phi_transmission',
        'param_values': np.linspace(1.0, 1.2, 11),
        'outputs': ['Eh', 'El'],
        'plot_type': '1D'
    },
    
    # Example 6: Immunity duration (delta)
    {
        'name': 'immunity_duration_sweep',
        'param_name': 'delta',
        'param_values': 1 / np.array([30, 60, 90, 180, 365]),  # convert days to rate
        'outputs': ['Eh', 'El', 'Rh', 'Rl'],
        'plot_type': '1D'
    }
]

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
        param_dict['delta_d'],
        param_dict['p_recover'],
        param_dict['phi_recover'],
        param_dict['phi_transmission'],
        param_dict['sigma'],
        param_dict['tau'],
        param_dict['theta']
    )

def run_simulation(params_dict, y0, t):
    """Run single simulation with given parameters."""
    params_tuple = build_params_tuple(params_dict)
    sol = odeint(SEIRS_model_v2, y0, t, args=(params_tuple,))
    
    # Extract compartments
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
    
    # Compute derived quantities
    compartments['I_high'] = compartments['Indh'] + compartments['Idh']
    compartments['I_low'] = compartments['Indl'] + compartments['Idl']
    compartments['I_total'] = compartments['I_high'] + compartments['I_low']
    
    return compartments

def compute_R0_from_params(params_dict):
    """Compute R0 for both strains given parameters."""
    beta_l = params_dict['contact_rate'] * params_dict['transmission_probability']
    beta_h = params_dict['phi_transmission'] * beta_l
    
    # Get parameters
    theta = params_dict['theta']
    p_recover = params_dict['p_recover']  # transmission reduction for treated
    sigma = params_dict['sigma']
    phi_recover = params_dict['phi_recover']
    
    # Effective transmission rates (weighted by treatment fraction)
    # beta_eff = beta * (fraction_untreated + p_recover * fraction_treated)
    beta_eff_high = beta_h * (1 - theta + p_recover * theta)
    beta_eff_low = beta_l * (1 - theta + p_recover * theta)
    
    # Effective recovery rates (phi_recover only affects high strain)
    sigma_eff_high = phi_recover * sigma
    sigma_eff_low = sigma
    
    R0_high = beta_eff_high / sigma_eff_high
    R0_low = beta_eff_low / sigma_eff_low
    
    return R0_low, R0_high

def beta_from_R0(R0_target, sigma):
    """Convert R0 target to beta (for R0 sweeps)."""
    return R0_target * sigma

# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def run_1D_sweep(experiment):
    """Run and plot 1D parameter sweep."""
    print(f"\n{'='*60}")
    print(f"Running 1D sweep: {experiment['name']}")
    print(f"{'='*60}")
    
    param_name = experiment['param_name']
    param_values = experiment['param_values']
    outputs = experiment['outputs']
    is_R0_sweep = experiment.get('R0_sweep', False)
    
    # Storage for results
    results = {out: [] for out in outputs}
    metadata = []
    
    for idx, param_val in enumerate(param_values):
        # Build parameter dictionary
        params_dict = BASE_PARAMS.copy()
        
        if is_R0_sweep:
            # Special handling: convert R0 to beta
            beta_l = beta_from_R0(param_val, params_dict['sigma'])
            params_dict['contact_rate'] = beta_l / params_dict['transmission_probability']
            R0_target = param_val
        else:
            params_dict[param_name] = param_val
            R0_target = None
        
        # Run simulation
        compartments = run_simulation(params_dict, INITIAL_CONDITIONS, TIME_GRID)
        
        # Store requested outputs
        for out in outputs:
            results[out].append(compartments[out])
        
        # Compute R0 values
        R0_low, R0_high = compute_R0_from_params(params_dict)
        
        metadata.append({
            'param_value': param_val,
            'R0_low': R0_low,
            'R0_high': R0_high,
            'R0_target': R0_target
        })
        
        # Diagnostic output
        print(f"[{idx+1}/{len(param_values)}] {param_name}={param_val:.4f}  "
              f"R0_low={R0_low:.3f}  R0_high={R0_high:.3f}")
    
    # Convert to arrays
    for out in outputs:
        results[out] = np.array(results[out])
    
    # Plotting
    n_outputs = len(outputs)
    fig, axes = plt.subplots(1, n_outputs, figsize=(6*n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    for ax_idx, out in enumerate(outputs):
        ax = axes[ax_idx]
        for i, param_val in enumerate(param_values):
            label = f"{param_name}={param_val:.3f}"
            if is_R0_sweep:
                label = f"R0={param_val:.2f}"
            ax.plot(TIME_GRID, results[out][i, :], color=colors[i], label=label)
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel(out)
        ax.set_title(f'{out} — {experiment["name"]}')
        ax.legend(fontsize='x-small', loc='best')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig_path = f'./Figures/{experiment["name"]}.png'
    plt.savefig(fig_path, dpi=600)
    print(f"Saved figure: {fig_path}")
    plt.show()
    
    # Save data
    df_list = []
    for i, param_val in enumerate(param_values):
        df_tmp = pd.DataFrame({'time': TIME_GRID})
        for out in outputs:
            df_tmp[out] = results[out][i, :]
        df_tmp['param_name'] = param_name
        df_tmp['param_value'] = param_val
        df_tmp.update(metadata[i])
        df_list.append(df_tmp)
    
    df = pd.concat(df_list, ignore_index=True)
    csv_path = f'./Tables/{experiment["name"]}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved data: {csv_path}")

def run_2D_sweep(experiment):
    """Run and plot 2D parameter sweep (heatmap)."""
    print(f"\n{'='*60}")
    print(f"Running 2D sweep: {experiment['name']}")
    print(f"{'='*60}")
    
    param1_name = experiment['param_name']
    param1_values = experiment['param_values']
    param2_name = experiment['secondary_param']
    param2_values = experiment['secondary_values']
    outputs = experiment['outputs']
    
    # Storage for peak/summary values
    results = {out: np.zeros((len(param1_values), len(param2_values))) 
               for out in outputs}
    
    for i, param1_val in enumerate(param1_values):
        for j, param2_val in enumerate(param2_values):
            # Build parameter dictionary
            params_dict = BASE_PARAMS.copy()
            params_dict[param1_name] = param1_val
            params_dict[param2_name] = param2_val
            
            # Run simulation
            compartments = run_simulation(params_dict, INITIAL_CONDITIONS, TIME_GRID)
            
            # Store peak values for requested outputs
            for out in outputs:
                if '_peak' in out:
                    base_out = out.replace('_peak', '')
                    results[out][i, j] = compartments[base_out].max()
                elif '_final' in out:
                    base_out = out.replace('_final', '')
                    results[out][i, j] = compartments[base_out][-1]
                else:
                    results[out][i, j] = compartments[out].max()
            
            print(f"[{i+1}/{len(param1_values)}, {j+1}/{len(param2_values)}] "
                  f"{param1_name}={param1_val:.3f}, {param2_name}={param2_val:.3f}")
    
    # Plotting
    n_outputs = len(outputs)
    fig, axes = plt.subplots(1, n_outputs, figsize=(7*n_outputs, 6))
    if n_outputs == 1:
        axes = [axes]
    
    for ax_idx, out in enumerate(outputs):
        ax = axes[ax_idx]
        im = ax.imshow(results[out], aspect='auto', cmap='YlOrRd', origin='lower')
        ax.set_xlabel(f'{param2_name}')
        ax.set_ylabel(f'{param1_name}')
        ax.set_title(f'{out} — {experiment["name"]}')
        
        # Set ticks
        x_idx = np.linspace(0, len(param2_values)-1, 6).round().astype(int)
        y_idx = np.linspace(0, len(param1_values)-1, 6).round().astype(int)
        ax.set_xticks(x_idx)
        ax.set_xticklabels([f'{param2_values[k]:.2f}' for k in x_idx])
        ax.set_yticks(y_idx)
        ax.set_yticklabels([f'{param1_values[k]:.2f}' for k in y_idx])
        
        plt.colorbar(im, ax=ax, label=out)
    
    plt.tight_layout()
    fig_path = f'./Figures/{experiment["name"]}.png'
    plt.savefig(fig_path, dpi=600)
    print(f"Saved figure: {fig_path}")
    plt.show()
    
    # Save data
    df_list = []
    for i, param1_val in enumerate(param1_values):
        for j, param2_val in enumerate(param2_values):
            row = {
                param1_name: param1_val,
                param2_name: param2_val
            }
            for out in outputs:
                row[out] = results[out][i, j]
            df_list.append(row)
    
    df = pd.DataFrame(df_list)
    csv_path = f'./Tables/{experiment["name"]}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved data: {csv_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SEIRS Parameter Testing Framework")
    print("="*60)
    print(f"Found {len(PARAMETER_EXPERIMENTS)} experiments to run")
    
    for exp in PARAMETER_EXPERIMENTS:
        if exp['plot_type'] == '1D':
            run_1D_sweep(exp)
        elif exp['plot_type'] == '2D':
            run_2D_sweep(exp)
        else:
            print(f"Unknown plot type: {exp['plot_type']}")
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)