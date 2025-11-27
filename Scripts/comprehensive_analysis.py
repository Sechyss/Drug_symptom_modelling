import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import sys

# Add project root to path (more robust method)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Models.SEIRS_Models import SEIRS_model_v2
from Models import params as default_params

#%% 1. BIFURCATION ANALYSIS
class BifurcationAnalyzer:
    """Find critical transitions in virulence dominance"""
    
    def __init__(self, model_func, t_max=365):
        self.model = model_func
        self.t = np.linspace(0, t_max, t_max)
        
    def run_to_equilibrium(self, y0, params, tol=1e-6):
        """Run until steady state or epidemic burnout"""
        solution = odeint(self.model, y0, self.t, args=(params,))
        
        # Check if reached equilibrium
        final_state = solution[-1]
        return final_state
    
    def compute_virulence_fraction(self, state):
        """Calculate proportion of high-virulence infections"""
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = state
        
        total_high = Indh + Idh
        total_low = Indl + Idl
        total_infected = total_high + total_low
        
        if total_infected < 1e-6:
            return np.nan
        
        return total_high / total_infected
    
    def phase_diagram(self, theta_range, kappa_scale_range, 
                      save_path='phase_diagram.png'):
        """
        2D bifurcation diagram: treatment coverage vs detection sensitivity
        """
        results = np.zeros((len(theta_range), len(kappa_scale_range)))
        
        # Initial conditions
        y0 = [default_params.S, 
              default_params.Eh, default_params.Indh, default_params.Idh, default_params.Rh,
              default_params.El, default_params.Indl, default_params.Idl, default_params.Rl]
        
        for i, theta in enumerate(theta_range):
            for j, kappa_scale in enumerate(kappa_scale_range):
                # Build parameter vector - 13 parameters for SEIRS_model_v2
                # (beta_l, birth_rate, death_rate, delta, delta_d_high, delta_d_low, 
                #  p_recover, phi_recover, phi_transmission, sigma, tau, kappa_scale, theta)
                kappa_high = default_params.kappa_base * (1 + kappa_scale * (default_params.phi_transmission - 1))
                delta_d_high = kappa_high / default_params.tau
                delta_d_low = default_params.kappa_base / default_params.tau
                
                params_list = (
                    default_params.beta_l,
                    default_params.birth_rate,
                    default_params.death_rate,
                    default_params.delta,
                    delta_d_high,
                    delta_d_low,
                    default_params.p_recover,
                    default_params.phi_recover,
                    default_params.phi_transmission,
                    default_params.sigma,
                    default_params.tau,
                    kappa_scale,
                    theta
                )
                
                final_state = self.run_to_equilibrium(y0, params_list)
                results[i, j] = self.compute_virulence_fraction(final_state)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.contourf(kappa_scale_range, theta_range, results, 
                         levels=20, cmap='RdYlGn_r')
        
        # Add contour line at 0.5 (equal prevalence)
        cs = ax.contour(kappa_scale_range, theta_range, results, 
                        levels=[0.5], colors='black', linewidths=2)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Equal prevalence')
        
        plt.colorbar(im, ax=ax, label='High-virulence fraction')
        ax.set_xlabel('Detection sensitivity (kappa_scale)', fontsize=12)
        ax.set_ylabel('Treatment coverage (θ)', fontsize=12)
        ax.set_title('Virulence Dominance Phase Diagram', fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def find_critical_thresholds(self, theta_range, kappa_scale=1.0):
        """Find theta values where virulence dominance switches"""
        virulence_fractions = []
        
        y0 = [default_params.S, 
              default_params.Eh, default_params.Indh, default_params.Idh, default_params.Rh,
              default_params.El, default_params.Indl, default_params.Idl, default_params.Rl]
        
        for theta in theta_range:
            kappa_high = default_params.kappa_base * (1 + kappa_scale * (default_params.phi_transmission - 1))
            delta_d_high = kappa_high / default_params.tau
            delta_d_low = default_params.kappa_base / default_params.tau
            
            params_list = (
                default_params.beta_l,
                default_params.birth_rate,
                default_params.death_rate,
                default_params.delta,
                delta_d_high,
                delta_d_low,
                default_params.p_recover,
                default_params.phi_recover,
                default_params.phi_transmission,
                default_params.sigma,
                default_params.tau,
                kappa_scale,
                theta
            )
            
            final_state = self.run_to_equilibrium(y0, params_list)
            vf = self.compute_virulence_fraction(final_state)
            virulence_fractions.append(vf)
        
        # Find local minima/maxima
        vf_array = np.array(virulence_fractions)
        dv = np.gradient(vf_array)
        
        critical_points = []
        for i in range(1, len(dv)-1):
            if dv[i-1] * dv[i+1] < 0:  # Sign change
                critical_points.append({
                    'theta': theta_range[i],
                    'virulence_fraction': vf_array[i],
                    'type': 'minimum' if dv[i-1] < 0 else 'maximum'
                })
        
        return critical_points, virulence_fractions

#%% 2. ELASTICITY ANALYSIS
class ElasticityAnalyzer:
    """Quantify parameter sensitivity using elasticity coefficients"""
    
    @staticmethod
    def compute_R0(params_dict):
        """
        Compute basic reproduction number for high-virulence strain
        R0 = beta / (sigma + delta_d*theta*kappa)
        """
        beta_h = params_dict['phi_transmission'] * params_dict['beta_l']
        
        # Detection-adjusted removal rate
        kappa_high = params_dict['kappa_base'] * (1 + params_dict['kappa_scale'] * 
                                                    (params_dict['phi_transmission'] - 1))
        delta_d_high = kappa_high / (1/3)
        
        removal_rate = params_dict['sigma'] + delta_d_high * params_dict['theta']
        
        R0_high = beta_h / removal_rate
        return R0_high
    
    def elasticity_matrix(self, baseline_params, param_names, delta=0.01):
        """
        Compute elasticity for all parameters
        Elasticity = (dR0/R0) / (dp/p)
        """
        baseline_R0 = self.compute_R0(baseline_params)
        elasticities = {}
        
        for param_name in param_names:
            if param_name not in baseline_params:
                continue
            
            # Perturb parameter
            perturbed_params = baseline_params.copy()
            original_value = perturbed_params[param_name]
            
            if original_value == 0:
                # For zero parameters, use absolute change
                perturbed_params[param_name] = delta
                new_R0 = self.compute_R0(perturbed_params)
                elasticities[param_name] = (new_R0 - baseline_R0) / baseline_R0 / delta
            else:
                perturbed_params[param_name] = original_value * (1 + delta)
                new_R0 = self.compute_R0(perturbed_params)
                elasticities[param_name] = ((new_R0 - baseline_R0) / baseline_R0) / delta
        
        return elasticities
    
    def plot_elasticities(self, elasticities, save_path='elasticity_analysis.png'):
        """Tornado plot of elasticities"""
        params = list(elasticities.keys())
        values = list(elasticities.values())
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(values))[::-1]
        params = [params[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Color code
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(params, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Elasticity (R₀ sensitivity)', fontsize=12)
        ax.set_title('Parameter Elasticity Analysis\n(Positive = increases R₀)', 
                     fontsize=14, weight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def sensitivity_at_critical_points(self, critical_thetas, param_ranges):
        """Compare elasticities at different treatment coverage levels"""
        results = []
        
        for theta in critical_thetas:
            params_dict = {
                'beta_l': default_params.beta_l,
                'sigma': default_params.sigma,
                'phi_transmission': default_params.phi_transmission,
                'theta': theta,
                'kappa_base': default_params.kappa_base,
                'kappa_scale': default_params.kappa_scale,
                'p_recover': default_params.p_recover
            }
            
            elasticities = self.elasticity_matrix(params_dict, list(params_dict.keys()))
            elasticities['theta_level'] = theta
            results.append(elasticities)
        
        df = pd.DataFrame(results)
        
        # Plot
        df_plot = df.drop('theta_level', axis=1)
        df_plot.index = df['theta_level']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot.T.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Elasticity', fontsize=12)
        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_title('Elasticity Variation Across Treatment Coverage Levels', 
                     fontsize=14, weight='bold')
        ax.legend(title='θ', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('elasticity_by_theta.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df

#%% 3. INVASION ANALYSIS
class InvasionAnalyzer:
    """Test evolutionary stability using invasion fitness"""
    
    def __init__(self, model_func, t_max=365):
        self.model = model_func
        self.t = np.linspace(0, t_max, t_max)
    
    def find_equilibrium(self, resident_type, treatment_params):
        """
        Find equilibrium with only resident strain
        resident_type: 'high' or 'low'
        """
        if resident_type == 'high':
            y0 = [9990, 0, 5, 0, 0, 0, 0, 0, 0]
        else:
            y0 = [9990, 0, 0, 0, 0, 0, 5, 0, 0]
        
        solution = odeint(self.model, y0, self.t, args=(treatment_params,))
        return solution[-1]
    
    def invasion_fitness(self, resident_type, mutant_type, treatment_params):
        """
        Compute invasion R0 for mutant in resident equilibrium
        Returns: mutant R0 when resident is at equilibrium
        """
        # Get resident equilibrium
        eq_state = self.find_equilibrium(resident_type, treatment_params)
        S_eq = eq_state[0]
        N_total = np.sum(eq_state)
        
        # Extract parameters
        theta = treatment_params[-1]
        beta_l = treatment_params[0]
        beta_h = treatment_params[8] * beta_l
        sigma = treatment_params[9]
        
        # Compute mutant R0 in resident environment
        if mutant_type == 'high':
            beta_mutant = beta_h
            # Assume detection follows kappa scaling
            removal_mutant = sigma + treatment_params[4] * theta
        else:
            beta_mutant = beta_l
            removal_mutant = sigma + treatment_params[5] * theta
        
        R0_invasion = (beta_mutant * S_eq / N_total) / removal_mutant
        
        return R0_invasion
    
    def invasion_matrix(self, theta_range, kappa_scale=1.0):
        """
        Compute invasion fitness for all combinations
        Returns: DataFrame with invasion outcomes
        """
        results = []
        
        for theta in theta_range:
            # Build params
            kappa_high = default_params.kappa_base * (1 + kappa_scale * 
                                                      (default_params.phi_transmission - 1))
            delta_d_high = kappa_high / (1/3)
            delta_d_low = default_params.kappa_base / (1/3)
            
            params_list = [
                default_params.beta_l, default_params.birth_rate, default_params.death_rate,
                default_params.delta, delta_d_high, delta_d_low, default_params.p_recover,
                default_params.phi_recover, default_params.phi_transmission,
                default_params.sigma, default_params.tau, theta
            ]
            
            # Test both invasion scenarios
            R0_high_invades_low = self.invasion_fitness('low', 'high', params_list)
            R0_low_invades_high = self.invasion_fitness('high', 'low', params_list)
            
            results.append({
                'theta': theta,
                'high_invades_low': R0_high_invades_low > 1,
                'low_invades_high': R0_low_invades_high > 1,
                'R0_high_inv': R0_high_invades_low,
                'R0_low_inv': R0_low_invades_high,
                'ESS': 'high' if (R0_high_invades_low > 1 and R0_low_invades_high < 1) else
                       'low' if (R0_high_invades_low < 1 and R0_low_invades_high > 1) else
                       'coexistence' if (R0_high_invades_low > 1 and R0_low_invades_high > 1) else
                       'bistable'
            })
        
        return pd.DataFrame(results)
    
    def plot_invasion_diagram(self, invasion_df, save_path='invasion_analysis.png'):
        """Visualize invasion outcomes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Invasion R0s
        ax1.plot(invasion_df['theta'], invasion_df['R0_high_inv'], 
                'r-o', label='High invades Low', linewidth=2, markersize=6)
        ax1.plot(invasion_df['theta'], invasion_df['R0_low_inv'], 
                'b-s', label='Low invades High', linewidth=2, markersize=6)
        ax1.axhline(1, color='black', linestyle='--', linewidth=1, label='R₀=1')
        ax1.set_xlabel('Treatment coverage (θ)', fontsize=12)
        ax1.set_ylabel('Invasion R₀', fontsize=12)
        ax1.set_title('Invasion Fitness', fontsize=14, weight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Panel 2: ESS classification
        ess_colors = {'high': 'red', 'low': 'blue', 'coexistence': 'purple', 'bistable': 'gray'}
        for ess_type in invasion_df['ESS'].unique():
            subset = invasion_df[invasion_df['ESS'] == ess_type]
            ax2.scatter(subset['theta'], [1]*len(subset), 
                       c=ess_colors[ess_type], s=100, label=ess_type, alpha=0.7)
        
        ax2.set_xlabel('Treatment coverage (θ)', fontsize=12)
        ax2.set_yticks([])
        ax2.set_title('Evolutionary Stable Strategy (ESS)', fontsize=14, weight='bold')
        ax2.legend()
        ax2.set_ylim(0.5, 1.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

#%% 4. TRANSIENT DYNAMICS ANALYSIS
class TransientAnalyzer:
    """Characterize epidemic trajectories"""
    
    def __init__(self, model_func, t_max=365):
        self.model = model_func
        self.t = np.linspace(0, t_max, t_max)
    
    def extract_metrics(self, solution):
        """Compute key epidemic metrics"""
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = solution.T
        
        high_infections = Indh + Idh
        low_infections = Indl + Idl
        total_infections = high_infections + low_infections
        
        metrics = {}
        
        # Peak timing
        if np.max(high_infections) > 1e-6:
            metrics['peak_time_high'] = self.t[np.argmax(high_infections)]
            metrics['peak_size_high'] = np.max(high_infections)
        else:
            metrics['peak_time_high'] = np.nan
            metrics['peak_size_high'] = 0
        
        if np.max(low_infections) > 1e-6:
            metrics['peak_time_low'] = self.t[np.argmax(low_infections)]
            metrics['peak_size_low'] = np.max(low_infections)
        else:
            metrics['peak_time_low'] = np.nan
            metrics['peak_size_low'] = 0
        
        # Cumulative burden
        metrics['cumulative_high'] = np.trapz(high_infections, self.t)
        metrics['cumulative_low'] = np.trapz(low_infections, self.t)
        metrics['total_burden'] = np.trapz(total_infections, self.t)
        
        # Crossover point (if exists)
        crossover_idx = np.where(np.diff(np.sign(high_infections - low_infections)))[0]
        if len(crossover_idx) > 0:
            metrics['crossover_time'] = self.t[crossover_idx[0]]
            metrics['crossover_count'] = len(crossover_idx)
        else:
            metrics['crossover_time'] = np.nan
            metrics['crossover_count'] = 0
        
        # Final state
        metrics['final_high_fraction'] = high_infections[-1] / (total_infections[-1] + 1e-10)
        
        # Attack rate (total infected / population)
        N = 10000
        total_recovered = Rh[-1] + Rl[-1]
        metrics['attack_rate'] = total_recovered / N
        
        return metrics
    
    def compare_scenarios(self, theta_range, save_path='transient_comparison.png'):
        """Compare epidemic dynamics across treatment levels"""
        all_metrics = []
        
        y0 = [default_params.S, 
              default_params.Eh, default_params.Indh, default_params.Idh, default_params.Rh,
              default_params.El, default_params.Indl, default_params.Idl, default_params.Rl]
        
        for theta in theta_range:
            kappa_high = default_params.kappa_base * (1 + default_params.kappa_scale * 
                                                      (default_params.phi_transmission - 1))
            delta_d_high = kappa_high / (1/3)
            delta_d_low = default_params.kappa_base / (1/3)
            
            params_list = [
                default_params.beta_l, default_params.birth_rate, default_params.death_rate,
                default_params.delta, delta_d_high, delta_d_low, default_params.p_recover,
                default_params.phi_recover, default_params.phi_transmission,
                default_params.sigma, default_params.tau, theta
            ]
            
            solution = odeint(self.model, y0, self.t, args=(params_list,))
            metrics = self.extract_metrics(solution)
            metrics['theta'] = theta
            all_metrics.append(metrics)
        
        df = pd.DataFrame(all_metrics)
        
        # Plotting
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Peak timing
        axes[0, 0].plot(df['theta'], df['peak_time_high'], 'ro-', label='High-virulence')
        axes[0, 0].plot(df['theta'], df['peak_time_low'], 'bs-', label='Low-virulence')
        axes[0, 0].set_xlabel('θ')
        axes[0, 0].set_ylabel('Peak time (days)')
        axes[0, 0].legend()
        axes[0, 0].set_title('Epidemic Peak Timing')
        axes[0, 0].grid(alpha=0.3)
        
        # Peak size
        axes[0, 1].plot(df['theta'], df['peak_size_high'], 'ro-', label='High-virulence')
        axes[0, 1].plot(df['theta'], df['peak_size_low'], 'bs-', label='Low-virulence')
        axes[0, 1].set_xlabel('θ')
        axes[0, 1].set_ylabel('Peak prevalence')
        axes[0, 1].legend()
        axes[0, 1].set_title('Epidemic Peak Size')
        axes[0, 1].grid(alpha=0.3)
        
        # Cumulative burden
        axes[0, 2].plot(df['theta'], df['cumulative_high'], 'ro-', label='High-virulence')
        axes[0, 2].plot(df['theta'], df['cumulative_low'], 'bs-', label='Low-virulence')
        axes[0, 2].set_xlabel('θ')
        axes[0, 2].set_ylabel('Cumulative infections')
        axes[0, 2].legend()
        axes[0, 2].set_title('Total Disease Burden')
        axes[0, 2].grid(alpha=0.3)
        
        # Crossover timing
        axes[1, 0].scatter(df['theta'], df['crossover_time'], s=100, alpha=0.6)
        axes[1, 0].set_xlabel('θ')
        axes[1, 0].set_ylabel('Crossover time (days)')
        axes[1, 0].set_title('Strain Dominance Switch')
        axes[1, 0].grid(alpha=0.3)
        
        # Final fraction
        axes[1, 1].plot(df['theta'], df['final_high_fraction'], 'mo-', linewidth=2)
        axes[1, 1].axhline(0.5, color='gray', linestyle='--', label='Equal')
        axes[1, 1].set_xlabel('θ')
        axes[1, 1].set_ylabel('High-virulence fraction')
        axes[1, 1].set_title('Final Strain Composition')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # Attack rate
        axes[1, 2].plot(df['theta'], df['attack_rate'], 'ko-', linewidth=2)
        axes[1, 2].set_xlabel('θ')
        axes[1, 2].set_ylabel('Attack rate')
        axes[1, 2].set_title('Total Population Impact')
        axes[1, 2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df

#%% 5. INITIAL CONDITIONS SCAN
class InitialConditionAnalyzer:
    """Test sensitivity to initial seeding"""
    
    def __init__(self, model_func, t_max=365):
        self.model = model_func
        self.t = np.linspace(0, t_max, t_max)
    
    def scan_seeding_ratios(self, high_range, low_range, treatment_params,
                           save_path='initial_conditions_scan.png'):
        """
        Test how initial strain composition affects outcome
        """
        results = np.zeros((len(high_range), len(low_range)))
        
        for i, Indh_init in enumerate(high_range):
            for j, Indl_init in enumerate(low_range):
                S_init = 10000 - Indh_init - Indl_init
                y0 = [S_init, 0, Indh_init, 0, 0, 0, Indl_init, 0, 0]
                
                try:
                    solution = odeint(self.model, y0, self.t, args=(treatment_params,))
                    
                    # Compute final high-virulence fraction
                    high_final = solution[-1, 2] + solution[-1, 3]
                    low_final = solution[-1, 6] + solution[-1, 7]
                    
                    if high_final + low_final > 1e-6:
                        results[i, j] = high_final / (high_final + low_final)
                    else:
                        results[i, j] = np.nan
                except:
                    results[i, j] = np.nan
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.contourf(low_range, high_range, results, levels=20, cmap='RdYlGn_r')
        cs = ax.contour(low_range, high_range, results, levels=[0.5], 
                       colors='black', linewidths=2)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Equal outcome')
        
        plt.colorbar(im, ax=ax, label='Final high-virulence fraction')
        ax.set_xlabel('Initial low-virulence infections', fontsize=12)
        ax.set_ylabel('Initial high-virulence infections', fontsize=12)
        ax.set_title('Basin of Attraction\n(Effect of initial strain composition)', 
                    fontsize=14, weight='bold')
        
        # Add diagonal line (equal initial seeding)
        max_val = max(high_range[-1], low_range[-1])
        ax.plot([0, max_val], [0, max_val], 'w--', linewidth=2, label='Equal initial')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def stochastic_seeding(self, n_replicates=100, theta=0.5, 
                          save_path='stochastic_outcomes.png'):
        """
        Random initial seeding to test outcome variability
        """
        outcomes = []
        
        kappa_high = default_params.kappa_base * (1 + default_params.kappa_scale * 
                                                  (default_params.phi_transmission - 1))
        delta_d_high = kappa_high / (1/3)
        delta_d_low = default_params.kappa_base / (1/3)
        
        params_list = [
            default_params.beta_l, default_params.birth_rate, default_params.death_rate,
            default_params.delta, delta_d_high, delta_d_low, default_params.p_recover,
            default_params.phi_recover, default_params.phi_transmission,
            default_params.sigma, default_params.tau, theta
        ]
        
        for _ in range(n_replicates):
            # Random seeding (Poisson distributed)
            Indh_init = np.random.poisson(5)
            Indl_init = np.random.poisson(5)
            S_init = 10000 - Indh_init - Indl_init
            
            y0 = [S_init, 0, Indh_init, 0, 0, 0, Indl_init, 0, 0]
            
            solution = odeint(self.model, y0, self.t, args=(params_list,))
            
            high_final = solution[-1, 2] + solution[-1, 3]
            low_final = solution[-1, 6] + solution[-1, 7]
            
            if high_final + low_final > 1e-6:
                high_frac = high_final / (high_final + low_final)
            else:
                high_frac = np.nan
            
            outcomes.append({
                'Indh_init': Indh_init,
                'Indl_init': Indl_init,
                'high_fraction': high_frac
            })
        
        df = pd.DataFrame(outcomes).dropna()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter of outcomes
        sc = axes[0].scatter(df['Indl_init'], df['Indh_init'], 
                            c=df['high_fraction'], s=100, cmap='RdYlGn_r', 
                            alpha=0.6, edgecolor='black')
        plt.colorbar(sc, ax=axes[0], label='Final high-virulence fraction')
        axes[0].set_xlabel('Initial low-virulence', fontsize=12)
        axes[0].set_ylabel('Initial high-virulence', fontsize=12)
        axes[0].set_title('Outcome Distribution', fontsize=14, weight='bold')
        
        # Histogram of outcomes
        axes[1].hist(df['high_fraction'], bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Equal')
        axes[1].set_xlabel('Final high-virulence fraction', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Outcome Variability (n={n_replicates})', 
                         fontsize=14, weight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df

#%% 6. EVOLUTIONARY TRAJECTORY ANALYSIS
class EvolutionaryTrajectoryAnalyzer:
    """Track strain dynamics over time"""
    
    def __init__(self, model_func, t_max=365):
        self.model = model_func
        self.t = np.linspace(0, t_max, t_max)
    
    def plot_trajectory(self, theta, save_path='evolutionary_trajectory.png'):
        """Plot time series of strain composition"""
        
        kappa_high = default_params.kappa_base * (1 + default_params.kappa_scale * 
                                                  (default_params.phi_transmission - 1))
        delta_d_high = kappa_high / (1/3)
        delta_d_low = default_params.kappa_base / (1/3)
        
        params_list = [
            default_params.beta_l, default_params.birth_rate, default_params.death_rate,
            default_params.delta, delta_d_high, delta_d_low, default_params.p_recover,
            default_params.phi_recover, default_params.phi_transmission,
            default_params.sigma, default_params.tau, theta
        ]
        
        y0 = [default_params.S, 
              default_params.Eh, default_params.Indh, default_params.Idh, default_params.Rh,
              default_params.El, default_params.Indl, default_params.Idl, default_params.Rl]
        
        solution = odeint(self.model, y0, self.t, args=(params_list,))
        
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = solution.T
        
        high_infections = Indh + Idh
        low_infections = Indl + Idl
        total_infections = high_infections + low_infections
        
        high_fraction = high_infections / (total_infections + 1e-10)
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Panel 1: Absolute infections
        axes[0].plot(self.t, high_infections, 'r-', linewidth=2, label='High-virulence')
        axes[0].plot(self.t, low_infections, 'b-', linewidth=2, label='Low-virulence')
        axes[0].fill_between(self.t, 0, high_infections, color='red', alpha=0.3)
        axes[0].fill_between(self.t, 0, low_infections, color='blue', alpha=0.3)
        axes[0].set_ylabel('Infections (count)', fontsize=12)
        axes[0].set_title(f'Epidemic Dynamics (θ={theta:.2f})', fontsize=14, weight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Panel 2: Relative composition
        axes[1].plot(self.t, high_fraction, 'purple', linewidth=2)
        axes[1].axhline(0.5, color='gray', linestyle='--', label='Equal prevalence')
        axes[1].fill_between(self.t, 0, high_fraction, color='red', alpha=0.3, label='High-virulence')
        axes[1].fill_between(self.t, high_fraction, 1, color='blue', alpha=0.3, label='Low-virulence')
        axes[1].set_ylabel('High-virulence fraction', fontsize=12)
        axes[1].set_title('Strain Composition Over Time', fontsize=14, weight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Panel 3: Susceptible depletion
        axes[2].plot(self.t, S, 'g-', linewidth=2, label='Susceptible')
        axes[2].plot(self.t, Rh + Rl, 'orange', linewidth=2, label='Recovered')
        axes[2].set_xlabel('Time (days)', fontsize=12)
        axes[2].set_ylabel('Population', fontsize=12)
        axes[2].set_title('Population Dynamics', fontsize=14, weight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_trajectories(self, theta_values, save_path='trajectory_comparison.png'):
        """Compare strain dynamics across treatment levels"""
        
        fig, axes = plt.subplots(len(theta_values), 2, figsize=(14, 4*len(theta_values)))
        
        if len(theta_values) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, theta in enumerate(theta_values):
            kappa_high = default_params.kappa_base * (1 + kappa_scale * 
                                                      (default_params.phi_transmission - 1))
            delta_d_high = kappa_high / (1/3)
            delta_d_low = default_params.kappa_base / (1/3)
            
            params_list = [
                default_params.beta_l, default_params.birth_rate, default_params.death_rate,
                default_params.delta, delta_d_high, delta_d_low, default_params.p_recover,
                default_params.phi_recover, default_params.phi_transmission,
                default_params.sigma, default_params.tau, theta
            ]
            
            y0 = [default_params.S, 
                  default_params.Eh, default_params.Indh, default_params.Idh, default_params.Rh,
                  default_params.El, default_params.Indl, default_params.Idl, default_params.Rl]
            
            solution = odeint(self.model, y0, self.t, args=(params_list,))
            
            S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = solution.T
            
            high_infections = Indh + Idh
            low_infections = Indl + Idl
            high_fraction = high_infections / (high_infections + low_infections + 1e-10)
            
            # Absolute infections
            axes[idx, 0].plot(self.t, high_infections, 'r-', linewidth=2, label='High')
            axes[idx, 0].plot(self.t, low_infections, 'b-', linewidth=2, label='Low')
            axes[idx, 0].set_ylabel('Infections', fontsize=11)
            axes[idx, 0].set_title(f'θ={theta:.2f}', fontsize=12, weight='bold')
            axes[idx, 0].legend()
            axes[idx, 0].grid(alpha=0.3)
            
            # Strain composition
            axes[idx, 1].plot(self.t, high_fraction, 'purple', linewidth=2)
            axes[idx, 1].axhline(0.5, color='gray', linestyle='--')
            axes[idx, 1].set_ylabel('High-virulence fraction', fontsize=11)
            axes[idx, 1].grid(alpha=0.3)
            axes[idx, 1].set_ylim(0, 1)
        
        axes[-1, 0].set_xlabel('Time (days)', fontsize=12)
        axes[-1, 1].set_xlabel('Time (days)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

#%% 7. POLICY OPTIMIZATION
class PolicyOptimizer:
    """Find optimal treatment strategies"""
    
    def __init__(self, model_func, t_max=365):
        self.model = model_func
        self.t = np.linspace(0, t_max, t_max)
    
    def objective_function(self, theta, objective='minimize_burden', 
                          virulence_penalty=1.0):
        """
        Compute objective value for given treatment coverage
        
        Objectives:
        - 'minimize_burden': minimize total infections
        - 'minimize_virulence': minimize high-virulence fraction
        - 'balanced': weighted combination
        """
        kappa_high = default_params.kappa_base * (1 + kappa_scale * 
                                                  (default_params.phi_transmission - 1))
        delta_d_high = kappa_high / (1/3)
        delta_d_low = default_params.kappa_base / (1/3)
        
        params_list = [
            default_params.beta_l, default_params.birth_rate, default_params.death_rate,
            default_params.delta, delta_d_high, delta_d_low, default_params.p_recover,
            default_params.phi_recover, default_params.phi_transmission,
            default_params.sigma, default_params.tau, theta
        ]
        
        y0 = [default_params.S, 
              default_params.Eh, default_params.Indh, default_params.Idh, default_params.Rh,
              default_params.El, default_params.Indl, default_params.Idl, default_params.Rl]
        
        solution = odeint(self.model, y0, self.t, args=(params_list,))
        
        S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = solution.T
        
        high_infections = Indh + Idh
        low_infections = Indl + Idl
        total_infections = high_infections + low_infections
        
        # Compute metrics
        total_burden = np.trapz(total_infections, self.t)
        high_fraction = np.mean(high_infections / (total_infections + 1e-10))
        
        if objective == 'minimize_burden':
            return total_burden
        elif objective == 'minimize_virulence':
            return high_fraction
        elif objective == 'balanced':
            return total_burden + virulence_penalty * high_fraction * 1000
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def find_optimal_policy(self, objective='balanced', save_path='policy_optimization.png'):
        """Search for optimal treatment coverage"""
        from scipy.optimize import minimize_scalar
        
        result = minimize_scalar(
            lambda theta: self.objective_function(theta, objective),
            bounds=(0, 1),
            method='bounded'
        )
        
        optimal_theta = result.x
        optimal_value = result.fun
        
        # Plot objective function
        theta_range = np.linspace(0, 1, 50)
        objectives = [self.objective_function(t, objective) for t in theta_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(theta_range, objectives, 'b-', linewidth=2)
        plt.axvline(optimal_theta, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal θ={optimal_theta:.3f}')
        plt.scatter([optimal_theta], [optimal_value], color='red', s=200, zorder=5)
        plt.xlabel('Treatment coverage (θ)', fontsize=12)
        plt.ylabel(f'Objective value ({objective})', fontsize=12)
        plt.title(f'Treatment Policy Optimization\n{objective}', 
                 fontsize=14, weight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_theta, optimal_value
    
    def pareto_frontier(self, save_path='pareto_frontier.png'):
        """
        Compute trade-off between total burden and virulence selection
        """
        theta_range = np.linspace(0, 1, 30)
        
        burden_values = []
        virulence_values = []
        
        for theta in theta_range:
            burden = self.objective_function(theta, 'minimize_burden')
            virulence = self.objective_function(theta, 'minimize_virulence')
            
            burden_values.append(burden)
            virulence_values.append(virulence)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pareto curve
        sc = ax1.scatter(virulence_values, burden_values, c=theta_range, 
                        s=100, cmap='viridis', edgecolor='black')
        plt.colorbar(sc, ax=ax1, label='θ')
        ax1.set_xlabel('High-virulence fraction', fontsize=12)
        ax1.set_ylabel('Total disease burden', fontsize=12)
        ax1.set_title('Pareto Frontier\n(Trade-off between objectives)', 
                     fontsize=14, weight='bold')
        ax1.grid(alpha=0.3)
        
        # Individual objectives
        ax2.plot(theta_range, burden_values, 'b-o', linewidth=2, label='Total burden')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(theta_range, virulence_values, 'r-s', linewidth=2, 
                     label='Virulence fraction')
        
        ax2.set_xlabel('Treatment coverage (θ)', fontsize=12)
        ax2.set_ylabel('Disease burden', fontsize=12, color='blue')
        ax2_twin.set_ylabel('Virulence fraction', fontsize=12, color='red')
        ax2.set_title('Dual Objectives', fontsize=14, weight='bold')
        ax2.grid(alpha=0.3)
        
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return pd.DataFrame({
            'theta': theta_range,
            'burden': burden_values,
            'virulence': virulence_values
        })

#%% MAIN EXECUTION
if __name__ == "__main__":
    print("Starting comprehensive analysis...")
    
    # Create output directory
    output_dir = Path("Analysis_Results")
    output_dir.mkdir(exist_ok=True)
    
    # Use model v2 (with detection parameters)
    model = SEIRS_model_v2
    
    # Analysis ranges
    theta_range = np.linspace(0, 1, 20)
    kappa_scale_range = np.linspace(0, 3, 20)
    
    #%% 1. Bifurcation Analysis
    print("\n1. Running bifurcation analysis...")
    bifurcation = BifurcationAnalyzer(model)
    phase_results = bifurcation.phase_diagram(
        theta_range, kappa_scale_range,
        save_path=output_dir / "phase_diagram.png"
    )
    
    critical_points, vf_curve = bifurcation.find_critical_thresholds(theta_range)
    print(f"Critical points found: {len(critical_points)}")
    for cp in critical_points:
        print(f"  θ={cp['theta']:.3f}, virulence_frac={cp['virulence_fraction']:.3f}, type={cp['type']}")
    
    #%% 2. Elasticity Analysis
    print("\n2. Running elasticity analysis...")
    elasticity = ElasticityAnalyzer()
    
    baseline_params = {
        'beta_l': default_params.beta_l,
        'sigma': default_params.sigma,
        'phi_transmission': default_params.phi_transmission,
        'theta': 0.5,
        'kappa_base': default_params.kappa_base,
        'kappa_scale': default_params.kappa_scale,
        'p_recover': default_params.p_recover
    }
    
    elasticities = elasticity.elasticity_matrix(baseline_params, list(baseline_params.keys()))
    elasticity.plot_elasticities(elasticities, save_path=output_dir / "elasticity_tornado.png")
    
    # Elasticity at critical points
    critical_thetas = [0.0, 0.3, 0.5, 0.7, 1.0]
    elasticity_df = elasticity.sensitivity_at_critical_points(
        critical_thetas, baseline_params
    )
    elasticity_df.to_csv(output_dir / "elasticity_by_theta.csv", index=False)
    
    #%% 3. Invasion Analysis
    print("\n3. Running invasion analysis...")
    invasion = InvasionAnalyzer(model)
    
    invasion_df = invasion.invasion_matrix(theta_range, kappa_scale=1.0)
    invasion.plot_invasion_diagram(invasion_df, 
                                  save_path=output_dir / "invasion_analysis.png")
    invasion_df.to_csv(output_dir / "invasion_results.csv", index=False)
    
    print("\nESS summary:")
    print(invasion_df['ESS'].value_counts())
    
    #%% 4. Transient Dynamics
    print("\n4. Analyzing transient dynamics...")
    transient = TransientAnalyzer(model)
    
    transient_df = transient.compare_scenarios(
        theta_range,
        save_path=output_dir / "transient_comparison.png"
    )
    transient_df.to_csv(output_dir / "transient_metrics.csv", index=False)
    
    #%% 5. Initial Conditions Scan
    print("\n5. Scanning initial conditions...")
    initial_cond = InitialConditionAnalyzer(model)
    
    # Build params for theta=0.5
    kappa_high = default_params.kappa_base * (1 + default_params.kappa_scale * 
                                              (default_params.phi_transmission - 1))
    delta_d_high = kappa_high / (1/3)
    delta_d_low = default_params.kappa_base / (1/3)
    
    params_mid = [
        default_params.beta_l, default_params.birth_rate, default_params.death_rate,
        default_params.delta, delta_d_high, delta_d_low, default_params.p_recover,
        default_params.phi_recover, default_params.phi_transmission,
        default_params.sigma, default_params.tau, 0.5
    ]
    
    high_range = np.linspace(1, 20, 15)
    low_range = np.linspace(1, 20, 15)
    
    basin_results = initial_cond.scan_seeding_ratios(
        high_range, low_range, params_mid,
        save_path=output_dir / "basin_of_attraction.png"
    )
    
    # Stochastic outcomes
    stochastic_df = initial_cond.stochastic_seeding(
        n_replicates=200, theta=0.5,
        save_path=output_dir / "stochastic_outcomes.png"
    )
    stochastic_df.to_csv(output_dir / "stochastic_seeding.csv", index=False)
    
    #%% 6. Evolutionary Trajectories
    print("\n6. Plotting evolutionary trajectories...")
    evolution = EvolutionaryTrajectoryAnalyzer(model)
    
    # Single trajectory at optimal point
    evolution.plot_trajectory(theta=0.5, 
                            save_path=output_dir / "evolutionary_trajectory.png")
    
    # Compare multiple treatment levels
    theta_comparison = [0.0, 0.3, 0.6, 0.9]
    evolution.compare_trajectories(theta_comparison,
                                  save_path=output_dir / "trajectory_comparison.png")
    
    #%% 7. Policy Optimization
    print("\n7. Optimizing treatment policy...")
    optimizer = PolicyOptimizer(model)
    
    # Find optimal for different objectives
    optimal_burden, burden_value = optimizer.find_optimal_policy(
        objective='minimize_burden',
        save_path=output_dir / "optimal_policy_burden.png"
    )
    print(f"Optimal θ (minimize burden): {optimal_burden:.3f}")
    
    optimal_virulence, virulence_value = optimizer.find_optimal_policy(
        objective='minimize_virulence',
        save_path=output_dir / "optimal_policy_virulence.png"
    )
    print(f"Optimal θ (minimize virulence): {optimal_virulence:.3f}")
    
    optimal_balanced, balanced_value = optimizer.find_optimal_policy(
        objective='balanced',
        save_path=output_dir / "optimal_policy_balanced.png"
    )
    print(f"Optimal θ (balanced): {optimal_balanced:.3f}")
    
    # Pareto frontier
    pareto_df = optimizer.pareto_frontier(save_path=output_dir / "pareto_frontier.png")
    pareto_df.to_csv(output_dir / "pareto_results.csv", index=False)
    
    #%% Summary Report
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey findings:")
    print(f"1. Critical thresholds found: {len(critical_points)}")
    print(f"2. Most elastic parameters: {max(elasticities, key=elasticities.get)}")
    print(f"3. Optimal treatment coverage (balanced): {optimal_balanced:.3f}")
    print(f"4. ESS distribution: {dict(invasion_df['ESS'].value_counts())}")
    
    # Save summary
    summary = {
        'critical_points': critical_points,
        'elasticities': elasticities,
        'optimal_theta_burden': optimal_burden,
        'optimal_theta_virulence': optimal_virulence,
        'optimal_theta_balanced': optimal_balanced,
        'ess_counts': dict(invasion_df['ESS'].value_counts())
    }
    
    import json
    with open(output_dir / "summary.json", 'w') as f:
        # Convert numpy types to native Python
        summary_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else 
                float(v) if isinstance(v, np.floating) else v)
            for k, v in summary.items()
        }
        json.dump(summary_serializable, f, indent=2)
    
    print(f"\nSummary saved to: {output_dir / 'summary.json'}")