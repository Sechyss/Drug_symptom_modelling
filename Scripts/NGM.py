import os
import sys
import numpy as np
from numpy.linalg import eigvals

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models import params as model_params

"""
NGM.py

Next Generation Matrix (NGM) computation for SEIRS model v2.
Computes R0 and related metrics using the NGM approach.
"""

def load_params():
    """Load parameters from Models.params into a dictionary."""
    params_dict = {
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
    return params_dict

def params_dict_to_tuple(params_dict):
    """Convert parameter dictionary to tuple for model functions."""
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

def new_infections(x, params_dict):
    """
    New infection terms F_i for NGM.
    State vector x = [Eh, El, Indh, Indl, Idh, Idl]
    
    Returns only the NEW INFECTION terms (transmission from infected to susceptible).
    """
    Eh, El, Indh, Indl, Idh, Idl = x
    
    # Extract parameters
    contact_rate = params_dict['contact_rate']
    trans_prob = params_dict['transmission_probability']
    p_recover = params_dict['p_recover']
    phi_transmission = params_dict['phi_transmission']
    
    # Compute betas
    beta_l = contact_rate * trans_prob
    beta_h = phi_transmission * beta_l
    
    # At DFE, S ≈ 1 (assume total population normalized to 1)
    S = 1.0
    
    # New infections (force of infection terms)
    # High strain: beta_h * S * (Indh + p_recover * Idh)
    # Low strain: beta_l * S * (Indl + p_recover * Idl)
    F = np.array([
        beta_h * S * (Indh + p_recover * Idh),  # dEh/dt (new infections)
        beta_l * S * (Indl + p_recover * Idl),  # dEl/dt (new infections)
        0.0,  # Indh (no new infections, just progression)
        0.0,  # Indl
        0.0,  # Idh
        0.0   # Idl
    ])
    
    return F

def deriv(x, params_dict):
    """
    All derivative terms for infected compartments.
    This includes both new infections (F) and transitions (V).
    
    Returns: dx/dt for x = [Eh, El, Indh, Indl, Idh, Idl]
    """
    Eh, El, Indh, Indl, Idh, Idl = x
    
    # Extract parameters
    contact_rate = params_dict['contact_rate']
    trans_prob = params_dict['transmission_probability']
    death_rate = params_dict['death_rate']
    kappa_base = params_dict['kappa_base']
    kappa_scale = params_dict['kappa_scale']
    p_recover = params_dict['p_recover']
    phi_recover = params_dict['phi_recover']
    phi_transmission = params_dict['phi_transmission']
    sigma = params_dict['sigma']
    tau = params_dict['tau']
    theta = params_dict['theta']
    
    # Compute betas
    beta_l = contact_rate * trans_prob
    beta_h = phi_transmission * beta_l
    
    # Compute kappa values
    virulence_excess = phi_transmission - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * virulence_excess)
    kappa_low = kappa_base
    
    # Safety: ensure kappa * theta ≤ 1
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)
    
    # At DFE, S ≈ 1
    S = 1.0
    
    # Derivative terms (matching SEIRS_model_v2)
    dxdt = np.array([
        # dEh/dt = new infections - progression - death
        beta_h * S * (Indh + p_recover * Idh) - tau * Eh - death_rate * Eh,
        
        # dEl/dt
        beta_l * S * (Indl + p_recover * Idl) - tau * El - death_rate * El,
        
        # dIndh/dt = progression from Eh (untreated fraction) - recovery - death
        (1 - kappa_high * theta) * tau * Eh - phi_recover * sigma * Indh - death_rate * Indh,
        
        # dIndl/dt
        (1 - kappa_low * theta) * tau * El - sigma * Indl - death_rate * Indl,
        
        # dIdh/dt = progression from Eh (treated fraction) - recovery - death
        kappa_high * theta * tau * Eh - phi_recover * sigma * Idh - death_rate * Idh,
        
        # dIdl/dt
        kappa_low * theta * tau * El - sigma * Idl - death_rate * Idl
    ])
    
    return dxdt

def jacobian(fun, x, params_dict, eps=1e-8):
    """
    Compute Jacobian matrix numerically using finite differences.
    
    Args:
        fun: function f(x, params_dict) returning array
        x: point at which to evaluate Jacobian
        params_dict: parameter dictionary
        eps: finite difference step size
    
    Returns:
        J: Jacobian matrix (m × n) where m = len(f(x)), n = len(x)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f0 = np.asarray(fun(x, params_dict), dtype=float)
    m = f0.size
    J = np.zeros((m, n), dtype=float)
    
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        f_plus = fun(x_plus, params_dict)
        J[:, j] = (f_plus - f0) / eps
    
    return J

def compute_NGM(params_dict):
    """
    Compute Next Generation Matrix and R0.
    
    The NGM approach:
    1. Identify disease-free equilibrium (DFE)
    2. Linearize around DFE: dx/dt = (F - V)x
       - F: new infections
       - V: transitions out of infected compartments
    3. Compute K = F V^(-1) (next generation matrix)
    4. R0 = spectral radius of K (largest eigenvalue)
    
    Args:
        params_dict: dictionary of model parameters
    
    Returns:
        dict with keys:
            - R0: basic reproduction number
            - eigenvalues: all eigenvalues of NGM
            - NGM: the next generation matrix K
            - F_matrix: Jacobian of new infections
            - V_matrix: Jacobian of transitions
    """
    # Disease-free equilibrium for infected compartments
    # x = [Eh, El, Indh, Indl, Idh, Idl] all zero
    x0 = np.zeros(6)
    
    # Compute Jacobians at DFE
    JF = jacobian(new_infections, x0, params_dict)  # dF/dx
    Jf = jacobian(deriv, x0, params_dict)           # df/dx
    
    # V = F - f  (since f = F - V  => V = F - f)
    JV = JF - Jf
    
    # Compute next generation matrix K = F V^(-1)
    try:
        V_inv = np.linalg.inv(JV)
        K = JF @ V_inv
    except np.linalg.LinAlgError:
        print("Warning: V matrix is singular, cannot compute NGM")
        return {
            'R0': np.nan,
            'eigenvalues': np.array([np.nan]),
            'NGM': np.full((6, 6), np.nan),
            'F_matrix': JF,
            'V_matrix': JV
        }
    
    # Compute eigenvalues of K
    eigs = eigvals(K)
    
    # R0 is the spectral radius (largest eigenvalue magnitude)
    R0 = np.max(np.abs(eigs))
    
    return {
        'R0': R0,
        'eigenvalues': eigs,
        'NGM': K,
        'F_matrix': JF,
        'V_matrix': JV
    }

if __name__ == '__main__':
    print("="*60)
    print("Next Generation Matrix Analysis")
    print("="*60)
    
    # Load parameters
    params = load_params()
    
    print("\nParameters:")
    for key, val in params.items():
        print(f"  {key:25s}: {val}")
    
    # Compute NGM
    result = compute_NGM(params)
    
    print(f"\n{'='*60}")
    print(f"R₀ = {result['R0']:.4f}")
    print(f"{'='*60}")
    
    print("\nEigenvalues of NGM:")
    for i, eig in enumerate(result['eigenvalues']):
        print(f"  λ_{i+1} = {eig:.4f}")
    
    print("\nNext Generation Matrix K:")
    print(result['NGM'])
    
    print("\nF matrix (new infections):")
    print(result['F_matrix'])
    
    print("\nV matrix (transitions):")
    print(result['V_matrix'])