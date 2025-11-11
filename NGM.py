import numpy as np
from numpy.linalg import eigvals, inv
from scipy.integrate import odeint
from Models.SEIRS_Models import SEIRS_first_model  # the ODE model function to integrate
from Models import params as model_params  # import default parameters

# Make any stochastic behaviour reproducible
np.random.seed(42)

"""
NGM.py

Compute the Next-Generation Matrix (NGM) and R0 for an SEIRS model.

This script uses the imported SEIRS_first_model and model_params directly.

Output:
- F, V, NGM matrices (restricted to infected compartments) and R0 (spectral radius)

Usage:
    python3 NGM.py
"""

# ---------------------------
# Load parameters from imported module
# ---------------------------
def load_params():
    """Extract parameters from model_params module"""
    params = {
        'beta_l': getattr(model_params, "beta_l", 0.25),
        'birth_rate': getattr(model_params, "birth_rate", 0.0),
        'death_rate': getattr(model_params, "death_rate", 0.0),
        'delta': getattr(model_params, "delta", 1/90),
        'delta_d': getattr(model_params, "delta_d", 1/3),
        'p_recover': getattr(model_params, "p_recover", 1.5),
        'phi_recover': getattr(model_params, "phi_recover", 1.0),
        'sigma': getattr(model_params, "sigma", 1/10),
        'tau': getattr(model_params, "tau", 1/3),
        'phi_transmission': getattr(model_params, "phi_transmission", 1.05),
        'theta': getattr(model_params, "theta", 0.3),
        'N': getattr(model_params, "S", 10000) + getattr(model_params, "Eh", 0) + 
             getattr(model_params, "Indh", 5) + getattr(model_params, "Idh", 0) + 
             getattr(model_params, "Rh", 0) + getattr(model_params, "El", 0) + 
             getattr(model_params, "Indl", 5) + getattr(model_params, "Idl", 0) + 
             getattr(model_params, "Rl", 0)
    }
    return params

# ---------------------------
# Define model structure
# ---------------------------
# Compartment order for SEIRS_first_model: [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
COMPARTMENTS = ['S', 'Eh', 'Indh', 'Idh', 'Rh', 'El', 'Indl', 'Idl', 'Rl']
INFECTED_COMPARTMENTS = ['Eh', 'Indh', 'Idh', 'El', 'Indl', 'Idl']  # All infected states

# ---------------------------
# Wrapper for derivative function
# ---------------------------
def deriv(x, params_dict):
    """Wrapper to call SEIRS_first_model with correct parameter format"""
    params_tuple = (
        params_dict['beta_l'],
        params_dict['birth_rate'],
        params_dict['death_rate'],
        params_dict['delta'],
        params_dict['delta_d'],
        params_dict['p_recover'],
        params_dict['phi_recover'],
        params_dict['phi_transmission'],
        params_dict['sigma'],
        params_dict['tau'],
        params_dict['theta']
    )
    # SEIRS_first_model expects (y, t, params)
    return SEIRS_first_model(x, 0, params_tuple)

# ---------------------------
# Define new infections function
# ---------------------------
def new_infections(x, params_dict):
    """
    Returns vector of NEW infection terms for each compartment.
    For SEIRS model, only S compartment generates new infections.
    """
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = x
    
    beta_l = params_dict['beta_l']
    phi_transmission = params_dict['phi_transmission']
    theta = params_dict['theta']
    
    # Force of infection from high-virulence strain
    lambda_h = beta_l * phi_transmission * (Indh + Idh)
    
    # Force of infection from low-virulence strain
    lambda_l = beta_l * (Indl + Idl)
    
    # New infections
    new_inf_high = lambda_h * S
    new_inf_low = lambda_l * S
    
    # Return vector: new infections per compartment
    # Only S→Eh and S→El have new infections, others are 0
    return np.array([
        0,              # S (loses infections, doesn't gain)
        new_inf_high,   # Eh (gains from S)
        0,              # Indh
        0,              # Idh
        0,              # Rh
        new_inf_low,    # El (gains from S)
        0,              # Indl
        0,              # Idl
        0               # Rl
    ])

# ---------------------------
# Numerical Jacobian
# ---------------------------
def jacobian(fun, x, params, eps=1e-8):
    x = np.asarray(x, dtype=float)
    n = x.size
    f0 = np.asarray(fun(x, params), dtype=float)
    m = f0.size
    J = np.zeros((m, n), dtype=float)
    # central differences
    for i in range(n):
        dx = np.zeros_like(x)
        h = eps * max(1.0, abs(x[i]))
        dx[i] = h
        f_plus = np.asarray(fun(x + dx, params), dtype=float)
        f_minus = np.asarray(fun(x - dx, params), dtype=float)
        J[:, i] = (f_plus - f_minus) / (2*h)
    return J

# ---------------------------
# Build disease-free equilibrium (DFE)
# ---------------------------
def build_DFE(params):
    """
    Disease-free equilibrium: all population in S, no infections.
    Returns state vector normalized to proportions.
    """
    N = params['N']
    # DFE: [S=N, Eh=0, Indh=0, Idh=0, Rh=0, El=0, Indl=0, Idl=0, Rl=0]
    x0 = np.array([N, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    # Normalize to proportions (as model expects fractions)
    return x0 / x0.sum()

# ---------------------------
# Main NGM computation
# ---------------------------
def compute_NGM(params):
    """Compute Next-Generation Matrix and R0"""
    
    # Get infected compartment indices
    infected_idx = [COMPARTMENTS.index(name) for name in INFECTED_COMPARTMENTS]
    
    # Build DFE state
    x0 = build_DFE(params)
    
    # Compute Jacobians at DFE
    JF = jacobian(new_infections, x0, params)  # dF/dx
    Jf = jacobian(deriv, x0, params)           # df/dx
    
    # V = F - f  (since f = F - V  => V = F - f)
    JV = JF - Jf
    
    # Restrict to infected compartments
    idx = np.ix_(infected_idx, infected_idx)
    F_block = JF[idx]
    V_block = JV[idx]
    
    # Invert V_block
    try:
        V_inv = inv(V_block)
    except Exception as e:
        raise np.linalg.LinAlgError(f"V matrix is singular or not invertible: {e}")
    
    # Next-Generation Matrix
    NGM = F_block @ V_inv
    
    # Compute R0 (spectral radius)
    eigs = eigvals(NGM)
    R0 = max(np.real(eigs))
    
    return {
        'compartments': COMPARTMENTS,
        'infected_names': INFECTED_COMPARTMENTS,
        'F': F_block,
        'V': V_block,
        'NGM': NGM,
        'eigenvalues': eigs,
        'R0': R0,
        'DFE': x0
    }

# ---------------------------
# CLI / run
# ---------------------------
def main():
    params = load_params()
    
    print("="*60)
    print("Next-Generation Matrix (NGM) Analysis")
    print("="*60)
    print("\nLoaded parameters:")
    for k, v in params.items():
        if k != 'N':
            print(f"  {k}: {v}")
    print(f"  Total population (N): {params['N']}")
    
    res = compute_NGM(params)
    
    # Print results
    np.set_printoptions(precision=6, suppress=True)
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"\nCompartment order: {res['compartments']}")
    print(f"Infected compartments: {res['infected_names']}")
    print(f"\nDFE state (normalized):\n{res['DFE']}")
    print(f"\nF matrix (new infections, infected block):\n{res['F']}")
    print(f"\nV matrix (transitions, infected block):\n{res['V']}")
    print(f"\nNext-Generation Matrix (NGM = F @ V^(-1)):\n{res['NGM']}")
    print(f"\nEigenvalues of NGM:\n{res['eigenvalues']}")
    print(f"\n{'='*60}")
    print(f"Basic Reproduction Number (R0): {res['R0']:.6f}")
    print(f"{'='*60}")
    
    # Interpretation
    if res['R0'] > 1:
        print("\n  R0 > 1: Disease will spread in the population")
    elif res['R0'] < 1:
        print("\n  R0 < 1: Disease will die out")
    else:
        print("\n  R0 = 1: Disease at threshold (endemic equilibrium)")

if __name__ == '__main__':
    main()