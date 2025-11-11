import numpy as np

#%% ODE Model
def SEIRS_first_model(y, t, params):
    """
    SEIRS model testing virulence-transmission trade-off with symptom-targeting drug.
    
    Biological assumptions:
    - Transmission requires symptoms (sneezing, coughing)
    - Drug reduces symptoms but doesn't eliminate pathogen
    - High-virulence: produces strong symptoms → treated individuals still transmit
    - Low-virulence: produces mild symptoms → treatment eliminates transmission
    
    This tests hypothesis: Can drugs that mask symptoms allow "super-virulent" strains
    to evolve by removing the constraint that high virulence = immobile/dead hosts?
    
    State vector y (length 9): [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
    
    Parameters (11):
    - beta_l: baseline transmission rate (low-virulence)
    - birth_rate, death_rate: demographic rates
    - delta: rate of waning immunity
    - delta_d: detection/treatment rate (proportion diagnosed)
    - p_recover: treatment efficacy (proportion that recover faster)
    - phi_recover: recovery rate modifier for high-strain (SET TO 1.0 for no effect currently)
    - phi_transmission: transmission multiplier for high-strain (e.g., 1.05 = 5% higher R0)
    - sigma: recovery rate
    - tau: 1/latent period
    - theta: treatment coverage (proportion of detected cases treated)
    
    Future extensions via phi_recover:
    - < 1.0: high-virulence has longer infectious period (more virulent = sicker longer)
    - > 1.0: high-virulence has shorter infectious period (burn out faster)
    """
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = y

    if not hasattr(params, "__len__") or len(params) != 11:
        raise ValueError("params must be a sequence of length 11")

    (beta_l, birth_rate, death_rate, delta, delta_d, p_recover,
     phi_recover, phi_transmission, sigma, tau, theta) = params

    # Transmission dynamics
    beta_h = phi_transmission * beta_l
    
    # HIGH-VIRULENCE: Strong symptoms → both treated & untreated transmit
    # (drug reduces symptoms but not enough to eliminate transmission)
    B_h = beta_h * (Indh + Idh)
    
    # LOW-VIRULENCE: Mild symptoms → only untreated transmit
    # (drug eliminates their weak symptoms → no transmission)
    B_l = beta_l * Indl

    # ODEs
    dSdt = birth_rate - (B_h + B_l) * S + delta * (Rh + Rl) - death_rate * S
    dEhdt = B_h * S - tau * Eh - death_rate * Eh
    dEldt = B_l * S - tau * El - death_rate * El

    # High-virulence progression and recovery
    dIndhdt = tau * Eh - delta_d * theta * Indh - phi_recover * sigma * Indh - death_rate * Indh
    dIdhdt = delta_d * theta * Indh - phi_recover * p_recover * sigma * Idh - death_rate * Idh

    # Low-virulence progression and recovery
    dIndldt = tau * El - delta_d * theta * Indl - sigma * Indl - death_rate * Indl
    dIdldt = delta_d * theta * Indl - p_recover * sigma * Idl - death_rate * Idl

    # Recovery compartments
    dRhdt = phi_recover * sigma * (p_recover * Idh + Indh) - delta * Rh - death_rate * Rh
    dRldt = sigma * (p_recover * Idl + Indl) - delta * Rl - death_rate * Rl

    # Mass balance check
    total = S + Eh + Indh + Idh + Rh + El + Indl + Idl + Rl
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Non-finite or non-positive total population")
    
    return dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt
