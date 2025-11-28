import numpy as np

#%% ODE Model
def SEIRS_model_v1(y, t, params):
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


def SEIRS_model_v2(y, t, params):
    """
    SEIRS model testing virulence-transmission trade-off with symptom-targeting drug.
    
    Parameters (12):  # Note: increased to 12
    - contact_rate, transmission_probability
    - birth_rate, death_rate
    - delta: immunity waning rate
    - kappa_base: baseline detection rate for low virulence
    - kappa_scale: sensitivity of detection to virulence increase
    - p_recover: transmission reduction for treated
    - phi_recover: recovery rate modifier for high-strain
    - phi_transmission: transmission multiplier for high-strain (virulence proxy)
    - sigma: recovery rate
    - tau: incubation rate
    - theta: treatment coverage
    
    Detection mechanism:
    - kappa_high = kappa_base * (1 + kappa_scale * (phi_transmission - 1))
    - Example: phi_transmission=1.1, kappa_scale=2 → 20% increase in detection
    """
    import numpy as np

    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = y

    # Expect: (contact_rate, transmission_probability, birth_rate, death_rate,
    #          delta, kappa_base, kappa_scale, p_recover, phi_recover,
    #          phi_transmission, sigma, tau, theta)
    if not hasattr(params, "__len__") or len(params) != 13:
        raise ValueError("SEIRS_model_v2 expects 13 params (contact_rate, transmission_probability, birth_rate, death_rate, delta, kappa_base, kappa_scale, p_recover, phi_recover, phi_transmission, sigma, tau, theta)")

    (contact_rate, transmission_probability, birth_rate, death_rate,
     delta, kappa_base, kappa_scale, p_recover, phi_recover,
     phi_transmission, sigma, tau, theta) = params

    # Build beta from both contact_rate and transmission_probability
    beta_l = contact_rate * transmission_probability
    beta_h = phi_transmission * beta_l

    # Detection scaling (kappa)
    vir_excess = phi_transmission - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * vir_excess)
    kappa_low  = kappa_base
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low  = min(kappa_low,  1.0 / theta)

    theta_high = kappa_high * theta
    theta_low  = kappa_low  * theta

    # Forces of infection (treated transmit partially via p_recover)
    B_h = beta_h * (Indh + p_recover * Idh)
    B_l = beta_l * (Indl + p_recover * Idl)

    # ODEs
    dSdt   = birth_rate - (B_h + B_l) * S + delta * (Rh + Rl) - death_rate * S
    dEhdt  = B_h * S - tau * Eh - death_rate * Eh
    dEldt  = B_l * S - tau * El - death_rate * El

    # Split at symptom onset (no Ind -> Id lag); p_recover does NOT affect recovery rates
    sigma_h = phi_recover * sigma
    sigma_l = sigma

    dIndhdt = (1.0 - theta_high) * tau * Eh - sigma_h * Indh - death_rate * Indh
    dIdhdt  = theta_high * tau * Eh - sigma_h * Idh  - death_rate * Idh

    dIndldt = (1.0 - theta_low)  * tau * El - sigma_l * Indl - death_rate * Indl
    dIdldt  = theta_low * tau * El - sigma_l * Idl  - death_rate * Idl

    # Recovery (no p_recover here; it only reduces transmission above)
    dRhdt = sigma_h * (Indh + Idh) - delta * Rh - death_rate * Rh
    dRldt = sigma_l * (Indl + Idl) - delta * Rl - death_rate * Rl

    return np.array([dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt])

def SEIRS_model_v3(y, t, params):
    """
    SEIRS model with split-at-onset treatment and drug effects that
    modify behavior for treated infectious individuals.

    Params (15):
      (contact_rate_low, transmission_probability_low, contact_rate_high, phi_transmission,
       drug_contact_multiplier, drug_transmission_multiplier,
       birth_rate, death_rate, delta, kappa_base, kappa_scale, phi_recover, sigma, tau, theta)

    Notes:
    - Split at symptom onset (no Ind -> Id lag).
    - Drug affects treated infectious classes only:
        c_treated = drug_contact_multiplier * contact_rate
        r_treated = drug_transmission_multiplier * transmission_probability
      so β_treated = c_treated * r_treated (and × φ for high strain).
    """
    import numpy as np

    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = y

    if not hasattr(params, "__len__") or len(params) != 15:
        raise ValueError(
            "SEIRS_model_v3 expects 15 params: "
            "(contact_rate_low, transmission_probability_low, contact_rate_high, phi_transmission, "
            "drug_contact_multiplier, drug_transmission_multiplier, "
            "birth_rate, death_rate, delta, kappa_base, kappa_scale, phi_recover, sigma, tau, theta)"
        )

    (c_low, r_low, c_high, phi_t,
     m_c_drug, m_r_drug,
     birth_rate, death_rate, delta, kappa_base, kappa_scale, phi_recover, sigma, tau, theta) = params

    # Betas for untreated vs treated (drug-modified) and low vs high
    beta_l_u = c_low * r_low
    beta_l_t = (c_low * m_c_drug) * (r_low * m_r_drug)

    beta_h_u = c_high * r_low * phi_t
    beta_h_t = (c_high * m_c_drug) * (r_low * m_r_drug) * phi_t

    # Detection scaling (kappa)
    vir_excess = phi_t - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * vir_excess)
    kappa_low  = kappa_base
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low  = min(kappa_low,  1.0 / theta)

    theta_high = kappa_high * theta
    theta_low  = kappa_low  * theta

    # Forces of infection (treated use drug-modified betas)
    B_h = beta_h_u * Indh + beta_h_t * Idh
    B_l = beta_l_u * Indl + beta_l_t * Idl

    # ODEs
    dSdt   = birth_rate - (B_h + B_l) * S + delta * (Rh + Rl) - death_rate * S
    dEhdt  = B_h * S - tau * Eh - death_rate * Eh
    dEldt  = B_l * S - tau * El - death_rate * El

    # Split at onset; same recovery speeds for treated/untreated within strain
    sigma_h = phi_recover * sigma
    sigma_l = sigma

    dIndhdt = (1.0 - theta_high) * tau * Eh - sigma_h * Indh - death_rate * Indh
    dIdhdt  = theta_high * tau * Eh              - sigma_h * Idh  - death_rate * Idh

    dIndldt = (1.0 - theta_low)  * tau * El - sigma_l * Indl - death_rate * Indl
    dIdldt  = theta_low * tau * El          - sigma_l * Idl  - death_rate * Idl

    dRhdt = sigma_h * (Indh + Idh) - delta * Rh - death_rate * Rh
    dRldt = sigma_l * (Indl + Idl) - delta * Rl - death_rate * Rl

    return np.array([dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt])
