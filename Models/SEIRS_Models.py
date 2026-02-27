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
    dIdhdt  = theta_high * tau * Eh - sigma_h * Idh  - death_rate * Idh

    dIndldt = (1.0 - theta_low)  * tau * El - sigma_l * Indl - death_rate * Indl
    dIdldt  = theta_low * tau * El - sigma_l * Idl  - death_rate * Idl

    dRhdt = sigma_h * (Indh + Idh) - delta * Rh - death_rate * Rh
    dRldt = sigma_l * (Indl + Idl) - delta * Rl - death_rate * Rl

    return np.array([dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt])


def SEIRS_model_v4(y, t, params):
    """
    SEIRS model variant simplified to a single strain with drug-modified behavior.

        Params (12):
      (contact_rate, transmission_probability, phi_transmission,
       drug_contact_multiplier, drug_transmission_multiplier,
             birth_rate, death_rate, kappa_base, kappa_scale,
             sigma, tau, theta)

    Notes:
    - Single strain (low-virulence naming kept for compatibility).
    - Split at symptom onset (no Ind -> Id lag).
    - Drug affects treated infectious classes only:
        c_treated = drug_contact_multiplier * contact_rate
        r_treated = drug_transmission_multiplier * transmission_probability
      so β_treated = c_treated * r_treated.
    - kappa_scale and phi_transmission are used to set an effective detection
      scaling for this single strain (consistent with v2/v3 logic).
    """
    import numpy as np

    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    S, El, Indl, Idl, Rl = y

    if not hasattr(params, "__len__") or len(params) != 12:
        raise ValueError(
            "SEIRS_model_v4 expects 12 params: "
            "(contact_rate, transmission_probability, phi_transmission, "
            "drug_contact_multiplier, drug_transmission_multiplier, "
            "birth_rate, death_rate, kappa_base, kappa_scale, "
            "sigma, tau, theta)"
        )

    (c_low, r_low, c_high, phi_t,
     m_c_drug, m_r_drug,
     birth_rate, death_rate, kappa_base, kappa_scale, sigma, tau, theta) = params

    # Demography OFF: ignore birth_rate, death_rate
    # Betas for untreated vs treated (drug-modified)
    beta_l_u = c_low * r_low
    beta_l_t = (c_low * m_c_drug) * (r_low * m_r_drug)
    beta_h_u = c_high * r_low * phi_t
    beta_h_t = (c_high * m_c_drug) * (r_low * m_r_drug) * phi_t

    # Detection scaling (single strain): allow kappa_scale to adjust detection
    # based on phi_transmission so parameter meanings remain consistent.
    kappa_low = kappa_base * (1.0 + kappa_scale * (phi_t - 1.0))
    if theta > 0:
        kappa_low = min(kappa_low, 1.0 / theta)

    theta_low = kappa_low * theta

    # Force of infection
    B_l = beta_l_u * Indl + beta_l_t * Idl

    # ODEs (no births/deaths, no waning)
    dSdt    = - B_l * S
    dEldt   = B_l * S - tau * El
    sigma_l = sigma
    dIndldt = (1.0 - theta_low) * tau * El - sigma_l * Indl
    dIdldt  = theta_low * tau * El - sigma_l * Idl
    dRldt   = sigma_l * (Indl + Idl)

    return np.array([dSdt, dEldt, dIndldt, dIdldt, dRldt])


def SEIRS_model_v5(y, t, params):
    """
    SEIRS model with split-at-onset treatment and drug effects that
    modify behavior for treated infectious individuals.

    Params (14):
      (contact_rate_low, transmission_probability_low, contact_rate_high, phi_transmission,
       drug_contact_multiplier, drug_transmission_multiplier,
       birth_rate, death_rate, kappa_base, kappa_scale, phi_recover, sigma, tau, theta)

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

    if not hasattr(params, "__len__") or len(params) != 14:
        raise ValueError(
            "SEIRS_model_v3 expects 14 params: "
            "(contact_rate_low, transmission_probability_low, contact_rate_high, phi_transmission, "
            "drug_contact_multiplier, drug_transmission_multiplier, "
            "birth_rate, death_rate, kappa_base, kappa_scale, phi_recover, sigma, tau, theta)"
        )

    (c_low, r_low, c_high, phi_t,
     m_c_drug, m_r_drug,
     birth_rate, death_rate, kappa_base, kappa_scale, phi_recover, sigma, tau, theta) = params

    # Demography OFF: ignore birth_rate, death_rate
    beta_l_u = c_low * r_low
    beta_l_t = (c_low * m_c_drug) * (r_low * m_r_drug)
    beta_h_u = c_high * r_low * phi_t
    beta_h_t = (c_high * m_c_drug) * (r_low * m_r_drug) * phi_t

    # Detection scaling (kappa)
    vir_excess = phi_t - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * vir_excess)
    kappa_low = kappa_base
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)

    theta_high = kappa_high * theta
    theta_low = kappa_low * theta

    B_h = beta_h_u * Indh + beta_h_t * Idh
    B_l = beta_l_u * Indl + beta_l_t * Idl

    # ODEs (no births/deaths, no waning)
    dSdt    = - (B_h + B_l) * S
    dEhdt   = B_h * S - tau * Eh
    dEldt   = B_l * S - tau * El

    sigma_h = phi_recover * sigma
    sigma_l = sigma

    dIndhdt = (1.0 - theta_high) * tau * Eh - sigma_h * Indh
    dIdhdt  = theta_high * tau * Eh - sigma_h * Idh
    dIndldt = (1.0 - theta_low)  * tau * El - sigma_l * Indl
    dIdldt  = theta_low * tau * El - sigma_l * Idl

    dRhdt   = sigma_h * (Indh + Idh)
    dRldt   = sigma_l * (Indl + Idl)

    return np.array([dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt])


def SEIRS_model_v6(y, t, params):
    """
    SEIRS model with split-at-onset treatment and drug effects that
    modify behavior for treated infectious individuals.
    
    ═══════════════════════════════════════════════════════════════════════════
    BIOLOGICAL CONTEXT
    ═══════════════════════════════════════════════════════════════════════════
    This model tests the hypothesis that symptom-targeting drugs can alter
    pathogen evolution by decoupling virulence from its fitness costs.
    
    Key biological assumptions:
    1. High virulence causes severe symptoms → reduced contact rate (sick people stay home)
    2. Drugs reduce symptoms → treated individuals resume normal contacts
    3. Two strains compete: high-virulence (h) vs low-virulence (l)
    
    ═══════════════════════════════════════════════════════════════════════════
    STATE VARIABLES (9 compartments)
    ═══════════════════════════════════════════════════════════════════════════
    y = [S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl]
    
    S     = Susceptible (can be infected by either strain)
    
    High-virulence strain (h):
      Eh   = Exposed/latent (infected but not yet infectious)
      Indh = Infectious, Not Detected (symptomatic but not diagnosed/treated)
      Idh  = Infectious, Detected (diagnosed and receiving treatment)
      Rh   = Recovered (immune, but immunity wanes)
    
    Low-virulence strain (l):
      El   = Exposed/latent
      Indl = Infectious, Not Detected
      Idl  = Infectious, Detected
      Rl   = Recovered
    
    ═══════════════════════════════════════════════════════════════════════════
    PARAMETERS (14 total)
    ═══════════════════════════════════════════════════════════════════════════
    (c_low, r_low, phi_t, m_c_drug, m_r_drug, birth_rate, death_rate, delta,
     kappa_base, kappa_scale, phi_recover, sigma, tau, theta)
    
    Transmission parameters:
      c_low   = baseline contact rate (contacts per person per day) for low-virulence
      r_low   = transmission probability per contact (low-virulence baseline)
      phi_t   = phi_transmission: virulence multiplier (≥1.0)
                - Increases per-contact transmission probability for high strain
                - Also triggers contact rate penalty (sicker → fewer contacts)
    
    Drug effect parameters:
      m_c_drug = drug_contact_multiplier: how drug affects contact rate
                 - >1: treated individuals have MORE contacts (feel better, go out)
                 - <1: treated individuals have FEWER contacts (stay home to recover)
      m_r_drug = drug_transmission_multiplier: how drug affects transmission probability
                 - <1: drug reduces infectiousness (e.g., antivirals reduce viral load)
                 - >1: drug increases transmission (e.g., cough suppressant stops droplet expulsion)
    
    Demographic parameters:
      birth_rate = rate of new susceptibles entering population
      death_rate = background mortality rate (all compartments)
      delta      = immunity waning rate (1/duration of immunity)
    
    Detection/treatment parameters:
      kappa_base  = baseline detection probability (for low-virulence)
      kappa_scale = sensitivity of detection to virulence increase
                    - Higher virulence → more severe symptoms → more likely to seek care
      theta       = treatment coverage (fraction of detected cases that get treated)
    
    Recovery parameters:
      phi_recover = recovery rate modifier for high-strain
                    - <1: slower recovery (longer infectious period)
                    - >1: faster recovery (burn out quickly)
      sigma       = baseline recovery rate (1/infectious period)
      tau         = incubation rate (1/latent period)

    ═══════════════════════════════════════════════════════════════════════════
    MODEL EQUATIONS AND DYNAMICS
    ═══════════════════════════════════════════════════════════════════════════
    """
    import numpy as np

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Sanitize state variables (ensure non-negative)
    # ─────────────────────────────────────────────────────────────────────────
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = y

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Validate and unpack parameters
    # ─────────────────────────────────────────────────────────────────────────
    if not hasattr(params, "__len__"):
        raise ValueError("params must be a sequence")

    if len(params) == 14:
        (c_low, r_low, phi_t,
         m_c_drug, m_r_drug,
         birth_rate, death_rate, delta,
         kappa_base, kappa_scale,
         phi_recover, sigma, tau, theta) = params

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Calculate high-strain contact rate with VIRULENCE PENALTY
        # ─────────────────────────────────────────────────────────────────────
        # 
        # BIOLOGICAL RATIONALE:
        # High virulence causes severe symptoms (fever, fatigue, pain)
        # → Sick individuals reduce their contacts (stay home, bedridden)
        # 
        # MATHEMATICAL FORMULATION:
        # c_high = c_low × exp(-α × max(0, φ - 1))
        # 
        # Where:
        #   α = 0.5 (penalty severity parameter)
        #   φ = phi_transmission (virulence proxy)
        # 
        # BEHAVIOR:
        #   - φ ≤ 1: c_high = c_low (no penalty, strains have equal contact)
        #   - φ = 1.5: c_high = c_low × exp(-0.5 × 0.5) ≈ 0.78 × c_low (22% reduction)
        #   - φ = 2.0: c_high = c_low × exp(-0.5 × 1.0) ≈ 0.61 × c_low (39% reduction)
        #   - φ = 3.0: c_high = c_low × exp(-0.5 × 2.0) ≈ 0.37 × c_low (63% reduction)
        # 
        # This creates a VIRULENCE-TRANSMISSION TRADE-OFF:
        # Higher virulence increases per-contact transmission but decreases contacts
        
        alpha = 0.5  # tune this: smaller = milder penalty, larger = steeper penalty
        vir_excess_pos = max(0.0, phi_t - 1.0)  # only penalize if φ > 1
        c_high = c_low * np.exp(-alpha * vir_excess_pos)

    else:
        raise ValueError(
            "SEIRS_model_v6 expects 14 params. "
            "14: (contact_rate_low, transmission_probability_low, phi_transmission, "
            "drug_contact_multiplier, drug_transmission_multiplier, "
            "birth_rate, death_rate, delta, kappa_base, kappa_scale, "
            "phi_recover, sigma, tau, theta). "
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Calculate transmission rates (β) for all scenarios
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # TRANSMISSION RATE FORMULA:
    # β = contact_rate × transmission_probability
    # 
    # We need 4 different β values:
    #   β_l_u = low-strain, untreated
    #   β_l_t = low-strain, treated (drug-modified)
    #   β_h_u = high-strain, untreated
    #   β_h_t = high-strain, treated (drug-modified)
    
    # LOW-STRAIN UNTREATED:
    # β_l_u = c_low × r_low
    # Standard transmission for low-virulence, no drug effects
    beta_l_u = c_low * r_low
    
    # LOW-STRAIN TREATED:
    # β_l_t = (c_low × m_c_drug) × (r_low × m_r_drug)
    # Drug modifies both contact rate and transmission probability
    # Example: m_c=1.2, m_r=0.75 → more contacts but less infectious per contact
    beta_l_t = (c_low * m_c_drug) * (r_low * m_r_drug)

    # HIGH-STRAIN UNTREATED:
    # β_h_u = c_high × r_low × φ
    # - c_high: reduced contact rate due to severe symptoms
    # - r_low × φ: enhanced per-contact transmission (φ > 1)
    # Net effect depends on balance of contact penalty vs transmission boost
    beta_h_u = c_high * r_low * phi_t
    
    # HIGH-STRAIN TREATED:
    # β_h_t = (c_high × m_c_drug) × (r_low × m_r_drug) × φ
    # Drug effects applied ON TOP OF virulence penalty
    # NOTE: In v6, drug does NOT restore contact rate lost to virulence
    #       (this is changed in v7)
    beta_h_t = (c_high * m_c_drug) * (r_low * m_r_drug) * phi_t

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Calculate detection/treatment probabilities (κ and θ)
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # BIOLOGICAL RATIONALE:
    # Higher virulence → more severe symptoms → more likely to seek medical care
    # 
    # DETECTION SCALING FORMULA:
    # κ_high = κ_base × (1 + κ_scale × (φ - 1))
    # 
    # EXAMPLES (with κ_base=1.0, κ_scale=1.0):
    #   φ = 1.0: κ_high = 1.0 (same as low strain)
    #   φ = 1.5: κ_high = 1.0 × (1 + 1.0 × 0.5) = 1.5 (50% more detection)
    #   φ = 2.0: κ_high = 1.0 × (1 + 1.0 × 1.0) = 2.0 (100% more detection)
    # 
    # CONSTRAINT: Effective treatment fraction cannot exceed 1.0
    # θ_effective = κ × θ, must have κ × θ ≤ 1, so κ ≤ 1/θ
    
    vir_excess = phi_t - 1.0
    kappa_high = kappa_base * (1 + kappa_scale * vir_excess)
    kappa_low = kappa_base
    
    # Apply constraint: ensure θ_high and θ_low don't exceed 1.0
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)

    # Effective treatment fractions
    # θ_high = fraction of high-strain infections that get detected AND treated
    # θ_low = fraction of low-strain infections that get detected AND treated
    theta_high = kappa_high * theta
    theta_low = kappa_low * theta

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Calculate Forces of Infection (λ)
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # FORCE OF INFECTION:
    # The rate at which susceptibles become infected
    # 
    # For each strain, both untreated (Ind) and treated (Id) contribute:
    # B_h = β_h_u × I_nd_h + β_h_t × I_d_h  (high-strain force)
    # B_l = β_l_u × I_nd_l + β_l_t × I_d_l  (low-strain force)
    # 
    # Total infection rate for susceptibles: (B_h + B_l) × S
    
    B_h = beta_h_u * Indh + beta_h_t * Idh
    B_l = beta_l_u * Indl + beta_l_t * Idl

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Define ODEs for each compartment
    # ─────────────────────────────────────────────────────────────────────────
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUSCEPTIBLE (S)
    # ═══════════════════════════════════════════════════════════════════════
    # dS/dt = births - infection - death + immunity waning
    # 
    # + birth_rate: new susceptibles enter (births, immigration)
    # - (B_h + B_l) × S: susceptibles become infected
    # + δ × (R_h + R_l): recovered individuals lose immunity → susceptible again
    # - death_rate × S: background mortality
    dSdt = birth_rate - (B_h + B_l) * S + delta * (Rh + Rl) - death_rate * S
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXPOSED (E_h, E_l)
    # ═══════════════════════════════════════════════════════════════════════
    # dE/dt = new infections - progression to infectious - death
    # 
    # + B × S: new infections from susceptibles
    # - τ × E: progression to infectious state (rate = 1/latent_period)
    # - death_rate × E: background mortality
    dEhdt = B_h * S - tau * Eh - death_rate * Eh
    dEldt = B_l * S - tau * El - death_rate * El

    # ═══════════════════════════════════════════════════════════════════════
    # RECOVERY RATES (strain-specific)
    # ═══════════════════════════════════════════════════════════════════════
    # σ_h = φ_recover × σ (high-strain recovery rate)
    # σ_l = σ (low-strain recovery rate, baseline)
    # 
    # If φ_recover < 1: high-strain has LONGER infectious period
    # If φ_recover > 1: high-strain has SHORTER infectious period
    sigma_h = phi_recover * sigma
    sigma_l = sigma

    # ═══════════════════════════════════════════════════════════════════════
    # INFECTIOUS NOT DETECTED (I_nd_h, I_nd_l)
    # ═══════════════════════════════════════════════════════════════════════
    # 
    # SPLIT AT SYMPTOM ONSET:
    # When individuals leave E and become infectious, they are immediately
    # split into detected (Id) vs not-detected (Ind) based on θ
    # 
    # dI_nd/dt = (1 - θ) × τ × E - σ × I_nd - death
    # 
    # + (1 - θ_high) × τ × E_h: fraction NOT detected entering from exposed
    # - σ_h × I_nd_h: recovery
    # - death_rate × I_nd_h: background mortality
    dIndhdt = (1.0 - theta_high) * tau * Eh - sigma_h * Indh - death_rate * Indh
    dIndldt = (1.0 - theta_low) * tau * El - sigma_l * Indl - death_rate * Indl
    
    # ═══════════════════════════════════════════════════════════════════════
    # INFECTIOUS DETECTED/TREATED (I_d_h, I_d_l)
    # ═══════════════════════════════════════════════════════════════════════
    # 
    # dI_d/dt = θ × τ × E - σ × I_d - death
    # 
    # + θ_high × τ × E_h: fraction detected entering from exposed
    # - σ_h × I_d_h: recovery (same rate as untreated in this model)
    # - death_rate × I_d_h: background mortality
    # 
    # NOTE: In this model, treatment affects TRANSMISSION (via β_t)
    #       but NOT recovery rate (both Ind and Id recover at rate σ)
    dIdhdt = theta_high * tau * Eh - sigma_h * Idh - death_rate * Idh
    dIdldt = theta_low * tau * El - sigma_l * Idl - death_rate * Idl

    # RECOVERED
    # dR/dt = σ × (I_nd + I_d) - δ × R - death
    dRhdt = sigma_h * (Indh + Idh) - delta * Rh - death_rate * Rh
    dRldt = sigma_l * (Indl + Idl) - delta * Rl - death_rate * Rl

    return np.array([dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt])


def SEIRS_model_v7(y, t, params):
    """
    v7: Drug restores contact rate for treated individuals (symptom masking).
    
    ═══════════════════════════════════════════════════════════════════════════
    KEY DIFFERENCE FROM v6
    ═══════════════════════════════════════════════════════════════════════════
    
    In v6: Drug effects (m_c, m_r) apply uniformly to all treated individuals
           but do NOT compensate for the virulence-induced contact penalty.
           
           c_high_treated_v6 = c_high × m_c_drug
                             = (c_low × exp(-α(φ-1))) × m_c_drug
           
           The virulence penalty exp(-α(φ-1)) remains!
    
    In v7: Drug RESTORES contact rate for high-strain treated individuals
           because the drug masks their severe symptoms.
           
           c_high_treated_v7 = c_high_untreated + ρ × (c_low - c_high_untreated)
           
           Where ρ = drug_contact_restore ∈ [0, 1]
           
           If ρ = 0: no restoration (same as v6 without m_c effect)
           If ρ = 1: full restoration to c_low (complete symptom masking)
           If ρ = 0.5: partial restoration (50% of lost contacts recovered)
    
    ═══════════════════════════════════════════════════════════════════════════
    BIOLOGICAL HYPOTHESIS
    ═══════════════════════════════════════════════════════════════════════════
    
    This tests whether symptom-masking drugs can SELECT FOR higher virulence:
    
    Without drug:
      High virulence → severe symptoms → fewer contacts → transmission penalty
      This creates natural selection AGAINST high virulence
    
    With symptom-masking drug:
      High virulence → severe symptoms → TREATMENT → symptoms masked → 
      contacts restored → no transmission penalty!
      
      Meanwhile, high virulence still provides transmission ADVANTAGE:
      β_h = φ × r_low × c (higher per-contact transmission)
      
    Result: Drug removes the FITNESS COST of high virulence while preserving
            the FITNESS BENEFIT, potentially selecting for "super-virulent" strains.
    
    ═══════════════════════════════════════════════════════════════════════════
    PARAMETERS (15 total) - adds drug_contact_restore
    ═══════════════════════════════════════════════════════════════════════════
    (c_low, r_low, phi_t, m_c_drug, m_r_drug, drug_contact_restore,
     birth_rate, death_rate, delta, kappa_base, kappa_scale, 
     phi_recover, sigma, tau, theta)
    
    NEW PARAMETER:
      drug_contact_restore (ρ) = degree to which drug restores contact rate
                                 for high-strain treated individuals
        - ρ = 0: no restoration (drug doesn't help mobility)
        - ρ = 1: full restoration (complete symptom masking)
        - ρ = 0.8 (default): substantial but not complete restoration
    
    All other parameters same as v6.

    ═══════════════════════════════════════════════════════════════════════════
    """
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Sanitize state variables
    # ─────────────────────────────────────────────────────────────────────────
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = y

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Validate and unpack parameters
    # ─────────────────────────────────────────────────────────────────────────
    if len(params) != 15:
        raise ValueError("SEIRS_model_v7 expects 15 params")

    (c_low, r_low, phi_t,
     m_c_drug, m_r_drug, drug_contact_restore,
     birth_rate, death_rate, delta,
     kappa_base, kappa_scale,
     phi_recover, sigma, tau, theta) = params

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Calculate contact rates with VIRULENCE PENALTY
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # Same as v6: high virulence reduces contact rate for UNTREATED individuals
    # c_high_untreated = c_low × exp(-α × max(0, φ - 1))
    
    alpha = 0.5  # virulence penalty severity
    vir_excess_pos = max(0.0, phi_t - 1.0)
    c_high_untreated = c_low * np.exp(-alpha * vir_excess_pos)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Drug RESTORES contact rate for TREATED high-strain (KEY v7 CHANGE)
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # SYMPTOM MASKING MECHANISM:
    # The drug reduces symptoms, allowing sick individuals to resume activities
    # 
    # FORMULA:
    # c_high_treated = c_high_untreated + ρ × (c_low - c_high_untreated)
    # 
    # INTERPRETATION:
    #   c_high_untreated = contact rate if sick and untreated
    #   c_low = contact rate if healthy (or mildly symptomatic)
    #   (c_low - c_high_untreated) = contacts LOST due to severe symptoms
    #   ρ × (c_low - c_high_untreated) = contacts RESTORED by drug
    # 
    # EXAMPLES (with α=0.5, c_low=10):
    #   φ = 1.5: c_high_untreated = 7.79
    #            Lost contacts = 10 - 7.79 = 2.21
    #            ρ = 0.8: restored = 0.8 × 2.21 = 1.77
    #            c_high_treated = 7.79 + 1.77 = 9.56 
    #   
    #   φ = 2.0: c_high_untreated = 6.07
    #            Lost contacts = 10 - 6.07 = 3.93
    #            ρ = 0.8: restored = 0.8 × 3.93 = 3.14
    #            c_high_treated = 6.07 + 3.14 = 9.21
    # 
    # KEY INSIGHT: Higher virulence has MORE to gain from treatment!
    # This creates ASYMMETRIC selection pressure favoring high virulence.
    
    c_high_treated = c_high_untreated + drug_contact_restore * (c_low - c_high_untreated)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Low-strain contact rates (standard drug effect)
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # Low-strain individuals have no virulence penalty, so drug just applies
    # the standard contact multiplier:
    # c_low_treated = c_low × m_c_drug
    # 
    # NOTE: m_c_drug effect is separate from drug_contact_restore
    #       m_c_drug: general behavioral change (e.g., feel better → go to work)
    #       drug_contact_restore: specifically restores contacts lost to symptoms

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Calculate all transmission rates (β)
    # ─────────────────────────────────────────────────────────────────────────
    
    # HIGH-STRAIN UNTREATED:
    # β_h_u = c_high_untreated × r_low × φ
    # - Reduced contacts (sick at home)
    # - Enhanced per-contact transmission (φ > 1)
    beta_h_u = c_high_untreated * r_low * phi_t
    
    # HIGH-STRAIN TREATED:
    # β_h_t = c_high_treated × (r_low × m_r_drug) × φ
    # - RESTORED contacts (drug masks symptoms) ← KEY v7 CHANGE
    # - Drug-modified transmission probability
    # - Still has virulence transmission boost (φ)
    # 
    # NOTE: m_r_drug typically < 1 (drug reduces viral load/infectiousness)
    #       but contact restoration can MORE than compensate!
    beta_h_t = c_high_treated * (r_low * m_r_drug) * phi_t
    
    # LOW-STRAIN UNTREATED:
    # β_l_u = c_low × r_low
    # Standard baseline transmission
    beta_l_u = c_low * r_low
    
    # LOW-STRAIN TREATED:
    # β_l_t = c_low_treated × (r_low × m_r_drug)
    # Drug effects on contact rate and transmission probability
    # No virulence boost (no φ term)
    beta_l_t = (c_low * m_c_drug) * (r_low * m_r_drug)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Detection/treatment probabilities (same as v6)
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # κ_high = κ_base × (1 + κ_scale × (φ - 1))
    # Higher virulence → more severe symptoms → higher detection
    # 
    # This creates an INTERESTING DYNAMIC:
    # High virulence is more likely to be DETECTED (κ_high > κ_low)
    # But treated high-virulence can now TRANSMIT effectively (via restoration)
    # 
    # So the pathway is:
    # High virulence → severe symptoms → seek treatment → get drug →
    # symptoms masked → resume contacts → spread disease!
    
    kappa_high = kappa_base * (1 + kappa_scale * vir_excess_pos)
    kappa_low = kappa_base
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)

    theta_high = kappa_high * theta
    theta_low = kappa_low * theta

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Forces of Infection
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # B_h = β_h_u × I_nd_h + β_h_t × I_d_h
    # B_l = β_l_u × I_nd_l + β_l_t × I_d_l
    # 
    # KEY v7 INSIGHT:
    # For high-strain, β_h_t can be LARGER than β_h_u if restoration is strong!
    # This means treated individuals may transmit MORE than untreated ones.
    # 
    # Example calculation:
    #   φ = 1.5, ρ = 0.8, m_r_drug = 0.75
    #   β_h_u = 7.79 × 0.025 × 1.5 = 0.292
    #   β_h_t = 9.56 × (0.025 × 0.75) × 1.5 = 0.269
    #   
    #   Here β_h_t < β_h_u because m_r_drug reduction outweighs restoration.
    #   
    #   But if m_r_drug = 1.0 (drug doesn't reduce infectiousness):
    #   β_h_t = 9.56 × 0.025 × 1.5 = 0.359 > 0.292 = β_h_u
    #   Treated transmit MORE than untreated!
    
    B_h = beta_h_u * Indh + beta_h_t * Idh
    B_l = beta_l_u * Indl + beta_l_t * Idl

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: ODEs (same structure as v6)
    # ─────────────────────────────────────────────────────────────────────────
    
    # SUSCEPTIBLES
    # dS/dt = births - infection - death + immunity waning
    dSdt = birth_rate - (B_h + B_l) * S + delta * (Rh + Rl) - death_rate * S
    
    # EXPOSED
    # dE/dt = new infections - progression - death
    dEhdt = B_h * S - tau * Eh - death_rate * Eh
    dEldt = B_l * S - tau * El - death_rate * El

    # RECOVERY RATES (strain-specific)
    sigma_h = phi_recover * sigma
    sigma_l = sigma

    # INFECTIOUS NOT DETECTED
    # dI_nd/dt = (1-θ) × τ × E - σ × I_nd - death
    dIndhdt = (1.0 - theta_high) * tau * Eh - sigma_h * Indh - death_rate * Indh
    dIndldt = (1.0 - theta_low) * tau * El - sigma_l * Indl - death_rate * Indl
    
    # INFECTIOUS DETECTED/TREATED
    # dI_d/dt = θ × τ × E - σ × I_d - death
    dIdhdt = theta_high * tau * Eh - sigma_h * Idh - death_rate * Idh
    dIdldt = theta_low * tau * El - sigma_l * Idl - death_rate * Idl

    # RECOVERED
    # dR/dt = σ × (I_nd + I_d) - δ × R - death
    dRhdt = sigma_h * (Indh + Idh) - delta * Rh - death_rate * Rh
    dRldt = sigma_l * (Indl + Idl) - delta * Rl - death_rate * Rl

    return np.array([dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt])


def SEIRS_model_v8(y, t, params):
    """
    v8: Unified contact restoration mechanism (virulence-aware symptom masking).
    
    ═══════════════════════════════════════════════════════════════════════════
    KEY DIFFERENCE FROM v7
    ═══════════════════════════════════════════════════════════════════════════
    
    In v7: Two separate mechanisms:
           - m_c_drug: general contact boost (applies to all treated)
           - drug_contact_restore (ρ): specific to high-strain
           Creates confusing interactions and biological ambiguity.
    
    In v8: Single unified mechanism:
           - restoration_efficiency (ρ): how well drug masks symptoms
           - Applies proportionally to symptomatic burden
           
           c_treated = c_untreated + ρ × (c_low - c_untreated)
           
           Where (c_low - c_untreated) = contacts LOST to symptoms
           
    ═══════════════════════════════════════════════════════════════════════════
    BIOLOGICAL RATIONALE
    ═══════════════════════════════════════════════════════════════════════════
    
    Symptoms reduce contacts proportional to their severity:
      - No symptoms (healthy): c = c_low (baseline)
      - Mild symptoms (low-strain): c ≈ c_low (stays home a bit)
      - Severe symptoms (high-strain untreated): c = c_high << c_low (bedridden)
    
    Drug masks symptoms → restores lost contacts proportional to severity:
      - High-strain treated: can restore much (lost 30% of contacts)
      - Low-strain treated: can restore little (lost 5% of contacts)
    
    This is BIOLOGICALLY REALISTIC:
      Drug effectiveness depends on how much symptom burden exists to mask!
      No symptoms → no restoration possible
      Severe symptoms → drug can restore substantial contacts
    
    ═══════════════════════════════════════════════════════════════════════════
    PARAMETERS (14 total) - simplified from v7's 15
    ═══════════════════════════════════════════════════════════════════════════
    (c_low, r_low, phi_t,
     restoration_efficiency, m_r_drug,
     birth_rate, death_rate, delta,
     kappa_base, kappa_scale,
     phi_recover, sigma, tau, theta)
    
    REMOVED:
      - m_c_drug (general contact multiplier) → replaced by restoration_efficiency
      - drug_contact_restore → merged into restoration_efficiency
    
    NEW/MODIFIED:
      restoration_efficiency (ρ) = degree to which drug restores contacts lost to symptoms
        - ρ = 0: no symptom masking (sick stay home regardless of treatment)
        - ρ = 1: complete symptom masking (treated feel/appear healthy)
        - ρ = 0.8 (typical): substantial but not complete restoration
        
    KEPT:
      - m_r_drug: transmission probability modifier (antivirals reduce viral load)
        Still acts separately from contact restoration
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Sanitize state variables
    # ─────────────────────────────────────────────────────────────────────────
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    S, Eh, Indh, Idh, Rh, El, Indl, Idl, Rl = y

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Validate and unpack parameters
    # ─────────────────────────────────────────────────────────────────────────
    if len(params) != 14:
        raise ValueError(
            "SEIRS_model_v8 expects 14 params: "
            "(c_low, r_low, phi_t, restoration_efficiency, m_r_drug, "
            "birth_rate, death_rate, delta, kappa_base, kappa_scale, "
            "phi_recover, sigma, tau, theta)"
        )

    (c_low, r_low, phi_t,
     restoration_efficiency, m_r_drug,
     birth_rate, death_rate, delta,
     kappa_base, kappa_scale,
     phi_recover, sigma, tau, theta) = params

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Calculate contact rates with VIRULENCE PENALTY
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # Untreated contact rates: virulence causes symptom-induced reduction
    # c_high_untreated = c_low × exp(-α × max(0, φ - 1))
    
    alpha = 0.5  # virulence penalty severity
    vir_excess_pos = max(0.0, phi_t - 1.0)
    c_high_untreated = c_low * np.exp(-alpha * vir_excess_pos)
    
    # LOW-STRAIN: mild symptoms, only minor contact reduction
    # For v8, we assume low-strain has minimal symptom burden
    # If you want to model it explicitly:
    # c_low_untreated = c_low × exp(-α_low × (φ - 1))
    # For now, c_low_untreated ≈ c_low (no penalty for low-strain)
    c_low_untreated = c_low

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: UNIFIED CONTACT RESTORATION (KEY v8 CHANGE)
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # FORMULA:
    # c_treated = c_untreated + ρ × (c_low - c_untreated)
    # 
    # Biological interpretation:
    #   - Baseline (c_low): healthy contact rate
    #   - Untreated: reduced contact rate due to symptoms
    #   - Drug restores: ρ fraction of lost contacts
    # 
    # EXAMPLE CALCULATIONS (c_low=10, φ=1.5, α=0.5):
    #   
    #   High-strain:
    #     c_high_untreated = 10 × exp(-0.5×0.5) = 7.79
    #     contacts_lost = 10 - 7.79 = 2.21
    #     
    #     ρ = 0: c_high_treated = 7.79 (no restoration)
    #     ρ = 0.5: c_high_treated = 7.79 + 0.5×2.21 = 8.90
    #     ρ = 0.8: c_high_treated = 7.79 + 0.8×2.21 = 9.56
    #     ρ = 1.0: c_high_treated = 10.00 (full restoration)
    #   
    #   Low-strain:
    #     c_low_untreated = 10 (no penalty)
    #     contacts_lost = 10 - 10 = 0
    #     c_low_treated = 10 + ρ×0 = 10 (no restoration needed)
    # 
    # KEY INSIGHT:
    # Drug effectiveness depends ONLY on symptom severity!
    # High virulence (more symptoms) → more restoration possible
    # Low virulence (mild symptoms) → little restoration possible
    # 
    # This avoids the awkward "why does low-strain get a boost?" question
    
    c_high_treated = c_high_untreated + restoration_efficiency * (c_low - c_high_untreated)
    c_low_treated = c_low_untreated + restoration_efficiency * (c_low - c_low_untreated)
    # Note: c_low_treated = c_low since c_low_untreated = c_low

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Calculate transmission rates (β)
    # ─────────────────────────────────────────────────────────────────────────
    
    # HIGH-STRAIN UNTREATED:
    # β_h_u = c_high_untreated × r_low × φ
    beta_h_u = c_high_untreated * r_low * phi_t
    
    # HIGH-STRAIN TREATED:
    # β_h_t = c_high_treated × (r_low × m_r_drug) × φ
    # Contact restoration + drug transmission reduction
    beta_h_t = c_high_treated * (r_low * m_r_drug) * phi_t
    
    # LOW-STRAIN UNTREATED:
    # β_l_u = c_low × r_low
    beta_l_u = c_low_untreated * r_low
    
    # LOW-STRAIN TREATED:
    # β_l_t = c_low_treated × (r_low × m_r_drug)
    # Since c_low_treated = c_low_untreated, only m_r_drug changes transmission
    beta_l_t = c_low_treated * (r_low * m_r_drug)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Detection/treatment probabilities
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # Higher virulence → more severe symptoms → higher detection
    # κ_high = κ_base × (1 + κ_scale × (φ - 1))
    
    kappa_high = kappa_base * (1 + kappa_scale * vir_excess_pos)
    kappa_low = kappa_base
    if theta > 0:
        kappa_high = min(kappa_high, 1.0 / theta)
        kappa_low = min(kappa_low, 1.0 / theta)

    theta_high = kappa_high * theta
    theta_low = kappa_low * theta

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Forces of Infection
    # ─────────────────────────────────────────────────────────────────────────
    
    B_h = beta_h_u * Indh + beta_h_t * Idh
    B_l = beta_l_u * Indl + beta_l_t * Idl

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: ODEs
    # ─────────────────────────────────────────────────────────────────────────
    
    dSdt = birth_rate - (B_h + B_l) * S + delta * (Rh + Rl) - death_rate * S
    
    dEhdt = B_h * S - tau * Eh - death_rate * Eh
    dEldt = B_l * S - tau * El - death_rate * El

    sigma_h = phi_recover * sigma
    sigma_l = sigma

    dIndhdt = (1.0 - theta_high) * tau * Eh - sigma_h * Indh - death_rate * Indh
    dIndldt = (1.0 - theta_low) * tau * El - sigma_l * Indl - death_rate * Indl
    
    dIdhdt = theta_high * tau * Eh - sigma_h * Idh - death_rate * Idh
    dIdldt = theta_low * tau * El - sigma_l * Idl - death_rate * Idl

    dRhdt = sigma_h * (Indh + Idh) - delta * Rh - death_rate * Rh
    dRldt = sigma_l * (Indl + Idl) - delta * Rl - death_rate * Rl

    return np.array([dSdt, dEhdt, dIndhdt, dIdhdt, dRhdt, dEldt, dIndldt, dIdldt, dRldt])
