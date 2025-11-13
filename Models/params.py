# Initial absolute counts for each compartment (counts)
S = 10000
Eh = 0
Indh = 5  # High-virulence initial infections
Idh = 0
Rh = 0
El = 0
Indl = 5  # Low-virulence initial infections
Idl = 0
Rl = 0

# Model parameters (units: days for rates unless noted)
# Treatment / biology
theta = 0.3            # fraction of DETECTED cases that get treated (0..1)
                       # Set low (0.3) as baseline for dose-response experiments
                       # Effective treatment rate = delta_d * theta ≈ 10% per day
p_recover = 0.5        # treated transmission rate multiplier (<1 means less transmission)

# Virulence modifiers
phi_transmission = 1.05 # multiplier for high-virulence transmission (beta_h = phi_transmission * beta_l)

phi_recover = 1.0       # modifier for high-virulence recovery rate (currently no effect)
                        # Future use: <1 = slower recovery (longer infectious period)
phi_mortality = 1.0     # modifier for high-virulence mortality rate (currently no effect)
                        # Future use: >1 = higher disease-induced mortality

# Rates (per day)
sigma = 1/5           # recovery rate (1 / infectious period = 5 days for untreated)
delta = 1/120           # immunity waning rate (1 / duration of immunity = 120 days)
tau = 1/3              # progression from exposed -> infectious (1 / incubation = 3 days)
# Demography and transmission
birth_rate = 0.0       # No births (short-term epidemic focus)
death_rate = 0.0       # No background mortality (disease-induced mortality could be added)
contact_rate = 10.0    # average contacts per person per day
transmission_probability = 0.025  # probability of transmission per contact
beta_l = contact_rate * transmission_probability  # baseline transmission rate for low-virulence strain
                                                  # Results in R0_low ≈ beta_l/sigma = 2.5 (COVID-like)
                                                  # Results in R0_high ≈ phi_transmission * R0_low = 2.625

# Time grid defaults (used if scripts import these)
t_max = 365            # simulation length in days
t_steps = 365          # number of time points (daily resolution)

# Detection/uptake parameters
kappa_base = 1.0        # baseline: no advantage for either strain
kappa_scale = 1.0       # moderate sensitivity to virulence
                        # Range: 0.5-3.0
                        # Examples:
                        # - kappa_scale=1: phi_trans=1.1 → 10% more detection
                        # - kappa_scale=2: phi_trans=1.1 → 20% more detection

# Biological interpretation:
# kappa_high * theta = effective treatment fraction for high-virulence
# Higher virulence → more severe symptoms → more people seek treatment
# Even if baseline coverage (theta) is fixed

# Experimental design notes:
# - theta kept LOW (0.3) to allow dose-response studies (vary 0.3 → 0.9)
# - Equal strain seeding (5 each) tests competitive dynamics
# - R0 difference (5%) tests subtle evolutionary advantage
# - Asymmetric transmission: Idh contributes, Idl doesn't (symptom-masking hypothesis)

# Optional: recommended ranges for sensitivity analysis
# theta: 0.0 - 0.9 (vary treatment coverage)
# p_recover: 1.1 - 2.0 (vary treatment efficacy)
# phi_transmission: 1.01 - 1.2 (vary virulence advantage)
# delta: 1/365 (1 year) - 1/30 (1 month) (vary immunity duration)
